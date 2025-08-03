import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from functools import partial
import time
import src.data.dlimp as dl
from src.data import obs_transforms
from src.data.utils.data_utils import get_dataset_statistics, normalize_action_and_proprio, NormalizationType
import os

log = logging.getLogger(__name__)

def log_execution_time(logger):
    """Decorator to log execution time of functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
            return result
        return wrapper
    return decorator

def restructure_traj(traj, image_obs_keys, depth_obs_keys, proprio_obs_key, language_key):
    """Restructure trajectory into a standard format."""
    traj_len = tf.shape(traj["action"])[0]
    old_obs = traj["observation"]
    new_obs = {}
    
    # Extract images
    for new, old in image_obs_keys.items():
        if old is None:
            new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
        else:
            new_obs[f"image_{new}"] = old_obs[old]

    # Extract depth images
    for new, old in depth_obs_keys.items():
        if old is None:
            new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
        else:
            new_obs[f"depth_{new}"] = old_obs[old]

    # Extract proprioception
    if proprio_obs_key is not None:
        new_obs["proprio"] = tf.cast(old_obs[proprio_obs_key], tf.float32)

    # Add timestep info
    new_obs["timestep"] = tf.range(traj_len)

    # Extract language instruction
    task = {}
    if language_key is not None and language_key in traj:
        task["language_instruction"] = traj[language_key]
        if task["language_instruction"].dtype != tf.string:
            task["language_instruction"] = tf.cast(task["language_instruction"], tf.string)

    return {
        "observation": new_obs,
        "task": task,
        "action": tf.cast(traj["action"], tf.float32),
        "dataset_name": tf.constant(traj.get("dataset_name", "unknown")),
        "traj_length": traj_len,
    }

def load_dataset_without_breaking(
    name,
    data_dir,
    train=True,
    split=None,
    image_obs_keys=None,
    depth_obs_keys=None,
    proprio_obs_key=None,
    language_key=None,
    normalize=True,
    filter_fn=None,
    max_traj_length=None,
    min_traj_length=0,
):
    """
    Load trajectories without breaking them up.
    
    Args:
        name: Name of the dataset
        data_dir: Directory containing the data
        train: Whether to load the training split
        split: Specific split to load (overrides train)
        image_obs_keys: Dict mapping new image names to observation keys
        depth_obs_keys: Dict mapping new depth image names to observation keys
        proprio_obs_key: Key for proprioception data
        language_key: Key for language instructions
        normalize: Whether to normalize actions and proprioception
        filter_fn: Optional function to filter trajectories
        max_traj_length: Maximum trajectory length (None for no limit)
        min_traj_length: Minimum trajectory length
    
    Returns:
        DLataset containing full trajectories
    """
    # Default values
    image_obs_keys = image_obs_keys or {}
    depth_obs_keys = depth_obs_keys or {}
    
    # Determine split
    if split is None:
        builder = tfds.builder(name, data_dir=data_dir)
        if "val" not in builder.info.splits:
            split = "train[:95%]" if train else "train[95%:]"
        else:
            split = "train" if train else "val"
    
    # Load the dataset
    dataset = dl.DLataset.from_rlds(
        tfds.builder(name, data_dir=data_dir),
        split=split,
        shuffle=False,  # No shuffle to maintain trajectory integrity
    )
    
    # Apply restructuring
    restructure_fn = partial(
        restructure_traj,
        image_obs_keys=image_obs_keys,
        depth_obs_keys=depth_obs_keys,
        proprio_obs_key=proprio_obs_key,
        language_key=language_key,
    )
    
    dataset = dataset.traj_map(restructure_fn)
    
    # Filter out empty trajectories
    dataset = dataset.filter(lambda x: tf.shape(x["action"])[0] > 0)
    
    # Filter by trajectory length
    if min_traj_length > 0:
        dataset = dataset.filter(lambda x: tf.shape(x["action"])[0] >= min_traj_length)
    
    if max_traj_length is not None:
        dataset = dataset.filter(lambda x: tf.shape(x["action"])[0] <= max_traj_length)
    
    # Apply custom filter if provided
    if filter_fn is not None:
        dataset = dataset.filter(filter_fn)
    
    # Compute or load statistics for normalization
    if normalize:
        stats = get_dataset_statistics(
            dataset,
            hash_dependencies=(name, str(proprio_obs_key)),
            save_dir=data_dir,
            force_recompute=False,
        )
        
        # Apply normalization
        dataset = dataset.traj_map(
            partial(
                normalize_action_and_proprio,
                metadata=stats,
                normalization_type=NormalizationType.BOUNDS,
            )
        )
    
    return dataset

def apply_image_processing(traj, resize_size=None, train=False, image_augment_kwargs=None):
    """Apply image processing to a trajectory without breaking it."""
    resize_size = resize_size or {}
    image_augment_kwargs = image_augment_kwargs or {}
    
    # Process all images in the trajectory
    obs = traj["observation"]
    
    # Find all image keys
    image_keys = [k for k in obs.keys() if k.startswith("image_")]
    
    for key in image_keys:
        # Extract the image name (after "image_")
        img_name = key[6:]
        
        # Decode images
        if img_name in resize_size:
            size = resize_size[img_name]
            images = tf.map_fn(
                lambda img: tf.cond(
                    tf.equal(img, ""),
                    lambda: tf.zeros((*size, 3), dtype=tf.uint8),
                    lambda: tf.image.resize(tf.io.decode_image(img, channels=3), size)
                ),
                obs[key],
                fn_output_signature=tf.TensorSpec((*size, 3), tf.uint8)
            )
            obs[key] = images
        
        # Apply augmentation if in training mode
        if train and img_name in image_augment_kwargs:
            aug_kwargs = image_augment_kwargs[img_name]
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            
            # Apply augmentation to each image in the trajectory
            images = tf.map_fn(
                lambda img: obs_transforms.augment_image(img, **aug_kwargs, seed=seed),
                obs[key],
                fn_output_signature=tf.TensorSpec(obs[key].shape[1:], tf.uint8)
            )
            obs[key] = images
    
    traj["observation"] = obs
    return traj

class TrajectorySampler:
    """Samples whole trajectories from multiple datasets according to weights."""
    
    def __init__(self, datasets, weights):
        """
        Initialize the trajectory sampler.
        
        Args:
            datasets: List of tf.data.Dataset objects containing trajectories
            weights: List of sampling weights for each dataset
        """
        self.datasets = datasets
        self.weights = np.array(weights) / np.sum(weights)
        self.dataset_iterators = [iter(dataset.repeat()) for dataset in datasets]
    
    def sample(self):
        """Sample a trajectory according to the weights."""
        # Choose a dataset according to weights
        dataset_idx = np.random.choice(len(self.datasets), p=self.weights)
        
        # Get the next trajectory from that dataset
        try:
            return next(self.dataset_iterators[dataset_idx])
        except StopIteration:
            # Reset iterator if needed (shouldn't happen with repeat())
            self.dataset_iterators[dataset_idx] = iter(self.datasets[dataset_idx].repeat())
            return next(self.dataset_iterators[dataset_idx])

class TorchTrajectoryDataset(Dataset):
    """PyTorch dataset that provides whole trajectories."""
    
    def __init__(self, tf_datasets, sample_weights, train=True, resize_size=None, image_augment_kwargs=None):
        """
        Initialize the dataset.
        
        Args:
            tf_datasets: List of TensorFlow datasets containing trajectories
            sample_weights: Weights for sampling from each dataset
            train: Whether in training mode (affects augmentation)
            resize_size: Dict mapping image names to resize dimensions
            image_augment_kwargs: Dict of image augmentation parameters
        """
        self.sampler = TrajectorySampler(tf_datasets, sample_weights)
        self.train = train
        self.resize_size = resize_size or {}
        self.image_augment_kwargs = image_augment_kwargs or {}
        
        # Calculate total number of trajectories (approx)
        self.total_trajectories = sum(
            [ds.reduce(0, lambda count, _: count + 1).numpy() for ds in tf_datasets]
        )
    
    def __len__(self):
        """Return the total number of trajectories."""
        return self.total_trajectories
    
    def __getitem__(self, idx):
        """Get a trajectory (ignores idx, uses weighted sampling)."""
        # Sample a trajectory
        traj = self.sampler.sample()
        
        # Process images
        traj = apply_image_processing(
            traj,
            resize_size=self.resize_size,
            train=self.train,
            image_augment_kwargs=self.image_augment_kwargs
        )
        
        # Convert to PyTorch tensors
        torch_traj = {}
        
        # Convert observations
        torch_traj["observation"] = {}
        for k, v in traj["observation"].items():
            if k.startswith("image_"):
                # Convert images to torch format (B, C, H, W)
                v = v.numpy().astype(np.float32) / 255.0
                v = np.transpose(v, (0, 3, 1, 2))
                torch_traj["observation"][k] = torch.tensor(v)
            elif k == "proprio":
                torch_traj["observation"][k] = torch.tensor(v.numpy(), dtype=torch.float32)
            elif k == "timestep":
                torch_traj["observation"][k] = torch.tensor(v.numpy(), dtype=torch.long)
        
        # Convert task information
        torch_traj["task"] = {}
        if "language_instruction" in traj["task"]:
            torch_traj["task"]["language_instruction"] = traj["task"]["language_instruction"].numpy().decode("utf-8")
        
        # Convert actions
        torch_traj["action"] = torch.tensor(traj["action"].numpy(), dtype=torch.float32)
        
        # Add metadata
        torch_traj["dataset_name"] = traj["dataset_name"].numpy().decode("utf-8") if isinstance(traj["dataset_name"], tf.Tensor) else traj["dataset_name"]
        torch_traj["traj_length"] = traj["traj_length"].numpy() if isinstance(traj["traj_length"], tf.Tensor) else traj["traj_length"]
        
        return torch_traj

class TrajectoryInterleavedDataset:
    @log_execution_time(log)
    def __init__(self, config, train=True):
        """
        Create a dataset of intact trajectories with interleaved sampling.
        
        Args:
            config: Configuration object with dataset parameters
            train: Whether in training mode
        """
        # Parse dataset mixture and weights
        dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
            config.dataset_mix,
            config.data_path,
            load_proprio=config.load_proprio,
            load_camera_views=config.get("load_camera_views", ("primary",)),
        )
        
        # Load individual datasets without breaking trajectories
        tf_datasets = []
        
        for kwargs in dataset_kwargs_list:
            # Extract basic dataset parameters
            dataset = load_dataset_without_breaking(
                name=kwargs["name"],
                data_dir=kwargs["data_dir"],
                train=train,
                split=config.get("split", None),
                image_obs_keys=kwargs.get("image_obs_keys", {}),
                depth_obs_keys=kwargs.get("depth_obs_keys", {}),
                proprio_obs_key=kwargs.get("proprio_obs_key"),
                language_key=kwargs.get("language_key"),
                normalize=not config.get("skip_norm", False),
                min_traj_length=config.get("min_traj_length", 0),
                max_traj_length=config.get("max_traj_length", None),
            )
            
            # Skip trajectories without language if needed
            if config.get("skip_unlabeled", False):
                dataset = dataset.filter(
                    lambda x: "language_instruction" in x["task"] and 
                             x["task"]["language_instruction"] != ""
                )
            
            tf_datasets.append(dataset)
        
        # Create PyTorch dataset with trajectory sampling
        self.dataset = TorchTrajectoryDataset(
            tf_datasets=tf_datasets,
            sample_weights=sample_weights,
            train=train,
            resize_size={
                "primary": (224, 224),
                "wrist": (224, 224),
                # Add other camera views as needed
            },
            image_augment_kwargs=config.get("image_augment_kwargs", {
                "primary": {
                    "random_brightness": [0.1],
                    "random_contrast": [0.9, 1.1],
                    "random_saturation": [0.9, 1.1],
                    "random_hue": [0.05],
                }
            }) if train else None,
        )
    
    def get_dataloader(self, batch_size=1, num_workers=4, shuffle=False):
        """
        Get a PyTorch DataLoader for this dataset.
        
        Note: batch_size should typically be 1 since each item is already a full trajectory
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_trajectories
        )
    
    def _collate_trajectories(self, batch):
        """
        Custom collate function for batching trajectories of different lengths.
        When batch_size=1, this basically passes through the trajectory.
        """
        if len(batch) == 1:
            return batch[0]
        
        # This is more complex and requires padding trajectories to same length
        # For simplicity, we recommend using batch_size=1 when working with trajectories
        # But a full implementation can be added if needed
        raise NotImplementedError(
            "Batching multiple trajectories requires padding implementation. "
            "For trajectory data, we recommend using batch_size=1."
        )

# Example usage
def make_oxe_dataset_kwargs_and_weights(dataset_mix, data_path, load_proprio=True, load_camera_views=("primary",)):
    """
    Helper function to create dataset kwargs and weights from config.
    You'll need to adapt this to your actual configuration format.
    """
    dataset_kwargs_list = []
    sample_weights = []
    
    for dataset_name, weight in dataset_mix.items():
        image_obs_keys = {}
        for view in load_camera_views:
            image_obs_keys[view] = f"{view}_camera"  # Adjust naming convention as needed
        
        dataset_kwargs_list.append({
            "name": dataset_name,
            "data_dir": data_path,
            "image_obs_keys": image_obs_keys,
            "depth_obs_keys": {},  # Add depth keys if needed
            "proprio_obs_key": "robot_state" if load_proprio else None,
            "language_key": "language_instruction",  # Adjust key name as needed
        })
        sample_weights.append(weight)
    
    return dataset_kwargs_list, sample_weights


class ExampleConfig:
    def __init__(self):
        self.dataset_mix = {
            "bridge_dataset": 1.0,
        }
        self.data_path = os.environ.get("VLA_DATA_DIR/resize_224")
        self.load_proprio = True
        self.load_camera_views = ["primary", "wrist"]
        self.skip_unlabeled = True
        self.min_traj_length = 10
        self.max_traj_length = 200
        self.image_augment_kwargs = {
            "primary": {
                "random_brightness": [0.1],
                "random_contrast": [0.9, 1.1],
                "random_saturation": [0.9, 1.1],
                "random_hue": [0.05],
            }
        }
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    



def main():
    # Create the dataset
    from PIL import Image
    import os
    logdir = "/n/fs/vlm-syn/vlm-syn/scripts/test"

    config = ExampleConfig()
    trajectory_dataset = TrajectoryInterleavedDataset(config, train=True)

    # Get a dataloader (batch_size=1 because each item is already a full trajectory)
    dataloader = trajectory_dataset.get_dataloader(batch_size=1, num_workers=4)

    # test the dataloader
    for traj in dataloader:
        # Each traj is a complete trajectory containing:
        # - traj["observation"] with image and proprio data for all timesteps
        # - traj["action"] with actions for all timesteps
        # - traj["task"]["language_instruction"] with the language instruction
        # - traj["traj_length"] with the trajectory length

        # save all images from trajectory to logdir
        for timestep, image in enumerate(traj["observation"]["image_primary"]):
            # Convert (C, H, W) -> (H, W, C)
            if image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))

            print(f"Transformed shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

            # Convert to uint8 (values are already in 0-255, no need to rescale)
            image = image.astype(np.uint8)
            print(f"Final shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")

            # Convert to PIL image and save
            image = Image.fromarray(image)
            image.save(os.path.join(logdir, f"image_{timestep}.png"))
            print(f"Saved image_{timestep}.png")

        print(f"Processed trajectory with length {traj['traj_length']}")
        exit()
        
        # Process your trajectory...
    
if __name__ == "__main__":
    main()

