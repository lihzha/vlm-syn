"""Create per-bridge data folders expected by downstream code."""

import os
import sys

from PIL import Image

sys.path.insert(0, "/n/fs/robot-data/vlm-syn")  # needed to add for this to work


# dataloading
import math

import einops
import hydra
import numpy as np

# dummy
# import pretty_errors
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.agent.dataset import TorchRLDSInterleavedDataset

# print(pretty_errors.__version__)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)


def load_bridge(config):
    train_dataloader = DataLoader(
        TorchRLDSInterleavedDataset(config.data.train, train=True).dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        collate_fn=custom_collate_fn,  # delete this line to use default collate function
    )
    return train_dataloader


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length trajectories and complex data types.

    Args:
        batch (list): List of individual trajectory samples

    Returns:
        dict: Collated batch with padded trajectories
    """
    # Identify keys in the batch
    keys = batch[0].keys()

    # Prepare output dictionary
    collated_batch = {}

    for key in keys:
        # Collect all values for this key
        values = [item[key] for item in batch]

        # Handle different types of data
        if isinstance(values[0], dict):
            # If the value is a dictionary, keep as list
            collated_batch[key] = values

        elif isinstance(values[0], (np.ndarray, torch.Tensor)):
            # Tensor/Array handling
            try:
                # Try to stack numeric arrays
                if all(
                    isinstance(v, (int, float, np.number))
                    for v in np.concatenate(values)
                ):
                    # Find the maximum length
                    max_len = max(v.shape[0] for v in values)

                    # Pad tensors to the maximum length
                    padded_values = []
                    mask_values = []
                    for v in values:
                        if v.shape[0] < max_len:
                            # Create padding
                            if len(v.shape) == 1:
                                # 1D tensor padding
                                pad_dtype = (
                                    v.dtype if hasattr(v, "dtype") else type(v[0])
                                )
                                pad = np.zeros(
                                    (max_len - v.shape[0],) + v.shape[1:],
                                    dtype=pad_dtype,
                                )
                                padded_v = np.concatenate([v, pad])
                                mask_v = np.concatenate(
                                    [
                                        np.ones(v.shape[0], dtype=bool),
                                        np.zeros(max_len - v.shape[0], dtype=bool),
                                    ]
                                )
                            else:
                                # Multi-dimensional tensor padding
                                pad_dtype = (
                                    v.dtype if hasattr(v, "dtype") else type(v[0][0])
                                )
                                pad_shape = (max_len - v.shape[0],) + v.shape[1:]
                                pad = np.zeros(pad_shape, dtype=pad_dtype)
                                padded_v = np.concatenate([v, pad], axis=0)
                                mask_v = np.concatenate(
                                    [
                                        np.ones(v.shape[0], dtype=bool),
                                        np.zeros(max_len - v.shape[0], dtype=bool),
                                    ]
                                )
                        else:
                            padded_v = v
                            mask_v = np.ones(max_len, dtype=bool)

                        padded_values.append(padded_v)
                        mask_values.append(mask_v)

                    # Convert to tensor
                    collated_batch[key] = torch.tensor(np.stack(padded_values))

                    # Add a mask to indicate valid entries
                    collated_batch[f"{key}_mask"] = torch.tensor(np.stack(mask_values))
                else:
                    # If not all numeric, keep as list
                    collated_batch[key] = values
            except Exception:
                # Fallback to keeping original values if stacking fails
                collated_batch[key] = values

        elif isinstance(values[0], (int, float, str)):
            # Simple types like numbers or strings
            try:
                collated_batch[key] = torch.tensor(values)
            except Exception:
                # If conversion fails, keep as list
                collated_batch[key] = values

        else:
            # For any other type, keep the original list
            collated_batch[key] = values

    return collated_batch


def preprocess_batch(batch):
    # TODO(ajhancock): add history
    images = batch["observation"]["image_primary"]  # should b
    images = einops.rearrange(
        images, "B T H W C -> B (T C) H W"
    ).float()  # move to float for vision encoders  # remove cond_steps dimension
    # proprios = batch["observation"]["proprio"]
    # actions = batch["action"].squeeze(1)  # remove the time dimension
    actions = batch["action"].to("cuda")
    # print(type(actions), actions.device, actions.dtype)  # Debugging step
    # print(f"actions shape: {actions.shape}")  # Debugging step
    # assume horizon and time dimension are the same, 1
    actions = einops.rearrange(actions, "B 1 1 D -> B D")

    texts = [text.decode("utf-8") for text in batch["task"]["language_instruction"]]
    texts = np.array(texts)
    # model_inputs = self.processor(text=texts, images=images)

    # specific to how VLA is defined
    inputs = {
        "image": images.to(torch.bfloat16),
        "actions": actions.to(torch.bfloat16),
    }

    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    return inputs


@hydra.main(
    version_base=None,
    config_path="/n/fs/robot-data/vlm-syn/config/robot_data/",
    config_name="oxe.yaml",
)  # defaults
def main(config: OmegaConf):
    OmegaConf.resolve(config)

    # set seeds
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)

    # load data
    print("Setting up dataloader")
    train_dataloader = DataLoader(
        TorchRLDSInterleavedDataset(config.data.train, train=True).dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        collate_fn=custom_collate_fn,  # delete this line to use default collate function
    )

    max_batches = config.max_batches
    batch_size = config.batch_size
    curr_batch = 0

    np.set_printoptions(
        precision=2, suppress=True
    )  # Limits decimal places, avoids scientific notation
    horizon = config.get("horizon_steps", 1)
    dof = 7  # 6 joints + gripper
    arms = 1

    for batch in train_dataloader:
        if curr_batch >= max_batches:
            break
        # loop through the batch
        if curr_batch % 50 == 0:
            print(f"curr_batch: {curr_batch} of {max_batches}")

        log_dir = config.get("log_dir", None)
        for traj_index in range(len(batch["observation"])):
            print(traj_index)
            dataset_name = batch["dataset_name"][traj_index][0].decode("utf-8")

            save_index = curr_batch * batch_size + traj_index
            save_dir = os.path.join(log_dir, dataset_name, f"traj_{save_index}")
            # don't save if already exists
            os.makedirs(save_dir, exist_ok=True)

            curr_obs = batch["observation"][traj_index]
            curr_task = batch["task"][traj_index]
            curr_action = batch["action"][traj_index]

            curr_image_primary = curr_obs["image_primary"]
            curr_task_completed = curr_obs["task_completed"]

            curr_language_instruction = curr_task["language_instruction"]
            curr_language_instruction = [
                text.decode("utf-8") for text in curr_language_instruction
            ]

            # curr_file_path = batch["file_path_tensor"][
            #     traj_index
            # ]  # trajectory length long
            # curr_file_path = [file_path.decode("utf-8") for file_path in curr_file_path]

            t_horizon = 1
            obs_to_save = list(range(0, len(curr_image_primary), t_horizon))
            obs_to_save.append(-1)  # get last image
            # print(f"obs_to_save: {obs_to_save}")

            init_state = np.zeros((horizon, arms, dof))

            init_state[0, 0, -1] = 1  # gripper open to start
            states = []
            states.append(init_state)

            # create directory for each trajectory
            # delete and override if exists
            traj_dir = save_dir
            # remove any old ifles
            # for file in os.listdir(traj_dir):
            # os.remove(os.path.join(traj_dir, file))

            state_action_path_name = "trajectory_log.txt"
            state_action_path = os.path.join(traj_dir, state_action_path_name)
            with open(state_action_path, "w") as file:
                # file.write(f"file path: {curr_file_path[0]}\n")
                # Write language instruction
                file.write(f"language instruction: {curr_language_instruction[0]}\n")
                state_to_write = np.squeeze(
                    init_state, axis=(0, 1)
                )  # states, actions are of size (H, 1, 7). We set H to 1
                file.write(f"state at t=0: {state_to_write}\n")

                for t in range(len(curr_action)):
                    curr_state = states[t]
                    # print(f"state at t={t} : {curr_state}")
                    action_t = curr_action[t]
                    action_to_write = np.squeeze(action_t, axis=(0, 1))
                    file.write(f"action at t={t} : {action_to_write}\n")
                    next_state = curr_state + action_t
                    # update gripper state
                    next_state[0, 0, -1] = action_t[0, 0, -1]  # gripper
                    state_to_write = np.squeeze(next_state, axis=(0, 1))
                    file.write(f"state at t={t + 1} : {state_to_write}\n")
                    states.append(next_state)

                    # check if t in t_horizon
                    if t in obs_to_save:
                        # save obs
                        image = curr_image_primary[t]
                        image = np.squeeze(image, axis=0)
                        image = Image.fromarray(image)
                        image.save(os.path.join(traj_dir, f"obs_{t}.jpg"))

                    if t == len(curr_action) - 1:
                        # save last image
                        image = curr_image_primary[-1]
                        image = np.squeeze(image, axis=0)
                        image = Image.fromarray(image)
                        image.save(os.path.join(traj_dir, f"obs_{t}.jpg"))

        curr_batch += 1

    print("Done!")


if __name__ == "__main__":
    main()
