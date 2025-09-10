import json
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

sys.path.append("/n/fs/robot-data/vlm-syn")

from droid_utils.gen_language_actions import (
    describe_base_movement,
)

os.environ["CURL_CA_BUNDLE"] = (
    "/etc/pki/tls/certs/ca-bundle.crt"  # Ensure the CA bundle is set for SSL verification
)

data_name = "droid"
# data_dir = "gs://gresearch/robotics"
# data_dir = "/n/fs/vla-mi/datasets/OXE/"
data_dir = "/n/fs/robot-data/data"
save_dir = "/n/fs/robot-data/vlm-syn/droid_base_lang_actions_json/"
os.makedirs(save_dir, exist_ok=True)


# builder = tfds.builder_from_directory(builder_dir="gs://gresearch/robotics/droid/1.0.1")
# ds = builder.as_dataset(split="test")

ds = tfds.load(data_name, data_dir=data_dir, split="train")
path_to_droid_repo = "/n/fs/robot-data/vlm-syn/droid_utils"

ds = ds.prefetch(tf.data.AUTOTUNE)
ds = ds.map(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

# Load the extrinsics
cam2base_extrinsics_path = f"{path_to_droid_repo}/cam2base_extrinsics.json"
with open(cam2base_extrinsics_path, "r") as f:
    cam2base_extrinsics = json.load(f)

# Load the intrinsics
intrinsics_path = f"{path_to_droid_repo}/intrinsics.json"
with open(intrinsics_path, "r") as f:
    intrinsics = json.load(f)

# Load mapping from episode ID to path, then invert
episode_id_to_path_path = f"{path_to_droid_repo}/episode_id_to_path.json"
with open(episode_id_to_path_path, "r") as f:
    episode_id_to_path = json.load(f)
episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}

# Load camera serials
camera_serials_path = f"{path_to_droid_repo}/camera_serials.json"
with open(camera_serials_path, "r") as f:
    camera_serials = json.load(f)

for i, example in enumerate(tqdm(ds)):
    file_path = example["episode_metadata"]["file_path"].numpy().decode()
    episode_path = file_path.split("r2d2-data-full/")[1].split("/trajectory")[0]
    if episode_path not in episode_path_to_id:
        continue
    episode_id = episode_path_to_id[episode_path]

    print(f"\n--- Processing episode {episode_id} ({i + 1}/{len(ds)}) ---")

    if os.path.exists(os.path.join(save_dir, f"{episode_id}_language_action.json")):
        print(f"Already processed {episode_id}")
        continue  # already processed
    if episode_id not in cam2base_extrinsics:
        continue
    if episode_id not in intrinsics:
        continue
    if episode_id not in camera_serials:
        continue

    # pick the calibration camera for this episode (as you did earlier)
    extr = cam2base_extrinsics[episode_id]
    intr = intrinsics[episode_id]
    cams = camera_serials[episode_id]

    # ----------- compute gripper pose in cam frame -----------
    for k, v in extr.items():
        if k.isdigit():
            camera_serial = k
            extracted_extrinsics = v
            break

    # Also lets us get the intrinsics
    if camera_serial not in intr:
        continue
    extracted_intrinsics = intr[camera_serial]

    # Using the camera serial, find the corresponding camera name (which is used to determine
    # which image stream in the episode to use)
    camera_serials_to_name = {v: k for k, v in cams.items()}
    if camera_serial not in camera_serials_to_name:
        continue

    gripper_actions = []
    cartesian_poses = []
    for curr_step in example["steps"]:
        gripper_actions.append(curr_step["action_dict"]["gripper_position"].numpy().item())
        cartesian_pose = curr_step["observation"]["cartesian_position"].numpy()
        cartesian_poses.append(cartesian_pose)

    language_actions = describe_base_movement(
        gripper_poses=np.array(cartesian_poses),
        gripper_actions=np.array(gripper_actions),
    )
    # language_actions is a list of string. Compress it and save it
    with open(os.path.join(save_dir, f"{episode_id}_language_action.json"), "w") as f:
        json.dump(language_actions, f)
        print(f"Saved {episode_id}_language_action.json")
