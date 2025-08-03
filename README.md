# Cross Embodiment Generalization Data Labelling

A pipeline for labelling language actions for cross embodiment learning. This repo lets you

1. **Ingest & pre‑process the OXE dataset** with `pipeline/01_load_oxe_dataset.py`.
2. **Explore Gemini‑based interaction** via `pipeline/gemini_playground.py` (optional).

---

## 1. Installation

```bash
# Clone the repository
git clone git@github.com:lihzha/vlm-syn.git
cd vlm‑syn

conda env create -f environment.yaml

conda activate ceg
```

---

## 2. Data Loading and Processing

The script provides an example about how to load the OXE dataset. You can design your data processing script based on this.

```bash
python pipeline/load_oxe_dataset.py \
       --input  data/raw/oxe_dataset \
       --output data/processed
```

---

## 3. Gemini Playground

Interact with processed images or run quick prompts through Gemini.
Requires a **Google API key** set as an environment variable (ask Lihan if you need one):

```bash
export GEMINI_VLA_API_KEY="<your‑key>"
# Modify the main function to load corresponding images and prompts
python pipeline/gemini_playground.py
```

---

## 4. Gripper Label and Tracking (only for Michael for now)

To try gripper labelling and tracking, you should look at `pipeline/gripper_label_and_track`. You will first need to implement `pipeline/gripper_label_and_track/utils/get_gripper_pos.py` using Molmo:
1. Molmo online playground: https://playground.allenai.org/
2. Using Molmo in code: https://github.com/allenai/molmo

Then, to initialize the tracking repo, run:
```bash
git submodule update --init --recursive

cd pipeline/gripper_label_and_track/utils/point_trackers/tapnet
pip install .

mkdir checkpoints
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
```

After this, you should be able to run `pipeline/gripper_label_and_track/fit_robo2cam.py`, which gives you the robot end effector trajectories in the camera frame.

---

## 5. Labelling human video (only for Jeremy for now)

I don't have any related scripts yet. In my mind, the overall pipeline should be:
1. Prepare a dataset: e.g. HOI4D and HOT3D
2. Label human hand poses: you can use modules like HaMeR (https://geopavlakos.github.io/hamer/). For some of the datasets the human pose information is already available so you don't even need to label.
3. Turn into language actions: you can apply some heuristics to turn human hand poses into language actions like "move left 4cm, rotate yaw clockwise for 20 degrees, and close the gripper."