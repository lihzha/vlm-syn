# Cross Embodiment Generalization Data Labelling

A pipeline for labelling language actions for cross embodiment learning. This repo lets you

1. **Ingest & pre‑process the OXE dataset** with `pipeline/01_load_oxe_dataset.py`.
2. **Explore Gemini‑based interaction** via `pipeline/gemini_playground.py` (optional).

---

## 1. Installation

```bash
# Clone the repository
git clone git@github.com:lihzha/oxe-language-label.git
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

To try gripper labelling and tracking, you should look at `pipeline/gripper_label_and_track`. You will need to implement `pipeline/gripper_label_and_track/utils/get_gripper_pos.py` using Molmo:
1. Molmo online playground: https://playground.allenai.org/
2. Using Molmo in code: https://github.com/allenai/molmo

```bash
export GEMINI_VLA_API_KEY="<your‑key>"
# Modify the main function to load corresponding images and prompts
python pipeline/gemini_playground.py
```
