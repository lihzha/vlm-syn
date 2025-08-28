"""Create per-bridge data folders expected by downstream code."""

import json
import os
import sys

sys.path.insert(0, "/n/fs/robot-data/vlm-syn")  # needed to add for this to work
sys.path.insert(0, "/n/fs/robot-data/vlm-syn/pipeline/utils/molmo")

import time
from typing import Any, Optional, cast

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.agent.dataset import RLDSInterleavedDataloader

# Molmo imports (optional, guarded below)
try:
    from olmo.data import build_mm_preprocessor
    from olmo.model import Molmo
    from olmo.util import extract_points

    _MOLMO_AVAILABLE = True
except Exception:
    # For type-checkers, assign Nones when imports are unavailable
    Molmo = None  # type: ignore[assignment]
    build_mm_preprocessor = None  # type: ignore[assignment]
    extract_points = None  # type: ignore[assignment]
    _MOLMO_AVAILABLE = False


# ---------------- Molmo gripper labeling helpers ---------------- #
_molmo_model: Optional[Any] = None
_molmo_preprocessor: Optional[Any] = None
_hf_molmo_model: Optional[Any] = None
_hf_molmo_processor: Optional[Any] = None

dataset_metadata = {
    "kuka": {
        "traj_identifier": "traj_index",
        "image_key": "image_primary",
        "image_size": (224, 224),
    },
}


# ---------------- Gemini verification helpers ---------------- #
_GEMINI_AVAILABLE = False
_gemini_model: Optional[Any] = None
_GEMINI_API_ENV = "GEMINI_VLA_API_KEY"
_GEMINI_MODEL_NAME = "gemini-2.5-flash"


def _init_gemini() -> bool:
    """Lazy-initialize Gemini model if API key and package are available."""
    global _GEMINI_AVAILABLE, _gemini_model
    if _gemini_model is not None:
        return True
    try:
        import os as _os

        api_key = _os.getenv(_GEMINI_API_ENV)
        if not api_key:
            _GEMINI_AVAILABLE = False
            _gemini_model = None
            return False
        import google.generativeai as _genai  # type: ignore

        _genai.configure(api_key=api_key)
        _gemini_model = _genai.GenerativeModel(_GEMINI_MODEL_NAME)
        _GEMINI_AVAILABLE = True
        return True
    except Exception:
        _GEMINI_AVAILABLE = False
        _gemini_model = None
        return False


def _gemini_json_from_image(prompt: str, pil_image: Image.Image) -> Optional[dict]:
    """Send prompt+image to Gemini and parse a top-level JSON object from its text.

    Returns a dict on success or None on failure.
    """
    if not _init_gemini():
        return None
    try:
        assert _gemini_model is not None
        response = _gemini_model.generate_content([prompt, pil_image], stream=False)
        text = getattr(response, "text", None)
        if not text:
            return None
        # Extract first JSON object braces
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return None
        json_str = text[first : last + 1]
        return json.loads(json_str)
    except Exception:
        return None


def gemini_check_gripper_visible(image_np: np.ndarray) -> Optional[bool]:
    """Return True if gripper visible, False if not, None if unavailable/uncertain."""
    try:
        pil_img = Image.fromarray(image_np)
    except Exception:
        return None
    prompt = (
        "You will receive one image. Determine if a robot gripper/end-effector is visible. "
        "Consider parallel-jaw or suction grippers as valid. If it is not in frame, fully occluded, "
        "or not a robot end-effector, answer false. Respond strictly as JSON with the schema: "
        '{"gripper_visible": true|false}. No extra text.'
    )
    data = _gemini_json_from_image(prompt, pil_img)
    if not data or "gripper_visible" not in data:
        return None
    val = data.get("gripper_visible")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes"}
    return None


def gemini_validate_molmo_label(image_np: np.ndarray, x_px: float, y_px: float) -> Optional[bool]:
    """Return True if Molmo's (x,y) lies on the gripper center, False if inaccurate, None if unsure."""
    try:
        pil_img = Image.fromarray(image_np)
    except Exception:
        return None
    prompt = (
        "You will receive one image and a proposed robot gripper center in pixel coordinates. "
        f"Proposed center: (x={x_px:.1f}, y={y_px:.1f}). "
        "Evaluate if this point is centered on the visible robot gripper/end-effector within ~10 pixels tolerance. "
        "If the gripper is not visible, treat as inaccurate. Respond strictly as JSON: "
        '{"label_is_correct": true|false}. No extra text.'
    )
    data = _gemini_json_from_image(prompt, pil_img)
    if not data or "label_is_correct" not in data:
        return None
    val = data.get("label_is_correct")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes"}
    return None


def _init_molmo_local():
    """Lazily initialize Molmo model and preprocessor from a checkpoint directory.

    To enable, set environment variable MOLMO_CHECKPOINT_DIR to a local Molmo checkpoint directory
    (untarred content containing config.yaml and weights).
    """
    global _molmo_model, _molmo_preprocessor
    if (not _MOLMO_AVAILABLE) or (Molmo is None) or (build_mm_preprocessor is None):
        return False
    if _molmo_model is not None and _molmo_preprocessor is not None:
        return True
    import os as _os

    ckpt_dir = _os.environ.get("MOLMO_CHECKPOINT_DIR")
    if not ckpt_dir:
        return False
    try:
        molmo_cls = cast(Any, Molmo)
        preproc_builder = cast(Any, build_mm_preprocessor)
        _molmo_model = molmo_cls.from_checkpoint(ckpt_dir, device="cuda")
        _molmo_model.eval()  # type: ignore[union-attr]
        _molmo_preprocessor = preproc_builder(
            _molmo_model.config,  # type: ignore[union-attr]
            for_inference=True,
            shuffle_messages=False,
        )
        return True
    except Exception:
        # Failed to initialize Molmo; disable
        _molmo_model = None
        _molmo_preprocessor = None
        return False


def _init_molmo_hf():
    """Initialize Molmo via Hugging Face (trust_remote_code)."""
    global _hf_molmo_model, _hf_molmo_processor
    if _hf_molmo_model is not None and _hf_molmo_processor is not None:
        return True
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor

        _hf_molmo_processor = AutoProcessor.from_pretrained(
            "allenai/Molmo-7B-D-0924", trust_remote_code=True, local_files_only=True
        )
        _hf_molmo_model = AutoModelForCausalLM.from_pretrained(
            "allenai/Molmo-7B-D-0924", trust_remote_code=True, local_files_only=True
        )
        _hf_molmo_model.eval()
        if torch.cuda.is_available():
            _hf_molmo_model.to("cuda")
        return True
    except Exception:
        _hf_molmo_model = None
        _hf_molmo_processor = None
        return False


def _init_molmo():
    """Initialize either local-checkpoint Molmo or HF Molmo.

    Preference: local checkpoint when MOLMO_CHECKPOINT_DIR is set; otherwise try HF.
    """
    # Try local first if explicitly configured
    import os as _os

    if _os.environ.get("MOLMO_CHECKPOINT_DIR"):
        if _init_molmo_local():
            return True
        # fall back to HF
        return _init_molmo_hf()
    # Otherwise try HF first
    if _init_molmo_hf():
        return True
    return _init_molmo_local()


def _extract_points_generic(text: str, image_w: int, image_h: int):
    """Lightweight point extractor for Molmo outputs with <point> or <points> tags."""
    import re as _re

    pts = []
    # Match single point tags
    for m in _re.finditer(
        r'x\s*=\s*"([0-9]+(?:\.[0-9]+)?)"\s*y\s*=\s*"([0-9]+(?:\.[0-9]+)?)"', text
    ):
        x = float(m.group(1))
        y = float(m.group(2))
        if max(x, y) <= 100.0:
            x = x / 100.0 * image_w
            y = y / 100.0 * image_h
        pts.append([x, y])
    return pts


def label_gripper_center_with_molmo(image_np):
    """Predict gripper center (x, y) in pixel coords for a single RGB image (H, W, 3) uint8.

    Returns: (x, y) as floats in pixel coordinates, or None if unavailable.
    """
    if not _init_molmo():
        return None

    h, w = image_np.shape[:2]
    text = None
    # Prefer HF if initialized
    if _hf_molmo_model is not None and _hf_molmo_processor is not None:
        try:
            from transformers import GenerationConfig

            pil_img = Image.fromarray(image_np)
            proc = _hf_molmo_processor
            mdl = _hf_molmo_model
            inputs = proc.process(images=[pil_img], text="Point to the robot gripper center.")
            # to device & batch dim
            inputs = {k: v.to(mdl.device).unsqueeze(0) for k, v in inputs.items()}
            with torch.inference_mode():
                out = mdl.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=128, stop_strings="<|endoftext|>"),
                    tokenizer=proc.tokenizer,
                )
            gen_tokens = out[0, inputs["input_ids"].size(1) :]
            text = proc.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        except Exception:
            text = None

    if text is None and _molmo_model is not None and _molmo_preprocessor is not None:
        # Fallback to local checkpoint path
        example = {
            "image": image_np,
            "style": "pointing",
            "label": "robot gripper center",
        }
        batch_np = _molmo_preprocessor(example, rng=np.random)
        tokenizer = _molmo_preprocessor.tokenizer  # type: ignore[attr-defined]
        input_ids = torch.tensor(
            batch_np["input_tokens"], dtype=torch.long, device="cuda"
        ).unsqueeze(0)
        images = torch.tensor(batch_np["images"], dtype=torch.float32, device="cuda").unsqueeze(0)
        image_input_idx = torch.tensor(
            batch_np["image_input_idx"], dtype=torch.int32, device="cuda"
        ).unsqueeze(0)
        image_masks = None
        if "image_masks" in batch_np:
            image_masks = torch.tensor(
                batch_np["image_masks"], dtype=torch.float32, device="cuda"
            ).unsqueeze(0)
        with torch.inference_mode():
            gen = _molmo_model.generate(  # type: ignore[union-attr]
                input_ids=input_ids,
                images=images,
                image_masks=image_masks,
                image_input_idx=image_input_idx,
                max_steps=128,
                is_distributed=False,
            )
        token_ids = gen.token_ids[0, 0].detach().cpu().numpy()
        text = tokenizer.decode(token_ids, truncate_at_eos=True)

    if text is None:
        return None

    # Parse points
    pts = []
    if extract_points is not None:
        try:
            pts = extract_points(text, w, h)  # type: ignore[operator]
        except Exception:
            pts = []
    if not pts:
        pts = _extract_points_generic(text, w, h)
    if len(pts) == 0:
        return None
    # Return the first point
    xy = pts[0]
    return float(xy[0]), float(xy[1])


def label_gripper_centers_with_molmo_batch(images_np_list):
    """Batch version of gripper center prediction using Molmo when available.

    Args:
        images_np_list: list of RGB images (H, W, 3) uint8.

    Returns:
        List of Optional[(x, y)] in pixel coordinates.
    """
    if not _init_molmo():
        return [None for _ in images_np_list]

    if _hf_molmo_model is not None and _hf_molmo_processor is not None and len(images_np_list) > 0:
        try:
            from transformers.generation.configuration_utils import GenerationConfig

            pil_images = [Image.fromarray(img) for img in images_np_list]
            prompts = ["Point to the robot gripper center."] * len(pil_images)
            proc = _hf_molmo_processor
            mdl = _hf_molmo_model
            inputs = proc.process(images=pil_images, text=prompts)
            inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = mdl.generate_from_batch(
                    inputs,
                    GenerationConfig(
                        max_new_tokens=128,
                        stop_strings="<|endoftext|>",
                        eos_token_id=proc.tokenizer.eos_token_id,
                        pad_token_id=proc.tokenizer.pad_token_id,
                    ),
                    tokenizer=proc.tokenizer,
                )
                generated_texts = proc.tokenizer.batch_decode(
                    out[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
                )

            results = []
            for i, img_np in enumerate(images_np_list):
                txt = generated_texts[i]
                h, w = img_np.shape[:2]
                pts = []
                if extract_points is not None:
                    try:
                        pts = extract_points(txt, w, h)  # type: ignore[operator]
                    except Exception:
                        pts = []
                if not pts:
                    pts = _extract_points_generic(txt, w, h)
                if len(pts) == 0:
                    results.append(None)
                else:
                    xy = pts[0]
                    results.append((float(xy[0]), float(xy[1])))
            return results
        except Exception:
            pass

    # Fallback: serial single-image path
    return [label_gripper_center_with_molmo(img) for img in images_np_list]


@hydra.main(
    version_base=None,
    config_path="/n/fs/robot-data/vlm-syn/config/",
    config_name="oxe.yaml",
)  # defaults
def main(config):
    OmegaConf.resolve(config)

    # load data
    print("Setting up dataloader")
    dataloader = iter(RLDSInterleavedDataloader(config.data.train, train=False, batch_size=1))

    np.set_printoptions(
        precision=3, suppress=True
    )  # Limits decimal places, avoids scientific notation

    # json_path = os.path.join(save_dir, "gripper_pixel_coords.json")

    visualize = True
    pixel_coords = []

    for batch in dataloader:
        # loop through the batch
        log_dir = config.get("log_dir", None)  # type: ignore[call-arg]
        dataset_name = batch["dataset_name"][0][0].decode("utf-8")
        metadata = dataset_metadata[dataset_name]
        os.makedirs(os.path.join(log_dir, dataset_name), exist_ok=True)
        metadata_path = os.path.join(log_dir, dataset_name, "metadata.json")
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w") as file:
                json.dump(metadata, file)

        obs = batch["observation"]

        molmo_gripper_xy = None

        # first_img = obs["image_primary"][0]
        # first_img = np.squeeze(first_img, axis=0)
        # if first_img.dtype != np.uint8:
        #     first_img = np.clip(first_img, 0, 255).astype(np.uint8)
        start_time = time.time()
        first_frames = obs["image_primary"][:, 0, 0]
        molmo_results = label_gripper_centers_with_molmo_batch(first_frames)
        end_time = time.time()
        print(f"Molmo time: {end_time - start_time} seconds")
        if molmo_results is None:
            continue

        # Gemini-in-the-loop filtering per trajectory sample
        for i, molmo_gripper_xy in enumerate(molmo_results):
            img_first = first_frames[i]
            # Check if gripper is visible in the first frame; if not, skip trajectory
            gripper_visible = gemini_check_gripper_visible(img_first)
            if gripper_visible is False:
                continue

            # If Molmo didn't produce a label, skip this sample
            if molmo_gripper_xy is None:
                continue

            # Validate Molmo label with Gemini when available
            x_px, y_px = molmo_gripper_xy
            label_ok = gemini_validate_molmo_label(img_first, float(x_px), float(y_px))
            if label_ok is False:
                continue

            pixel_coords.append(
                {
                    metadata["traj_identifier"]: batch[metadata["traj_identifier"]][i][0],
                    "pixel_coords": [float(x_px), float(y_px)],
                    "label_frame_idx": 0,
                }
            )

        # just for visualization
        if visualize:
            for i, (img, molmo_gripper_xy) in enumerate(zip(first_frames, molmo_results)):
                save_index = batch[metadata["traj_identifier"]][i]
                save_dir = os.path.join(log_dir, dataset_name, f"traj_{save_index}")
                os.makedirs(save_dir, exist_ok=True)

                with open(os.path.join(save_dir, "trajectory_log.txt"), "w") as file:
                    gripper_visible = gemini_check_gripper_visible(img)
                    if gripper_visible is False:
                        file.write("skipped: gripper not visible in first frame (Gemini)\n")
                        continue
                    if molmo_gripper_xy is not None:
                        x_px, y_px = molmo_gripper_xy
                        label_ok = gemini_validate_molmo_label(img, float(x_px), float(y_px))
                        if label_ok is False:
                            file.write(
                                f"skipped: Molmo label rejected by Gemini at ({x_px:.1f}, {y_px:.1f})\n"
                            )
                            continue
                        file.write(
                            f"molmo_gripper_center_px: [{x_px:.1f}, {y_px:.1f}] (Gemini OK)\n"
                        )
                        from PIL import ImageDraw

                        im_vis = Image.fromarray(img).convert("RGB")
                        draw = ImageDraw.Draw(im_vis)
                        x_px, y_px = molmo_gripper_xy
                        # Clamp to image bounds
                        w_vis, h_vis = im_vis.size
                        x_i = int(max(0, min(w_vis - 1, x_px)))
                        y_i = int(max(0, min(h_vis - 1, y_px)))
                        r = max(4, int(0.01 * min(w_vis, h_vis)))
                        draw.ellipse([x_i - r, y_i - r, x_i + r, y_i + r], outline="red", width=3)
                        draw.line([x_i - 2 * r, y_i, x_i + 2 * r, y_i], fill="red", width=2)
                        draw.line([x_i, y_i - 2 * r, x_i, y_i + 2 * r], fill="red", width=2)
                        im_vis.save(os.path.join(save_dir, "obs_gripper.jpg"))

                    else:
                        file.write("molmo_gripper_center_px: None\n")

    print("Done!")


if __name__ == "__main__":
    main()
