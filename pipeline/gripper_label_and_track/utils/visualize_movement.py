#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Describe robot end-effector motions in plain English using **camera-frame**
poses that were saved by the calibration script (``ee_camera_frame.npz``).

It prints a sentence for every consecutive frame like::

    frame 000 → 001: move right 2.1 cm and up 0.9 cm and rotate yaw -19.8°

Fix for axis mix-ups
--------------------
Users reported left/right flips and "up" being labelled backward.  Those are
pure sign/axis-ordering issues that depend on *your* camera convention and how
the calibration recovered the extrinsics.  Adjust the ``AXIS_PERM`` and
``AXIS_SIGN`` constants below and re-run - no other changes needed.

Coordinate conventions assumed here (OpenCV default):
    * **X (right)**  - increases to the right in the image.
    * **Y (down)**   - increases downward in the image.
    * **Z (forward)**- increases *into* the scene (away from camera).

Natural-language mapping implemented:
    +X → right   | -X → left
    -Y → up      | +Y → down
    +Z → forward | -Z → backward

If your system uses a different handedness (e.g., +X left, +Y up, +Z out of
image, ROS REP 103, etc.), just tweak the two arrays and you are done.
"""

import argparse
import os
from glob import glob
from typing import List

import cv2 as cv
import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R

# ───────────────────────── user-tweakable constants ────────────────────────── #

# Re-order camera-frame (x,y,z) to (x',y',z') used for NL description.
# Example: (0,1,2) keeps XYZ order; (1,2,0) would map cam-Y→x', cam-Z→y', cam-X→z'.
AXIS_PERM = np.array([0, 1, 2])  # do **not** change order unless axes are swapped

# Flip signs on the permuted axes.  +1 keeps sign, -1 flips sign.
# If your script said "right" when the robot moved left, set first element to -1.
AXIS_SIGN = np.array([-1, 1, 1])  # fix LEFT/RIGHT flip & keep others as-is

# Threshold (in cm) below which a motion component is considered negligible.
THRESH_CM = 0.3

ARROW_SCALE_PX_PER_CM = 30  # 1 cm motion = 5 pixels arrow length
FONT = 1

# ───────────────────────── helper functions ──────────────────────────────── #


def describe_translation(t_cam: np.ndarray) -> List[str]:
    """Return list of NL phrases for a 3-D translation *in metres*."""
    # Permute + flip + convert to cm ----------------------------------------
    v = (AXIS_SIGN * t_cam[AXIS_PERM]) * 100.0  # centimetres after transform
    dx, dy, dz = v  # semantics after transform: right, down, forward axes

    parts = []
    if abs(dx) > THRESH_CM:
        parts.append(f"move {'right' if dx > 0 else 'left'} {abs(dx):.1f} cm")
    if abs(dy) > THRESH_CM:
        parts.append(f"move {'forward' if dy > 0 else 'backward'} {abs(dy):.1f} cm")
    if abs(dz) > THRESH_CM:
        parts.append(f"move {'down' if dz > 0 else 'up'} {abs(dz):.1f} cm")
    return parts


def describe_rotation(dR: np.ndarray) -> str:
    """Return a yaw-only phrase (ignore pitch/roll for simplicity)."""
    yaw_deg = R.from_matrix(dR).as_euler("zyx", degrees=True)[0]
    if abs(yaw_deg) < 1.0:
        return None
    return f"rotate yaw {yaw_deg:+.1f}°"


################################################################################
# VISUALISATION
################################################################################


def _draw_arrow(
    frame: np.ndarray, t_cam: np.ndarray, text: str, scale_px_cm: float
) -> np.ndarray:
    """Overlay arrow and NL text onto *frame* and return a new image."""
    img = frame.copy()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 2-D arrow tip ----------------------------------------------------------
    v = (AXIS_SIGN * t_cam[AXIS_PERM]) * 100.0  # cm (dx,dy,dz)
    dx_cm, _, dz_cm = v  # arrow only uses right/left & up/down

    end_x = int(cx + dx_cm * scale_px_cm)
    end_y = int(cy + dz_cm * scale_px_cm)  # +dz (down) increases y

    cv.arrowedLine(img, (cx, cy), (end_x, end_y), (0, 0, 255), 2, tipLength=0.25)

    # put text at top-left ---------------------------------------------------
    cv.putText(img, text, (10, 30), FONT, 0.7, (0, 255, 0), 2, cv.LINE_AA)

    # draw a circle
    radius = 5
    cv.circle(img, (cx, cy), radius, (255, 0, 0), -1)
    return img


def visualize_movement(
    frames_path: str,
    T_cam_ee_path: str,
    out_path: str = "motion_vis.mp4",
    fps: int = 3,
) -> None:
    """Save *out_path* video with arrow + text overlays for each transition."""
    frames = load_frames_from_dir(frames_path)
    T_cam_ee = np.load(T_cam_ee_path)["T_cam_ee"]

    txt_path = os.path.splitext(out_path)[0] + ".txt"

    with (
        imageio.get_writer(out_path, fps=fps, codec="libx264", quality=4) as writer,
        open(txt_path, "a") as f,
    ):
        writer.append_data(frames[0][..., ::-1])  # BGR to RGB

        for i in range(len(T_cam_ee) - 1):
            dT = np.linalg.inv(T_cam_ee[i]) @ T_cam_ee[i + 1]
            t = dT[:3, 3]
            dR = dT[:3, :3]

            parts = describe_translation(t)
            rot_part = describe_rotation(dR)

            if rot_part:
                parts.append(rot_part)
            sentence = " and ".join(parts) if parts else "(negligible motion)"
            frame_vis = _draw_arrow(frames[i + 1], t, sentence, ARROW_SCALE_PX_PER_CM)

            writer.append_data(frame_vis[..., ::-1])  # BGR to RGB
            f.write(f"frame {i:03d} → {i + 1:03d}: {sentence}\n")


def _visualise_movements(
    frames: List[np.ndarray],
    T_cam_ee: np.ndarray,
    out_path: str = "motion_vis.mp4",
    fps: int = 3,
) -> None:
    """Save *out_path* video with arrow + text overlays for each transition."""
    assert len(frames) == len(T_cam_ee)

    h, w = frames[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    vw = cv.VideoWriter(out_path, fourcc, fps, (w, h))

    vw.write(frames[0])

    for i in range(len(T_cam_ee) - 1):
        dT = np.linalg.inv(T_cam_ee[i]) @ T_cam_ee[i + 1]
        t = dT[:3, 3]
        dR = dT[:3, :3]

        parts = describe_translation(t)
        rot_part = describe_rotation(dR)

        if rot_part:
            parts.append(rot_part)
        sentence = " and ".join(parts) if parts else "(negligible motion)"
        frame_vis = _draw_arrow(frames[i + 1], t, sentence, ARROW_SCALE_PX_PER_CM)
        vw.write(frame_vis)
        print(f"frame {i:03d} → {i + 1:03d}: {sentence}")

    # write the last frame without arrow (copy previous sentence)
    vw.release()
    print(f"[✓] Video saved to {out_path}")


# ───────────────────────────── main routine ──────────────────────────────── #

################################################################################
# MAIN
################################################################################


def load_frames_from_dir(dir_path: str) -> List[np.ndarray]:
    """Load images/*.png|jpg in alphanumeric order."""
    paths = (
        sorted(glob(os.path.join(dir_path, "*.png")))
        + sorted(glob(os.path.join(dir_path, "*.jpg")))
        + sorted(glob(os.path.join(dir_path, "*.jpeg")))
    )
    if not paths:
        raise FileNotFoundError(f"no images in {dir_path}")
    return [cv.imread(p) for p in paths]


def load_frames_from_video(video_path: str) -> List[np.ndarray]:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def main(argv: List[str] = None) -> None:
    p = argparse.ArgumentParser(description="Describe & visualise robot motion")
    p.add_argument("poses", help="ee_camera_frame.npz from calibration")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="directory containing RGB frames")
    src.add_argument("--video", help="input video file")

    p.add_argument("--out", default="motion_vis.mp4", help="output video path")
    p.add_argument("--fps", type=int, default=3, help="output FPS (default 3)")
    args = p.parse_args(argv)

    # load camera-frame poses ----------------------------------------------
    data = np.load(args.poses)
    T_cam_ee = data["T_cam_ee"]  # (N,4,4)

    # load frames -----------------------------------------------------------
    if args.images:
        frames = load_frames_from_dir(args.images)
    else:
        frames = load_frames_from_video(args.video)

    assert len(frames) == len(T_cam_ee), (
        "#frames ≠ #poses - make sure they correspond 1-to-1"
    )

    _visualise_movements(frames, T_cam_ee, args.out, args.fps)


if __name__ == "__main__":
    main()
