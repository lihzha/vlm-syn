import argparse
import os
import re
from typing import Optional, Tuple

from PIL import Image, ImageDraw


def parse_gripper_point(log_path: str) -> Optional[Tuple[float, float]]:
    """Parse molmo_gripper_center_px from a trajectory log file.

    Expected line format in the log:
      molmo_gripper_center_px: [x, y]
    Returns (x, y) in pixels, or None if not found or None.
    """
    if not os.path.isfile(log_path):
        return None
    pattern = re.compile(
        r"molmo_gripper_center_px:\s*\[\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*\]"
    )
    with open(log_path, "r") as f:
        for line in f:
            if "molmo_gripper_center_px:" not in line:
                continue
            if "None" in line:
                return None
            m = pattern.search(line)
            if m:
                x = float(m.group(1))
                y = float(m.group(2))
                return x, y
    return None


def draw_marker(
    image_path: str, point_xy: Tuple[float, float], out_path: str, color: str = "red"
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x_px, y_px = point_xy
    w, h = image.size
    x_i = int(max(0, min(w - 1, x_px)))
    y_i = int(max(0, min(h - 1, y_px)))
    r = max(4, int(0.01 * min(w, h)))
    draw.ellipse([x_i - r, y_i - r, x_i + r, y_i + r], outline=color, width=3)
    draw.line([x_i - 2 * r, y_i, x_i + 2 * r, y_i], fill=color, width=2)
    draw.line([x_i, y_i - 2 * r, x_i, y_i + 2 * r], fill=color, width=2)
    image.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Molmo-labeled gripper point for a trajectory."
    )
    parser.add_argument(
        "traj_dir",
        type=str,
        help="Path to trajectory directory (contains obs_*.jpg and trajectory_log.txt)",
    )
    parser.add_argument(
        "--frame", type=int, default=0, help="Frame index to visualize (default: 0)"
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="Optional explicit image filename (overrides --frame)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output annotated image path (defaults to <image> with _gripper suffix)",
    )
    parser.add_argument("--color", type=str, default="red", help="Marker color (default: red)")
    args = parser.parse_args()

    traj_dir = os.path.abspath(args.traj_dir)
    log_path = os.path.join(traj_dir, "trajectory_log.txt")
    point = parse_gripper_point(log_path)
    if point is None:
        raise SystemExit(f"No gripper point found in {log_path} (or value is None)")

    if args.image_name is not None:
        image_path = os.path.join(traj_dir, args.image_name)
    else:
        image_path = os.path.join(traj_dir, f"obs_{args.frame}.jpg")

    if not os.path.isfile(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    if args.out is not None:
        out_path = args.out
    else:
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_gripper{ext}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    draw_marker(image_path, point, out_path, color=args.color)
    print(f"Saved annotated image to: {out_path}")


if __name__ == "__main__":
    main()
