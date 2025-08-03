import glob
import os

from PIL import Image


def main():
    top_folder = "/n/fs/robot-data/vlm-syn/data/oxe/berkeley_autolab_ur5"
    pattern = os.path.join(top_folder, "*/gripper_point_vis_2.png")
    image_paths = sorted(glob.glob(pattern))
    scores = {0: 0, 1: 0, 2: 0}

    for img_path in image_paths:
        print(f"Showing: {img_path}")
        img = Image.open(img_path)
        img.show()
        while True:
            try:
                score = int(input("Rate this image (0, 1, 2): "))
                if score in [0, 1, 2]:
                    scores[score] += 1
                    break
                else:
                    print("Please enter 0, 1, or 2.")
            except Exception:
                print("Invalid input. Please enter 0, 1, or 2.")
        img.close()

    print("\nScore summary:")
    for k in [0, 1, 2]:
        print(f"Score {k}: {scores[k]}")


if __name__ == "__main__":
    main()
