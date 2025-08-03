import os
import sys

import google.generativeai as genai
from PIL import Image

API_KEY_ENV_VAR = "GEMINI_VLA_API_KEY"
MODEL_NAME = "gemini-2.5-flash"


def configure_gemini():
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        print(f"Error: Environment variable {API_KEY_ENV_VAR} not set.")
        sys.exit(1)
    genai.configure(api_key=api_key)


def main():
    image_path = "/n/fs/robot-data/vlm-syn/0.png"  # Replace with your image path
    prompt = "Move the apple to the location of the tomato. The positive x-axis is towards the top-left, parallel to the counter's edge. The positive y-axis is towards the bottom-left, parallel to the grid lines. Each grid square is 10cm x 10cm. \
         Output the required displacement as a JSON object with 'x' and 'y' keys, representing the distance in centimeters. For example: {'x': -5, 'y': +5}. Compare the two objects' positions directly to get the answer, and don't use the box as reference."

    image_path_2 = "/n/fs/robot-data/vlm-syn/tomato.png"

    configure_gemini()
    model = genai.GenerativeModel(MODEL_NAME)

    img = Image.open(image_path)
    img_2 = Image.open(image_path_2)

    response = model.generate_content(
        [prompt, img, img_2],
        stream=False,
    )
    if not hasattr(response, "text") or not response.text:
        raise ValueError("Model did not return any text output.")

    print("Model output:", response.text)


if __name__ == "__main__":
    main()
