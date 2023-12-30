import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def is_binarized(image_path):
    # Check if the image at image_path is already binarized
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    unique_values = np.unique(image)
    return len(unique_values) == 2 and set(unique_values).issubset({0, 255})


def binarize_image(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Unable to read image at {image_path}")
        return

    # Binarize the image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    return output_path, binary_image


def erode_shapes(img, erosion_size):
    # Threshold the image to get the shapes
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Define the erosion kernel
    kernel = np.ones((erosion_size, erosion_size), np.uint8)

    # Erode the image
    eroded_image = cv2.erode(thresh, kernel, iterations=1)

    return eroded_image


def rename_and_binarize(file_path, output_dir, input_dir, erode=False):
    relative_path = os.path.relpath(file_path, input_dir)
    output_path = os.path.join(output_dir, relative_path)

    # Rename the file in the output path if necessary
    dir_name, basename = os.path.split(output_path)
    if basename.startswith('2023') and '_' in basename:
        parts = basename.split('_', 1)
        if len(parts) == 2 and len(parts[0]) == 14:  # Check for a 14-digit prefix
            new_filename = parts[1]
            new_file_path = os.path.join(dir_name, new_filename)
            output_path = new_file_path

    # Only process and copy the file if it doesn't already exist as a binarized file
    if not os.path.exists(output_path) or not is_binarized(output_path):
        # print("Binarized", file_path)
        output_path, binary_image = binarize_image(file_path, output_path)

        if erode:
            # print("Eroded", file_path)
            binary_image = erode_shapes(img=binary_image, erosion_size=5)

        cv2.imwrite(output_path, binary_image)
        # print("Saved", output_path)


def process_files_in_subdirectories(directory, output_dir):
    all_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if
                 file.endswith('.png')]

    for file_path in tqdm(all_files, desc="Processing files"):
        erode = True if "inklabels.png" in file_path else False
        rename_and_binarize(file_path, output_dir, directory, erode=erode)


def main():
    input_dir = os.path.join("data", "labels", "twelve_layer_unetr_it2", "processed")
    output_dir = os.path.join("data", "labels", "twelve_layer_unetr_it2", "binarized")

    process_files_in_subdirectories(input_dir, output_dir)


if __name__ == "__main__":
    main()
