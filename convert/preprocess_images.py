import os
import shutil

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm


def create_lut(control_points):
    # Extract the x and y coordinates of the control points
    x_points, y_points = zip(*control_points)

    # Create a cubic spline passing through the control points
    spline = CubicSpline(x_points, y_points)

    # Generate the LUT by evaluating the spline for each input intensity
    lut = spline(np.arange(256))

    # Clip the LUT to the valid range and convert to uint8
    lut = np.clip(lut, 0, 255).astype('uint8')
    return lut


def apply_lut(image, lut):
    # Apply the LUT to each channel of the image
    if len(image.shape) == 3:
        return cv2.merge([cv2.LUT(channel, lut) for channel in cv2.split(image)])
    else:
        return cv2.LUT(image, lut)


def process_image(file_path, lut, output_dir):
    transformed_path = file_path.replace('data', output_dir)
    if os.path.exists(transformed_path):
        return  # Skip processing if the file already exists

    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is not None:
        transformed_image = apply_lut(image, lut)
        os.makedirs(os.path.dirname(transformed_path), exist_ok=True)
        cv2.imwrite(transformed_path, transformed_image)


def copy_mask(file_dir, output_dir):
    mask_path = os.path.join(file_dir, 'mask.png')
    if os.path.exists(mask_path):
        output_mask_path = mask_path.replace('data', output_dir)
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        shutil.copy(mask_path, output_mask_path)


def main():
    control_points = [(0, 0), (114, 55), (255, 255)]
    lut = create_lut(control_points)

    input_dir = os.path.join('data', 'fragments')
    output_dir = os.path.join('data', 'fragments_contrasted')
    all_files = []

    print("Searching for files in", input_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.tif'):
                all_files.append(os.path.join(root, file))
            if file == 'mask.png':
                copy_mask(root, output_dir)

    print("Processing Files")
    for file_path in tqdm(all_files, desc="Processing images"):
        process_image(file_path, lut, output_dir)


if __name__ == '__main__':
    main()
