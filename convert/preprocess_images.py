import os
import shutil

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm


def apply_lut_16bit(image, lut):
    # Manually apply the LUT for 16-bit images
    if len(image.shape) == 2:  # Grayscale
        transformed_image = lut[image]
    elif len(image.shape) == 3:  # Color
        transformed_image = np.stack([lut[image[:, :, i]] for i in range(3)], axis=-1)
    else:
        raise ValueError("Unsupported image format")

    return transformed_image


def create_lut_16bit(control_points):
    # Adjust the control points for 16-bit range
    # For example, scale them up from the 8-bit range
    control_points_16bit = [(x * 257, y * 257) for x, y in control_points]

    # Rest of the code remains the same, but adjust the range for LUT generation
    x_points, y_points = zip(*control_points_16bit)
    spline = CubicSpline(x_points, y_points)
    lut = spline(np.linspace(0, 65535, 65536))

    lut = np.clip(lut, 0, 65535).astype('uint16')
    return lut


def process_image(file_path, lut, output_dir):
    transformed_path = file_path.replace('data', output_dir)
    if os.path.exists(transformed_path):
        return  # Skip processing if the file already exists

    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is not None:
        transformed_image = apply_lut_16bit(image, lut)

        os.makedirs(os.path.dirname(transformed_path), exist_ok=True)
        cv2.imwrite(transformed_path, transformed_image)


def copy_mask(file_dir, output_dir):
    mask_path = os.path.join(file_dir, 'mask.png')
    if os.path.exists(mask_path):
        output_mask_path = mask_path.replace('data', output_dir)
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        shutil.copy(mask_path, output_mask_path)


def main():
    control_points = [(0, 0), (128, 64), (255, 255)]
    lut = create_lut_16bit(control_points)
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
