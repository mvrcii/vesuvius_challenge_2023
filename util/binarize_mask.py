import argparse
import os

import cv2
import numpy as np


def process_image(img_src_path):
    # Generate target image path
    base, ext = os.path.splitext(img_src_path)
    img_target_path = f"{base}_processed{ext}"

    # Load label image
    label_img = cv2.imread(img_src_path, 0)
    label_arr = np.asarray(label_img)

    # Binarize the label
    binary_label = np.where(label_arr > 1, 1, 0)

    # Scale the binary_label to full 8-bit range
    scaled_label = (binary_label * 255).astype(np.uint8)

    # Save the scaled image
    cv2.imwrite(img_target_path, scaled_label)

    return img_target_path


def main():
    parser = argparse.ArgumentParser(description='Process a binary mask image.')
    parser.add_argument('img_src_path', type=str, help='Path to the source image')

    args = parser.parse_args()

    img_target_path = process_image(args.img_src_path)
    print(f"Processed image saved to: {img_target_path}")


if __name__ == "__main__":
    main()
