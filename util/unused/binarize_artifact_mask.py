import argparse
import os
import re
import sys

import cv2
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def process_image(img_src_path):
    try:
        # Extract the fragment ID using regular expression
        frag_id_match = re.search(r'\d+', img_src_path)
        if not frag_id_match:
            raise ValueError("No fragment ID found in the filename")
        frag_id = frag_id_match.group()

        # Generate target image path
        target_path = f"data/fragments/fragment{frag_id}/"
        file_name = "artifact_mask.png"
        img_target_path = os.path.join(target_path, file_name)

        os.makedirs(target_path, exist_ok=True)

        # Load label image
        label_img = cv2.imread(img_src_path, 0)
        label_arr = np.asarray(label_img)

        # Binarize the label
        binary_label = np.where(label_arr > 1, 1, 0)

        # Scale the binary_label to full 8-bit range
        scaled_label = (binary_label * 255).astype(np.uint8)

        # Save the scaled image
        cv2.imwrite(img_target_path, scaled_label)

        return True, img_target_path
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Process binary artifact mask files in a directory.')
    parser.add_argument('directory', nargs='?', type=str, default="data/base_artifact_mask_files",
                        help='Path to the directory with source images. '
                             'Defaults to data/base_artifact_mask_files if not provided.')

    args = parser.parse_args()

    success_count = 0
    failure_count = 0

    for file in os.listdir(args.directory):
        if file.endswith(".png") and "artifact_mask" in file:
            img_src_path = os.path.join(args.directory, file)
            success, message = process_image(img_src_path)
            if success:
                print(colored(f"Success: Processed {message}", "green"))
                success_count += 1
            else:
                print(colored(f"Failure: {message}", "red"))
                failure_count += 1

    print("\nReport:")
    print(colored(f"Total Successful: {success_count}", "green"))
    print(colored(f"Total Failures: {failure_count}", "red"))


if __name__ == "__main__":
    main()
