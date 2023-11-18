import argparse
import os
import re
import sys

import cv2
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from conf import CFG


def process_image(img_src_path, target_dir):
    try:
        # Generate target image path
        file_name = "inferenced_scroll.png"
        target_path = os.path.join(target_dir, file_name)

        # Load label image
        label_img = np.load(img_src_path)
        label_arr = np.asarray(label_img)

        # Binarize the label
        binary_label = np.where(label_arr > 1, 1, 0)

        # Scale the binary_label to full 8-bit range
        scaled_label = (binary_label * 255).astype(np.uint8)

        # Save the scaled image
        cv2.imwrite(target_path, scaled_label)

    except Exception as e:
        return False, str(e)


def main():
    config = load_config(CFG)

    dir = "inference/results/fragment20231024093300/20231118-023621/"
    img_src_path = os.path.join(dir, "frag_20231024093300_20231118-023621result.npy")
    process_image(img_src_path, dir)



if __name__ == "__main__":
    main()
