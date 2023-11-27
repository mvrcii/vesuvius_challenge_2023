import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_handler import Config
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def read_fragment(work_dir, fragment_id, depth):
    images = []

    # Create a tqdm object with the range
    with tqdm(range(0, depth)) as t:
        for i in t:
            img_path = os.path.join(work_dir, "data", "fragments", f"fragment{fragment_id}", "slices", f"{i:05}.tif")

            # Set the description of the tqdm object to the current file ID
            t.set_description(f"Processing {i:05}.tif")

            image = np.asarray(cv2.imread(img_path, 0))
            image = image / 255
            images.append(image)

    images = np.stack(images, axis=0)
    return images


def get_sys_args():
    if len(sys.argv) < 2:
        print("Usage: python ./data/create_dataset_slice.py <config_path>")
        sys.exit(1)

    return sys.argv[1]


if __name__ == '__main__':
    # Load config
    config_path = get_sys_args()
    config = Config.load_from_file(config_path)
    slice_length = config.slice_length
    slice_depth = config.slice_depth

    fragment_id = "2"
    fragment_dir = os.path.join(config.work_dir, "data", "fragments", f"fragment{fragment_id}")
    slices_path = os.path.join(fragment_dir, "slices")
    mask_path = os.path.join(fragment_dir, f"mask.png")
    target_path = os.path.join(config.work_dir, "data", "slice_inference_results")
    os.makedirs(target_path, exist_ok=True)

    if not os.path.isfile(mask_path):
        raise ValueError(f"Mask file does not exist for fragment: {mask_path}")
    mask_arr = np.asarray(Image.open(mask_path))
    stack = read_fragment(config.work_dir, "2", 64)  # 5 (depth) x (14830) width x (9506) height
    height, width = stack[0].shape

    x = 0
    y = 0
    out_x = 0
    out_y = 0
    out_height = height // slice_length
    out_width = width // slice_length
    progress_bar = tqdm(total=out_height*out_width, desc="Doing inference", unit="iteration")
    result = np.zeros((out_height + 10, out_width + 10))

    ink_7_27 = 0.411844052997497
    no_ink_7_27 = 0.40435661216492697
    # ink_10_29 = 0.41478473769614066
    # no_ink_10_29 = 0.40962726891154916
    # ink_30_46 = 0.3767836464618817
    # no_ink_30_46 = 0.37998289621719006

    while x < width - slice_length:
        while y < height - slice_length:
            progress_bar.update(1)
            # Check if patch is valid
            mask_patch = mask_arr[y:y + slice_length, x:x + slice_length]
            if mask_patch.sum() == 0:
                y += slice_length
                out_y += 1
                continue

            cube = stack[:, y:y + slice_length, x:x + slice_length]
            mean_stack = np.mean(cube, axis=(1, 2))
            mean_7_27 = np.mean(mean_stack[7:28])
            if abs(ink_7_27 - mean_7_27) < abs(no_ink_7_27 - mean_7_27):
                result[out_y, out_x] = 1

            y += slice_length
            out_y += 1
        x += slice_length
        out_x += 1
    plt.imshow(result, cmap='gray')
    plt.show()
