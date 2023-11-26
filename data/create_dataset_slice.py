import os
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm
from random import random, sample

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_handler import Config
from PIL import Image


def read_fragment(work_dir, fragment_id, depth):
    images = []
    print(f"attempting to read fragment {fragment_id}")

    for i in tqdm(range(0, depth)):
        img_path = os.path.join(work_dir, "data", "fragments", f"fragment{fragment_id}", "slices", f"{i:05}.tif")
        print(img_path)

        image = cv2.imread(img_path, 0)
        assert 1 < np.asarray(image).max() <= 255, "Invalid image"
        images.append(image)
    images = np.stack(images, axis=0)

    return images

def create_single_val_dataset(data_root_dir, train_split=0.8):
    train_dir = os.path.join(data_root_dir, 'train')
    train_img_dir = os.path.join(train_dir, 'images')

    if not (os.path.exists(train_dir) and os.path.exists(train_img_dir)):
        print("Train directory, images, or labels are missing!")
        sys.exit(1)

    val_dir = os.path.join(data_root_dir, 'val')
    val_img_dir = os.path.join(val_dir, 'images')

    # Create val directories if they don't exist
    os.makedirs(val_img_dir, exist_ok=True)

    train_images = os.listdir(train_img_dir)

    num_to_select = int(len(train_images) * (1 - train_split))
    sampled_images = sample(train_images, min(num_to_select, len(train_images)))

    for image in sampled_images:
        shutil.move(os.path.join(train_img_dir, image), os.path.join(val_img_dir, image))

    print(f"Moved {num_to_select} patch images from train to val")


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
    label_path = os.path.join(fragment_dir, f"inklabels.png")
    target_path = os.path.join(config.work_dir, "data", "datasets", "slice")
    target_train_img_path = os.path.join(target_path, "train", "images")

    os.makedirs(target_path, exist_ok=True)
    os.makedirs(target_train_img_path, exist_ok=True)

    if not os.path.isfile(mask_path):
        raise ValueError(f"Mask file does not exist for fragment: {mask_path}")
    mask_arr = np.asarray(Image.open(mask_path))
    stack = read_fragment(config.work_dir, "2", slice_depth)  # 5 (depth) x (14830) width x (9506) height
    print(stack.shape)

    label_arr = np.asarray(cv2.imread(label_path, cv2.IMREAD_GRAYSCALE))
    label_arr = (label_arr / 255).astype(np.uint8)

    progress_bar = tqdm(total=config.count, desc="Searching patches", unit="iteration")

    big_ink_slices = 0
    small_ink_slices = 0
    no_ink_slices = 0
    saved_samples = 0

    while saved_samples < config.count:
        # Randomly select a patch
        x = np.random.randint(0, stack.shape[1] - slice_length)
        y = np.random.randint(0, stack.shape[2] - slice_length)

        length = slice_length
        width = 1

        # Randomly slice horizontally or vertically
        if random() < 0.5:
            length, width = width, length

        # Check if patch is valid
        mask_patch = mask_arr[x:x + length, y:y + width]
        if mask_patch.sum() == 0:
            continue

        patch = stack[:, x:x + length, y:y + width]
        patch = patch.reshape((slice_length, slice_length))

        label_patch = label_arr[x:x + length, y:y + width]
        label = label_patch.sum() / (length * width)
        label = int(label * 100)

        if label == 0 and (big_ink_slices + small_ink_slices) < no_ink_slices:
            continue

        if label > 50:
            big_ink_slices += 1
        elif label > 0:
            small_ink_slices += 1
        else:
            no_ink_slices += 1

        # Set filename to label, with 2 decimals
        filename = f"{saved_samples}_{label}.npy"
        patch_path = os.path.join(target_train_img_path, filename)
        np.save(patch_path, patch)
        saved_samples += 1
        progress_bar.update(1)

    print(f"Total:\t\t\t\t{config.count}")
    print(f"Slices with > 50% ink:\t\t{big_ink_slices}")
    print(f"Slices with 1-50% ink:\t\t{small_ink_slices}")
    print(f"Slices with 0% ink:\t\t{no_ink_slices}")

    create_single_val_dataset(data_root_dir=target_path,
                              )


