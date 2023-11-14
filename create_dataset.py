import argparse
import logging
import os
import random
import shutil

import cv2
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

from conf import CFG


def read_image(fragment_id):
    images = []

    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)

    for i in tqdm(idxs):
        img_path = os.path.join(CFG.fragment_root_dir, "fragments", f"fragment{fragment_id}", "slices", f"{i:05}.tif")

        image = cv2.imread(img_path, 0)
        assert 1 < np.asarray(image).max() <= 255, f"Invalid image"

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size) % CFG.tile_size
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size) % CFG.tile_size

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=0)

    label_path = os.path.join(CFG.fragment_root_dir, "fragments/fragment2/inklabels.png")
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label = np.pad(label, [(0, pad0), (0, pad1)], mode='constant', constant_values=0)
    label = (label / 255).astype(np.uint8)
    assert set(np.unique(np.array(label))) == {0, 1}, "Invalid label"

    return images, label


def create_dataset(data_root_dir, fragment_id=2):
    data_dir = os.path.join(data_root_dir, 'train')

    img_path = os.path.join(data_dir, "images")
    label_path = os.path.join(data_dir, "labels")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    images, label = read_image(fragment_id)

    x1_list = list(range(0, images.shape[2] - CFG.tile_size + 1, CFG.stride))
    y1_list = list(range(0, images.shape[1] - CFG.tile_size + 1, CFG.stride))

    progress_bar = tqdm(total=len(x1_list) * len(y1_list), desc="Train Dataset: Processing images and labels")

    skip_counter_black_image = 0
    skip_counter_label = 0
    skip_counter_low_ink = 0
    requirement = (128 ** 2) * 0.1

    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            progress_bar.update(1)

            img_patch = images[:, y1:y2, x1:x2]

            # Check that the train image is not full black
            if img_patch.max() == 0:
                skip_counter_black_image += 1
                continue

            # Scale label down to match segformer output
            label_patch = resize(label[y1:y2, x1:x2], (128, 128), order=0, preserve_range=True, anti_aliasing=False)

            # Check that the label has two classes
            if len(np.unique(label_patch)) != 2:
                skip_counter_label += 1
                continue

            # Check that the label contains at least 10% ink
            if label_patch.sum() <= requirement:
                skip_counter_low_ink += 1
                continue

            file_name = f"{x1}_{y1}_{x2}_{y2}.npy"
            img_file_path = os.path.join(img_path, file_name)
            label_file_path = os.path.join(label_path, file_name)

            np.save(img_file_path, img_patch)
            np.save(label_file_path, label_patch)

    progress_bar.close()

    print("Patches skipped due to unary label:", skip_counter_label)
    print("Patches skipped due to black source image:", skip_counter_black_image)
    print("Patches skipped due to low ink:", skip_counter_low_ink)


def move_files(src_dir, dest_dir, files, pbar):
    for file in files:
        shutil.move(os.path.join(src_dir, file), os.path.join(dest_dir, file))
        pbar.update(1)


def create_val_from_train(data_root_dir, train_split=0.8):
    train_dir = os.path.join(data_root_dir, 'train')
    train_img_dir = os.path.join(train_dir, 'images')
    train_label_dir = os.path.join(train_dir, 'labels')

    if not (os.path.exists(train_dir) and os.path.exists(train_img_dir) and os.path.exists(train_label_dir)):
        logging.error("Train directory, images, or labels are missing!")

    val_dir = os.path.join(data_root_dir, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_label_dir = os.path.join(val_dir, 'labels')

    # Create val directories if they don't exist
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # List all files in the train images and labels directories
    image_files = os.listdir(train_img_dir)
    label_files = os.listdir(train_label_dir)

    # Assuming the image and label files have a one-to-one correspondence and the same naming convention
    assert len(image_files) == len(label_files)

    # Randomly select 20% of the files
    num_files_to_select = int(len(image_files) * (1 - train_split))
    selected_files = random.sample(image_files, num_files_to_select)

    progress_bar = tqdm(total=len(image_files) * 2, desc="Validation Dataset: Processing images and labels")

    # Move the selected image and label files
    move_files(train_img_dir, val_img_dir, selected_files, pbar=progress_bar)
    move_files(train_label_dir, val_label_dir, selected_files, pbar=progress_bar)

    progress_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset.')
    parser.add_argument('--patch_size', type=int, default=512, help='Size of the patch.')
    parser.add_argument('--split', type=float, default=0.8, help='Train and validation split.')
    parser.add_argument('--skip_train', action="store_true", help='Skip the creation of train dataset.')

    args = parser.parse_args()

    # Update CFG with the patch_size argument
    CFG.tile_size = args.patch_size
    CFG.size = CFG.tile_size

    # train
    if not args.skip_train:
        create_dataset(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)))

    # val
    create_val_from_train(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)), train_split=args.split)
