import argparse
import logging
import multiprocessing
import os
import random
import shutil

import cv2
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from conf import CFG


def read_fragment(fragment_id):
    images = []
    pad0, pad1 = None, None

    fragment_dir = os.path.join(CFG.fragment_root_dir, "fragments", f"fragment{fragment_id}")
    assert os.path.isdir(fragment_dir), "Fragment directory does not exist"

    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    layers = range(start, end)

    for layer in tqdm(layers):
        img_path = os.path.join(fragment_dir, "slices", f"{layer:05}.tif")
        assert os.path.isfile(img_path), "Fragment file does not exist"

        image = cv2.imread(img_path, 0)

        if image is None or image.shape[0] == 0:
            print("Image is empty or not loaded correctly:", img_path)
        assert 1 < np.asarray(image).max() <= 255, f"Invalid image"

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size) % CFG.tile_size
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size) % CFG.tile_size

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=0)

    label_path = os.path.join(CFG.fragment_root_dir, f"../inklabels/fragment{fragment_id}/inklabels.png")
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if label is None or label.shape[0] == 0:
        print("Label is empty or not loaded correctly:", label_path)

    assert pad0 is not None and pad1 is not None, "Padding is None or not set"

    label = np.pad(label, [(0, pad0), (0, pad1)], mode='constant', constant_values=0)
    label = (label / 255).astype(np.uint8)

    assert set(np.unique(np.array(label))) == {0, 1}, "Invalid label"

    return images, label


def process_fragment(data_root_dir, fragment_id, data_type):
    create_dataset(data_root_dir, fragment_id, data_type)


def create_k_fold_train_val_dataset(data_root_dir, train_frag_ids=None, val_frag_ids=None):
    print(f"Creating k_fold dataset with {len(train_frag_ids)} training and {len(val_frag_ids)} validation fragments")

    train_args = [(data_root_dir, frag_id, 'train') for frag_id in train_frag_ids]
    val_args = [(data_root_dir, frag_id, 'val') for frag_id in val_frag_ids]

    process_map(process_fragment, train_args, max_workers=multiprocessing.cpu_count())
    process_map(process_fragment, val_args, max_workers=multiprocessing.cpu_count())



def create_dataset(data_root_dir, fragment_id=2, data_type='train'):
    data_dir = os.path.join(data_root_dir, data_type)

    img_path = os.path.join(data_dir, "images")
    label_path = os.path.join(data_dir, "labels")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    images, label = read_fragment(fragment_id)

    x1_list = list(range(0, images.shape[2] - CFG.tile_size + 1, CFG.stride))
    y1_list = list(range(0, images.shape[1] - CFG.tile_size + 1, CFG.stride))

    progress_bar = tqdm(total=len(x1_list) * len(y1_list), desc=f"{data_type.capitalize()} Dataset: Processing images "
                                                                f"and labels")

    skip_counter_black_image = 0
    skip_counter_label = 0
    skip_counter_low_ink = 0
    requirement = (128 ** 2) * 0.1
    segformer_output_dim = (128, 128)

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
            label_patch = resize(label[y1:y2, x1:x2], segformer_output_dim, order=0, preserve_range=True,
                                 anti_aliasing=False)

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


def create_single_val_dataset(data_root_dir, train_split=0.8):
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


def build_k_fold_folder(cfg):
    train_ids_formatted = ['TF{}'.format(idx) for idx in cfg.train_frag_ids]
    val_ids_formatted = ['VF{}'.format(idx) for idx in cfg.val_frag_ids]
    train_ids_str = '_'.join(train_ids_formatted)
    val_ids_str = '_'.join(val_ids_formatted)

    return train_ids_str, val_ids_str


if __name__ == '__main__':
    """
    The label files are expected to be located within inklabels directory, so for example: 
    "inklabels/fragment1/inklabels.png"
    """
    parser = argparse.ArgumentParser(description="Run k-fold or single train-val dataset creation.")
    parser.add_argument('--k_fold', type=bool, default=None,
                        help='Enable k_fold dataset creation. Overrides CFG.k_fold if provided.')

    # Parse the arguments
    args = parser.parse_args()

    if args.k_fold is not None:
        CFG.k_fold = args.k_fold

    if CFG.k_fold:
        train_ids_str, val_ids_str = build_k_fold_folder(CFG)
        data_root_dir = os.path.join(CFG.data_root_dir, f'k_fold_{train_ids_str}_{val_ids_str}', str(CFG.size))
        create_k_fold_train_val_dataset(data_root_dir=data_root_dir,
                                        train_frag_ids=CFG.train_frag_ids,
                                        val_frag_ids=CFG.val_frag_ids)
    else:
        data_root_dir = os.path.join(CFG.data_root_dir, f'single_TF{CFG.single_train_frag_id}', str(CFG.size))
        create_dataset(data_root_dir=data_root_dir, fragment_id=CFG.single_train_frag_id)
        create_single_val_dataset(data_root_dir=data_root_dir, train_split=CFG.train_split)
