import argparse
import gc
import logging
import os
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

from multilayer_approach.data_validation import validate_fragments
from data.utils import write_to_config
from utility.configs import Config
from utility.fragments import get_frag_name_from_id

Image.MAX_IMAGE_PIXELS = None


def extract_patches(config: Config, frags, label_dir):
    frag_id_2_channel = validate_fragments(config, frags, label_dir)
    logging.info(f"Starting to extract image and label patches..")

    for fragment_id, channels in frag_id_2_channel.items():
        process_fragment(label_dir=label_dir, config=config, fragment_id=fragment_id, channels=channels)

    root_dir = os.path.join(config.dataset_target_dir, str(config.patch_size))
    df = pd.DataFrame(LABEL_INFO_LIST, columns=['filename', 'frag_id', 'channels', 'ink_p', 'ignore_p'])
    os.makedirs(root_dir, exist_ok=True)
    df.to_csv(os.path.join(root_dir, "label_infos.csv"))

    write_to_config(os.path.join(root_dir),
                    patch_size=config.patch_size,
                    label_size=config.label_size,
                    stride=config.stride,
                    in_chans=config.in_chans,
                    fragment_names=[get_frag_name_from_id(frag_id).upper() for frag_id in frag_id_2_channel.keys()],
                    frag_id_2_channel=frag_id_2_channel)


def process_fragment(config: Config, fragment_id, channels, label_dir):
    frag_name = '_'.join([get_frag_name_from_id(fragment_id)]).upper()
    target_dir = os.path.join(config.dataset_target_dir, str(config.patch_size), frag_name)

    write_to_config(target_dir, frag_id=fragment_id, channels=channels)

    create_dataset(target_dir=target_dir,
                   label_dir=label_dir,
                   config=config,
                   frag_id=fragment_id,
                   channels=channels)


def clear_dataset(config: Config):
    print("Clearing patch size", config.patch_size)
    root_dir = os.path.join(config.dataset_target_dir, str(config.patch_size))
    if os.path.isdir(root_dir):
        shutil.rmtree(root_dir)
        print("Deleted dataset directory:", root_dir)
    else:
        print("Dataset directory does not exist yet, nothing delete:", root_dir)


def create_dataset(target_dir, config: Config, frag_id, channels, label_dir):
    os.makedirs(target_dir, exist_ok=True)

    fragment = "fragments_contrasted" if getattr(config, 'contrasted', False) else "fragments"

    fragment_dir = os.path.join(config.data_root_dir, fragment, f"fragment{frag_id}")
    if not os.path.isdir(fragment_dir):
        raise ValueError(f"Fragment directory does not exist: {fragment_dir}")

    label_dir = os.path.join(label_dir, frag_id)
    if not os.path.isdir(label_dir):
        raise ValueError(f"Label directory does not exist: {fragment_dir}")
    label_path = os.path.join(label_dir, f"inklabels.png")
    ignore_path = os.path.join(label_dir, f"ignore.png")

    # Load mask
    mask_path = os.path.join(fragment_dir, f"mask.png")
    if not os.path.isfile(mask_path):
        raise ValueError(f"Mask file does not exist for fragment: {frag_id}")

    # print("reading mask")
    mask = read_label(label_path=mask_path, patch_size=config.patch_size)

    start_channel = min(channels)
    end_channel = max(channels)

    read_chans = range(start_channel, end_channel + 1)

    image_tensor = read_fragment_images_for_channels(root_dir=fragment_dir, patch_size=config.patch_size,
                                                     channels=read_chans, ch_block_size=config.in_chans)
    # print("reading label")
    label_arr = read_label(label_path=label_path, patch_size=config.patch_size)
    ignore_arr = read_label(label_path=ignore_path, patch_size=config.patch_size)

    # assert label arr is binary
    assert set(np.unique(label_arr)) == {0, 1}, "Invalid label, not binary: " + str(np.unique(label_arr))

    # assert ignore arr is binary
    assert set(np.unique(ignore_arr)) == {0, 1}, "Invalid ignore, not binary: " + str(np.unique(ignore_arr))

    # assert both have sum > 0
    assert label_arr.sum() > 0, "Label array is empty"
    assert ignore_arr.sum() > 0, "Ignore array is empty"

    assert ignore_arr.shape == label_arr.shape == mask.shape == image_tensor[0].shape, (
        f"Shape mismatch for Fragment {frag_id}: Img={image_tensor[0].shape} "
        f"Mask={mask.shape} Label={label_arr.shape} Ignore Arr={ignore_arr.shape}")

    patch_cnt, skipped_cnt, ignore_skipped_count = process_channel_stack(config=config,
                                                                         target_dir=target_dir,
                                                                         frag_id=frag_id,
                                                                         mask=mask,
                                                                         image_tensor=image_tensor,
                                                                         label_arr=label_arr,
                                                                         ignore_arr=ignore_arr,
                                                                         start_channel=start_channel)

    del image_tensor, label_arr, ignore_arr
    gc.collect()

    total_patches = patch_cnt + skipped_cnt + ignore_skipped_count
    print(f"- Total={total_patches} | Patches={patch_cnt} | Skipped={skipped_cnt} | Ignored = {ignore_skipped_count}")


# Extracts image/label patches for one label (12 best layers)
def process_channel_stack(config: Config, target_dir, frag_id, mask, image_tensor, label_arr, ignore_arr,
                          start_channel):
    # image tensor is (12, height, width)
    x1_list = list(range(0, image_tensor.shape[2] - config.patch_size + 1, config.stride))
    y1_list = list(range(0, image_tensor.shape[1] - config.patch_size + 1, config.stride))

    pbar = tqdm(total=len(x1_list) * len(y1_list), desc=f"Processing: Fragment: {get_frag_name_from_id(frag_id)}")

    mask_skipped = 0
    ignore_skipped = 0
    label_shape = (config.label_size, config.label_size)

    STACK_PATCH_INFOS = []

    if label_arr.ndim != 2:
        raise ValueError(f"Invalid label arr shape: {label_arr.shape}")

    if ignore_arr.ndim != 2:
        raise ValueError(f"Invalid ignore arr shape: {ignore_arr.shape}")

    if image_tensor.ndim != 3 or image_tensor.shape[0] != cfg.in_chans or len(image_tensor[0]) + len(
            image_tensor[1]) == 0:
        raise ValueError(
            f"Expected tensor with shape ({cfg.in_chans}, height, width), got {image_tensor.shape}")

    img_dest_dir = os.path.join(target_dir, "images")
    label_dest_dir = os.path.join(target_dir, "labels")
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(label_dest_dir, exist_ok=True)

    for y1 in y1_list:
        for x1 in x1_list:
            pbar.update(1)
            y2 = y1 + config.patch_size
            x2 = x1 + config.patch_size

            # Gte mask patch
            mask_patch = mask[y1:y2, x1:x2]

            # Check if patch is fully in mask => discard
            if mask_patch.all() != 1:
                mask_skipped += 1
                continue

            # Check if mask patch shape is valid
            if mask_patch.shape != (config.patch_size, config.patch_size):
                mask_skipped += 1
                continue

            # Get label, image and ignore patch
            label_patch = label_arr[y1:y2, x1:x2]
            ignore_patch = ignore_arr[y1:y2, x1:x2]
            image_patch = image_tensor[:, y1:y2, x1:x2]

            # Set label patch to 0 where ignore patch is 1 (to not count ink towards ink_p when it is ignored)
            label_patch[ignore_patch == 1] = 0

            # Create keep_patch by inverting ignore patch
            keep_patch = np.logical_not(ignore_patch)

            # Check shapes
            assert image_patch.shape == (
                cfg.in_chans, config.patch_size, config.patch_size), f"Image patch wrong shape: {image_patch.shape}"
            assert label_patch.shape == (
                config.patch_size, config.patch_size), f"Label patch wrong shape: {label_patch.shape}"

            # scale label and keep_patch patch down to label size
            label_patch = resize(label_patch, label_shape, order=0, preserve_range=True, anti_aliasing=False)
            keep_patch = resize(keep_patch, label_shape, order=0, preserve_range=True, anti_aliasing=False)

            # assert label patch has > 0 pixels
            label_pixel_count = np.prod(label_patch.shape)
            assert label_pixel_count != 0

            # calculate ink percentage of scaled down label patch
            ink_percentage = int((label_patch.sum() / label_pixel_count) * 100)
            assert 0 <= ink_percentage <= 100

            # calculate keep percentage of scaled down keep patch
            keep_percent = int((keep_patch.sum() / np.prod(keep_patch.shape)) * 100)
            assert 0 <= keep_percent <= 100
            ignore_percent = 100 - keep_percent

            # Discard images with less than 5% keep pixels
            if keep_percent < 5:
                ignore_skipped += 1
                continue

            # Create file name and save image
            file_name = f"f{frag_id}_ch{start_channel:02d}_{x1}_{y1}_{x2}_{y2}.npy"
            np.save(os.path.join(img_dest_dir, file_name), image_patch)

            # stack label and keep patch => save
            label_patch = np.stack([label_patch, keep_patch], axis=0)
            label_patch = np.packbits(label_patch.flatten())
            np.save(os.path.join(label_dest_dir, file_name), label_patch)

            STACK_PATCH_INFOS.append((file_name, frag_id, start_channel, ink_percentage, ignore_percent))

    LABEL_INFO_LIST.extend(STACK_PATCH_INFOS)
    return len(STACK_PATCH_INFOS), mask_skipped, ignore_skipped


def read_fragment_images_for_channels(root_dir, patch_size, channels, ch_block_size):
    images = []

    for channel in channels:
        img_path = os.path.join(root_dir, "layers", f"{channel:05}.tif")

        assert os.path.isfile(img_path), "Fragment file does not exist"

        image = cv2.imread(img_path, 0)

        if image is None or image.shape[0] == 0:
            print("Image is empty or not loaded correctly:", img_path)
        assert 1 < np.asarray(image).max() <= 255, f"Invalid image"

        pad0 = (patch_size - image.shape[0] % patch_size) % patch_size
        pad1 = (patch_size - image.shape[1] % patch_size) % patch_size
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)

    images = np.stack(images, axis=0)
    assert images.ndim == 3 and images.shape[0] == ch_block_size, f"Images shape {images.shape}, ch_block_size {ch_block_size}"

    return np.array(images)


def read_label(label_path, patch_size):
    if not os.path.isfile(label_path):
        return None

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # print("before padding ", label.shape)
    assert label is not None and label.shape[0] != 0 and label.shape[
        1] != 0, "Label is empty or not loaded correctly"

    pad0 = (patch_size - label.shape[0] % patch_size) % patch_size
    pad1 = (patch_size - label.shape[1] % patch_size) % patch_size
    assert pad0 is not None and pad1 is not None, "Padding is None or not set"

    label = np.pad(label, [(0, pad0), (0, pad1)], mode='constant', constant_values=0)
    label = (label / 255).astype(np.uint8)
    assert set(np.unique(np.array(label))) == {0, 1}, "Invalid label"

    # Expand dimensions for label stacking
    # print("after padding ", label.shape)

    return label


def get_sys_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some strings.')

    # Required string argument
    parser.add_argument('config_path', type=str)

    args = parser.parse_args()
    return args.config_path


if __name__ == '__main__':
    config_path = get_sys_args()
    cfg = Config.load_from_file(config_path)

    LABEL_INFO_LIST = []

    label_dir = os.path.join(cfg.work_dir, "data", "labels", "6_twelve_layer_unetr_it3", "binarized")
    print("Using label dir:", label_dir)

    train_fragments = cfg.fragment_ids
    val_fragments = cfg.validation_fragments
    fragments = list(set(train_fragments).union(val_fragments))
    clear_dataset(config=cfg)

    extract_patches(cfg, fragments, label_dir)
