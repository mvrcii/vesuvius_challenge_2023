import argparse
import logging
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utility.configs import Config
from utility.fragments import get_frag_name_from_id, JETFIRE_FRAG_ID
from multilayer_approach.data_validation_multilayer import validate_fragments
from skimage.transform import resize

Image.MAX_IMAGE_PIXELS = None


def extract_patches(config: Config, frags, label_dir):
    frag_id_2_channel = validate_fragments(config, frags, label_dir)

    logging.info(f"Starting to extract sanity sample")

    for fragment_id, channels in frag_id_2_channel.items():
        process_fragment(label_dir=label_dir, config=config, fragment_id=fragment_id, channels=channels)


def process_fragment(config: Config, fragment_id, channels, label_dir):
    frag_name = '_'.join([get_frag_name_from_id(fragment_id)]).upper()
    target_dir = os.path.join(config.dataset_target_dir, "sanity", frag_name)

    create_dataset(target_dir=target_dir,
                   label_dir=label_dir,
                   config=config,
                   frag_id=fragment_id,
                   channels=channels)


def clear_dataset(config: Config):
    print("clearing with patch size ", config.patch_size)
    root_dir = os.path.join(config.dataset_target_dir, str(config.patch_size))
    if os.path.isdir(root_dir):
        shutil.rmtree(root_dir)
        print("Deleted dataset directory:", root_dir)
    else:
        print("Dataset directory does not exist yet, nothing delete:", root_dir)


def create_dataset(target_dir, config: Config, frag_id, channels, label_dir):
    os.makedirs(target_dir, exist_ok=True)

    fragment_dir = os.path.join(config.data_root_dir, "../../fragments", f"fragment{frag_id}")
    if not os.path.isdir(fragment_dir):
        raise ValueError(f"Fragment directory does not exist: {fragment_dir}")

    label_dir = os.path.join(label_dir, frag_id)
    if not os.path.isdir(label_dir):
        raise ValueError(f"Label directory does not exist: {fragment_dir}")
    label_path = os.path.join(label_dir, f"{frag_id}_inklabels.png")
    ignore_path = os.path.join(label_dir, f"{frag_id}_ignore.png")

    # Load mask
    mask_path = os.path.join(fragment_dir, f"mask.png")
    if not os.path.isfile(mask_path):
        raise ValueError(f"Mask file does not exist for fragment: {frag_id}")
    # mask = np.asarray(Image.open(mask_path))

    mask = read_label(label_path=mask_path, patch_size=config.patch_size)

    start_channel = min(channels)
    end_channel = max(channels)

    read_chans = range(start_channel, end_channel + 1)

    image_tensor = read_fragment_images_for_channels(root_dir=fragment_dir, patch_size=config.patch_size,
                                                     channels=read_chans, ch_block_size=config.in_chans)
    label_arr = read_label(label_path=label_path, patch_size=config.patch_size)
    ignore_arr = read_label(label_path=ignore_path, patch_size=config.patch_size)

    # assert label arr is binary
    assert set(np.unique(label_arr)) == {0, 1}, "Invalid label, not binary: " + str(np.unique(label_arr))

    # assert ignore arr is binary
    assert set(np.unique(ignore_arr)) == {0, 1}, "Invalid ignore, not binary: " + str(np.unique(ignore_arr))

    # assert both have sum > 0
    assert label_arr.sum() > 0, "Label array is empty"
    assert ignore_arr.sum() > 0, "Ignore array is empty"

    print("Image Shape:", image_tensor.shape)
    print("Label Shape:", label_arr.shape)
    print("Mask Shape:", mask.shape)

    assert label_arr.shape == mask.shape == image_tensor[0].shape, (
        f"Shape mismatch for Fragment {frag_id}: Img={image_tensor[0].shape} "
        f"Mask={mask.shape} Label={label_arr.shape}")

    process_channel_stack(config=config,
                          target_dir=target_dir,
                          frag_id=frag_id,
                          mask=mask,
                          image_tensor=image_tensor,
                          label_arr=label_arr,
                          ignore_arr=ignore_arr,
                          start_channel=start_channel)


# Extracts image/label patches for one label (12 best layers)
def process_channel_stack(config: Config, target_dir, frag_id, mask, image_tensor, label_arr, ignore_arr,
                          start_channel):
    # image tensor is (12, height, width)
    x1_list = list(range(0, image_tensor.shape[2] - config.patch_size + 1, config.stride))
    y1_list = list(range(0, image_tensor.shape[1] - config.patch_size + 1, config.stride))


    label_shape = (config.label_size, config.label_size)

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

    x1 = 17134
    y1 = 2873

    y2 = y1 + config.patch_size
    x2 = x1 + config.patch_size

    mask_patch = mask[y1:y2, x1:x2]

    # Check if patch is fully in mask => discard
    if mask_patch.all() != 1:
        print("mask patch ignored")
        return

    if mask_patch.shape != (config.patch_size, config.patch_size):
        print("invalid shape")
        return

    # Get label patch, calculate ink percentage
    label_patch = label_arr[y1:y2, x1:x2]

    label_pixel_count = np.prod(label_patch.shape)
    assert label_pixel_count != 0
    ink_percentage = int((label_patch.sum() / label_pixel_count) * 100)
    print("ink percentage", ink_percentage)

    # Get image patch
    image_patch = image_tensor[:, y1:y2, x1:x2]
    ignore_patch = ignore_arr[y1:y2, x1:x2]

    # invert ignore_patch so "ignored" areas are 0 and rest is 1
    keep_patch = np.logical_not(ignore_patch)
    keep_percent = int((keep_patch.sum() / np.prod(keep_patch.shape)) * 100)
    assert 0 <= keep_percent <= 100

    # if ignore patch is fully black, discard
    if keep_percent < 5:
        print("fully ignored")
        return

    # Check shapes
    assert image_patch.shape == (
        cfg.in_chans, config.patch_size, config.patch_size), f"Image patch wrong shape: {image_patch.shape}"
    assert label_patch.shape == (
        config.patch_size, config.patch_size), f"Label patch wrong shape: {label_patch.shape}"

    file_name = f"f{frag_id}_ch{start_channel:02d}_{x1}_{y1}_{x2}_{y2}.npy"

    # save image
    np.save(os.path.join(img_dest_dir, file_name), image_patch)

    # scale label and ignore patch down to label size
    label_patch = resize(label_patch, label_shape, order=0, preserve_range=True, anti_aliasing=False)
    keep_patch = resize(keep_patch, label_shape, order=0, preserve_range=True, anti_aliasing=False)

    # stack label and keep patch
    label_patch = np.stack([label_patch, keep_patch], axis=0)

    # save label
    label_patch = np.packbits(label_patch.flatten())
    np.save(os.path.join(label_dest_dir, file_name), label_patch)


def read_fragment_images_for_channels(root_dir, patch_size, channels, ch_block_size):
    print("Reading channels to create sanity label")
    images = []

    for channel in tqdm(channels):
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
    assert images.ndim == 3 and images.shape[0] == ch_block_size

    return np.array(images)


def read_label(label_path, patch_size):
    if not os.path.isfile(label_path):
        return None

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    assert label is not None and label.shape[0] != 0 and label.shape[
        1] != 0, "Label is empty or not loaded correctly"

    pad0 = (patch_size - label.shape[0] % patch_size) % patch_size
    pad1 = (patch_size - label.shape[1] % patch_size) % patch_size
    assert pad0 is not None and pad1 is not None, "Padding is None or not set"

    label = np.pad(label, [(0, pad0), (0, pad1)], mode='constant', constant_values=0)
    label = (label / 255).astype(np.uint8)
    assert set(np.unique(np.array(label))) == {0, 1}, "Invalid label"

    # Expand dimensions for label stacking

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

    label_dir = os.path.join(cfg.work_dir, "multilayer_approach", "../base_labels", "3_binarized")
    print("Using label dir:", label_dir)

    fragments = [JETFIRE_FRAG_ID]
    extract_patches(cfg, fragments, label_dir)
