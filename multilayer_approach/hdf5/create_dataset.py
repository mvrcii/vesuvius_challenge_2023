import argparse
import logging
import os
import shutil

import cv2
import h5py
import numpy as np
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

from data.utils import write_to_config
from multilayer_approach.data_validation import validate_fragments
from utility.configs import Config
from utility.fragments import get_frag_name_from_id

Image.MAX_IMAGE_PIXELS = None


def validate_directory(path, directory_type="directory"):
    """Raise an error if the directory does not exist."""
    if not os.path.isdir(path):
        raise ValueError(f"{directory_type.capitalize()} directory does not exist: {path}")


def validate_file(path, file_description="file"):
    """Raise an error if the file does not exist."""
    if not os.path.isfile(path):
        raise ValueError(f"{file_description.capitalize()} file does not exist: {path}")


def extract_patches(config, label_dir):
    # Get fragments
    frags = list(set(cfg.fragment_ids).union(cfg.validation_fragments))

    dataset_path = os.path.join(config.dataset_target_dir, str(config.patch_size), "dataset.hdf5")
    if os.path.isfile(dataset_path):
        with h5py.File(dataset_path, 'r') as dataset:
            for frag_id in frags:
                frag_name = get_frag_name_from_id(frag_id)

                if frag_name in dataset:
                    print(f"Fragment {frag_id} already exists")
                    frags.remove(frag_id)

    frag_id_2_layer = validate_fragments(config, frags, label_dir)
    print("\nSTAGE 2:\tEXTRACTING PATCHES")
    print("FRAGMENTS:")
    for frag_id in frag_id_2_layer.keys():
        print(frag_id, "\t", get_frag_name_from_id(frag_id))
    print("\n")

    for fragment_id, layers in frag_id_2_layer.items():
        process_fragment(label_dir=label_dir, config=config, frag_id=fragment_id, layers=layers)

    root_dir = os.path.join(config.dataset_target_dir, str(config.patch_size))

    write_to_config(path=root_dir,
                    patch_size=config.patch_size,
                    label_size=config.label_size,
                    stride=config.stride,
                    in_chans=config.in_chans,
                    fragment_names={frag_id: get_frag_name_from_id(frag_id).upper() for frag_id in
                                    frag_id_2_layer.keys()},
                    frag_id_2_layer=frag_id_2_layer)


def clear_dataset(config: Config):
    print("STAGE 1:\tCLEANING DATASET")
    root_dir = os.path.join(config.dataset_target_dir, str(config.patch_size))
    if os.path.isdir(root_dir):
        shutil.rmtree(root_dir)
        print("DELETED:\t", root_dir)
    print("\n")


def process_fragment(config, frag_id, layers, label_dir):
    target_dir = os.path.join(config.dataset_target_dir, str(config.patch_size))
    os.makedirs(target_dir, exist_ok=True)

    fragment = "fragments_contrasted" if getattr(config, 'contrasted', False) else "fragments"
    fragment_dir = os.path.join(config.data_root_dir, fragment, f"fragment{frag_id}")
    validate_directory(fragment_dir, "Fragment")

    label_dir = os.path.join(label_dir, frag_id)
    validate_directory(label_dir, "Label")

    paths = {name: os.path.join(label_dir, f"{name}.png") for name in ["inklabels", "ignore"]}
    paths["mask"] = os.path.join(fragment_dir, "mask.png")

    for path in paths.values():
        validate_file(path)

    tensors = {
        name: load_padded_binary_image(label_path=path, patch_size=config.patch_size)
        for name, path in paths.items()
    }

    tensors['image'] = read_fragment_images_for_layers(root_dir=fragment_dir, patch_size=config.patch_size,
                                                       layers=range(min(layers), max(layers) + 1))

    # assert label and ignore arr are binary
    # assert set(np.unique(label_arr)) == {0, 1}, "Invalid label, not binary: " + str(np.unique(label_arr))
    # assert set(np.unique(ignore_arr)) == {0, 1}, "Invalid ignore, not binary: " + str(np.unique(ignore_arr))

    # assert both have sum > 0
    # assert label_arr.sum() > 0, "Label array is empty"
    # assert ignore_arr.sum() > 0, "Ignore array is empty"

    # assert ignore_arr.shape == label_arr.shape == mask.shape == image_tensor[0].shape, (
    #     f"Shape mismatch for Fragment {frag_id}: Img={image_tensor[0].shape} "
    #     f"Mask={mask.shape} Label={label_arr.shape} Ignore Arr={ignore_arr.shape}")

    patch_cnt, skipped_cnt, ignore_skipped_count = process_layers(tensors=tensors,
                                                                  config=config,
                                                                  target_dir=target_dir,
                                                                  frag_id=frag_id,
                                                                  start_layer=min(layers))

    total_patches = patch_cnt + skipped_cnt + ignore_skipped_count
    print(f"Total={total_patches} | Patches={patch_cnt} | Skipped={skipped_cnt} | Ignored = {ignore_skipped_count}")


def process_layers(tensors, config: Config, target_dir, frag_id, start_layer):
    """ Extracts image/label/keep patches for one label (12 best layers)
    :param tensors: Dictionary with image, label, ignore and mask tensors
    :param config: Config object
    :param target_dir: Target directory
    :param frag_id: Fragment ID
    :param start_layer: Start layer
    """
    image_tensor = tensors['image']
    label_arr = tensors['inklabels']
    ignore_arr = tensors['ignore']
    mask = tensors['mask']

    x1_list = list(range(0, image_tensor.shape[2] - config.patch_size + 1, config.stride))
    y1_list = list(range(0, image_tensor.shape[1] - config.patch_size + 1, config.stride))

    pbar = tqdm(total=len(x1_list) * len(y1_list), desc=f"Processing: Fragment: {get_frag_name_from_id(frag_id)}")

    mask_skipped = 0
    ignore_skipped = 0
    label_shape = (config.label_size, config.label_size)

    if label_arr.ndim != 2:
        raise ValueError(f"Invalid label arr shape: {label_arr.shape}")

    if ignore_arr.ndim != 2:
        raise ValueError(f"Invalid ignore arr shape: {ignore_arr.shape}")

    if image_tensor.ndim != 3 or image_tensor.shape[0] != cfg.in_chans or len(image_tensor[0]) + len(
            image_tensor[1]) == 0:
        raise ValueError(
            f"Expected tensor with shape ({cfg.in_chans}, height, width), got {image_tensor.shape}")

    hdf_filepath = os.path.join(target_dir, "dataset.hdf5")
    with h5py.File(hdf_filepath, 'a') as hdf_file:
        frag_name = get_frag_name_from_id(frag_id).upper()
        frag_group = hdf_file.create_group(frag_name) if frag_name not in hdf_file else hdf_file[frag_name]

        frag_meta_data = {
            "frag_id": frag_id,
            "start_layer": start_layer,
            "patch_size": config.patch_size,
            "stride": config.stride,
            "label_size": config.label_size,
            "in_chans": config.in_chans,
            "frag_name": frag_name,
            "label_dims": label_arr.shape
        }

        for key, value in frag_meta_data.items():
            frag_group.attrs[key] = value

        patch_cnt = 0

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
                # assert image_patch.shape == (
                #     cfg.in_chans, config.patch_size, config.patch_size), f"Image patch wrong shape: {image_patch.shape}"
                # assert label_patch.shape == (
                #     config.patch_size, config.patch_size), f"Label patch wrong shape: {label_patch.shape}"

                # scale label and keep_patch patch down to label size
                label_patch = resize(label_patch, label_shape, order=0, preserve_range=True, anti_aliasing=False)
                keep_patch = resize(keep_patch, label_shape, order=0, preserve_range=True, anti_aliasing=False)

                # assert label patch has > 0 pixels
                label_pixel_count = np.prod(label_patch.shape)
                # assert label_pixel_count != 0

                # Calculate ink percentage of scaled down label patch
                ink_percentage = int((label_patch.sum() / label_pixel_count) * 100)
                # assert 0 <= ink_percentage <= 100

                # Discard images with less than 5% ink pixels
                if ink_percentage < cfg.ink_ratio:
                    ignore_skipped += 1
                    continue

                # TODO: Temporarily remember patches with ink_percentage == 0 and randomly save as many as we have patches
                # TODO: with ink_percentage > ink_ratio

                # Calculate keep percentage of scaled down keep patch
                keep_percent = int((keep_patch.sum() / np.prod(keep_patch.shape)) * 100)
                # assert 0 <= keep_percent <= 100
                ignore_percent = 100 - keep_percent

                # Discard images with less than 5% keep pixels
                if keep_percent < 5:
                    ignore_skipped += 1
                    continue

                if f"img_patch_{patch_cnt}" not in frag_group:
                    patch_group = frag_group.create_group(f"img_patch_{patch_cnt}")
                    patch_cnt += 1
                else:
                    patch_group = frag_group[f"img_patch_{patch_cnt}"]

                # Save patches to fragment patch group within the hdf5 file
                patch_group.create_dataset("image_patch", data=image_patch)
                patch_group.create_dataset("label_patch", data=label_patch)
                patch_group.create_dataset("keep_patch", data=keep_patch)

                # Additionally add metadata to the patch group
                patch_meta_data = {
                    "frag_id": frag_id,
                    "start_layer": start_layer,
                    "ink_percentage": ink_percentage,
                    "ignore_percent": ignore_percent,
                    "bbox": (x1, y1, x2, y2)
                }

                for key, value in patch_meta_data.items():
                    patch_group.attrs[key] = value

    return patch_cnt, mask_skipped, ignore_skipped


def read_fragment_images_for_layers(root_dir, patch_size, layers):
    images = []

    for layer in layers:
        img_path = os.path.join(root_dir, "layers", f"{layer:05}.tif")

        # assert os.path.isfile(img_path), "Fragment file does not exist"

        image = cv2.imread(img_path, 0)

        if image is None or image.shape[0] == 0:
            print("Image is empty or not loaded correctly:", img_path)
        # assert 1 < np.asarray(image).max() <= 255, f"Invalid image"

        pad0 = (patch_size - image.shape[0] % patch_size) % patch_size
        pad1 = (patch_size - image.shape[1] % patch_size) % patch_size
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)

    images = np.stack(images, axis=0)
    # assert images.ndim == 3 and images.shape[
    #     0] == ch_block_size, f"Images shape {images.shape}, ch_block_size {ch_block_size}"

    return np.array(images)


def load_padded_binary_image(label_path, patch_size):
    if not os.path.isfile(label_path):
        return None

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # assert label is not None and label.shape[0] != 0 and label.shape[1] != 0, "Label is empty or not loaded correctly"

    pad0 = (patch_size - label.shape[0] % patch_size) % patch_size
    pad1 = (patch_size - label.shape[1] % patch_size) % patch_size

    # assert pad0 is not None and pad1 is not None, "Padding is None or not set"

    label = np.pad(label, [(0, pad0), (0, pad1)], mode='constant', constant_values=0)
    label = (label / 255).astype(np.uint8)

    # assert set(np.unique(np.array(label))) == {0, 1}, "Invalid label"

    return label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset for segmentation based on a given config')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    cfg = Config.load_from_file(args.config_path)

    label_dir = os.path.join(cfg.work_dir, "data", "labels", "6_twelve_layer_unetr_it3", "binarized")

    print("STAGE 0:")
    print("LABEL_DIR:\t", label_dir)
    print("PATCH_SIZE:\t", cfg.patch_size, "\n")

    # clear_dataset(config=cfg)

    extract_patches(cfg, label_dir)
