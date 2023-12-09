import argparse
import gc
import logging
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_handler import Config
from constants import get_frag_name_from_id, FRAGMENTS
from data.data_validation import validate_fragments, format_ranges
from data.utils import write_to_config
from data_modules.utils import balance_dataset

Image.MAX_IMAGE_PIXELS = None


def extract_patches(config: Config, label_dir):
    frag_id_2_channel = validate_fragments(config, label_dir)

    logging.info(f"Starting to extract image and label patches..")

    for fragment_id, channels in frag_id_2_channel.items():
        process_fragment(label_dir=label_dir, config=config, fragment_id=fragment_id, channels=channels)

    root_dir = os.path.join(config.dataset_target_dir, str(config.patch_size))
    df = pd.DataFrame(LABEL_INFO_LIST, columns=['filename', 'frag_id', 'channels', 'ink_p', 'artefact_p'])
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


def clean_all_fragment_label_dirs(config: Config):
    root_dir = os.path.join(config.dataset_target_dir, str(config.patch_size))
    for fragment_id in FRAGMENTS.values():
        frag_name = '_'.join([get_frag_name_from_id(fragment_id)]).upper()
        label_dir = os.path.join(root_dir, frag_name, 'labels')
        if os.path.isdir(label_dir):
            shutil.rmtree(label_dir)

    for file in ['label_infos.csv', 'config.json']:
        file_path = os.path.join(root_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def create_dataset(target_dir, config: Config, frag_id, channels, label_dir):
    target_dir = os.path.join(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    fragment_dir = os.path.join(config.data_root_dir, "fragments", f"fragment{frag_id}")
    if not os.path.isdir(fragment_dir):
        raise ValueError(f"Fragment directory does not exist: {fragment_dir}")

    label_dir = os.path.join(label_dir, frag_id)
    if not os.path.isdir(label_dir):
        raise ValueError(f"Label directory does not exist: {fragment_dir}")

    # Load mask
    mask_path = os.path.join(fragment_dir, f"mask.png")
    if not os.path.isfile(mask_path):
        raise ValueError(f"Mask file does not exist for fragment: {frag_id}")
    mask = np.asarray(Image.open(mask_path))

    total_skipped_cnt = 0
    total_patch_cnt = 0
    total_pruned_cnt = 0
    pbar = tqdm(total=(len(channels)), desc="Initializing...")

    first_channel_processed = False
    for start_channel in channels[::cfg.in_chans]:
        end_channel = start_channel + cfg.in_chans - 1

        pbar.set_description(f"\033[94mProcessing: Fragment: {get_frag_name_from_id(frag_id)}\033[0m "
                             f"Channel: {start_channel:02d}-{end_channel:02d} in "
                             f"{format_ranges(sorted(list(channels)), '')}")

        read_chans = range(start_channel, end_channel + 1)

        # Tensor may either be images or labels
        image_tensor = read_fragment_images_for_channels(root_dir=fragment_dir, patch_size=config.patch_size,
                                                         channels=read_chans, ch_block_size=config.in_chans)
        label_tensor = read_fragment_labels_for_channels(root_dir=label_dir, patch_size=config.patch_size,
                                                         channels=read_chans, ch_block_size=config.in_chans)

        # Only required for TQDM
        if not first_channel_processed:
            x1_list = list(range(0, image_tensor.shape[2] - config.patch_size + 1, config.stride))
            y1_list = list(range(0, image_tensor.shape[1] - config.patch_size + 1, config.stride))
            total_length = len(channels[::config.in_chans]) * len(x1_list) * len(y1_list)
            pbar.reset(total=total_length)
            first_channel_processed = True

        patch_cnt, skipped_cnt, pruned_cnt = process_channel_stack(config=config,
                                                                   target_dir=target_dir,
                                                                   frag_id=frag_id,
                                                                   mask=mask,
                                                                   image_tensor=image_tensor,
                                                                   label_tensor=label_tensor,
                                                                   start_channel=start_channel,
                                                                   pbar=pbar)

        total_patch_cnt += patch_cnt
        total_skipped_cnt += skipped_cnt
        total_pruned_cnt += pruned_cnt
        del image_tensor, label_tensor

        gc.collect()

    total_patches = total_patch_cnt + total_skipped_cnt + total_pruned_cnt
    pbar.set_description(f"\033[94mProcessing: Fragment: {get_frag_name_from_id(frag_id)}\033[0m "
                         f"\033[92mDone! Channels: {format_ranges(sorted(list(channels)), '')}\033[0m "
                         f"- Total={total_patches} |"
                         f" Patches={total_patch_cnt} |"
                         f" Skipped={total_skipped_cnt} |"
                         f" Pruned={total_pruned_cnt}")
    pbar.close()


def process_channel_stack(config: Config, target_dir, frag_id, mask, image_tensor, label_tensor, start_channel, pbar):
    """Extracts image/label patches for one 'label stack', corresponding to cfg.in_chans consecutive slice layers.

    :param label_tensor:
    :param image_tensor:
    :param pbar:
    :param start_channel:
    :param target_dir:
    :param config:
    :param frag_id:
    :param mask:
    :return:
    """
    x1_list = list(range(0, image_tensor.shape[2] - config.patch_size + 1, config.stride))
    y1_list = list(range(0, image_tensor.shape[1] - config.patch_size + 1, config.stride))

    patches = 0
    mask_skipped = 0

    STACK_PATCHES = {}
    STACK_PATCH_INFOS = []

    for y1 in y1_list:
        for x1 in x1_list:
            pbar.update(1)
            y2 = y1 + config.patch_size
            x2 = x1 + config.patch_size

            # Check mask for processing type 'images' and 'labels'
            if mask[y1:y2, x1:x2].all() != 1:  # Patch is not contained in mask
                mask_skipped += 1
                continue

            file_name = f"f{frag_id}_ch{start_channel:02d}_{x1}_{y1}_{x2}_{y2}.npy"

            if label_tensor.ndim != 3 or label_tensor.shape[0] != 2 or len(label_tensor[0]) + len(label_tensor[1]) == 0:
                raise ValueError(f"Expected tensor with shape (2, height, width), got {label_tensor.shape}")

            if image_tensor.ndim != 3 or image_tensor.shape[0] != cfg.in_chans or len(image_tensor[0]) + len(
                    image_tensor[1]) == 0:
                raise ValueError(
                    f"Expected tensor with shape ({cfg.in_chans}, height, width), got {image_tensor.shape}")

            ink_percentage = 0
            artefact_percentage = 0
            label_exists = False
            label_patch = None

            # tensor images: (4x512x512) labels (2x512x512)
            # If label is existent
            if label_tensor[0][0][0] != -1:
                label_exists = True
                base_label_patch = label_tensor[0, y1:y2, x1:x2]

                shape_product = np.prod(base_label_patch.shape)
                assert shape_product != 0

                ink_percentage = int((base_label_patch.sum() / shape_product) * 100)
                assert 0 <= ink_percentage <= 100

                label_patch = base_label_patch

            # If artefact label is existent
            if label_tensor[1][0][0] != -1:
                artefact_label_patch = label_tensor[1, y1:y2, x1:x2]

                artefact_percentage = int((artefact_label_patch.sum() / np.prod(artefact_label_patch.shape)) * 100)
                assert 0 <= artefact_percentage <= 100

                # If no label exists for this patch -> create zero-filled label patch
                if not label_exists:
                    artefact_label_patch = np.zeros_like(artefact_label_patch)

                    # Check that the label contains no 0 shape
                    shape_product = np.prod(artefact_label_patch.shape)
                    assert shape_product != 0

                    label_patch = artefact_label_patch
            assert label_patch is not None, "Label patch is None, Stack as processed without existing label or artefact label"

            image_patch = image_tensor[:, y1:y2, x1:x2]

            assert image_patch.shape == (
                cfg.in_chans, config.patch_size, config.patch_size), f"Image patch wrong shape: {image_patch.shape}"
            assert label_patch.shape == (
                config.patch_size, config.patch_size), f"Label patch wrong shape: {label_patch.shape}"

            STACK_PATCHES[file_name] = (image_patch, label_patch)
            STACK_PATCH_INFOS.append((file_name, frag_id, start_channel, ink_percentage, artefact_percentage))

    all_patches = len(STACK_PATCHES)
    assert all_patches > 0, "No patches were created for this fragment"
    df = pd.DataFrame(STACK_PATCH_INFOS, columns=['filename', 'frag_id', 'channels', 'ink_p', 'artefact_p'])
    balanced_df, _, _, _ = balance_dataset(cfg, df)
    LABEL_INFO_LIST.extend(balanced_df.values.tolist())

    for _, row in balanced_df.iterrows():
        file_name = row['filename']
        image_patch, label_patch = STACK_PATCHES[file_name]
        img_dir = os.path.join(target_dir, "images")
        label_dir = os.path.join(target_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        np.save(os.path.join(img_dir, file_name), image_patch)
        np.save(os.path.join(label_dir, file_name), label_patch)

    patches += len(balanced_df)
    pruned = all_patches - len(balanced_df)
    return patches, mask_skipped, pruned


def read_fragment_images_for_channels(root_dir, patch_size, channels, ch_block_size):
    images = []

    for channel in channels:
        img_path = os.path.join(root_dir, "slices", f"{channel:05}.tif")

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


def read_fragment_labels_for_channels(root_dir, patch_size, channels, ch_block_size):
    def read_label(label_path):
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
        label = np.expand_dims(label, 0)
        assert label.shape[0] == 1

        return label

    start_channel = channels[0]

    label_step_size = 4

    label_blocks = []
    for i in range(ch_block_size // label_step_size):
        base_label_path = os.path.join(root_dir, f'inklabels_{start_channel}_{start_channel + 3}.png')
        neg_label_path = os.path.join(root_dir, f'negatives_{start_channel}_{start_channel + 3}.png')
        start_channel += label_step_size

        base_label = read_label(base_label_path)
        neg_label = read_label(neg_label_path)

        if base_label is None and neg_label is None:
            print("Label and neg Label none")
            sys.exit(1)

        if base_label is None:
            base_label = np.ones_like(neg_label) * -1
        if neg_label is None:
            neg_label = np.ones_like(base_label) * -1

        label_stack = np.concatenate([base_label, neg_label], axis=0)
        assert label_stack.shape[0] == 2

        label_blocks.append(label_stack)

    result = np.zeros_like(label_blocks[0])
    output_shape = label_blocks[0][0]

    valid = [False, False]
    for block in label_blocks:
        for label_idx in range(2):
            # Check if the label part of the block for XOR does not contain -1
            if block[label_idx][0][0] != -1:
                result[label_idx] = result[label_idx] | block[label_idx]
                valid[label_idx] = True

    for valid_idx in range(2):
        if not valid[valid_idx]:
            result[valid_idx] = np.ones_like(output_shape) * -1

    assert result.shape == (2, output_shape.shape[0], output_shape.shape[1])

    return result


def get_sys_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some strings.')

    # Required string argument
    parser.add_argument('config_path', type=str)

    # Optional string argument
    parser.add_argument('--label_dir', type=str, default='handmade',
                        help='The label directory to be used for dataset creation, defaults to handmade')

    args = parser.parse_args()
    return args.config_path, args.label_dir


if __name__ == '__main__':
    config_path, label_dir = get_sys_args()
    cfg = Config.load_from_file(config_path)

    # TODO: add automatic binarize label layered for files that changed
    LABEL_INFO_LIST = []

    if label_dir == "handmade":
        label_dir = os.path.join(cfg.work_dir, "data", "base_label_files", "handmade")
    else:
        label_dir = os.path.join(cfg.work_dir, "data", "base_label_binarized", label_dir)

    clean_all_fragment_label_dirs(config=cfg)
    extract_patches(cfg, label_dir)
