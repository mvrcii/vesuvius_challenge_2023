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

from fragment import FragmentHandler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_handler import Config
from constants import get_frag_name_from_id
from data.data_validation import validate_fragments, format_ranges
from data.utils import write_to_config
from skimage.transform import resize

Image.MAX_IMAGE_PIXELS = None


def extract_patches(config: Config, frags, label_dir):
    frag_id_2_channel = validate_fragments(config, frags, label_dir)

    logging.info(f"Starting to extract image and label patches")

    for fragment_id, channels in frag_id_2_channel.items():
        process_fragment(label_dir=label_dir, config=config, fragment_id=fragment_id, channels=channels)

    root_dir = os.path.join(config.dataset_target_dir, str(config.patch_size))
    df = pd.DataFrame(LABEL_INFO_LIST, columns=['filename', 'frag_id', 'channels', 'ink_p'])
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
    for fragment_id in FragmentHandler().get_ids():
        frag_name = '_'.join([get_frag_name_from_id(fragment_id)]).upper()
        frag_dir = os.path.join(root_dir, frag_name)
        if os.path.isdir(frag_dir):
            shutil.rmtree(frag_dir)
            print(f"Deleted dataset fragment directory: {frag_dir}")

    for file in ['label_infos.csv', 'config.json']:
        file_path = os.path.join(root_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def create_dataset(target_dir, config: Config, frag_id, channels, label_dir):
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

        channel_str = f"{start_channel:02d}" if cfg.in_chans == 1 else f"{start_channel:02d}-{end_channel:02d}"
        pbar.set_description(f"\033[94mProcessing: Fragment: {get_frag_name_from_id(frag_id)}\033[0m "
                             f"Channel: {channel_str} in "
                             f"{format_ranges(sorted(list(channels)), '')}")

        read_chans = range(start_channel, end_channel + 1)

        image_tensor = read_fragment_images_for_channels(root_dir=fragment_dir, patch_size=config.patch_size,
                                                         channels=read_chans, ch_block_size=config.in_chans)
        label_tensor = read_fragment_labels_for_channels(root_dir=label_dir, patch_size=config.patch_size,
                                                         channels=read_chans)

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


def extract_img_patch(_cfg: Config, tensor, x1, x2, y1, y2):
    if tensor.ndim != 3 or tensor.shape[0] != cfg.in_chans or tensor.shape[0] + tensor.shape[1] == 0:
        raise ValueError(f"Expected tensor with shape ({cfg.in_chans}, height, width), got {tensor.shape}")

    assert tensor, "Image tensor is None"

    img_patch = tensor[:, y1:y2, x1:x2]

    assert img_patch is not None, "Image patch is None"
    assert img_patch.shape == (_cfg.in_chans, _cfg.patch_size, _cfg.patch_size), (f"Image patch wrong shape: "
                                                                                  f"{img_patch.shape}")
    return img_patch


def extract_label_patch(_cfg: Config, tensor, x1, x2, y1, y2):
    if tensor.ndim != 3 or len(tensor[0]) + len(tensor[1]) == 0:
        raise ValueError(f"Expected tensor with shape (1, height, width), got {tensor.shape}")

    assert tensor, "Label tensor is None"

    label_patch = tensor[0, y1:y2, x1:x2]

    shape_product = np.prod(label_patch.shape)
    assert shape_product != 0

    ink_percentage = int((label_patch.sum() / shape_product) * 100)
    assert 0 <= ink_percentage <= 100

    assert label_patch is not None, "Label patch is None"
    assert label_patch.shape == (1, _cfg.patch_size, _cfg.patch_size), f"Label patch wrong shape: {label_patch.shape}"

    return label_patch


def process_channel_stack(config: Config, target_dir, frag_id, mask, image_tensor, label_tensor, start_channel, pbar):
    x1_list = list(range(0, image_tensor.shape[2] - config.patch_size + 1, config.stride))
    y1_list = list(range(0, image_tensor.shape[1] - config.patch_size + 1, config.stride))

    patches = 0
    patches_skipped_mask = 0

    patch_stack = {}
    patch_metainfo_stack = []

    for y1 in y1_list:
        for x1 in x1_list:
            pbar.update(1)
            y2 = y1 + config.patch_size
            x2 = x1 + config.patch_size

            # Check that patch is not contained in mask
            if mask[y1:y2, x1:x2].all() != 1:
                patches_skipped_mask += 1
                continue

            file_name = f"f{frag_id}_ch{start_channel:02d}_{x1}_{y1}_{x2}_{y2}.npy"

            label_patch, ink_p = extract_label_patch(_cfg=config, tensor=label_tensor, x1=x1, x2=x2, y1=y1, y2=y2)
            image_patch = extract_img_patch(_cfg=config, tensor=image_tensor, x1=x1, x2=x2, y1=y1, y2=y2)

            patch_stack[file_name] = (image_patch, label_patch)
            patch_metainfo_stack.append((file_name, frag_id, start_channel, ink_p))

    assert len(patch_stack) > 0, f"Warning: No patches were created for fragment {frag_id} and channel {start_channel}"

    patch_df = pd.DataFrame(patch_metainfo_stack, columns=['filename', 'frag_id', 'channels', 'ink_p'])

    # BALANCING / PRUNING
    patch_df = balance_dataset(config, patch_df)

    LABEL_INFO_LIST.extend(patch_df.values.tolist())

    img_dir = os.path.join(target_dir, "images")
    label_dir = os.path.join(target_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for _, row in patch_df.iterrows():
        file_name = row['filename']
        image_patch, label_patch = patch_stack[file_name]

        # save image
        np.save(os.path.join(img_dir, file_name), image_patch)

        # save label
        label_output_shape = (config.label_size, config.label_size)
        label_patch = resize(label_patch, label_output_shape, order=0, preserve_range=True, anti_aliasing=False)
        label_patch = np.packbits(label_patch.flatten())
        np.save(os.path.join(label_dir, file_name), label_patch)

    patches += len(patch_df)
    pruned = len(patch_stack) - len(patch_df)

    return patches, patches_skipped_mask, pruned


def balance_dataset(_cfg: Config, patch_df):
    data = patch_df[patch_df['frag_id'].isin(_cfg.fragment_ids)]

    # Skip balancing
    if _cfg.ink_ratio == -1:
        return data

    # BALANCING IS DONE ON CREATION
    # Step 1: Filter out rows where ink_p > ratio
    ink_samples = data[data['ink_p'] > _cfg.ink_ratio]

    # Step 2: Decide how many no-ink samples
    non_ink_sample_count = int(len(ink_samples) * cfg.no_ink_sample_percentage)

    # Step 3: Filter out rows where ink_p <= ink_ratio and limit the number of rows
    non_ink_samples = data[data['ink_p'] <= _cfg.ink_ratio].head(non_ink_sample_count)

    # Step 4: Concatenate the two DataFrames
    df = pd.concat([ink_samples, non_ink_samples])

    return df


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


def read_fragment_labels_for_channels(root_dir, patch_size, channels):
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

    file_pattern = f"_{start_channel}_{start_channel + cfg.in_chans - 1}"
    if cfg.in_chans == 1:
        file_pattern = f"_{start_channel}"

    base_label_path = os.path.join(root_dir, f'inklabels{file_pattern}.png')
    base_label = read_label(base_label_path)

    if base_label is None:
        raise Exception(f"Label data not found for layer {start_channel}")

    return base_label


def get_sys_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Create the dataset.')

    # Required string argument
    parser.add_argument('config_path', type=str)

    args = parser.parse_args()
    return args.config_path


if __name__ == '__main__':
    config_path = get_sys_args()
    cfg = Config.load_from_file(config_path)

    LABEL_INFO_LIST = []

    label_dir = os.path.join(cfg.work_dir, "data", "base_label_binarized_single",
                             "upbeat-tree-741-segformer-b2-231210-210131")
    fragments = cfg.fragment_ids

    clean_all_fragment_label_dirs(config=cfg)
    extract_patches(cfg, fragments, label_dir)
