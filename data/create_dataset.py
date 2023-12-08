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

Image.MAX_IMAGE_PIXELS = None


def extract_patches(config, processing_type):
    frag_id_2_channel = validate_fragments(config, processing_type=processing_type)

    logging.info(f"Starting to extract {processing_type} patches..")

    for fragment_id, channels in frag_id_2_channel.items():
        process_fragment(config=config, fragment_id=fragment_id, channels=channels, processing_type=processing_type)

    if processing_type == 'labels':
        root_dir = os.path.join(config.dataset_target_dir, str(config.patch_size))
        df = pd.DataFrame(LABEL_INFO_LIST, columns=['filename', 'frag_id', 'channels', 'ink_p', 'artefact_p'])
        df.to_csv(os.path.join(root_dir, "label_infos.csv"))

        write_to_config(os.path.join(root_dir),
                        patch_size=config.patch_size,
                        label_size=config.label_size,
                        stride=config.stride,
                        in_chans=config.in_chans,
                        fragment_names=[get_frag_name_from_id(frag_id).upper() for frag_id in frag_id_2_channel.keys()],
                        frag_id_2_channel=frag_id_2_channel)


def process_fragment(config, fragment_id, channels, processing_type):
    frag_name = '_'.join([get_frag_name_from_id(fragment_id)]).upper()
    target_dir = os.path.join(config.dataset_target_dir, str(config.patch_size), frag_name)

    if processing_type == 'images':
        write_to_config(target_dir, frag_id=fragment_id, channels=channels)

    create_dataset(target_dir=target_dir,
                   config=config,
                   frag_id=fragment_id,
                   channels=channels,
                   processing_type=processing_type)

    # TODO: PRUNE LABEL_INFO_LIST for blacks


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


def create_dataset(target_dir, config, frag_id, channels, processing_type):
    target_dir = os.path.join(target_dir, processing_type)

    os.makedirs(target_dir, exist_ok=True)

    fragment_dir = os.path.join(config.data_root_dir, "fragments", f"fragment{frag_id}")
    if not os.path.isdir(fragment_dir):
        raise ValueError(f"Fragment directory does not exist: {fragment_dir}")

    # Load mask
    mask_path = os.path.join(fragment_dir, f"mask.png")
    if not os.path.isfile(mask_path):
        raise ValueError(f"Mask file does not exist for fragment: {frag_id}")
    mask = np.asarray(Image.open(mask_path))

    total_skipped_cnt = 0
    total_patch_cnt = 0
    pbar = tqdm(total=(len(channels)), desc="Initializing...")

    first_channel_processed = False
    for start_channel in channels[::cfg.in_chans]:
        end_channel = start_channel + cfg.in_chans - 1

        pbar.set_description(f"\033[94mProcessing: Fragment: {get_frag_name_from_id(frag_id)}\033[0m "
                             f"Channel: {start_channel:02d}-{end_channel:02d} in "
                             f"{format_ranges(sorted(list(channels)), '')}")

        # Tensor may either be images or labels
        tensor = read_fragment_for_channels(frag_dir=fragment_dir,
                                            processing_type=processing_type,
                                            channels=range(start_channel, end_channel + 1),
                                            patch_size=config.patch_size,
                                            ch_block_size=config.in_chans)

        # Only required for TQDM
        if not first_channel_processed:
            x1_list = list(range(0, tensor.shape[2] - config.patch_size + 1, config.stride))
            y1_list = list(range(0, tensor.shape[1] - config.patch_size + 1, config.stride))
            total_length = len(channels[::config.in_chans]) * len(x1_list) * len(y1_list)
            pbar.reset(total=total_length)
            first_channel_processed = True

        patch_cnt, skipped_cnt = process_channel_stack(config=config,
                                                       target_dir=target_dir,
                                                       frag_id=frag_id,
                                                       mask=mask,
                                                       tensor=tensor,
                                                       start_channel=start_channel,
                                                       processing_type=processing_type,
                                                       pbar=pbar)

        total_patch_cnt += patch_cnt
        total_skipped_cnt += skipped_cnt
        del tensor
        gc.collect()
    pbar.set_description(f"\033[94mProcessing: Fragment: {get_frag_name_from_id(frag_id)}\033[0m "
                         f"\033[92mDone! Channels: {format_ranges(sorted(list(channels)), '')}\033[0m")
    pbar.close()


def process_channel_stack(config, target_dir, frag_id, mask, tensor, start_channel, processing_type, pbar):
    """Extracts image/label patches for one 'label stack', corresponding to cfg.in_chans consecutive slice layers.

    :param pbar:
    :param processing_type:
    :param start_channel:
    :param target_dir:
    :param config:
    :param frag_id:
    :param mask:
    :param tensor:
    :return:
    """
    x1_list = list(range(0, tensor.shape[2] - config.patch_size + 1, config.stride))
    y1_list = list(range(0, tensor.shape[1] - config.patch_size + 1, config.stride))

    patches = 0
    mask_skipped = 0
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
            file_path = os.path.join(target_dir, file_name)

            if processing_type == 'labels':
                if tensor.ndim < 3 or tensor.shape[0] < 2 or len(tensor[0]) + len(tensor[1]) == 0:
                    raise ValueError(f"Expected tensor with shape (2, height, width), got {tensor.shape}")

                ink_percentage = 0
                artefact_percentage = 0
                label_exists = False

                # If label is existent
                if tensor[0][0][0] != -1:
                    label_exists = True
                    base_label_patch = tensor[0, y1:y2, x1:x2]

                    shape_product = np.prod(base_label_patch.shape)
                    assert shape_product != 0

                    ink_percentage = int((base_label_patch.sum() / shape_product) * 100)
                    assert 0 <= ink_percentage <= 100

                    np.save(file_path, base_label_patch)
                    patches += 1

                # If artefact label is existent
                if tensor[1][0][0] != -1:
                    artefact_label_patch = tensor[1, y1:y2, x1:x2]

                    artefact_percentage = int((artefact_label_patch.sum() / np.prod(artefact_label_patch.shape)) * 100)
                    assert 0 <= artefact_percentage <= 100

                    # If no label exists for this patch -> create zero-filled label patch
                    if not label_exists:
                        artefact_label_patch = np.zeros_like(artefact_label_patch)

                        # Check that the label contains no 0 shape
                        shape_product = np.prod(artefact_label_patch.shape)
                        assert shape_product != 0

                        np.save(file_path, artefact_label_patch)
                        patches += 1

                LABEL_INFO_LIST.append((file_name, frag_id, start_channel, ink_percentage, artefact_percentage))

            elif processing_type == 'images':
                if os.path.isfile(file_path):
                    continue
                image_patch = tensor[:, y1:y2, x1:x2]
                np.save(file_path, image_patch)
                patches += 1
            else:
                raise ValueError("Unknown processing type:", processing_type)

    return patches, mask_skipped


def read_fragment_for_channels(frag_dir, channels, patch_size, processing_type, ch_block_size):
    if processing_type == 'images':
        image_patch = read_fragment_images_for_channels(frag_dir, patch_size, channels, ch_block_size=ch_block_size)
        return image_patch
    elif processing_type == 'labels':
        label_patch = read_fragment_labels_for_channels(frag_dir, patch_size, channels, ch_block_size=ch_block_size)
        return label_patch
    else:
        raise ValueError("Processing Type unknown:", processing_type)


def read_fragment_images_for_channels(fragment_dir, patch_size, channels, ch_block_size):
    images = []

    for channel in channels:
        img_path = os.path.join(fragment_dir, "slices", f"{channel:05}.tif")

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


def read_fragment_labels_for_channels(fragment_dir, patch_size, channels, ch_block_size):
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

    root_dir = os.path.join(fragment_dir, 'layered')
    label_step_size = 4

    label_blocks = []
    for i in range(ch_block_size // label_step_size):
        base_label_path = os.path.join(root_dir, f'inklabels_{start_channel}_{start_channel + 3}.png')
        neg_label_path = os.path.join(root_dir, f'negatives_{start_channel}_{start_channel + 3}.png')
        start_channel += label_step_size

        base_label = read_label(base_label_path)
        neg_label = read_label(neg_label_path)

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
    if len(sys.argv) < 2:
        print("Usage: python ./data/create_dataset.py <config_path>")
        sys.exit(1)

    return sys.argv[1]


if __name__ == '__main__':
    config_path = get_sys_args()
    cfg = Config.load_from_file(config_path)

    # TODO: add automatic binarize label layered for files that changed
    LABEL_INFO_LIST = []

    clean_all_fragment_label_dirs(config=cfg)
    for proc_type in ['images', 'labels']:
        print("Processing", proc_type)
        extract_patches(cfg, processing_type=proc_type)
