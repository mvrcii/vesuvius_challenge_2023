import gc
import logging
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from config_handler import Config
from constants import get_frag_name_from_id
from data.data_validation import validate_fragments, format_ranges
from data.utils import write_config

Image.MAX_IMAGE_PIXELS = None


def extract_image_patches(config):
    # TODO: add automatic binarize label layered for files that changed
    frag_id_2_channel = validate_fragments(config)

    logging.info("Starting to extract image patches..")

    for fragment_id, channels in frag_id_2_channel.items():
        process_fragment(config=config, fragment_id=fragment_id, channels=channels)


def process_fragment(config, fragment_id, channels):
    target_dir = write_config(config=config, frag_id=fragment_id, channels=channels)
    create_dataset(target_dir=target_dir, config=config, frag_id=fragment_id, channels=channels)


def create_dataset(target_dir, config, frag_id, channels):
    img_path = os.path.join(target_dir, "images")
    os.makedirs(img_path, exist_ok=True)

    fragment_dir = os.path.join(config.data_root_dir, "fragments", f"fragment{frag_id}")
    if not os.path.isdir(fragment_dir):
        raise ValueError(f"Fragment directory does not exist: {fragment_dir}")

    # Load mask
    mask_path = os.path.join(fragment_dir, f"mask.png")
    if not os.path.isfile(mask_path):
        raise ValueError(f"Mask file does not exist for fragment: {frag_id}")
    mask = np.asarray(Image.open(mask_path))

    assert len(channels) % 4 == 0, "Channels are not divisible by 4"

    # Load slice images
    total_skipped_cnt = 0
    total_patch_cnt = 0

    pbar = tqdm(total=(len(channels)), desc="Initializing...")

    first_channel_processed = False
    for start_channel in channels[::4]:
        end_channel = start_channel + 3

        pbar.set_description(f"\033[94mProcessing: Fragment: {get_frag_name_from_id(frag_id)}\033[0m "
                             f"Channel: {start_channel:02d}-{end_channel:02d} from"
                             f"{format_ranges(sorted(list(channels)), '')}")

        images = read_fragment_images_for_channels(fragment_dir=fragment_dir,
                                                   channels=range(start_channel, end_channel + 1),
                                                   patch_size=config.patch_size)

        if not first_channel_processed:
            x1_list = list(range(0, images.shape[2] - config.patch_size + 1, config.stride))
            y1_list = list(range(0, images.shape[1] - config.patch_size + 1, config.stride))
            total_length = len(channels[::4]) * len(x1_list) * len(y1_list)
            pbar.reset(total=total_length)
            first_channel_processed = True

        patch_cnt, skipped_cnt = process_label_stack(config=config,
                                                     target_dir=img_path,
                                                     frag_id=frag_id,
                                                     mask=mask,
                                                     images=images,
                                                     start_channel=start_channel,
                                                     pbar=pbar)
        total_patch_cnt += patch_cnt
        total_skipped_cnt += skipped_cnt
        del images
        gc.collect()
    pbar.set_description(f"\033[94mProcessing: Fragment: {get_frag_name_from_id(frag_id)}\033[0m "
                         f"\033[92mDone! Channels: {format_ranges(sorted(list(channels)), '')}\033[0m")
    pbar.close()


def process_label_stack(config, target_dir, frag_id, mask, images, start_channel, pbar):
    """Extracts patches for one 'label stack', corresponding to 4 consecutive slice/image layers.

    :param start_channel:
    :param target_dir:
    :param config:
    :param frag_id:
    :param mask:
    :param images:
    :return:
    """
    x1_list = list(range(0, images.shape[2] - config.patch_size + 1, config.stride))
    y1_list = list(range(0, images.shape[1] - config.patch_size + 1, config.stride))

    patches = 0
    mask_skipped = 0
    for y1 in y1_list:
        for x1 in x1_list:
            pbar.update(1)
            y2 = y1 + config.patch_size
            x2 = x1 + config.patch_size

            if mask[y1:y2, x1:x2].all() != 1:  # Patch is not contained in mask
                mask_skipped += 1
                continue

            img_patch = images[:, y1:y2, x1:x2]
            file_name = f"f{frag_id}_ch{start_channel:02d}_{x1}_{y1}_{x2}_{y2}.npy"
            np.save(os.path.join(target_dir, file_name), img_patch)
            patches += 1
    return patches, mask_skipped


def read_fragment_images_for_channels(fragment_dir, patch_size, channels):
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

    return np.array(images)


def get_sys_args():
    if len(sys.argv) < 2:
        print("Usage: python ./data/extract_image_patches.py <config_path>")
        sys.exit(1)

    return sys.argv[1]


if __name__ == '__main__':
    # Process command arguments
    config_path = get_sys_args()

    # Load config
    cfg = Config.load_from_file(config_path)

    # Build dataset
    extract_image_patches(cfg)
