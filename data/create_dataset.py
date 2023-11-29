import gc
import json
import logging
import os
import random
import re
import shutil
import sys
from multiprocessing import Manager, Pool

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_handler import Config
from constants import get_frag_name_from_id

Image.MAX_IMAGE_PIXELS = None


def write_dataset_cfg(path, **kwargs):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'config.json')

    with open(path, 'w') as file:
        json.dump(kwargs, file, indent=4)


def write_config(_cfg, frag_2_channels):
    """
    Wrapper for config writing. Returns the target directory for the dataset.
    """
    return write_k_fold_cfg(_cfg, frag_2_channels) if _cfg.k_fold else write_single_fold_cfg(_cfg, frag_2_channels)


def build_dataset_dir_from_config(config):
    train_frags = [get_frag_name_from_id(frag_id) for frag_id in config.train_frag_ids]
    result_dir_name = '_'.join(train_frags)
    k_fold_str = 'k_fold' if config.k_fold else 'single'

    return os.path.join(config.dataset_target_dir, k_fold_str, f"{str(config.patch_size)}px", result_dir_name)


def write_single_fold_cfg(_cfg, frag_2_channels):
    logging.info("Starting single-fold dataset creation process...")
    path = build_dataset_dir_from_config(config=_cfg)

    write_dataset_cfg(path,
                      single_train_frag_ids=_cfg.train_frag_ids,
                      frag_2_channels=frag_2_channels,
                      patch_size=_cfg.patch_size,
                      label_size=_cfg.label_size,
                      mode='single_fold')

    logging.info(f"Data root directory for single-fold dataset: {path}")
    logging.info(f"Creating single-fold dataset with fragment ids {_cfg.train_frag_ids}")

    return path


def write_k_fold_cfg(_cfg, frag_2_channels):
    logging.info("Starting k-fold dataset creation process...")
    path = build_dataset_dir_from_config(config=_cfg)

    write_dataset_cfg(path,
                      frag_2_channels=frag_2_channels,
                      train_frag_ids=_cfg.train_frag_ids,
                      val_frag_ids=_cfg.val_frag_ids,
                      patch_size=_cfg.patch_size,
                      label_size=_cfg.label_size,
                      k_fold=len(_cfg.train_frag_ids) + len(_cfg.val_frag_ids),
                      mode='k_fold')

    logging.info(f"Data root directory for k-fold: {path}")
    logging.info(
        f"Creating k-fold dataset with {len(_cfg.train_frag_ids)} training and {len(_cfg.val_frag_ids)} validation fragments")

    return path


def read_fragment(fragment_dir, channels, patch_size):
    images = []
    labels = []
    negative_labels = []
    pad0, pad1 = None, None

    for channel in tqdm(channels):
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

    for channel in channels[::4]:
        channel_used = False
        label_path = os.path.join(fragment_dir, 'layered', f"inklabels_{channel}_{channel + 3}.png")
        negative_label_path = os.path.join(fragment_dir, 'layered', f"negatives_{channel}_{channel + 3}.png")

        # LABELS
        label = None
        if os.path.isfile(label_path):
            channel_used = True
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            if label is None or label.shape[0] == 0:
                print("Label is empty or not loaded correctly:", label_path)
                sys.exit(1)

            assert pad0 is not None and pad1 is not None, "Padding is None or not set"
            label = np.pad(label, [(0, pad0), (0, pad1)], mode='constant', constant_values=0)
            label = (label / 255).astype(np.uint8)
            assert set(np.unique(np.array(label))) == {0, 1}, "Invalid label"
        labels.append(label)

        # NEGATIVE LABELS
        # This is optional, negative labels don't have to exist
        negative_label = None
        if os.path.isfile(negative_label_path):
            channel_used = True
            negative_label = cv2.imread(negative_label_path, cv2.IMREAD_GRAYSCALE)
            if negative_label is None or negative_label.shape[0] == 0:
                print("Negative Label is empty or not loaded correctly:", label_path)
                sys.exit(1)
        negative_labels.append(negative_label)

        if not channel_used:
            print(f"Channel {channel} listed but not used for any label. Check it!")
            sys.exit(1)

    # images = np.stack(images, axis=0)
    # labels = np.stack(labels, axis=0)
    # negative_labels = np.stack(negative_labels, axis=0)

    return images, labels, negative_labels


def process_fragment(dataset_information, fragment_id, data_type):
    return create_dataset(dataset_information=dataset_information, fragment_ids=fragment_id, data_type=data_type)


def validate_fragments(_cfg):
    all_errors = []
    frag_2_channels = {}

    for frag_id in _cfg.train_frag_ids:
        val_errors, frag_channels = validate_fragment_files(frag_id=frag_id, cfg=_cfg)
        frag_2_channels[frag_id] = frag_channels

        frag_str = f"Fragment: '{get_frag_name_from_id(frag_id)} ({frag_id})'"
        if val_errors:
            all_errors.extend(val_errors)
            print_checks([frag_str] + val_errors, [])
        else:
            print_checks([], [frag_str])

    if all_errors:
        sys.exit(1)

    return frag_2_channels


def print_checks(errors, valids):
    for valid in valids:
        print(f"\033[92mValid: {valid}\033[0m")
    for error in errors:
        print(f"\033[91mError: {error}\033[0m")


def validate_fragment_files(frag_id, cfg):
    errors = []
    frag_dir = os.path.join(cfg.work_dir, "data", "fragments", f"fragment{frag_id}")

    errors.extend(validate_fragment_dir(frag_dir))

    valid_errors, valid_channels = validate_labels(frag_dir)
    errors.extend(valid_errors)

    errors.extend(validate_masks(frag_dir))

    return errors, valid_channels


def validate_fragment_dir(frag_dir):
    if not os.path.isdir(frag_dir):
        return [f"Fragment directory '{frag_dir}' does not exist"]

    return []


def extract_label_indices(directory):
    ranges = []
    pattern = r'inklabels_(\d+)_(\d+).png'

    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            start_layer, end_layer = map(int, match.groups())
            ranges.append((start_layer, end_layer))

    indices = []
    for start, end in ranges:
        indices.extend(range(start, end + 1))
    indices.sort()

    return indices


def extract_indices(directory, pattern):
    indices = []

    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            groups = match.groups()
            if len(groups) == 1:
                # Single number (e.g., \d+.tif)
                number = int(groups[0])
                indices.append(number)
            elif len(groups) == 2:
                # Range of numbers (e.g., inklabels_(\d+)_(\d+).png)
                start_layer, end_layer = map(int, groups)
                indices.extend(range(start_layer, end_layer + 1))

    indices = list(set(indices))  # Remove duplicates if any
    indices.sort()  # Sort the indices in ascending order
    return indices


def is_valid_png(file_path):
    try:
        with Image.open(file_path) as img:
            return img.format == "PNG"
    except IOError:
        return False


def validate_labels(frag_dir):
    errors = []
    label_dir = os.path.join(frag_dir, 'layered')
    slice_dir = os.path.join(frag_dir, 'slices')

    if not os.path.isdir(label_dir):
        errors.append(f"Missing '{label_dir}' directory")
        return errors, []

    if not os.path.isdir(slice_dir):
        errors.append(f"Missing '{slice_dir}' directory")
        return errors, []

    existing_negative_channels = set(extract_indices(label_dir, pattern=r'negatives_(\d+)_(\d+).png'))
    existing_label_channels = set(extract_indices(label_dir, pattern=r'inklabels_(\d+)_(\d+).png'))
    existing_slice_channels = set(extract_indices(slice_dir, pattern=r'(\d+).tif'))
    required_channels = existing_label_channels.union(existing_negative_channels)

    valid_channels = existing_slice_channels.intersection(required_channels)

    missing_slice_channels = required_channels - existing_slice_channels
    if missing_slice_channels:
        errors.append(
            f"Missing slice channel files: {format_ranges(sorted(list(missing_slice_channels)), file_ending='png')}")

    return errors, sorted(list(valid_channels))


def calc_req_dataset_channels(slice_dir, label_dir):
    errors = []
    required_dirs = [slice_dir, label_dir]
    req_dirs_present = True

    for req_dir in required_dirs:
        if not os.path.isdir(req_dir):
            errors.append(f"Missing '{req_dir}' directory")
            req_dirs_present = False

    if req_dirs_present:
        existing_label_channels = set(extract_indices(label_dir, pattern=r'inklabels_(\d+)_(\d+).png'))
        existing_slice_channels = set(extract_indices(slice_dir, pattern=r'\d+.tif'))

        return errors, existing_slice_channels, existing_label_channels

    return errors, None, None


def validate_masks(frag_dir):
    file = 'mask.png'
    if not os.path.isfile(os.path.join(frag_dir, file)):
        return [f"Missing '{file}' file"]

    return []


def format_ranges(numbers, file_ending="tif"):
    """Convert a list of numbers into a string of ranges."""
    if not numbers:
        return ""

    ranges = []
    start = end = numbers[0]

    for n in numbers[1:]:
        if n - 1 == end:  # Part of the range
            end = n
        else:  # New range
            ranges.append((start, end))
            start = end = n
    ranges.append((start, end))

    return ', '.join(
        [f"{s:05d}.{file_ending} - {e:05d}.{file_ending}" if s != e else f"{s:05d}.tif" for s, e in ranges])


def build_dataset(_cfg):
    frag_2_channels = validate_fragments(_cfg=_cfg)
    build_k_fold_dataset(_cfg, frag_2_channels) if _cfg.k_fold else build_single_fold_dataset(_cfg, frag_2_channels)


def build_single_fold_dataset(_cfg, frag_2_channels):
    print("Dataset Type: Single-Fold")
    target_dir = write_config(_cfg=_cfg, frag_2_channels=frag_2_channels)

    dataset_information = {
        "target_dir": target_dir,
        "data_root_dir": _cfg.data_root_dir,
        "frag_2_channels": frag_2_channels,
        "patch_size": _cfg.patch_size,
        "stride": _cfg.stride,
        "ink_ratio": _cfg.ink_ratio,
        "min_negative_patches": _cfg.min_negative_patches_per_channel_block
    }

    train_coord_dict = process_fragment(dataset_information=dataset_information,
                                        fragment_id=_cfg.train_frag_ids,
                                        data_type='train')

    create_single_val_dataset(patch_size=_cfg.patch_size,
                              data_root_dir=target_dir,
                              train_split=_cfg.train_split,
                              train_coord_dict=train_coord_dict)


def build_k_fold_dataset(_cfg, frag_2_channels):
    target_dir = write_config(_cfg=_cfg, frag_2_channels=frag_2_channels)
    print("Dataset Type: K-Fold")

    with Manager() as manager:
        progress_tracker = manager.Value('i', 0)  # Shared integer for progress tracking

        dataset_information = {
            "target_dir": target_dir,
            "data_root_dir": _cfg.data_root_dir,
            "frag_2_channels": frag_2_channels,
            "patch_size": _cfg.patch_size,
            "stride": _cfg.stride,
            "ink_ratio": _cfg.ink_ratio,
            "min_negative_patches": _cfg.min_negative_patches_per_channel_block
        }

        all_args = [(dataset_information, frag_id, 'train', progress_tracker) for frag_id in _cfg.train_frag_ids]
        all_args += [(dataset_information, frag_id, 'val', progress_tracker) for frag_id in _cfg.val_frag_ids]

        with Pool() as pool:
            result = pool.starmap_async(process_fragment, all_args)
            results = result.get()

        errors = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                print(f"Error in processing fragment {all_args[i][1]}: {res}")
                errors.append(res)

        if errors:
            print(f"Errors occurred in {len(errors)} fragments.")
        else:
            print("All fragments processed successfully.")


def create_dataset(dataset_information, fragment_ids, data_type='train'):
    target_dir = dataset_information["target_dir"]
    data_root_dir = dataset_information["data_root_dir"]
    frag_2_channels = dataset_information["frag_2_channels"]
    patch_size = dataset_information["patch_size"]
    stride = dataset_information["stride"]
    ink_ratio = dataset_information["ink_ratio"]
    min_negative_patches = dataset_information["min_negative_patches"]

    target_data_dir = os.path.join(target_dir, data_type)

    img_path = os.path.join(target_data_dir, "images")
    label_path = os.path.join(target_data_dir, "labels")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    total_patch_count_white = 0
    total_patch_count_black = 0
    total_patch_count_negative = 0

    coord_dict = {}

    for frag_id in fragment_ids:
        coord_dict[frag_id] = {}
        fragment_dir = os.path.join(data_root_dir, "fragments", f"fragment{frag_id}")

        if not os.path.isdir(fragment_dir):
            raise ValueError(f"Fragment directory does not exist: {fragment_dir}")

        mask_path = os.path.join(fragment_dir, f"mask.png")
        if not os.path.isfile(mask_path):
            raise ValueError(f"Mask file does not exist for fragment: {mask_path}")
        mask_arr = np.asarray(Image.open(mask_path))

        channels = frag_2_channels[frag_id]
        assert len(channels) % 4 == 0, "Channels are not divisible by 4"

        images, labels, negative_labels = read_fragment(fragment_dir, frag_2_channels[frag_id], patch_size)
        images = np.array(images)
        labels = np.array(labels, dtype=object)
        negative_labels = np.array(negative_labels, dtype=object)

        x1_list = list(range(0, images.shape[2] - patch_size + 1, stride))
        y1_list = list(range(0, images.shape[1] - patch_size + 1, stride))

        pbar_channels = tqdm(total=(len(channels)),
                             desc=f"Fragment {get_frag_name_from_id(frag_id)} ({frag_id})': Processing channels")

        patch_count_skipped_mask = 0
        patch_count_white_total = 0
        patch_count_black_total = 0
        patch_count_negative_total = 0

        for label_idx, channel in enumerate(channels[::4]):
            channel_idx = channels.index(channel)
            coord_dict[frag_id][label_idx] = {}
            patch_count_white = 0
            patch_count_black = 0
            patch_count_negative = 0
            start_coord_list = [(x, y) for x in x1_list for y in y1_list]
            white_start_coords = []
            black_start_coords = []
            pbar_channels.update(4)

            for y1 in y1_list:
                for x1 in x1_list:
                    y2 = y1 + patch_size
                    x2 = x1 + patch_size

                    if mask_arr[y1:y2, x1:x2].all() != 1:  # Patch is not contained in mask
                        patch_count_skipped_mask += 1
                        start_coord_list.remove((x1, y1))
                        continue

                    img_patch = images[channel_idx:channel_idx + 4, y1:y2, x1:x2]
                    coord = (x1, y1)

                    # LABELS
                    if labels[label_idx] is not None:
                        label_patch = labels[label_idx][y1:y2, x1:x2]

                        # Add white patch if it contains sufficient ink
                        if label_patch.sum() > np.prod(label_patch.shape) * ink_ratio:
                            patch_count_white += 1
                            white_start_coords.append(coord)
                            start_coord_list.remove(coord)
                            file_name = f"f{frag_id}_l{label_idx}_{x1}_{y1}_{x2}_{y2}.npy"

                            img_file_path = os.path.join(img_path, file_name)
                            label_file_path = os.path.join(label_path, file_name)

                            np.save(img_file_path, img_patch)
                            np.save(label_file_path, label_patch)

                    # NEGATIVE LABELS
                    # Check if negative labels exist for this channel block
                    elif negative_labels[label_idx] is not None:
                        negative_label_patch = negative_labels[label_idx][y1:y2, x1:x2]
                        # If negative labels exist, check if this patch is overlapping a negative label
                        if negative_label_patch.sum() > 0:
                            # Here we are adding artefact patches
                            patch_count_negative += 1
                            black_start_coords.append(coord)
                            start_coord_list.remove(coord)

            # At this point start_coord_list only contains black patches that aren't outside the mask, are not
            # specifically labelled good negatives, and are below the specified ink threshold
            # black_start_coords at this point contains all the patches overlapping good_negative labels
            # negative_patch_count is the amount of negative patches that will now be sampled from start_coord_list,
            # it will be at least the amount min_negative_patches, or more if there are more white patches
            negative_patch_count = min(max(patch_count_white, min_negative_patches), len(start_coord_list))
            # Here we are adding regular dark patches
            black_start_coords.extend(random.sample(start_coord_list, negative_patch_count))
            for x1, y1 in black_start_coords:
                y2 = y1 + patch_size
                x2 = x1 + patch_size

                img_patch = images[channel_idx:channel_idx + 4, y1:y2, x1:x2]

                label_patch = np.zeros_like(img_patch[0])
                if labels[label_idx] is not None:
                    label_patch = labels[label_idx][y1:y2, x1:x2]

                patch_count_black += 1

                file_name = f"f{frag_id}_l{label_idx}_{x1}_{y1}_{x2}_{y2}.npy"

                img_file_path = os.path.join(img_path, file_name)
                label_file_path = os.path.join(label_path, file_name)

                np.save(img_file_path, img_patch)
                np.save(label_file_path, label_patch)

            coord_dict[frag_id][label_idx]['white_coord_list'] = white_start_coords
            coord_dict[frag_id][label_idx]['black_coord_list'] = black_start_coords

            patch_count_white_total += patch_count_white
            patch_count_black_total += (patch_count_black - patch_count_negative)
            patch_count_negative_total += patch_count_negative

        del images, labels, negative_labels
        gc.collect()

        pbar_channels.close()
        print(
            f"Balanced: Patch Count Fragment {get_frag_name_from_id(frag_id)}: "
            f"Ink={patch_count_white_total}, "
            f"Black={patch_count_black_total}, "
            f"Negative={patch_count_negative_total}")
        print(f"After Masking: Patch Count Fragment {get_frag_name_from_id(frag_id)}: Mask={patch_count_skipped_mask}")

        total_patch_count_white += patch_count_white_total
        total_patch_count_black += patch_count_black_total
        total_patch_count_negative += patch_count_negative_total

    print("\n================== SUMMARY TRAIN DATASET ==================")
    print("Total Patch Count White:\t", total_patch_count_white)
    print("Total Patch Count Black:\t", total_patch_count_black)
    print("Total Patch Count Negative:\t", total_patch_count_negative)
    print("Total Patch Count:\t\t\t", total_patch_count_white + total_patch_count_black + total_patch_count_negative)

    return coord_dict


def plot_patch_count_per_label_layer(data):
    plt.figure(figsize=(10, 5))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.xlabel('Label Layer')
    plt.ylabel('Patch Count')
    plt.title('Patch Count per Label Layer')
    plt.xticks(list(data.keys()))
    plt.show()


def move_files(src_dir, dest_dir, files):
    for file in files:
        shutil.move(os.path.join(src_dir, file), os.path.join(dest_dir, file))


def create_single_val_dataset(patch_size, data_root_dir, train_coord_dict, train_split=0.8):
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

    total_patch_count_white = 0
    total_patch_count_black = 0

    for frag_id, channels in train_coord_dict.items():
        for label_idx, coord_lists in channels.items():
            for list_name, coords in coord_lists.items():
                num_coords_to_select = int(len(coords) * (1 - train_split))
                sampled_coords = random.sample(coords, min(num_coords_to_select, len(coords)))

                files = []
                for coord in sampled_coords:
                    x1, y1 = coord
                    x2, y2 = x1 + patch_size, y1 + patch_size
                    file_name = f"f{frag_id}_l{label_idx}_{x1}_{y1}_{x2}_{y2}.npy"
                    files.append(file_name)

                for file in files:
                    shutil.move(os.path.join(train_img_dir, file), os.path.join(val_img_dir, file))
                    shutil.move(os.path.join(train_label_dir, file), os.path.join(val_label_dir, file))

                if list_name == 'black_coord_list':
                    total_patch_count_black += len(files)
                elif list_name == 'white_coord_list':
                    total_patch_count_white += len(files)

    total_count = total_patch_count_white + total_patch_count_black

    print("\n================== SUMMARY VAL DATASET ==================")
    print("Total Patch Count White:\t", total_patch_count_white)
    print("Total Patch Count Black:\t", total_patch_count_black)
    print("Total Patch Count:\t\t\t", total_count)


def plot_2d_arrays(label, image=None):
    """
    Plots one or two 2D numpy arrays as images. If two arrays are provided,
    they are plotted side by side.

    Parameters:
    array_1 (numpy.ndarray): The first 2D numpy array.
    array_2 (numpy.ndarray, optional): The second 2D numpy array. Default is None.

    Returns:
    None
    """
    figsize = 15, 15

    if image is not None:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        axs[0].imshow(label, cmap='gray', interpolation='none')
        axs[0].set_title("Label")
        axs[0].axis('off')

        axs[1].imshow(image[33], cmap='gray', interpolation='none')
        axs[1].set_title("Image")
        axs[1].axis('off')
    else:
        plt.figure(figsize=figsize)
        plt.imshow(label, cmap='gray', interpolation='none')
        plt.title("Label")
        plt.axis('off')

    plt.show()


def get_sys_args():
    if len(sys.argv) < 2:
        print("Usage: python ./data/create_dataset.py <config_path>")
        sys.exit(1)

    return sys.argv[1]


if __name__ == '__main__':
    # Process command arguments
    config_path = get_sys_args()

    # Load config
    config = Config.load_from_file(config_path)

    # Build dataset
    build_dataset(_cfg=config)
