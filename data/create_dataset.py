import gc
import json
import logging
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from multiprocessing import Manager, Pool

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm

from config_handler import Config
from constants import get_frag_name_from_id

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def write_single_fold_cfg(_cfg, frag_2_channels):
    logging.info("Starting single-fold dataset creation process...")
    result_dir_name = f'single_fold_{str(_cfg.patch_size)}px'
    path = os.path.join(_cfg.dataset_target_dir, result_dir_name)

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

    result_dir_name = f'k_fold_{str(_cfg.patch_size)}px'
    path = os.path.join(_cfg.dataset_target_dir, result_dir_name)

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

    for channel in range(min(channels), max(channels), 4):
        label_path = os.path.join(fragment_dir, 'layered', f"inklabels_{channel}_{channel + 3}.png")

        if not os.path.isfile(label_path):
            print("Label file not found:", label_path)
            sys.exit(1)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if label is None or label.shape[0] == 0:
            print("Label is empty or not loaded correctly:", label_path)

        assert pad0 is not None and pad1 is not None, "Padding is None or not set"

        label = np.pad(label, [(0, pad0), (0, pad1)], mode='constant', constant_values=0)
        label = (label / 255).astype(np.uint8)
        assert set(np.unique(np.array(label))) == {0, 1}, "Invalid label"

        labels.append(label)

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)

    return images, labels


def process_fragment(dataset_information, fragment_id, data_type, progress_tracker):
    try:
        create_dataset(dataset_information=dataset_information, fragment_ids=fragment_id, data_type=data_type)
        progress_tracker.value += 1

        return True
    except Exception as e:
        return e


def calc_original_channel_ids(channels):
    mid = 65 // 2
    start = mid - channels // 2
    end = mid + channels // 2
    return list(range(start, end))


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
    print("\n")


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

    existing_label_channels = set(extract_indices(label_dir, pattern=r'inklabels_(\d+)_(\d+).png'))
    existing_slice_channels = set(extract_indices(slice_dir, pattern=r'(\d+).tif'))

    missing_slice_channels = existing_label_channels - existing_slice_channels
    valid_channels = existing_slice_channels.intersection(existing_label_channels)

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
        "calc_mean_std": _cfg.calc_mean_std
    }

    with Manager() as manager:
        progress_tracker = manager.Value('i', 0)

        all_args = [(dataset_information, _cfg.train_frag_ids, 'train', progress_tracker)]

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

        create_single_val_dataset(data_root_dir=target_dir, train_split=_cfg.train_split)

    if _cfg.calc_mean_std:
        plot_mean_std(target_dir)


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
            "calc_mean_std": _cfg.calc_mean_std
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

    if _cfg.calc_mean_std:
        plot_mean_std(target_dir)


def create_dataset(dataset_information, fragment_ids, data_type='train'):
    target_dir = dataset_information["target_dir"]
    data_root_dir = dataset_information["data_root_dir"]
    frag_2_channels = dataset_information["frag_2_channels"]
    patch_size = dataset_information["patch_size"]
    stride = dataset_information["stride"]
    ink_ratio = dataset_information["ink_ratio"]
    calc_mean_std = dataset_information["calc_mean_std"]

    target_data_dir = os.path.join(target_dir, data_type)

    img_path = os.path.join(target_data_dir, "images")
    label_path = os.path.join(target_data_dir, "labels")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    total_patch_count = 0
    label_size = patch_size // 4

    for frag_id in fragment_ids:
        fragment_dir = os.path.join(data_root_dir, "fragments", f"fragment{frag_id}")

        if not os.path.isdir(fragment_dir):
            raise ValueError(f"Fragment directory does not exist: {fragment_dir}")

        mask_path = os.path.join(fragment_dir, f"mask.png")
        if not os.path.isfile(mask_path):
            raise ValueError(f"Mask file does not exist for fragment: {mask_path}")
        mask_arr = np.asarray(Image.open(mask_path))

        channels = frag_2_channels[frag_id]
        assert len(channels) % 4 == 0, "Channels are not divisible by 4"

        images, label = read_fragment(fragment_dir, frag_2_channels[frag_id], patch_size)

        x1_list = list(range(0, images.shape[2] - patch_size + 1, stride))
        y1_list = list(range(0, images.shape[1] - patch_size + 1, stride))

        progress_bar = tqdm(total=len(x1_list) * len(y1_list) * (len(channels) // 4),
                            desc=f"{data_type.capitalize()} Dataset Fragment "
                                 f"'{get_frag_name_from_id(frag_id)} ({frag_id})': Processing images "
                                 f"and labels")

        patch_count_for_fragment = 0

        patch_count_for_label_layer = defaultdict(int)
        try:
            for y1 in y1_list:
                for x1 in x1_list:
                    y2 = y1 + patch_size
                    x2 = x1 + patch_size

                    for channel in range(0, max(channels) - min(channels) + 1, 4):
                        progress_bar.update(1)

                        img_patch = images[channel:channel + 4, y1:y2, x1:x2]

                        if mask_arr[y1:y2, x1:x2].all() != 1:  # Patch is not contained in mask
                            continue

                        # Scale label down to match segformer output
                        label_idx = channel // 4
                        label_patch = label[label_idx, y1:y2, x1:x2]
                        label_patch = resize(label_patch,
                                             (label_size, label_size),
                                             order=0, preserve_range=True, anti_aliasing=False)

                        # Check that the label contains at least N % ink
                        if label_patch.sum() < np.prod(label_patch.shape) * ink_ratio:
                            continue

                        file_name = f"f{frag_id}_l{label_idx}_{x1}_{y1}_{x2}_{y2}.npy"
                        img_file_path = os.path.join(img_path, file_name)
                        label_file_path = os.path.join(label_path, file_name)

                        np.save(img_file_path, img_patch)
                        np.save(label_file_path, label_patch)

                        patch_count_for_fragment += 1
                        patch_count_for_label_layer[label_idx] += 1

            total_patch_count += patch_count_for_fragment

            progress_bar.close()

            plot_patch_count_per_label_layer(dict(sorted(patch_count_for_label_layer.items())))

            if calc_mean_std:
                process_mean_and_std(images, data_type, frag_id, target_data_dir)

            del images, label
            gc.collect()

            print(f"Patch Count for Fragment '{get_frag_name_from_id(frag_id)}'", patch_count_for_fragment)
        except RuntimeError as e:
            print(e)

    print("Total Patch Count:", total_patch_count)


def plot_patch_count_per_label_layer(data):
    plt.figure(figsize=(10, 5))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.xlabel('Label Layer')
    plt.ylabel('Patch Count')
    plt.title('Patch Count per Label Layer')
    plt.xticks(list(data.keys()))
    plt.show()


def process_mean_and_std(images, data_type, frag_id, data_root_dir):
    if data_type != 'train':
        return

    mean_ch_wise = np.mean(images, axis=(1, 2), dtype=np.float64)
    std_ch_wise = np.std(images, axis=(1, 2), dtype=np.float64)

    json_file_path = os.path.join(data_root_dir, 'fragments_mean_std.json')

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    channels = images.shape[0]
    channel_ids = calc_original_channel_ids(channels=channels)

    channel_stats = {}
    for i, ch_id in enumerate(channel_ids):
        channel_stats[str(ch_id)] = {
            'mean': mean_ch_wise[i],
            'std': std_ch_wise[i]
        }

    data[f'fragment{frag_id}'] = channel_stats

    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)


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

    progress_bar = tqdm(total=num_files_to_select * 2, desc="Validation Dataset: Processing images and labels")

    # Move the selected image and label files
    move_files(train_img_dir, val_img_dir, selected_files, pbar=progress_bar)
    move_files(train_label_dir, val_label_dir, selected_files, pbar=progress_bar)

    progress_bar.close()


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


def plot_mean_std(data_root_dir):
    mean_std_path = os.path.join(data_root_dir, 'fragments_mean_std.json')

    with open(mean_std_path, 'r') as f:
        data = json.load(f)

    # Assume you know the number of channels, or extract it dynamically
    N_channels = max(len(fragment_channels) for fragment_channels in data.values())

    # Number of fragments
    M_fragments = len(data)

    # Initialize arrays to store data
    means = np.zeros((M_fragments, N_channels))
    stds = np.zeros((M_fragments, N_channels))
    fragments = [str(frag.replace('fragment', '')) for frag in data.keys()]

    channel_indices = {}
    for fragment_channels in data.values():
        for ch_id in fragment_channels.keys():
            if ch_id not in channel_indices:
                channel_indices[ch_id] = len(channel_indices)

    for i, (fragment_id, channels_data) in enumerate(data.items()):
        for ch_id, stats in channels_data.items():
            channel_index = channel_indices[ch_id]
            means[i, channel_index] = stats['mean']
            stds[i, channel_index] = stats['std']

    # Create subplots for mean and standard deviation
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    plot_type = 'scatter' if M_fragments == 1 else 'line'

    # Plot
    for ch_id, channel_index in channel_indices.items():
        if plot_type == 'scatter':
            ax[0].scatter(fragments, means[:, channel_index], label=f'Channel {ch_id}')
            ax[1].scatter(fragments, stds[:, channel_index], label=f'Channel {ch_id}')
        else:
            ax[0].plot(fragments, means[:, channel_index], label=f'Channel {ch_id}')
            ax[1].plot(fragments, stds[:, channel_index], label=f'Channel {ch_id}')

    ax[0].set_title('Channel-wise Mean across Fragments')
    ax[0].set_xlabel('Fragment')
    ax[0].set_ylabel('Mean')
    ax[0].legend()

    ax[1].set_title('Channel-wise Std across Fragments')
    ax[1].set_xlabel('Fragment')
    ax[1].set_ylabel('Std')
    ax[1].legend()

    plt.tight_layout()

    img_path = os.path.join(data_root_dir, 'mean_std.png')
    plt.savefig(img_path, dpi=300)

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
