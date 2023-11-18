import argparse
import gc
import json
import logging
import os
import random
import shutil
import sys
import time
from multiprocessing import Manager, Pool

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conf import CFG
from util.train_utils import load_config

Image.MAX_IMAGE_PIXELS = None


def write_dataset_cfg(path, **kwargs):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'config.json')

    with open(path, 'w') as file:
        json.dump(kwargs, file, indent=4)


def write_config(_cfg):
    """
    Wrapper for config writing. Returns the target directory for the dataset.
    """
    return write_k_fold_cfg(_cfg) if _cfg.k_fold else write_single_fold_cfg(_cfg)


def write_single_fold_cfg(_cfg):
    logging.info("Starting single-fold dataset creation process...")
    result_dir_name = f'single_fold_{str(_cfg.patch_size)}px_{str(_cfg.dataset_in_chans)}ch'
    path = os.path.join(_cfg.dataset_target_dir, result_dir_name)

    channel_ids = calc_original_channel_ids(_cfg.dataset_in_chans)

    write_dataset_cfg(path,
                      single_train_frag_ids=_cfg.single_train_frag_id,
                      patch_size=_cfg.patch_size,
                      channels=_cfg.dataset_in_chans,
                      channel_ids=channel_ids,
                      mode='single_fold')

    logging.info(f"Data root directory for single-fold dataset: {path}")
    logging.info(f"Creating single-fold dataset with fragment id {_cfg.single_train_frag_id}")

    return path


def write_k_fold_cfg(_cfg):
    logging.info("Starting k-fold dataset creation process...")

    result_dir_name = f'k_fold_{str(_cfg.patch_size)}px_{str(_cfg.dataset_in_chans)}ch'
    path = os.path.join(_cfg.dataset_target_dir, result_dir_name)

    channel_ids = calc_original_channel_ids(_cfg.dataset_in_chans)

    write_dataset_cfg(path,
                      train_frag_ids=_cfg.train_frag_ids,
                      val_frag_ids=_cfg.val_frag_ids,
                      patch_size=_cfg.patch_size,
                      channels=_cfg.dataset_in_chans,
                      channel_ids=channel_ids,
                      k_fold=len(_cfg.train_frag_ids) + len(_cfg.val_frag_ids),
                      mode='k_fold')

    logging.info(f"Data root directory for k-fold: {path}")
    logging.info(
        f"Creating k-fold dataset with {len(_cfg.train_frag_ids)} training and {len(_cfg.val_frag_ids)} validation fragments")

    return path


def read_fragment(fragment_dir, data_root_dir, dataset_in_chans, patch_size):
    images = []
    pad0, pad1 = None, None

    print(f"Using {dataset_in_chans} channels for the dataset")
    channels = calc_original_channel_ids(dataset_in_chans)

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
    images = np.stack(images, axis=0)

    label_path = os.path.join(data_root_dir, fragment_dir, "inklabels.png")
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if label is None or label.shape[0] == 0:
        print("Label is empty or not loaded correctly:", label_path)

    assert pad0 is not None and pad1 is not None, "Padding is None or not set"

    label = np.pad(label, [(0, pad0), (0, pad1)], mode='constant', constant_values=0)
    label = (label / 255).astype(np.uint8)

    assert set(np.unique(np.array(label))) == {0, 1}, "Invalid label"

    return images, label


def process_fragment(dataset_information, fragment_id, data_type, progress_tracker):
    print(f"\nStart processing fragment {fragment_id} for {data_type} dataset..")
    create_dataset(dataset_information=dataset_information, fragment_id=fragment_id, data_type=data_type)
    progress_tracker.value += 1


def validate_fragments(_cfg):
    try:
        if _cfg.k_fold:
            for frag_id in _cfg.train_frag_ids:
                validate_fragment_files(frag_id=frag_id, channels=_cfg.dataset_in_chans)
        else:
            validate_fragment_files(frag_id=_cfg.single_train_frag_id, channels=_cfg.dataset_in_chans)

    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def calc_original_channel_ids(channels):
    mid = 65 // 2
    start = mid - channels // 2
    end = mid + channels // 2
    return list(range(start, end))


def validate_fragment_files(frag_id, channels):
    used_channels = calc_original_channel_ids(channels)

    fragments_path = os.path.join("data", "fragments")
    frag_dir = os.path.join(fragments_path, f"fragment{frag_id}")
    if not os.path.isdir(frag_dir):
        raise NotADirectoryError(f"Required fragment directory '{frag_dir}' does not exist.")

    for ch_idx in used_channels:
        ch_file_path = os.path.join(frag_dir, 'slices', f"{ch_idx:05d}.tif")
        if not os.path.isfile(ch_file_path):
            raise FileNotFoundError(f"Required channel {ch_idx} for fragment {frag_id} does not exist.")


def build_dataset(_cfg):
    validate_fragments(_cfg=_cfg)
    build_k_fold_dataset(_cfg) if _cfg.k_fold else build_single_fold_dataset(_cfg)


def build_single_fold_dataset(_cfg):
    target_dir = write_config(_cfg=_cfg)

    dataset_information = {
        "target_dir": target_dir,
        "data_root_dir": _cfg.data_root_dir,
        "dataset_in_chans": _cfg.dataset_in_chans,
        "patch_size": _cfg.patch_size,
        "calc_mean_std": _cfg.calc_mean_std
    }

    create_dataset(dataset_information=dataset_information, fragment_id=_cfg.single_train_frag_id)
    create_single_val_dataset(data_root_dir=target_dir, train_split=_cfg.train_split)

    if _cfg.calc_mean_std:
        plot_mean_std(target_dir)


def build_k_fold_dataset(_cfg):
    target_dir = write_config(_cfg=_cfg)

    with Manager() as manager:
        progress_tracker = manager.Value('i', 0)  # Shared integer for progress tracking
        total_tasks = len(_cfg.train_frag_ids) + len(_cfg.val_frag_ids)

        dataset_information = {
            "target_dir": target_dir,
            "data_root_dir": _cfg.data_root_dir,
            "dataset_in_chans": _cfg.dataset_in_chans,
            "patch_size": _cfg.patch_size,
            "calc_mean_std": _cfg.calc_mean_std
        }

        all_args = [(dataset_information, frag_id, 'train', progress_tracker) for frag_id in _cfg.train_frag_ids]
        all_args += [(dataset_information, frag_id, 'val', progress_tracker) for frag_id in _cfg.val_frag_ids]

        with Pool() as pool:
            # Use asynchronous map with callback to update the progress
            result = pool.starmap_async(process_fragment, all_args)

            while not result.ready():
                time.sleep(1)  # Update interval
                print(f"\rProgress: {progress_tracker.value}/{total_tasks}", end="")

        print("\nAll tasks completed.")

    if _cfg.calc_mean_std:
        plot_mean_std(target_dir)


def create_dataset(dataset_information, fragment_id, data_type='train'):
    target_dir = dataset_information["target_dir"]
    data_root_dir = dataset_information["data_root_dir"]
    dataset_in_chans = dataset_information["dataset_in_chans"]
    patch_size = dataset_information["patch_size"]
    calc_mean_std = dataset_information["calc_mean_std"]

    target_data_dir = os.path.join(target_dir, data_type)

    img_path = os.path.join(target_data_dir, "images")
    label_path = os.path.join(target_data_dir, "labels")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    fragment_dir = os.path.join(data_root_dir, "fragments", f"fragment{fragment_id}")
    if not os.path.isdir(fragment_dir):
        raise ValueError(f"Fragment directory does not exist: {fragment_dir}")

    mask_path = os.path.join(fragment_dir, f"mask.png")
    if not os.path.isfile(mask_path):
        raise ValueError(f"Mask file does not exist for fragment: {mask_path}")

    images, label = read_fragment(fragment_dir, data_root_dir, dataset_in_chans, patch_size)

    mask_arr = np.asarray(Image.open(mask_path))

    x1_list = list(range(0, images.shape[2] - CFG.patch_size + 1, CFG.stride))
    y1_list = list(range(0, images.shape[1] - CFG.patch_size + 1, CFG.stride))

    progress_bar = tqdm(total=len(x1_list) * len(y1_list),
                        desc=f"{data_type.capitalize()} Dataset Fragment {fragment_id}: Processing images "
                             f"and labels")

    skip_counter_not_in_mask = 0
    segformer_output_dim = CFG.SEGFORMER_OUTPUT_DIM

    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.patch_size
            x2 = x1 + CFG.patch_size
            progress_bar.update(1)

            img_patch = images[:, y1:y2, x1:x2]

            if mask_arr[y1:y2, x1:x2].all() != 1:  # Patch is not contained in mask
                skip_counter_not_in_mask += 1
                # plot_2d_arrays(label=label[y1:y2, x1:x2], image=img_patch)
                continue

            # Scale label down to match segformer output
            label_patch = resize(label[y1:y2, x1:x2], (128, 128), order=0, preserve_range=True,
                                 anti_aliasing=False)

            img_patch_resized = resize(images[:, y1:y2, x1:x2], (64, 128, 128), order=0, preserve_range=True,
                                 anti_aliasing=False)

            # Check that the label contains at least 10% ink
            if label_patch.sum() < np.prod(label_patch.shape) * CFG.REQUIRED_LABEL_INK_PERCENTAGE:
                # plot_2d_arrays(label=label_patch, image=img_patch)
                continue

            # if label_patch.sum() >= np.prod(label_patch.shape) * 0.25:
            #     # plot_2d_arrays(label=label_patch, image=img_patch_resized)
            #     continue

            file_name = f"f{fragment_id}_{x1}_{y1}_{x2}_{y2}.npy"
            img_file_path = os.path.join(img_path, file_name)
            label_file_path = os.path.join(label_path, file_name)

            np.save(img_file_path, img_patch)
            np.save(label_file_path, label_patch)

    progress_bar.close()

    if calc_mean_std:
        process_mean_and_std(images, data_type, fragment_id, target_data_dir)

    del images, label
    gc.collect()

    print("Patches skipped due to not in mask:", skip_counter_not_in_mask)


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


def merge_cfg_and_args(_args, _cfg):
    # Load config and eventually merge local config
    _cfg = load_config(CFG)

    if _args.k_fold is not None:
        logging.info(f"K-fold dataset creation mode set to {_cfg.k_fold} from command line argument.")
        _cfg.k_fold = args.k_fold


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run k-fold or single train-val dataset creation.")
    parser.add_argument('--k_fold', type=bool, default=None,
                        help='Enable k_fold dataset creation. Overrides CFG.k_fold if provided.')

    # Parse the arguments
    args = parser.parse_args()

    # Load the config and merge with local config
    config = load_config(CFG)

    # Merge config with arguments
    merge_cfg_and_args(_args=args, _cfg=config)

    # Build dataset
    build_dataset(_cfg=config)
