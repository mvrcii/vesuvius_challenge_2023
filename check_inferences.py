import argparse
import os
import re

import numpy as np
from PIL import Image
from skimage.transform import resize

from slurm_inference import print_colored
from utility.checkpoints import CHECKPOINTS
from utility.configs import Config
from utility.fragments import get_frag_name_from_id, SUPERSEDED_FRAGMENTS, FRAGMENTS_IGNORE


# # Parse command-line arguments
# parser = argparse.ArgumentParser(description='Check for files with > x% zeros.')
# parser.add_argument('runs_to_check', type=str, help='Comma-separated list of run names')
# parser.add_argument('report_zero_percent', type=float, help='Percent threshold for zeros')
# args = parser.parse_args()


def extract_info_from_paths(paths, work_dir, inference_root_dir):
    for path in paths:
        # Remove the work directory part from the path
        relative_path = path.replace(inference_root_dir, '')
        del_path = path.replace(work_dir, '')

        # Extract the fragment ID, stride, and model name
        parts = relative_path.split('/')
        fragment_id = parts[1].replace('fragment', '')
        file_name = parts[-1]
        if 'stride-2' in file_name:
            stride = 'S2'
        elif 'stride-4' in file_name:
            stride = 'S4'
        else:
            stride = 'S?'

        if 'tta' in file_name:
            stride += '+ TTA'

        model_name_parts = parts[-2].split('_')[:2][-1].split('-')[0:2]  # First two substrings after the timestamp
        model_name = '-'.join(model_name_parts)

        frag_name = get_frag_name_from_id(fragment_id)

        # Print the formatted information
        print_colored(message=f"PLEASE CHECK:\t{frag_name:20} {fragment_id:10} {stride} ({model_name})", color="purple")
        print_colored(f"FULL PATH:\t{path}", color="purple")


def binarize_image(array):
    return np.where(array > 0, 1, 0)


# Function to check if an array has more than x% zeros
def calc_black_percentage(image, mask, downsample_factor=8):
    new_height = image.shape[0] // downsample_factor
    new_width = image.shape[1] // downsample_factor

    image_resized = resize(image, (new_height, new_width), anti_aliasing=False)
    image_binarized = binarize_image(image_resized)

    mask_resized = resize(mask, (new_height, new_width), anti_aliasing=False)
    mask_binarized = binarize_image(mask_resized)

    unique_black = (image_binarized == 0) & (mask_binarized == 0)
    unique_black_count = np.count_nonzero(unique_black)

    non_black_mask_count = np.count_nonzero(mask_binarized == 0)

    black_pixel_percentage = round(unique_black_count / non_black_mask_count if non_black_mask_count > 0 else 0, 6)

    return black_pixel_percentage


def get_sys_args():
    parser = argparse.ArgumentParser(description="Check a given inference directory with numpy files.")
    # parser.add_argument('checkpoint_list', type=str, help='Comma-separated list of checkpoint names')
    parser.add_argument('no_ink_ratio', type=float, default=0.5, help='A threshold that determines how '
                                                                      'much minimum percentage of non-ink (pixels with value 0)'
                                                                      'must be present in a .npy file.')
    return parser.parse_args()


def find_group_name_in_filename(filename, group_names):
    pattern = '|'.join(re.escape(name) for name in group_names)

    matches = re.findall(pattern, filename)

    if len(matches) == 1:
        return matches[0]
    else:
        return None


def detect_outliers(data, m=1.5):
    """
    Detect outliers in data. An outlier is defined as a value that is more than m standard deviations from the mean.
    Returns a tuple of (outliers, mean, standard deviation).
    Outliers are returned as a list of tuples, each tuple containing the index and the value of the outlier.
    """
    if not data:
        return [], None, None

    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    outliers = [(index, value) for index, value in enumerate(data) if abs(value - mean) > m * std_dev]

    return outliers, mean, std_dev


def custom_sort_key(file_name):
    if 'stride-4' in file_name:
        priority = 3
    elif 'tta-stride-2' in file_name:
        priority = 2
    elif 'stride-2' in file_name:
        priority = 1
    else:
        priority = 4  # Default for any other file names
    return priority, file_name


def check_fragment_dir(checkpoints_to_check, inference_root_dir, work_dir):
    zero_ints = []
    fail_load = []
    skip_list = []

    for fragment_dir in os.listdir(inference_root_dir):
        fragment_id = fragment_dir.split('fragment')[-1]

        if fragment_id in SUPERSEDED_FRAGMENTS:
            skip_list.append(f"SKIP:\t{get_frag_name_from_id(fragment_id):15} {fragment_id:15}\tis superseded")
            continue

        if fragment_id in FRAGMENTS_IGNORE:
            skip_list.append(f"SKIP:\t{get_frag_name_from_id(fragment_id):15} {fragment_id:15}\tis ignored")
            continue

        print_colored(f"INFO:\t{get_frag_name_from_id(fragment_id):15} {fragment_id:15}", color="blue")

        fragment_path = os.path.join(inference_root_dir, fragment_dir)

        groups = ['stride-2', 'stride-4', 'tta-stride-2']

        if not os.path.isdir(fragment_path):
            continue

        for checkpoint in os.listdir(fragment_path):
            checkpoint_str = checkpoint.split('_')[-1]

            for ckpt in checkpoints_to_check:
                if checkpoint_str in ckpt:
                    checkpoint_dir = os.path.join(fragment_path, checkpoint)

                    print_colored(f"INFO:\t{checkpoint_str.upper()}", color='blue')
                    black_group_stats = {group: [] for group in groups}

                    if not os.path.isdir(checkpoint_dir):
                        continue

                    for npy_file in sorted(os.listdir(checkpoint_dir), key=custom_sort_key):
                        if not npy_file.endswith('.npy'):
                            continue

                        npy_file_path = os.path.join(checkpoint_dir, npy_file)

                        group_name = find_group_name_in_filename(filename=npy_file, group_names=groups)
                        if not group_name:
                            print_colored('WARNING:\t No group name found for', npy_file_path, 'red')

                        image = np.load(npy_file_path)
                        mask_path = os.path.join(work_dir, "data", "fragments",
                                                 f"fragment{fragment_id}", "mask.png")
                        if not os.path.isfile(mask_path):
                            raise ValueError(f"Mask file does not exist for fragment: {fragment_id}")
                        mask = np.asarray(Image.open(mask_path))

                        if mask is None:
                            print_colored(f"ERROR:\tMask is none: {mask_path}", 'red')

                        black_pixel_percentage = calc_black_percentage(image=image, mask=mask)
                        print(f"{npy_file:50} -> {black_pixel_percentage:.4f}")
                        file_path = npy_file_path.replace(work_dir + os.sep, '')
                        black_group_stats[group_name].append((file_path, black_pixel_percentage))

                    for group, tuples in black_group_stats.items():
                        black_values = []
                        file_paths = []

                        for file_path, black_value in tuples:
                            black_values.append(black_value)
                            file_paths.append(file_path)

                        outliers, mean, std_dev = detect_outliers(black_values, m=2)

                        for idx, outlier_val in outliers:
                            print_colored(f"OUTLIER: in {group}: {outlier_val} -> {file_paths[idx]}", 'red')

    for message in skip_list:
        print_colored(message, color='blue')

    return zero_ints, fail_load


def main():
    Image.MAX_IMAGE_PIXELS = None

    # args = get_sys_args()
    config = Config().load_local_cfg()

    work_dir = os.path.expanduser(config.work_dir)
    inference_root_dir = os.path.join(work_dir, "inference", "results")

    checkpoints_to_check = [CHECKPOINTS['wise-energy'],
                            CHECKPOINTS['olive-wind'],
                            CHECKPOINTS['curious-rain']]

    print_colored("Checkpoints to check:", 'purple')
    for checkpoint in checkpoints_to_check:
        print_colored(f"{checkpoint}", color="purple")

    zero_ints, fail_load = check_fragment_dir(checkpoints_to_check=checkpoints_to_check,
                                              inference_root_dir=inference_root_dir,
                                              work_dir=work_dir)


if __name__ == '__main__':
    main()
