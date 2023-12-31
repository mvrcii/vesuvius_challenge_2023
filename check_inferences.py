import argparse
import os

import numpy as np
from PIL import Image
from skimage.transform import resize

from slurm_inference import print_colored
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
    # Binarize the image: 0 remains 0, anything greater becomes 255
    return np.where(array > 0, 255, 0)

# Function to check if an array has more than x% zeros
def has_more_than_x_percent_zeros(array, threshold, mask=None, tolerance=0.1):
    # array = binarize_image(array)
    print(np.min(array), np.max(array))
    print(np.min(mask), np.max(mask))
    print(mask)
    if mask is not None:
        # Resize the mask to match the array dimensions
        mask_resized = resize(mask, array.shape, anti_aliasing=True)

        # Create a boolean mask for near-zero values in the resized mask
        mask_near_zero = mask_resized < tolerance

        # Calculate unique black pixels in the array not covered by the mask
        unique_black = (array == 0) & mask_near_zero
        unique_black_count = np.count_nonzero(unique_black)

        # Count non-black pixels in the mask
        non_black_mask_count = np.count_nonzero(mask_near_zero)
    else:
        print("Mask is None")
        unique_black_count = np.count_nonzero(array == 0)
        non_black_mask_count = np.size(array)

    # Calculate percentage and check against threshold
    black_pixel_percentage = unique_black_count / non_black_mask_count if non_black_mask_count > 0 else 0
    print("Black pixel percentage:", black_pixel_percentage)
    return black_pixel_percentage > threshold


def get_sys_args():
    parser = argparse.ArgumentParser(description="Check a given inference directory with numpy files.")
    parser.add_argument('checkpoint_list', type=str, help='Comma-separated list of checkpoint names')
    parser.add_argument('no_ink_ratio', type=float, default=0.5, help='A threshold that determines how '
                                                                      'much minimum percentage of non-ink (pixels with value 0)'
                                                                      'must be present in a .npy file.')
    return parser.parse_args()


def check_fragment_dir(checkpoints_to_check, inference_root_dir, threshold, work_dir):
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
        if os.path.isdir(fragment_path):
            for sub_dir in os.listdir(fragment_path):
                for checkpoint in checkpoints_to_check:
                    if checkpoint not in sub_dir:
                        continue
                    run_path = os.path.join(fragment_path, sub_dir)
                    if os.path.isdir(run_path):
                        for file in os.listdir(run_path):
                            if file.endswith(".npy"):
                                file_path = os.path.join(run_path, file)
                                array = np.load(file_path)
                                mask_path = os.path.join(work_dir, "data", "fragments",
                                                         f"fragment{fragment_id}", "mask.png")
                                if not os.path.isfile(mask_path):
                                    raise ValueError(f"Mask file does not exist for fragment: {fragment_id}")
                                mask = np.asarray(Image.open(mask_path))
                                mask = resize(mask, (array.shape[0], array.shape[1]), anti_aliasing=True)

                                if has_more_than_x_percent_zeros(array, threshold, mask=mask):
                                    zero_ints.append(file_path)

    for message in skip_list:
        print_colored(message, color='blue')

    return zero_ints, fail_load


def main():
    Image.MAX_IMAGE_PIXELS = None

    args = get_sys_args()
    config = Config().load_local_cfg()

    work_dir = os.path.expanduser(config.work_dir)
    inference_root_dir = os.path.join(work_dir, "inference", "results")

    checkpoints_to_check = args.checkpoint_list.split(',')
    for checkpoint in checkpoints_to_check:
        print_colored(f"INFO:\tChecking {checkpoint}", color="purple")

    zero_ints, fail_load = check_fragment_dir(checkpoints_to_check=checkpoints_to_check,
                                              inference_root_dir=inference_root_dir,
                                              threshold=args.no_ink_ratio,
                                              work_dir=work_dir)

    print_colored(f"\nFiles with > {args.no_ink_ratio} zero percentage:", color="purple")
    print_colored("----------------------------------------------------", color="purple")
    extract_info_from_paths(paths=zero_ints, work_dir=work_dir, inference_root_dir=inference_root_dir)
    if len(zero_ints) == 0:
        print_colored("None", color="purple")
    print_colored("----------------------------------------------------", color="purple")

    print(f"Files that failed to load:")
    print("----------------------------------------------------")
    for x in fail_load:
        print(x)
    if len(fail_load) == 0:
        print("None")


if __name__ == '__main__':
    main()
