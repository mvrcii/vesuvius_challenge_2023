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
    # Binarize the image: 0 remains 0, anything greater becomes 255
    return np.where(array > 0.01, 1, 0)


# Function to check if an array has more than x% zeros
def calc_black_percentage(image, mask):
    image = binarize_image(image)

    mask_resized = resize(mask, image.shape, anti_aliasing=True)
    mask_resized = mask_resized == 0.0

    unique_black = (image == 0) & mask_resized
    unique_black_count = np.count_nonzero(unique_black)

    non_black_mask_count = np.count_nonzero(mask_resized)

    black_pixel_percentage = round(unique_black_count / non_black_mask_count if non_black_mask_count > 0 else 0, 6)

    return black_pixel_percentage


def get_sys_args():
    parser = argparse.ArgumentParser(description="Check a given inference directory with numpy files.")
    parser.add_argument('checkpoint_list', type=str, help='Comma-separated list of checkpoint names')
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

        groups = ['stride-2', 'stride-4', 'tta-stride-2']
        black_group_stats = {group: [] for group in groups}

        if not os.path.isdir(fragment_path):
            continue

        for checkpoint in os.listdir(fragment_path):
            checkpoint = checkpoint.split('_')[-1]

            for ckpt in checkpoints_to_check:
                if checkpoint in ckpt:
                    print(checkpoint, ckpt)
                    checkpoint_dir = os.path.join(fragment_path, checkpoint)

                    if not os.path.isdir(checkpoint_dir):
                        continue

                    for npy_file in os.listdir(checkpoint_dir):
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
                        # mask = resize(mask, (image.shape[0], image.shape[1]), anti_aliasing=True)

                        if mask is None:
                            print_colored(f"ERROR:\tMask is none: {mask_path}", 'red')

                        black_pixel_percentage = calc_black_percentage(image=image, mask=mask)
                        black_group_stats[group_name].append(black_pixel_percentage)

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

    checkpoints_to_check = [CHECKPOINTS['wise-energy'],
                            CHECKPOINTS['olive-wind'],
                            CHECKPOINTS['curious-rain']]

    print_colored("Checkpoints to check:", 'purple')
    for checkpoint in checkpoints_to_check:
        print_colored(f"{checkpoint}", color="purple")

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
