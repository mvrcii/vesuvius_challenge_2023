import argparse
import os

import numpy as np

from slurm_inference import print_colored
from utility.configs import Config
from utility.fragments import get_frag_name_from_id, SUPERSEDED_FRAGMENTS, FRAGMENTS_IGNORE


# # Parse command-line arguments
# parser = argparse.ArgumentParser(description='Check for files with > x% zeros.')
# parser.add_argument('runs_to_check', type=str, help='Comma-separated list of run names')
# parser.add_argument('report_zero_percent', type=float, help='Percent threshold for zeros')
# args = parser.parse_args()


def extract_info_from_paths(paths, work_dir):
    for path in paths:
        # Remove the work directory part from the path
        relative_path = path.replace(work_dir, '')

        # Extract the fragment ID, stride, and model name
        parts = relative_path.split('/')
        fragment_id = parts[2].replace('fragment', '')
        file_name = parts[-1]
        if 'stride-2' in file_name:
            stride = 'S2'
        elif 'stride-4' in file_name:
            stride = 'S4'
        else:
            stride = 'S?'
        model_name_parts = parts[-2].split('_')[:2]  # First two substrings after the timestamp
        model_name = '-'.join(model_name_parts)

        # Retrieve the fragment name
        frag_name = get_frag_name_from_id(fragment_id)

        # Print the formatted information
        print_colored(message=f"PLEASE CHECK: {frag_name:20} {fragment_id:10} {stride} ({model_name})", color="purple")


# Function to check if an array has more than x% zeros
def has_more_than_x_percent_zeros(array, x):
    return np.count_nonzero(array == 0) / array.size > x


def get_sys_args():
    parser = argparse.ArgumentParser(description="Check a given inference directory with numpy files.")
    parser.add_argument('checkpoint_list', type=str, help='Comma-separated list of checkpoint names')
    parser.add_argument('no_ink_ratio', type=float, default=0.5, help='A threshold that determines how '
                                                                      'much minimum percentage of non-ink (pixels with value 0)'
                                                                      'must be present in a .npy file.')
    return parser.parse_args()


def check_fragment_dir(checkpoints_to_check, inference_root_dir, threshold):
    zero_ints = []
    fail_load = []

    for fragment_id in os.listdir(inference_root_dir):
        fragment_id = fragment_id.split('fragment')[-1]

        if fragment_id in SUPERSEDED_FRAGMENTS:
            print_colored(f"SKIP:\t {get_frag_name_from_id(fragment_id)} is superseded", color='blue')
            continue

        if fragment_id in FRAGMENTS_IGNORE:
            print_colored(f"SKIP:\t {get_frag_name_from_id(fragment_id)} is being ignored", color='blue')
            continue

        print_colored(f"Checking {get_frag_name_from_id(fragment_id):15} '{fragment_id}'", color="blue")
        fragment_path = os.path.join(inference_root_dir, fragment_id)
        if os.path.isdir(fragment_path):
            # Check each run name directory
            for sub_dir in os.listdir(fragment_path):
                for checkpoint in checkpoints_to_check:
                    if checkpoint not in sub_dir:
                        continue
                    run_path = os.path.join(fragment_path, sub_dir)
                    if os.path.isdir(run_path):
                        for file in os.listdir(run_path):
                            if file.endswith(".npy"):
                                file_path = os.path.join(run_path, file)
                                try:
                                    array = np.load(file_path)
                                    if has_more_than_x_percent_zeros(array, threshold):
                                        zero_ints.append(file_path)
                                except Exception as e:
                                    fail_load.append(file_path)
    return zero_ints, fail_load


def main():
    args = get_sys_args()
    config = Config().load_local_cfg()

    work_dir = os.path.expanduser(config.work_dir)
    inference_root_dir = os.path.join(work_dir, "inference", "results")

    checkpoints_to_check = args.checkpoint_list.split(',')
    for checkpoint in checkpoints_to_check:
        print_colored(f"INFO:\tChecking {checkpoint}", color="purple")

    zero_ints, fail_load = check_fragment_dir(checkpoints_to_check=checkpoints_to_check,
                                              inference_root_dir=inference_root_dir,
                                              threshold=args.no_ink_ratio)

    print_colored(f"\nFiles with > {args.no_ink_ratio} zero percentage:", color="purple")
    print_colored("----------------------------------------------------", color="purple")
    extract_info_from_paths(paths=zero_ints, work_dir=work_dir)
    if len(zero_ints) == 0:
        print_colored("None", color="purple")
    print_colored("----------------------------------------------------", color="purple")


    # print("")
    # print(f"Files that failed to load:")
    # print("----------------------------------------------------")
    # for x in fail_load:
    #     print(x)
    # if len(fail_load) == 0:
    #     print("None")


if __name__ == '__main__':
    main()
