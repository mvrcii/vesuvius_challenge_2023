import argparse
import os

import numpy as np

from slurm_inference import print_colored
from utility.configs import Config
from utility.fragments import get_frag_name_from_id

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Check for files with > x% zeros.')
parser.add_argument('runs_to_check', type=str, help='Comma-separated list of run names')
parser.add_argument('report_zero_percent', type=float, help='Percent threshold for zeros')
args = parser.parse_args()

report_zero_percent = args.report_zero_percent
runs_to_check = args.runs_to_check.split(',')

print("Checking:")
for x in runs_to_check:
    print(x)

# Path to the fragments directory
fragments_dir = os.path.expanduser(
    "~/kaggle1stReimp/inference/results")  # Update with the path to your fragments directory
print("Searching in", fragments_dir)


def extract_info_from_paths(paths, work_dir):
    for path in paths:
        # Remove the work directory part from the path
        relative_path = path.replace(work_dir, '')

        # Extract the fragment ID, stride, and model name
        parts = relative_path.split('/')
        fragment_id = parts[2].replace('fragment', '')
        file_name = parts[-1]
        stride = 'S2' if 'stride-2' in file_name else 'Unknown'  # Adjust this logic as needed
        model_name_parts = parts[-2].split('_')[:2]  # First two substrings after the timestamp
        model_name = '-'.join(model_name_parts)

        # Retrieve the fragment name
        frag_name = get_frag_name_from_id(fragment_id)

        # Print the formatted information
        print_colored(message=f"TO BE CHECKED: {frag_name} {stride} ({model_name})", color="purple")


# Function to check if an array has more than x% zeros
def has_more_than_x_percent_zeros(array, x):
    return np.count_nonzero(array == 0) / array.size > x


def get_sys_args():
    parser = argparse.ArgumentParser(description="Check a given inference directory with numpy files.")
    parser.add_argument('fragments_root_dir', type=str, help='The root directory of all fragment inference results to check.')
    parser.add_argument('ink_threshold', type=float, default=0.5, help='A threshold that determines how '
                                                                     'much minimum percentage of non-ink (pixels with value 0)'
                                                                     'must be present in a .npy file.')
    return parser.parse_args()

def check_fragment_dir(fragments_dir):
    zero_ints = []
    fail_load = []
    # Iterate over each fragment directory
    for fragment_id in os.listdir(fragments_dir):
        print_colored(f"Checking {get_frag_name_from_id(fragment_id):15} '{fragment_id}'", color="blue")
        fragment_path = os.path.join(fragments_dir, fragment_id)
        if os.path.isdir(fragment_path):
            # Check each run name directory
            for sub_dir in os.listdir(fragment_path):
                # print("Checking", sub_dir)
                for run_name in runs_to_check:
                    if run_name not in sub_dir:
                        # print(sub_dir)
                        # print("does not end with ", run_name)
                        continue
                    run_path = os.path.join(fragment_path, sub_dir)
                    if os.path.isdir(run_path):
                        # print("searching in ", run_path)
                        # Process each .npy file
                        for file in os.listdir(run_path):
                            if file.endswith(".npy"):
                                file_path = os.path.join(run_path, file)
                                try:
                                    array = np.load(file_path)
                                    if has_more_than_x_percent_zeros(array, report_zero_percent):
                                        zero_ints.append(file_path)
                                except Exception as e:
                                    fail_load.append(file_path)
    return zero_ints, fail_load


def main():
    args = get_sys_args()
    config = Config().load_local_cfg()

    zero_ints, fail_load = check_fragment_dir(fragments_dir=args.fragments_root_dir)
    extract_info_from_paths(paths=zero_ints, work_dir=config.work_dir)
    #
    # print("")
    # print(f"Files with > {report_zero_percent} zero percentage:")
    # print("----------------------------------------------------")
    # for x in zero_ints:
    #     print(x)
    # if len(zero_ints) == 0:
    #     print("None")
    # print("")
    # print(f"Files that failed to load:")
    # print("----------------------------------------------------")
    # for x in fail_load:
    #     print(x)
    # if len(fail_load) == 0:
    #     print("None")


if __name__ == '__main__':
    main()
