import os
import numpy as np
import argparse

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
fragments_dir = os.path.expanduser("~/kaggle1stReimp/inference/results")  # Update with the path to your fragments directory
print("Searching in", fragments_dir)


# Function to check if an array has more than x% zeros
def has_more_than_x_percent_zeros(array, x):
    return np.count_nonzero(array == 0) / array.size > x

zero_ints = []
fail_load = []
# Iterate over each fragment directory
for fragment_id in os.listdir(fragments_dir):
    print("Checking", fragment_id)
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
                            # try:
                            array = np.load(file_path)
                            if has_more_than_x_percent_zeros(array, report_zero_percent):
                                zero_ints.append(file_path)
                            # except Exception as e:
                            #     fail_load.append(file_path)

print(f"Files with > {report_zero_percent} zero percentage:")
print("----------------------------------------------------")
for x in zero_ints:
    print(x)
print("")
print(f"Files that failed to load:")
print("----------------------------------------------------")
for x in fail_load:
    print(x)
