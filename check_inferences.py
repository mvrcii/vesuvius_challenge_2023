import os

import numpy as np

report_zero_percent = 0.03  # if file has > 60% zeros, print something
runs_to_ckeck = ["olive-wind-1194-unetr-sf-b5-231231-064008"]

print("Checking:")
for x in runs_to_ckeck:
    print(x)

# Path to the fragments directory
fragments_dir = os.path.expanduser("~/kaggle1stReimp/inference/results")  # Update with the path to your fragments directory
print("Searching in ", fragments_dir)


# Function to check if an array has more than 60% zeros
def has_more_than_60_percent_zeros(array):
    return np.count_nonzero(array == 0) / array.size > report_zero_percent


# Iterate over each fragment directory
for fragment_id in os.listdir(fragments_dir):
    print("Checking", fragment_id)
    fragment_path = os.path.join(fragments_dir, fragment_id)
    if os.path.isdir(fragment_path):
        # Check each run name directory
        for sub_dir in os.listdir(fragment_path):
            for run_name in runs_to_ckeck:
                if not sub_dir.endswith(run_name):
                    continue
                run_path = os.path.join(fragment_path, sub_dir)
                if os.path.isdir(run_path):
                    # Process each .npy file
                    for file in os.listdir(run_path):
                        if file.endswith(".npy"):
                            file_path = os.path.join(run_path, file)
                            array = np.load(file_path)
                            if has_more_than_60_percent_zeros(array):
                                print(f"File with >{report_zero_percent} zeros: {file_path}")
                            else:
                                print(f"{file} is correct")
