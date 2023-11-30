import os
import sys

import numpy as np

def is_valid_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        # Check if any dimension is zero
        if 0 in data.shape:
            return False
        return True
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return False

def check_dataset(directory):
    invalid_files = []
    check_count = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.npy'):
                file_path = os.path.join(root, filename)
                check_count += 1
                if not is_valid_data(file_path):
                    invalid_files.append(file_path)
    print("Checked", check_count, "files")
    return invalid_files

if len(sys.argv) != 2:
    print("Usage: python util/check_dataset_dims.py <dataset_root_dir>")
    sys.exit(1)

dataset_dir = sys.argv[1]

# Directory containing your dataset
# data_directory = 'data/datasets/single/512px/OPTIMUS_STARSCREAM'
print("Dataset Directory to check:", dataset_dir)

# Check the dataset
invalid_data = check_dataset(dataset_dir)
print(f"Number of invalid data items: {len(invalid_data)}")
if invalid_data:
    print("Invalid data files:", invalid_data)
