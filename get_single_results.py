import os
import sys
from PIL import Image

import numpy as np


def mean_of_npy_files(directory_path, fragment_id):
    # List to store the numpy arrays
    arrays = []

    # Iterating over each file in the directory
    for file in os.listdir(directory_path):
        if file.endswith(".npy"):
            # Load the numpy array and add it to the list
            file_path = os.path.join(directory_path, file)
            arrays.append(np.load(file_path))

    # Check if there are any numpy files
    if not arrays:
        print("No .npy files found in the directory.")
        return

    # Calculate the mean across all numpy arrays
    mean_array = np.mean(arrays, axis=0)

    # Save the mean array as a new .npy file with the fragment_id in the filename
    mean_file_path = os.path.join(directory_path, f'mean_array_{fragment_id}.npy')
    np.save(mean_file_path, mean_array)
    print(f"Mean array saved to {mean_file_path}")


if __name__ == "__main__":
    out_dir = os.path.join("fragments", "single_results")
    os.makedirs(out_dir, exist_ok=True)
    if len(sys.argv) != 3:
        print("Usage: python script.py <fragment_id> <checkpoint_name>")
        exit()

    fragment_id = sys.argv[1]
    checkpoint_name = sys.argv[2]

    path = os.path.join("inference", "results", f"fragment{fragment_id}")
    c_dir = None
    for x in os.listdir(path):
        if checkpoint_name in x:
            c_dir = x
            break
    if c_dir is None:
        print(f"No prediction found for fragment {fragment_id} with checkpoint {checkpoint_name}")

    c_path = os.path.join(path, c_dir)
    npy_files = []
    for x in os.listdir(c_path):
        if x.endswith(".npy"):
            file_path = os.path.join(c_path, x)
            print("found ", file_path)
            npy_files.append(np.load(file_path))
    result = np.mean(np.stack(npy_files), axis=0)
    result = (255 - (result * 255)).astype(np.uint8)

    # Save the images
    out_path = os.path.join(out_dir, f'fragment{fragment_id}_{checkpoint_name}.png')
    Image.fromarray(result).save(os.path.join(out_path))
    print("Saving to ", out_path)
