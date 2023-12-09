import os

import numpy as np


def print_array_info(path1, path2):
    # Load the arrays
    array1 = np.load(path1)
    array2 = np.load(path2)

    # Print shape, type, and filesize for each array
    print(f"Array 1: Shape - {array1.shape}, Type - {array1.dtype}, Filesize - {os.path.getsize(path1)} bytes")
    print(f"Array 2: Shape - {array2.shape}, Type - {array2.dtype}, Filesize - {os.path.getsize(path2)} bytes")


# Example usage
path_a = r"~/kaggle1stReimp/data/datasets/512/JETFIRE/labels/f20231005123336_ch60_9984_9728_10496_10240.npy"
path_b = r"~/kaggle1stReimp/data/datasets/512/JETFIRE/images/f20231005123336_ch60_9984_9728_10496_10240.npy"
print_array_info(path_a, path_b)
