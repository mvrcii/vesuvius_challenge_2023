import os
import numpy as np
from tqdm import tqdm


def check_images(directory, expected_shape, value_range):
    print(f"Checking {directory}")
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.npy'):  # assuming the files are saved in .npy format
            file_path = os.path.join(directory, filename)
            array = np.load(file_path)

            # Check shape
            if array.shape != expected_shape:
                print(f"{filename} has an incorrect shape: {array.shape}")
                exit()

            # Check value range
            if array.min() < value_range[0] or array.max() > value_range[1]:
                print(f"{filename} has values outside the range {value_range}")
                exit()


# Check images in data/train/images
check_images('../data/512/train/images', (16, 512, 512), (0, 255))
check_images('../data/512/val/images', (16, 512, 512), (0, 255))

# Check labels in data/train/labels
check_images('../data/512/train/labels', (128, 128), (0, 1))
check_images('../data/512/val/labels', (128, 128), (0, 1))

print("Checking complete, all files in valid format.")
