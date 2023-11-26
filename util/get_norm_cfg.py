import os
import glob
from tqdm import tqdm
import numpy as np

# Define the path to the images
image_path = '../data/512/train/images'

# List all numpy array files in the directory
image_files = glob.glob(os.path.join(image_path, '*.npy'))

# Initialize variables to store mean, M2 (sum of squares of differences from the current mean), and count
mean_channels = np.zeros(16)
M2_channels = np.zeros(16)
num_pixels = 0

# Iterate over each file and calculate the mean and M2
for file in tqdm(image_files):
    # Load the numpy array
    image = np.load(file)

    # Ensure the image has the correct shape
    if image.shape == (16, 512, 512):
        num_pixels += 512 * 512
        for i in range(16):
            delta = np.sum(image[i, :, :]) - mean_channels[i] * 512 * 512
            mean_channels[i] += delta / num_pixels
            squared_diff = (image[i, :, :] - mean_channels[i]) ** 2
            M2_channels[i] += np.sum(squared_diff)

# Calculate the standard deviation for each channel
std_dev_channels = np.sqrt(M2_channels / num_pixels)

# Adjusting precision to three decimal places
mean_channels = np.around(mean_channels, 3)
std_dev_channels = np.around(std_dev_channels, 3)

# Save the results to a Python file
norm_cfg_content = f"mean = {list(mean_channels)}\nstd_dev = {list(std_dev_channels)}"

with open('../data/512/train/norm_cfg.py', 'w') as file:
    file.write(norm_cfg_content)
