import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def combine_and_plot_arrays(directory_path, save_path):
    arrays = []

    # Load all numpy arrays from the directory
    for filename in tqdm(os.listdir(directory_path)):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory_path, filename)
            array = np.load(file_path)
            arrays.append(array)

    # Combine arrays by taking the maximum for each pixel
    combined_array = np.maximum.reduce(arrays)

    # Display and save the plot
    plt.imshow(combined_array, cmap='gray')  # Change colormap if needed
    plt.colorbar()
    plt.savefig(save_path)


if __name__ == "__main__":
    # Check if the user provided a folder path as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
    else:
        # Example usage:
        folder_path = sys.argv[1]
        combine_and_plot_arrays(folder_path, folder_path + 'max_combined_image.png')
        print("Saved image to " + folder_path + "/max_combined_image.png")
