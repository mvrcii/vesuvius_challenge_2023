import os
import sys

import numpy as np
from PIL import Image


def combine_arrays_to_image(directory, output_filename):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Initialize an array to store the combined result
    combined_array = None

    for file in files:
        # Load the numpy array from file
        array = np.load(os.path.join(directory, file))

        if combined_array is None:
            # Initialize the combined array with the first array
            combined_array = array
        else:
            # Take the maximum of the current and the new array
            combined_array = np.maximum(combined_array, array)

    # Normalize the combined array to the range 0-255
    combined_array = (combined_array * 255).astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(combined_array)

    # Save the image
    image.save(output_filename)

    return output_filename


if __name__ == "__main__":
    # Check if the user provided a folder path as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
    else:
        # Example usage:
        folder_path = sys.argv[1]
        combine_arrays_to_image(folder_path, folder_path + 'max_combined_image.png')
