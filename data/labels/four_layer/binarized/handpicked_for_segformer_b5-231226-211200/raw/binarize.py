from PIL import Image
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = None

def check_and_binarize_image(image_path, threshold=128):
    try:
        image = Image.open(image_path).convert('L')
        image_np = np.array(image)

        # Check if the image is already binarized
        if np.all(np.logical_or(image_np <= threshold, image_np >= 255)):
            print(f"'{image_path}' is already binarized, skipping.")
            return

        # Binarize and save the image
        image_np = (image_np > threshold) * 255
        binarized_image = Image.fromarray(np.uint8(image_np))
        binarized_image.save(image_path)
        print(f"Binarized '{image_path}'")

    except IOError:
        print(f"Skipping file (IOError/Decompression Bomb): {image_path}")


def rename_and_process_files_in_subdirectories(directory):
    for subdir, _, files in os.walk(directory):

        for filename in files:
            if filename.endswith(".png"):
                file_path = os.path.join(subdir, filename)
                dir_name, basename = os.path.split(file_path)

                # Correcting renaming logic
                if basename.startswith('2023') and '_' in basename:
                    parts = basename.split('_', 1)
                    if len(parts) == 2 and len(parts[0]) == 14:  # Check for a 14-digit prefix
                        new_filename = parts[1]
                        new_file_path = os.path.join(dir_name, new_filename)
                        os.rename(file_path, new_file_path)
                        print(f"Renamed '{file_path}' to '{new_file_path}'")
                        file_path = new_file_path

                # Check if binarizing is necessary
                check_and_binarize_image(file_path)


if __name__ == "__main__":
    # Provide the directory relative to the script location or an absolute path
    directory = "."  # This goes one directory up from the script's location
    rename_and_process_files_in_subdirectories(directory)
