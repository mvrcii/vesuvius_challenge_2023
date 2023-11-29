import os
import sys

import cv2
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def process_image(img_src_dir, img_src_name, frag_id):
    try:
        target_dir = f"data/fragments/fragment{frag_id}/layered"
        os.makedirs(target_dir, exist_ok=True)

        pos = img_src_name.find('_')
        file_name = img_src_name[pos + 1:]

        img_target_dir = os.path.join(target_dir, file_name)

        # Load label image
        label_img = cv2.imread(os.path.join(img_src_dir, img_src_name), 0)
        label_arr = np.asarray(label_img)

        # Binarize the label
        binary_label = np.where(label_arr > 1, 1, 0)

        # Scale the binary_label to full 8-bit range
        scaled_label = (binary_label * 255).astype(np.uint8)

        # Save the scaled image
        cv2.imwrite(img_target_dir, scaled_label)

        return True, img_target_dir
    except Exception as e:
        return False, str(e)


def main(keyword="inklabels"):
    success_count = 0
    failure_count = 0

    layer_dir = os.path.join("data", "base_label_files", "layered")

    if len(os.listdir(layer_dir)) == 0:
        print("Error: No sub-directories found for:", layer_dir)
        return

    valid_file_count = 0
    for fragment_id in os.listdir(layer_dir):

        # Only process fragment directories within the layered folder
        sub_dir = os.path.join(layer_dir, fragment_id)
        print(sub_dir)
        if not os.path.isdir(sub_dir):
            print("Skipping element ", fragment_id)
            continue

        for file in os.listdir(sub_dir):
            if not file.endswith(".png") or keyword not in file:
                continue

            valid_file_count += 1
            success, message = process_image(img_src_dir=sub_dir, img_src_name=file, frag_id=fragment_id)

            if success:
                print(colored(f"Success: Processed {message}", "green"))
                success_count += 1
            else:
                print(colored(f"Failure: {message}", "red"))
                failure_count += 1

    print("\nReport:")
    print(colored(f"Total Files: {valid_file_count}", "blue"))
    print(colored(f"Total Successful: {success_count}", "green"))
    print(colored(f"Total Failures: {failure_count}", "red"))


if __name__ == "__main__":
    main("inklabels")
    main("negatives")
