import os
import sys

import cv2
import numpy as np
from termcolor import colored

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def process_image(img_src_dir, img_src_name, out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)

        pos = img_src_name.find('_')

        if 'superseded' in img_src_name:
            pos = img_src_name.find('_', pos + 1)

        file_name = img_src_name[pos + 1:]

        img_target_path = os.path.join(out_dir, file_name)

        # Load label image
        label_img = cv2.imread(os.path.join(img_src_dir, img_src_name), 0)
        label_arr = np.asarray(label_img)

        # Normalize label array
        label_arr = label_arr / 255.0

        # Binarize the label
        binary_label = np.where(label_arr > 0.2, 1, 0)

        # Scale the binary_label to full 8-bit range
        scaled_label = (binary_label * 255).astype(np.uint8)

        # Save the scaled image
        cv2.imwrite(img_target_path, scaled_label)

        return True, img_target_path
    except Exception as e:
        return False, str(e)


def main(keyword="inklabels"):
    os.makedirs(os.path.join("data", "base_label_binarized"))
    success_count = 0
    failure_count = 0
    valid_file_count = 0

    model_dir = os.path.join("data", "base_label_files", "model_generated")
    for run_dir in os.listdir(model_dir):
        for fragment_id in os.listdir(os.path.join(model_dir, run_dir)):
            # Only process fragment directories within the layered folder
            sub_dir = os.path.join(model_dir, run_dir, fragment_id)
            out_dir = os.path.join("data", "base_label_binarized", run_dir, fragment_id)

            if not os.path.isdir(sub_dir):
                print("Skipping element ", fragment_id)
                continue

            for file in os.listdir(sub_dir):
                if not file.endswith(".png") or "inklabels" not in file:
                    continue

                valid_file_count += 1
                success, message = process_image(img_src_dir=sub_dir, img_src_name=file, out_dir=out_dir)

                if success:
                    print(colored(f"Success: Processed {message}", "green"))
                    success_count += 1
                else:
                    print(colored(f"Failure: {message}", "red"))
                    failure_count += 1

    print(f"\nReport {keyword}:")
    print(colored(f"Total Files: {valid_file_count}", "blue"))
    print(colored(f"Total Successful: {success_count}", "green"))
    print(colored(f"Total Failures: {failure_count}", "red"))


if __name__ == "__main__":
    main("inklabels")
