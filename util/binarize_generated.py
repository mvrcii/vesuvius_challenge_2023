import os
import sys

import cv2
import numpy as np
from termcolor import colored
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('../')

from constants import get_frag_name_from_id
from meta import AlphaBetaMeta


def process_image(img_src_dir, img_target_dir):
    try:
        # Load label image
        label_img = cv2.imread(img_src_dir, 0)
        label_arr = np.asarray(label_img)

        # Normalize label array
        label_arr = label_arr / 255.0

        # Binarize the label
        binary_label = np.where(label_arr > 0.2, 1, 0)

        # Scale the binary_label to full 8-bit range
        scaled_label = (binary_label * 255).astype(np.uint8)

        # Save the scaled image
        cv2.imwrite(img_target_dir, scaled_label)

        return True, img_target_dir
    except Exception as e:
        return False, str(e)


def extract_image_target_name(full_image_name):
    pos = full_image_name.find('_')

    if 'superseded' in full_image_name:
        pos = full_image_name.find('_', pos + 1)

    file_name = full_image_name[pos + 1:]

    return file_name


def main(keyword="inklabels"):
    model_label_path = AlphaBetaMeta().get_current_label_dir()
    model = AlphaBetaMeta().get_current_model()
    label_target_path = AlphaBetaMeta().get_label_target_dir()

    os.makedirs(label_target_path, exist_ok=True)
    success_count = 0
    exist_count = 0
    failure_count = 0
    valid_file_count = 0

    print(f"Binarizing label files for model checkpoint: {model}")
    for fragment_id in os.listdir(model_label_path):
        frag_name = get_frag_name_from_id(fragment_id).upper()

        # Only process fragment directories within the layered folder
        sub_dir = os.path.join(model_label_path, fragment_id)
        out_dir = os.path.join(label_target_path, model, fragment_id)

        if not os.path.isdir(sub_dir):
            print("Skipping element ", fragment_id)
            continue

        for img_src_name in tqdm(os.listdir(sub_dir), desc=f"Processing {frag_name}"):
            if not img_src_name.endswith(".png") or "inklabels" not in img_src_name:
                continue

            valid_file_count += 1

            img_target_name = extract_image_target_name(img_src_name)
            img_target_path = os.path.join(out_dir, img_target_name)
            img_src_path = os.path.join(sub_dir, img_src_name)

            if os.path.isfile(img_target_path):
                exist_count += 1
                continue

            os.makedirs(out_dir, exist_ok=True)

            success, message = process_image(img_src_dir=img_src_path, img_target_dir=img_target_path)

            if success:
                # print(colored(f"Success: Processed {message}", "green"))
                success_count += 1
            else:
                # print(colored(f"Failure: {message}", "red"))
                failure_count += 1

    print(f"\n")
    print(colored(f"Total Files:\t\t{valid_file_count}", "blue"))
    print(colored(f"Total Existing:\t\t{exist_count}", "blue"))
    print(colored(f"Total Successful:\t{success_count}", "green"))
    print(colored(f"Total Failures:\t\t{failure_count}", "red"))


if __name__ == "__main__":
    main("inklabels")
