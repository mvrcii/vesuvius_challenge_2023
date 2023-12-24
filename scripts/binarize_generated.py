import os
import sys

import cv2
import numpy as np
from termcolor import colored
from tqdm import tqdm

from utility.fragments import get_frag_name_from_id
from utility.meta_data import AlphaBetaMeta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('../')



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
    model_label_path = AlphaBetaMeta().get_current_base_label_dir()
    model = AlphaBetaMeta().get_previous_model()
    label_target_path = AlphaBetaMeta().get_label_target_dir()
    it = AlphaBetaMeta().get_current_iteration()
    phase = AlphaBetaMeta().get_current_phase()
    fragments = AlphaBetaMeta().get_current_train_fragments()
    fragments_str = " ".join(fragments)

    os.makedirs(label_target_path, exist_ok=True)
    success_count = 0
    exist_count = 0
    failure_count = 0
    valid_file_count = 0

    if not model:
        print("Make sure that ")
        raise Exception("Make sure that the iteration is set correctly.")

    print(f">>> BINARIZE LABELS <<<")
    print(f"Iteration:\t\t\t{it}\nTrain Phase:\t\t{phase}")
    print(f"Train Fragments:\t{fragments_str}")
    print(f"Label Model:\t\t{model}\t")
    for fragment_id in os.listdir(model_label_path):
        frag_name = get_frag_name_from_id(fragment_id).upper()

        # Only process fragment directories within the layered folder
        sub_dir = os.path.join(model_label_path, fragment_id)
        out_dir = os.path.join(label_target_path, model, fragment_id)

        if not os.path.isdir(sub_dir):
            print("Skipping element ", fragment_id)
            continue

        for img_src_name in tqdm(os.listdir(sub_dir), desc=f"Processing {frag_name} '{fragment_id}'"):
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

    print(colored(f"\nTotal Files:\t\t{valid_file_count}", "blue"))
    print(colored(f"Total Existing:\t\t{exist_count}", "blue"))
    print(colored(f"Total Successful:\t{success_count}", "green"))
    print(colored(f"Total Failures:\t\t{failure_count}", "red"))


if __name__ == "__main__":
    main("inklabels")
