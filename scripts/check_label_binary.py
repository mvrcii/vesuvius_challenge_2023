import os

import cv2
import numpy as np

root_dir = os.path.join('data', 'base_label_binarized_single')

for checkpoint in os.listdir(root_dir):
    checkpoint_dir = os.path.join(root_dir, checkpoint)

    for frag_id in os.listdir(checkpoint_dir):
        frag_dir = os.path.join(checkpoint_dir, frag_id)

        for label_file in os.listdir(frag_dir):
            label_path = os.path.join(frag_dir, label_file)
            print("Checking:", label_path)

            label_img = cv2.imread(label_path)

            unique = np.unique(label_img)
            if len(unique) > 2:
                print(unique, label_path)
