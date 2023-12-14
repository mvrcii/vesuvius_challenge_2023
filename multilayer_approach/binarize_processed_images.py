import os
import cv2
import numpy as np
from tqdm import tqdm

source_path = os.path.join("base_labels", "2_processed")
destination_path = os.path.join("base_labels", "3_binarized")

for frag_id in tqdm(os.listdir(source_path)):
    frag_dir_destination = os.path.join(destination_path, frag_id)
    frag_dir_source = os.path.join(source_path, frag_id)

    os.makedirs(frag_dir_destination, exist_ok=True)

    for img_name in os.listdir(frag_dir_source):
        img_src_path = os.path.join(frag_dir_source, img_name)
        img_dest_path = os.path.join(destination_path, frag_id, img_name)

        img = cv2.imread(img_src_path, 0)
        img_arr = np.asarray(img)

        # Binarize the label
        binary_img = np.where(img_arr > 128, 1, 0).astype(np.uint8)
        binary_img *= 255

        # Assert image is binary
        assert np.all(np.unique(binary_img) == [0, 255])

        cv2.imwrite(img_dest_path, binary_img)
        print("saved to ", img_dest_path)









