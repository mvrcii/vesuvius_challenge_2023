import os

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from multilayer_approach.hdf5.create_dataset import load_padded_binary_image
from utility.configs import Config
from utility.fragments import get_frag_id_from_name, FragmentHandler


def draw_bounding_boxes_on_label(label, bbox_list, color=(255, 0, 0), opacity=0.4):
    """
    Draws bounding boxes on the label image using matrix operations for efficiency.

    :param label: Binary label image as a numpy array.
    :param bbox_list: List of bounding boxes, where each bbox is (x1, y1, x2, y2).
    :param color: Color of the bounding box in BGR format (default is red).
    :param opacity: Opacity of the bounding box.
    :return: RGB image with bounding boxes.
    """
    # Convert binary label to RGB
    label_rgb = np.stack([label] * 3, axis=-1) * 255

    # Preparing color overlay with opacity
    overlay_color = np.array(color) * opacity

    for x1, y1, x2, y2 in tqdm(bbox_list):
        # Fill bounding box area
        label_rgb[y1:y2, x1:x2] = (label_rgb[y1:y2, x1:x2] * (1 - opacity) + overlay_color).astype(np.uint8)
        label_rgb[y1:y2, x1:x2] += 1

    return label_rgb


if __name__ == '__main__':
    cfg = Config.load_local_cfg()
    label_base_dir = os.path.join(cfg.work_dir, "data", "labels", "6_twelve_layer_unetr_it3", "binarized")

    dataset_path = 'data/datasets/unetrpp_contrasted/64/'
    dataset = h5py.File('data/datasets/unetrpp_contrasted/64/dataset.hdf5', 'r')

    for frag_name in dataset.keys():
        print("Processing fragment", frag_name)

        if os.path.isfile(os.path.join(dataset_path, frag_name + '.jpg')):
            continue

        frag_id = get_frag_id_from_name(frag_name)
        label_dir = os.path.join(label_base_dir, frag_id, "inklabels.png")

        padded_label_arr = load_padded_binary_image(label_path=label_dir, patch_size=64)

        assert all(np.array(padded_label_arr.shape) == dataset[frag_name].attrs['label_dims'])

        bbox_list = []
        for patch in dataset[frag_name].keys():
            bbox = dataset[frag_name][patch].attrs['bbox']
            bbox_list.append(bbox)

        img = draw_bounding_boxes_on_label(padded_label_arr, bbox_list)

        frag_handler = FragmentHandler()
        rotate_num = frag_handler.get_rotation(frag_id=frag_id)
        flip_num = frag_handler.get_flip(frag_id=frag_id)

        if rotate_num is not None:
            img = np.rot90(img, rotate_num)
        if flip_num is not None:
            img = np.flip(img, flip_num)

        # Saving the image
        save_path = os.path.join(dataset_path, frag_name + '.jpg')
        cv2.imwrite(save_path, img)