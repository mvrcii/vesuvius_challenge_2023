import argparse
import gc
import os

import numpy as np
from tqdm import tqdm
import cv2

from conf import CFG


def read_image(fragment_id):
    images = []

    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)

    path = r"C:\Users\Marce\Git-Master\Privat\vesuv\data"
    for i in tqdm(idxs):
        # image = cv2.imread(CFG.comp_dataset_path + f"{mode}/{fragment_id}/surface_volume/{i:02}.tif", 0)
        # img_path = os.path.join("A:\\", "projects_a", "Python", "vesuv", "data", "fragments", "fragment2", "slices",
        #                         f"{i:05}.tif")
        img_path = os.path.join(path, "fragments", f"fragment{fragment_id}", "slices", f"{i:05}.tif")

        image = cv2.imread(img_path, 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size) % CFG.tile_size
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size) % CFG.tile_size

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)

    label_path = os.path.join(path, "fragments/fragment2/inklabels_original.png")
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label = np.pad(label, [(0, pad0), (0, pad1)], mode='constant', constant_values=0)

    return images, label


def create_dataset(data_root_dir, dataset_type='train', fragment_id=2):
    data_dir = os.path.join(data_root_dir, dataset_type)

    img_path = os.path.join(data_dir, "images")
    label_path = os.path.join(data_dir, "labels")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    images, label = read_image(fragment_id)

    x1_list = list(range(0, images.shape[1] - CFG.tile_size + 1, CFG.stride))
    y1_list = list(range(0, images.shape[0] - CFG.tile_size + 1, CFG.stride))

    progress_bar = tqdm(total=len(x1_list) * len(y1_list), desc="Processing images")

    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if images[y1:y2, x1:x2].max() != 0:
                file_name = f"{x1}_{y1}_{x2}_{y2}.npy"

                img_file_path = os.path.join(img_path, file_name)
                label_file_path = os.path.join(label_path, file_name)

                if not os.path.exists(img_file_path):
                    np.save(img_file_path, images[y1:y2, x1:x2])

                if not os.path.exists(label_file_path):
                    np.save(label_file_path, label[y1:y2, x1:x2])

            progress_bar.update(1)
    progress_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dataset.')
    parser.add_argument('patch_size', type=int, help='Size of the patch.')
    parser.add_argument('dataset_type', type=str, choices=['train', 'test', 'val'],
                        help='Type of the dataset (train, test, val).')

    args = parser.parse_args()

    # Update CFG with the patch_size argument
    CFG.tile_size = args.patch_size
    CFG.size = CFG.tile_size

    create_dataset(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)), dataset_type=args.dataset_type)
