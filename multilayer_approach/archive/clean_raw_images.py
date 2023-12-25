import os

import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


# Load the image
# path to current directory
base_path = "../base_labels"
source_path = os.path.join(base_path, '0_raw')
target_Path = os.path.join(base_path, '1_cleaned')
for file in tqdm(os.listdir(source_path)):
    if not file.endswith('.png'):
        continue
    image = np.asarray(Image.open(os.path.join(source_path, file)))
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Threshold to get a binary image
    # Remove small white dots
    kernel = np.ones((20, 20), np.uint8)
    cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # assert image only has 0 and 255 values
    assert np.array_equal(np.unique(cleaned), np.array([0, 255]))

    os.makedirs(target_Path, exist_ok=True)
    # Save the cleaned image
    save_path = os.path.join(target_Path, file)
    cv2.imwrite(save_path, cleaned)

print("Done cleaning images in", source_path)





