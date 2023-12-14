import os

import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


# Load the image
# path to current directory
path = "base_labels/0_raw"
for file in tqdm(os.listdir(path)):
    os.makedirs(os.path.join(path, 'cleaned'), exist_ok=True)

    if not file.endswith('.png'):
        continue
    image = np.asarray(Image.open(os.path.join(path, file)))
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Threshold to get a binary image
    # Remove small white dots
    kernel = np.ones((20, 20), np.uint8)
    cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # assert image only has 0 and 255 values
    assert np.array_equal(np.unique(cleaned), np.array([0, 255]))

    # Save the cleaned image
    save_path = os.path.join(path, "cleaned", file)
    cv2.imwrite(save_path, cleaned)
print("Done cleaning images in base_labels/0_raw")





