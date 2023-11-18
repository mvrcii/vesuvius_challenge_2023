import os

import numpy as np
import cv2

# Load the logits file
file_b2 = np.load(os.path.join("20231118-042028", "frag_20230522181603_20231118-042028result.npy"))
file_b0 = np.load(os.path.join("20231118-043435", "frag_20230522181603_20231118-043435result.npy"))

file = (file_b2 + file_b0) / 2

# Find the min and max values in the array
min_val = file.min()
max_val = file.max()

# Normalize the logits to the range 0-255
normalized_logits = (file - min_val) / (max_val - min_val) * 255

# Convert to uint8 for image representation
image_data = normalized_logits.astype(np.uint8)

# Resize the image to the desired dimensions
resized_image = cv2.resize(image_data, (14848, 5632), interpolation=cv2.INTER_LINEAR)

# Save the resized image as grayscale
cv2.imwrite("ensemble_logits.png", resized_image)
