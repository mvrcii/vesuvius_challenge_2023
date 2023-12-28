import cv2
import numpy as np


def erode_shapes(image_path, erosion_size):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to get the shapes
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Define the erosion kernel
    kernel = np.ones((erosion_size, erosion_size), np.uint8)

    # Erode the image
    eroded_image = cv2.erode(thresh, kernel, iterations=1)

    return eroded_image


# Usage
eroded_shapes = erode_shapes('jetfire_ink.png', 20)  # Adjust erosion size as needed
cv2.imwrite('eroded_shapes20.jpg', eroded_shapes)
