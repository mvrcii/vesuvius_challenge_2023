from PIL import Image
import numpy as np
import os


def binarize_image(input_path, output_path, threshold=128):
    # Load the image
    image = Image.open(input_path).convert('L')  # Convert to grayscale if not already

    # Convert to numpy array for processing
    image_np = np.array(image)

    # Apply threshold
    image_np = (image_np > threshold) * 255  # Convert to 0 and 255

    # Convert back to PIL image and save
    binarized_image = Image.fromarray(np.uint8(image_np))
    binarized_image.save(output_path)

for img in os.listdir("."):
    if img.endswith(".png"):
        binarize_image(img, img)

