import os
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# Create the subfolder if it doesn't exist
if not os.path.exists('scaled_down'):
    os.makedirs('scaled_down')

# Iterate through all files in the current directory
for filename in tqdm(os.listdir('')):
    if filename.endswith('.png'):
        img = Image.open(filename)
        # Scale the image down to 25% of its size
        factor = 2
        img = img.resize((img.width // factor, img.height // factor))
        # Save the scaled image in the subfolder
        img.save(f'scaled_down/{filename}')
