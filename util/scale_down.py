import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Create the subfolder if it doesn't exist
if not os.path.exists('scaled_down'):
    os.makedirs('scaled_down')

# Iterate through all files in the current directory
for filename in os.listdir('.'):
    if filename.endswith('.png'):
        img = Image.open(filename)
        # Scale the image down to 25% of its size
        img = img.resize((img.width // 4, img.height // 4), Image.ANTIALIAS)
        # Save the scaled image in the subfolder
        img.save(f'scaled_down/{filename}')
