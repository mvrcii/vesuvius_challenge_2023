import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# List all files in the current directory
files = os.listdir('.')

# Filter out the .tif files that are numbered from 00000.tif to 00063.tif
tif_files = [f for f in files if f.endswith('.tif') and '00000.tif' <= f <= '00063.tif']

# Sort the files to maintain the order
tif_files.sort()

# Read each image and save it in an array of numpy arrays
images = []
for file in tqdm(tif_files):
    img = cv2.imread(file, 0)
    img_array = np.asarray(img)
    images.append(img_array)


# Display the number of images loaded and the shape of the first image (as a sample)
slices = []
y = 9970
x = 1815
w = 680
h = 1
for image in images:
    s = image[y: y+h, x: x+w]
    slices.append(s)

x = np.stack(slices, axis=0)
x = np.squeeze(x)
print(x.shape)
plt.imshow(x, cmap='gray')
plt.show()


