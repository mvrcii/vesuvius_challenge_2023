import cv2

import numpy as np
import matplotlib.pyplot as plt

file = np.load("frag_20231024093300_20231118-022252result.npy")
plt.imshow(file, cmap='gray')
