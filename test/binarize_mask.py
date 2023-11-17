import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
Fixes mask to ensure it only has values 0-1 (can contain other values due
to soft brush painting)
'''
mask_img = cv2.imread('mask3300.png', 0)
mask_arr = np.asarray(mask_img)
binary_mask = np.where(mask_arr > 1, 1, 0)
print(np.unique(binary_mask))
plt.imshow(binary_mask)
plt.show()


