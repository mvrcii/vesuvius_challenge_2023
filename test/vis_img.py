import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

img = np.load("../data/512/train/labels/5120_7168_5632_7680.npy")
# img = resize(img, (512, 512), order=0, preserve_range=True, anti_aliasing=False)
print(img.shape)
# print(img.max())
# print(img.min())
# plt.imshow(img, cmap='gray')
# plt.show()