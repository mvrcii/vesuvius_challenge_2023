import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


path_imgs = "datasets/256/GRIMLARGE/images"
path_labels = "datasets/256/GRIMLARGE/labels"

for img_name in os.listdir(path_imgs):
    img = np.load(os.path.join(path_imgs, img_name))
    label = np.load(os.path.join(path_labels, img_name))
    label = np.unpackbits(label).reshape((2, 64, 64))

    # Scale label up to patch shape
    label = resize(label, (2, 128, 128), order=0, preserve_range=True, anti_aliasing=False)
    # show both of them next to each other using pyplot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img[0], cmap="gray")
    ax2.imshow(label[0], cmap="gray")
    ax3.imshow(label[1], cmap="gray")
    plt.show()


