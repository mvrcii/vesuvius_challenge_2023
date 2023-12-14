import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


path_imgs = "datasets/512/GRIMLARGE/images"
path_labels = "datasets/512/GRIMLARGE/labels"

for img_name in os.listdir(path_imgs):
    img = np.load(os.path.join(path_imgs, img_name))
    label = np.load(os.path.join(path_labels, img_name))
    label = np.unpackbits(label).reshape((128, 128))

    # Scale label up to patch shape
    label = resize(label, (512, 512), order=0, preserve_range=True, anti_aliasing=False)
    # show both of them next to each other using pyplot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img[0], cmap="gray")
    ax2.imshow(label, cmap="gray")
    plt.show()


