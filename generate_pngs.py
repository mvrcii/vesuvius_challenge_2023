import matplotlib.pyplot as plt
import numpy as np

prediction = np.load("20231113-092302result.npy")
plt.imshow(prediction, cmap='gray')
plt.savefig('logits_fragment1.png', bbox_inches='tight', dpi=500, pad_inches=0)
plt.imshow(prediction > 0.5, cmap='gray')
plt.savefig('binarized_fragment1.png', bbox_inches='tight', dpi=1500, pad_inches=0)
