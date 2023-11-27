import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ink_files = [file for file in os.listdir("datasets/slice/train/images") if file.endswith("_100.npy")]
no_ink_files = [file for file in os.listdir("datasets/slice/train/images") if file.endswith("_0.npy")]

means_ink = np.zeros(64)
counts_ink = 0
means_no_ink = np.zeros(64)
counts_no_ink = 0

for file in tqdm(ink_files):
    arr = np.load(os.path.join("train/images", file))
    arr = arr / 255
    mean = np.mean(arr, axis=(1, 2))
    means_ink += mean
    counts_ink += 1
means_ink = means_ink / counts_ink

for file in tqdm(no_ink_files):
    arr = np.load(os.path.join("train/images", file))
    arr = arr / 255
    mean = np.mean(arr, axis=(1, 2))
    means_no_ink += mean
    counts_no_ink += 1
means_no_ink = means_no_ink / counts_no_ink

print(f"Ink mean: {np.mean(means_ink)}")
print(f"No-Ink mean: {np.mean(means_no_ink)}")

print(f"Ink mean 10-29: {np.mean(means_ink[10:29])}")
print(f"No-Ink mean 10-29: {np.mean(means_no_ink[10:29])}")

print(f"Ink mean 30-46: {np.mean(means_ink[30:46])}")
print(f"No-Ink mean 30-46: {np.mean(means_no_ink[30:46])}")

array1 = means_ink
array2 = means_no_ink
#
# ink_mean = np.mean(array1)
# no_ink_mean = np.mean(array2)
# avg = (ink_mean + no_ink_mean) / 2
#
# array1 -= avg
# array2 -= avg

# Creating the plot
x = np.arange(len(array1))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, array1, width, label='Ink', color='green')
rects2 = ax.bar(x + width/2, array2, width, label='No ink', color='blue')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Means value')
ax.set_title('Comparing ink to no-ink')
ax.set_xticks(x)
ax.set_xticklabels([f'{i+1}' for i in x])
ax.legend()

plt.show()