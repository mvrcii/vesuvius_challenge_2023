import os

import numpy as np

for frag_dir in os.listdir(os.path.join('data', 'datasets', '512')):
    nan_count = 0
    for file in os.listdir(os.path.join('data', 'datasets', '512', frag_dir, 'images')):
        arr = np.load(file)
        if np.isnan(arr).any():
            nan_count += 1
            print("NAN found in image:", file)

    print("NaN Count:", nan_count)
