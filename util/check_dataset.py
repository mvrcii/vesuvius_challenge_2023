import os

import numpy as np

for frag_dir in os.listdir(os.path.join('data', 'datasets', '512')):
    nan_count = 0
    unique_error_count = 0

    if os.path.isdir(os.path.join('data', 'datasets', '512', frag_dir)):
        for file in os.listdir(os.path.join('data', 'datasets', '512', frag_dir, 'images')):
            arr = np.load(os.path.join('data', 'datasets', '512', frag_dir, 'images', file))
            if np.isnan(arr).any():
                nan_count += 1
                print("NAN found in image:", file)

        for file in os.listdir(os.path.join('data', 'datasets', '512', frag_dir, 'labels')):
            arr = np.load(os.path.join('data', 'datasets', '512', frag_dir, 'labels', file))
            if np.isnan(arr).any():
                nan_count += 1
                print("NAN found in labels:", file)
            if len(np.unique(arr)) > 2:
                unique_error_count += 1

        print("NaN Count:", nan_count)
        print("Unique Error Count:", unique_error_count)

