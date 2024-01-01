import glob
import os

import numpy as np
from PIL import Image

from utility.fragments import IRONHIDE_FRAG_ID, SUNSTREAKER_FRAG_ID, THUNDERCRACKER_FRAG_ID, ULTRA_MAGNUS_FRAG_ID, GRIMHUGE_FRAG_ID, JAZZBIGGER_FRAG_ID, HOT_ROD_FRAG_ID, BLASTER_FRAG_ID, JETFIRE_FRAG_ID, BLUEBIGGER_FRAG_ID, DEVASBIGGER_FRAG_ID, SKYGLORIOUS_FRAG_ID, TRAILBIGGER_FRAG_ID


def find_smallest_array_size(arrays):
    """ Finds the smallest array size among the given numpy arrays. """
    min_rows = min(arr.shape[0] for arr in arrays)
    min_cols = min(arr.shape[1] for arr in arrays)
    return min_rows, min_cols


def find_npy_files(directory):
    """
    Find all .npy files in the specified directory and its subdirectories.

    Parameters:
    directory (str): The path to the directory where the search should be performed.

    Returns:
    list: A list of paths to the .npy files found.
    """
    # Create the search pattern to find all .npy files
    search_pattern = os.path.join(directory, '**', '*.npy')

    # Use glob to find files matching the pattern
    npy_files = glob.glob(search_pattern, recursive=True)
    return npy_files


def crop_arrays(arrays, size):
    """ Crops the given arrays to the specified size. """
    return [arr[:size[0], :size[1]] for arr in arrays]


def weighted_ensemble(arrays_with_weights):
    """
    Calculate a weighted ensemble of 2D arrays.

    Parameters:
    arrays_with_weights (list of tuples): A list where each tuple contains a NumPy file path and a weight.
                                         Each NumPy file should contain a 2D array.

    Returns:
    np.ndarray: A 2D array representing the weighted ensemble of the input arrays.
    """
    weighted_sum = None
    total_weight = 0

    for array, weight in arrays_with_weights:
        if weighted_sum is None:
            weighted_sum = np.zeros_like(array)

        weighted_sum += array * weight
        total_weight += weight

    if total_weight > 0:
        # Normalize the array to have values between 0 and 1
        weighted_ensemble = weighted_sum / total_weight
        weighted_ensemble = np.clip(weighted_ensemble, 0, 1)
        return weighted_ensemble
    else:
        raise ValueError("Total weight must be greater than 0.")


def is_relevant(inf_dir, checkpoint_names):
    for c in checkpoint_names:
        if c in inf_dir:
            return c
    return "invalid"


# Goes over all importants fragments (13)
relevant_fragments = [IRONHIDE_FRAG_ID]
checkpoint_names = ["desert-sea-1183", "efficient-aardvark"]
checkpoint_weights = {"desert-sea-1183": 0.5, "efficient-aardvark": 0.5}
out_dir = os.path.join("data", "ensemble_results")

inf_dir = os.path.join("inference", "results")
for frag_id in relevant_fragments:
    # list holding all npy files to find smallest common size
    npy_files = []

    # list holding model npy files to allow ensembling them together first
    model_files = {}
    for c in checkpoint_names:
        model_files[c] = []

    fragment_dir = os.path.join(inf_dir, f"fragment{frag_id}")
    for inf_dir in os.listdir(fragment_dir):  # e.g. 20231227-171128_fine-donkey-1133-unetr.sf
        c_name = is_relevant(inf_dir, checkpoint_names)
        if c_name == "invalid":
            continue
        inf_dir_path = os.path.join(fragment_dir, inf_dir)
        for file in os.listdir(inf_dir_path):
            if file.endswith(".npy"):
                file_path = os.path.join(inf_dir_path, file)
                arr = np.load(file_path)
                print(arr.shape)
                npy_files.append(arr)
                model_files[c_name].append(arr)
    smallest_size = find_smallest_array_size(npy_files)

    # Crop all arrays to the smallest size
    cropped_arrays = crop_arrays(npy_files, smallest_size)
    for c in model_files.keys():
        # replace list of npy files with mean
        model_files[c] = np.mean(np.stack(model_files[c]), axis=0)
    print(len(cropped_arrays))

    ensemble_arr = []
    for c in model_files.keys():
        ensemble_arr.append((model_files[c], checkpoint_weights[c]))
    ensemble = weighted_ensemble(ensemble_arr)
    ensemble = (255 - (ensemble * 255)).astype(np.uint8)

    # Ensure the directory exists
    os.makedirs(out_dir, exist_ok=True)

    # todo remove
    ### debug
    weight_string = ""
    for c in checkpoint_weights.keys():
        weight_string += c[:5] + str(checkpoint_weights[c])
    ### debug

    # Save the images
    out_path = os.path.join(out_dir, f'{weight_string}_fragment{frag_id}_ensemble.png')
    Image.fromarray(ensemble).save(os.path.join(out_path))
    print("Saving to ", out_path)
