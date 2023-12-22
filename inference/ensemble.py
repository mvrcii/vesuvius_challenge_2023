import os
import sys

import numpy as np

from config_handler import Config
from fragment import FragmentHandler
from util.combine_predictions import has_valid_ckpt_dirs


def load_and_process_data(directory, frag_id, architecture):
    start_layer, end_layer = FragmentHandler().get_best_12_layers(frag_id)  # inclusive
    layer_indices = set(range(start_layer, end_layer + 1))

    files = [file for file in os.listdir(directory) if file.startswith('sigmoid_logits') and
             file.endswith('.npy') and not file.startswith('maxed_logits')]
    files.sort(key=lambda x: int(x.split('_')[2]))

    data = []
    for filename in files:
        layer_start_idx = int(filename.split('_')[2])

        if layer_start_idx not in layer_indices:
            continue

        file_path = os.path.join(directory, filename)
        prediction = np.load(file_path)
        normalized_prediction = normalize_data(prediction)
        data.append(normalized_prediction)

    if architecture == 'segformer':
        return np.mean(data, axis=0)
    else:
        return data[0]


def identify_architecture(directory):
    if 'segformer' in directory:
        return 'segformer'
    elif 'unetr-sf' in directory:
        return 'unetr-sf'
    elif 'unet3d-sf' in directory:
        return 'unet3d-sf'
    else:
        raise ValueError(f"Unknown architecture for directory {directory}")


def process_ensemble_strategy(strategy, data):
    if strategy == 'mean':
        return np.mean(data, axis=0)
    else:
        raise Exception("Unknown ensemble strategy")


def normalize_data(data):
    # Normalize the data (scale to mean 0 and std 1)
    # Ensure the data is not a single value to avoid division by zero
    if data.size > 1:
        return (data - np.mean(data)) / np.std(data)
    return data


def extract_first_name(directory):
    return directory.split('_')[1].split('-')[0]


def ensemble_results(root_dir, frag_id, selected_directories, all_directories):
    processed_data = []
    first_names = []
    strategy = 'mean'

    for index in selected_directories:
        directory = all_directories[index]
        architecture = identify_architecture(directory)
        data = load_and_process_data(os.path.join(root_dir, directory), frag_id, architecture)
        processed_data.append(data)
        first_names.append(extract_first_name(directory))

    # Generate ensemble info ID
    ensemble_info_id = 'ensemble_' + '-'.join(first_names) + f"_{strategy}"

    ensemble_result = process_ensemble_strategy(strategy=strategy, data=processed_data)

    return ensemble_result, ensemble_info_id


def list_directories(root_dir):
    return [d for d in os.listdir(root_dir)
            if
            os.path.isdir(os.path.join(root_dir, d)) and not d.startswith("ensemble") and not d.startswith("snapshot")]


if __name__ == "__main__":
    config = Config().load_local_cfg()
    frag_infos = FragmentHandler().get_name_2_id()
    frag_root = os.path.join(config.work_dir, 'inference', 'results')

    # Filter valid fragments only
    valid_frags = [(name, frag_id) for (name, frag_id) in frag_infos if has_valid_ckpt_dirs(frag_root, frag_id)]

    print("Available Fragments:")
    for i, (name, frag_id) in enumerate(valid_frags, start=1):
        print(f"\033[92m{i:2}. {name:15} {frag_id}\033[0m")  # Display only valid fragments in green

    user_input = input("Fragment Number: ")

    # Check if input is a number and within the range of valid fragments
    if user_input.isdigit():
        user_number = int(user_input)
        if 1 <= user_number <= len(valid_frags):
            name, fragment_id = valid_frags[user_number - 1]
        else:
            print("Invalid selection. Please enter a valid number.")
            sys.exit(1)
    else:
        print("Invalid input. Please enter a number.")
        sys.exit(1)

    fragment_dir = os.path.join(config.work_dir, 'inference', 'results', f'fragment{fragment_id}')

    all_directories = list_directories(fragment_dir)

    # Display the directories for the user to choose from
    print("Select directories for ensemble:")
    for i, directory in enumerate(all_directories):
        print(f"{i}: {directory}")

    # User input
    selected_indices = input("Enter the numbers of the directories you want to ensemble, separated by commas: ")
    selected_indices = [int(x.strip()) for x in selected_indices.split(',')]

    # Ensure indices are within range
    selected_indices = [i for i in selected_indices if 0 <= i < len(all_directories)]

    print("Selected indices:", selected_indices)

    ensemble_result, ensemble_dir = ensemble_results(root_dir=fragment_dir, frag_id=fragment_id,
                                                     selected_directories=selected_indices,
                                                     all_directories=all_directories)

    # Save the result
    ensemble_dir = os.path.join(fragment_dir, f"ensemble_{ensemble_dir}")
    os.makedirs(ensemble_dir, exist_ok=True)
    np.save(os.path.join(ensemble_dir, f"sigmoid_logits_0_3.npy"), ensemble_result)
    print("Ensemble result saved in:", ensemble_dir)
