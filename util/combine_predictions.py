import datetime
import os
import sys
from tkinter import Tk, Scale, HORIZONTAL, Label, Button, Frame, IntVar, Radiobutton

import cv2
import numpy as np
from PIL import Image, ImageTk
from PIL.Image import Resampling
from tqdm import tqdm

import constants
from constants import get_all_frag_infos

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_handler import Config


def combine_layers(predictions, max_distance):
    """
    Combine predictions from multiple layers with a weighting that decreases
    with the distance between layers.

    :param predictions: List of 2D numpy arrays, each representing a layer.
    :param max_distance: Maximum layer distance to consider for weighting.
    :return: Combined 2D numpy array.
    """
    num_layers = len(predictions)
    layer_shape = predictions[0].shape
    combined = np.zeros(layer_shape)

    # Create weight matrix
    weight_matrix = np.zeros((num_layers, num_layers))
    for i in tqdm(range(num_layers)):
        for j in range(num_layers):
            distance = abs(i - j)
            weight_matrix[i, j] = max(0, 1 - distance / max_distance)

    # Apply weights and combine
    for i in tqdm(range(num_layers)):
        for j in range(num_layers):
            weighted_prediction = predictions[j] * weight_matrix[i, j]
            combined += weighted_prediction

    # Normalize the combined array
    combined /= np.max(combined)

    return combined


def get_common_layers(folder_paths):
    common_layers = set()

    for i, folder_path in enumerate(folder_paths):
        paths = [x for x in os.listdir(folder_path) if x.endswith('.npy') and not x.startswith('maxed_logits')]
        layers = set(int(x.split('_')[2]) for x in paths)

        if i == 0:
            common_layers = layers
        else:
            common_layers = common_layers.intersection(layers)

    return common_layers


def get_valid_ckpt_dirs(folder_path):
    def is_valid(path):
        return os.path.isdir(os.path.join(folder_path, path)) and not path.__contains__('snapshots')

    if not os.path.isdir(folder_path):
        return None

    return [_dir for _dir in os.listdir(folder_path) if is_valid(path=_dir)]


def get_selected_ckpt_dirs(frag_dir):
    if not os.path.exists(frag_dir):
        print(f"No such directory: {frag_dir}")
        sys.exit(1)

    sub_dirs = get_valid_ckpt_dirs(frag_dir)
    if not sub_dirs:
        print(f"No subdirectories found in {frag_dir}")
        sys.exit(1)

    if len(sub_dirs) == 1:
        selected_subdirs = [sub_dirs[0]]
    else:
        print("\nAvailable model predictions:")
        for idx, subdir in enumerate(sub_dirs):
            print(f"{idx}: {subdir}")
        selected_indices = input("Enter numbers of subdirectories you want to use (comma-separated): ")
        try:
            selected_indices = [int(idx.strip()) for idx in selected_indices.split(',')]
            selected_subdirs = [sub_dirs[idx] for idx in selected_indices if 0 <= idx < len(sub_dirs)]
            if not selected_subdirs:
                raise ValueError
        except ValueError:
            print("Invalid selection. Please enter valid numbers.")
            sys.exit(1)

    return [os.path.join(frag_dir, subdir) for subdir in selected_subdirs]


def has_valid_ckpt_dirs(work_dir, frag_id):
    frag_dir = os.path.join(work_dir, 'inference', 'results', f'fragment{frag_id}')
    return bool(get_valid_ckpt_dirs(frag_dir))


def get_target_dims(work_dir, frag_id):
    frag_dir = os.path.join(work_dir, "data", "fragments", f"fragment{frag_id}")
    assert os.path.isdir(frag_dir)

    target_dims = None

    slice_dir = os.path.join(frag_dir, "slices")
    if os.path.isdir(slice_dir):
        for i in range(0, 63):
            if target_dims:
                return target_dims

            img_path = os.path.join(slice_dir, f"{i:05}.tif")
            if os.path.isfile(img_path):
                image = cv2.imread(img_path, 0)
                target_dims = image.shape

    return target_dims


def main():
    config = Config().load_local_cfg()
    frag_infos = get_all_frag_infos()

    # Filter valid fragments only
    valid_frags = [(name, frag_id) for (name, frag_id) in frag_infos if has_valid_ckpt_dirs(config.work_dir, frag_id)]

    print("Available Fragments:")
    for i, (name, frag_id) in enumerate(valid_frags, start=1):
        print(f"\033[92m{i:2}. {name:15} {frag_id}\033[0m")  # Display only valid fragments in green

    user_input = input("Fragment Number: ")

    # Check if input is a number and within the range of valid fragments
    if user_input.isdigit():
        user_number = int(user_input)
        if 1 <= user_number <= len(valid_frags):
            name, frag_id = valid_frags[user_number - 1]
        else:
            print("Invalid selection. Please enter a valid number.")
            sys.exit(1)
    else:
        print("Invalid input. Please enter a number.")
        sys.exit(1)

    frag_dir = os.path.join(config.work_dir, 'inference', 'results', f'fragment{frag_id}')
    sub_dirs = get_selected_ckpt_dirs(frag_dir=frag_dir)

    common_layers = get_common_layers(sub_dirs)
    target_dims = get_target_dims(work_dir=config.work_dir, frag_id=frag_id)

    model_arrays = {}
    for sub_dir in sub_dirs:
        arrays = combine_npy_preds(root_dir=sub_dir, layer_indices=common_layers, ignore_percent=0)
        model_arrays[sub_dir] = arrays

    Visualization(frag_id=frag_id, root_dir=frag_dir, target_dims=target_dims, model_arrays=model_arrays)


class Visualization:
    def __init__(self, frag_id, root_dir, target_dims, model_arrays: dict):
        assert isinstance(target_dims, tuple) and len(target_dims) == 2, "target_dims must be a tuple of two elements"
        assert isinstance(model_arrays, dict) and model_arrays, "model_arrays must be a non-empty dictionary"

        self.root_dir = root_dir
        self.target_dims = target_dims

        self.rotate_num = 0
        rot_value = constants.get_rotate_value(frag_id=frag_id)
        if rot_value:
            self.rotate_num = rot_value

        self.model_names = list(model_arrays.keys())
        self.models = list(model_arrays.values())
        self.model_count = len(self.models)

        assert all(isinstance(model, list) and model for model in
                   self.models), "Each value in model_arrays must be a non-empty list"
        assert all(len(model[0].shape) == 2 for model in self.models), "Each numpy array must be two-dimensional"

        self.array_maxs = []
        self.array_sums = []
        self.curr_weight_th_val = 0.5

        # Initially calculate max and sum once
        for model in self.models:
            self.array_maxs.append(np.maximum.reduce(model))
            self.array_sums.append(np.sum(model, axis=0))

        if self.model_count == 1:
            self.array = self.array_maxs[0]
        else:
            self.array = self.calc_weighted_arr(array=self.array_maxs, weight=self.curr_weight_th_val)

        # Slider idxs
        self.curr_layer_val = 0
        self.curr_th_val = 0.5

        # Variable to hold the selected mode
        self.inverted = False

        # Create main window
        self.root = Tk()
        self.root.title("Threshold Visualizer")
        self.mode_var = IntVar(value=0)  # Default to mode 0

        # Utility frame
        utility_frame = Frame(self.root)
        utility_frame.pack(side='top')
        save_button = Button(utility_frame, text="Save Snapshot", command=lambda: self.save_snapshot())
        save_button.pack(side="left")
        invert_button = Button(utility_frame, text="Invert colors", command=lambda: self.invert_colors())
        invert_button.pack(side="left")

        # Create a frame for the mode selection buttons
        mode_frame = Frame(self.root)
        mode_frame.pack(side='top')

        # List of modes
        self.modes = ["Sum", "Max", "Layers"]

        # Create a Radiobutton for each mode
        for index, mode in enumerate(self.modes):
            Radiobutton(mode_frame, text=mode, variable=self.mode_var, value=index, command=self.mode_changed).pack(
                side='left')

        # Create a frame for the slider and buttons
        control_frame = Frame(self.root)
        control_frame.pack(side='top')
        left_button = Button(control_frame, text="  -  ", command=self.decrease_slider)
        left_button.pack(side='left')
        self.slider = Scale(control_frame, from_=0, to=1, orient=HORIZONTAL, resolution=0.001, length=500,
                            command=self.update_image)
        self.slider.set(0.5)
        self.slider.pack(side='left')
        right_button = Button(control_frame, text="  +  ", command=self.increase_slider)
        right_button.pack(side='left')

        # Create another control frame for weighting
        if self.model_count > 1:
            control_frame_weighting = Frame(self.root)
            control_frame_weighting.pack(side='top')
            left_button_weighting = Button(control_frame_weighting, text="  --  ",
                                           command=self.decrease_slider_weighting)
            left_button_weighting.pack(side='left')
            self.slider_weighting = Scale(control_frame_weighting, from_=0, to=1, orient=HORIZONTAL, resolution=0.1,
                                          length=500,
                                          command=self.update_weighting)
            self.slider_weighting.set(self.curr_weight_th_val)
            self.slider_weighting.pack(side='left')
            right_button_weighting = Button(control_frame_weighting, text="  ++  ",
                                            command=self.increase_slider_weighting)
            right_button_weighting.pack(side='left')

        # Display area for the image
        self.label = Label(self.root)
        self.label.pack()

        # Start the application
        self.root.mainloop()

    @staticmethod
    def calc_weighted_arr(array, weight):
        array_cp = array.copy()
        assert len(array) == 2
        return array_cp[0] * weight + array_cp[1] * (1 - weight)

    def get_threshold(self):
        mode = self.mode_var.get()

        if mode == 2:
            threshold = self.curr_layer_val
        else:
            threshold = self.curr_th_val

        return threshold

    def set_threshold(self, threshold):
        mode = self.mode_var.get()

        if mode == 2:
            self.curr_layer_val = threshold
        else:
            self.curr_th_val = threshold

        return threshold

    def create_ensemble_dir_simple_names(self, dir_list, prefix):
        # Extracting only the first part of the name after '_' (ignoring the numbers and model names)
        parts = [d.split('_')[1].split('-')[0:2] for d in dir_list]
        # Joining the first two parts (words) of each directory name
        simplified_parts = ["-".join(p) for p in parts]

        # Current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Creating the new directory name
        new_dir_name = prefix + "_".join(simplified_parts) + "_" + timestamp
        return new_dir_name

    def save_snapshot(self):
        # Save in fragment root dir for ensemble
        target_dir = os.path.join(self.root_dir, "snapshots")
        file_prefix = 'ensemble_'

        # Save in model dir within fragment for single model
        if self.model_count == 1:
            assert len(self.model_names) == 1
            model_dir = self.model_names[0]
            target_dir = os.path.join(self.root_dir, model_dir, "snapshots")
            file_prefix = ''

        os.makedirs(target_dir, exist_ok=True)

        model_names_str = self.create_ensemble_dir_simple_names(self.model_names, file_prefix)

        mode = self.modes[self.mode_var.get()]
        threshold = self.get_threshold()
        inverted_str = f'inverted' if self.inverted else ""

        file_path = os.path.join(target_dir, f"{model_names_str}_{mode.lower()}_{threshold}_{inverted_str}.png")

        image = self.process_image(array=self.array, max_size=self.target_dims)
        #
        # if image.mode != 'RGBA':
        #     image = image.convert('RGBA')
        #
        # data = np.array(image)  # Convert image to numpy array
        # data[:, :, 3] = np.where(data[:, :, 0:3].sum(axis=2) == 0, 0, 255)  # Set alpha to 0 where RGB sums to 0, else 255
        # image = Image.fromarray(data)

        image.save(file_path)

    def invert_colors(self):
        self.inverted = not self.inverted
        self.update_image()

    @staticmethod
    def normalize_npy_preds(array):
        min_value = array.min()
        max_value = array.max()
        normalized_array = (array - min_value) / (max_value - min_value)

        return normalized_array

    @staticmethod
    def rot90(array, count):
        return np.rot90(array, count)

    def get_layer_weighted_arr(self, layer, weight=None):
        if not weight:
            weight = self.curr_weight_th_val
        if self.model_count == 2:
            return self.calc_weighted_arr(array=[model[layer] for model in self.models], weight=weight)
        else:
            return self.models[0][layer]

    def get_array_max(self, weight=None):
        if not weight:
            weight = self.curr_weight_th_val
        if self.model_count == 2:
            return self.calc_weighted_arr(array=self.array_maxs, weight=weight)
        else:
            return self.array_maxs[0]

    def get_array_sum(self, weight=None):
        if not weight:
            weight = self.curr_weight_th_val
        if self.model_count == 2:
            return self.calc_weighted_arr(array=self.array_sums, weight=weight)
        else:
            return self.array_sums[0]

    def mode_changed(self):
        mode = self.mode_var.get()
        print("Selected Mode", self.modes[mode])
        threshold = self.get_threshold()

        if mode in [0, 1]:
            if mode == 0:
                self.array = self.get_array_max()
            else:  # mode == 1
                self.array = self.get_array_sum()

            self.slider.config(resolution=0.001, from_=0, to=1)
            self.slider.set(threshold)

        elif mode == 2:
            self.slider.config(resolution=1, from_=0, to=len(self.models[0]) - 1)
            self.slider.set(threshold)
            self.array = self.get_layer_weighted_arr(int(threshold))

        self.update_image()

    def update_weighting(self, weight=None):
        if self.mode_var.get() == 0:
            self.array = self.get_array_max(weight=float(weight))
        elif self.mode_var.get() == 1:
            self.array = self.get_array_sum(weight=float(weight))
        else:
            self.array = self.get_layer_weighted_arr(layer=int(self.get_threshold()), weight=float(weight))

        new_img = self.process_image(array=self.array)
        imgtk = ImageTk.PhotoImage(image=new_img)
        self.label.imgtk = imgtk
        self.label.config(image=imgtk)

    def update_image(self, threshold=None):
        if threshold:
            self.set_threshold(threshold)

        if self.mode_var.get() == 2:
            layer = int(self.get_threshold())
            print("Selected layer", layer)
            self.array = self.get_layer_weighted_arr(layer=layer)
            self.set_threshold(layer)

        new_img = self.process_image(array=self.array)
        imgtk = ImageTk.PhotoImage(image=new_img)
        self.label.imgtk = imgtk
        self.label.config(image=imgtk)

    def process_image(self, array, max_size=(1500, 800)):
        processed = array.copy()
        processed = self.normalize_npy_preds(processed)  # Normalize
        processed = self.rot90(processed, self.rotate_num)  # Rotate
        # processed = np.flip(processed, 1)

        threshold = self.get_threshold()
        if self.mode_var.get() == 2:
            threshold = 0

        # Apply threshold
        processed[processed < float(threshold)] = 0
        if self.inverted:
            processed = 1 - processed

        image = Image.fromarray(np.uint8(processed * 255), 'L')
        aspect_ratio = image.width / image.height

        # Determine new dimensions
        if aspect_ratio > 1:
            new_width = min(max_size[0], image.width * 4)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_size[1], image.height * 4)
            new_width = int(new_height * aspect_ratio)

        # Resize with high-quality resampling
        return image.resize((new_width, new_height), Resampling.LANCZOS)

    def decrease_slider(self):
        mode = self.mode_var.get()
        step = 0.001

        if mode == 2:
            step = 1  # step size = 1 for layers
        current_value = self.slider.get()
        new_threshold = max(current_value - step, 0)  # Ensure the value does not go below the minimum

        self.set_threshold(new_threshold)
        self.slider.set(new_threshold)

    def increase_slider(self):
        mode = self.mode_var.get()
        step = 0.001
        max_val = 1

        if mode == 2:
            step = 1
            max_val = len(self.models[0]) - 1
        current_value = self.slider.get()
        new_threshold = min(current_value + step, max_val)  # Ensure the value does not exceed the maximum

        self.set_threshold(new_threshold)
        self.slider.set(new_threshold)

    def decrease_slider_weighting(self):
        step = 0.1
        current_value = self.slider_weighting.get()
        new_threshold = max(current_value - step, 0)
        self.curr_weight_th_val = new_threshold
        self.slider_weighting.set(new_threshold)

    def increase_slider_weighting(self):
        step = 0.1
        max_val = 1
        current_value = self.slider_weighting.get()
        new_threshold = min(current_value + step, max_val)  # Ensure the value does not exceed the maximum
        self.curr_weight_th_val = new_threshold
        self.slider_weighting.set(new_threshold)


def combine_npy_preds(root_dir, layer_indices=None, ignore_percent=0):
    npy_preds_array = []

    paths = [x for x in os.listdir(root_dir) if x.endswith('.npy') and not x.startswith('maxed_logits')]
    paths.sort(key=lambda x: int(x.split('_')[2]))

    if layer_indices is not None:
        paths = [path for path in paths if int(path.split('_')[2]) in layer_indices]

    # Load all numpy arrays from the directory
    for filename in paths:
        file_path = os.path.join(root_dir, filename)
        array = np.load(file_path)
        npy_preds_array.append(array)

    if ignore_percent > 0:
        ignore_count = int(ignore_percent * len(npy_preds_array))
        npy_preds_array = npy_preds_array[ignore_count:-ignore_count]

    return npy_preds_array


if __name__ == "__main__":
    main()
