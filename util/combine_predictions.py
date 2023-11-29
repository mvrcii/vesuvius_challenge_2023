import os
import sys
from tkinter import Tk, Scale, HORIZONTAL, Label, Button, Frame, IntVar, Radiobutton

import cv2
import numpy as np
from PIL import Image, ImageTk
from PIL.Image import Resampling
from tqdm import tqdm

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


def get_sys_args():
    if len(sys.argv) != 2:
        print("Usage: python combine_predictions.py <fragment_id>")
        sys.exit(1)

    frag_id = sys.argv[1]
    config = Config().load_local_cfg()

    folder_path = os.path.join(config.work_dir, 'inference', 'results', f'fragment{frag_id}')

    if not os.path.exists(folder_path):
        print(f"No such folder: {folder_path}")
        sys.exit(1)

    sub_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    if not sub_dirs:
        print(f"No subdirectories found in {folder_path}")
        sys.exit(1)

    print("Available model predictions:")
    for idx, subdir in enumerate(sub_dirs):
        print(f"{idx}: {subdir}")

    selected_index = input("Enter the number of the subdirectory you want to use: ")

    try:
        selected_index = int(selected_index)
        if selected_index < 0 or selected_index >= len(sub_dirs):
            raise ValueError
    except ValueError:
        print("Invalid selection. Please enter a valid number.")
        sys.exit(1)

    selected_subdir = sub_dirs[selected_index]
    folder_path = os.path.join(folder_path, selected_subdir)

    return frag_id, config.work_dir, folder_path


def main():
    frag_id, work_dir, folder_path = get_sys_args()

    layer_idxs = None
    target_dims = (10479, 10360)  # TODO extract from original tif image dimension

    npy_preds_array = combine_npy_preds(idxs=layer_idxs, root_dir=folder_path, ignore_percent=0)

    Visualization(root_dir=folder_path, target_dims=target_dims, arrays=npy_preds_array)


class Visualization:
    def __init__(self, root_dir, target_dims, arrays):
        self.root_dir = root_dir
        self.target_dims = target_dims

        self.rotate_num = -1

        # Initially normalize and rotate combined predictions
        self.arrays = arrays
        self.array_max = np.maximum.reduce(self.arrays)
        self.array_sum = np.sum(self.arrays, axis=0)
        self.array = self.array_max  # initially set view to array_sum

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

        # Add left button for decreasing the slider value
        left_button = Button(control_frame, text="  -  ", command=self.decrease_slider)
        left_button.pack(side='left')

        # Add a slider
        self.slider = Scale(control_frame, from_=0, to=1, orient=HORIZONTAL, resolution=0.001, length=500,
                            command=self.update_image)
        self.slider.set(0.5)
        self.slider.pack(side='left')

        # Add right button for increasing the slider value
        right_button = Button(control_frame, text="  +  ", command=self.increase_slider)
        right_button.pack(side='left')

        # Display area for the image
        self.label = Label(self.root)
        self.label.pack()

        # Initially change mode
        # self.mode_changed()

        # Start the application
        self.root.mainloop()

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

    def save_snapshot(self):
        target_dir = os.path.join(self.root_dir, "snapshots")
        os.makedirs(target_dir, exist_ok=True)
        mode = self.modes[self.mode_var.get()]
        threshold = self.get_threshold()
        inverted_str = f'_inverted' if self.inverted else ""
        file_path = os.path.join(target_dir, f"{mode.lower()}_{threshold:.2f}{inverted_str}.png")

        image = self.process_image(array=self.array, max_size=self.target_dims)
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

    def mode_changed(self):
        mode = self.mode_var.get()
        print("Selected Mode", self.modes[mode])
        threshold = self.get_threshold()

        if mode in [0, 1]:
            if mode == 0:
                self.array = self.array_max
            else:  # mode == 1
                self.array = self.array_sum

            self.slider.config(resolution=0.001, from_=0, to=1)
            self.slider.set(threshold)

        elif mode == 2:
            self.slider.config(resolution=1, from_=0, to=len(self.arrays) - 1)
            self.slider.set(threshold)
            self.array = self.arrays[int(threshold)]

        self.update_image()

    def update_image(self, threshold=None):
        if threshold:
            self.set_threshold(threshold)

        if self.mode_var.get() == 2:
            layer = int(self.get_threshold())
            print("Selected layer", layer)
            self.array = self.arrays[layer]
            self.set_threshold(layer)

        new_img = self.process_image(array=self.array)
        imgtk = ImageTk.PhotoImage(image=new_img)
        self.label.imgtk = imgtk
        self.label.config(image=imgtk)

    def process_image(self, array, max_size=(1500, 800)):
        processed = array.copy()
        processed = self.normalize_npy_preds(processed)  # Normalize
        processed = self.rot90(processed, self.rotate_num)  # Rotate

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
            max_val = len(self.arrays) - 1
        current_value = self.slider.get()
        new_threshold = min(current_value + step, max_val)  # Ensure the value does not exceed the maximum

        self.set_threshold(new_threshold)
        self.slider.set(new_threshold)


def combine_npy_preds(root_dir, idxs=None, ignore_percent=0):
    npy_preds_array = []

    paths = [x for x in os.listdir(root_dir) if x.endswith('.npy') and not x.startswith('maxed_logits')]
    paths.sort(key=lambda x: int(x.split('_')[2]))

    if idxs:
        start, end = idxs
        paths = paths[start:end]

    # Load all numpy arrays from the directory
    for filename in tqdm(paths):
        if filename.endswith('.npy'):
            file_path = os.path.join(root_dir, filename)
            array = np.load(file_path)
            npy_preds_array.append(array)

    if ignore_percent > 0:
        print(f"Ignoring {ignore_percent * 100}% of top and bottom layers")
        print("Before:", len(npy_preds_array))
        ignore_count = int(ignore_percent * len(npy_preds_array))
        combined_npy_preds = npy_preds_array[ignore_count:-ignore_count]
        print("After:", len(combined_npy_preds))

    return npy_preds_array


if __name__ == "__main__":
    main()
