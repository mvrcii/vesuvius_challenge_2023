import argparse
import datetime
import os
import subprocess
import sys
from tkinter import Tk, Scale, HORIZONTAL, Label, Button, Frame, IntVar, Radiobutton, Entry, StringVar

import cv2
import numpy as np
from PIL import Image, ImageTk
from PIL.Image import Resampling
from skimage.transform import resize
from tqdm import tqdm

from utility.configs import Config
from utility.fragments import FragmentHandler, SUPERSEDED_FRAGMENTS, FRAGMENTS_IGNORE


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


def get_start_layer_idx(filename, single_layer):
    if single_layer:
        return int(filename.split('_')[2].split('.')[0])
    else:
        return int(filename.split('_')[2])


def get_common_layers(model_dir, selected_layers, single_layer):
    paths = [x for x in os.listdir(model_dir) if x.endswith('.npy') and not x.startswith('maxed_logits')]

    layers = []
    for filename in paths:
        layers.append(get_start_layer_idx(filename, single_layer))

    layers = set(layers).intersection(selected_layers)

    if len(layers) == 0:
        print("No intersecting layers found for the given range.")
        sys.exit(1)

    return layers


def get_valid_model_dirs(folder_path):
    def is_valid(path):
        return os.path.isdir(os.path.join(folder_path, path)) and not path.__contains__('snapshots')

    if not os.path.isdir(folder_path):
        return None

    return [_dir for _dir in os.listdir(folder_path) if is_valid(path=_dir)]


def get_selected_model_name(frag_dir):
    if not os.path.exists(frag_dir):
        print(f"No such directory: {frag_dir}")
        sys.exit(1)

    model_dirs = get_valid_model_dirs(frag_dir)
    if not model_dirs:
        print(f"No subdirectories found in {frag_dir}")
        sys.exit(1)

    if len(model_dirs) == 1:
        selected_subdir = model_dirs[0]
    else:
        print("\nAvailable model predictions:")
        for idx, subdir in enumerate(model_dirs):
            print(f"{idx}: {subdir}")
        while True:
            selected_index = input("Enter the number of the subdirectory you want to use: ")
            try:
                selected_index = int(selected_index.strip())
                if 0 <= selected_index < len(model_dirs):
                    selected_subdir = model_dirs[selected_index]
                    break
                else:
                    print("Invalid selection. Please enter a valid number.")
            except ValueError:
                print("Invalid selection. Please enter a valid number.")

    return os.path.join(frag_dir, selected_subdir), selected_subdir


def has_valid_ckpt_dirs(frag_root, frag_id):
    frag_dir = os.path.join(frag_root, f'fragment{frag_id}')
    return bool(get_valid_model_dirs(frag_dir))


def get_target_dims(work_dir, frag_id):
    frag_dir = os.path.join(work_dir, "fragments", f"fragment{frag_id}")

    target_dims = None

    slice_dir = os.path.join(frag_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)

    if os.path.isdir(slice_dir):
        for i in range(0, 63):
            if target_dims:
                return target_dims

            img_path = os.path.join(slice_dir, f"{i:05}.tif")
            if not os.path.isfile(img_path):
                continue

            image = cv2.imread(img_path, 0)
            target_dims = image.shape

    if target_dims is None:
        if frag_id in SUPERSEDED_FRAGMENTS:
            print("Warning: Fragment superseded, added suffix for download!")
            frag_id += "_superseded"
        print(f"Downloading Slice file for dimensions: {os.path.join(frag_id, 'slices', f'{i:05}.tif')}")
        command = ['bash', "./scripts/utils/download_fragment.sh", frag_id, f'{i:05} {i:05}']
        subprocess.run(command, check=True)
        image = cv2.imread(img_path, 0)
        target_dims = image.shape
    assert target_dims, "Target dimensions are none!"

    return target_dims


def parse_args():
    parser = argparse.ArgumentParser(description='Batch Infer Layered Script')
    parser.add_argument('--save_all_layers', action='store_true', help='Save all layer files (in total 61)')
    parser.add_argument('--max_ensemble', action='store_true', help='Show the max ensemble between two models')
    parser.add_argument('--submission', action='store_true', help='Formats file names in submission mode')
    parser.add_argument('--single_layer', action='store_true', help='Combine predictions for single layer checkpoints')

    args = parser.parse_args()

    return args


def get_selected_layer_range(start_idx, end_idx):
    default_start = 0
    default_end = 62

    # Adjust end_idx if it exceeds the default_end
    if end_idx > default_end:
        end_idx = default_end

    if start_idx < default_start:
        start_idx = default_start

    # Return the range between start_idx and end_idx
    return list(range(start_idx, end_idx + 1))


def main():
    args = parse_args()

    config = Config().load_local_cfg()

    single_layer = args.single_layer
    results_dir = 'single_results' if single_layer else 'results'
    frag_root = os.path.join(config.work_dir, 'inference', results_dir)

    all_frag_dirs = [frag_id.split('fragment')[1] for frag_id in os.listdir(frag_root)]

    # Filter valid fragments only
    valid_frags = []
    for frag_id in all_frag_dirs:
        if frag_id in SUPERSEDED_FRAGMENTS or frag_id in FRAGMENTS_IGNORE:
            continue
        if not has_valid_ckpt_dirs(frag_root, frag_id):
            continue
        frag_name = FragmentHandler().get_name(frag_id=frag_id)
        valid_frags.append((frag_name, frag_id))

    print("Available Fragments:")
    for i, (name, frag_id) in enumerate(valid_frags, start=1):
        print(f"\033[92m{i:2}. {name:20} {frag_id}\033[0m")  # Display only valid fragments in green

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

    frag_dir = os.path.join(frag_root, f'fragment{frag_id}')
    model_dir, model_name = get_selected_model_name(frag_dir=frag_dir)

    global sparent, save_all_layers, max_ensemble, submission_mode
    save_all_layers = args.save_all_layers
    max_ensemble = args.max_ensemble
    submission_mode = args.submission

    target_dims = get_target_dims(work_dir=config.work_dir, frag_id=frag_id)

    Visualization(frag_id=frag_id, root_dir=frag_dir, target_dims=target_dims, model_name=model_name,
                  model_dir=model_dir, single_layer=single_layer)


class Visualization:
    def __init__(self, frag_id, root_dir, target_dims, model_name, model_dir, single_layer):
        assert isinstance(target_dims, tuple) and len(target_dims) == 2, "target_dims must be a tuple of two elements"

        self.root_dir = root_dir
        self.target_dims = target_dims
        self.frag_id = frag_id

        frag_handler = FragmentHandler()
        self.rotate_num = frag_handler.get_rotation(frag_id=frag_id)
        self.flip_num = frag_handler.get_flip(frag_id=frag_id)
        self.center_layers = frag_handler.get_best_layers(frag_id=frag_id)

        # start layer idx = start index for a label/numpy file
        # start and end layer idx are both inclusive!
        start_layer_idx, end_layer_idx = 0, 60
        selected_layers = list(range(start_layer_idx, end_layer_idx + 1))
        valid_layers = get_common_layers(model_dir=model_dir, selected_layers=selected_layers,
                                         single_layer=single_layer)

        self.model_layer_idcs, self.model_layer_values, self.file_names = load_predictions(root_dir=model_dir,
                                                                                           layer_indices=valid_layers,
                                                                                           single_layer=single_layer)
        start_layer_idx, end_layer_idx = FragmentHandler().get_best_layers(frag_id=frag_id)

        self.model_name = model_name
        self.model_dir = model_dir

        # Slider idxs
        self.curr_layer_val = 0
        self.curr_th_val = 0.5

        self.curr_weight_th_val = 0.5

        # Variable to hold the selected mode
        self.inverted = False
        self.min_max_mode = 'max'

        self.max_ensemble = max_ensemble
        self.save_all_layers = save_all_layers
        self.submission_mode = submission_mode

        # Create main window
        self.root = Tk()
        self.root.title("Threshold Visualizer")

        self.mode_var = IntVar(value=1)  # Default to mode 0
        self.start_layer_var = IntVar(value=start_layer_idx)  # inclusive
        self.end_layer_var = IntVar(value=end_layer_idx)  # inclusive

        # threshold = FragmentHandler().get_boost_threshold(frag_id=frag_id)  # Deactivate for now
        threshold = 0.01
        self.threshold_var = StringVar(value=str(threshold))

        # Utility frame
        utility_frame = Frame(self.root)
        utility_frame.pack(side='top')
        save_button = Button(utility_frame, text="Save Snapshot", command=lambda: self.save_snapshot())
        save_button.pack(side="left")
        invert_button = Button(utility_frame, text="Invert colors", command=lambda: self.invert_colors())
        invert_button.pack(side="left")
        save_transparent_button = Button(utility_frame, text="Save Transparent", command=lambda: self.save_snapshot(transparent=True))
        save_transparent_button.pack(side="left")

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

        self.threshold_layer = Label(self.root, text="Threshold:")
        self.threshold_layer.pack()
        self.threshold_input = Entry(self.root, textvariable=self.threshold_var)
        self.threshold_input.bind("<Return>", self.on_threshold_input_enter)
        self.threshold_input.pack()

        self.left_button = Button(control_frame, text="  -  ", command=self.decrease_slider)
        self.slider = Scale(control_frame, from_=0, to=len(self.model_layer_idcs) - 1, orient=HORIZONTAL, resolution=1,
                            length=500, command=self.update_image)
        self.slider.set(self.curr_layer_val)
        self.layer_label = Label(control_frame, text="Current Layer:")
        print(self.model_layer_idcs)
        self.layer_label.config(text=f"{self.model_layer_idcs[self.curr_layer_val]}")
        self.right_button = Button(control_frame, text="  +  ", command=self.increase_slider)

        self.start_layer_input = Entry(control_frame, textvariable=self.start_layer_var)
        self.start_layer_input.bind("<Return>", self.on_start_layer_input_enter)
        self.start_layer_input.pack(side='left')
        self.start_layer_label = Label(control_frame, text="Start Layer:")
        self.start_layer_label.pack(side='left')

        self.end_layer_input = Entry(control_frame, textvariable=self.end_layer_var)
        self.end_layer_input.bind("<Return>", self.on_end_layer_input_enter)
        self.end_layer_input.pack(side='left')
        self.end_layer_label = Label(control_frame, text="End Layer:")
        self.end_layer_label.pack(side='left')

        self.clear_focus_button = Button(self.root, text="Toggle Min-Max", command=self.toggle_min_max)
        self.clear_focus_button.pack()
        # self.root.bind("<KeyPress>", self.on_key_press)

        # Added: Slider for start and end layer index
        # self.layer_range_frame = Frame(self.root)
        # self.layer_range_frame.pack(side='top')
        #
        # Label(self.layer_range_frame, text="Start Layer:").pack(side='left')
        # Scale(self.layer_range_frame, from_=0, to=len(self.model_layer_idcs) - 1, orient=HORIZONTAL,
        #       variable=self.start_layer_var, command=self.update_layer_range).pack(side='left')
        #
        # Label(self.layer_range_frame, text="End Layer:").pack(side='left')
        # Scale(self.layer_range_frame, from_=0, to=len(self.model_layer_idcs) - 1, orient=HORIZONTAL,
        #       variable=self.end_layer_var, command=self.update_layer_range).pack(side='left')

        # Display area for the image
        self.label = Label(self.root)
        self.label.pack()

        image = self.process_image()
        imgtk = ImageTk.PhotoImage(image=image)
        self.label.imgtk = imgtk
        self.label.config(image=imgtk)

        # Start the application
        self.root.mainloop()

    def toggle_min_max(self):
        if self.min_max_mode == 'min':
            print("Toggled to Max Mode")
            self.min_max_mode = 'max'
        else:
            print("Toggled to Min Mode")
            self.min_max_mode = 'min'

        self.update_image()

    def on_key_press(self, event):
        if event.keysym == "Left":
            self.end_layer_var.set(max(self.end_layer_var.get() - 1, 0))
            self.update_image()
        elif event.keysym == "Right":
            self.end_layer_var.set(min(self.end_layer_var.get() + 1, len(self.model_layer_idcs)))
            self.update_image()
        elif event.keysym == "a":
            self.start_layer_var.set(max(self.start_layer_var.get() - 1, 0))
            self.update_image()
        elif event.keysym == "d":
            self.start_layer_var.set(min(self.start_layer_var.get() + 1, len(self.model_layer_idcs)))
            self.update_image()
        print("Key pressed")

    def on_threshold_input_enter(self, event):
        try:
            threshold_val = self.threshold_input.get()
            self.threshold_var.set(threshold_val)

            self.update_image()
            # new_img = self.process_image(array=self.array)
            # imgtk = ImageTk.PhotoImage(image=new_img)
            # self.label.imgtk = imgtk
            # self.label.config(image=imgtk)

            print(f"Entered value: {float(threshold_val)}")
        except ValueError:
            print("Please enter a valid float number")

    def on_start_layer_input_enter(self, event):
        try:
            start_layer = self.start_layer_var.get()
            self.start_layer_var.set(start_layer)
            self.update_image()

            print(f"Entered value: {start_layer}")
        except ValueError:
            print("Please enter a valid int number")

    def on_end_layer_input_enter(self, event):
        try:
            end_layer = self.end_layer_var.get()
            self.end_layer_var.set(end_layer)
            self.update_image()

            print(f"Entered value: {end_layer}")
        except ValueError:
            print("Please enter a valid int number")

    def get_threshold(self):
        mode = self.mode_var.get()

        if mode == 2:
            threshold = self.curr_layer_val
        else:
            threshold = float(self.threshold_var.get())

        return threshold

    def set_threshold(self, threshold):
        mode = self.mode_var.get()

        if mode == 2:
            self.curr_layer_val = threshold
        else:
            self.threshold_var.set(threshold)

        return threshold

    def create_ensemble_dir_simple_names(self, model_dir, prefix):
        parts = []
        if 'superseded' in model_dir:
            part = model_dir.split('_')[2].split('-')[0:2]
        else:
            part = model_dir.split('_')[1].split('-')[0:2]
        parts.append(part)

        # Joining the first two parts (words) of each directory name
        simplified_parts = ["-".join(p) for p in parts]

        # Current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Creating the new directory name
        new_dir_name = prefix + "_".join(simplified_parts) + "_" + timestamp
        return new_dir_name

    def save_snapshot(self, transparent = False):
        global type_list
        # Save in model dir within fragment for single model
        model_dir = self.model_dir
        target_dir = os.path.join(self.root_dir, model_dir, "snapshots")
        file_prefix = ''

        os.makedirs(target_dir, exist_ok=True)

        if self.submission_mode:
            model_names_str = str(self.frag_id)
        else:
            model_names_str = self.create_ensemble_dir_simple_names(self.model_dir, file_prefix)

        mode_key = self.mode_var.get()
        mode = self.modes[mode_key].upper()
        inverted_str = f'_inverted' if self.inverted else ""
        transparent_str = f'_transparent' if transparent else ""
        type_str = f'_{type_list[0]}' if len(type_list[0]) > 0 else ""

        if mode == 'MAX' and self.min_max_mode == 'min':
            mode = 'MIN'

        # Layer mode
        if mode_key == 2:
            if self.save_all_layers:
                start, end = self.center_layers
                center_layers = set(range(start, end + 1))
                intersecting_layers = center_layers.intersection(self.model_layer_idcs)

                for layer_idx in sorted(list(intersecting_layers)):
                    # Check if the current layer is within the specified range and is part of model_layer_indices
                    if start <= layer_idx <= end and layer_idx in self.model_layer_idcs:
                        threshold = float(self.threshold_var.get())
                        file_name = f"{model_names_str}_mode={mode}_layer={layer_idx}_th={threshold:g}{inverted_str}.png"
                        file_path = os.path.join(target_dir, file_name)
                        self.curr_layer_val = int(layer_idx)
                        image = self.process_image(save_img=True)
                        print(f"Saving {file_name}")
                        image.save(file_path)
            else:
                layer = int(self.get_threshold())
                self.curr_layer_val = layer
                threshold = float(self.threshold_var.get())
                file_name = f"{os.path.basename(model_names_str)}_mode={mode}_layer={layer}_th={threshold:g}{inverted_str}.png"
                file_path = os.path.join(target_dir, file_name)
                image = self.process_image(save_img=True)
                image.save(file_path)
                print(f"Saving {file_name}")
        else:
            threshold = self.get_threshold()
            start_layer_idx = int(self.start_layer_var.get())
            last_start_layer_idx = self.model_layer_idcs[-1]

            layer_str = f'{start_layer_idx}-{last_start_layer_idx}'
            name = f"{os.path.basename(model_names_str)}_mode={mode}_layer={layer_str}_th={float(threshold):g}{inverted_str}{transparent_str}{type_str}.png"
            file_path = os.path.join(target_dir, name)

            image = self.process_image(save_img=True)

            if transparent:
                print("Start converting transparent image")
                processed = image.convert('RGBA')
                data = np.array(processed)

                # Calculate the inverse brightness as alpha
                # Brightness is the average of the R, G, and B channels
                brightness = np.mean(data[..., :3], axis=-1)
                alpha = 255 - brightness.astype(np.uint8)

                # Set all pixels to black and use calculated alpha
                data[..., :3] = 0  # Set R, G, B to 0 (black)
                data[..., 3] = alpha  # Set alpha channel

                # Convert back to Image
                processed = Image.fromarray(data)
                image = processed
                print("Done converting transparent image")

            print("Saved file at", file_path)
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

        if mode in [0, 1]:
            self.layer_label.pack_forget()
            self.left_button.pack_forget()
            self.right_button.pack_forget()
            self.slider.pack_forget()

        elif mode == 2:
            self.layer_label.pack(side='right')
            self.left_button.pack(side='left')
            self.slider.pack(side='left')
            self.right_button.pack(side='left')

        self.update_image()

    def update_image(self, threshold=None):
        if threshold:
            self.set_threshold(threshold)

        if self.mode_var.get() == 2:
            layer = self.curr_layer_val
            print("Selected layer", layer)
            self.layer_label.config(text=f"Current Layer: {self.file_names[int(layer)]}")

        image = self.process_image()
        imgtk = ImageTk.PhotoImage(image=image)
        self.label.imgtk = imgtk
        self.label.config(image=imgtk)

    def process_image(self, save_img=False):
        max_size = (1500, 800)
        if save_img:
            max_size = self.target_dims

        start_layer = self.start_layer_var.get()  # inclusive  38
        end_layer = self.end_layer_var.get()  # inclusive  40

        offset = self.model_layer_idcs[0]
        rel_start_idx = start_layer - offset
        rel_end_idx = end_layer - offset

        mode = self.modes[self.mode_var.get()].lower()
        processed = None

        if mode == "sum":
            processed = np.mean(self.model_layer_values[rel_start_idx:rel_end_idx + 1], axis=0)
        elif mode == "max":
            if self.min_max_mode == 'min':
                processed = np.minimum.reduce(self.model_layer_values[rel_start_idx:rel_end_idx + 1])
            else:
                processed = np.maximum.reduce(self.model_layer_values[rel_start_idx:rel_end_idx + 1])
        elif mode == "layers":
            layer = self.curr_layer_val
            processed = self.model_layer_values.copy()[int(layer)]

        assert processed is not None and len(processed.shape) == 2

        processed = self.normalize_npy_preds(processed)  # Normalize

        if not save_img:
            print(self.rotate_num)
            processed = self.rot90(processed, self.rotate_num)  # Rotate

            if self.flip_num is not None:
                print(self.flip_num)
                processed = np.flip(processed, self.flip_num)

        threshold = float(self.threshold_var.get())

        if threshold > 0:
            processed = (processed > threshold).astype(int)
            processed[processed >= float(threshold)] = 1  # clamp all values that are not 0 (or threshold) to 1
        # else:
        if threshold < 0:
            processed = processed * (threshold * -1)
            processed[processed > 1] = 1
            # OUTLINE BOOSTING
            # processed[processed < 1] = 0

        # Apply threshold
        print("Applied threshold:", float(threshold))

        if self.inverted:
            processed = 1 - processed

        image = np.uint8(processed * 255)
        aspect_ratio = image.shape[1] / image.shape[0]

        new_width = image.shape[1] * 4  # width
        new_height = image.shape[0] * 4  # height

        # Save image
        if save_img:
            original_height, original_width = max_size
            upscaled_image = resize(image, (new_height, new_width),
                                    order=0, preserve_range=True, anti_aliasing=False)

            assert new_width >= original_width and new_height >= original_height

            width_diff = new_width - original_width
            height_diff = new_height - original_height

            out_width = -width_diff
            out_height = -height_diff

            if width_diff == 0:
                out_width = new_width

            if height_diff == 0:
                out_height = new_height

            return Image.fromarray(np.array(upscaled_image)[:out_height, :out_width])

        # Determine new dimensions
        if aspect_ratio > 1:
            new_width = min(max_size[0], new_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_size[1], new_height)
            new_width = int(new_height * aspect_ratio)

        # Resize with high-quality resampling
        image = Image.fromarray(np.uint8(processed * 255), 'L')
        return image.resize((new_width, new_height), Resampling.LANCZOS)

    def decrease_slider(self):
        mode = self.mode_var.get()

        if mode == 2:
            current_value = self.slider.get()
            new_threshold = max(current_value - 1, 0)  # Ensure the value does not go below the minimum

            self.set_threshold(new_threshold)
            self.slider.set(new_threshold)

            # layer = self.curr_layer_val
            print("Selected layer", new_threshold)
            self.layer_label.config(text=f"Current Layer: {self.file_names[int(new_threshold)]}")

    def increase_slider(self):
        mode = self.mode_var.get()

        if mode == 2:
            max_val = len(self.model_layer_values) - 1
            current_value = self.slider.get()
            new_layer = min(current_value + 1, max_val)  # Ensure the value does not exceed the maximum

            self.set_threshold(new_layer)
            self.slider.set(new_layer)
            print("Selected layer", new_layer)
            self.layer_label.config(text=f"Current Layer: {self.file_names[int(new_layer)]}")


def load_predictions(root_dir, single_layer, layer_indices=None):
    global type_list
    layer_idcs = list()
    layer_values = []
    file_names = list()

    type_set = set()

    file_paths = [x for x in os.listdir(root_dir) if x.endswith('.npy')]
    for npy_file_name in file_paths:
        type_set.add(npy_file_name.split("sigmoid")[0])

    type_list = list(type_set)
    for i, name in enumerate(type_list):
        if type_list[i].endswith("-"):
            type_list[i] = type_list[i][:-1]

    if len(type_list) > 1:
        print("Which inference type do you want to see?")

        for i, type_name in enumerate(type_list):
            print(f"{i}: {type_name}")
        selection = int(input())
        check_start = type_list[selection]
        # if selection == 0:
        #     check_start = "sigmoid"

        type_list[0] = type_list[selection]
        file_paths = [x for x in file_paths if x.startswith(check_start)]

    file_paths.sort(key=lambda x: get_start_layer_idx(x, single_layer))

    for filename in file_paths:
        layer_start_idx = get_start_layer_idx(filename, single_layer)

        # If layer_indices are given, check if layer_start_idx is contained
        if layer_indices and layer_start_idx not in layer_indices:
            continue

        file_path = os.path.join(root_dir, filename)
        array = np.load(file_path)

        layer_idcs.append(layer_start_idx)
        layer_values.append(array)
        file_names.append(filename)

    if len(layer_idcs) == 0 or len(layer_values) == 0 or len(file_names) == 0:
        print("No valid layer files found")
        sys.exit(1)

    return layer_idcs, layer_values, file_names


if __name__ == "__main__":
    type_list = []
    save_all_layers = False
    max_ensemble = False
    submission_mode = False

    main()
