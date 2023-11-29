import os
from tkinter import Tk, Scale, HORIZONTAL, Label, Button, Frame, IntVar, Radiobutton

import numpy as np
from PIL import Image, ImageTk
from PIL.Image import Resampling
from tqdm import tqdm

arrays = []
threshold = 0.5
slider = None
inverted = False


def process_image(array, threshold, max_size=(1500, 800), inverted=False):
    processed = array.copy()

    # Apply threshold
    processed[processed < threshold] = 0
    if inverted:
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


def update_image(value):
    if mode_var.get() == 2:
        layer = int(value)
        global array
        array = arrays[layer]
        array = np.rot90(array, -1)
        print("setting to layers", layer)
        new_img = process_image(array, 0)
        imgtk = ImageTk.PhotoImage(image=new_img)
        label.imgtk = imgtk
        label.config(image=imgtk)
        return

    global threshold
    threshold = float(value)
    # Update the image based on the slider's value
    new_img = process_image(array, threshold)
    imgtk = ImageTk.PhotoImage(image=new_img)
    label.imgtk = imgtk
    label.config(image=imgtk)


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


def combine_arrays(directory_path, ignore_percent=0):
    global arrays

    paths = [x for x in os.listdir(directory_path) if x.endswith('.npy') and not x.startswith('maxed_logits')]
    paths.sort(key=lambda x: int(x.split('_')[2]))

    # paths = paths[28:38]

    # Load all numpy arrays from the directory
    for filename in tqdm(paths):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory_path, filename)
            array = np.load(file_path)
            arrays.append(array)

    if ignore_percent > 0:
        print(f"Ignoring {ignore_percent * 100}% of top and bottom layers")
        print("Before:", len(arrays))
        ignore_count = int(ignore_percent * len(arrays))
        arrays = arrays[ignore_count:-ignore_count]
        print("After:", len(arrays))

    combined_array = np.maximum.reduce(arrays)
    # combined_array = np.sum(arrays, axis=0)
    # combined_array = combine_layers(arrays, max_distance=2)  # max_distance: how far can layer influence each other

    min_value = combined_array.min()
    max_value = combined_array.max()
    normalized_array = (combined_array - min_value) / (max_value - min_value)

    rotated_array = np.rot90(normalized_array, -1)
    return rotated_array


def increase_slider():
    step = 0.001
    max_val = 1
    if mode_var.get() == 2:
        step = 1
        max_val = len(arrays) - 1
    current_value = slider.get()
    new_value = min(current_value + step, max_val)  # Ensure the value does not exceed the maximum
    slider.set(new_value)
    update_image(new_value)


def decrease_slider():
    step = 0.001
    if mode_var.get() == 2:
        step = 1
    current_value = slider.get()
    new_value = max(current_value - step, 0)  # Ensure the value does not go below the minimum
    slider.set(new_value)
    update_image(new_value)


def mode_changed():
    global slider
    global array
    global threshold
    # This function will be called when the mode is changed.
    # You can use mode_var.get() to get the current mode value.
    mode = mode_var.get()
    if mode == 0:
        combined_array = np.maximum.reduce(arrays)
        # combined_array = np.sum(arrays, axis=0)
        # combined_array = combine_layers(arrays, max_distance=2)  # max_distance: how far can layer influence each other

        min_value = combined_array.min()
        max_value = combined_array.max()
        normalized_array = (combined_array - min_value) / (max_value - min_value)
        array = np.rot90(normalized_array, -1)
        update_image(threshold)
        slider.config(resolution=0.001)
        slider.config(from_=0)
        slider.config(to=1)
        slider.set(0.5)
    elif mode == 1:
        combined_array = np.sum(arrays, axis=0)
        min_value = combined_array.min()
        max_value = combined_array.max()
        normalized_array = (combined_array - min_value) / (max_value - min_value)
        array = np.rot90(normalized_array, -1)
        update_image(threshold)
        slider.config(resolution=0.001)
        slider.config(from_=0)
        slider.config(to=1)
        slider.set(0.5)
    elif mode == 2:
        slider.config(resolution=1)
        slider.config(from_=0)
        slider.config(to=len(arrays) - 1)
        slider.set(0)
        array = np.rot90(arrays[0], -1)
        threshold = 0
        update_image(threshold)


def save_snapshot():
    new_img = process_image(array, threshold, max_size=(10479, 10360))
    os.makedirs("snapshots", exist_ok=True)
    new_img.save(os.path.join("snapshots", f"{frag_id}_snapshot_{threshold}.png"))


def invert_colors():
    global inverted
    inverted = not inverted
    update_image(threshold)


if __name__ == "__main__":
    # folder_path = r"A:\handlabel\test\20231123-205540"  # big (330?)
    # folder_path = r"A:\handlabel\test\20231123-212933"  # small (35)
    # folder_path = r"A:\handlabel\test\20231125-220804"  # huge 336
    # folder_path = r"C:\Users\Marce\Git-Master\Privat\kaggle1stReimp\inference\results\fragment20230702185753\20231128-224709"
    folder_path = r"C:\Users\Marce\Git-Master\Privat\kaggle1stReimp\inference\results\fragment20230702185753\20231129-023606"
    array = combine_arrays(folder_path, ignore_percent=0)

    frag_id = folder_path.split('\\')[-2]

    # Create main window
    root = Tk()
    root.title("Threshold Visualizer")

    # Variable to hold the selected mode
    mode_var = IntVar(value=0)  # Default to mode 0

    # Utility frame
    utility_frame = Frame(root)
    utility_frame.pack(side='top')
    save_button = Button(utility_frame, text="Save Snapshot", command=save_snapshot)
    save_button.pack(side="left")
    invert_button = Button(utility_frame, text="Invert colors", command=invert_colors)
    invert_button.pack(side="left")

    # Create a frame for the mode selection buttons
    mode_frame = Frame(root)
    mode_frame.pack(side='top')

    # List of modes
    modes = ["Sum", "Max", "Layers"]

    # Create a Radiobutton for each mode
    for index, mode in enumerate(modes):
        Radiobutton(mode_frame, text=mode, variable=mode_var, value=index, command=mode_changed).pack(side='left')

    # Create a frame for the slider and buttons
    control_frame = Frame(root)
    control_frame.pack(side='top')

    # Add left button for decreasing the slider value
    left_button = Button(control_frame, text="  -  ", command=decrease_slider)
    left_button.pack(side='left')

    # Add a slider
    slider = Scale(control_frame, from_=0, to=1, orient=HORIZONTAL, resolution=0.001, length=500, command=update_image)
    slider.set(0.5)  # Set initial value of the slider
    slider.pack(side='left')

    # Add right button for increasing the slider value
    right_button = Button(control_frame, text="  +  ", command=increase_slider)
    right_button.pack(side='left')

    # Display area for the image
    label = Label(root)
    label.pack()

    # Initial image display
    update_image(0.5)

    # Start the application
    root.mainloop()
