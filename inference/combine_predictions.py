import os
from tkinter import Tk, Scale, HORIZONTAL, Label, Button, Frame, IntVar, Radiobutton

import numpy as np
from PIL import Image, ImageTk
from tqdm import tqdm

arrays = []
threshold = 0.5


def process_image(array, threshold, max_size=(1800, 1000)):
    processed = array.copy()

    # Apply threshold
    # processed = np.where(array > threshold, 1, 0)

    # reverse threshold
    processed[processed < threshold] = 0
    # processed = array

    # Brightness
    # image = array
    # current_min = np.min(image)
    # current_max = np.max(image)
    # new_max = 1
    # new_min = threshold
    # processed = (image - current_min) / (current_max - current_min) * (new_max - new_min) + new_min

    image = Image.fromarray(np.uint8(processed * 255), 'L')

    # Resize while maintaining aspect ratio
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        # Width is greater than height
        new_width = min(image.width, max_size[0])
        new_height = int(new_width / aspect_ratio)
    else:
        # Height is greater than or equal to width
        new_height = min(image.height, max_size[1])
        new_width = int(new_height * aspect_ratio)

    return image.resize((new_width, new_height), Image.ANTIALIAS)


def update_image(value):
    global threshold
    # Update the image based on the slider's value
    threshold = float(value)
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

    paths = [x for x in os.listdir(directory_path) if x.endswith('.npy')]
    paths.sort(key=lambda x: int(x.split('_')[2]))

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
    return normalized_array


def increase_slider():
    current_value = slider.get()
    new_value = min(current_value + 0.001, 1)  # Ensure the value does not exceed the maximum
    slider.set(new_value)
    update_image(new_value)


def decrease_slider():
    current_value = slider.get()
    new_value = max(current_value - 0.001, 0)  # Ensure the value does not go below the minimum
    slider.set(new_value)
    update_image(new_value)


def mode_changed():
    global array
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
        array = normalized_array
    elif mode == 1:
        combined_array = np.sum(arrays, axis=0)
        min_value = combined_array.min()
        max_value = combined_array.max()
        normalized_array = (combined_array - min_value) / (max_value - min_value)
        array = normalized_array
    update_image(threshold)


if __name__ == "__main__":
    # folder_path = r"A:\handlabel\test\20231123-205540"  # big (330?)
    folder_path = r"A:\handlabel\test\20231123-212933"  # small (35)
    # folder_path = r"A:\handlabel\test\20231125-220804"  # huge 336
    array = combine_arrays(folder_path, ignore_percent=0.25)
    # print(array.shape)
    # exit()

    # Create main window
    root = Tk()
    root.title("Threshold Visualizer")

    # Variable to hold the selected mode
    mode_var = IntVar(value=0)  # Default to mode 0

    # Create a frame for the mode selection buttons
    mode_frame = Frame(root)
    mode_frame.pack(side='top')

    # List of modes
    modes = ["Sum", "Max"]

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
