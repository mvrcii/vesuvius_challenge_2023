import os
from tkinter import Tk, Scale, HORIZONTAL, Label, Button

import numpy as np
from PIL import Image, ImageTk
from tqdm import tqdm


def process_image(array, threshold, max_size=(1300, 800)):
    # Apply threshold
    processed = np.where(array > threshold, 1, 0)
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
    arrays = []

    paths = [x for x in os.listdir(directory_path) if x.endswith('.npy')]
    paths.sort(key=lambda x: int(x.split('_')[2]))

    # Load all numpy arrays from the directory
    for filename in tqdm(paths):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory_path, filename)
            array = np.load(file_path)
            arrays.append(array)

    ignore_percent = 0.1
    if ignore_percent > 0:
        print(f"Ignoring {ignore_percent * 100}% of top and bottom layers")
        print("Before:", len(arrays))
        ignore_count = int(ignore_percent * len(arrays))
        arrays = arrays[ignore_count:-ignore_count]
        print("After:", len(arrays))

    combined_array = combine_layers(arrays, max_distance=2)  # max_distance: how far can layer influence each other

    min_value = combined_array.min()
    max_value = combined_array.max()
    normalized_array = (combined_array - min_value) / (max_value - min_value)
    return normalized_array
    # plt.imshow(normalized_array > 0.2, cmap='gray')

    # Display and save the plot
    # plt.imshow(combined_array, cmap='gray')  # Change colormap if needed
    # plt.colorbar()
    # plt.show()
    # plt.savefig(save_path)


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


if __name__ == "__main__":
    folder_path = r"A:\handlabel\test\20231123-205540"  # big (330?)
    # folder_path = r"A:\handlabel\test\20231123-212933"  # small (535)
    array = combine_arrays(folder_path, ignore_percent=0.2)
    # print(array.shape)
    # exit()

    # Create main window
    root = Tk()
    root.title("Threshold Visualizer")

    # Add left button for decreasing the slider value
    left_button = Button(root, text="Left", command=decrease_slider)
    left_button.pack(side='left')

    # Add a slider
    slider = Scale(root, from_=0, to=1, orient=HORIZONTAL, resolution=0.001, length=500, command=update_image)
    slider.set(0.5)  # Set initial value of the slider
    slider.pack(side='left')

    # Add right button for increasing the slider value
    right_button = Button(root, text="Right", command=increase_slider)
    right_button.pack(side='left')

    # Display area for the image
    label = Label(root)
    label.pack()

    # Initial image display
    update_image(0.5)

    # Start the application
    root.mainloop()
