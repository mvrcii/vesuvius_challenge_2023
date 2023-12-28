import random

import ipywidgets as widgets
import matplotlib.pyplot as plt
import torch

from models.losses.focal_loss import MaskedFocalLoss


def combined_plot(y_pred, y_true, y_mask, loss, gamma, alpha, title):
    """
    Create a combined plot with the test case (y_pred, y_true, y_mask) on the left
    and the current loss value for a specific combination of alpha and gamma on the right.
    :param y_pred: Predicted labels tensor.
    :param y_true: True labels tensor.
    :param y_mask: Mask tensor.
    :param loss: Current loss value.
    :param gamma: Current gamma value.
    :param alpha: Current alpha value.
    :param title: Title for the combined plot.
    """
    fig = plt.figure(figsize=(16, 6))

    # Plot for y_pred, y_true, y_mask
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(y_pred[0], cmap='Blues', alpha=0.5, label='y_pred')  # Blue for y_pred
    ax1.imshow(y_true[0], cmap='Reds', alpha=0.5, label='y_true')   # Red for y_true
    ax1.set_title('y_pred (Blue) and y_true (Red) Overlap')
    ax1.grid(False)

    # Loss value plot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.text(0.5, 0.5, f'Loss: {loss:.4f}\nGamma: {gamma}\nAlpha: {alpha}',
             horizontalalignment='center', verticalalignment='center',
             fontsize=16, transform=ax2.transAxes)
    ax2.set_title('Current Loss Value')
    ax2.axis('off')

    plt.suptitle(title)
    plt.show()

def create_test_case(overlap_percentage, positive_pixels_percentage):
    tensor_size = (1, 256, 256)
    total_pixels = tensor_size[1] * tensor_size[2]
    num_positive_pixels = int(total_pixels * positive_pixels_percentage / 100)
    num_overlap_pixels = int(num_positive_pixels * overlap_percentage / 100)

    # Initialize tensors
    y_pred = torch.zeros(tensor_size)
    y_true = torch.zeros(tensor_size)

    # Randomly select positions for positive pixels
    all_pixels = [(i, j) for i in range(tensor_size[1]) for j in range(tensor_size[2])]
    positive_positions = set(random.sample(all_pixels, num_positive_pixels))

    # Assign overlapping positive pixels
    overlap_positions = set(random.sample(positive_positions, num_overlap_pixels))
    for pos in overlap_positions:
        y_pred[0, pos[0], pos[1]] = 1
        y_true[0, pos[0], pos[1]] = 1

    # Assign non-overlapping positive pixels
    non_overlap_pred = positive_positions - overlap_positions
    non_overlap_true = set(random.sample(all_pixels, num_positive_pixels - len(overlap_positions))) - overlap_positions

    for pos in non_overlap_pred:
        y_pred[0, pos[0], pos[1]] = 1

    for pos in non_overlap_true:
        y_true[0, pos[0], pos[1]] = 1

    # Mask remains the same
    y_mask = torch.ones(tensor_size)

    return y_pred, y_true, y_mask


def update_plot(gamma, alpha, overlap, positive_pixels):
    y_pred, y_true, y_mask = create_test_case(overlap_percentage=overlap, positive_pixels_percentage=positive_pixels)
    loss_fn = MaskedFocalLoss(gamma=gamma, alpha=alpha)
    loss = loss_fn(y_pred_unmasked=y_pred, y_true_unmasked=y_true, y_mask=y_mask).item()
    combined_plot(y_pred, y_true, y_mask, loss, gamma, alpha, "Interactive Focal Loss Visualization")


def main():
    gamma_slider = widgets.FloatSlider(value=1.0, min=1.0, max=5.0, step=1.0, description='Gamma:')
    alpha_slider = widgets.FloatSlider(value=0.25, min=0.0, max=1.0, step=0.05, description='Alpha:')
    overlap_slider = widgets.IntSlider(value=50, min=0, max=100, step=1, description='Overlap %:')
    positive_pixels_slider = widgets.IntSlider(value=50, min=0, max=100, step=1, description='Positive Pixels %:')

    widgets.interactive(update_plot, gamma=gamma_slider, alpha=alpha_slider,
                        overlap=overlap_slider, positive_pixels=positive_pixels_slider)


