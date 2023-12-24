import matplotlib.pyplot as plt
import torch
from torch.nn import BCELoss


# Function to plot the arrays
def plot_arrays(output, label, mask):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plotting each array with a title
    im0 = axes[0].imshow(output[0], cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Output')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(label[0], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Label')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(mask[0], cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Mask')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])

    plt.show()


def calculate_masked_metrics_batchwise(outputs, labels, mask):
    # Ensure batch dimension is maintained during operations
    outputs = (outputs > 0.5).float()
    batch_size = outputs.size(0)

    # Flatten tensors except for the batch dimension
    outputs_flat = (outputs * mask).view(batch_size, -1)
    labels_flat = (labels * mask).view(batch_size, -1)

    # Calculate True Positives, False Positives, and False Negatives for each batch
    true_positives = (outputs_flat * labels_flat).sum(dim=1)
    false_positives = (outputs_flat * (1 - labels_flat)).sum(dim=1)
    false_negatives = ((1 - outputs_flat) * labels_flat).sum(dim=1)

    # Calculate metrics for each batch
    iou = true_positives / (
            true_positives + false_positives + false_negatives + 1e-6)  # Added epsilon for numerical stability
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # Added epsilon for F1 calculation

    return iou, precision, recall, f1


def binary_cross_entropy_with_mask_batch(outputs, labels, mask):
    # all 3 input variables should be shape (batch_size, label_size, label_size)
    # outputs should be sigmoided (0-1)
    # labels should be binary
    # mask should be binary
    criterion = BCELoss(reduction='none')
    bce_loss = criterion(outputs, labels)

    # Apply mask and keep batch dimension
    masked_bce_loss = bce_loss * mask

    # Average over all dimensions
    return masked_bce_loss.mean(dim=[0, 1, 2])


full_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
empty_mask = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
half_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

mask = half_mask

# Case 1: Perfectly predicted only 0
# output_batch = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]])
# label_batch = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
# mask_batch = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])
# mask_batch = torch.ones_like(label_batch)

output_batch = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
label_batch = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])
mask_batch = torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])

iou, precision, recall, f1 = calculate_masked_metrics_batchwise(output_batch, label_batch, mask_batch)
# print metrics in one string
print("IoU: " + str(iou.item()) + " Precision: " + str(precision.item()) + " Recall: " + str(
    recall.item()) + " F1: " + str(f1.item()))

loss = dice_loss_with_mask_batch_fixed(output_batch, label_batch, mask_batch)
print("Loss: " + str(loss.item()))
plot_arrays(output_batch, label_batch, mask_batch)

# loss_bce = binary_cross_entropy_with_mask_batch(output_batch, label_batch, mask_batch)
# loss_dice = dice_loss_with_mask_batch(output_batch, label_batch, mask_batch)
# print("Loss bce: " + str(loss_bce))
# print("Loss dice: " + str(loss_dice))
