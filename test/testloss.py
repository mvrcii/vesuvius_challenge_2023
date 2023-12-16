import torch
import matplotlib.pyplot as plt
from torch.nn import BCELoss


def dice_loss_with_mask(outputs, labels, mask):
    smooth = 1.0
    # Apply mask
    outputs_masked = outputs * mask
    labels_masked = labels * mask
    intersection = (outputs_masked * labels_masked).sum()
    union = outputs_masked.sum() + labels_masked.sum() - intersection
    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
    return dice_loss


def binary_cross_entropy_with_mask(outputs, labels, mask):
    # THIS EXPECTS SIGMOIDED OUTPUT
    criterion = BCELoss(reduction='none')
    bce_loss = criterion(outputs, labels)
    masked_bce_loss = bce_loss * mask
    return masked_bce_loss.mean()


# Function to visualize tensors as images
# Function to visualize tensors as images with borders
# Function to visualize tensors with border
def visualize_tensors_with_border(tensors, titles):
    plt.figure(figsize=(10, 10))
    for i, (tensor, title) in enumerate(zip(tensors, titles), 1):
        ax = plt.subplot(3, 3, i)  # Changed to a 3x3 grid
        plt.imshow(tensor, cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.axis('off')
    plt.show()


full_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
empty_mask = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
half_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

mask = full_mask

# Case 1: Perfectly predicted only 0
output1 = torch.tensor([[0.0, 0.1], [0.1, 0.0]])
label1 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
loss1_bce = binary_cross_entropy_with_mask(output1, label1, mask)
loss1_dice = dice_loss_with_mask(output1, label1, mask)

# Case 2: Predicted 50% correct
output2 = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
label2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
loss2_bce = binary_cross_entropy_with_mask(output2, label2, mask)
loss2_dice = dice_loss_with_mask(output2, label2, mask)

# Case 3: Predicted 1 when should be all 0
output3 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
label3 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
loss3_bce = binary_cross_entropy_with_mask(output3, label3, mask)
loss3_dice = dice_loss_with_mask(output3, label3, mask)

# Visualizing the test cases
test_cases = [(output1, "Output 1"), (label1, "Label 1"), (mask, "Mask 1"),
              (output2, "Output 2"), (label2, "Label 2"), (mask, "Mask 2"),
              (output3, "Output 3"), (label3, "Label 3"), (mask, "Mask 3")]

print("Loss 1 BCE: ", loss1_bce.item(), "Loss 1 dice: ", loss1_dice)
print("Loss 2 BCE: ", loss2_bce.item(), "Loss 2 dice: ", loss2_dice)
print("Loss 3 BCE: ", loss3_bce.item(), "Loss 3 dice: ", loss3_dice)

visualize_tensors_with_border([tc[0] for tc in test_cases], [tc[1] for tc in test_cases])
