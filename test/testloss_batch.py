import torch
import matplotlib.pyplot as plt
from torch.nn import BCELoss


def dice_loss_with_mask_batch(outputs, labels, mask):
    # all 3 input variables should be shape (batch_size, label_size, label_size)
    # outputs should be sigmoided (0-1)
    # labels should be binary
    # mask should be binary
    smooth = 1.0
    # Apply mask
    outputs_masked = outputs * mask
    labels_masked = labels * mask

    # Sum over spatial dimensions, keep batch dimension
    intersection = (outputs_masked * labels_masked).sum(dim=[1, 2])
    union = outputs_masked.sum(dim=[1, 2]) + labels_masked.sum(dim=[1, 2]) - intersection

    # Compute dice loss per batch and average
    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
    return dice_loss.mean()


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


def dice_loss_with_mask_batch(outputs, labels, mask):
    # all 3 input variables should be shape (batch_size, label_size, label_size)
    # outputs should be sigmoided (0-1)
    # labels should be binary
    # mask should be binary
    smooth = 1.0
    # Apply mask
    outputs_masked = outputs * mask
    labels_masked = labels * mask

    # Sum over spatial dimensions, keep batch dimension
    intersection = (outputs_masked * labels_masked).sum(dim=[1, 2])
    union = outputs_masked.sum(dim=[1, 2]) + labels_masked.sum(dim=[1, 2])

    # Compute dice loss per batch and average
    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
    return dice_loss.mean()


def dice_loss_with_mask_batch_fixed(outputs, labels, mask):
    outputs_masked = outputs * mask
    labels_masked = labels * mask

    # Calculate intersection and union with masking
    intersection = (outputs_masked * labels_masked).sum(axis=(1, 2))
    print(intersection)
    union = (outputs_masked + labels_masked).sum(axis=(1, 2))

    # Compute dice loss per batch and average
    dice_loss = 1 - (2. * intersection) / union
    return dice_loss.mean()


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

output_batch = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
label_batch = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
mask_batch = torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])

loss = dice_loss_with_mask_batch_fixed(output_batch, label_batch, mask_batch)
print(loss)

# loss_bce = binary_cross_entropy_with_mask_batch(output_batch, label_batch, mask_batch)
# loss_dice = dice_loss_with_mask_batch(output_batch, label_batch, mask_batch)
# print("Loss bce: " + str(loss_bce))
# print("Loss dice: " + str(loss_dice))
