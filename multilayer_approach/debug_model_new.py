import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize
from torch import float16
from torch.nn import BCELoss
from tqdm import tqdm

from config_handler import Config
from models.architectures.unetr_segformer import UNETR_Segformer


def calculate_masked_metrics_batchwise(outputs, labels, mask):
    # Ensure batch dimension is maintained during operations
    outputs = (outputs > 0.5).float()
    batch_size = outputs.size(0)

    # Flatten tensors except for the batch dimension
    outputs_flat = (outputs * mask).view(batch_size, -1)
    labels_flat = (labels * mask).view(batch_size, -1)
    # print("Positives: ", outputs_flat.sum(dim=1))

    # Calculate True Positives, False Positives, and False Negatives for each batch
    true_positives = (outputs_flat * labels_flat).sum(dim=1)
    false_positives = (outputs_flat * (1 - labels_flat)).sum(dim=1)
    false_negatives = ((1 - outputs_flat) * labels_flat).sum(dim=1)
    # print("True Positives:", true_positives, "False Positives:", false_positives, "False Negatives:", false_negatives)

    # Calculate metrics for each batch
    iou = true_positives / (
                true_positives + false_positives + false_negatives + 1e-6)  # Added epsilon for numerical stability
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # Added epsilon for F1 calculation

    return iou, precision, recall, f1


def dice_loss_with_mask_batch_new(outputs, labels, mask):
    # all 3 input variables should be shape (batch_size, label_size, label_size)
    # outputs should be sigmoided (0-1)
    # labels should be binary
    # mask should be binary
    outputs_masked = outputs * mask
    labels_masked = labels * mask

    # Calculate intersection and union with masking
    intersection = (outputs_masked * labels_masked).sum(axis=(1, 2))
    union = (outputs_masked + labels_masked).sum(axis=(1, 2))

    # Compute dice loss per batch and average
    dice_loss = 1 - (2. * intersection) / union
    return dice_loss.mean()
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


def create_data(cfg):
    label = np.ones((cfg.patch_size, cfg.patch_size))
    start = (cfg.patch_size // 2) - 32
    end = (cfg.patch_size // 2) + 32
    label[start + 70:end + 70, start:end] = 0
    label[0:64, -64:-1] = 0

    # get random numpy array of label shape
    rarr = np.random.rand(*label.shape)


    # image = np.stack([label] * 16)
    image = np.stack([np.ones_like(label)] * 16)
    image = np.stack([rarr] * 16)
    # make image random vectors of the same shape


    label = resize(label, ((cfg.patch_size // 4), (cfg.patch_size // 4)), order=0, preserve_range=True, anti_aliasing=False)
    plt.imshow(label, cmap='gray', vmin=0, vmax=1)
    plt.title("label")
    plt.show()

    return image, label


if __name__ == "__main__":

    cfg = Config.load_from_file("configs/unet_sf/config_debug.py")
    image, label = create_data(cfg)

    model = UNETR_Segformer(cfg)

    if torch.cuda.is_available():
        model.to('cuda')
    else:
        print("CUDA is not available. The model will remain on the CPU.")

    mask_np = np.ones_like(label)
    # mask_np[:, 24:40] = 0
    # plt.imshow(mask_np, cmap='gray')
    # plt.show()

    plt.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
    plt.title("mask")
    plt.show()

    mask = torch.tensor(mask_np, dtype=float16).to('cuda')

    label = torch.tensor(label, dtype=float16).to('cuda')

    # plt.imshow(label, cmap='gray')
    # plt.imshow(image, cmap='gray')
    # plt.show()

    image = torch.Tensor(image).unsqueeze(0).unsqueeze(0).float()
    assert image.dim() == 5
    print("Image Shape:", image.shape)

    image = image.to('cuda')

    # 256, dice, 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 200
    criterion = torch.nn.BCEWithLogitsLoss()
    for x in tqdm(range(epochs)):
        # Forward Pass
        logits = model(image)
        logits = logits.half()

        probabilities = torch.sigmoid(logits)

        # bce_loss = binary_cross_entropy_with_mask_batch(probabilities, label.unsqueeze(0), mask.unsqueeze(0))
        dice_loss = dice_loss_with_mask_batch_new(probabilities, label.unsqueeze(0), mask.unsqueeze(0))
        iou, precision, recall, f1 = calculate_masked_metrics_batchwise(probabilities, label.unsqueeze(0), mask.unsqueeze(0))
        # print all metrics in one line
        print("IoU:", iou.item(), "Precision:", precision.item(), "Recall:", recall.item(), "F1:", f1.item())

        # total_loss = bce_loss + dice_loss
        # total_loss = dice_loss
        total_loss = criterion(logits, label.unsqueeze(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print("Loss:", total_loss)

        prediction = probabilities.squeeze().detach().cpu().numpy()
        plt.imshow(prediction, cmap='gray')
        plt.show()
