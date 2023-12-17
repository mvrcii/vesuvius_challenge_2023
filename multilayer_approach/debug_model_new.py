import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize
from torch import float16

from config_handler import Config
from losses.binary_dice_loss import MaskedBinaryDiceLoss
from losses.focal_loss import MaskedFocalLoss
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


def create_data(cfg):
    label = np.zeros((cfg.patch_size, cfg.patch_size))
    start = (cfg.patch_size // 2) - 32
    end = (cfg.patch_size // 2) + 32
    label[start + 70:end + 70, start:end] = 1
    label[start + 20:start + 40, :] = 1
    label[0:64, -64:-1] = 1

    # get random numpy array of label shape
    rarr = np.random.rand(*label.shape)

    # image = np.stack([label] * 16)
    image = np.stack([np.zeros_like(label)] * 16)
    image = np.stack([rarr] * 16)
    # make image random vectors of the same shape

    label = resize(label, ((cfg.patch_size // 4), (cfg.patch_size // 4)), order=0, preserve_range=True,
                   anti_aliasing=False)
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
    # print("Image Shape:", image.shape)

    image = image.to('cuda')

    # 256, dice, 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    epochs = 200

    # criterion = torch.nn.BCEWithLogitsLoss()
    focal_loss_fn = MaskedFocalLoss(gamma=0.5)
    dice_loss_fn = MaskedBinaryDiceLoss(from_logits=True)

    for x in range(epochs):
        # Forward Pass
        logits = model(image)  # N, H, W
        logits = logits.half()

        focal_loss = focal_loss_fn(logits, label)

        probabilities = torch.sigmoid(logits)

        # bce_loss = binary_cross_entropy_with_mask_batch(probabilities, label.unsqueeze(0), mask.unsqueeze(0))
        dice_loss = MaskedBinaryDiceLoss(logits, label.unsqueeze(0), mask.unsqueeze(0))

        # iou, precision, recall, f1 = calculate_masked_metrics_batchwise(probabilities, label.unsqueeze(0),
        #                                                                 mask.unsqueeze(0))
        # print all metrics in one line
        # print("IoU:", iou.item(), "Precision:", precision.item(), "Recall:", recall.item(), "F1:", f1.item())

        total_loss = dice_loss + focal_loss
        # total_loss = dice_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print("Loss:", total_loss)

        prediction = probabilities.squeeze().detach().cpu().numpy()
        plt.imshow(prediction, cmap='gray')
        plt.show()
