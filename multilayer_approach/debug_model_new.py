import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize
from torch import float16

from utility.config_handler import Config
from models.losses.binary_bce_loss import MaskedBinaryBCELoss
from models.losses import MaskedBinaryDiceLoss
from models.losses import MaskedFocalLoss
from models.architectures.unet3d_segformer import UNET3D_Segformer


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

    cfg = Config.load_from_file("configs/unet3d/config_debug.py")
    image, label = create_data(cfg)

    # model = UNETR_Segformer(cfg)
    model = UNET3D_Segformer(cfg)

    if torch.cuda.is_available():
        model.to('cuda')
    else:
        print("CUDA is not available. The model will remain on the CPU.")

    mask_np = np.ones_like(label)

    plt.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
    plt.title("mask")
    plt.show()

    mask = torch.tensor(mask_np, dtype=float16).to('cuda')
    label = torch.tensor(label, dtype=float16).to('cuda')

    image = torch.Tensor(image).unsqueeze(0).unsqueeze(0).float()
    image = image.to('cuda')

    epochs = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # criterion = torch.nn.BCEWithLogitsLoss()
    # focal_loss_fn = MaskedFocalLoss(gamma=0.5)
    dice_loss_fn = MaskedBinaryDiceLoss(from_logits=True)
    focal_loss_fn = MaskedFocalLoss(gamma=2.0, alpha=0.25)
    bce_loss_fn = MaskedBinaryBCELoss(from_logits=True)

    for x in range(epochs):
        logits = model(image)  # N, H, W
        logits = logits.half()

        probabilities = torch.sigmoid(logits)

        bce_loss = bce_loss_fn(logits, label.unsqueeze(0), mask.unsqueeze(0))
        dice_loss = dice_loss_fn(logits, label.unsqueeze(0), mask.unsqueeze(0))
        focal_loss = focal_loss_fn(logits, label.unsqueeze(0), mask.unsqueeze(0))

        # iou, precision, recall, f1 = calculate_masked_metrics_batchwise(probabilities, label.unsqueeze(0),
        #                                                                 mask.unsqueeze(0))
        # print all metrics in one line
        # print("IoU:", iou.item(), "Precision:", precision.item(), "Recall:", recall.item(), "F1:", f1.item())

        total_loss = focal_loss + dice_loss
        # total_loss = dice_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Epoch={x} Lr={scheduler.get_last_lr()} Loss:", total_loss)

        if x % 20 == 0:
            prediction = probabilities.squeeze().detach().cpu().numpy()
            plt.imshow(prediction, cmap='gray')
            plt.show()
