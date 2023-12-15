import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import float16
from torchvision.utils import make_grid

from models.abstract_model import AbstractVesuvLightningModule
from models.architectures.unetr_segformer import UNETR_Segformer


def dice_loss_with_mask_batch(outputs, labels, mask):
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


def load_test_image(cfg):
    path = os.path.join("multilayer_approach/datasets/sanity/JETFIRE")
    image = np.load(os.path.join(path, 'images', 'f20231005123336_ch30_17134_2873_17646_3385.npy'))
    label = np.load(os.path.join(path, 'labels', 'f20231005123336_ch30_17134_2873_17646_3385.npy'))
    label = np.unpackbits(label).reshape((2, cfg.label_size, cfg.label_size))  # 2, 128, 128

    label = label[0]  # 128, 128 now

    label = torch.from_numpy(label).to(dtype=float16, device='cuda')
    image = torch.from_numpy(image).to(dtype=float16, device='cuda')

    image = image.unsqueeze(0)
    print("Image Shape", image.shape)

    pad_array = torch.zeros(1, 16 - image.shape[1], cfg.patch_size, cfg.patch_size).to(dtype=float16, device='cuda')
    print("Pad Shape", pad_array.shape)

    image = torch.cat([image, pad_array], dim=1)

    return image, label


class UNETR_SFModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.cfg = cfg
        self.model = UNETR_Segformer(cfg=cfg)

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        data, label = batch
        probabilities = torch.sigmoid(self.forward(data))

        target = label[:, 0]
        keep_mask = label[:, 1]

        dice_loss = dice_loss_with_mask_batch(probabilities, target, keep_mask)

        self.update_unetr_training_metrics(dice_loss)

        # if self.global_step % 20 == 0:
        #     with torch.no_grad():
        #         test_img_tensor, test_label_tensor = load_test_image(cfg=self.cfg)
        #
        #         test_logits = self.forward(test_img_tensor)
        #         test_probs = torch.sigmoid(test_logits)
        #
        #         combined = torch.cat([test_probs, test_label_tensor], dim=2)
        #
        #         # Convert your output tensor to an image or grid of images
        #         grid = make_grid(combined).detach().cpu()
        #
        #         test_image = wandb.Image(grid, caption="Step {}".format(self.global_step))
        #
        #         self.log({"test_image_pred": test_image})

        return dice_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        probabilities = torch.sigmoid(self.forward(data))

        target = label[:, 0]
        keep_mask = label[:, 1]

        dice_loss = dice_loss_with_mask_batch(probabilities, target, keep_mask)

        iou, precision, recall, f1 = calculate_masked_metrics_batchwise(probabilities, target, keep_mask)

        self.update_unetr_validation_metrics(dice_loss, iou, precision, recall, f1)

    def update_unetr_validation_metrics(self, loss, iou, precision, recall, f1):
        self.log(f'val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_precision', precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_recall', recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def update_unetr_training_metrics(self, loss):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log(f'train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
