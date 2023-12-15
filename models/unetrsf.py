import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import float16
from torchvision.utils import make_grid

from models.abstract_model import AbstractVesuvLightningModule
from models.architectures.unetr_segformer import UNETR_Segformer


def binary_cross_entropy_with_mask(outputs, labels, mask):
    bce_loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
    masked_bce_loss = bce_loss * mask
    return masked_bce_loss.mean()


def dice_loss_with_mask(outputs, labels, mask):
    smooth = 1.0
    # Apply mask
    outputs_masked = outputs * mask
    labels_masked = labels * mask
    intersection = (outputs_masked * labels_masked).sum()
    union = outputs_masked.sum() + labels_masked.sum()
    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
    return dice_loss


def calculate_masked_metrics(outputs, labels, mask):
    # Flatten tensors for simplicity
    outputs_flat = (outputs * mask).view(-1)
    labels_flat = (labels * mask).view(-1)

    # Calculate True Positives, False Positives, and False Negatives
    true_positives = (outputs_flat * labels_flat).sum()
    false_positives = (outputs_flat * (1 - labels_flat)).sum()
    false_negatives = ((1 - outputs_flat) * labels_flat).sum()

    # Calculate metrics
    iou = true_positives / (true_positives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

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
        self.model = UNETR_Segformer(cfg=cfg)
        self.test_image, self.test_label = load_test_image(cfg=cfg)

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        probabilities = torch.sigmoid(logits)

        target = label[:, 0]
        keep_mask = label[:, 1]

        bce_loss = binary_cross_entropy_with_mask(logits, target, keep_mask)
        dice_loss = dice_loss_with_mask(probabilities, target, keep_mask)
        total_loss = bce_loss + dice_loss

        self.update_unetr_training_metrics(total_loss)

        if self.global_step % 20 == 0:
            test_logits = self.forward(self.test_image)
            test_probs = torch.sigmoid(test_logits)

            combined = torch.cat([test_probs, self.test_label], dim=2)

            # Convert your output tensor to an image or grid of images
            grid = make_grid(combined).detach().cpu()

            test_image = wandb.Image(grid, caption="Step {}".format(self.global_step))

            self.log({"test_image_pred": test_image})

        return total_loss

    def validation_step(self, batch, batch_idx):
        # data shape is (batch_size, 1, 16, patch_size, patch_size) (1 for channel)
        data, label = batch
        logits = self.forward(data)
        probabilities = torch.sigmoid(logits)

        target = label[:, 0]
        keep_mask = label[:, 1]

        bce_loss = binary_cross_entropy_with_mask(logits, target, keep_mask)
        dice_loss = dice_loss_with_mask(probabilities, target, keep_mask)
        total_loss = bce_loss + dice_loss

        iou, precision, recall, f1 = calculate_masked_metrics(probabilities, target, keep_mask)

        self.update_unetr_validation_metrics(total_loss, iou, precision, recall, f1)

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
