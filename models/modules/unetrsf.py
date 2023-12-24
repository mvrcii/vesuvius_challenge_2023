import torch
import wandb
from segmentation_models_pytorch.utils.metrics import IoU
from torchmetrics import Dice
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall
from torchvision.utils import make_grid

from models.architectures.unetr_segformer import UNETR_Segformer
from models.losses.utils import get_loss_functions
from models.modules.abstract_module import AbstractLightningModule


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

    return iou.mean(), precision.mean(), recall.mean(), f1.mean()


class UNETR_SFModule(AbstractLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.model = UNETR_Segformer(cfg=cfg)

        self.load_weights()

    def training_step(self, batch, batch_idx):
        self.train_step += 1
        data, label = batch

        probabilities = torch.sigmoid(self.forward(data))

        target = label[:, 0]
        keep_mask = label[:, 1]

        dice_loss = dice_loss_with_mask_batch(probabilities, target, keep_mask)

        self.update_unetr_training_metrics(dice_loss)

        if batch_idx == 5 or batch_idx == 10:
            with torch.no_grad():
                combined = torch.cat([probabilities[0], target[0], keep_mask[0]], dim=1)
                grid = make_grid(combined).detach().cpu()

                test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))

                wandb.log({"Train Image": test_image})

        return dice_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        probabilities = torch.sigmoid(self.forward(data))

        target = label[:, 0]
        keep_mask = label[:, 1]

        dice_loss = dice_loss_with_mask_batch(probabilities, target, keep_mask)

        iou, precision, recall, f1 = calculate_masked_metrics_batchwise(probabilities, target, keep_mask)

        self.update_unetr_validation_metrics(dice_loss, iou, precision, recall, f1)

        if batch_idx == 5 or batch_idx == 10:
            with torch.no_grad():
                combined = torch.cat([probabilities[0], target[0], keep_mask[0]], dim=1)
                grid = make_grid(combined).detach().cpu()

                test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))

                wandb.log({"Validation Image": test_image})

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
