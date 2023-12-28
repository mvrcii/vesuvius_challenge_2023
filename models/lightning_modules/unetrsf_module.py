import torch
import wandb
from torchvision.utils import make_grid

from models.architectures.unetr_segformer import UNETR_Segformer
from models.lightning_modules.abstract_module import AbstractLightningModule


def calculate_masked_metrics_batchwise(epsilon, outputs, labels, mask):
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
            true_positives + false_positives + false_negatives + epsilon)  # Added epsilon for numerical stability
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)  # Added epsilon for F1 calculation

    return iou.mean(), precision.mean(), recall.mean(), f1.mean()


class UNETR_SFModule(AbstractLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.epsilon = cfg.epsilon

        self.model = UNETR_Segformer(cfg=cfg)
        self.load_weights()

    def training_step(self, batch, batch_idx):
        data, label = batch
        y_true = label[:, 0]
        y_mask = label[:, 1]

        if torch.isnan(data).any() or torch.isinf(data).any():
            print("Warning: Input Data is nan:", data)

        if torch.isnan(y_true).any():
            print("Warning: Label is nan:", y_true)

        if torch.isnan(y_mask).any():
            print("Warning: Ignore Mask is nan:", y_mask)

        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        if torch.isnan(logits).any():
            print("Warning: Logits are nan:", logits)

        total_loss, losses = self.calculate_masked_weighted_loss(logits, y_true, y_mask)
        self.log_losses_to_wandb(losses, 'train')

        if torch.isnan(total_loss).any():
            print("Warning: Total train loss is nan:", total_loss)

        self.update_unetr_training_metrics(total_loss)
        self.train_step += 1

        if batch_idx == 5 and self.trainer.is_global_zero:
            with torch.no_grad():
                combined = torch.cat([y_pred[0], y_true[0], y_mask[0]], dim=1)
                grid = make_grid(combined).detach().cpu()
                test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))
                wandb.log({"Train Image": test_image})

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        y_true = label[:, 0]
        y_mask = label[:, 1]

        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        total_loss, losses = self.calculate_masked_weighted_loss(logits, y_true, y_mask)
        self.log_losses_to_wandb(losses, 'val')

        iou, precision, recall, f1 = calculate_masked_metrics_batchwise(self.epsilon, y_pred, y_true, y_mask)
        self.update_unetr_validation_metrics(total_loss, iou, precision, recall, f1)

        if batch_idx % 20 == 0 and self.trainer.is_global_zero:
            with torch.no_grad():
                combined = torch.cat([y_pred[0], y_true[0], y_mask[0]], dim=1)
                grid = make_grid(combined).detach().cpu()
                test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))
                wandb.log({"Validation Image": test_image})

    def calculate_masked_weighted_loss(self, logits, y_true, y_mask):
        losses = [(name, weight, loss_function(logits, y_true.float(), y_mask)) for (name, weight, loss_function) in
                  self.loss_functions]
        total_loss = sum([weight * value for (_, weight, value) in losses])
        losses.append(("total", 1.0, total_loss))

        return total_loss, losses

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
