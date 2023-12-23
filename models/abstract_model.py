import torch
from einops import rearrange
from lightning import LightningModule
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR
from torchmetrics import Dice
from torchmetrics.classification import (BinaryF1Score, BinaryPrecision, BinaryRecall,
                                         BinaryAccuracy, BinaryJaccardIndex as IoU)
from torchvision.utils import make_grid

from losses.utils import get_loss_functions


class AbstractVesuvLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_step = 0
        self.epochs = cfg.epochs
        self.lr = cfg.lr
        self.eta_min = cfg.eta_min
        self.weight_decay = cfg.weight_decay
        self.optimizer = cfg.optimizer
        self.step_lr_steps = cfg.step_lr_steps
        self.step_lr_factor = cfg.step_lr_factor

        self.loss_functions = get_loss_functions(cfg)

        self.f1 = BinaryF1Score()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.iou = IoU()
        self.dice_coefficient = Dice(multiclass=False, threshold=0.5)

    def configure_optimizers(self):
        if self.optimizer == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError()

        if self.epochs == -1:
            scheduler = StepLR(
                optimizer,
                step_size=self.step_lr_steps,
                gamma=self.step_lr_factor
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=self.eta_min
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        output = self.model(x.float())
        output = rearrange(output.logits, 'b 1 h w -> b h w')

        return output

    def training_step(self, batch, batch_idx):
        data, y_true = batch
        y_pred = self.forward(data)

        # Calculate individual losses, store them as (name, weight, value) tuples in list
        losses = [(name, weight, loss_function(y_pred, y_true.float())) for (name, weight, loss_function) in
                  self.loss_functions]
        total_loss = sum([weight * value for (_, weight, value) in losses])
        losses.append(("total", 1.0, total_loss))

        self.update_training_metrics(losses=losses)
        self.train_step += 1

        if batch_idx % 100 == 0 and self.trainer.is_global_zero:
            with torch.no_grad():
                probs = torch.sigmoid(y_pred)
                combined = torch.cat([probs[0], y_true[0]], dim=1)
                grid = make_grid(combined).detach().cpu()
                test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))
                wandb.log({"Train Image": test_image})

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, y_true = batch
        y_pred = self.forward(data)
        probs = torch.sigmoid(y_pred)

        # Calculate individual losses, store them as (name, weight, value) tuples in list
        losses = [(name, weight, loss_function(y_pred, y_true.float())) for (name, weight, loss_function) in
                  self.loss_functions]

        # Combine individual losses based on their weight
        total_loss = sum([weight * value for (_, weight, value) in losses])

        # Append total loss to list which is being logged
        losses.append(("total", 1.0, total_loss))

        # Combine the losses
        self.update_validation_metrics(losses=losses, output_logits=y_pred, target=y_true)

        if batch_idx == 5 and self.trainer.is_global_zero:
            with torch.no_grad():
                combined = torch.cat([probs[0], y_true[0]], dim=1)
                grid = make_grid(combined).detach().cpu()
                test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))
                wandb.log({"Validation Image": test_image})

    def update_training_metrics(self, losses):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for (name, weight, value) in losses:
            if name == 'total':
                self.log(f'train_loss', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            else:
                self.log(f'train_loss_{name}', value * weight, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def update_validation_metrics(self, losses, output_logits, target):
        """
        Update and log validation metrics.

        This function logs validation loss and calculates metrics such as accuracy, precision, recall, F1 score, and
        IOU. These metrics are logged at the end of each epoch.

        :param loss:            The loss value computed during the validation phase.
        :param output_logits:   Tensor of shape (N, C, H, W).
        :param target:          Tensor of shape (N, H, W) or (N, C, H, W).
        """
        if target.ndim != output_logits.ndim and target.ndim == 3 and output_logits.ndim == 4:
            target = target.unsqueeze(1)
        target = target.int()

        for (name, weight, value) in losses:
            if name == 'total':
                self.log(f'val_loss', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            else:
                self.log(f'val_loss_{name}', value * weight, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.log('val_accuracy', self.accuracy(output_logits, target), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.precision(output_logits, target), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.recall(output_logits, target), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.f1(output_logits, target), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.iou(output_logits, target), on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('val_dice_coefficient', self.dice_coefficient(torch.sigmoid(output_logits), target), on_step=False,
                 on_epoch=True,
                 prog_bar=True, sync_dist=True)
