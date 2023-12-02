import torch
from einops import rearrange
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics.classification import (BinaryF1Score, BinaryPrecision, BinaryRecall,
                                         BinaryAccuracy, BinaryJaccardIndex as IoU)

from losses.bce_with_logits_with_label_smoothing import BCEWithLogitsLossWithLabelSmoothing
from util.losses import BinaryDiceLoss


class AbstractVesuvLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.epochs = cfg.epochs
        self.lr = cfg.lr
        self.eta_min = cfg.eta_min
        self.weight_decay = cfg.weight_decay
        self.optimizer = cfg.optimizer

        # False Negatives (FNs) are twice as impactful on the loss as False Positives (FPs)
        # pos_weight = torch.tensor([cfg.pos_weight]).to(self.device)

        self.bce_loss = BCEWithLogitsLossWithLabelSmoothing(label_smoothing=cfg.label_smoothing)
        self.dice_loss = BinaryDiceLoss(from_logits=True)

        self.f1 = BinaryF1Score()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.iou = IoU()

    def configure_optimizers(self):
        if self.optimizer == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError()

        if self.epochs == -1:
            # Set T_0 to a reasonable value based on your dataset and model
            # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)
            scheduler = StepLR(
                optimizer,
                step_size=2,
                gamma=0.99
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
        data, target = batch
        output = self.forward(data)

        # Compute both BCE loss (with label smoothing) and Dice loss
        bce_loss = self.bce_loss(output, target.float())
        dice_loss = self.dice_loss(torch.sigmoid(output), target.float())

        # Combine the losses
        total_loss = bce_loss + dice_loss

        self.update_training_metrics(loss=total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        # Compute both BCE loss (with label smoothing) and Dice loss
        bce_loss = self.bce_loss(output, target.float())
        dice_loss = self.dice_loss(torch.sigmoid(output), target.float())

        # Combine the losses
        total_loss = bce_loss + dice_loss

        self.update_validation_metrics(loss=total_loss, output_logits=output, target=target)

    def update_training_metrics(self, loss):
        """
        Update and log training metrics.

        This function logs the learning rate and training loss. The metrics are logged at the end of each epoch.

        :param loss: The loss value computed during the training phase.
        """
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def update_validation_metrics(self, loss, output_logits, target):
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

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', self.accuracy(output_logits, target), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.precision(output_logits, target), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.recall(output_logits, target), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.f1(output_logits, target), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.iou(output_logits, target), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
