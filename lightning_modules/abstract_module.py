import torch
import torch.nn.functional as F
from bitsandbytes.optim import Adam8bit
from einops import rearrange
from lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import (BinaryF1Score, BinaryPrecision, BinaryRecall,
                                         BinaryAccuracy, BinaryAUROC, BinaryJaccardIndex as IoU, BinaryAveragePrecision)


from util.losses import BinaryDiceLoss


class BCEWithLogitsLossWithLabelSmoothing(torch.nn.Module):
    def __init__(self, label_smoothing=0.1, pos_weight=None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(inputs, targets_smooth, pos_weight=self.pos_weight)


class AbstractVesuvLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.epochs = cfg.epochs
        self.lr = cfg.lr
        self.eta_min = cfg.eta_min
        self.weight_decay = cfg.weight_decay
        self.optimizer = cfg.optimizer

        # False Negatives (FNs) are twice as impactful on the loss as False Positives (FPs)
        pos_weight = torch.tensor([cfg.pos_weight]).to(device='cuda')

        self.bce_loss = BCEWithLogitsLossWithLabelSmoothing(label_smoothing=cfg.label_smoothing, pos_weight=pos_weight)
        self.dice_loss = BinaryDiceLoss(from_logits=True)

        self.f1 = BinaryF1Score()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.auc = BinaryAUROC(thresholds=None)
        self.iou = IoU()
        self.map = BinaryAveragePrecision()

    def configure_optimizers(self):
        if self.optimizer == 'adamw':
            optimizer = Adam8bit(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            # optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError()

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

        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        # Compute both BCE loss (with label smoothing) and Dice loss
        # bce_loss = self.bce_loss(output, target.float())
        # dice_loss = self.dice_loss(torch.sigmoid(output), target.float())

        # Combine the losses
        # total_loss = bce_loss + dice_loss

        # Update metrics
        # self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log('val_accuracy', self.accuracy(output, target.int()), on_step=False, on_epoch=True, prog_bar=False)
        # self.log('val_precision', self.precision(output, target.int()), on_step=False, on_epoch=True, prog_bar=False)
        # self.log('val_recall', self.recall(output, target.int()), on_step=False, on_epoch=True, prog_bar=False)

        self.log('val_f1', self.f1(output, target.int()), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auc', self.auc(output, target.int()), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.iou(output, target.int()), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_map', self.map(output, target.int()), on_step=False, on_epoch=True, prog_bar=False)