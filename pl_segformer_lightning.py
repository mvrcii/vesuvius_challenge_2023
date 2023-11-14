import torch
from einops import rearrange
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchmetrics import AveragePrecision
from torchmetrics.classification import (BinaryF1Score, BinaryPrecision, BinaryRecall,
                                         BinaryAccuracy, BinaryAUROC, BinaryJaccardIndex as IoU, BinaryAveragePrecision)
from transformers import SegformerForSemanticSegmentation
from util.losses import BinaryDiceLoss
from conf import CFG


class SegFormerLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            CFG.seg_pretrained,
            num_labels=1,
            num_channels=16,
            ignore_mismatched_sizes=True,
        )

        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = BinaryDiceLoss(from_logits=True)

        self.f1 = BinaryF1Score()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.auc = BinaryAUROC(thresholds=5)
        self.iou = IoU()
        self.map = BinaryAveragePrecision()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.999)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=True
        )
        # scheduler = CosineAnnealingLR(
        #     optimizer,
        #     T_max=100,
        #     eta_min=0
        # )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_iou"}

    def forward(self, x):
        output = self.model(x.float())
        output = rearrange(output.logits, 'b 1 h w -> b h w')

        return output

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        bce_loss = self.bce_loss(output, target.float())
        dice_loss = self.dice_loss(torch.sigmoid(output), target.float())
        total_loss = bce_loss + dice_loss

        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss', total_loss, on_epoch=True, prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        bce_loss = self.bce_loss(output, target.float())
        dice_loss = self.dice_loss(torch.sigmoid(output), target.float())
        total_loss = bce_loss + dice_loss

        # Update metrics
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.accuracy(output, target.int()), on_epoch=True, prog_bar=False)
        self.log('val_precision', self.precision(output, target.int()), on_epoch=True, prog_bar=False)
        self.log('val_recall', self.recall(output, target.int()), on_epoch=True, prog_bar=False)
        self.log('val_f1', self.f1(output, target.int()), on_epoch=True, prog_bar=True)
        self.log('val_auc', self.auc(output, target.int()), on_epoch=True, prog_bar=True)
        self.log('val_iou', self.iou(output, target.int()), on_epoch=True, prog_bar=True)
        self.log('val_map', self.map(output, target.int()), on_epoch=True, prog_bar=False)
