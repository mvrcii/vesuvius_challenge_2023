import numpy as np
import torch
from lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.regression import MeanSquaredError
from transformers import ResNetForImageClassification, ResNetConfig


class ResNet50Module(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.epochs = cfg.epochs
        self.lr = cfg.lr
        self.eta_min = cfg.eta_min
        self.weight_decay = cfg.weight_decay
        self.optimizer = cfg.optimizer

        configuration = ResNetConfig(num_channels=cfg.in_chans, num_labels=1)
        self.model = ResNetForImageClassification(configuration)

        self.mse_loss = MSELoss()
        self.mse_metric = MeanSquaredError()

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.eta_min
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        data, labels = batch
        data = data.unsqueeze(1)
        data = data.float()
        labels = labels.float()

        logits = self.model(data).logits
        preds = torch.sigmoid(logits).squeeze()

        mse_loss = self.mse_loss(preds, labels).float()

        self.log("train_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        data = data.unsqueeze(1)
        data = data.float()
        labels = labels.float()

        logits = self.model(data).logits
        preds = torch.sigmoid(logits).squeeze()

        mse_loss = self.mse_loss(preds, labels).float()
        mse_metric = self.mse_metric(preds, labels)

        self.log('val_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse_metric', mse_metric, on_step=False, on_epoch=True, prog_bar=True)

