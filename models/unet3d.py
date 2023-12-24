import torch
from lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchmetrics import MeanSquaredError
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall, AUROC

from models.architectures.pytorch3dunet.unet3d.model import get_model


class UNet3DEncoder(nn.Module):
    def __init__(self, cfg):
        super(UNet3DEncoder, self).__init__()
        unet3d = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": cfg.unet3d_out_channels,
                            "f_maps": 16, "num_groups": 8, "is_segmentation": False})
        self.encoders = unet3d.encoders

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x


class BinaryClassificationDecoder(nn.Module):
    def __init__(self):
        super(BinaryClassificationDecoder, self).__init__()

        self.flatten = nn.Flatten()

        # Define additional layers
        self.linear_layers = nn.Sequential(
            nn.Linear(128 * 1 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x


class UNET3D(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = UNet3DEncoder(cfg=cfg)
        self.decoder = BinaryClassificationDecoder()

    def forward(self, x):
        encoder_output = self.encoder(x)
        logits = self.decoder(encoder_output)
        return logits.squeeze(1)


class UNET3D_Module(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.epochs = cfg.epochs
        self.lr = cfg.lr
        self.eta_min = cfg.eta_min
        self.weight_decay = cfg.weight_decay
        self.optimizer = cfg.optimizer
        self.step_lr_steps = cfg.step_lr_steps
        self.step_lr_factor = cfg.step_lr_factor

        self.model = UNET3D(cfg=cfg)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.f1 = BinaryF1Score()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.mse = MeanSquaredError()
        self.auc = AUROC(task='binary')

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
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, y_true = batch
        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        # loss = self.loss_fn(input=y_pred, target=y_true)
        mse_loss = self.mse(y_pred, y_true)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log('train_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        data, y_true = batch
        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        loss = self.loss_fn(input=logits, target=y_true)
        mse_loss = self.mse(y_pred, y_true)

        auc = self.auc(torch.sigmoid(logits), y_true)

        self.log(f'val_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log('val_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_auc', auc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log(f'val_precision', self.precision(y_pred, y_true), on_step=False, on_epoch=True,
        #          prog_bar=True, sync_dist=True)
        # self.log(f'val_recall', self.recall(y_pred, y_true), on_step=False, on_epoch=True,
        #          prog_bar=True, sync_dist=True)
        # self.log(f'val_f1', self.f1(y_pred, y_true), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
