import torch
from torchmetrics import MeanSquaredError
from torchmetrics.classification import AUROC

from models.architectures.unet3d import UNET3D
from models.modules.abstract_module import AbstractLightningModule


class UNET3D_Module(AbstractLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.model = UNET3D(cfg=cfg)

        self.load_weights()

        self.mse = MeanSquaredError()
        self.auc = AUROC(task='binary')

    def training_step(self, batch, batch_idx):
        data, y_true = batch
        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        mse_loss = self.mse(y_pred, y_true)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        data, y_true = batch
        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        mse_loss = self.mse(y_pred, y_true)

        auc = self.auc(torch.sigmoid(logits), y_true)

        self.log(f'val_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_auc', auc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
