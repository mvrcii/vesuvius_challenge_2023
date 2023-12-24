import torch
from torchmetrics.classification import AUROC

from models.architectures.unet3d import UNET3D
from models.lightning_modules.abstract_module import AbstractLightningModule


class UNET3D_Module(AbstractLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.model = UNET3D(cfg=cfg)
        self.load_weights()

        self.auc = AUROC(task='binary')

    def training_step(self, batch, batch_idx):
        data, y_true = batch
        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        total_loss, losses = self.calculate_weighted_loss(y_pred=y_pred, y_true=y_true)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_losses_to_wandb(losses, 'train')

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, y_true = batch
        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        _, losses = self.calculate_weighted_loss(y_pred=y_pred, y_true=y_true)
        self.log_losses_to_wandb(losses, 'val')

        auc = self.auc(torch.sigmoid(logits), y_true)
        self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
