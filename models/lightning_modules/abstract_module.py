import os

import torch
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from models.losses.utils import get_loss_functions


class AbstractLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_step = 0

        # Unpack config parameters since self.cfg = cfg leads to pickle issues
        self.epochs = getattr(cfg, 'epochs', -1)
        self.lr = getattr(cfg, 'lr', 1e-4)
        self.eta_min = getattr(cfg, 'eta_min', 1e-7)
        self.weight_decay = getattr(cfg, 'weight_decay', 0.001)
        self.optimizer = getattr(cfg, 'optimizer', 'adamw')
        self.step_lr_steps = getattr(cfg, 'step_lr_steps', 1)
        self.step_lr_factor = getattr(cfg, 'step_lr_factor', 0.98)
        self.from_checkpoint = getattr(cfg, 'from_checkpoint', None)
        self.from_pretrained = getattr(cfg, 'from_pretrained', None)

        self.loss_functions = get_loss_functions(cfg)

    def load_weights(self):
        if self.from_checkpoint:
            checkpoint_root_path = os.path.join("checkpoints", self.cfg.from_checkpoint)
            checkpoint_files = [file for file in os.listdir(checkpoint_root_path) if file.startswith('best-checkpoint')]
            checkpoint_path = os.path.join(checkpoint_root_path, checkpoint_files[-1])
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint
            state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
            print("Loaded model from checkpoint:", self.cfg.from_checkpoint)

        if self.from_pretrained:
            print("Loaded model from pretrained:", self.cfg.from_pretrained)

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

    def calculate_weighted_loss(self, y_pred, y_true):
        losses = [(name, weight, loss_function(y_pred, y_true.float())) for (name, weight, loss_function) in
                  self.loss_functions]
        total_loss = sum([weight * value for (_, weight, value) in losses])
        losses.append(("total", 1.0, total_loss))

        return total_loss, losses

    def log_losses_to_wandb(self, losses, prefix):
        for (name, weight, value) in losses:
            if name == 'total':
                self.log(f'{prefix}_loss', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            else:
                self.log(f'{prefix}_loss_{name}', value * weight, on_step=False, on_epoch=True, prog_bar=False,
                         sync_dist=True)

    def forward(self, x):
        return self.model(x)
