import os

import torch
from lightning import LightningModule
from segmentation_models_pytorch.utils.metrics import IoU
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics import Dice
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall

from models.losses.utils import get_loss_functions


class AbstractLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_step = 0
        self.loss_functions = get_loss_functions(cfg)
        self.f1 = BinaryF1Score()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.iou = IoU()
        self.dice_coefficient = Dice(multiclass=False, threshold=0.5)

    def load_weights(self):
        from_checkpoint = getattr(self.cfg, 'from_checkpoint', None)
        if from_checkpoint:
            checkpoint_root_path = os.path.join("checkpoints", self.cfg.from_checkpoint)
            checkpoint_files = [file for file in os.listdir(checkpoint_root_path) if file.startswith('best-checkpoint')]
            checkpoint_path = os.path.join(checkpoint_root_path, checkpoint_files[-1])
            print(checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint
            state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
            print("Loaded model from checkpoint:", self.cfg.from_checkpoint)
        else:
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

    def forward(self, x):
        return self.model(x)
