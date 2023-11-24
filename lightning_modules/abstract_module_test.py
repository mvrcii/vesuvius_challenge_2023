from torch.nn import BCEWithLogitsLoss
from einops import rearrange
from lightning import LightningModule
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import (BinaryF1Score)
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class AbstractVesuvLightningModuleTest(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.epochs = cfg.epochs
        self.lr = cfg.lr
        self.eta_min = cfg.eta_min
        self.optimizer = cfg.optimizer
        self.weight_decay = cfg.weight_decay

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=cfg.from_pretrained,
            ignore_mismatched_sizes=True,
            num_labels=1,
            num_channels=cfg.in_chans,
        )

        self.bce_loss = BCEWithLogitsLoss()
        self.f1 = BinaryF1Score()

    def configure_optimizers(self):
        if self.optimizer == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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

        # Compute both BCE loss
        bce_loss = self.bce_loss(output, target.float())

        # Log learning rate
        self.log('train_loss', bce_loss, on_step=True, prog_bar=False)

        return bce_loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        # Compute both BCE loss
        bce_loss = self.bce_loss(output, target.float())

        # Update metrics
        self.log('val_loss', bce_loss, on_step=True, prog_bar=True)
        self.log('val_f1', self.f1(output, target.int()), on_step=True, prog_bar=True)
