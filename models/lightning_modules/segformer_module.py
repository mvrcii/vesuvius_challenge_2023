import torch
import wandb
from einops import rearrange
from torchvision.utils import make_grid
from transformers import SegformerForSemanticSegmentation

from models.lightning_modules.abstract_module import AbstractLightningModule


class SegformerModule(AbstractLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=cfg.from_pretrained,
            ignore_mismatched_sizes=True,
            num_labels=1,
            num_channels=cfg.in_chans,
        )

        self.load_weights()

    def forward(self, x):
        output = self.model(x.float())
        output = rearrange(output.logits, 'b 1 h w -> b h w')
        return output

    def training_step(self, batch, batch_idx):
        data, y_true = batch
        y_pred = self.forward(data)

        # Calculate individual losses, store them as (name, weight, value) tuples in list
        losses = [(name, weight, loss_function(y_pred, y_true.float())) for (name, weight, loss_function) in
                  self.loss_functions]
        total_loss = sum([weight * value for (_, weight, value) in losses])
        losses.append(("total", 1.0, total_loss))

        self.update_training_metrics(losses=losses)
        self.train_step += 1

        if batch_idx % 100 == 0 and self.trainer.is_global_zero:
            with torch.no_grad():
                probs = torch.sigmoid(y_pred)
                combined = torch.cat([probs[0], y_true[0]], dim=1)
                grid = make_grid(combined).detach().cpu()
                test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))
                wandb.log({"Train Image": test_image})

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, y_true = batch
        y_pred = self.forward(data)
        probs = torch.sigmoid(y_pred)

        # Calculate individual losses, store them as (name, weight, value) tuples in list
        losses = [(name, weight, loss_function(y_pred, y_true.float())) for (name, weight, loss_function) in
                  self.loss_functions]

        # Combine individual losses based on their weight
        total_loss = sum([weight * value for (_, weight, value) in losses])

        # Append total loss to list which is being logged
        losses.append(("total", 1.0, total_loss))

        # Combine the losses
        self.update_validation_metrics(losses=losses, output_logits=y_pred, target=y_true)

        if batch_idx == 5 and self.trainer.is_global_zero:
            with torch.no_grad():
                combined = torch.cat([probs[0], y_true[0]], dim=1)
                grid = make_grid(combined).detach().cpu()
                test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))
                wandb.log({"Validation Image": test_image})

    def update_training_metrics(self, losses):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for (name, weight, value) in losses:
            if name == 'total':
                self.log(f'train_loss', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            else:
                self.log(f'train_loss_{name}', value * weight, on_step=False, on_epoch=True, prog_bar=False,
                         sync_dist=True)

    def update_validation_metrics(self, losses, output_logits, target):
        if target.ndim != output_logits.ndim and target.ndim == 3 and output_logits.ndim == 4:
            target = target.unsqueeze(1)
        target = target.int()

        for (name, weight, value) in losses:
            if name == 'total':
                self.log(f'val_loss', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            else:
                self.log(f'val_loss_{name}', value * weight, on_step=False, on_epoch=True, prog_bar=False,
                         sync_dist=True)

        self.log('val_accuracy', self.accuracy(output_logits, target), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.precision(output_logits, target), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.recall(output_logits, target), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.f1(output_logits, target), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.iou(output_logits, target), on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('val_dice_coefficient', self.dice_coefficient(torch.sigmoid(output_logits), target), on_step=False,
                 on_epoch=True,
                 prog_bar=True, sync_dist=True)
