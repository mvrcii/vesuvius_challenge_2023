import torch
from torchmetrics.classification import AUROC

from models.architectures.vit3d import ViT3D
from models.lightning_modules.abstract_module import AbstractLightningModule


class Vit3D_Module(AbstractLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        image_size = (48, 48)
        layers = 12
        num_classes = 1
        channels = 1

        # working example
        image_patch_size = (8, 8)  # Size of patches each frame is divided into (height, width)
        frame_patch_size = 3  # Number of frames grouped into a single temporal patch
        dim = 512  # Dimensionality of token embeddings in the transformer
        depth = 6  # Number of layers (blocks) in the transformer
        heads = 8  # Number of attention heads in each transformer layer
        mlp_dim = 1024  # Dimensionality of the feedforward network in each transformer layer
        dim_head = 64  # Dimensionality of each attention head
        dropout = 0.1  # Dropout rate used in attention and feedforward networks
        emb_dropout = 0.1  # Dropout rate for token

        # Instantiate the model
        self.model = ViT3D(
            image_size=image_size,
            image_patch_size=image_patch_size,
            frames=layers,
            frame_patch_size=frame_patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool='cls',  # Pooling method ('cls' for class token, 'mean' for mean pooling)
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

        self.load_weights()

        self.auc = AUROC(task='binary')

    def training_step(self, batch, batch_idx):
        data, y_true = batch
        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        total_loss, losses = self.calculate_weighted_loss(y_pred=y_pred.squeeze(1), y_true=y_true)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_losses_to_wandb(losses, 'train')

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, y_true = batch
        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        _, losses = self.calculate_weighted_loss(y_pred=y_pred.squeeze(1), y_true=y_true)
        self.log_losses_to_wandb(losses, 'val')

        auc = self.auc(torch.sigmoid(logits), y_true)
        self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Calculate and log accuracy
        # Convert predictions and labels to binary using 0.5 threshold
        preds_binary = (y_pred > 0.5).float()
        labels_binary = (y_true > 0.5).float()
        accuracy = torch.mean((preds_binary == labels_binary).float())
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
