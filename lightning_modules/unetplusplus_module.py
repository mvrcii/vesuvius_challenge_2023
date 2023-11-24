import segmentation_models_pytorch as smp
import torch

from lightning_modules.abstract_module import AbstractVesuvLightningModule


class UnetPlusPlusModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=cfg.in_chans,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )

        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        params = smp.encoders.get_preprocessing_params("resnet34")

        std = torch.tensor(params["std"])
        std = torch.cat((std, std[0].unsqueeze(0)), 0)
        self.register_buffer("std", std.view(1, 4, 1, 1))

        mean = torch.tensor(params["mean"])
        mean = torch.cat((mean, mean[0].unsqueeze(0)), 0)
        self.register_buffer("mean", mean.view(1, 4, 1, 1))

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        # Compute Dice loss
        loss = self.dice_loss(torch.sigmoid(output), target.float())

        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        target = target.unsqueeze(1)

        # Shape of image should be (b, c, h, w)
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be (b, c, h, w)
        assert target.ndim == 4

        # Check that mask values in between 0 and 1 for binary segmentation
        assert target.max() <= 1.0 and target.min() >= 0

        output = self.forward(image)
        loss = self.dice_loss(torch.sigmoid(output), target.float())

        # Update metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', self.accuracy(output, target.int()), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_precision', self.precision(output, target.int()), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_recall', self.recall(output, target.int()), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_f1', self.f1(output, target.int()), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auc', self.auc(output, target.int()), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.iou(output, target.int()), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_map', self.map(output, target.int()), on_step=False, on_epoch=True, prog_bar=False)
