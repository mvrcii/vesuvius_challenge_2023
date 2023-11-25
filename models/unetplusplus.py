import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss, DiceLoss
import torch

from models.abstract_model import AbstractVesuvLightningModule


class UnetPlusPlusModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        encoder_name, encoder_weights = cfg.encoder_name_and_weights

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,  # Encoders with weights https://smp.readthedocs.io/en/latest/encoders.html
            encoder_weights=encoder_weights,
            in_channels=cfg.in_chans,
            classes=1,
        )

        self.loss_fn = DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # Get normalization params
        params = smp.encoders.get_preprocessing_params(encoder_name)

        std = torch.tensor(params["std"])
        std = torch.cat((std, std[0].unsqueeze(0)), 0)
        self.register_buffer("std", std.view(1, 4, 1, 1))

        mean = torch.tensor(params["mean"])
        mean = torch.cat((mean, mean[0].unsqueeze(0)), 0)
        self.register_buffer("mean", mean.view(1, 4, 1, 1))

    def forward(self, x):
        # Normalize image here
        x = (x - self.mean) / self.std
        output = self.model(x)

        return output

    def training_step(self, batch, batch_idx):
        batch, batch_idx = self.assert_dims(batch, batch_idx)

        image, target = batch
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, target)

        super().update_training_metrics(loss)

    def validation_step(self, batch, batch_idx):
        batch, batch_idx = self.assert_dims(batch, batch_idx)

        image, target = batch
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, target)

        super().update_validation_metrics(loss, logits_mask, target)

    @staticmethod
    def assert_dims(batch, batch_idx):
        data, target = batch

        # Assertions
        assert data.ndim == 4
        h, w = data.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        # assert target.ndim == 4
        assert target.max() <= 1.0 and target.min() >= 0

        return (data, target), batch_idx
