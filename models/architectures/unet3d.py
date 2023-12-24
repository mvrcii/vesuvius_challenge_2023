from torch import nn

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
