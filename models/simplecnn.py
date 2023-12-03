import torch
import torch.nn as nn

from models.abstract_model import AbstractVesuvLightningModule


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.upconv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(4, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            ResidualBlock(64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            ResidualBlock(128),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            UpConvBlock(128, 64),
            ConvBlock(64, 64),
            UpConvBlock(64, 32),
            ConvBlock(32, 32),
            UpConvBlock(32, 16),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze()


class SimpleCNNModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.model = ComplexCNN()

    def forward(self, x):
        output = self.model(x.float())

        return output

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        # Compute both BCE loss (with label smoothing) and Dice loss
        bce_loss = self.bce_loss(output, target.float())
        dice_loss = self.dice_loss(torch.sigmoid(output), target.float())

        # Combine the losses
        total_loss = bce_loss + dice_loss

        self.update_validation_metrics(loss=total_loss, output_logits=output, target=target)
