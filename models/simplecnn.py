import torch.nn as nn

from models.abstract_model import AbstractVesuvLightningModule
import torch


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze()


class SimpleCNNModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.model = SimpleCNN()

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
