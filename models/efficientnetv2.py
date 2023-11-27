import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import BinaryAccuracy


class SuperduperultralightCNN(nn.Module):
    def _init_(self):
        super(SuperduperultralightCNN, self)._init_()
        # Single convolutional layer with minimal channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)  # Larger stride for more reduction

        # Extremely simplified fully connected layer
        self.fc = nn.Linear(4 * 16 * 16, 1)  # Large pooling reduces to 16x16

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 4 * 16 * 16)
        x = torch.sigmoid(self.fc(x))

        return x


class SuperduperlightCNN(nn.Module):
    def _init_(self):
        super(SuperduperlightCNN, self)._init_()
        # Single convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Simple fully connected layer
        self.fc = nn.Linear(8 * 32 * 32, 1)  # After one pooling, image size is 32x32

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 8 * 32 * 32)
        x = torch.sigmoid(self.fc(x))

        return x


class SuperlightCNN(nn.Module):
    def _init_(self):
        super(SuperlightCNN, self)._init_()
        # Further reduced convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Very simple fully connected layer
        self.fc = nn.Linear(16 * 16 * 16, 1)  # After two poolings, image size is 16x16

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 16 * 16)
        x = torch.sigmoid(self.fc(x))  # Sigmoid for binary classification

        return x


class LightCNN(nn.Module):
    def _init_(self):
        super(LightCNN, self)._init_()
        # Convolutional layers with reduced channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Simplified fully connected layer
        self.fc = nn.Linear(32 * 8 * 8, 1)  # Directly to output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 8 * 8)
        x = torch.sigmoid(self.fc(x))  # Sigmoid for binary classification

        return x


class SimpleCNN(nn.Module):
    def _init_(self):
        super(SimpleCNN, self)._init_()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 64x64 image pooled three times becomes 8x8
        self.fc2 = nn.Linear(128, 1)  # Binary classification

    def forward(self, x):
        # Apply convolutions and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 8 * 8)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Use sigmoid for binary classification

        return x


class EfficientNetV2Module(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.epochs = cfg.epochs
        self.eta_min = cfg.eta_min

        self.model = timm.create_model(cfg.from_pretrained,
                                       in_chans=cfg.in_chans,
                                       pretrained=True,
                                       num_classes=1)

        # Freeze the parameters in feature extractor
        layers = list(self.model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1280, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # self.model = SimpleCNN()

        self.loss = BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy()

    def forward(self, data):
        features = self.feature_extractor(data)
        output = self.classifier(features)
        # output = self.model(data)
        return output

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.eta_min
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.float()
        labels = (labels >= 0.5).view(-1, 1).float()

        outputs = self.forward(images)

        loss = self.loss(outputs, labels).float()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.float()
        labels = (labels >= 0.5).view(-1, 1).float()

        outputs = self.forward(images)

        loss = self.loss(outputs, labels).float()

        acc = self.accuracy(outputs, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
