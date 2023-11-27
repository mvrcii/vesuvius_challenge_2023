import timm
from lightning import LightningModule
from torch import nn
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.regression import MeanSquaredError


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
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        self.mse_loss = MSELoss()
        self.mse_metric = MeanSquaredError()

    def forward(self, data):
        features = self.feature_extractor(data)
        output = self.classifier(features)
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
        labels = (labels >= 0.4).view(-1, 1).float()

        outputs = self.forward(images)

        mse_loss = self.mse_loss(outputs, labels).float()
        self.log("train_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.float()
        labels = (labels >= 0.4).view(-1, 1).float()

        outputs = self.forward(images)

        mse_loss = self.mse_loss(outputs, labels).float()
        self.log("train_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True)

        mse_loss = self.mse_loss(outputs, labels).float()
        mse_metric = self.mse_metric(outputs, labels)

        self.log('val_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse_metric', mse_metric, on_step=False, on_epoch=True, prog_bar=True)
