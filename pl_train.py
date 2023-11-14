from datetime import datetime

from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from lightning_fabric.accelerators import find_usable_cuda_devices

from conf import CFG
from pl_segformer_datamodule import SegFormerDataModule
from pl_segformer_lightning import SegFormerLightningModule


def main():
    seed_value = 42
    seed_everything(seed_value)

    wandb_logger = WandbLogger(project="Kaggle1stReimp", entity="wuesuv")
    wandb_run_name = wandb_logger.experiment.name
    timestamp = datetime.now().strftime("%y%m%d-%H%M")  # Format: YYMMDD-HHMM

    model = SegFormerLightningModule()
    data_module = SegFormerDataModule()

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{timestamp}-{wandb_run_name}",
        filename="best-checkpoint-{epoch}-{val_iou:.2f}",
        save_top_k=1,
        monitor="val_iou",
        mode="max",  # IoU should be maximized
        every_n_epochs=1,  # Save a checkpoint every epoch
    )

    trainer = Trainer(
        max_epochs=CFG.epochs,
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback],
        accelerator="auto",
        devices=find_usable_cuda_devices(),  # all available gpus
        enable_progress_bar=True
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
