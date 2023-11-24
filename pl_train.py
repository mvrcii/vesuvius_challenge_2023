import os
import sys
import types
import warnings
from datetime import datetime

import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer

from config_handler import Config
from lightning_modules.abstract_module_test import AbstractVesuvLightningModuleTest
from lightning_modules.cnn3d_segformer_module import CNN3D_SegformerModule
from lightning_modules.segformer_module import SegformerModule
from pl_segformer_datamodule import SegFormerDataModule
from util.train_utils import get_device_configuration

torch.set_float32_matmul_precision('medium')

warnings.filterwarnings("ignore",
                        message="Some weights * were not initialized from the model checkpoint")
warnings.filterwarnings("ignore",
                        message="You should probably TRAIN this model on a down-stream task")

warnings.simplefilter("ignore", category=Warning)


def get_sys_args():
    if len(sys.argv) < 2:
        print("Usage: python pl_train.py <config_path>")
        sys.exit(1)

    return sys.argv[1]


def get_model(config):
    architecture = config.architecture

    if architecture == 'segformer':
        return SegformerModule(cfg=config)
    elif architecture == 'segformer-test':
        return AbstractVesuvLightningModuleTest(cfg=config)
    elif architecture == 'cnn3d_segformer':
        return CNN3D_SegformerModule(cfg=config)
    else:
        print("Invalid architecture:", architecture)
        sys.exit(1)


def log_wandb_hyperparams(config, wandb_logger):
    config_dict = vars(config)  # Convert config object to a dictionary

    # Remove non-serializable items
    cleaned_config = {k: v for k, v in config_dict.items() if not isinstance(v, types.ModuleType)}

    # Log the cleaned hyperparameters
    wandb_logger.log_hyperparams(cleaned_config)


def main():
    config_path = get_sys_args()
    config = Config.load_from_file(config_path)

    seed_value = config.seed
    seed_everything(seed_value)

    wandb_logger = WandbLogger(project="Kaggle1stReimp", entity="wuesuv")

    log_wandb_hyperparams(config=config, wandb_logger=wandb_logger)

    # Model name related stuff
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    model_name = config.model_name if hasattr(config, 'model_name') else "default_model"
    wandb_generated_name = wandb_logger.experiment.name
    model_run_name = f"{wandb_generated_name}-{model_name}-{timestamp}"
    wandb_logger.experiment.name = model_run_name
    model_run_dir = os.path.join("checkpoints", model_run_name)

    model = get_model(config=config)
    data_module = SegFormerDataModule(cfg=config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_run_dir,
        filename="best-checkpoint-{epoch}-{val_iou:.2f}--{val_auc:.2f}",
        save_top_k=1,
        monitor="val_iou",
        mode="max",
        every_n_epochs=1,
    )

    devices = get_device_configuration()
    print("Using Devices:", devices)

    trainer = Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval='epoch'), checkpoint_callback],
        accelerator="auto",
        devices=devices,
        enable_progress_bar=True,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, data_module)

    os.makedirs(model_run_dir, exist_ok=True)
    config.save_to_file(model_run_dir)


if __name__ == '__main__':
    main()
