import os
from datetime import datetime

import numpy as np
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from config_handler import Config
from data_modules.efficientnetv2 import EfficientNetV2DataModule
from models.efficientnetv2 import EfficientNetV2Module
from train import get_sys_args, log_wandb_hyperparams


def load_norm_params(cfg):
    stats = np.load(os.path.join(cfg.dataset_target_dir, "stats.npz"))
    mean = stats['mean']
    std = stats['std']

    return mean, std


def main():
    config_path = get_sys_args()
    config = Config.load_from_file(config_path)

    if config.seed != -1:
        seed_everything(config.seed)

    wandb_logger = WandbLogger(project="Kaggle1stReimp", entity="wuesuv")
    log_wandb_hyperparams(config=config, wandb_logger=wandb_logger)

    # Model name related stuff
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    model_name = config.model_name if hasattr(config, 'model_name') else "default_model"
    wandb_generated_name = wandb_logger.experiment.name
    model_run_name = f"{wandb_generated_name}-{model_name}-{timestamp}"
    wandb_logger.experiment.name = model_run_name
    model_run_dir = os.path.join(config.work_dir, "checkpoints", model_run_name)

    model = EfficientNetV2Module(cfg=config)
    data_module = EfficientNetV2DataModule(cfg=config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_run_dir,
        filename="best-checkpoint-{epoch}-{val_mse_metric:.2f}",
        save_top_k=1,
        save_weights_only=True,
        monitor="val_mse_metric",
        mode="max",
        every_n_epochs=1
    )

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        accelerator="auto",
        enable_progress_bar=True,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_callback],
        devices=1,
    )

    trainer.fit(model, data_module)

    os.makedirs(model_run_dir, exist_ok=True)
    config.save_to_file(model_run_dir)


if __name__ == '__main__':
    main()
