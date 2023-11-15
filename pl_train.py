import os
from datetime import datetime

from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from lightning_fabric.accelerators import find_usable_cuda_devices

from conf import CFG
from create_dataset import build_k_fold_folder
from pl_segformer_datamodule import SegFormerDataModule
from pl_segformer_lightning import SegFormerLightningModule


def main():
    seed_value = CFG.seed
    seed_everything(seed_value)

    wandb_logger = WandbLogger(project="Kaggle1stReimp", entity="wuesuv")
    wandb_run_name = wandb_logger.experiment.name
    timestamp = datetime.now().strftime("%y%m%d-%H%M")  # Format: YYMMDD-HHMM

    k_fold = True

    # Fragment Parameters
    train_frag_ids = [2, 3, 4]
    val_frag_ids = [1]
    single_train_frag_id = 2
    img_patch_size = 512

    if k_fold:
        train_ids_str, val_ids_str = build_k_fold_folder(train_frag_ids, val_frag_ids)
        data_root_dir = os.path.join(CFG.data_root_dir, f'k_fold_{train_ids_str}_{val_ids_str}', str(img_patch_size))
    else:
        data_root_dir = os.path.join(CFG.data_root_dir, f'single_TF{single_train_frag_id}', str(img_patch_size))

    model = SegFormerLightningModule()
    data_module = SegFormerDataModule(data_root_dir=data_root_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{timestamp}-{wandb_run_name}",
        filename="best-checkpoint-{epoch}-{val_iou:.2f}",
        save_top_k=1,
        monitor="val_iou",
        mode="max",  # IoU should be maximized
        every_n_epochs=1,  # Save a checkpoint every epoch
    )
    print("GPU DEVICES", find_usable_cuda_devices())
    trainer = Trainer(
        max_epochs=CFG.epochs,
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback],
        accelerator="auto",
        precision=16,
        devices=find_usable_cuda_devices(),  # all available gpus
        enable_progress_bar=True
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
