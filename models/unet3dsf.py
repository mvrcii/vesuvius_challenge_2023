import os

import torch
import wandb
from torchvision.utils import make_grid

from losses.binary_bce_loss import MaskedBinaryBCELoss
from losses.binary_dice_loss import MaskedBinaryDiceLoss
from losses.focal_loss import MaskedFocalLoss
from models.abstract_model import AbstractVesuvLightningModule
from models.architectures.unet3d_segformer import UNET3D_Segformer


def calculate_masked_metrics_batchwise(outputs, labels, mask):
    # Ensure batch dimension is maintained during operations
    outputs = (outputs > 0.5).float()
    batch_size = outputs.size(0)

    # Flatten tensors except for the batch dimension
    outputs_flat = (outputs * mask).view(batch_size, -1)
    labels_flat = (labels * mask).view(batch_size, -1)

    # Calculate True Positives, False Positives, and False Negatives for each batch
    true_positives = (outputs_flat * labels_flat).sum(dim=1)
    false_positives = (outputs_flat * (1 - labels_flat)).sum(dim=1)
    false_negatives = ((1 - outputs_flat) * labels_flat).sum(dim=1)

    # Calculate metrics for each batch
    iou = true_positives / (
            true_positives + false_positives + false_negatives + 1e-6)  # Added epsilon for numerical stability
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # Added epsilon for F1 calculation

    return iou.mean(), precision.mean(), recall.mean(), f1.mean()


# def load_test_image(cfg):
#     path = os.path.join("multilayer_approach/datasets/sanity/JETFIRE")
#     image = np.load(os.path.join(path, 'images', 'f20231005123336_ch30_17134_2873_17646_3385.npy'))
#     label = np.load(os.path.join(path, 'labels', 'f20231005123336_ch30_17134_2873_17646_3385.npy'))
#     label = np.unpackbits(label).reshape((2, cfg.label_size, cfg.label_size))  # 2, 128, 128
#
#     # Measure RAM usage of numpy arrays
#     image_size_bytes = sys.getsizeof(image)
#     label_size_bytes = sys.getsizeof(label)
#     print(f"Image size in RAM: {image_size_bytes} bytes")
#     print(f"Label size in RAM: {label_size_bytes} bytes")
#
#     label = label[0]  # 128, 128 now
#
#     label = torch.from_numpy(label).to(dtype=float16, device='cuda')
#     image = torch.from_numpy(image).to(dtype=float16, device='cuda')
#
#     image = image.unsqueeze(0)
#     print("Image Shape", image.shape)
#
#     pad_array = torch.zeros(1, 16 - image.shape[1], cfg.patch_size, cfg.patch_size).to(dtype=float16, device='cuda')
#     print("Pad Shape", pad_array.shape)
#
#     image = torch.cat([image, pad_array], dim=1)
#
#     # Measure VRAM usage
#     vram_usage_bytes = torch.cuda.memory_allocated()
#     print(f"VRAM used for image and label: {vram_usage_bytes} bytes")
#
#     return image, label


class UNET3D_SFModule(AbstractVesuvLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.cfg = cfg
        self.train_step = 0

        self.model = UNET3D_Segformer(cfg=cfg)

        self.focal_loss_fn = MaskedFocalLoss(gamma=0.5)
        self.dice_loss_fn = MaskedBinaryDiceLoss(from_logits=True)
        self.bce_loss_fn = MaskedBinaryBCELoss(from_logits=True)

        from_checkpoint = getattr(cfg, 'from_checkpoint', None)
        if from_checkpoint:
            checkpoint_root_path = os.path.join("checkpoints", cfg.from_checkpoint)
            checkpoint_files = [file for file in os.listdir(checkpoint_root_path) if file.startswith('best-checkpoint')]
            checkpoint_path = os.path.join(checkpoint_root_path, checkpoint_files[-1])

            checkpoint = torch.load(checkpoint_path)
            assert checkpoint

            state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
            print("Loaded model from checkpoint:", cfg.from_checkpoint)
        else:
            print("Loaded blank unetr and segformer from pretrained:", cfg.segformer_from_pretrained)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        y_true = label[:, 0]
        y_mask = label[:, 1]

        y_pred = self.forward(data)

        dice_loss = self.dice_loss_fn(y_pred, y_true, y_mask)
        focal_loss = self.focal_loss_fn(y_pred, y_true, y_mask)
        bce_loss = self.bce_loss_fn(y_pred, y_true, y_mask)

        self.log(f'train_dice_loss', dice_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train_focal_loss', focal_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train_bce_loss', bce_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        total_loss = dice_loss + bce_loss

        self.update_unetr_training_metrics(total_loss)
        self.train_step += 1

        if batch_idx % 100 == 0 and self.trainer.is_global_zero:
            with torch.no_grad():
                probs = torch.sigmoid(y_pred)
                combined = torch.cat([probs[0], y_true[0], y_mask[0]], dim=1)
                grid = make_grid(combined).detach().cpu()
                test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))
                wandb.log({"Train Image": test_image})

        return dice_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        y_true = label[:, 0]
        y_mask = label[:, 1]

        y_pred = self.forward(data)
        probs = torch.sigmoid(y_pred)

        dice_loss = self.dice_loss_fn(y_pred, y_true, y_mask)
        focal_loss = self.focal_loss_fn(y_pred, y_true, y_mask)
        bce_loss = self.bce_loss_fn(y_pred, y_true, y_mask)

        self.log(f'val_dice_loss', dice_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_focal_loss', focal_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_bce_loss', bce_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        total_loss = dice_loss + bce_loss

        iou, precision, recall, f1 = calculate_masked_metrics_batchwise(probs, y_true, y_mask)
        self.update_unetr_validation_metrics(total_loss, iou, precision, recall, f1)

        if batch_idx == 5 and self.trainer.is_global_zero:
            with torch.no_grad():
                combined = torch.cat([probs[0], y_true[0], y_mask[0]], dim=1)
                grid = make_grid(combined).detach().cpu()
                test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))
                wandb.log({"Validation Image": test_image})

    def on_after_backward(self):
        if self.trainer.global_step % 100 == 0:  # Log every 25 steps
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    wandb.log({f"grads/{name}": wandb.Histogram(param.grad.cpu().numpy())})

    def update_unetr_validation_metrics(self, loss, iou, precision, recall, f1):
        self.log(f'val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_precision', precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_recall', recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def update_unetr_training_metrics(self, loss):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log(f'train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
