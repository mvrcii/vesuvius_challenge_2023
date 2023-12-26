import torch
from torch.nn import BCEWithLogitsLoss
from torchmetrics import MeanSquaredError

from models.losses.binary_dice_loss import BinaryDiceLoss, MaskedBinaryDiceLoss
from models.losses.entropy_loss import EntropyLoss
from models.losses.focal_loss import FocalLoss, MaskedFocalLoss


def get_loss_functions(cfg):
    assert len(cfg.losses) > 0, "You must specify a loss type in your config!!!"

    focal_gamma = getattr(cfg, 'focal_gamma', 2)
    focal_alpha = getattr(cfg, 'focal_alpha', 0.25)

    available_losses = {"dice": BinaryDiceLoss(from_logits=True),
                        "bce": BCEWithLogitsLoss(),
                        "focal": FocalLoss(gamma=focal_gamma, alpha=focal_alpha),
                        "entropy": EntropyLoss(),
                        "mse": MeanSquaredError()}

    if torch.cuda.is_available():
        available_losses = {k: v.to('cuda') for k, v in available_losses.items()}

    return [(name, weight, available_losses[name]) for (name, weight) in cfg.losses]
