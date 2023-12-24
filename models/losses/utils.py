from torch.nn import BCEWithLogitsLoss

from models.losses.binary_dice_loss import BinaryDiceLoss
from models.losses.entropy_loss import EntropyLoss
from models.losses.focal_loss import FocalLoss


def get_loss_functions(cfg):
    available_losses = {"dice": BinaryDiceLoss(from_logits=True),
                        "bce": BCEWithLogitsLoss(),
                        "focal": FocalLoss(gamma=5.0, alpha=0.75),
                        "entropy": EntropyLoss()}

    return [(name, weight, available_losses[name]) for (name, weight) in cfg.losses]
