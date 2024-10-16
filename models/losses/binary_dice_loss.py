import sys

import torch
import torch.nn.functional as F
from torch import nn


class BinaryDiceLoss(nn.Module):

    def __init__(
            self,
            batch_dice: bool = False,
            from_logits: bool = True,
            log_loss: bool = False,
            smooth: float = 0.0,
            eps: float = 1e-3,
    ):
        """Implementation of Dice loss for binary image segmentation tasks

        Args:
            batch_dice: dice per sample and average or treat batch as a single volumetric sample (default)
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(BinaryDiceLoss, self).__init__()
        self.batch_dice = batch_dice
        self.from_logits = from_logits
        self.log_loss = log_loss
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)

        y_true = y_true.view(bs, -1)  # bs x num_elems
        y_pred = y_pred.view(bs, -1)  # bs x num_elems

        if self.batch_dice == True:
            intersection = torch.sum(y_pred * y_true)  # float
            cardinality = torch.sum(y_pred + y_true)  # float
        else:
            intersection = torch.sum(y_pred * y_true, dim=-1)  # bs x float
            cardinality = torch.sum(y_pred + y_true, dim=-1)  # bs x float

        dice_scores = (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps)
        if self.log_loss:
            losses = -torch.log(dice_scores.clamp_min(self.eps))
        else:
            losses = 1.0 - dice_scores
        return losses.mean()


class MaskedBinaryDiceLoss(nn.Module):
    def __init__(self, from_logits: bool = True, smoothing: float = 0.05, eps: float = 1e-7):
        super().__init__()
        self.from_logits = from_logits
        self.smoothing = smoothing
        self.eps = eps
        """Implementation of Dice loss for binary image segmentation tasks with mask

        Args:
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)
             - **y_mask** - torch.Tensor of shape (N, H, W)
        """

    def forward(self, y_pred_unmasked: torch.Tensor, y_true_unmasked: torch.Tensor,
                y_mask: torch.Tensor) -> torch.Tensor:
        assert y_true_unmasked.size(0) == y_pred_unmasked.size(0) == y_mask.size(0)

        # Mask prediction and ground truth
        y_pred = y_pred_unmasked * y_mask
        y_true = y_true_unmasked * y_mask

        y_true = y_true * (1 - self.smoothing) + 0.5 * self.smoothing

        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        intersection = (y_pred * y_true).sum(axis=(1, 2))
        union = (y_pred + y_true).sum(axis=(1, 2))

        losses = 1. - (2. * intersection) / union.clamp_min(self.eps)

        mean_loss = losses.mean()

        if torch.isnan(mean_loss).any():
            print("Warning: Mean Dice Loss is nan:", mean_loss)
            print("Warning: Losses:", losses)
            print("Warning: Union:", union)
            print("Warning: Intersection:", intersection)
            print("Warning: Torch Unique y_pred:", torch.unique(y_pred))
            print("Warning: Torch Unique y_true:", torch.unique(y_true))
            print("Warning: Torch Unique y_mask:", torch.unique(y_mask))
            sys.exit(1)

        return mean_loss
