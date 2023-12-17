import torch
from torch import nn


class BinaryDiceLoss(nn.Module):

    def __init__(
            self,
            batch_dice: bool = False,
            from_logits: bool = True,
            log_loss: bool = False,
            smooth: float = 0.0,
            eps: float = 1e-7,
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
