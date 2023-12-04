import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss


def get_loss_function(cfg):
    available_losses = {"dice": BinaryDiceLoss(from_logits=True),
                        "bce": BCEWithLogitsLoss()}

    def loss_composite(output, target):
        return sum(available_losses[name](output, target) * weight for (name, weight) in cfg.losses)

    return loss_composite


# https://github.com/agaldran/lesion_losses_ood/blob/main/utils/losses.py
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


class MCCLoss(_Loss):
    def __init__(self, eps: float = 1e-5):
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.
        It only supports binary mode.

        Args:
            eps (float): Small epsilon to handle situations where all the samples in the dataset belong to one class

        Reference:
            https://github.com/kakumarabhishek/MCC-Loss
        """
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MCC loss

        Args:
            y_pred (torch.Tensor): model prediction of shape (N, H, W) or (N, 1, H, W)
            y_true (torch.Tensor): ground truth labels of shape (N, H, W) or (N, 1, H, W)

        Returns:
            torch.Tensor: loss value (1 - mcc)
        """

        bs = y_true.shape[0]

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss


class TopKLoss(nn.BCEWithLogitsLoss):
    """
    TopKLoss for 2d binary segmentation
    Requires specifying percentage of pixels kept
    Expects input and target of same shape, target can be int or float
    pos_weight (Tensor, optional) – a weight of positive examples. Must be a vector
    with length equal to the number of classes. A multiplier for class frequencies.

    l_bce = nn.BCEWithLogitsLoss()
    l_top10 = TopKLoss(k=10)
    l_top100 = TopKLoss(k=100)

    print(l_top10(input.squeeze(), target))
    print(l_bce(input.squeeze(), target))
    print(l_top100(input.squeeze(), target))

    See https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/TopK_loss.py
    for a more general implementation (multi-class, 3d)
    """

    def __init__(self, pos_weight=None, k=10):
        self.k = k
        super(TopKLoss, self).__init__(reduction='none', pos_weight=pos_weight)

    def forward(self, inp, target):
        res = super(TopKLoss, self).forward(inp, target)
        num_pixels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1,)), int(num_pixels * self.k / 100), sorted=False)
        return res.mean()


class FocalLoss(nn.Module):
    """
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25, (optional) weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore. Default = 0.25
    gamma: float = 2,  exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
    reduction: str = 'mean'

    See https://amaarora.github.io/2020/06/29/FocalLoss.html for an explanation
    """

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inp, target):
        return torchvision.ops.sigmoid_focal_loss(inp, target, self.alpha, self.gamma, self.reduction)
