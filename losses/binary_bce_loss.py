import torch
import torch.nn.functional as F
from torch import nn


class MaskedBinaryBCELoss(nn.Module):
    def __init__(self, from_logits: bool = True):
        super().__init__()
        self.from_logits = from_logits
        """Implementation of BCE loss for binary image segmentation tasks with mask

        Args:
            from_logits: If True, assumes input is raw logits
        Shape
             - **y_pred** - torch.Tensor of shape (N, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)
             - **y_mask** - torch.Tensor of shape (N, H, W)
        """

    def forward(self, y_pred_unmasked: torch.Tensor, y_true_unmasked: torch.Tensor,
                y_mask: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            y_pred_unmasked = torch.sigmoid(y_pred_unmasked)

        losses = F.binary_cross_entropy(input=y_pred_unmasked, target=y_true_unmasked, reduction='none')
        losses = losses * y_mask  # Apply mask to losses

        # Normalizing by number of unmasked elements
        return losses.sum() / y_mask.sum()
