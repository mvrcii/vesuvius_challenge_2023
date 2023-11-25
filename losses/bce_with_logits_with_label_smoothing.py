import torch
import torch.nn.functional as F


class BCEWithLogitsLossWithLabelSmoothing(torch.nn.Module):
    def __init__(self, label_smoothing=0.1, pos_weight=None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true):
        """
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        """
        y_true_smooth = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Allows y_true with a Tensor shape of (N, H, W) or (N, C, H, W)
        assert y_pred.ndim == 4
        if y_true_smooth.ndim != y_pred.ndim:
            y_true_smooth = y_true_smooth.unsqueeze(1)

        return F.binary_cross_entropy_with_logits(y_pred, y_true_smooth, pos_weight=self.pos_weight)
