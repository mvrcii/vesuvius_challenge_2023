import torch


class BinaryJaccardIndex:
    def __init__(self):
        pass

    def calculate(self, logits, targets):
        probabilities = torch.sigmoid(logits)

        # Convert probabilities to binary predictions
        predictions = probabilities > 0.5

        # Calculate intersection and union
        intersection = (predictions & targets).float().sum((1, 2))  # Sum over height and width dimensions
        union = (predictions | targets).float().sum((1, 2))  # Sum over height and width dimensions

        # Compute the IoU and avoid division by zero
        iou = intersection / union.clamp(min=1e-6)

        mean_iou = torch.mean(iou)

        return mean_iou
