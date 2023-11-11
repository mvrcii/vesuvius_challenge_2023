import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def calculate_incremental_metrics(metric_accumulator, y_true, y_pred, threshold):
    """
    Update the metric accumulator with new batch metrics.
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > threshold).astype(np.int32)

    # Update confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true.reshape(-1), y_pred_binary.reshape(-1)).ravel()
    metric_accumulator['tp'] += tp
    metric_accumulator['fp'] += fp
    metric_accumulator['fn'] += fn
    metric_accumulator['tn'] += tn

    # Update AUC accumulator
    metric_accumulator['auc'] += roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))
    metric_accumulator['count'] += 1

def calculate_final_metrics(metric_accumulator):
    """
    Calculate final metrics based on the aggregated values.
    """
    tp, fp, fn, tn = metric_accumulator['tp'], metric_accumulator['fp'], metric_accumulator['fn'], \
    metric_accumulator['tn']

    # mIoU
    mean_mIoU = tp / (tp + fp + fn)

    # mAP
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    mean_mAP = precision * recall

    # F1-score
    mean_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # AUC
    mean_auc = metric_accumulator['auc'] / metric_accumulator['count']

    return mean_mIoU, mean_mAP, mean_auc, mean_f1