import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


def compute_binary_metrics(y_true, y_logits):

    y_true = y_true.cpu().numpy()
    y_probs = torch.sigmoid(y_logits).cpu().numpy()
    y_pred = (y_probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_probs),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics