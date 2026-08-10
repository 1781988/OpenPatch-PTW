from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def bit_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    return float(((pred > threshold) == (target > 0.5)).float().mean().item())


def mask_scores(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    p = pred.detach().float().cpu().reshape(-1).numpy()
    t = target.detach().float().cpu().reshape(-1).numpy().astype(np.int32)
    pb = (p >= threshold).astype(np.int32)
    inter = np.logical_and(pb == 1, t == 1).sum()
    union = np.logical_or(pb == 1, t == 1).sum()
    iou = float(inter / max(union, 1))
    f1 = float(f1_score(t, pb, zero_division=0))
    auc = float("nan") if np.unique(t).size < 2 else float(roc_auc_score(t, p))
    return {"f1": f1, "iou": iou, "auc": auc}


def status_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, object]:
    prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(np.int32)
    pred = prob.argmax(axis=1)
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    # Treat Valid (class 1) as positive for deployment-style accept/reject analysis.
    valid_target = (y == 1).astype(np.int32)
    valid_score = prob[:, 1]
    auroc = float("nan") if np.unique(valid_target).size < 2 else float(roc_auc_score(valid_target, valid_score))
    return {
        "macro_f1": macro_f1,
        "valid_auroc": auroc,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def far_at_tpr(valid_scores: np.ndarray, valid_labels: np.ndarray, target_tpr: float = 0.95) -> float:
    """FAR at target TPR where labels are 1 for valid and 0 otherwise."""
    pos = valid_scores[valid_labels == 1]
    neg = valid_scores[valid_labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    threshold = float(np.quantile(pos, 1.0 - target_tpr))
    return float((neg >= threshold).mean())


def forgery_acceptance_rate(status_logits: torch.Tensor) -> float:
    """Fraction of forged samples classified as Valid (class 1)."""
    pred = status_logits.argmax(dim=1)
    return float((pred == 1).float().mean().item())
