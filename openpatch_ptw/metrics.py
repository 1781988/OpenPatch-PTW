from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def bit_accuracy_per_sample(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return ((prediction > threshold) == (target > 0.5)).float().mean(dim=1)


def bit_accuracy(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    return float(bit_accuracy_per_sample(prediction, target, threshold).mean().item())


def _boundary(binary: torch.Tensor, radius: int = 2) -> torch.Tensor:
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(binary, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-binary, kernel, stride=1, padding=radius)
    return (dilated - eroded > 0).float()


def boundary_f1(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, radius: int = 2) -> float:
    pred = (prediction >= threshold).float()
    gt = (target >= 0.5).float()
    pred_boundary = _boundary(pred, radius)
    gt_boundary = _boundary(gt, radius)
    gt_dilated = F.max_pool2d(gt_boundary, radius * 2 + 1, stride=1, padding=radius)
    pred_dilated = F.max_pool2d(pred_boundary, radius * 2 + 1, stride=1, padding=radius)
    precision = (pred_boundary * gt_dilated).sum() / pred_boundary.sum().clamp_min(1.0)
    recall = (gt_boundary * pred_dilated).sum() / gt_boundary.sum().clamp_min(1.0)
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-8)).item())


def mask_scores(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probability = prediction.detach().float().cpu().reshape(-1).numpy()
    truth = target.detach().float().cpu().reshape(-1).numpy().astype(np.int32)
    binary = (probability >= threshold).astype(np.int32)
    intersection = np.logical_and(binary == 1, truth == 1).sum()
    union = np.logical_or(binary == 1, truth == 1).sum()
    iou = float(intersection / max(union, 1))
    f1 = float(f1_score(truth, binary, zero_division=0))
    auc = float("nan") if np.unique(truth).size < 2 else float(roc_auc_score(truth, probability))
    return {
        "f1": f1,
        "iou": iou,
        "auc": auc,
        "boundary_f1": boundary_f1(prediction, target, threshold),
    }


def mask_scores_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> list[Dict[str, float]]:
    return [mask_scores(prediction[i : i + 1], target[i : i + 1], threshold) for i in range(prediction.shape[0])]


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int32)
    if np.unique(labels).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def status_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, object]:
    probability = torch.softmax(logits, dim=1).detach().cpu().numpy()
    truth = labels.detach().cpu().numpy().astype(np.int32)
    prediction = probability.argmax(axis=1)
    valid_target = (truth == 1).astype(np.int32)
    valid_score = probability[:, 1]
    valid_auroc = (
        float("nan")
        if np.unique(valid_target).size < 2
        else float(roc_auc_score(valid_target, valid_score))
    )
    return {
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "valid_auroc": valid_auroc,
        "valid_eer": equal_error_rate(valid_score, valid_target),
        "per_class_f1": f1_score(
            truth, prediction, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
    }


def far_at_tpr(valid_scores: np.ndarray, valid_labels: np.ndarray, target_tpr: float = 0.95) -> float:
    positive = valid_scores[valid_labels == 1]
    negative = valid_scores[valid_labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    threshold = float(np.quantile(positive, 1.0 - target_tpr))
    return float((negative >= threshold).mean())


def forgery_acceptance_rate(status_logits: torch.Tensor) -> float:
    return float((status_logits.argmax(dim=1) == 1).float().mean().item())
