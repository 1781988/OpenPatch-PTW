from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def local_code_loss(
    predicted_code: torch.Tensor,
    expected_code: torch.Tensor,
    tamper_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Supervise local code only on intact regions for valid watermarked samples."""
    if expected_code.shape[-2:] != predicted_code.shape[-2:]:
        expected_code = F.interpolate(expected_code, predicted_code.shape[-2:], mode="bilinear", align_corners=False)
    diff = (predicted_code - expected_code).abs()
    if tamper_mask is None:
        return diff.mean()
    if tamper_mask.shape[-2:] != predicted_code.shape[-2:]:
        tamper_mask = F.interpolate(tamper_mask, predicted_code.shape[-2:], mode="nearest")
    valid = 1.0 - tamper_mask
    denom = valid.sum().clamp_min(1.0) * predicted_code.shape[1]
    return (diff * valid).sum() / denom


def status_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels.long())


def mask_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 1.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return bce + float(dice_weight) * dice_loss(logits, target)
