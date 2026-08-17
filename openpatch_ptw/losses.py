from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def boundary_map(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def balanced_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    max_pos_weight: float = 20.0,
    edge_weight: float = 0.0,
) -> torch.Tensor:
    positive = target.sum(dim=(1, 2, 3), keepdim=True)
    total = float(target[0].numel())
    negative = total - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, float(max_pos_weight))
    element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    element = element * torch.where(target > 0.5, pos_weight, torch.ones_like(pos_weight))
    if edge_weight > 0:
        element = element * (1.0 + float(edge_weight) * boundary_map(target))
    return element.mean()


def local_code_loss(
    predicted_code: torch.Tensor,
    expected_code: torch.Tensor,
    tamper_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if expected_code.shape[-2:] != predicted_code.shape[-2:]:
        expected_code = F.interpolate(
            expected_code, predicted_code.shape[-2:], mode="bilinear", align_corners=False
        )
    difference = (predicted_code - expected_code).abs()
    if tamper_mask is None:
        return difference.mean()
    if tamper_mask.shape[-2:] != predicted_code.shape[-2:]:
        tamper_mask = F.interpolate(tamper_mask, predicted_code.shape[-2:], mode="nearest")
    intact = 1.0 - tamper_mask
    denominator = intact.sum().clamp_min(1.0) * predicted_code.shape[1]
    return (difference * intact).sum() / denominator


def status_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(logits, labels.long(), weight=class_weights)


def mask_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 1.0,
    edge_weight: float = 0.0,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    bce = balanced_bce_with_logits(
        logits,
        target,
        max_pos_weight=max_pos_weight,
        edge_weight=edge_weight,
    )
    return bce + float(dice_weight) * dice_loss(logits, target)
