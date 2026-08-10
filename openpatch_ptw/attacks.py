from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class ForgeryResult:
    image: torch.Tensor
    mask: torch.Tensor
    kind: str


def _clip(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-1.0, 1.0)


def residual_transfer(
    target_plain: torch.Tensor,
    donor_plain: torch.Tensor,
    donor_watermarked: torch.Tensor,
    beta_range: Tuple[float, float] = (0.5, 1.5),
) -> ForgeryResult:
    """Transfer an estimated watermark residual from donor to unrelated target."""
    residual = donor_watermarked - donor_plain
    b = target_plain.shape[0]
    beta = torch.empty((b, 1, 1, 1), device=target_plain.device, dtype=target_plain.dtype)
    beta.uniform_(beta_range[0], beta_range[1])
    fake = _clip(target_plain + beta * residual)
    mask = torch.zeros((b, 1, target_plain.shape[-2], target_plain.shape[-1]), device=target_plain.device)
    return ForgeryResult(fake, mask, "residual_transfer")


def cross_image_patch_transfer(
    target: torch.Tensor,
    donor: torch.Tensor,
    mask: torch.Tensor,
) -> ForgeryResult:
    """Paste a watermarked region from a different image/message."""
    if mask.shape[1] == 1 and target.shape[1] == 3:
        m = mask.expand(-1, 3, -1, -1)
    else:
        m = mask
    fake = target * (1.0 - m) + donor * m
    return ForgeryResult(_clip(fake), mask, "cross_image_patch")


def copy_move(
    image: torch.Tensor,
    mask: torch.Tensor,
    max_shift_ratio: float = 0.35,
) -> ForgeryResult:
    """Copy a source region from another position of the same image.

    The destination mask is known; the source is produced with a random spatial roll.
    This attack is particularly relevant to position-bound watermarks because the
    global message remains unchanged while local coordinates become inconsistent.
    """
    h, w = image.shape[-2:]
    dy = random.randint(max(1, int(0.05 * h)), max(2, int(max_shift_ratio * h)))
    dx = random.randint(max(1, int(0.05 * w)), max(2, int(max_shift_ratio * w)))
    if random.random() < 0.5:
        dy = -dy
    if random.random() < 0.5:
        dx = -dx
    source = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
    m = mask.expand(-1, image.shape[1], -1, -1)
    fake = image * (1.0 - m) + source * m
    return ForgeryResult(_clip(fake), mask, "copy_move")


def batch_roll_donor(x: torch.Tensor) -> torch.Tensor:
    """Use another sample as donor without extra I/O; for B=1 use a spatial flip."""
    if x.shape[0] > 1:
        return torch.roll(x, shifts=1, dims=0)
    return torch.flip(x, dims=[-1])
