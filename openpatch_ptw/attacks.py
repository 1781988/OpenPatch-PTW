from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch


@dataclass
class ForgeryResult:
    image: torch.Tensor
    mask: torch.Tensor
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _clip(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(-1.0, 1.0)


def batch_roll_donor(tensor: torch.Tensor) -> torch.Tensor:
    """Use another batch item as donor; B=1 falls back to horizontal flip."""
    if tensor.shape[0] > 1:
        return torch.roll(tensor, shifts=1, dims=0)
    return torch.flip(tensor, dims=[-1])


def residual_transfer(
    target_plain: torch.Tensor,
    donor_plain: torch.Tensor,
    donor_watermarked: torch.Tensor,
    beta_range: Tuple[float, float] = (0.5, 1.5),
    beta: torch.Tensor | None = None,
) -> ForgeryResult:
    residual = donor_watermarked - donor_plain
    batch = target_plain.shape[0]
    if beta is None:
        beta = torch.empty((batch, 1, 1, 1), device=target_plain.device, dtype=target_plain.dtype)
        beta.uniform_(float(beta_range[0]), float(beta_range[1]))
    fake = _clip(target_plain + beta * residual)
    mask = torch.zeros(
        (batch, 1, target_plain.shape[-2], target_plain.shape[-1]),
        device=target_plain.device,
        dtype=target_plain.dtype,
    )
    return ForgeryResult(fake, mask, "residual_transfer", {"beta": beta.detach().flatten().cpu().tolist()})


def cross_image_patch_transfer(
    target: torch.Tensor,
    donor: torch.Tensor,
    mask: torch.Tensor,
) -> ForgeryResult:
    expanded = mask.expand(-1, target.shape[1], -1, -1) if mask.shape[1] == 1 else mask
    fake = target * (1.0 - expanded) + donor * expanded
    return ForgeryResult(_clip(fake), mask, "cross_image_patch")


def copy_move(
    image: torch.Tensor,
    mask: torch.Tensor,
    max_shift_ratio: float = 0.35,
    rng: random.Random | None = None,
) -> ForgeryResult:
    rng = rng or random
    height, width = image.shape[-2:]
    min_dy = max(1, int(0.05 * height))
    max_dy = max(min_dy, int(max_shift_ratio * height))
    min_dx = max(1, int(0.05 * width))
    max_dx = max(min_dx, int(max_shift_ratio * width))
    dy = rng.randint(min_dy, max_dy) * (-1 if rng.random() < 0.5 else 1)
    dx = rng.randint(min_dx, max_dx) * (-1 if rng.random() < 0.5 else 1)
    source = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
    expanded = mask.expand(-1, image.shape[1], -1, -1)
    fake = image * (1.0 - expanded) + source * expanded
    return ForgeryResult(_clip(fake), mask, "copy_move", {"dx": dx, "dy": dy})


def local_plain_replacement(
    watermarked: torch.Tensor,
    donor_plain: torch.Tensor,
    mask: torch.Tensor,
) -> ForgeryResult:
    expanded = mask.expand(-1, watermarked.shape[1], -1, -1)
    edited = watermarked * (1.0 - expanded) + donor_plain * expanded
    return ForgeryResult(_clip(edited), mask, "local_plain_replacement")
