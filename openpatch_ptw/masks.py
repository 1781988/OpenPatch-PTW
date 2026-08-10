from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch


def sample_area_ratio(bins: Sequence[Sequence[float]]) -> float:
    """bins: [[lo, hi, probability], ...]."""
    probs = np.asarray([b[2] for b in bins], dtype=np.float64)
    probs = probs / probs.sum()
    idx = int(np.random.choice(len(bins), p=probs))
    lo, hi, _ = bins[idx]
    return random.uniform(float(lo), float(hi))


def _rectangle_mask(h: int, w: int, ratio: float) -> np.ndarray:
    target = max(1, int(h * w * ratio))
    aspect = math.exp(random.uniform(math.log(0.35), math.log(2.8)))
    rh = int(math.sqrt(target / aspect))
    rw = int(math.sqrt(target * aspect))
    rh = max(1, min(rh, h - 1))
    rw = max(1, min(rw, w - 1))
    y = random.randint(0, max(0, h - rh))
    x = random.randint(0, max(0, w - rw))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + rh, x : x + rw] = 1
    return mask


def _brush_mask(h: int, w: int, ratio: float) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    target = h * w * ratio
    strokes = random.randint(1, 4)
    width = max(2, int(min(h, w) * random.uniform(0.01, 0.06)))
    tries = 0
    while mask.sum() < target * 0.8 and tries < strokes * 8:
        tries += 1
        n = random.randint(2, 6)
        pts = np.stack(
            [np.random.randint(0, w, size=n), np.random.randint(0, h, size=n)], axis=1
        ).astype(np.int32)
        cv2.polylines(mask, [pts], False, 1, thickness=width, lineType=cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(mask, (int(px), int(py)), width // 2, 1, -1)
    return mask


def _thin_mask(h: int, w: int, ratio: float) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    target = h * w * ratio
    thickness = max(1, int(min(h, w) * random.uniform(0.003, 0.015)))
    tries = 0
    while mask.sum() < target * 0.8 and tries < 40:
        tries += 1
        p1 = (random.randrange(w), random.randrange(h))
        p2 = (random.randrange(w), random.randrange(h))
        cv2.line(mask, p1, p2, 1, thickness=thickness, lineType=cv2.LINE_AA)
    return mask


def generate_multiscale_mask(
    resolution: int,
    bins: Sequence[Sequence[float]],
    shape_probs: dict | None = None,
) -> torch.Tensor:
    shape_probs = shape_probs or {"rectangle": 0.4, "brush": 0.4, "thin": 0.2}
    names = list(shape_probs)
    probs = np.asarray([shape_probs[n] for n in names], dtype=np.float64)
    probs = probs / probs.sum()
    shape = str(np.random.choice(names, p=probs))
    ratio = sample_area_ratio(bins)
    if shape in ("rectangle", "coco"):
        mask = _rectangle_mask(resolution, resolution, ratio)
    elif shape == "brush":
        mask = _brush_mask(resolution, resolution, ratio)
    elif shape == "thin":
        mask = _thin_mask(resolution, resolution, ratio)
    else:
        raise ValueError(f"Unknown mask shape: {shape}")
    return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)


def mask_area(mask: torch.Tensor) -> torch.Tensor:
    return mask.float().mean(dim=(-1, -2, -3))
