from __future__ import annotations

import math
import random
from typing import Sequence

import cv2
import numpy as np
import torch


def _rng_pair(seed: int | None = None):
    return random.Random(seed), np.random.default_rng(seed)


def sample_area_ratio(
    bins: Sequence[Sequence[float]],
    py_rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> float:
    py_rng = py_rng or random
    np_rng = np_rng or np.random.default_rng()
    probabilities = np.asarray([item[2] for item in bins], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    index = int(np_rng.choice(len(bins), p=probabilities))
    low, high, _ = bins[index]
    return py_rng.uniform(float(low), float(high))


def _rectangle_mask(h: int, w: int, ratio: float, rng: random.Random) -> np.ndarray:
    target = max(1, int(h * w * ratio))
    aspect = math.exp(rng.uniform(math.log(0.35), math.log(2.8)))
    rh = max(1, min(int(math.sqrt(target / aspect)), h))
    rw = max(1, min(int(math.sqrt(target * aspect)), w))
    y = rng.randint(0, max(0, h - rh))
    x = rng.randint(0, max(0, w - rw))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + rh, x : x + rw] = 1
    return mask


def _brush_mask(
    h: int,
    w: int,
    ratio: float,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    target = h * w * ratio
    width = max(2, int(min(h, w) * rng.uniform(0.008, 0.05)))
    attempts = 0
    while mask.sum() < target * 0.9 and attempts < 48:
        attempts += 1
        point_count = rng.randint(2, 7)
        points = np.stack(
            [np_rng.integers(0, w, size=point_count), np_rng.integers(0, h, size=point_count)],
            axis=1,
        ).astype(np.int32)
        cv2.polylines(mask, [points], False, 1, thickness=width, lineType=cv2.LINE_8)
        for px, py in points:
            cv2.circle(mask, (int(px), int(py)), max(1, width // 2), 1, -1)
    return mask


def _thin_mask(h: int, w: int, ratio: float, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    target = h * w * ratio
    thickness = max(1, int(min(h, w) * rng.uniform(0.002, 0.012)))
    attempts = 0
    while mask.sum() < target * 0.9 and attempts < 96:
        attempts += 1
        p1 = (rng.randrange(w), rng.randrange(h))
        p2 = (rng.randrange(w), rng.randrange(h))
        cv2.line(mask, p1, p2, 1, thickness=thickness, lineType=cv2.LINE_8)
    return mask


def _ellipse_mask(h: int, w: int, ratio: float, rng: random.Random) -> np.ndarray:
    target = max(1, h * w * ratio)
    aspect = math.exp(rng.uniform(math.log(0.4), math.log(2.5)))
    axis_y = max(1, min(int(math.sqrt(target / (math.pi * aspect))), h // 2))
    axis_x = max(1, min(int(axis_y * aspect), w // 2))
    center = (rng.randint(axis_x, max(axis_x, w - axis_x - 1)), rng.randint(axis_y, max(axis_y, h - axis_y - 1)))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, (axis_x, axis_y), rng.uniform(0, 180), 0, 360, 1, -1)
    return mask


def generate_mask_for_ratio(
    resolution: int,
    ratio: float,
    shape: str,
    seed: int | None = None,
) -> torch.Tensor:
    rng, np_rng = _rng_pair(seed)
    candidates = []
    for _ in range(4):
        if shape == "rectangle":
            mask = _rectangle_mask(resolution, resolution, ratio, rng)
        elif shape == "brush":
            mask = _brush_mask(resolution, resolution, ratio, rng, np_rng)
        elif shape == "thin":
            mask = _thin_mask(resolution, resolution, ratio, rng)
        elif shape == "ellipse":
            mask = _ellipse_mask(resolution, resolution, ratio, rng)
        else:
            raise ValueError(f"Unknown mask shape: {shape}")
        candidates.append((abs(float(mask.mean()) - ratio), mask))
    candidates.sort(key=lambda item: item[0])
    return torch.from_numpy(candidates[0][1].astype(np.float32)).unsqueeze(0)


def generate_multiscale_mask(
    resolution: int,
    bins: Sequence[Sequence[float]],
    shape_probs: dict[str, float] | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    shape_probs = shape_probs or {"rectangle": 0.3, "ellipse": 0.15, "brush": 0.35, "thin": 0.2}
    rng, np_rng = _rng_pair(seed)
    names = list(shape_probs)
    probabilities = np.asarray([shape_probs[name] for name in names], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    shape = str(np_rng.choice(names, p=probabilities))
    ratio = sample_area_ratio(bins, rng, np_rng)
    return generate_mask_for_ratio(resolution, ratio, shape, seed=rng.randint(0, 2**31 - 1))


def mask_area(mask: torch.Tensor) -> torch.Tensor:
    return mask.float().mean(dim=(-1, -2, -3))
