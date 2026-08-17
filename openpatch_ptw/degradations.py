from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


@dataclass(frozen=True)
class DegradationSpec:
    name: str
    kwargs: dict[str, Any]


def _to_unit(image: torch.Tensor) -> torch.Tensor:
    return (image / 2.0 + 0.5).clamp(0.0, 1.0)


def _from_unit(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(0.0, 1.0) * 2.0 - 1.0


def apply_degradation(image: torch.Tensor, name: str, **kwargs) -> torch.Tensor:
    """Evaluation degradations. Input/output are in [-1,1]."""
    if name == "clean":
        return image
    unit = _to_unit(image)
    if name == "noise":
        sigma = float(kwargs.get("sigma", 0.03))
        return _from_unit((unit + torch.randn_like(unit) * sigma).clamp(0, 1))
    if name == "blur":
        kernel = int(kwargs.get("kernel", 5))
        if kernel % 2 == 0:
            kernel += 1
        sigma = float(kwargs.get("sigma", 1.2))
        return _from_unit(TF.gaussian_blur(unit, [kernel, kernel], [sigma, sigma]))
    if name == "brightness":
        factor = float(kwargs.get("factor", 1.2))
        return _from_unit(TF.adjust_brightness(unit, factor))
    if name == "contrast":
        factor = float(kwargs.get("factor", 1.2))
        return _from_unit(TF.adjust_contrast(unit, factor))
    if name == "resize":
        scale = float(kwargs.get("scale", 0.75))
        height, width = unit.shape[-2:]
        resized = F.interpolate(unit, scale_factor=scale, mode="bilinear", align_corners=False)
        restored = F.interpolate(resized, size=(height, width), mode="bilinear", align_corners=False)
        return _from_unit(restored)
    if name == "jpeg":
        quality = int(kwargs.get("quality", 70))
        outputs = []
        for sample in unit.detach().cpu():
            pil = TF.to_pil_image(sample)
            buffer = io.BytesIO()
            pil.save(buffer, format="JPEG", quality=quality, subsampling=0)
            buffer.seek(0)
            decoded = Image.open(buffer).convert("RGB")
            outputs.append(TF.to_tensor(decoded))
        return _from_unit(torch.stack(outputs).to(device=image.device, dtype=image.dtype))
    raise ValueError(f"Unknown degradation: {name}")


def random_training_degradation(image: torch.Tensor, cfg: dict, rng: random.Random | None = None) -> torch.Tensor:
    """Cheap differentiable augmentation for training; JPEG is kept for evaluation."""
    rng = rng or random
    probability = float(cfg.get("probability", 0.5))
    if rng.random() > probability:
        return image
    choices = cfg.get("train_choices", ["noise", "blur", "brightness", "contrast", "resize"])
    name = rng.choice(list(choices))
    if name == "noise":
        return apply_degradation(image, name, sigma=rng.uniform(0.005, 0.04))
    if name == "blur":
        return apply_degradation(image, name, kernel=rng.choice([3, 5, 7]), sigma=rng.uniform(0.4, 1.6))
    if name in {"brightness", "contrast"}:
        return apply_degradation(image, name, factor=rng.uniform(0.75, 1.25))
    if name == "resize":
        return apply_degradation(image, name, scale=rng.uniform(0.6, 0.95))
    return image
