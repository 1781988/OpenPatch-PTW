from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import OpenPatchModels


def save_checkpoint(
    path: str | Path,
    step: int,
    models: OpenPatchModels,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    cfg: dict,
    best_metric: float | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "step": int(step),
        "position_sf": models.vae.decoder.watermark_2.state_dict(),
        "code_head": models.code_head.state_dict(),
        "status_head": models.status_head.state_dict(),
        "localizer": models.localizer.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": cfg,
        "best_metric": best_metric,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    models: OpenPatchModels,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    strict: bool = True,
) -> dict:
    payload = torch.load(path, map_location="cpu")
    models.vae.decoder.watermark_2.load_state_dict(payload["position_sf"], strict=strict)
    models.code_head.load_state_dict(payload["code_head"], strict=strict)
    models.status_head.load_state_dict(payload["status_head"], strict=strict)
    models.localizer.load_state_dict(payload["localizer"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload


def find_checkpoint(output_dir: str | Path, explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    root = Path(output_dir)
    candidates = list(root.glob("**/best.pth")) or list(root.glob("**/step_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)
