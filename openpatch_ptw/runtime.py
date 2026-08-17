from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=True) + "\n")


def read_id_manifest(path: str | Path | None) -> list[int] | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(int(line))
    return ids


def _run_optional(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except Exception:
        return None


def environment_report(repo_root: str | Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp_utc": utc_timestamp(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        report["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2),
                "capability": list(torch.cuda.get_device_capability(i)),
            }
            for i in range(torch.cuda.device_count())
        ]
    report["nvidia_smi"] = _run_optional(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    report["pip_freeze"] = _run_optional([sys.executable, "-m", "pip", "freeze"])
    if repo_root:
        root = Path(repo_root)
        report["git_commit"] = _run_optional(["git", "-C", str(root), "rev-parse", "HEAD"])
        report["git_status"] = _run_optional(["git", "-C", str(root), "status", "--short"])
    return report


def count_trainable_parameters(modules: Iterable[torch.nn.Module]) -> tuple[int, int]:
    total = 0
    trainable = 0
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
            if parameter.requires_grad:
                trainable += parameter.numel()
    return total, trainable


def safe_symlink_or_copy(source: str | Path, target: str | Path) -> None:
    source = Path(source).resolve()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        import shutil

        shutil.copy2(source, target)
