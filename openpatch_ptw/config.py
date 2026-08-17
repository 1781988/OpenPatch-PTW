from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

_PATH_KEYS = {
    "root",
    "vae",
    "checkpoint_dir",
    "convnext",
    "lama",
    "sd_inpaint",
    "output_dir",
    "train_img_dir",
    "train_ann_file",
    "dev_img_dir",
    "dev_ann_file",
    "test_img_dir",
    "test_ann_file",
    "manifests_dir",
    "train_manifest",
    "dev_manifest",
    "test_manifest",
    "real_edit_cache",
}


def _deep_merge(base: MutableMapping[str, Any], patch: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _parse_override(text: str) -> tuple[list[str], Any]:
    if "=" not in text:
        raise ValueError(f"Override must be KEY=VALUE, got: {text}")
    key, raw = text.split("=", 1)
    path = [part for part in key.split(".") if part]
    if not path:
        raise ValueError(f"Invalid override key: {text}")
    return path, yaml.safe_load(raw)


def _set_nested(mapping: MutableMapping[str, Any], path: list[str], value: Any) -> None:
    node = mapping
    for key in path[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, MutableMapping):
            raise ValueError(f"Cannot override {'.'.join(path)} because {key} is not a mapping")
        node = child
    node[path[-1]] = value


def _resolve_paths(node: Any, repo_root: Path, parent_key: str | None = None) -> Any:
    if isinstance(node, MutableMapping):
        for key, value in list(node.items()):
            node[key] = _resolve_paths(value, repo_root, key)
        return node
    if isinstance(node, list):
        return [_resolve_paths(value, repo_root, parent_key) for value in node]
    if isinstance(node, str) and parent_key in _PATH_KEYS and node:
        p = Path(node).expanduser()
        return str(p if p.is_absolute() else (repo_root / p).resolve())
    return node


def validate_config(cfg: Mapping[str, Any]) -> None:
    required = ["project", "upstream", "data", "model", "train", "mask", "loss", "eval"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Missing top-level config sections: {missing}")

    mix = cfg.get("sample_mix", {})
    total = sum(float(mix.get(k, 0.0)) for k in ("valid", "unwatermarked", "forged"))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"sample_mix probabilities must sum to 1, got {total}")

    bins = cfg["mask"].get("bins", [])
    if not bins:
        raise ValueError("mask.bins must not be empty")
    prob = 0.0
    for item in bins:
        if len(item) != 3:
            raise ValueError(f"Each mask bin must be [lo, hi, prob], got {item}")
        lo, hi, p = map(float, item)
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"Invalid mask bin: {item}")
        prob += p
    if prob <= 0:
        raise ValueError("mask bin probability sum must be positive")

    bit_dim = int(cfg["model"].get("bit_dim", 0))
    code_dim = int(cfg["model"].get("code_dim", 0))
    if bit_dim <= 0 or code_dim <= 0:
        raise ValueError("model.bit_dim and model.code_dim must be positive")

    accum = int(cfg["train"].get("gradient_accumulation_steps", 1))
    if accum < 1:
        raise ValueError("train.gradient_accumulation_steps must be >= 1")


def load_config(
    config_path: str | Path,
    variant_path: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    resolve_paths: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if variant_path:
        variant_path = Path(variant_path).expanduser().resolve()
        with variant_path.open("r", encoding="utf-8") as handle:
            variant = yaml.safe_load(handle) or {}
        _deep_merge(cfg, variant)

    for text in overrides or []:
        path, value = _parse_override(text)
        _set_nested(cfg, path, value)

    repo_root = REPO_ROOT
    cfg.setdefault("runtime", {})
    cfg["runtime"]["config_path"] = str(config_path)
    cfg["runtime"]["variant_path"] = str(variant_path) if variant_path else None
    cfg["runtime"]["repo_root"] = str(repo_root)

    if resolve_paths:
        cfg = _resolve_paths(cfg, repo_root)
    validate_config(cfg)
    return cfg


def dump_config(cfg: Mapping[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(cfg), handle, allow_unicode=True, sort_keys=False)
