from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import torch

from openpatch_ptw.config import load_config
from openpatch_ptw.data import build_dataset_from_config
from openpatch_ptw.models import build_openpatch_models, encode_decode_pair, extract_openpatch, sample_bits
from openpatch_ptw.runtime import environment_report, read_id_manifest, write_json


def parse_args():
    parser = argparse.ArgumentParser("Validate OpenPatch-PTW environment, data and checkpoints")
    parser.add_argument("--config", default="configs/openpatch_ptw.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--forward", action="store_true", help="Run one full VAE/embed/extract forward pass")
    return parser.parse_args()


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_path(path: str, kind: str, required=True):
    p = Path(path)
    exists = p.is_dir() if kind == "dir" else p.is_file()
    return {"path": str(p), "kind": kind, "required": required, "ok": exists}


def main():
    args = parse_args()
    cfg = load_config(args.config, args.variant, args.override)
    checks = []
    checks.extend(
        [
            check_path(cfg["upstream"]["root"], "dir"),
            check_path(cfg["upstream"]["vae"], "dir"),
            check_path(cfg["upstream"]["checkpoint_dir"], "dir"),
            check_path(cfg["upstream"]["convnext"], "file"),
            check_path(cfg["upstream"]["lama"], "file", required=False),
            check_path(cfg["upstream"]["sd_inpaint"], "dir", required=False),
            check_path(cfg["data"]["train_img_dir"], "dir"),
            check_path(cfg["data"]["train_ann_file"], "file"),
            check_path(cfg["data"]["test_img_dir"], "dir"),
            check_path(cfg["data"]["test_ann_file"], "file"),
            check_path(cfg["data"]["train_manifest"], "file"),
            check_path(cfg["data"]["dev_manifest"], "file"),
            check_path(cfg["data"]["test_manifest"], "file"),
        ]
    )
    checkpoint_dir = Path(cfg["upstream"]["checkpoint_dir"])
    for name in ("msg_decoder.pth", "localizer.pth", "diffusion_pytorch_model.safetensors"):
        checks.append(check_path(str(checkpoint_dir / name), "file"))

    packages = {
        name: package_version(name)
        for name in [
            "torch",
            "torchvision",
            "diffusers",
            "transformers",
            "accelerate",
            "timm",
            "kornia",
            "lpips",
            "pycocotools",
            "opencv-python-headless",
            "scikit-learn",
            "safetensors",
        ]
    }
    missing_packages = [name for name, version in packages.items() if version is None]
    required_failures = [item for item in checks if item["required"] and not item["ok"]]

    warnings = []
    if bool(cfg.get("experiment", {}).get("use_forgery_training", True)) and int(cfg["train"].get("batch_size", 1)) < 2:
        warnings.append(
            "Formal forgery training requires train.batch_size >= 2 so donor images/messages come from a different sample. "
            "Batch size 1 is supported only for smoke debugging."
        )
    if int(cfg["eval"].get("batch_size", 1)) < 2:
        warnings.append(
            "Formal open-set/forgery evaluation requires eval.batch_size >= 2 for true cross-image donor pairing."
        )

    report = {
        "environment": environment_report(cfg["runtime"]["repo_root"]),
        "packages": packages,
        "missing_packages": missing_packages,
        "checks": checks,
        "required_failures": required_failures,
        "warnings": warnings,
        "disk_free_gb": round(shutil.disk_usage(cfg["runtime"]["repo_root"]).free / 2**30, 2),
        "manifest_counts": {
            split: len(read_id_manifest(cfg["data"].get(f"{split}_manifest")) or [])
            for split in ("train", "dev", "test")
        },
    }

    if args.forward and not required_failures and not missing_packages:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = build_dataset_from_config(cfg, "dev", deterministic=True)
        sample = dataset[0]
        image = sample["pixel_values"].unsqueeze(0).to(device)
        mask = sample["masks"].unsqueeze(0).to(device)
        models = build_openpatch_models(cfg, device, strict_assets=True).eval()
        bits = sample_bits(1, int(cfg["model"]["bit_dim"]), device, image.dtype)
        with torch.no_grad():
            plain, watermarked, _, values = encode_decode_pair(models.vae, image, bits)
            zero_mask = torch.zeros_like(mask)
            output = extract_openpatch(models, watermarked, zero_mask)
        report["forward"] = {
            "ok": True,
            "image": list(image.shape),
            "plain": list(plain.shape),
            "watermarked": list(watermarked.shape),
            "position_code": list(values[-1].shape),
            "decoded_bits": list(output["decoded_bits"].shape),
            "pred_mask": list(output["pred_mask"].shape),
            "status_logits": list(output["status_logits"].shape),
        }

    output = Path(cfg["project"]["output_dir"]) / "system" / "doctor_report.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    if args.strict and (required_failures or missing_packages):
        sys.exit(1)


if __name__ == "__main__":
    main()
