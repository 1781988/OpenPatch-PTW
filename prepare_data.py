from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

from openpatch_ptw.config import load_config
from openpatch_ptw.runtime import sha256_file, write_json


def parse_args():
    parser = argparse.ArgumentParser("Download/validate COCO2017 and create deterministic manifests")
    parser.add_argument("--config", default="configs/openpatch_ptw.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--download", action="store_true", help="Run scripts/download_coco.sh first")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-images", type=int, default=200, help="Number of image files checked per split; -1 checks all")
    return parser.parse_args()


def load_image_records(annotation_file: str) -> list[dict]:
    path = Path(annotation_file)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("images")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No images found in annotation file: {path}")
    return records


def verify_files(image_dir: str, records: list[dict], limit: int) -> list[str]:
    selected = records if limit < 0 else records[:limit]
    missing = []
    root = Path(image_dir)
    for record in selected:
        path = root / record["file_name"]
        if not path.exists():
            missing.append(str(path))
    return missing


def write_manifest(path: str, image_ids: list[int], force: bool) -> dict:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if path_obj.exists() and not force:
        existing = [int(line) for line in path_obj.read_text(encoding="utf-8").splitlines() if line.strip()]
        if existing != image_ids:
            raise FileExistsError(
                f"Manifest exists with different content: {path_obj}. Use --force to overwrite."
            )
    else:
        path_obj.write_text("\n".join(str(value) for value in image_ids) + "\n", encoding="utf-8")
    return {"path": str(path_obj), "count": len(image_ids), "sha256": sha256_file(path_obj)}


def main():
    args = parse_args()
    cfg = load_config(args.config, args.variant, args.override)
    if args.download:
        dataset_root = Path(cfg["data"]["train_img_dir"]).parent
        subprocess.run(
            ["bash", str(Path(cfg["runtime"]["repo_root"]) / "scripts" / "download_coco.sh"), str(dataset_root)],
            check=True,
        )

    train_records = load_image_records(cfg["data"]["train_ann_file"])
    test_records = load_image_records(cfg["data"]["test_ann_file"])
    train_missing = verify_files(cfg["data"]["train_img_dir"], train_records, args.verify_images)
    test_missing = verify_files(cfg["data"]["test_img_dir"], test_records, args.verify_images)
    if train_missing or test_missing:
        raise FileNotFoundError(
            f"Dataset is incomplete. Missing train={len(train_missing)}, test={len(test_missing)}; "
            f"examples={(train_missing + test_missing)[:5]}"
        )

    seed = int(cfg["train"]["seed"])
    rng = random.Random(seed)
    train_ids_all = [int(record["id"]) for record in train_records]
    test_ids_all = [int(record["id"]) for record in test_records]
    rng.shuffle(train_ids_all)
    rng.shuffle(test_ids_all)

    dev_size = int(cfg["data"].get("dev_max_images", 1000))
    train_size = int(cfg["data"].get("train_max_images", 0))
    test_size = int(cfg["data"].get("test_max_images", 0))
    dev_ids = train_ids_all[:dev_size]
    train_pool = train_ids_all[dev_size:]
    train_ids = train_pool if train_size <= 0 else train_pool[:train_size]
    test_ids = test_ids_all if test_size <= 0 else test_ids_all[:test_size]

    reports = {
        "train": write_manifest(cfg["data"]["train_manifest"], train_ids, args.force),
        "dev": write_manifest(cfg["data"]["dev_manifest"], dev_ids, args.force),
        "test": write_manifest(cfg["data"]["test_manifest"], test_ids, args.force),
    }
    if set(train_ids).intersection(dev_ids):
        raise RuntimeError("Train and dev manifests overlap")

    report = {
        "seed": seed,
        "source_counts": {"train2017": len(train_records), "val2017": len(test_records)},
        "manifests": reports,
        "paths": {
            "train_img_dir": cfg["data"]["train_img_dir"],
            "train_ann_file": cfg["data"]["train_ann_file"],
            "test_img_dir": cfg["data"]["test_img_dir"],
            "test_ann_file": cfg["data"]["test_ann_file"],
        },
    }
    output = Path(cfg["data"]["manifests_dir"]) / "dataset_report.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
