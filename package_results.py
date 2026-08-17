from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openpatch_ptw.config import load_config
from openpatch_ptw.runtime import sha256_file, write_json


def parse_args():
    parser = argparse.ArgumentParser("Package OpenPatch-PTW experiment outputs for analysis")
    parser.add_argument("--config", default="configs/openpatch_ptw.yaml")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--max-visuals-per-suite", type=int, default=12)
    return parser.parse_args()


def copy_selected(source: Path, destination: Path, include_checkpoints: bool, max_visuals: int):
    allowed_names = {
        "resolved_config.yaml",
        "environment.json",
        "parameter_report.json",
        "upstream_load_report.json",
        "training_complete.json",
        "train_log.jsonl",
        "validation_log.jsonl",
        "evaluation_summary.json",
        "pipeline_state.json",
        "doctor_report.json",
    }
    visual_counts: dict[str, int] = {}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.suffix in {".pth", ".safetensors"} and not include_checkpoints:
            continue
        include = (
            path.name in allowed_names
            or path.suffix in {".csv", ".json", ".jsonl", ".yaml", ".yml", ".log", ".md", ".txt"}
        )
        if "visuals" in relative.parts and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            key = str(relative.parent)
            visual_counts.setdefault(key, 0)
            if visual_counts[key] >= max_visuals:
                continue
            visual_counts[key] += 1
            include = True
        if not include:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def flatten_group_summaries(package_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in package_root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        group_summary = payload.get("group_summary") if isinstance(payload, dict) else None
        if not isinstance(group_summary, dict):
            continue
        relative = path.relative_to(package_root)
        parts = relative.parts
        run_name = parts[0] if parts else ""
        model = "openpatch" if "openpatch" in parts else ("genptw" if "genptw" in parts else "")
        suite = path.stem
        for group, metrics in group_summary.items():
            row: dict[str, Any] = {
                "run": run_name,
                "model": model,
                "suite": suite,
                "group": group,
                "count": metrics.get("count"),
            }
            for metric, stats in metrics.items():
                if isinstance(stats, dict) and "mean" in stats:
                    row[f"{metric}_mean"] = stats.get("mean")
                    row[f"{metric}_std"] = stats.get("std")
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_root = Path(cfg["project"]["output_dir"])
    run_dirs = sorted(path for path in output_root.glob(f"{args.run_prefix}*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No runs matching {args.run_prefix} under {output_root}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_dir = output_root / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path(args.output) if args.output else bundle_dir / f"{args.run_prefix}_{timestamp}.zip"

    with tempfile.TemporaryDirectory(prefix="openpatch_bundle_") as temp:
        package_root = Path(temp) / f"OpenPatch-PTW_{args.run_prefix}"
        package_root.mkdir()
        for run_dir in run_dirs:
            copy_selected(
                run_dir,
                package_root / run_dir.name,
                args.include_checkpoints,
                args.max_visuals_per_suite,
            )

        rows = flatten_group_summaries(package_root)
        write_csv(package_root / "paper_summary.csv", rows)
        analysis_guide = f"""# OpenPatch-PTW 实验结果包

本结果包由 `package_results.py` 自动生成，用于后续统计分析和论文修改。

## 包含内容

- `paper_summary.csv`：各模型、实验套件、攻击条件的均值与标准差；
- `*_samples.csv/jsonl`：逐样本指标，可用于按篡改面积、攻击类型和失败案例筛选；
- `evaluation_summary.json`：每次完整评估摘要；
- `train_log.jsonl`、`validation_log.jsonl`：训练曲线与验证指标；
- `visuals/`：有限数量的原图、水印图、篡改图、GT、预测与一致性图；
- `resolved_config.yaml`、环境和 checkpoint 加载报告。

## 上传分析时建议说明

1. 本次运行使用的 GPU 与训练时长；
2. 是否完整跑过 `real_edits`；
3. 训练中是否出现 NaN/OOM/中断；
4. 希望优先优化小区域定位、开放集认证还是伪造拒绝。

默认不包含大 checkpoint 和 COCO 图像，以控制压缩包体积。
"""
        (package_root / "ANALYSIS_GUIDE.md").write_text(analysis_guide, encoding="utf-8")

        files = []
        for path in package_root.rglob("*"):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(package_root)),
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest = {
            "run_prefix": args.run_prefix,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "included_runs": [path.name for path in run_dirs],
            "include_checkpoints": args.include_checkpoints,
            "files": files,
        }
        write_json(package_root / "bundle_manifest.json", manifest)

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in package_root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(package_root.parent))

    checksum = sha256_file(archive_path)
    print(json.dumps({"archive": str(archive_path), "size": archive_path.stat().st_size, "sha256": checksum}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
