from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class ResultWriter:
    """Write aggregate JSON and sample-level CSV/JSONL without retaining all tensors."""

    def __init__(self, output_dir: str | Path, suite: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.suite = suite
        self.jsonl_path = self.output_dir / f"{suite}_samples.jsonl"
        self.csv_path = self.output_dir / f"{suite}_samples.csv"
        self._rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        clean = {}
        for key, value in row.items():
            if hasattr(value, "item") and callable(value.item):
                value = value.item()
            if isinstance(value, np.generic):
                value = value.item()
            clean[key] = value
        self._rows.append(clean)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean, ensure_ascii=False, allow_nan=True) + "\n")

    @staticmethod
    def _numeric(values: Iterable[Any]) -> list[float]:
        out = []
        for value in values:
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                value = float(value)
                if np.isfinite(value):
                    out.append(value)
        return out

    def summarize(self, group_keys: tuple[str, ...] = ("attack",)) -> dict[str, Any]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in self._rows:
            groups[tuple(row.get(key) for key in group_keys)].append(row)
        summary = {}
        for group, rows in groups.items():
            name = "/".join(str(item) for item in group)
            keys = sorted({key for row in rows for key in row})
            metrics = {"count": len(rows)}
            for key in keys:
                values = self._numeric(row.get(key) for row in rows)
                if values:
                    metrics[key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "median": float(np.median(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
            summary[name] = metrics
        return summary

    def finalize(self, payload: dict[str, Any], group_keys: tuple[str, ...] = ("attack",)) -> dict[str, Any]:
        if self._rows:
            fields = sorted({key for row in self._rows for key in row})
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self._rows)
        payload = dict(payload)
        payload["sample_count"] = len(self._rows)
        payload["sample_files"] = {
            "jsonl": str(self.jsonl_path),
            "csv": str(self.csv_path),
        }
        payload["group_summary"] = self.summarize(group_keys)
        aggregate_path = self.output_dir / f"{self.suite}.json"
        with aggregate_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
        return payload
