from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.datasets import CocoDetection
from torchvision.transforms import InterpolationMode

from .masks import generate_multiscale_mask, sample_area_ratio
from .runtime import read_id_manifest


class OpenPatchCocoDataset(CocoDetection):
    """COCO images with semantic or synthetic multi-scale tamper masks.

    A manifest contains COCO image IDs, not dataset indices. Evaluation can be
    deterministic: the same image ID always receives the same target mask.
    """

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        resolution: int = 512,
        bins: Sequence[Sequence[float]] = (
            (0.01, 0.05, 0.25),
            (0.05, 0.15, 0.35),
            (0.15, 0.30, 0.25),
            (0.30, 0.50, 0.15),
        ),
        shape_probs: Optional[dict[str, float]] = None,
        coco_prob: float = 0.35,
        manifest_file: str | None = None,
        deterministic: bool = False,
        seed: int = 2026,
    ):
        super().__init__(img_dir, ann_file)
        self.resolution = int(resolution)
        self.bins = [tuple(map(float, item)) for item in bins]
        self.shape_probs = shape_probs or {
            "rectangle": 0.30,
            "ellipse": 0.15,
            "brush": 0.35,
            "thin": 0.20,
        }
        self.coco_prob = float(coco_prob)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.manifest_file = manifest_file

        selected = read_id_manifest(manifest_file)
        if selected is not None:
            available = set(self.ids)
            missing = [image_id for image_id in selected if image_id not in available]
            if missing:
                raise ValueError(
                    f"Manifest {manifest_file} contains {len(missing)} IDs absent from annotation file; "
                    f"first missing IDs: {missing[:5]}"
                )
            self.ids = selected

    def _transform_image(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(image, self.resolution, interpolation=InterpolationMode.BILINEAR, antialias=True)
        image = TF.center_crop(image, [self.resolution, self.resolution])
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, [0.5] * 3, [0.5] * 3)

    def _resize_mask(self, mask: np.ndarray) -> torch.Tensor:
        image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        image = TF.resize(image, self.resolution, interpolation=InterpolationMode.NEAREST)
        image = TF.center_crop(image, [self.resolution, self.resolution])
        array = (np.asarray(image) > 127).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)

    def _semantic_mask(self, annotations, target_ratio: float) -> Optional[torch.Tensor]:
        candidates = []
        for annotation in annotations or []:
            try:
                mask = mask_utils.decode(self.coco.annToRLE(annotation))
                transformed = self._resize_mask(mask)
            except Exception:
                continue
            ratio = float(transformed.mean())
            if 0.005 <= ratio <= 0.55:
                candidates.append((abs(ratio - target_ratio), transformed))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def __getitem__(self, index):
        image, annotations = super().__getitem__(index)
        image = image.convert("RGB")
        image_id = int(self.ids[index])
        metadata = self.coco.loadImgs([image_id])[0]

        local_seed = self.seed + image_id * 1009 if self.deterministic else None
        py_rng = random.Random(local_seed) if local_seed is not None else random
        np_rng = np.random.default_rng(local_seed)

        target_ratio = sample_area_ratio(self.bins, py_rng, np_rng)
        mask = None
        if py_rng.random() < self.coco_prob:
            mask = self._semantic_mask(annotations, target_ratio)
        if mask is None:
            synthetic_seed = int(np_rng.integers(0, 2**31 - 1))
            mask = generate_multiscale_mask(
                self.resolution,
                self.bins,
                self.shape_probs,
                seed=synthetic_seed,
            )

        return {
            "pixel_values": self._transform_image(image),
            "masks": mask.float(),
            "index": int(index),
            "image_id": image_id,
            "file_name": metadata.get("file_name", ""),
        }


def collate_openpatch(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "masks": torch.stack([item["masks"] for item in batch]),
        "indices": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "image_ids": torch.tensor([item["image_id"] for item in batch], dtype=torch.long),
        "file_names": [item["file_name"] for item in batch],
    }


def build_dataset_from_config(cfg: dict, split: str, deterministic: bool | None = None):
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown split: {split}")
    data_cfg = cfg["data"]
    image_key = "train_img_dir" if split in {"train", "dev"} else "test_img_dir"
    annotation_key = "train_ann_file" if split in {"train", "dev"} else "test_ann_file"
    manifest_key = f"{split}_manifest"
    if deterministic is None:
        deterministic = split != "train"
    return OpenPatchCocoDataset(
        data_cfg[image_key],
        data_cfg[annotation_key],
        resolution=data_cfg["resolution"],
        bins=cfg["mask"]["bins"],
        shape_probs=cfg["mask"]["shapes"],
        coco_prob=float(cfg["mask"].get("coco_prob", 0.35)),
        manifest_file=data_cfg.get(manifest_key),
        deterministic=deterministic,
        seed=int(cfg["train"]["seed"]),
    )
