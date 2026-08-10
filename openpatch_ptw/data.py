from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.datasets import CocoDetection
from torchvision.transforms import InterpolationMode

from .masks import generate_multiscale_mask, sample_area_ratio


class OpenPatchCocoDataset(CocoDetection):
    """COCO image dataset with semantic-or-synthetic multi-scale tamper masks.

    Unlike the upstream recursive 15%-25% mask filtering, this dataset samples
    a target area bin first. If a COCO instance matching that bin exists it is
    used; otherwise a synthetic rectangle/brush/thin mask is generated.
    """

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        resolution: int = 512,
        bins: Sequence[Sequence[float]] = ((0.01, 0.05, 0.25), (0.05, 0.15, 0.35), (0.15, 0.30, 0.25), (0.30, 0.50, 0.15)),
        shape_probs: Optional[dict] = None,
        coco_prob: float = 0.35,
    ):
        super().__init__(img_dir, ann_file)
        self.resolution = int(resolution)
        self.bins = bins
        self.shape_probs = shape_probs or {"rectangle": 0.35, "brush": 0.40, "thin": 0.25}
        self.coco_prob = float(coco_prob)

    def _transform_image(self, img: Image.Image) -> torch.Tensor:
        img = TF.resize(img, self.resolution, interpolation=InterpolationMode.BILINEAR, antialias=True)
        img = TF.center_crop(img, [self.resolution, self.resolution])
        x = TF.to_tensor(img)
        return TF.normalize(x, [0.5] * 3, [0.5] * 3)

    def _resize_mask(self, mask: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        pil = TF.resize(pil, self.resolution, interpolation=InterpolationMode.NEAREST)
        pil = TF.center_crop(pil, [self.resolution, self.resolution])
        arr = (np.asarray(pil) > 127).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    def _semantic_mask(self, anns, target_ratio: float) -> Optional[torch.Tensor]:
        if not anns:
            return None
        # Find an instance near the sampled target ratio; avoid recursive resampling.
        candidates = []
        for ann in anns:
            try:
                rle = self.coco.annToRLE(ann)
                m = mask_utils.decode(rle)
            except Exception:
                continue
            ratio = float(m.mean())
            if 0.005 <= ratio <= 0.55:
                candidates.append((abs(ratio - target_ratio), m))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return self._resize_mask(candidates[0][1])

    def __getitem__(self, index):
        img, anns = super().__getitem__(index)
        img = img.convert("RGB")
        image = self._transform_image(img)

        target_ratio = sample_area_ratio(self.bins)
        mask = None
        if random.random() < self.coco_prob:
            mask = self._semantic_mask(anns, target_ratio)
        if mask is None:
            mask = generate_multiscale_mask(self.resolution, self.bins, self.shape_probs)

        return {"pixel_values": image, "masks": mask, "index": int(index)}


def collate_openpatch(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "masks": torch.stack([x["masks"] for x in batch]),
        "indices": torch.tensor([x["index"] for x in batch], dtype=torch.long),
    }
