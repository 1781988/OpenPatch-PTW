from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _import_upstream_ops():
    try:
        from Localizer.high_frequency_feature_extraction import HighDctFrequencyExtractor
        from model_utils import GatedRes
        return HighDctFrequencyExtractor, GatedRes
    except ImportError as exc:
        raise ImportError(
            "未找到 GenPTW 上游模块。请先运行 bash scripts/bootstrap_genptw.sh，"
            "并在入口脚本中调用 add_genptw_to_path()."
        ) from exc


class OpenPatchConvNeXt(timm.models.convnext.ConvNeXt):
    """ConvNeXt-Tiny backbone with 5 input channels.

    Channels: 3 high-frequency RGB + 1 watermark feature + 1 consistency map.
    """

    def __init__(self, conv_pretrain: bool = False, conv_ckpt: Optional[str] = None):
        super().__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        if conv_pretrain and conv_ckpt:
            base = timm.create_model("convnext_tiny", pretrained=False)
            state = torch.load(conv_ckpt, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            base.load_state_dict(state, strict=False)
            self.load_state_dict(base.state_dict(), strict=False)

        old = self.stem[0]
        new = nn.Conv2d(
            5,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            new.weight[:, :3].copy_(old.weight[:, :3])
            mean = old.weight[:, :3].mean(dim=1, keepdim=True)
            new.weight[:, 3:4].copy_(mean)
            new.weight[:, 4:5].copy_(mean)
        self.stem[0] = new

    def forward_features(self, x):
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        x = self.norm_pre(x)
        return x, outs

    def forward(self, x):
        return self.forward_features(x)


class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        _, GatedRes = _import_upstream_ops()
        self.upsamplec2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        self.upsamplec3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
        )
        self.upsamplec4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            GatedRes(384, 96, kernel_size=1, stride=1, padding=0),
            GatedRes(96, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        c2 = self.upsamplec2(c2)
        c3 = self.upsamplec3(c3)
        c4 = self.upsamplec4(c4)
        return self.decoder(torch.cat([c1, c2, c3, c4], dim=1))


class OpenPatchLocalizer(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        conv_pretrain: bool = False,
        conv_ckpt: Optional[str] = None,
    ):
        super().__init__()
        HighDctFrequencyExtractor, _ = _import_upstream_ops()
        self.high_dct = HighDctFrequencyExtractor()
        self.convnext = OpenPatchConvNeXt(conv_pretrain=conv_pretrain, conv_ckpt=conv_ckpt)
        self.maskdecoder = MaskDecoder()
        self.resize = nn.Upsample(size=(image_size, image_size), mode="bilinear", align_corners=False)

    def forward(
        self,
        image: torch.Tensor,
        wm_feature: torch.Tensor,
        residual_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        high_freq = self.high_dct(image)
        if wm_feature.shape[-2:] != image.shape[-2:]:
            wm_feature = F.interpolate(wm_feature, image.shape[-2:], mode="bilinear", align_corners=False)
        if residual_map.shape[-2:] != image.shape[-2:]:
            residual_map = F.interpolate(residual_map, image.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([high_freq, wm_feature.clamp(-1, 1), residual_map.clamp(0, 1)], dim=1)
        _, outs = self.convnext(x)
        logits = self.resize(self.maskdecoder(outs))
        return {"pred_mask_logits": logits, "pred_mask": torch.sigmoid(logits)}


def load_genptw_localizer_weights(model: OpenPatchLocalizer, checkpoint_path: str) -> dict:
    """Warm-start a 5-channel localizer from the official 4-channel GenPTW checkpoint.

    The new fifth input channel is initialized as the mean of the first three
    image channels. Shape-incompatible tensors are skipped instead of causing
    strict=False size mismatch errors.
    """
    raw = torch.load(checkpoint_path, map_location="cpu")
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    current = model.state_dict()
    transferred = {}
    skipped = []

    for key, value in state.items():
        k = key.replace("module.", "")
        if k not in current:
            skipped.append(k)
            continue
        if current[k].shape == value.shape:
            transferred[k] = value
            continue
        if k.endswith("convnext.stem.0.weight") and value.ndim == 4 and value.shape[1] == 4:
            expanded = current[k].clone()
            expanded[:, :4] = value
            expanded[:, 4:5] = value[:, :3].mean(dim=1, keepdim=True)
            transferred[k] = expanded
        else:
            skipped.append(k)

    msg = model.load_state_dict(transferred, strict=False)
    return {
        "loaded": len(transferred),
        "skipped": skipped,
        "missing": list(msg.missing_keys),
        "unexpected": list(msg.unexpected_keys),
    }
