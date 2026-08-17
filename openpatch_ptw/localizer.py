from __future__ import annotations

from typing import Dict, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Official 4-channel GenPTW ConvNeXt plus a zero-init consistency stem.

    Keeping the original 4-channel stem unchanged makes the initial output exactly
    compatible with the GenPTW localizer. The fifth cue is injected through a
    separate trainable projection rather than expanding and perturbing the old stem.
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
        four_channel = nn.Conv2d(
            4,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            four_channel.weight[:, :3].copy_(old.weight[:, :3])
            four_channel.weight[:, 3:4].copy_(old.weight[:, :3].mean(dim=1, keepdim=True))
        self.stem[0] = four_channel
        self.consistency_stem = nn.Conv2d(
            1,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        nn.init.zeros_(self.consistency_stem.weight)

    def forward_features(self, base_input: torch.Tensor, consistency: torch.Tensor):
        x = self.stem[0](base_input) + self.consistency_stem(consistency)
        x = self.stem[1](x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return self.norm_pre(x), outputs

    def forward(self, base_input: torch.Tensor, consistency: torch.Tensor):
        return self.forward_features(base_input, consistency)


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
        enable_consistency: bool = True,
    ):
        super().__init__()
        HighDctFrequencyExtractor, _ = _import_upstream_ops()
        self.high_dct = HighDctFrequencyExtractor()
        self.convnext = OpenPatchConvNeXt(conv_pretrain=conv_pretrain, conv_ckpt=conv_ckpt)
        self.maskdecoder = MaskDecoder()
        self.resize = nn.Upsample(size=(image_size, image_size), mode="bilinear", align_corners=False)
        self.enable_consistency = bool(enable_consistency)

    def set_consistency_enabled(self, enabled: bool) -> None:
        self.enable_consistency = bool(enabled)

    def forward(
        self,
        image: torch.Tensor,
        wm_feature: torch.Tensor,
        residual_map: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        high_frequency = self.high_dct(image)
        if wm_feature.shape[-2:] != image.shape[-2:]:
            wm_feature = F.interpolate(wm_feature, image.shape[-2:], mode="bilinear", align_corners=False)
        if residual_map is None or not self.enable_consistency:
            residual_map = torch.zeros_like(wm_feature[:, :1])
        elif residual_map.shape[-2:] != image.shape[-2:]:
            residual_map = F.interpolate(residual_map, image.shape[-2:], mode="bilinear", align_corners=False)
        base_input = torch.cat([high_frequency, wm_feature[:, :1].clamp(-1, 1)], dim=1)
        _, outputs = self.convnext(base_input, residual_map[:, :1].clamp(0, 1))
        logits = self.resize(self.maskdecoder(outputs))
        return {"pred_mask_logits": logits, "pred_mask": torch.sigmoid(logits)}


def _state_dict_from_file(path: str):
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "module"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def load_genptw_localizer_weights(
    model: OpenPatchLocalizer,
    checkpoint_path: str,
    strict_min_loaded: int = 1,
) -> dict:
    """Load official 4-channel GenPTW localizer exactly; consistency stem stays zero."""
    source = _state_dict_from_file(checkpoint_path)
    current = model.state_dict()
    transferred = {}
    skipped = []
    for key, value in source.items():
        key = key.replace("module.", "").replace("_orig_mod.", "")
        candidates = [key, key.removeprefix("model."), key.removeprefix("localizer.")]
        matched = False
        for candidate in dict.fromkeys(candidates):
            if candidate in current and current[candidate].shape == value.shape:
                transferred[candidate] = value
                matched = True
                break
        if not matched:
            skipped.append(key)
    message = model.load_state_dict(transferred, strict=False)
    if len(transferred) < strict_min_loaded:
        raise RuntimeError(f"No compatible localizer weights loaded from {checkpoint_path}")
    return {
        "loaded": len(transferred),
        "skipped": skipped,
        "missing": list(message.missing_keys),
        "unexpected": list(message.unexpected_keys),
    }
