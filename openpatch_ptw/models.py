from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL

from .genptw_bridge import (
    add_genptw_to_path,
    build_upstream_message_decoder,
    inject_openpatch_adapter,
    load_message_decoder_weights,
    warmstart_adapter_from_genptw,
)
from .heads import LocalCodeHead, OpenSetStatusHead, consistency_map
from .localizer import OpenPatchLocalizer, load_genptw_localizer_weights


@dataclass
class OpenPatchModels:
    vae: torch.nn.Module
    message_decoder: torch.nn.Module
    code_head: torch.nn.Module
    status_head: torch.nn.Module
    localizer: torch.nn.Module
    load_report: dict[str, Any]

    def modules(self):
        return [self.vae, self.message_decoder, self.code_head, self.status_head, self.localizer]

    def train(self, mode: bool = True):
        for module in self.modules():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)


@dataclass
class BaselineModels:
    vae: torch.nn.Module
    message_decoder: torch.nn.Module
    localizer: torch.nn.Module

    def modules(self):
        return [self.vae, self.message_decoder, self.localizer]

    def eval(self):
        for module in self.modules():
            module.eval()
        return self


def _required_file(path: Path, description: str, strict: bool) -> Path | None:
    if path.exists():
        return path
    if strict:
        raise FileNotFoundError(f"Missing {description}: {path}")
    return None


def build_openpatch_models(
    cfg: dict,
    device: torch.device,
    strict_assets: bool = True,
) -> OpenPatchModels:
    add_genptw_to_path(cfg["upstream"]["root"])
    model_cfg = cfg["model"]
    resolution = int(cfg["data"]["resolution"])

    vae = AutoencoderKL.from_pretrained(cfg["upstream"]["vae"])
    vae = inject_openpatch_adapter(
        vae,
        bit_dim=int(model_cfg["bit_dim"]),
        image_size=resolution,
        code_dim=int(model_cfg["code_dim"]),
        fourier_bands=int(model_cfg["fourier_bands"]),
        enable_position=bool(model_cfg.get("enable_position", True)),
    )

    checkpoint_dir = Path(cfg["upstream"]["checkpoint_dir"])
    load_report: dict[str, Any] = {}
    adapter_path = _required_file(
        checkpoint_dir / "diffusion_pytorch_model.safetensors",
        "official GenPTW adapter checkpoint",
        strict_assets,
    )
    if adapter_path:
        load_report["adapter"] = warmstart_adapter_from_genptw(
            vae,
            str(adapter_path),
            require_base_sf=bool(model_cfg.get("keep_base_sf", True)),
        )

    message_decoder = build_upstream_message_decoder(
        bit_dim=int(model_cfg["bit_dim"]), image_size=resolution
    )
    message_path = _required_file(
        checkpoint_dir / "msg_decoder.pth",
        "official GenPTW message decoder",
        strict_assets,
    )
    if message_path:
        load_report["message_decoder"] = load_message_decoder_weights(
            message_decoder, str(message_path)
        )

    code_head = LocalCodeHead(code_dim=int(model_cfg["code_dim"]))
    status_head = OpenSetStatusHead(hidden_dim=int(model_cfg["status_hidden_dim"]))
    localizer = OpenPatchLocalizer(
        image_size=resolution,
        conv_pretrain=bool(model_cfg.get("localizer_pretrained", False)),
        conv_ckpt=cfg["upstream"].get("convnext"),
        enable_consistency=bool(model_cfg.get("enable_consistency", True)),
    )
    localizer_path = _required_file(
        checkpoint_dir / "localizer.pth",
        "official GenPTW localizer",
        strict_assets,
    )
    if localizer_path:
        load_report["localizer"] = load_genptw_localizer_weights(localizer, str(localizer_path))

    for module in (vae, message_decoder, code_head, status_head, localizer):
        module.to(device)
    return OpenPatchModels(vae, message_decoder, code_head, status_head, localizer, load_report)


def build_genptw_baseline(cfg: dict, device: torch.device) -> BaselineModels:
    root = add_genptw_to_path(cfg["upstream"]["root"])
    from aaai_final_adapter import Decoder, inject_wmadapter
    from Localizer.model import Localizer
    from safetensors.torch import load_file

    resolution = int(cfg["data"]["resolution"])
    bit_dim = int(cfg["model"]["bit_dim"])
    vae = AutoencoderKL.from_pretrained(cfg["upstream"]["vae"])
    vae = inject_wmadapter(vae, bit_dim=bit_dim, image_size=resolution)
    message_decoder = Decoder(input_channels=3, output_length=bit_dim, image_size=resolution)
    localizer = Localizer(
        conv_pretrain=True,
        conv_ckpt=cfg["upstream"]["convnext"],
        image_size=resolution,
    )
    checkpoint_dir = Path(cfg["upstream"]["checkpoint_dir"])
    vae.load_state_dict(
        load_file(str(checkpoint_dir / "diffusion_pytorch_model.safetensors"), device="cpu"),
        strict=False,
    )
    message_decoder.load_state_dict(
        torch.load(checkpoint_dir / "msg_decoder.pth", map_location="cpu"), strict=True
    )
    localizer.load_state_dict(
        torch.load(checkpoint_dir / "localizer.pth", map_location="cpu"), strict=True
    )
    for module in (vae, message_decoder, localizer):
        module.to(device)
        module.requires_grad_(False)
        module.eval()
    return BaselineModels(vae, message_decoder, localizer)


def sample_bits(batch_size: int, bit_dim: int, device, dtype=torch.float32) -> torch.Tensor:
    probability = torch.full((batch_size, bit_dim), 0.5, device=device, dtype=dtype)
    return torch.bernoulli(probability)


@torch.no_grad()
def encode_decode_pair(
    vae,
    image: torch.Tensor,
    bits: torch.Tensor,
    scaling_factor: float = 0.18215,
):
    latents = vae.encode(image).latent_dist.sample() * scaling_factor
    plain = vae.decode_plain(latents / scaling_factor, return_dict=False)[0].clamp(-1, 1)
    wm_values = vae.decode_wm(latents / scaling_factor, bits, return_dict=False)
    watermarked = wm_values[0].clamp(-1, 1)
    return plain, watermarked, latents, wm_values


def extract_openpatch(
    models: OpenPatchModels,
    image: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    step: int = 0,
    detach_expected: bool = True,
) -> dict[str, torch.Tensor]:
    if target_mask is None:
        target_mask = torch.zeros(
            (image.shape[0], 1, image.shape[-2], image.shape[-1]),
            device=image.device,
            dtype=image.dtype,
        )
    decoded_bits, wm_feature = models.message_decoder(image, target_mask, step)
    predicted_code = models.code_head(wm_feature)
    expected_code = models.vae.decoder.watermark_2.code_generator(
        decoded_bits.detach() if detach_expected else decoded_bits,
        predicted_code.shape[-2:],
    )
    residual = consistency_map(predicted_code, expected_code, detach_expected=detach_expected)
    if not bool(models.localizer.enable_consistency):
        residual_for_model = torch.zeros_like(residual)
    else:
        residual_for_model = residual
    localizer_output = models.localizer(image, wm_feature, residual_for_model)
    status_logits = models.status_head(wm_feature, decoded_bits, residual_for_model)
    return {
        "decoded_bits": decoded_bits,
        "wm_feature": wm_feature,
        "predicted_code": predicted_code,
        "expected_code": expected_code,
        "consistency": residual,
        "status_logits": status_logits,
        **localizer_output,
    }


@torch.no_grad()
def extract_baseline(
    models: BaselineModels,
    image: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    step: int = 0,
) -> dict[str, torch.Tensor]:
    if target_mask is None:
        target_mask = torch.zeros(
            (image.shape[0], 1, image.shape[-2], image.shape[-1]),
            device=image.device,
            dtype=image.dtype,
        )
    decoded_bits, wm_feature = models.message_decoder(image, target_mask, step)
    localizer_output = models.localizer(image, target_mask, wm_feature)
    bit_confidence = (decoded_bits - 0.5).abs().mul(2.0).mean(dim=1)
    return {
        "decoded_bits": decoded_bits,
        "wm_feature": wm_feature,
        "pred_mask": localizer_output["pred_mask"],
        "pred_mask_logits": localizer_output.get("pred_mask_toloss"),
        "valid_score": bit_confidence,
    }
