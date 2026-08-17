from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch

from .position_code import PositionBoundSpatialInjection


def add_genptw_to_path(root: str = "third_party/GenPTW") -> str:
    root = str(Path(root).expanduser().resolve())
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"GenPTW upstream not found: {root}. Run: bash scripts/bootstrap_genptw.sh"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _upstream_adapter_classes():
    from aaai_final_adapter import Decoder, MessageEncoder, Z_WatermarkCrossAttention

    return MessageEncoder, Z_WatermarkCrossAttention, Decoder


def build_upstream_message_decoder(bit_dim: int = 64, image_size: int = 512):
    _, _, Decoder = _upstream_adapter_classes()
    return Decoder(input_channels=3, output_length=bit_dim, image_size=image_size)


def inject_openpatch_adapter(
    vae,
    bit_dim: int = 64,
    image_size: int = 512,
    code_dim: int = 8,
    fourier_bands: int = 4,
    enable_position: bool = True,
):
    """Inject GenPTW CAF modules and the OpenPatch spatial fusion into a VAE."""
    MessageEncoder, Z_WatermarkCrossAttention, _ = _upstream_adapter_classes()

    if not hasattr(vae.decoder, "old_forward"):
        vae.decoder.old_forward = vae.decoder.forward
    if not hasattr(vae, "old_decode"):
        vae.old_decode = vae.decode

    decoder = vae.decoder
    size = image_size // 8
    decoder.watermarkEncoder = MessageEncoder(input_length=bit_dim, output_channels=4, blocks_size=size)
    decoder.watermark_0 = Z_WatermarkCrossAttention(wm_channels=4, z_channels=512, hidden_dim=512)
    decoder.watermark_1 = Z_WatermarkCrossAttention(wm_channels=512, z_channels=512, hidden_dim=512)
    decoder.watermark_2 = PositionBoundSpatialInjection(
        wm_latent_dim=4 * size * size,
        z_channels=256,
        bit_dim=bit_dim,
        code_dim=code_dim,
        fourier_bands=fourier_bands,
        enable_position=enable_position,
    )

    def decoder_forward_wm(self_obj, z: torch.Tensor, bit_vector: torch.Tensor):
        wm_latent, wm_vector = self_obj.watermarkEncoder(z, bit_vector)
        sample = z + wm_latent
        sample = self_obj.conv_in(sample)
        sample = self_obj.mid_block(sample)

        z0 = self_obj.up_blocks[0](sample)
        z0, wm0 = self_obj.watermark_0(z0, wm_latent)
        z1 = self_obj.up_blocks[1](z0)
        z1, _wm1 = self_obj.watermark_1(z1, wm0)
        z2 = self_obj.up_blocks[2](z1)
        z2, position_code = self_obj.watermark_2(z2, wm_vector, bit_vector)
        z3 = self_obj.up_blocks[3](z2)

        sample = self_obj.conv_norm_out(z3)
        sample = self_obj.conv_act(sample)
        sample = self_obj.conv_out(sample)
        return sample, z0, z1, z2, z3, position_code

    def decoder_forward_plain(self_obj, z: torch.Tensor):
        sample = self_obj.conv_in(z)
        sample = self_obj.mid_block(sample)
        z0 = self_obj.up_blocks[0](sample)
        z1 = self_obj.up_blocks[1](z0)
        z2 = self_obj.up_blocks[2](z1)
        z3 = self_obj.up_blocks[3](z2)
        sample = self_obj.conv_norm_out(z3)
        sample = self_obj.conv_act(sample)
        sample = self_obj.conv_out(sample)
        return sample, z0, z1, z2, z3

    decoder.forward_wm = decoder_forward_wm.__get__(decoder, decoder.__class__)
    decoder.forward_plain = decoder_forward_plain.__get__(decoder, decoder.__class__)

    def decode_wm(self_obj, z, bit_vector, return_dict=True):
        values = self_obj.decoder.forward_wm(self_obj.post_quant_conv(z), bit_vector)
        if not return_dict:
            return values
        from diffusers.utils import BaseOutput

        class OpenPatchDecoderOutput(BaseOutput):
            sample: torch.FloatTensor
            z0: torch.FloatTensor
            z1: torch.FloatTensor
            z2: torch.FloatTensor
            z3: torch.FloatTensor
            position_code: torch.FloatTensor

        return OpenPatchDecoderOutput(
            sample=values[0],
            z0=values[1],
            z1=values[2],
            z2=values[3],
            z3=values[4],
            position_code=values[5],
        )

    def decode_plain(self_obj, z, return_dict=True):
        values = self_obj.decoder.forward_plain(self_obj.post_quant_conv(z))
        if not return_dict:
            return values
        from diffusers.utils import BaseOutput

        class PlainDecoderOutput(BaseOutput):
            sample: torch.FloatTensor
            z0: torch.FloatTensor
            z1: torch.FloatTensor
            z2: torch.FloatTensor
            z3: torch.FloatTensor

        return PlainDecoderOutput(
            sample=values[0], z0=values[1], z1=values[2], z2=values[3], z3=values[4]
        )

    vae.decode_wm = decode_wm.__get__(vae, vae.__class__)
    vae.decode_plain = decode_plain.__get__(vae, vae.__class__)
    return vae


def _load_state_file(path: str) -> Dict[str, torch.Tensor]:
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "module"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint object in {path}: {type(raw)!r}")
    return raw


def _canonical_candidates(key: str) -> list[str]:
    key = key.replace("module.", "").replace("_orig_mod.", "")
    candidates = [key]
    for prefix in ("vae.", "model."):
        if key.startswith(prefix):
            candidates.append(key[len(prefix) :])
    if key.startswith("decoder."):
        candidates.append(key)
    else:
        candidates.append("decoder." + key)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def warmstart_adapter_from_genptw(vae, checkpoint_path: str, require_base_sf: bool = True) -> dict:
    """Load official GenPTW message encoder, CAFs and original SF base branch.

    The official SF is remapped to `global_proj` and `base_fuse`; the new
    position branch remains zero-initialized.
    """
    source = _load_state_file(checkpoint_path)
    current = vae.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    def remap(key: str) -> str:
        key = key.replace(".watermark_2.proj.", ".watermark_2.global_proj.")
        key = key.replace(".watermark_2.fuse.", ".watermark_2.base_fuse.")
        return key

    allowed = ("watermarkEncoder", "watermark_0", "watermark_1", "watermark_2.proj", "watermark_2.fuse")
    for source_key, value in source.items():
        if not any(fragment in source_key for fragment in allowed):
            continue
        matched = False
        for candidate in _canonical_candidates(source_key):
            candidate = remap(candidate)
            if candidate in current and current[candidate].shape == value.shape:
                transferred[candidate] = value
                matched = True
                break
        if not matched:
            skipped.append(source_key)

    message = vae.load_state_dict(transferred, strict=False)
    base_loaded = sum(
        1
        for key in transferred
        if "watermark_2.global_proj" in key or "watermark_2.base_fuse" in key
    )
    caf_loaded = sum(
        1 for key in transferred if "watermarkEncoder" in key or "watermark_0" in key or "watermark_1" in key
    )
    if require_base_sf and base_loaded == 0:
        raise RuntimeError(
            "Official GenPTW SF weights were not loaded. Check that "
            "diffusion_pytorch_model.safetensors is the official checkpoint."
        )
    if caf_loaded == 0:
        raise RuntimeError("Official GenPTW watermark encoder/CAF weights were not loaded.")
    return {
        "loaded": len(transferred),
        "base_sf_loaded": base_loaded,
        "caf_loaded": caf_loaded,
        "skipped": skipped,
        "missing": list(message.missing_keys),
        "unexpected": list(message.unexpected_keys),
    }


def load_message_decoder_weights(decoder, checkpoint_path: str, strict_min_loaded: int = 1) -> dict:
    source = _load_state_file(checkpoint_path)
    current = decoder.state_dict()
    transferred = {}
    skipped = []
    for key, value in source.items():
        matched = False
        for candidate in _canonical_candidates(key):
            if candidate in current and current[candidate].shape == value.shape:
                transferred[candidate] = value
                matched = True
                break
        if not matched:
            skipped.append(key)
    message = decoder.load_state_dict(transferred, strict=False)
    if len(transferred) < strict_min_loaded:
        raise RuntimeError(f"No compatible message-decoder weights loaded from {checkpoint_path}")
    return {
        "loaded": len(transferred),
        "skipped": skipped,
        "missing": list(message.missing_keys),
        "unexpected": list(message.unexpected_keys),
    }


def set_module_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def freeze_batch_norm_stats(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()
