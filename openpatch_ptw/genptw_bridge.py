from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .position_code import PositionBoundSpatialInjection


def add_genptw_to_path(root: str = "third_party/GenPTW") -> str:
    root = str(Path(root).resolve())
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"GenPTW upstream not found: {root}. Run: bash scripts/bootstrap_genptw.sh"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _upstream_adapter_classes():
    from aaai_final_adapter import MessageEncoder, Z_WatermarkCrossAttention, Decoder
    return MessageEncoder, Z_WatermarkCrossAttention, Decoder


def build_upstream_message_decoder(bit_dim: int = 64, image_size: int = 512):
    _, _, Decoder = _upstream_adapter_classes()
    return Decoder(input_channels=3, output_length=bit_dim, image_size=image_size)


def inject_openpatch_adapter(
    vae,
    bit_dim: int = 64,
    image_size: int = 512,
    code_dim: int = 8,
    hidden_dim: int = 64,
    fourier_bands: int = 4,
):
    """Inject OpenPatch modules into a diffusers AutoencoderKL.

    CAF1/CAF2 and the upstream message encoder are structurally preserved.
    Only Spatial Fusion is replaced by PositionBoundSpatialInjection.
    """
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
        hidden_dim=hidden_dim,
        fourier_bands=fourier_bands,
    )

    def decoder_forward_wm(self_obj, z: torch.Tensor, bit_vector: torch.Tensor):
        sample = z
        wm_latent_o, wm_latent_b = self_obj.watermarkEncoder(sample, bit_vector)
        sample = sample + wm_latent_o
        sample = self_obj.conv_in(sample)
        sample = self_obj.mid_block(sample)

        z0 = self_obj.up_blocks[0](sample)
        z0, wm0 = self_obj.watermark_0(z0, wm_latent_o)
        z1 = self_obj.up_blocks[1](z0)
        z1, wm1 = self_obj.watermark_1(z1, wm0)
        z2 = self_obj.up_blocks[2](z1)
        z2, position_code = self_obj.watermark_2(z2, wm_latent_b, bit_vector)
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
        z = self_obj.post_quant_conv(z)
        values = self_obj.decoder.forward_wm(z, bit_vector)
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
            sample=values[0], z0=values[1], z1=values[2], z2=values[3], z3=values[4], position_code=values[5]
        )

    def decode_plain(self_obj, z, return_dict=True):
        z = self_obj.post_quant_conv(z)
        values = self_obj.decoder.forward_plain(z)
        if not return_dict:
            return values
        from diffusers.utils import BaseOutput

        class PlainDecoderOutput(BaseOutput):
            sample: torch.FloatTensor
            z0: torch.FloatTensor
            z1: torch.FloatTensor
            z2: torch.FloatTensor
            z3: torch.FloatTensor

        return PlainDecoderOutput(sample=values[0], z0=values[1], z1=values[2], z2=values[3], z3=values[4])

    vae.decode_wm = decode_wm.__get__(vae, vae.__class__)
    vae.decode_plain = decode_plain.__get__(vae, vae.__class__)
    return vae


def _load_state_file(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def warmstart_adapter_from_genptw(vae, checkpoint_path: str) -> dict:
    """Transfer shape-compatible official GenPTW weights.

    The original Spatial Fusion (`watermark_2`) is intentionally not loaded,
    because OpenPatch replaces it with a position-bound module.
    """
    source = _load_state_file(checkpoint_path)
    current = vae.state_dict()
    transferred = {}
    skipped = []
    allowed_fragments = ("watermarkEncoder", "watermark_0", "watermark_1")

    for key, value in source.items():
        k = key.replace("module.", "")
        if not any(fragment in k for fragment in allowed_fragments):
            continue
        if k in current and current[k].shape == value.shape:
            transferred[k] = value
        else:
            skipped.append(k)

    msg = vae.load_state_dict(transferred, strict=False)
    return {
        "loaded": len(transferred),
        "skipped": skipped,
        "missing": list(msg.missing_keys),
        "unexpected": list(msg.unexpected_keys),
    }


def load_message_decoder_weights(decoder, checkpoint_path: str) -> dict:
    state = _load_state_file(checkpoint_path)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    msg = decoder.load_state_dict(cleaned, strict=False)
    return {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}
