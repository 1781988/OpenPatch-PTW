from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class FourierPositionEncoding(nn.Module):
    """Parameter-free 2-D Fourier positional encoding in [-1, 1]."""

    def __init__(self, num_bands: int = 4, include_xy: bool = True):
        super().__init__()
        self.num_bands = int(num_bands)
        self.include_xy = bool(include_xy)

    @property
    def out_channels(self) -> int:
        return (2 if self.include_xy else 0) + 4 * self.num_bands

    def forward(self, batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        feats = []
        if self.include_xy:
            feats.extend([xx, yy])
        for i in range(self.num_bands):
            freq = (2.0 ** i) * math.pi
            feats.extend([
                torch.sin(freq * xx), torch.cos(freq * xx),
                torch.sin(freq * yy), torch.cos(freq * yy),
            ])
        return torch.stack(feats, dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


class PositionCodeField(nn.Module):
    """Deterministic local code C(u,v) conditioned on watermark bits and coordinates.

    The code generator deliberately contains no trainable parameters. This avoids a
    trivial collapse in which the embedding network and local-code decoder jointly
    learn a constant code field. The fixed mapping is saved as buffers, is
    reproducible, and can be regenerated during inference from decoded bits.
    """

    def __init__(
        self,
        bit_dim: int = 64,
        code_dim: int = 8,
        hidden_dim: int = 64,  # kept for config compatibility; intentionally unused
        fourier_bands: int = 4,
    ):
        super().__init__()
        self.bit_dim = int(bit_dim)
        self.code_dim = int(code_dim)
        self.fourier_bands = max(1, int(fourier_bands))

        i = torch.arange(1, self.bit_dim + 1, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(1, self.code_dim + 1, dtype=torch.float32).unsqueeze(0)
        proj = torch.sin(i * j * 0.754877666) + torch.cos(i * (j + 0.5) * 1.324717957)
        proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("bit_projection", proj, persistent=True)

        c = torch.arange(self.code_dim, dtype=torch.float32)
        fx = 1.0 + torch.remainder(c, self.fourier_bands)
        fy = 1.0 + torch.remainder(torch.floor(c / self.fourier_bands), self.fourier_bands)
        phase = 2.0 * math.pi * (c + 0.5) / max(self.code_dim, 1)
        self.register_buffer("freq_x", fx, persistent=True)
        self.register_buffer("freq_y", fy, persistent=True)
        self.register_buffer("phase", phase, persistent=True)

    def forward(self, bits: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[1] != self.bit_dim:
            raise ValueError(f"bits must be [B,{self.bit_dim}], got {tuple(bits.shape)}")
        h, w = spatial_size
        dtype, device = bits.dtype, bits.device
        projection = self.bit_projection.to(device=device, dtype=dtype)
        signed = bits.mul(2.0).sub(1.0)
        msg = torch.tanh(signed @ projection).unsqueeze(-1).unsqueeze(-1)

        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        fx = self.freq_x.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        fy = self.freq_y.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        phase = self.phase.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        carrier = torch.sin(math.pi * fx * xx.view(1, 1, h, w) + math.pi * fy * yy.view(1, 1, h, w) + phase)

        # Message identity shifts the local carrier; both message and coordinate are required.
        return torch.tanh(msg + 0.75 * carrier)


class PositionBoundSpatialInjection(nn.Module):
    """Drop-in replacement for GenPTW Spatial Fusion.

    The original global watermark latent is retained, while a deterministic
    position-bound code field adds local identity. Residual injection starts from
    exactly zero and a small learned scale, minimizing visual-quality regression.
    """

    def __init__(
        self,
        wm_latent_dim: int,
        z_channels: int = 256,
        bit_dim: int = 64,
        code_dim: int = 8,
        hidden_dim: int = 64,
        fourier_bands: int = 4,
        init_logit: float = -4.0,
    ):
        super().__init__()
        self.z_channels = int(z_channels)
        self.global_proj = nn.Linear(wm_latent_dim, z_channels)
        self.code_generator = PositionCodeField(
            bit_dim=bit_dim,
            code_dim=code_dim,
            hidden_dim=hidden_dim,
            fourier_bands=fourier_bands,
        )
        in_ch = z_channels + z_channels + code_dim
        self.gate = nn.Sequential(nn.Conv2d(in_ch, z_channels, 1), nn.Sigmoid())
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, z_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(z_channels, z_channels, 3, padding=1),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(float(init_logit)))
        nn.init.zeros_(self.residual[-1].weight)
        if self.residual[-1].bias is not None:
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, z: torch.Tensor, wm_latent: torch.Tensor, bits: torch.Tensor):
        b, _, h, w = z.shape
        global_feat = self.global_proj(wm_latent.reshape(b, -1))
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        code = self.code_generator(bits, (h, w))
        x = torch.cat([z, global_feat, code], dim=1)
        delta = self.gate(x) * self.residual(x)
        alpha = torch.sigmoid(self.alpha_logit)
        return z + alpha * delta, code

    @torch.no_grad()
    def expected_code(self, bits: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        return self.code_generator(bits, spatial_size)
