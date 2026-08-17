from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class PositionCodeField(nn.Module):
    """Deterministic local code C(u,v) conditioned on message bits and coordinates.

    The mapping is parameter-free. This prevents the embedder and local-code head
    from collapsing to a jointly learned constant field, and makes the expected
    code exactly reproducible at evaluation time.
    """

    def __init__(self, bit_dim: int = 64, code_dim: int = 8, fourier_bands: int = 4):
        super().__init__()
        self.bit_dim = int(bit_dim)
        self.code_dim = int(code_dim)
        self.fourier_bands = max(1, int(fourier_bands))
        if self.bit_dim <= 0 or self.code_dim <= 0:
            raise ValueError("bit_dim and code_dim must be positive")

        i = torch.arange(1, self.bit_dim + 1, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(1, self.code_dim + 1, dtype=torch.float32).unsqueeze(0)
        projection = torch.sin(i * j * 0.754877666) + torch.cos(i * (j + 0.5) * 1.324717957)
        projection = projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("bit_projection", projection, persistent=True)

        channel = torch.arange(self.code_dim, dtype=torch.float32)
        # Non-integer frequencies avoid equal code values at opposite borders.
        freq_x = 0.73 + 0.61 * torch.remainder(channel, self.fourier_bands)
        freq_y = 1.11 + 0.47 * torch.remainder(
            torch.floor(channel / self.fourier_bands), self.fourier_bands
        )
        phase = 2.0 * math.pi * (channel + 0.5) / max(self.code_dim, 1)
        self.register_buffer("freq_x", freq_x, persistent=True)
        self.register_buffer("freq_y", freq_y, persistent=True)
        self.register_buffer("phase", phase, persistent=True)

    def forward(self, bits: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[1] != self.bit_dim:
            raise ValueError(f"bits must be [B,{self.bit_dim}], got {tuple(bits.shape)}")
        height, width = map(int, spatial_size)
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid spatial_size: {spatial_size}")

        projection = self.bit_projection.to(device=bits.device, dtype=bits.dtype)
        signed_bits = bits.mul(2.0).sub(1.0)
        message = torch.tanh(signed_bits @ projection).unsqueeze(-1).unsqueeze(-1)

        y = torch.linspace(-1.0, 1.0, height, device=bits.device, dtype=bits.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=bits.device, dtype=bits.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        fx = self.freq_x.to(device=bits.device, dtype=bits.dtype).view(1, -1, 1, 1)
        fy = self.freq_y.to(device=bits.device, dtype=bits.dtype).view(1, -1, 1, 1)
        phase = self.phase.to(device=bits.device, dtype=bits.dtype).view(1, -1, 1, 1)
        carrier = torch.sin(
            math.pi * fx * xx.view(1, 1, height, width)
            + math.pi * fy * yy.view(1, 1, height, width)
            + phase
        )
        return torch.tanh(message + 0.75 * carrier)


class _ConvBNSelu(nn.Module):
    """State-compatible equivalent of GenPTW's ConvBNSelu."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.bn = nn.SyncBatchNorm(out_channels)
        self.selu = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu(self.bn(self.conv(x)))


class PositionBoundSpatialInjection(nn.Module):
    """GenPTW-compatible Spatial Fusion plus a zero-initialized position branch.

    `global_proj` and `base_fuse` mirror the original GenPTW Spatial Fusion and
    can be warm-started exactly. The position branch starts at zero, so replacing
    the module does not initially perturb the official baseline output.
    """

    def __init__(
        self,
        wm_latent_dim: int,
        z_channels: int = 256,
        bit_dim: int = 64,
        code_dim: int = 8,
        fourier_bands: int = 4,
        init_logit: float = -4.0,
        enable_position: bool = True,
    ):
        super().__init__()
        self.z_channels = int(z_channels)
        self.enable_position = bool(enable_position)
        self.global_proj = nn.Sequential(nn.Linear(wm_latent_dim, z_channels))
        self.base_fuse = nn.Sequential(_ConvBNSelu(z_channels * 2, z_channels))
        self.code_generator = PositionCodeField(
            bit_dim=bit_dim,
            code_dim=code_dim,
            fourier_bands=fourier_bands,
        )

        position_in = z_channels * 2 + code_dim
        self.position_gate = nn.Sequential(nn.Conv2d(position_in, z_channels, 1), nn.Sigmoid())
        self.position_residual = nn.Sequential(
            nn.Conv2d(position_in, z_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(z_channels, z_channels, 3, padding=1),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(float(init_logit)))
        nn.init.zeros_(self.position_residual[-1].weight)
        if self.position_residual[-1].bias is not None:
            nn.init.zeros_(self.position_residual[-1].bias)

        # Before official warm-start, keep the base branch neutral as well.
        nn.init.zeros_(self.base_fuse[0].conv.weight)
        if self.base_fuse[0].conv.bias is not None:
            nn.init.zeros_(self.base_fuse[0].conv.bias)

    def set_position_enabled(self, enabled: bool) -> None:
        self.enable_position = bool(enabled)

    def forward(self, z: torch.Tensor, wm_latent: torch.Tensor, bits: torch.Tensor):
        batch, _, height, width = z.shape
        global_feature = self.global_proj(wm_latent.reshape(batch, -1))
        global_feature = global_feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

        base_delta = self.base_fuse(torch.cat([z, global_feature], dim=1))
        code = self.code_generator(bits, (height, width))
        if not self.enable_position:
            return z + base_delta, code

        position_input = torch.cat([z, global_feature, code], dim=1)
        position_delta = self.position_gate(position_input) * self.position_residual(position_input)
        alpha = torch.sigmoid(self.alpha_logit)
        return z + base_delta + alpha * position_delta, code

    @torch.no_grad()
    def expected_code(self, bits: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        return self.code_generator(bits, spatial_size)
