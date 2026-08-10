from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCodeHead(nn.Module):
    """Predict the local position code from GenPTW's shared watermark feature map."""

    def __init__(self, in_channels: int = 1, code_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, code_dim, 1),
            nn.Tanh(),
        )

    def forward(self, wm_feature: torch.Tensor) -> torch.Tensor:
        return self.net(wm_feature)


def consistency_map(
    predicted_code: torch.Tensor,
    expected_code: torch.Tensor,
    detach_expected: bool = False,
) -> torch.Tensor:
    """Channel-mean absolute residual, normalized per image to a stable [0,1] range."""
    if expected_code.shape[-2:] != predicted_code.shape[-2:]:
        expected_code = F.interpolate(
            expected_code,
            size=predicted_code.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if detach_expected:
        expected_code = expected_code.detach()
    residual = (predicted_code - expected_code).abs().mean(dim=1, keepdim=True)
    # Avoid aggressive min-max normalization: absolute residual magnitude is useful.
    return residual.clamp(0.0, 2.0) / 2.0


class OpenSetStatusHead(nn.Module):
    """Three-way classifier: Unwatermarked / Valid / Forged.

    Inputs are deliberately compact to reduce overfitting to attack-specific artifacts:
    pooled shared watermark feature, bit confidence, and consistency statistics.
    """

    def __init__(self, wm_channels: int = 1, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(wm_channels + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        wm_feature: torch.Tensor,
        decoded_bits: torch.Tensor,
        residual_map: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pool(wm_feature).flatten(1)
        bit_conf = (decoded_bits - 0.5).abs().mul(2.0).mean(dim=1, keepdim=True)
        r_mean = residual_map.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        r_max = residual_map.amax(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        x = torch.cat([pooled, bit_conf, r_mean, r_max], dim=1)
        return self.net(x)
