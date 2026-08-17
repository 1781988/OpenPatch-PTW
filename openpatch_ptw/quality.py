from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.metrics import psnr as kornia_psnr
from kornia.metrics import ssim as kornia_ssim


@dataclass
class QualityLossOutput:
    total: torch.Tensor
    reconstruction: torch.Tensor
    lpips: torch.Tensor
    jnd: torch.Tensor


class QualityObjective(nn.Module):
    """GenPTW-style fidelity objective with optional LPIPS and JND."""

    def __init__(
        self,
        use_lpips: bool = True,
        use_jnd: bool = True,
        lpips_net: str = "vgg",
    ):
        super().__init__()
        self.use_lpips = bool(use_lpips)
        self.use_jnd = bool(use_jnd)
        self.lpips_model = None
        self.jnd_model = None
        if self.use_lpips:
            import lpips

            self.lpips_model = lpips.LPIPS(net=lpips_net)
            self.lpips_model.requires_grad_(False)
            self.lpips_model.eval()
        if self.use_jnd:
            try:
                from JND import JND
            except ImportError as exc:
                raise ImportError("JND requires the upstream GenPTW repository on sys.path") from exc
            self.jnd_model = JND(in_channels=3, out_channels=3, blue=False)
            self.jnd_model.requires_grad_(False)
            self.jnd_model.eval()

    @staticmethod
    def _unit(image: torch.Tensor) -> torch.Tensor:
        return (image / 2.0 + 0.5).clamp(0.0, 1.0)

    def forward(
        self,
        watermarked: torch.Tensor,
        plain: torch.Tensor,
        rec_weight: float = 0.5,
        lpips_weight: float = 1.0,
        jnd_weight: float = 1.5,
    ) -> QualityLossOutput:
        reconstruction = F.mse_loss(watermarked, plain.detach())
        lpips_loss = watermarked.new_zeros(())
        jnd_loss = watermarked.new_zeros(())

        if self.lpips_model is not None and lpips_weight > 0:
            self.lpips_model.to(device=watermarked.device, dtype=torch.float32)
            lpips_loss = self.lpips_model(watermarked.float(), plain.detach().float()).mean()
        if self.jnd_model is not None and jnd_weight > 0:
            self.jnd_model.to(device=watermarked.device, dtype=torch.float32)
            plain_unit = self._unit(plain.detach().float())
            wm_unit = self._unit(watermarked.float())
            with torch.no_grad():
                heatmap = self.jnd_model(plain_unit)
                cost = torch.exp(-8.0 * heatmap)
            jnd_loss = (cost * (wm_unit - plain_unit).abs()).mean()

        total = (
            float(rec_weight) * reconstruction
            + float(lpips_weight) * lpips_loss
            + float(jnd_weight) * jnd_loss
        )
        return QualityLossOutput(total, reconstruction, lpips_loss, jnd_loss)


@torch.no_grad()
def image_quality_per_sample(
    watermarked: torch.Tensor,
    plain: torch.Tensor,
    lpips_model: nn.Module | None = None,
) -> list[dict[str, float]]:
    wm = (watermarked.float() / 2.0 + 0.5).clamp(0, 1)
    clean = (plain.float() / 2.0 + 0.5).clamp(0, 1)
    psnr_values = [
        float(kornia_psnr(wm[i : i + 1], clean[i : i + 1], max_val=1.0).item())
        for i in range(wm.shape[0])
    ]
    ssim_map = kornia_ssim(wm, clean, window_size=5)
    ssim_values = ssim_map.mean(dim=(1, 2, 3)).cpu().tolist()
    lpips_values = [float("nan")] * wm.shape[0]
    if lpips_model is not None:
        lpips_model.to(device=watermarked.device, dtype=torch.float32)
        lpips_values = lpips_model(watermarked.float(), plain.float()).flatten().float().cpu().tolist()
    return [
        {"psnr": psnr_values[i], "ssim": float(ssim_values[i]), "lpips": float(lpips_values[i])}
        for i in range(wm.shape[0])
    ]
