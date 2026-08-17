from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _rgb(tensor: torch.Tensor):
    tensor = tensor.detach().float().cpu()
    if tensor.shape[0] == 1:
        return tensor[0].clamp(0, 1).numpy(), "gray"
    tensor = (tensor / 2.0 + 0.5).clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy(), None


def save_forensic_grid(
    path: str | Path,
    plain: torch.Tensor,
    watermarked: torch.Tensor,
    attacked: torch.Tensor,
    target_mask: torch.Tensor,
    predicted_mask: torch.Tensor,
    consistency: torch.Tensor,
    max_items: int = 4,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_items, plain.shape[0])
    titles = ["Plain", "Watermarked", "Attacked", "GT", "Prediction", "Consistency"]
    figure, axes = plt.subplots(count, len(titles), figsize=(18, 3 * count), squeeze=False)
    for row in range(count):
        tensors = [plain[row], watermarked[row], attacked[row], target_mask[row], predicted_mask[row], consistency[row]]
        for column, (title, tensor) in enumerate(zip(titles, tensors)):
            image, cmap = _rgb(tensor)
            axes[row, column].imshow(image, cmap=cmap, vmin=0, vmax=1)
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
