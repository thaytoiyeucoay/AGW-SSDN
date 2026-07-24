"""Original-resolution sliding-window inference."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def window_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """Return window starts that cover every pixel, including the far edge."""
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if stride > patch_size:
        raise ValueError("stride cannot exceed patch_size because it would leave gaps")
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last_position = length - patch_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


@torch.inference_mode()
def sliding_window_logits(
    model: nn.Module,
    image: torch.Tensor,
    *,
    patch_size: int = 512,
    stride: int = 384,
    num_classes: int = 3,
) -> torch.Tensor:
    """Average overlapping patch logits and restore the original image size."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape [1, C, H, W]")

    _, _, original_height, original_width = image.shape
    pad_height = max(patch_size - original_height, 0)
    pad_width = max(patch_size - original_width, 0)
    if pad_height or pad_width:
        can_reflect = (
            original_height > 1
            and original_width > 1
            and pad_height < original_height
            and pad_width < original_width
        )
        image = F.pad(
            image,
            (0, pad_width, 0, pad_height),
            mode="reflect" if can_reflect else "replicate",
        )

    _, _, padded_height, padded_width = image.shape
    logits_sum = image.new_zeros(
        (1, num_classes, padded_height, padded_width)
    )
    prediction_count = image.new_zeros((1, 1, padded_height, padded_width))

    for top in window_positions(padded_height, patch_size, stride):
        for left in window_positions(padded_width, patch_size, stride):
            patch = image[
                :, :, top : top + patch_size, left : left + patch_size
            ]
            output = model(patch)
            if isinstance(output, dict):
                output = output["main"]
            elif isinstance(output, tuple):
                output = output[0]
            if output.shape[-2:] != patch.shape[-2:]:
                output = F.interpolate(
                    output,
                    size=patch.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            logits_sum[
                :, :, top : top + patch_size, left : left + patch_size
            ] += output
            prediction_count[
                :, :, top : top + patch_size, left : left + patch_size
            ] += 1

    averaged = logits_sum / prediction_count.clamp_min(1)
    return averaged[:, :, :original_height, :original_width]


@torch.inference_mode()
def sliding_window_predict(
    model: nn.Module,
    image: torch.Tensor,
    *,
    patch_size: int = 512,
    stride: int = 384,
    num_classes: int = 3,
) -> torch.Tensor:
    """Return an original-resolution 2D class-index mask."""
    logits = sliding_window_logits(
        model,
        image,
        patch_size=patch_size,
        stride=stride,
        num_classes=num_classes,
    )
    return logits.argmax(dim=1).squeeze(0)
