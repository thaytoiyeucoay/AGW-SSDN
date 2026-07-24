"""Loss functions used to train AGW-SSDN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class WeightedDiceLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, smooth: float = 1e-3) -> None:
        super().__init__()
        self.smooth = smooth
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        intersection = (probabilities * one_hot).sum(dim=(2, 3))
        cardinality = probabilities.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.weight is None:
            return 1 - dice.mean()
        weights = self.weight.view(1, -1)
        return 1 - ((dice * weights).sum(dim=1) / weights.sum()).mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        dice_weight: float = 0.7,
        cross_entropy_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.dice = WeightedDiceLoss(weight)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.cross_entropy_weight * self.cross_entropy(logits, targets)
        )


def _lovasz_gradient(sorted_foreground: torch.Tensor) -> torch.Tensor:
    pixels = len(sorted_foreground)
    foreground_sum = sorted_foreground.sum()
    intersection = foreground_sum - sorted_foreground.cumsum(0)
    union = foreground_sum + (1 - sorted_foreground).cumsum(0)
    jaccard = 1 - intersection / union
    if pixels > 1:
        jaccard[1:pixels] -= jaccard[:-1]
    return jaccard


def _lovasz_flat(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    classes: str | Sequence[int] = "present",
) -> torch.Tensor:
    if probabilities.numel() == 0:
        return probabilities.sum() * 0
    class_indices = (
        range(probabilities.shape[1])
        if isinstance(classes, str)
        else classes
    )
    losses: list[torch.Tensor] = []
    for class_index in class_indices:
        foreground = (labels == class_index).float()
        if classes == "present" and foreground.sum() == 0:
            continue
        errors = (foreground - probabilities[:, class_index]).abs()
        sorted_errors, permutation = torch.sort(errors, descending=True)
        sorted_foreground = foreground[permutation]
        losses.append(
            torch.dot(sorted_errors, _lovasz_gradient(sorted_foreground))
        )
    return torch.stack(losses).mean() if losses else probabilities.sum() * 0


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, per_image: bool = True) -> None:
        super().__init__()
        self.per_image = per_image

    @staticmethod
    def _flatten(
        probabilities: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channels = probabilities.shape[1]
        probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, channels)
        return probabilities, labels.reshape(-1)

    def forward(self, probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.per_image:
            losses = [
                _lovasz_flat(*self._flatten(prob[None], label[None]))
                for prob, label in zip(probabilities, labels)
            ]
            return torch.stack(losses).mean()
        return _lovasz_flat(*self._flatten(probabilities, labels))


class SymmetricLovaszLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.lovasz = LovaszSoftmaxLoss(per_image=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            0.2 * self.cross_entropy(logits, targets)
            + 0.8 * self.lovasz(logits.softmax(dim=1), targets)
        )


def generate_boundary_target(
    mask: torch.Tensor,
    kernel_size: int = 3,
    plant_background_weight: float = 0.3,
    crop_weed_weight: float = 1.0,
) -> torch.Tensor:
    """Build a dual-weighted plant/background and crop/weed boundary target."""
    padding = kernel_size // 2
    plant = (mask >= 1).float().unsqueeze(1)
    dilated_plant = F.max_pool2d(plant, kernel_size, stride=1, padding=padding)
    eroded_plant = -F.max_pool2d(-plant, kernel_size, stride=1, padding=padding)
    plant_background = (dilated_plant != eroded_plant).float()

    crop = (mask == 1).float().unsqueeze(1)
    weed = (mask == 2).float().unsqueeze(1)
    dilated_crop = F.max_pool2d(crop, kernel_size, stride=1, padding=padding)
    dilated_weed = F.max_pool2d(weed, kernel_size, stride=1, padding=padding)
    crop_weed = dilated_crop * dilated_weed
    return (
        plant_background_weight * plant_background
        + crop_weed_weight * crop_weed
    ).clamp(0, 1)


class BoundaryLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.7,
        dice_weight: float = 0.3,
        positive_weight: float = 10.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.register_buffer("positive_weight", torch.tensor([positive_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.flatten()
        targets = targets.flatten()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.positive_weight
        )
        probabilities = logits.sigmoid()
        intersection = (probabilities * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            probabilities.sum() + targets.sum() + self.smooth
        )
        return self.bce_weight * bce + self.dice_weight * (1 - dice)


class FeatureDecorrelationLoss(nn.Module):
    """Squared cosine similarity between global-trend and local-detail features."""

    def forward(self, trend: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        trend_vector = F.normalize(F.adaptive_avg_pool2d(trend, 1).flatten(1), dim=1)
        detail_vector = F.normalize(
            F.adaptive_avg_pool2d(detail, 1).flatten(1), dim=1
        )
        return ((trend_vector * detail_vector).sum(dim=1) ** 2).mean()


# Backward-compatible alias used by the original script.
MIMLoss = FeatureDecorrelationLoss
