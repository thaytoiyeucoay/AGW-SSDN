"""Segmentation metrics based on a streaming confusion matrix."""

from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )

    def reset(self) -> None:
        self.confusion_matrix.fill(0)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predicted = predictions.detach().cpu().numpy().reshape(-1)
        target = targets.detach().cpu().numpy().reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        indices = self.num_classes * target[valid] + predicted[valid]
        self.confusion_matrix += np.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    @property
    def class_iou(self) -> np.ndarray:
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1)
            + self.confusion_matrix.sum(axis=0)
            - intersection
        )
        return intersection / np.maximum(union, 1)

    @property
    def mean_iou(self) -> float:
        return float(np.nanmean(self.class_iou))

    @property
    def pixel_accuracy(self) -> float:
        total = self.confusion_matrix.sum()
        return float(np.diag(self.confusion_matrix).sum() / max(total, 1))

    @property
    def class_accuracy(self) -> np.ndarray:
        denominator = self.confusion_matrix.sum(axis=1)
        return np.diag(self.confusion_matrix) / np.maximum(denominator, 1)

    def as_dict(self, class_names: list[str] | None = None) -> dict[str, object]:
        names = class_names or [
            f"class_{index}" for index in range(self.num_classes)
        ]
        return {
            "mean_iou": self.mean_iou,
            "pixel_accuracy": self.pixel_accuracy,
            "class_iou": dict(zip(names, self.class_iou.tolist())),
            "class_accuracy": dict(zip(names, self.class_accuracy.tolist())),
        }
