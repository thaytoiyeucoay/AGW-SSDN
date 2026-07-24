"""Training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .inference import sliding_window_logits
from .losses import (
    BoundaryLoss,
    FeatureDecorrelationLoss,
    generate_boundary_target,
)
from .metrics import SegmentationMetrics
from .utils import load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class LossWeights:
    auxiliary: tuple[float, float, float] = (0.2, 0.3, 0.5)
    boundary: float = 0.5
    decorrelation: float = 0.1


def _resize_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] == targets.shape[-2:]:
        return logits
    return F.interpolate(
        logits, size=targets.shape[-2:], mode="bilinear", align_corners=True
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    segmentation_loss: nn.Module,
    boundary_loss: BoundaryLoss,
    decorrelation_loss: FeatureDecorrelationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_weights: LossWeights,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0,
        "main": 0.0,
        "auxiliary": 0.0,
        "boundary": 0.0,
        "decorrelation": 0.0,
    }

    for step, (images, masks) in enumerate(tqdm(loader, desc="Train"), start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images)
        if not isinstance(outputs, dict):
            raise RuntimeError("AGWSSDN must return training outputs in train mode")

        main_loss = segmentation_loss(
            _resize_logits(outputs["main"], masks), masks
        )
        auxiliary = images.new_zeros(())
        for key, weight in zip(
            ("aux1", "aux2", "aux3"), loss_weights.auxiliary
        ):
            if key in outputs:
                auxiliary = auxiliary + weight * segmentation_loss(
                    _resize_logits(outputs[key], masks), masks
                )

        boundary = images.new_zeros(())
        if "boundary" in outputs:
            targets = generate_boundary_target(masks)
            boundary = boundary_loss(
                _resize_logits(outputs["boundary"], targets), targets
            )

        decorrelation = decorrelation_loss(
            outputs["trend"], outputs["detail"]
        )
        total = (
            main_loss
            + auxiliary
            + loss_weights.boundary * boundary
            + loss_weights.decorrelation * decorrelation
        )
        (total / accumulation_steps).backward()

        if step % accumulation_steps == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        totals["loss"] += total.item()
        totals["main"] += main_loss.item()
        totals["auxiliary"] += auxiliary.item()
        totals["boundary"] += boundary.item()
        totals["decorrelation"] += decorrelation.item()

    return {key: value / max(len(loader), 1) for key, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    segmentation_loss: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    patch_size: int = 512,
    stride: int = 384,
) -> tuple[float, SegmentationMetrics]:
    model.eval()
    metrics = SegmentationMetrics(num_classes)
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="Evaluate"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if images.shape[0] != 1:
            raise ValueError(
                "Evaluation loader must use batch_size=1 for variable-size images"
            )
        logits = sliding_window_logits(
            model,
            images,
            patch_size=patch_size,
            stride=stride,
            num_classes=num_classes,
        )
        total_loss += segmentation_loss(logits, masks).item()
        metrics.update(logits.argmax(dim=1), masks)

    return total_loss / max(len(loader), 1), metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    early_loss: nn.Module,
    late_loss: nn.Module,
    boundary_loss: BoundaryLoss,
    decorrelation_loss: FeatureDecorrelationLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int,
    loss_switch_epoch: int,
    loss_weights: LossWeights,
    accumulation_steps: int,
    num_classes: int,
    patch_size: int,
    stride: int,
    checkpoint_path: str | Path,
    config: dict[str, Any],
) -> nn.Module:
    best_mean_iou = float("-inf")
    model.to(device)

    for epoch in range(epochs):
        segmentation_loss = early_loss if epoch < loss_switch_epoch else late_loss
        train_stats = train_one_epoch(
            model,
            train_loader,
            segmentation_loss,
            boundary_loss,
            decorrelation_loss,
            optimizer,
            device,
            loss_weights=loss_weights,
            accumulation_steps=accumulation_steps,
        )
        validation_loss, metrics = evaluate(
            model,
            validation_loader,
            segmentation_loss,
            device,
            num_classes=num_classes,
            patch_size=patch_size,
            stride=stride,
        )
        scheduler.step()

        class_iou = ", ".join(f"{value:.4f}" for value in metrics.class_iou)
        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"train={train_stats['loss']:.4f} | val={validation_loss:.4f} | "
            f"mIoU={metrics.mean_iou:.4f} | IoU=[{class_iou}] | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if metrics.mean_iou > best_mean_iou:
            best_mean_iou = metrics.mean_iou
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_mean_iou=best_mean_iou,
                config=config,
            )
            print(f"Saved best checkpoint to {checkpoint_path}")

    load_checkpoint(Path(checkpoint_path), model, device=device)
    return model
