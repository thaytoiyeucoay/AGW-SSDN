"""Train AGW-SSDN from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from agw_ssdn.config import load_config, save_config
from agw_ssdn.data import PlantDataset, calculate_class_weights, calculate_mean_std
from agw_ssdn.engine import LossWeights, fit
from agw_ssdn.losses import (
    BoundaryLoss,
    DiceCrossEntropyLoss,
    FeatureDecorrelationLoss,
    SymmetricLovaszLoss,
)
from agw_ssdn.models import AGWSSDN
from agw_ssdn.utils import count_parameters, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml")
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:<index>"
    )
    return parser.parse_args()


def make_dataset(
    config: dict,
    split: str,
    *,
    training: bool,
    mean: list[float],
    std: list[float],
) -> PlantDataset:
    data = config["data"]
    return PlantDataset(
        data[f"{split}_images"],
        data[f"{split}_masks"],
        patch_size=tuple(data["patch_size"]),
        training=training,
        mean=mean,
        std=std,
        weed_probability=float(data["weed_probability"]),
        image_token=data["image_token"],
        mask_token=data["mask_token"],
        mask_suffix=data["mask_suffix"],
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(args.device)
    data = config["data"]

    mean = data.get("mean")
    std = data.get("std")
    if mean is None or std is None:
        print("Calculating training-set normalization statistics...")
        mean, std = calculate_mean_std(data["train_images"])
        config["data"]["mean"] = mean
        config["data"]["std"] = std
    print(f"Normalization mean={mean}, std={std}")

    train_dataset = make_dataset(
        config, "train", training=True, mean=mean, std=std
    )
    validation_dataset = make_dataset(
        config, "val", training=False, mean=mean, std=std
    )
    weight_dataset = make_dataset(
        config, "train", training=False, mean=mean, std=std
    )

    training = config["training"]
    workers = int(data["num_workers"])
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )

    num_classes = int(config["model"]["num_classes"])
    print("Calculating median-frequency class weights...")
    class_weights = calculate_class_weights(weight_dataset, num_classes).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    model = AGWSSDN(
        in_channels=int(config["model"]["in_channels"]),
        num_classes=num_classes,
        use_auxiliary_heads=bool(config["model"]["use_auxiliary_heads"]),
        use_boundary_head=bool(config["model"]["use_boundary_head"]),
    ).to(device)
    print(
        f"Model parameters: {count_parameters(model) / 1_000_000:.2f}M "
        f"({count_parameters(model, trainable_only=True) / 1_000_000:.2f}M trainable)"
    )

    early_loss = DiceCrossEntropyLoss(
        class_weights,
        dice_weight=float(config["loss"]["dice_weight"]),
        cross_entropy_weight=float(config["loss"]["cross_entropy_weight"]),
    ).to(device)
    late_loss = SymmetricLovaszLoss(class_weights).to(device)
    boundary_loss = BoundaryLoss(
        bce_weight=float(config["loss"]["boundary_bce_weight"]),
        dice_weight=float(config["loss"]["boundary_dice_weight"]),
        positive_weight=float(config["loss"]["boundary_positive_weight"]),
    ).to(device)
    decorrelation_loss = FeatureDecorrelationLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"])
    )
    loss_weights = LossWeights(
        auxiliary=tuple(float(value) for value in config["loss"]["auxiliary_weights"]),
        boundary=float(config["loss"]["boundary_weight"]),
        decorrelation=float(config["loss"]["decorrelation_weight"]),
    )
    output_path = Path(config["output"]["directory"]) / config["output"]["checkpoint"]
    resolved_config_path = output_path.parent / "resolved_config.yaml"
    save_config(config, resolved_config_path)
    print(f"Saved resolved configuration to {resolved_config_path}")

    fit(
        model,
        train_loader,
        validation_loader,
        early_loss=early_loss,
        late_loss=late_loss,
        boundary_loss=boundary_loss,
        decorrelation_loss=decorrelation_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=int(training["epochs"]),
        loss_switch_epoch=int(training["loss_switch_epoch"]),
        loss_weights=loss_weights,
        accumulation_steps=int(training["accumulation_steps"]),
        num_classes=num_classes,
        patch_size=int(config["inference"]["patch_size"]),
        stride=int(config["inference"]["stride"]),
        checkpoint_path=output_path,
        config=config,
    )


if __name__ == "__main__":
    main()
