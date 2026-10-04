"""Evaluate a trained AGW-SSDN checkpoint at original image resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from agw_ssdn.config import load_config
from agw_ssdn.data import PlantDataset
from agw_ssdn.engine import evaluate
from agw_ssdn.losses import DiceCrossEntropyLoss
from agw_ssdn.models import AGWSSDN
from agw_ssdn.utils import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="Optional JSON metrics path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    data = config["data"]
    if data.get("mean") is None or data.get("std") is None:
        raise ValueError(
            "Set data.mean and data.std in the config before evaluation"
        )

    dataset = PlantDataset(
        data["test_images"],
        data["test_masks"],
        patch_size=tuple(data["patch_size"]),
        training=False,
        mean=data["mean"],
        std=data["std"],
        image_token=data["image_token"],
        mask_token=data["mask_token"],
        mask_suffix=data["mask_suffix"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(data["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model_config = config["model"]
    model = AGWSSDN(
        in_channels=int(model_config["in_channels"]),
        num_classes=int(model_config["num_classes"]),
        use_auxiliary_heads=bool(model_config["use_auxiliary_heads"]),
        use_boundary_head=bool(model_config["use_boundary_head"]),
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    loss = DiceCrossEntropyLoss().to(device)
    validation_loss, metrics = evaluate(
        model,
        loader,
        loss,
        device,
        num_classes=int(model_config["num_classes"]),
        patch_size=int(config["inference"]["patch_size"]),
        stride=int(config["inference"]["stride"]),
    )
    results = metrics.as_dict(config["class_names"])
    results["loss"] = validation_loss
    print(json.dumps(results, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
