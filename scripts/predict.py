"""Run original-resolution prediction and save a colorized mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from agw_ssdn.config import load_config
from agw_ssdn.data import DEFAULT_PALETTE, colorize_mask
from agw_ssdn.inference import sliding_window_predict
from agw_ssdn.models import AGWSSDN
from agw_ssdn.utils import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    data = config["data"]
    if data.get("mean") is None or data.get("std") is None:
        raise ValueError("Set data.mean and data.std in the config before prediction")

    model_config = config["model"]
    model = AGWSSDN(
        in_channels=int(model_config["in_channels"]),
        num_classes=int(model_config["num_classes"]),
        use_auxiliary_heads=bool(model_config["use_auxiliary_heads"]),
        use_boundary_head=bool(model_config["use_boundary_head"]),
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor = transforms.Normalize(data["mean"], data["std"])(
        TF.to_tensor(image)
    ).unsqueeze(0).to(device)
    prediction = sliding_window_predict(
        model,
        tensor,
        patch_size=int(config["inference"]["patch_size"]),
        stride=int(config["inference"]["stride"]),
        num_classes=int(model_config["num_classes"]),
    )
    color_mask = colorize_mask(
        prediction.cpu().numpy().astype(np.int64), DEFAULT_PALETTE
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color_mask).save(args.output)
    print(f"Saved {prediction.shape[1]}x{prediction.shape[0]} mask to {args.output}")


if __name__ == "__main__":
    main()
