"""Write the fixed train/val/test image lists and training-set statistics.

The validation set is a seeded random subset of the original training folder.
Normalization statistics and median-frequency class weights are computed on
the remaining training images only. The output JSON is what ``data.split_file``
points to in the per-dataset configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agw_ssdn.config import load_config
from agw_ssdn.data import (
    IMAGE_EXTENSIONS,
    PlantDataset,
    calculate_class_weights,
    calculate_mean_std,
    split_train_val,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config["data"]
    seed = int(config.get("seed", 42))
    val_ratio = float(data.get("val_split", 0.2))

    train_paths, val_paths = split_train_val(
        data["train_images"], val_ratio=val_ratio, seed=seed
    )
    test_names = sorted(
        path.name
        for path in Path(data["test_images"]).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    mean, std = calculate_mean_std(data["train_images"], train_paths)
    weight_dataset = PlantDataset(
        data["train_images"],
        data["train_masks"],
        training=False,
        mean=mean,
        std=std,
        image_token=data["image_token"],
        mask_token=data["mask_token"],
        mask_suffix=data["mask_suffix"],
        image_paths=train_paths,
    )
    class_weights = calculate_class_weights(
        weight_dataset, int(config["model"]["num_classes"])
    )

    split = {
        "seed": seed,
        "val_ratio": val_ratio,
        "train": [path.name for path in train_paths],
        "val": [path.name for path in val_paths],
        "test": test_names,
        "mean": mean,
        "std": std,
        "class_weights": class_weights.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=1) + "\n", encoding="utf-8")
    print(
        f"{args.output}: {len(split['train'])} train, {len(split['val'])} val, "
        f"{len(split['test'])} test"
    )


if __name__ == "__main__":
    main()
