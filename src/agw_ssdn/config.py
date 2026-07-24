"""YAML configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    validate_config(config)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a resolved experiment configuration to YAML."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"data", "model", "training", "loss", "inference", "output"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    patch_size = int(config["inference"]["patch_size"])
    stride = int(config["inference"]["stride"])
    if patch_size <= 0 or stride <= 0 or stride > patch_size:
        raise ValueError(
            "inference.patch_size and stride must be positive, with stride <= patch_size"
        )
    if int(config["model"]["num_classes"]) < 2:
        raise ValueError("model.num_classes must be at least 2")
    num_classes = int(config["model"]["num_classes"])
    if len(config.get("class_names", [])) != num_classes:
        raise ValueError("class_names must contain one name per model class")
    if int(config["training"]["epochs"]) <= 0:
        raise ValueError("training.epochs must be positive")
    if int(config["training"]["batch_size"]) <= 0:
        raise ValueError("training.batch_size must be positive")
    if int(config["training"]["accumulation_steps"]) <= 0:
        raise ValueError("training.accumulation_steps must be positive")
    if len(config["loss"]["auxiliary_weights"]) != 3:
        raise ValueError("loss.auxiliary_weights must contain exactly three values")


def with_overrides(
    config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Return a copy with dotted-key CLI overrides applied."""
    updated = deepcopy(config)
    for dotted_key, value in overrides.items():
        target = updated
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    validate_config(updated)
    return updated
