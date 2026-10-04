from __future__ import annotations

import numpy as np
from PIL import Image

from agw_ssdn.data import PlantDataset


def test_evaluation_dataset_preserves_original_resolution(tmp_path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "annotations"
    image_dir.mkdir()
    mask_dir.mkdir()

    height, width = 73, 109
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros_like(image)
    mask[10:30, 20:50] = (0, 255, 0)
    mask[40:50, 60:70] = (255, 0, 0)
    Image.fromarray(image).save(image_dir / "001_image.png")
    Image.fromarray(mask).save(mask_dir / "001_annotation.png")

    dataset = PlantDataset(
        image_dir,
        mask_dir,
        training=False,
        mean=[0, 0, 0],
        std=[1, 1, 1],
    )
    image_tensor, mask_tensor = dataset[0]

    assert image_tensor.shape == (3, height, width)
    assert mask_tensor.shape == (height, width)
    assert set(mask_tensor.unique().tolist()) == {0, 1, 2}


def test_split_file_round_trip(tmp_path) -> None:
    import json

    from agw_ssdn.data import load_split, split_train_val

    for index in range(10):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            tmp_path / f"{index:03d}_image.png"
        )
    train, val = split_train_val(tmp_path, val_ratio=0.2, seed=42)
    assert len(val) == 2 and not set(train) & set(val)
    assert (train, val) == split_train_val(tmp_path, val_ratio=0.2, seed=42)

    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"train": [p.name for p in train], "val": [p.name for p in val]})
    )
    assert load_split(split_file, tmp_path) == (train, val)
