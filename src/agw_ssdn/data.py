"""Dataset and preprocessing utilities for crop-weed segmentation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


DEFAULT_PALETTE = np.asarray(
    [
        [0, 0, 0],      # Background
        [0, 255, 0],    # Crop
        [255, 0, 0],    # Weed
    ],
    dtype=np.uint8,
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def rgb_to_class_mask(
    rgb_mask: np.ndarray, palette: np.ndarray = DEFAULT_PALETTE
) -> np.ndarray:
    """Map every RGB pixel to its nearest palette color."""
    pixels = rgb_mask.astype(np.float32)
    colors = palette.astype(np.float32)
    distances = ((pixels[..., None, :] - colors[None, None, ...]) ** 2).sum(axis=-1)
    return distances.argmin(axis=-1).astype(np.int64)


def colorize_mask(
    class_mask: np.ndarray, palette: np.ndarray = DEFAULT_PALETTE
) -> np.ndarray:
    """Convert a 2D class-index mask to RGB."""
    return palette[class_mask]


class PlantDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Paired crop-weed images and three-class RGB masks."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        *,
        patch_size: tuple[int, int] = (512, 512),
        training: bool = True,
        mean: Sequence[float],
        std: Sequence[float],
        weed_probability: float = 0.8,
        image_token: str = "image",
        mask_token: str = "annotation",
        mask_suffix: str = ".png",
        palette: np.ndarray = DEFAULT_PALETTE,
        image_paths: Sequence[Path] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size
        self.training = training
        self.weed_probability = weed_probability
        self.image_token = image_token
        self.mask_token = mask_token
        self.mask_suffix = mask_suffix
        self.palette = palette
        self.normalize = transforms.Normalize(list(mean), list(std))
        if image_paths is not None:
            self.images = sorted(image_paths)
        else:
            self.images = sorted(
                path
                for path in self.image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        self._weed_coordinates: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.images)

    def mask_path_for(self, image_path: Path) -> Path:
        mask_stem = image_path.stem.replace(self.image_token, self.mask_token)
        mask_path = self.mask_dir / f"{mask_stem}{self.mask_suffix}"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask for {image_path.name} was not found at {mask_path}"
            )
        return mask_path

    def _weed_coords(self, mask: Image.Image, mask_path: Path) -> np.ndarray:
        if mask_path not in self._weed_coordinates:
            class_mask = rgb_to_class_mask(np.asarray(mask), self.palette)
            self._weed_coordinates[mask_path] = np.argwhere(class_mask == 2)
        return self._weed_coordinates[mask_path]

    @staticmethod
    def _bounded_crop_origin(
        center: int, crop_length: int, image_length: int, jitter: int
    ) -> int:
        origin = center - crop_length // 2 + jitter
        return max(0, min(origin, max(image_length - crop_length, 0)))

    def _crop(
        self, image: Image.Image, mask: Image.Image, mask_path: Path
    ) -> tuple[Image.Image, Image.Image]:
        if not self.training:
            return image, mask

        width, height = image.size
        crop_width, crop_height = self.patch_size
        top = random.randint(0, max(height - crop_height, 0))
        left = random.randint(0, max(width - crop_width, 0))

        if random.random() < self.weed_probability:
            coordinates = self._weed_coords(mask, mask_path)
            if len(coordinates):
                center_y, center_x = coordinates[random.randrange(len(coordinates))]
                margin_y = max(crop_height // 2 - 50, 0)
                margin_x = max(crop_width // 2 - 50, 0)
                top = self._bounded_crop_origin(
                    int(center_y),
                    crop_height,
                    height,
                    random.randint(-margin_y, margin_y),
                )
                left = self._bounded_crop_origin(
                    int(center_x),
                    crop_width,
                    width,
                    random.randint(-margin_x, margin_x),
                )

        return (
            TF.crop(image, top, left, crop_height, crop_width),
            TF.crop(mask, top, left, crop_height, crop_width),
        )

    def _augment(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        if random.random() < 0.5:
            angle = random.uniform(-15.0, 15.0)
            image = TF.rotate(image, angle)
            mask = TF.rotate(
                mask, angle, interpolation=transforms.InterpolationMode.NEAREST
            )
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        mask_path = self.mask_path_for(image_path)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        if image.size != mask.size:
            raise ValueError(
                f"Image and mask sizes differ for {image_path.name}: "
                f"{image.size} vs {mask.size}"
            )

        image, mask = self._crop(image, mask, mask_path)
        if self.training:
            image, mask = self._augment(image, mask)

        image_tensor = self.normalize(TF.to_tensor(image))
        mask_tensor = torch.from_numpy(
            rgb_to_class_mask(np.asarray(mask), self.palette)
        ).long()
        return image_tensor, mask_tensor


def split_train_val(
    image_dir: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Deterministically split training images into train and val subsets."""
    paths = sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    return sorted(shuffled[val_count:]), sorted(shuffled[:val_count])


def load_split(
    split_file: str | Path, train_dir: str | Path
) -> tuple[list[Path], list[Path]]:
    """Read the fixed train/val image lists written by ``scripts/make_split.py``."""
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    train_dir = Path(train_dir)
    return (
        [train_dir / name for name in split["train"]],
        [train_dir / name for name in split["val"]],
    )


def calculate_mean_std(
    image_dir: str | Path, image_paths: Sequence[Path] | None = None
) -> tuple[list[float], list[float]]:
    """Calculate pixel-weighted RGB mean and standard deviation.

    If ``image_paths`` is given, only those images are used (e.g. the training
    subset after the train/val split); otherwise every image in ``image_dir``.
    """
    if image_paths is not None:
        paths = sorted(image_paths)
    else:
        paths = sorted(
            path
            for path in Path(image_dir).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for path in paths:
        tensor = TF.to_tensor(Image.open(path).convert("RGB")).to(torch.float64)
        flat = tensor.reshape(3, -1)
        channel_sum += flat.sum(dim=1)
        channel_squared_sum += flat.square().sum(dim=1)
        pixel_count += flat.shape[1]

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - mean.square()
    std = variance.clamp_min(0).sqrt()
    return mean.tolist(), std.tolist()


def calculate_class_weights(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], num_classes: int = 3
) -> torch.Tensor:
    """Median-frequency class weights computed from full-resolution masks."""
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for index in range(len(dataset)):
        _, mask = dataset[index]
        counts += torch.bincount(mask.flatten(), minlength=num_classes)[:num_classes]
    frequencies = counts / counts.sum()
    nonzero = frequencies[frequencies > 0]
    median = nonzero.median()
    return (median / frequencies.clamp_min(1e-10)).to(torch.float32)
