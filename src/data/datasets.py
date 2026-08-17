"""Torch dataloaders and the augmentation policy shared by training and serving."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.config import CLASS_NAMES, NORM_MEAN, NORM_STD, load_params


def build_train_transform(image_size: int, augmentation: dict) -> transforms.Compose:
    """Augmentation for the training split only: flips, rotation, crop and jitter."""
    jitter = float(augmentation.get("color_jitter", 0.2))
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(float(augmentation.get("random_resized_crop_scale", 0.8)), 1.0),
            ),
            transforms.RandomHorizontalFlip(p=float(augmentation.get("horizontal_flip", 0.5))),
            transforms.RandomRotation(degrees=float(augmentation.get("rotation_degrees", 15))),
            transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic transform used for validation, test and live inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def build_dataloaders(
    processed_dir: Path | None = None,
    image_size: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[dict[str, DataLoader], list[str]]:
    """Return {'train','val','test'} dataloaders over the processed image folders."""
    params = load_params()
    processed_dir = Path(processed_dir or params["data"]["processed_dir"])
    image_size = int(image_size or params["preprocess"]["image_size"])
    batch_size = int(batch_size or params["train"]["batch_size"])
    num_workers = int(params["train"]["num_workers"] if num_workers is None else num_workers)

    train_tf = build_train_transform(image_size, params["train"]["augmentation"])
    eval_tf = build_eval_transform(image_size)

    loaders: dict[str, DataLoader] = {}
    for split in ["train", "val", "test"]:
        dataset = ImageFolder(processed_dir / split, transform=train_tf if split == "train" else eval_tf)
        if dataset.classes != CLASS_NAMES:
            raise ValueError(f"Unexpected class order in {split}: {dataset.classes}")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders, CLASS_NAMES


def load_flattened_split(
    split: str,
    image_size: int,
    processed_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a split as flattened pixel vectors for the logistic-regression baseline."""
    params = load_params()
    processed_dir = Path(processed_dir or params["data"]["processed_dir"])

    features: list[np.ndarray] = []
    labels: list[int] = []
    for label_index, class_name in enumerate(CLASS_NAMES):
        for path in sorted((processed_dir / split / class_name).glob("*.jpg")):
            with Image.open(path) as img:
                arr = np.asarray(
                    img.convert("RGB").resize((image_size, image_size), Image.BILINEAR),
                    dtype=np.float32,
                )
            features.append((arr / 255.0).reshape(-1))
            labels.append(label_index)

    return np.stack(features), np.asarray(labels, dtype=np.int64)


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """Single-image preprocessing used by the inference service."""
    return build_eval_transform(image_size)(image.convert("RGB")).unsqueeze(0)
