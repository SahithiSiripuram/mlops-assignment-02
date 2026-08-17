"""Stage 2 of the pipeline: clean, resize and split the raw images.

Produces a stratified 80/10/10 train/validation/test split of 224x224 RGB JPEGs:

    data/processed/train/{cats,dogs}/
    data/processed/val/{cats,dogs}/
    data/processed/test/{cats,dogs}/
    data/processed/metadata.json

The individual helpers (`is_valid_image`, `resize_image`, `stratified_split`) are
pure functions so they can be unit-tested without touching the full dataset.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageFile

from src.config import (
    CLASS_NAMES,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_dirs,
    load_params,
    write_json,
)

# A handful of images in the Kaggle set are truncated; Pillow can still decode them.
ImageFile.LOAD_TRUNCATED_IMAGES = True

SPLITS = ["train", "val", "test"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def is_valid_image(path: Path) -> bool:
    """True when Pillow can fully decode `path` as an image.

    The Kaggle Cats-vs-Dogs data ships with a few zero-byte and corrupt files;
    filtering them here keeps the training loop from crashing mid-epoch.
    """
    if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
        return False
    if path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()  # header check
        with Image.open(path) as img:
            img.convert("RGB").load()  # full decode
        return True
    except Exception:
        return False


def resize_image(source: Path, target: Path, size: int) -> Path:
    """Convert to RGB, resize to `size` x `size` and save as JPEG."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        rgb = img.convert("RGB").resize((size, size), Image.BILINEAR)
        rgb.save(target, format="JPEG", quality=92)
    return target


def stratified_split(
    items: list[Path],
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Shuffle `items` deterministically and cut them into train/val/test.

    Ratios must sum to 1.0. Every item lands in exactly one split, and the
    remainder from rounding is given to the training split.
    """
    total_ratio = train_split + val_split + test_split
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_val = int(round(n * val_split))
    n_test = int(round(n * test_split))
    n_train = n - n_val - n_test

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def preprocess(force: bool = False) -> dict:
    params = load_params()
    data_params = params["data"]
    pre_params = params["preprocess"]

    raw_dir = Path(data_params.get("raw_dir", RAW_DIR))
    processed_dir = Path(data_params.get("processed_dir", PROCESSED_DIR))
    image_size = int(pre_params["image_size"])

    if processed_dir.exists() and force:
        shutil.rmtree(processed_dir)
    ensure_dirs(*[processed_dir / split / cls for split in SPLITS for cls in CLASS_NAMES])

    metadata: dict = {
        "image_size": image_size,
        "class_names": CLASS_NAMES,
        "splits": {split: {} for split in SPLITS},
        "skipped_corrupt": 0,
        "seed": int(pre_params["seed"]),
    }

    for class_name in CLASS_NAMES:
        source_files = sorted(
            p for p in (raw_dir / class_name).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        )
        valid_files = [p for p in source_files if is_valid_image(p)]
        metadata["skipped_corrupt"] += len(source_files) - len(valid_files)
        print(
            f"[preprocess] {class_name}: {len(valid_files)} valid / {len(source_files)} raw images"
        )

        # Optional per-class cap, sampled deterministically so runs stay reproducible.
        max_per_class = pre_params.get("max_per_class")
        if max_per_class and len(valid_files) > int(max_per_class):
            sampler = random.Random(int(pre_params["seed"]))
            valid_files = sorted(sampler.sample(valid_files, int(max_per_class)))
            print(f"[preprocess] {class_name}: capped to {len(valid_files)} images")
            metadata["max_per_class"] = int(max_per_class)

        split_map = stratified_split(
            valid_files,
            float(pre_params["train_split"]),
            float(pre_params["val_split"]),
            float(pre_params["test_split"]),
            seed=int(pre_params["seed"]),
        )

        for split, files in split_map.items():
            for index, source in enumerate(files):
                target = processed_dir / split / class_name / f"{class_name[:-1]}_{index:05d}.jpg"
                resize_image(source, target, image_size)
            metadata["splits"][split][class_name] = len(files)

    for split in SPLITS:
        metadata["splits"][split]["total"] = sum(
            metadata["splits"][split][cls] for cls in CLASS_NAMES
        )
    metadata["total_images"] = sum(metadata["splits"][s]["total"] for s in SPLITS)

    write_json(processed_dir / "metadata.json", metadata)
    print(f"[preprocess] wrote {metadata['total_images']} images to {processed_dir}")
    for split in SPLITS:
        print(f"[preprocess]   {split}: {metadata['splits'][split]}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess and split the raw dataset.")
    parser.add_argument("--force", action="store_true", help="Rebuild the processed dataset.")
    args = parser.parse_args()
    preprocess(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
