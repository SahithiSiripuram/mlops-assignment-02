"""Exploratory data analysis over the raw dataset.

Writes three figures to `docs/plots/`:

  eda_class_balance.png      class counts per split
  eda_sample_grid.png        a grid of sample images from each class
  eda_image_dimensions.png   the raw width/height distribution that motivates 224x224

    python scripts/eda.py
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, PROCESSED_DIR, RAW_DIR, read_json  # noqa: E402

PLOTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "plots"
SPLITS = ["train", "val", "test"]


def plot_class_balance() -> Path:
    metadata = read_json(PROCESSED_DIR / "metadata.json")
    fig, ax = plt.subplots(figsize=(6, 4))

    width = 0.35
    positions = range(len(SPLITS))
    for offset, class_name in zip([-width / 2, width / 2], CLASS_NAMES, strict=True):
        counts = [metadata["splits"][split][class_name] for split in SPLITS]
        bars = ax.bar([p + offset for p in positions], counts, width, label=class_name)
        ax.bar_label(bars, fontsize=9)

    ax.set_xticks(list(positions), [s.capitalize() for s in SPLITS])
    ax.set_ylabel("images")
    ax.set_title("Class balance across the 80/10/10 split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output = PLOTS_DIR / "eda_class_balance.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def plot_sample_grid(per_class: int = 5) -> Path:
    rng = random.Random(42)
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(per_class * 2, len(CLASS_NAMES) * 2.2))

    for row, class_name in enumerate(CLASS_NAMES):
        files = sorted((PROCESSED_DIR / "train" / class_name).glob("*.jpg"))
        for col, path in enumerate(rng.sample(files, min(per_class, len(files)))):
            ax = axes[row][col]
            with Image.open(path) as img:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(class_name, fontsize=11)

    fig.suptitle("Preprocessed training samples (224x224 RGB)")
    fig.tight_layout()

    output = PLOTS_DIR / "eda_sample_grid.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def plot_image_dimensions(sample_size: int = 400) -> Path:
    """Raw images vary a lot in size -- this is why everything is resized to 224x224."""
    rng = random.Random(42)
    widths: list[int] = []
    heights: list[int] = []
    modes: Counter[str] = Counter()

    for class_name in CLASS_NAMES:
        files = sorted((RAW_DIR / class_name).glob("*"))
        for path in rng.sample(files, min(sample_size // 2, len(files))):
            try:
                with Image.open(path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    modes[img.mode] += 1
            except Exception:
                continue

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(widths, heights, s=8, alpha=0.4)
    axes[0].axvline(224, color="crimson", linestyle="--", linewidth=1, label="224 px target")
    axes[0].axhline(224, color="crimson", linestyle="--", linewidth=1)
    axes[0].set_xlabel("width (px)")
    axes[0].set_ylabel("height (px)")
    axes[0].set_title(f"Raw image dimensions (n={len(widths)})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(list(modes.keys()), list(modes.values()))
    axes[1].set_title("Colour modes in the raw data")
    axes[1].set_ylabel("images")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output = PLOTS_DIR / "eda_image_dimensions.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def main() -> int:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for figure in [plot_class_balance(), plot_sample_grid(), plot_image_dimensions()]:
        print(f"[eda] wrote {figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
