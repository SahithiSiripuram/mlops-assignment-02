"""Stage 1 of the pipeline: fetch the Cats vs Dogs dataset into `data/raw/`.

Two sources are supported (selected by `data.source` in params.yaml):

* ``kaggle`` -- the full Kaggle "Dogs vs Cats" dataset via the Kaggle CLI.
  Requires an API token at ``~/.kaggle/kaggle.json``.
* ``url``    -- a curated Cats-vs-Dogs subset served over plain HTTPS. No
  credentials are needed, which keeps CI runs and graders unblocked.

Both sources land in the same layout so every downstream stage is identical:

    data/raw/cats/*.jpg
    data/raw/dogs/*.jpg
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from src.config import CLASS_NAMES, RAW_DIR, ensure_dirs, load_params, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Kaggle and the mirrored subset use different folder words for the same classes.
CLASS_ALIASES = {
    "cats": {"cat", "cats"},
    "dogs": {"dog", "dogs"},
}


def _download_url(url: str, destination: Path) -> None:
    print(f"[download] GET {url}")
    with urlopen(url) as response, open(destination, "wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)
    size_mb = destination.stat().st_size / 1024 / 1024
    print(f"[download] saved {destination.name} ({size_mb:.1f} MB)")


def _download_kaggle(dataset: str, destination_dir: Path) -> None:
    """Shell out to the Kaggle CLI; it handles auth from ~/.kaggle/kaggle.json."""
    print(f"[download] kaggle datasets download -d {dataset}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(destination_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle download failed. Install the CLI (`pip install kaggle`) and place "
            f"your API token at ~/.kaggle/kaggle.json.\n{result.stderr.strip()}"
        )


def _extract_all(archive_dir: Path, target_dir: Path) -> None:
    for archive in sorted(archive_dir.glob("*.zip")):
        print(f"[download] extracting {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
    # Kaggle archives frequently nest a second zip inside the first one.
    for nested in sorted(target_dir.rglob("*.zip")):
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(nested.parent)
        nested.unlink()


def _class_of(path: Path) -> str | None:
    """Infer the label from any parent folder name (works for both sources)."""
    parts = {part.lower() for part in path.parts}
    for canonical, aliases in CLASS_ALIASES.items():
        if parts & aliases:
            return canonical
    # Kaggle's flat test folder names files `cat.123.jpg` / `dog.123.jpg`.
    stem = path.name.lower()
    if stem.startswith("cat"):
        return "cats"
    if stem.startswith("dog"):
        return "dogs"
    return None


def _collect_into_raw(extracted_dir: Path, raw_dir: Path) -> dict[str, int]:
    """Flatten an arbitrary extracted tree into data/raw/<class>/."""
    ensure_dirs(*[raw_dir / name for name in CLASS_NAMES])
    counts = {name: 0 for name in CLASS_NAMES}

    for image_path in sorted(extracted_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
            continue
        label = _class_of(image_path.relative_to(extracted_dir))
        if label is None:
            continue
        counts[label] += 1
        # Prefix with the running count to avoid name collisions between the
        # train/validation folders of the source archive.
        target = raw_dir / label / f"{label[:-1]}_{counts[label]:05d}{image_path.suffix.lower()}"
        shutil.copyfile(image_path, target)

    return counts


def download(source: str | None = None, force: bool = False) -> dict[str, int]:
    params = load_params()["data"]
    source = source or params["source"]
    raw_dir = Path(params.get("raw_dir", RAW_DIR))

    existing = {
        name: len(list((raw_dir / name).glob("*"))) if (raw_dir / name).exists() else 0
        for name in CLASS_NAMES
    }
    if not force and all(count > 0 for count in existing.values()):
        print(f"[download] raw data already present: {existing} (use --force to refetch)")
        return existing

    if force and raw_dir.exists():
        shutil.rmtree(raw_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive_dir = tmp_path / "archives"
        extract_dir = tmp_path / "extracted"
        ensure_dirs(archive_dir, extract_dir)

        if source == "kaggle":
            _download_kaggle(params["kaggle_dataset"], archive_dir)
        elif source == "url":
            _download_url(params["url"], archive_dir / "dataset.zip")
        else:
            raise ValueError(f"Unknown data source '{source}' (expected 'kaggle' or 'url')")

        _extract_all(archive_dir, extract_dir)
        counts = _collect_into_raw(extract_dir, raw_dir)

    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Download produced an empty class: {counts}")

    write_json(
        raw_dir / "download_summary.json",
        {"source": source, "counts": counts, "total": sum(counts.values())},
    )
    print(f"[download] raw images ready: {counts} (total {sum(counts.values())})")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Cats vs Dogs dataset.")
    parser.add_argument("--source", choices=["url", "kaggle"], default=None)
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists.")
    args = parser.parse_args()
    download(source=args.source, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
