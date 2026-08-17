"""Shared pytest fixtures.

The API tests must not depend on a real training run, so `stub_model_dir` builds a
tiny TorchScript model with the same contract as the exported one (224x224x3 in,
2 logits out) plus a matching `metadata.json`.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.config import CLASS_NAMES, NORM_MEAN, NORM_STD

IMAGE_SIZE = 224


def make_image(color: tuple[int, int, int] = (120, 90, 60), size: int = 320) -> Image.Image:
    """A deterministic RGB test image with some structure (not a flat fill)."""
    image = Image.new("RGB", (size, size), color)
    pixels = image.load()
    for x in range(size):
        for y in range(0, size, 8):
            pixels[x, y] = ((x * 3) % 256, (y * 5) % 256, (x + y) % 256)
    return image


def image_bytes(image: Image.Image | None = None, fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    (image or make_image()).save(buffer, format=fmt)
    return buffer.getvalue()


class _TinyNet(nn.Module):
    """Stand-in for the trained CNN: same input/output shapes, no training needed."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, len(CLASS_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    return make_image()


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    return image_bytes(sample_image)


@pytest.fixture(scope="session")
def stub_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model_dir = tmp_path_factory.mktemp("models")
    scripted = torch.jit.trace(_TinyNet().eval(), torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    scripted.save(str(model_dir / "model.pt"))

    metadata = {
        "model_name": "stub-cnn",
        "model_family": "pytorch",
        "artifact": "model.pt",
        "artifact_format": "torchscript",
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
        "val_metrics": {"accuracy": 0.99},
        "test_metrics": {"accuracy": 0.98, "f1": 0.98},
        "mlflow_run_id": "stub-run",
        "trained_at": "2026-01-01T00:00:00",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return model_dir


@pytest.fixture()
def api_client(stub_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with the stub model wired in through MODEL_DIR."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(stub_model_dir))
    from src.api import main as api_main

    api_main.stats.feedback_path = tmp_path / "feedback.jsonl"
    with TestClient(api_main.app) as client:
        yield client
