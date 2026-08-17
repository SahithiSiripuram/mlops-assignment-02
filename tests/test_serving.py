"""Unit tests for the model/inference utilities (M3.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import CLASS_NAMES
from src.data.datasets import image_to_tensor
from src.models.architectures import SimpleCNN, build_model, count_parameters
from src.models.evaluate import compute_metrics
from src.models.serving import (
    MAX_IMAGE_BYTES,
    ModelNotAvailableError,
    Predictor,
    decode_image,
    softmax,
)
from tests.conftest import image_bytes, make_image


class TestDecodeImage:
    def test_decodes_jpeg_bytes_to_rgb(self, sample_image_bytes: bytes) -> None:
        image = decode_image(sample_image_bytes)
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    def test_decodes_png_bytes(self) -> None:
        assert decode_image(image_bytes(make_image(), fmt="PNG")).mode == "RGB"

    def test_rejects_empty_payload(self) -> None:
        with pytest.raises(ValueError, match="Empty image payload"):
            decode_image(b"")

    def test_rejects_non_image_bytes(self) -> None:
        with pytest.raises(ValueError, match="Could not decode"):
            decode_image(b"not an image at all")

    def test_rejects_oversized_payload(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            decode_image(b"x" * (MAX_IMAGE_BYTES + 1))


def test_softmax_returns_a_probability_distribution() -> None:
    probabilities = softmax(np.array([2.0, 1.0]))
    assert pytest.approx(probabilities.sum(), abs=1e-6) == 1.0
    assert probabilities[0] > probabilities[1]


def test_image_to_tensor_shape_and_normalization() -> None:
    tensor = image_to_tensor(make_image(size=300), image_size=224)
    assert tuple(tensor.shape) == (1, 3, 224, 224)
    # ImageNet normalization pushes values outside [0, 1].
    assert tensor.min() < 0


class TestArchitectures:
    def test_simple_cnn_output_shape(self) -> None:
        import torch

        logits = SimpleCNN().eval()(torch.zeros(2, 3, 224, 224))
        assert tuple(logits.shape) == (2, len(CLASS_NAMES))

    def test_transfer_model_freezes_the_backbone(self) -> None:
        model = build_model("transfer")
        total, trainable = count_parameters(model)
        assert trainable < total

    def test_unknown_model_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("resnet999")


class TestPredictor:
    def test_predict_returns_the_expected_contract(self, stub_model_dir: Path) -> None:
        result = Predictor(model_dir=stub_model_dir).predict(make_image())

        assert result["label"] in CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 1.0
        assert set(result["probabilities"]) == set(CLASS_NAMES)
        assert pytest.approx(sum(result["probabilities"].values()), abs=1e-4) == 1.0
        assert result["confidence"] == pytest.approx(max(result["probabilities"].values()), abs=1e-4)

    def test_predict_bytes_matches_predict(self, stub_model_dir: Path) -> None:
        predictor = Predictor(model_dir=stub_model_dir)
        image = make_image()
        assert predictor.predict_bytes(image_bytes(image))["label"] == predictor.predict(image)["label"]

    def test_predict_bytes_rejects_garbage(self, stub_model_dir: Path) -> None:
        with pytest.raises(ValueError):
            Predictor(model_dir=stub_model_dir).predict_bytes(b"nonsense")

    def test_info_exposes_model_metadata(self, stub_model_dir: Path) -> None:
        info = Predictor(model_dir=stub_model_dir).info()
        assert info["model_name"] == "stub-cnn"
        assert info["image_size"] == 224
        assert info["class_names"] == CLASS_NAMES

    def test_missing_model_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotAvailableError):
            Predictor(model_dir=tmp_path / "does-not-exist")


def test_compute_metrics_on_a_perfect_prediction() -> None:
    y_true = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y_true, y_true, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
