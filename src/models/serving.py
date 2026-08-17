"""Inference utilities shared by the API, the smoke test and the unit tests.

`Predictor` hides the difference between the exported TorchScript model and the
scikit-learn baseline, so the API only ever deals with `predict(image) -> dict`.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile

from src.config import CLASS_NAMES, MODELS_DIR, read_json

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB upload cap


class ModelNotAvailableError(RuntimeError):
    """Raised when no exported model can be found on disk."""


def decode_image(payload: bytes) -> Image.Image:
    """Bytes -> RGB PIL image, with explicit errors for bad or oversized uploads."""
    if not payload:
        raise ValueError("Empty image payload.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except Exception as exc:  # Pillow raises a variety of decode errors
        raise ValueError(f"Could not decode the uploaded file as an image: {exc}") from exc
    return image.convert("RGB")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class Predictor:
    """Loads the exported model once and serves single-image predictions."""

    def __init__(self, model_dir: Path | str = MODELS_DIR, threshold: float = 0.5) -> None:
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.metadata: dict[str, Any] = self._load_metadata()
        self.class_names: list[str] = self.metadata.get("class_names", CLASS_NAMES)
        self.image_size: int = int(self.metadata.get("image_size", 224))
        self.family: str = self.metadata.get("model_family", "pytorch")
        self.model_name: str = self.metadata.get("model_name", "unknown")
        self._model = self._load_model()

    # -- loading ---------------------------------------------------------

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotAvailableError(
                f"No metadata.json in {self.model_dir}. Run `python -m src.models.train` first."
            )
        return read_json(metadata_path)

    def _load_model(self) -> Any:
        artifact = self.model_dir / self.metadata.get("artifact", "model.pt")
        if not artifact.exists():
            raise ModelNotAvailableError(f"Model artifact missing: {artifact}")

        if self.family == "pytorch":
            import torch

            model = torch.jit.load(str(artifact), map_location="cpu")
            model.eval()
            return model

        import joblib

        return joblib.load(artifact)

    # -- inference -------------------------------------------------------

    def _preprocess_torch(self, image: Image.Image):
        from src.data.datasets import image_to_tensor

        return image_to_tensor(image, self.image_size)

    def _preprocess_sklearn(self, image: Image.Image) -> np.ndarray:
        size = int(self.metadata.get("baseline_image_size", 64))
        arr = np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        return arr.reshape(1, -1)

    def predict_proba(self, image: Image.Image) -> np.ndarray:
        """Return class probabilities in the order of `self.class_names`."""
        if self.family == "pytorch":
            import torch

            with torch.no_grad():
                logits = self._model(self._preprocess_torch(image))
                return torch.softmax(logits, dim=1)[0].numpy()
        return self._model.predict_proba(self._preprocess_sklearn(image))[0]

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Full prediction payload: label, confidence and per-class probabilities."""
        probabilities = self.predict_proba(image)
        index = int(np.argmax(probabilities))
        return {
            "label": self.class_names[index],
            "confidence": round(float(probabilities[index]), 6),
            "probabilities": {
                name: round(float(prob), 6)
                for name, prob in zip(self.class_names, probabilities, strict=True)
            },
            "model_name": self.model_name,
        }

    def predict_bytes(self, payload: bytes) -> dict[str, Any]:
        return self.predict(decode_image(payload))

    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_family": self.family,
            "artifact_format": self.metadata.get("artifact_format"),
            "image_size": self.image_size,
            "class_names": self.class_names,
            "trained_at": self.metadata.get("trained_at"),
            "mlflow_run_id": self.metadata.get("mlflow_run_id"),
            "test_metrics": self.metadata.get("test_metrics", {}),
        }
