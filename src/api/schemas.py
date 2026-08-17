"""Request/response models for the inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool
    model_name: str | None = None
    version: str
    uptime_seconds: float


class PredictionResponse(BaseModel):
    request_id: str
    label: str = Field(..., examples=["dogs"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_name: str
    latency_ms: float
    filename: str | None = None


class Base64PredictionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG or PNG bytes.")
    filename: str | None = None


class FeedbackRequest(BaseModel):
    """Post-deployment ground truth for one prediction."""

    request_id: str | None = None
    predicted: str = Field(..., examples=["dogs"])
    actual: str = Field(..., examples=["cats"])


class FeedbackResponse(BaseModel):
    recorded: bool
    correct: bool
    live_accuracy: float
    samples: int


class StatsResponse(BaseModel):
    service: dict[str, Any]
    model: dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
