"""Monitoring for the inference service: structured logs, counters and Prometheus.

Three layers, all cheap enough to run in-process:

* structured JSON request/response logs (never the image bytes themselves)
* in-app counters exposed at ``/stats`` -- request count, latency, class mix
* Prometheus metrics at ``/metrics`` for scraping
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from src.config import LOGS_DIR

REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total HTTP requests handled by the inference service.",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "End-to-end request latency in seconds.",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Predictions produced, labelled by predicted class.",
    ["label"],
    registry=REGISTRY,
)
PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Confidence of the predicted class.",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Prediction requests that failed.",
    ["reason"],
    registry=REGISTRY,
)
FEEDBACK_COUNT = Counter(
    "model_feedback_total",
    "Ground-truth labels submitted after deployment.",
    ["correct"],
    registry=REGISTRY,
)
LIVE_ACCURACY = Gauge(
    "model_live_accuracy",
    "Rolling accuracy computed from post-deployment feedback.",
    registry=REGISTRY,
)
MODEL_INFO = Gauge(
    "model_info",
    "Static model metadata, exposed as labels with a constant value of 1.",
    ["model_name", "model_family", "trained_at"],
    registry=REGISTRY,
)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """JSON logs on stdout so Docker/Kubernetes log drivers can parse them."""

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if hasattr(record, "extra_fields"):
                payload.update(record.extra_fields)  # type: ignore[attr-defined]
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    logger = logging.getLogger("cats_vs_dogs_api")
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log a structured event. Callers pass metadata only -- never raw image bytes."""
    logger.info(message, extra={"extra_fields": fields})


class ServiceStats:
    """Thread-safe in-app counters backing ``/stats``.

    Also keeps the post-deployment feedback ledger: predictions paired with the
    true label are appended to ``logs/feedback.jsonl`` and summarised as a rolling
    accuracy, which is what `model_live_accuracy` exports to Prometheus.
    """

    def __init__(self, window: int = 200, feedback_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.class_counts: dict[str, int] = {}
        self.latencies_ms: deque[float] = deque(maxlen=window)
        self.confidences: deque[float] = deque(maxlen=window)
        self.feedback_total = 0
        self.feedback_correct = 0
        self.feedback_path = feedback_path or (LOGS_DIR / "feedback.jsonl")

    def record_request(self, latency_ms: float) -> None:
        with self._lock:
            self.request_count += 1
            self.latencies_ms.append(latency_ms)

    def record_prediction(self, label: str, confidence: float) -> None:
        with self._lock:
            self.prediction_count += 1
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
            self.confidences.append(confidence)

    def record_error(self) -> None:
        with self._lock:
            self.error_count += 1

    def record_feedback(self, predicted: str, actual: str, request_id: str | None = None) -> dict:
        """Persist one ground-truth label and return the updated rolling accuracy."""
        correct = predicted == actual
        with self._lock:
            self.feedback_total += 1
            self.feedback_correct += int(correct)
            accuracy = self.feedback_correct / self.feedback_total

        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "request_id": request_id,
                        "predicted": predicted,
                        "actual": actual,
                        "correct": correct,
                    }
                )
                + "\n"
            )

        FEEDBACK_COUNT.labels(correct=str(correct).lower()).inc()
        LIVE_ACCURACY.set(accuracy)
        return {"correct": correct, "live_accuracy": round(accuracy, 4), "samples": self.feedback_total}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self.latencies_ms)
            confidences = list(self.confidences)
            summary = {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "requests_total": self.request_count,
                "predictions_total": self.prediction_count,
                "errors_total": self.error_count,
                "predictions_by_class": dict(self.class_counts),
                "latency_ms": {
                    "count": len(latencies),
                    "avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                    "p50": round(latencies[len(latencies) // 2], 2) if latencies else 0.0,
                    "p95": round(latencies[int(len(latencies) * 0.95) - 1], 2)
                    if len(latencies) >= 20
                    else (round(latencies[-1], 2) if latencies else 0.0),
                    "max": round(latencies[-1], 2) if latencies else 0.0,
                },
                "avg_confidence": round(sum(confidences) / len(confidences), 4)
                if confidences
                else 0.0,
                "post_deployment": {
                    "feedback_samples": self.feedback_total,
                    "correct": self.feedback_correct,
                    "live_accuracy": round(self.feedback_correct / self.feedback_total, 4)
                    if self.feedback_total
                    else None,
                },
            }
        return summary
