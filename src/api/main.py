"""FastAPI inference service for the Cats-vs-Dogs classifier.

Endpoints
---------
GET  /                health//docs pointers and build info
GET  /health          liveness + readiness probe (model loaded?)
GET  /model           metadata of the currently served model
POST /predict         multipart image upload  -> label + class probabilities
POST /predict/base64  JSON base64 image       -> label + class probabilities
POST /feedback        submit the true label for post-deployment tracking
GET  /stats           in-app counters: requests, latency, class mix, live accuracy
GET  /metrics         Prometheus exposition format
"""

from __future__ import annotations

import base64
import binascii
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.monitoring import (
    LIVE_ACCURACY,
    MODEL_INFO,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    REGISTRY,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ServiceStats,
    configure_logging,
    log_event,
)
from src.api.schemas import (
    Base64PredictionRequest,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictionResponse,
)
from src.config import MODELS_DIR, load_params
from src.models.serving import ModelNotAvailableError, Predictor

SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
IMAGE_TAG = os.getenv("IMAGE_TAG", "local")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger = configure_logging(LOG_LEVEL)
stats = ServiceStats()

# Populated during startup; `None` means the service is live but not ready.
state: dict[str, object | None] = {"predictor": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup rather than per request."""
    model_dir = os.getenv("MODEL_DIR", str(MODELS_DIR))
    try:
        params = load_params()
        threshold = float(params["serving"]["default_threshold"])
        predictor = Predictor(model_dir=model_dir, threshold=threshold)
        state["predictor"] = predictor
        info = predictor.info()
        MODEL_INFO.labels(
            model_name=info["model_name"],
            model_family=info["model_family"],
            trained_at=str(info["trained_at"]),
        ).set(1)
        log_event(logger, "model_loaded", **info)
    except (ModelNotAvailableError, Exception) as exc:  # noqa: BLE001 - reported via /health
        state["load_error"] = str(exc)
        log_event(logger, "model_load_failed", error=str(exc), model_dir=model_dir)
    yield
    log_event(logger, "service_shutdown", uptime_seconds=stats.snapshot()["uptime_seconds"])


app = FastAPI(
    title="Cats vs Dogs Inference API",
    description=(
        "Binary image classification service for a pet adoption platform. "
        "Built for AIMLCZG523 MLOps Assignment 2."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


def get_predictor() -> Predictor:
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded: {state.get('load_error') or 'unknown error'}",
        )
    return predictor  # type: ignore[return-value]


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Assign a request id, time the call, and emit one structured log line."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:  # noqa: BLE001 - convert to a logged 500
        latency_ms = (time.perf_counter() - started) * 1000
        stats.record_error()
        log_event(
            logger,
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            latency_ms=round(latency_ms, 2),
            error=str(exc),
        )
        REQUEST_COUNT.labels(request.method, request.url.path, "500").inc()
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error", "request_id": request_id}
        )

    latency_ms = (time.perf_counter() - started) * 1000
    stats.record_request(latency_ms)
    REQUEST_COUNT.labels(request.method, request.url.path, str(status_code)).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(latency_ms / 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"

    # Metadata only -- request bodies and image bytes are deliberately not logged.
    log_event(
        logger,
        "request_handled",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=status_code,
        latency_ms=round(latency_ms, 2),
        client=request.client.host if request.client else None,
    )
    return response


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "cats-vs-dogs-inference",
        "version": SERVICE_VERSION,
        "image_tag": IMAGE_TAG,
        "docs": "/docs",
        "endpoints": ["/health", "/model", "/predict", "/predict/base64", "/feedback", "/stats", "/metrics"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness + readiness. Returns 200 with `model_loaded=false` if the model failed."""
    predictor = state.get("predictor")
    snapshot = stats.snapshot()
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_name=predictor.model_name if predictor is not None else None,  # type: ignore[union-attr]
        version=SERVICE_VERSION,
        uptime_seconds=snapshot["uptime_seconds"],
    )


@app.get("/model", tags=["meta"])
def model_info(predictor: Predictor = Depends(get_predictor)) -> dict:
    return {**predictor.info(), "image_tag": IMAGE_TAG, "service_version": SERVICE_VERSION}


def _predict_payload(
    payload: bytes,
    filename: str | None,
    request_id: str,
    predictor: Predictor,
) -> PredictionResponse:
    started = time.perf_counter()
    try:
        result = predictor.predict_bytes(payload)
    except ValueError as exc:
        PREDICTION_ERRORS.labels(reason="invalid_image").inc()
        stats.record_error()
        log_event(logger, "prediction_rejected", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - started) * 1000
    stats.record_prediction(result["label"], result["confidence"])
    PREDICTION_COUNT.labels(label=result["label"]).inc()
    PREDICTION_CONFIDENCE.observe(result["confidence"])

    log_event(
        logger,
        "prediction",
        request_id=request_id,
        filename=filename,
        label=result["label"],
        confidence=result["confidence"],
        inference_ms=round(latency_ms, 2),
        model_name=result["model_name"],
    )
    return PredictionResponse(
        request_id=request_id,
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_name=result["model_name"],
        latency_ms=round(latency_ms, 2),
        filename=filename,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    request: Request,
    file: UploadFile = File(..., description="JPEG or PNG image of a cat or a dog."),
    predictor: Predictor = Depends(get_predictor),
) -> PredictionResponse:
    """Classify an uploaded image and return the label with class probabilities."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    payload = await file.read()
    return _predict_payload(payload, file.filename, request_id, predictor)


@app.post("/predict/base64", response_model=PredictionResponse, tags=["inference"])
def predict_base64(
    request: Request,
    body: Base64PredictionRequest,
    predictor: Predictor = Depends(get_predictor),
) -> PredictionResponse:
    """Same as /predict, but takes JSON -- convenient for Postman and smoke tests."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    try:
        payload = base64.b64decode(body.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        PREDICTION_ERRORS.labels(reason="invalid_base64").inc()
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}") from exc
    return _predict_payload(payload, body.filename, request_id, predictor)


@app.post("/feedback", response_model=FeedbackResponse, tags=["monitoring"])
def feedback(
    request: Request, body: FeedbackRequest, predictor: Predictor = Depends(get_predictor)
) -> FeedbackResponse:
    """Record the true label for a served prediction (post-deployment tracking)."""
    valid = set(predictor.class_names)
    if body.actual not in valid or body.predicted not in valid:
        raise HTTPException(status_code=400, detail=f"Labels must be one of {sorted(valid)}")

    result = stats.record_feedback(
        predicted=body.predicted,
        actual=body.actual,
        request_id=body.request_id or getattr(request.state, "request_id", None),
    )
    log_event(logger, "feedback_recorded", **result, predicted=body.predicted, actual=body.actual)
    return FeedbackResponse(recorded=True, **result)


@app.get("/stats", tags=["monitoring"])
def service_stats() -> dict:
    """Human-readable counters: request volume, latency percentiles, live accuracy."""
    predictor = state.get("predictor")
    return {
        "service": {
            "version": SERVICE_VERSION,
            "image_tag": IMAGE_TAG,
            **stats.snapshot(),
        },
        "model": predictor.info() if predictor is not None else {"loaded": False},
    }


@app.get("/metrics", tags=["monitoring"])
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    snapshot = stats.snapshot()
    live = snapshot["post_deployment"]["live_accuracy"]
    if live is not None:
        LIVE_ACCURACY.set(live)
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
