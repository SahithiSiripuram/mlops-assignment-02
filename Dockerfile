# Multi-stage build: wheels are compiled in the builder, only the venv ships.
# Result is a slim, non-root image that serves the exported model over FastAPI.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps the image ~1.5 GB instead of ~6 GB with CUDA wheels.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt


FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="cats-vs-dogs-inference" \
      org.opencontainers.image.description="Cats vs Dogs binary image classifier - FastAPI inference service" \
      org.opencontainers.image.source="https://github.com/SahithiSiripuram/mlops-assignment-02" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_DIR=/app/models \
    LOG_LEVEL=INFO \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser params.yaml ./
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser models ./models

RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
