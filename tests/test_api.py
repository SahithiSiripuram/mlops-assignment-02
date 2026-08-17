"""API contract tests for the inference service (M2.1 / M3.1)."""

from __future__ import annotations

import base64

from src.config import CLASS_NAMES


class TestHealth:
    def test_health_reports_a_loaded_model(self, api_client) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["model_name"] == "stub-cnn"
        assert body["uptime_seconds"] >= 0

    def test_root_lists_the_endpoints(self, api_client) -> None:
        body = api_client.get("/").json()
        assert "/predict" in body["endpoints"]
        assert "/health" in body["endpoints"]

    def test_model_endpoint_returns_metadata(self, api_client) -> None:
        body = api_client.get("/model").json()
        assert body["model_name"] == "stub-cnn"
        assert body["class_names"] == CLASS_NAMES

    def test_every_response_carries_a_request_id(self, api_client) -> None:
        response = api_client.get("/health")
        assert response.headers["X-Request-ID"]
        assert float(response.headers["X-Process-Time-Ms"]) >= 0


class TestPredict:
    def test_predict_returns_label_and_probabilities(self, api_client, sample_image_bytes) -> None:
        response = api_client.post(
            "/predict", files={"file": ("cat.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200

        body = response.json()
        assert body["label"] in CLASS_NAMES
        assert 0.0 <= body["confidence"] <= 1.0
        assert set(body["probabilities"]) == set(CLASS_NAMES)
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
        assert body["filename"] == "cat.jpg"
        assert body["latency_ms"] >= 0
        assert body["request_id"]

    def test_predict_rejects_a_non_image_upload(self, api_client) -> None:
        response = api_client.post(
            "/predict", files={"file": ("notes.txt", b"plain text", "text/plain")}
        )
        assert response.status_code == 400
        assert "decode" in response.json()["detail"].lower()

    def test_predict_requires_a_file(self, api_client) -> None:
        assert api_client.post("/predict").status_code == 422

    def test_predict_base64_matches_multipart(self, api_client, sample_image_bytes) -> None:
        encoded = base64.b64encode(sample_image_bytes).decode()
        response = api_client.post(
            "/predict/base64", json={"image_base64": encoded, "filename": "dog.jpg"}
        )
        assert response.status_code == 200
        assert response.json()["label"] in CLASS_NAMES

    def test_predict_base64_rejects_bad_payload(self, api_client) -> None:
        response = api_client.post("/predict/base64", json={"image_base64": "!!!not-base64!!!"})
        assert response.status_code == 400


class TestMonitoring:
    def test_feedback_updates_live_accuracy(self, api_client, sample_image_bytes) -> None:
        prediction = api_client.post(
            "/predict", files={"file": ("cat.jpg", sample_image_bytes, "image/jpeg")}
        ).json()

        response = api_client.post(
            "/feedback",
            json={
                "request_id": prediction["request_id"],
                "predicted": prediction["label"],
                "actual": prediction["label"],
            },
        )
        assert response.status_code == 200

        body = response.json()
        assert body["recorded"] is True
        assert body["correct"] is True
        assert body["samples"] >= 1
        assert 0.0 <= body["live_accuracy"] <= 1.0

    def test_feedback_rejects_an_unknown_label(self, api_client) -> None:
        response = api_client.post(
            "/feedback", json={"predicted": "cats", "actual": "hamsters"}
        )
        assert response.status_code == 400

    def test_stats_counts_requests_and_predictions(self, api_client, sample_image_bytes) -> None:
        api_client.post("/predict", files={"file": ("a.jpg", sample_image_bytes, "image/jpeg")})

        body = api_client.get("/stats").json()
        assert body["service"]["requests_total"] >= 1
        assert body["service"]["predictions_total"] >= 1
        assert body["service"]["latency_ms"]["avg"] >= 0
        assert sum(body["service"]["predictions_by_class"].values()) >= 1
        assert body["model"]["model_name"] == "stub-cnn"

    def test_metrics_endpoint_exposes_prometheus_series(self, api_client, sample_image_bytes) -> None:
        api_client.post("/predict", files={"file": ("a.jpg", sample_image_bytes, "image/jpeg")})

        response = api_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

        text = response.text
        assert "inference_requests_total" in text
        assert "model_predictions_total" in text
        assert "inference_request_latency_seconds" in text
