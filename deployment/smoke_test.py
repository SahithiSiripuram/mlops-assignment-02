"""Post-deployment smoke test (M4.3).

Verifies a freshly deployed service end to end and exits non-zero on any failure,
which is what fails the CD pipeline:

  1. /health returns 200 and reports the model as loaded
  2. /model exposes the served model's metadata
  3. /predict classifies a real image and returns a valid probability distribution
  4. /predict rejects a malformed upload with 400
  5. /metrics exposes the Prometheus series

    python deployment/smoke_test.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sample_image_bytes() -> tuple[bytes, str]:
    """Use a real test image when the dataset is present, else a synthetic one."""
    for class_name in ["cats", "dogs"]:
        candidates = sorted((PROJECT_ROOT / "data" / "processed" / "test" / class_name).glob("*.jpg"))
        if candidates:
            return candidates[0].read_bytes(), candidates[0].name
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), (130, 110, 90)).save(buffer, format="JPEG")
    return buffer.getvalue(), "synthetic.jpg"


def wait_for_service(base_url: str, timeout: int = 120) -> None:
    """Poll /health until the service is up, or fail the pipeline."""
    deadline = time.time() + timeout
    last_error = "no attempt made"
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"[smoke] service reachable after {timeout - int(deadline - time.time())}s")
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(3)
    raise SystemExit(f"[smoke] FAIL: service never became reachable ({last_error})")


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[smoke] {status}: {name}{f' -- {detail}' if detail else ''}")
    return condition


def run_smoke_tests(base_url: str) -> int:
    base_url = base_url.rstrip("/")
    results: list[bool] = []

    # 1. Health check
    health = requests.get(f"{base_url}/health", timeout=10)
    body = health.json()
    results.append(check("health returns 200", health.status_code == 200))
    results.append(check("model is loaded", body.get("model_loaded") is True, str(body)))
    results.append(check("status is healthy", body.get("status") == "healthy"))

    # 2. Model metadata
    model = requests.get(f"{base_url}/model", timeout=10)
    results.append(check("model metadata available", model.status_code == 200))
    if model.status_code == 200:
        meta = model.json()
        print(f"[smoke]   serving '{meta.get('model_name')}' trained at {meta.get('trained_at')}")

    # 3. Prediction on a real image
    payload, filename = _sample_image_bytes()
    started = time.perf_counter()
    prediction = requests.post(
        f"{base_url}/predict",
        files={"file": (filename, payload, "image/jpeg")},
        timeout=30,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    results.append(check("predict returns 200", prediction.status_code == 200))

    if prediction.status_code == 200:
        result = prediction.json()
        probabilities = result.get("probabilities", {})
        results.append(check("label is a known class", result.get("label") in {"cats", "dogs"}, str(result.get("label"))))
        results.append(check("confidence in [0, 1]", 0.0 <= result.get("confidence", -1) <= 1.0))
        results.append(
            check(
                "probabilities sum to 1",
                abs(sum(probabilities.values()) - 1.0) < 1e-3,
                str(probabilities),
            )
        )
        print(f"[smoke]   {filename} -> {result.get('label')} "
              f"({result.get('confidence'):.4f}) in {latency_ms:.1f} ms")

    # 4. Malformed input is rejected, not crashed on
    bad = requests.post(
        f"{base_url}/predict", files={"file": ("bad.txt", b"not an image", "text/plain")}, timeout=10
    )
    results.append(check("malformed upload rejected with 400", bad.status_code == 400, f"got {bad.status_code}"))

    # 5. Monitoring endpoints
    metrics = requests.get(f"{base_url}/metrics", timeout=10)
    results.append(
        check(
            "prometheus metrics exposed",
            metrics.status_code == 200 and "inference_requests_total" in metrics.text,
        )
    )
    stats = requests.get(f"{base_url}/stats", timeout=10)
    results.append(check("stats endpoint available", stats.status_code == 200))

    passed, total = sum(results), len(results)
    print(f"\n[smoke] {passed}/{total} checks passed")
    if passed != total:
        print("[smoke] SMOKE TESTS FAILED -- deployment is not healthy")
        return 1
    print("[smoke] ALL SMOKE TESTS PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-deployment smoke test.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--wait", type=int, default=120, help="Seconds to wait for readiness.")
    args = parser.parse_args()

    wait_for_service(args.base_url, timeout=args.wait)
    return run_smoke_tests(args.base_url)


if __name__ == "__main__":
    sys.exit(main())
