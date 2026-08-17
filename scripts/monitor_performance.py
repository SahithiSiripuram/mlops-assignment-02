"""Post-deployment model performance tracking (M5.2).

Replays a batch of held-out test images against the *deployed* service, submits the
true labels back through `/feedback`, and reports how the live model is behaving:

  * live accuracy / precision / recall / F1 against the ground truth
  * latency percentiles measured client-side
  * predicted-class balance and confidence distribution (a simple drift signal)
  * a comparison against the offline test accuracy recorded at training time

    python scripts/monitor_performance.py --base-url http://localhost:8000 --samples 60
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
CLASSES = ["cats", "dogs"]


def collect_samples(samples: int, seed: int = 7) -> list[tuple[Path, str]]:
    """Take a balanced random batch from the held-out test split."""
    rng = random.Random(seed)
    batch: list[tuple[Path, str]] = []
    per_class = max(1, samples // len(CLASSES))

    for class_name in CLASSES:
        files = sorted((TEST_DIR / class_name).glob("*.jpg"))
        if not files:
            raise SystemExit(
                f"No test images in {TEST_DIR / class_name}. "
                "Run `python -m src.data.preprocess` first."
            )
        batch += [(path, class_name) for path in rng.sample(files, min(per_class, len(files)))]

    rng.shuffle(batch)
    return batch


def evaluate_live(base_url: str, batch: list[tuple[Path, str]], send_feedback: bool = True) -> dict:
    base_url = base_url.rstrip("/")
    latencies: list[float] = []
    confidences: list[float] = []
    predicted_counts = {name: 0 for name in CLASSES}
    # confusion[true][pred]
    confusion = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    correct = 0
    failures = 0

    for path, true_label in batch:
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/predict",
                files={"file": (path.name, path.read_bytes(), "image/jpeg")},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            failures += 1
            print(f"[monitor] request failed for {path.name}: {exc}")
            continue

        latencies.append((time.perf_counter() - started) * 1000)
        result = response.json()
        predicted = result["label"]

        predicted_counts[predicted] += 1
        confidences.append(result["confidence"])
        confusion[true_label][predicted] += 1
        correct += int(predicted == true_label)

        if send_feedback:
            requests.post(
                f"{base_url}/feedback",
                json={
                    "request_id": result.get("request_id"),
                    "predicted": predicted,
                    "actual": true_label,
                },
                timeout=10,
            )

    total = len(batch) - failures
    if total == 0:
        raise SystemExit("[monitor] every request failed -- is the service running?")

    # "dogs" is the positive class, matching the offline metrics.
    tp = confusion["dogs"]["dogs"]
    fp = confusion["cats"]["dogs"]
    fn = confusion["dogs"]["cats"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ordered = sorted(latencies)

    return {
        "samples": total,
        "failed_requests": failures,
        "live_accuracy": round(correct / total, 4),
        "precision_dogs": round(precision, 4),
        "recall_dogs": round(recall, 4),
        "f1_dogs": round(f1, 4),
        "confusion_matrix": confusion,
        "predicted_class_counts": predicted_counts,
        "confidence": {
            "mean": round(statistics.mean(confidences), 4),
            "min": round(min(confidences), 4),
            "low_confidence_rate": round(
                sum(1 for c in confidences if c < 0.7) / len(confidences), 4
            ),
        },
        "latency_ms": {
            "mean": round(statistics.mean(ordered), 2),
            "p50": round(ordered[len(ordered) // 2], 2),
            "p95": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 2),
            "max": round(ordered[-1], 2),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track deployed model performance.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--no-feedback", action="store_true", help="Skip posting true labels.")
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "docs" / "monitoring_report.json")
    )
    args = parser.parse_args()

    batch = collect_samples(args.samples)
    print(f"[monitor] replaying {len(batch)} held-out test images against {args.base_url}")
    report = evaluate_live(args.base_url, batch, send_feedback=not args.no_feedback)

    # Compare live behaviour against the metrics recorded when the model was trained.
    metadata_path = PROJECT_ROOT / "models" / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        offline_accuracy = metadata.get("test_metrics", {}).get("accuracy")
        if offline_accuracy is not None:
            drift = round(report["live_accuracy"] - offline_accuracy, 4)
            report["offline_test_accuracy"] = round(offline_accuracy, 4)
            report["accuracy_delta_vs_offline"] = drift
            report["status"] = "OK" if drift > -0.05 else "DEGRADED"
        report["model_name"] = metadata.get("model_name")

    # Pull the service's own view of itself for cross-checking.
    try:
        report["service_stats"] = requests.get(f"{args.base_url}/stats", timeout=10).json()["service"]
    except requests.RequestException:
        pass

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "service_stats"}, indent=2))
    print(f"\n[monitor] report written to {output_path}")
    return 0 if report.get("status", "OK") == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
