# Screen recording script (< 5 minutes)

The deliverable asks for a recording that demonstrates the complete MLOps workflow
**from a code change to a deployed model prediction**. This is a shot-by-shot plan
with the exact commands, timed to fit in five minutes.

Record at 1920×1080. Keep a terminal, a browser and the repository open beforehand.

---

## 0:00 – 0:25 · Repository tour

Show the repo tree and say what it covers.

```bash
tree -L 2 -I '.venv|data|mlruns|__pycache__'
git log --oneline | head -12
```

Point out: `src/`, `tests/`, `deployment/`, `.github/workflows/`, `dvc.yaml`, `params.yaml`.

## 0:25 – 0:55 · Data versioning (M1.1)

```bash
dvc dag                 # download -> preprocess -> train
dvc status              # data is up to date with the tracked hashes
cat data/processed/metadata.json | head -20
```

Say: raw and processed images are tracked by DVC, not Git — Git holds only the hashes.

## 0:55 – 1:35 · Experiment tracking (M1.3)

```bash
make mlflow             # http://localhost:5000
```

In the MLflow UI, show:

* the three runs — `baseline`, `cnn`, `transfer`
* the comparison view sorted by `val_accuracy`
* open the `transfer` run → **Parameters**, **Metrics** (per-epoch curves),
  **Artifacts** → `plots/transfer_confusion_matrix.png` and the logged model

## 1:35 – 2:05 · Make a code change and run the tests (M3.1)

Make a small, visible change — for example bump `SERVICE_VERSION` in
`src/api/main.py` from `1.0.0` to `1.1.0`, or add a field to the `/` response.

```bash
make test               # 42 tests pass
git add -A && git commit -m "Bump service version to 1.1.0"
```

## 2:05 – 2:35 · Container build and local run (M2.3)

```bash
make docker-build
make docker-run
curl -s http://localhost:8000/health | jq
```

## 2:35 – 3:10 · Live prediction (M2.1)

```bash
curl -s -X POST http://localhost:8000/predict \
  -F "file=@data/processed/test/dogs/dog_00003.jpg" | jq

curl -s -X POST http://localhost:8000/predict \
  -F "file=@data/processed/test/cats/cat_00003.jpg" | jq
```

Then open `http://localhost:8000/docs` and run `/predict` once from the Swagger UI
(this is also the Postman-equivalent shot).

## 3:10 – 3:55 · CI/CD (M3.2, M3.3, M4.2)

```bash
git push origin main
```

In the browser, open the **Actions** tab and show:

* the **CI** run: lint → tests → build → smoke test → push to GHCR
* the published package under **Packages** (`ghcr.io/…/mlops-assignment-02`)
* the **CD** run triggered by CI: pull image → kind cluster → `kubectl apply` →
  rollout status → post-deploy smoke test passing

## 3:55 – 4:25 · Deployment target (M4.1, M4.3)

```bash
kubectl get pods,svc
kubectl rollout status deployment/cats-vs-dogs-inference
python deployment/smoke_test.py --base-url http://localhost:8080
```

Show the smoke test printing `ALL SMOKE TESTS PASSED`.

## 4:25 – 5:00 · Monitoring (M5.1, M5.2)

```bash
curl -s http://localhost:8000/stats | jq '.service'
curl -s http://localhost:8000/metrics | grep -E "inference_requests_total|model_predictions_total|model_live_accuracy"
python scripts/monitor_performance.py --base-url http://localhost:8000 --samples 60
```

Close on the monitoring report: live accuracy versus the offline test accuracy,
latency percentiles and the predicted-class balance.

---

### Recording tips

* `⌘⇧5` on macOS records the screen; pick "Record Selected Portion" for a tight crop.
* Pre-pull the Docker base images so the build does not stall on the download.
* Increase the terminal font size to at least 16pt so the output is readable.
* Keep `jq` installed — the JSON responses are much easier to read on camera.
