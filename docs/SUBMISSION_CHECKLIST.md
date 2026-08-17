# Submission checklist — Assignment 2 (50 marks)

Every rubric item, where it is implemented, and how to demonstrate it. Tick the
"evidence" column as you capture each screenshot or recording segment.

## M1 · Model Development & Experiment Tracking (10M)

| Task | Implementation | How to show it |
|---|---|---|
| Git source versioning | Repository with a feature-wise commit history | `git log --oneline` |
| DVC dataset versioning | `dvc.yaml` (3 stages), `dvc.lock`, `.dvc/config` with a configured remote | `dvc dag`, `dvc status` |
| Baseline model | Logistic regression on flattened pixels + a from-scratch CNN + MobileNetV3 transfer model | `src/models/architectures.py`, training console output |
| Serialized model | `models/model.pt` (TorchScript), `models/model_state_dict.pt`; the sklearn baseline is logged to its MLflow run | `ls -la models/` |
| Experiment tracking | MLflow: 3 runs, params, per-epoch metrics, artifacts | MLflow UI — run comparison + artifacts tab |
| Confusion matrix & loss curves | Logged to each run under `plots/` | MLflow UI artifacts, `docs/plots/` |

## M2 · Model Packaging & Containerization (10M)

| Task | Implementation | How to show it |
|---|---|---|
| REST API | FastAPI in `src/api/main.py` | `http://localhost:8000/docs` |
| Health endpoint | `GET /health` → status, model_loaded, version, uptime | `curl -s localhost:8000/health` |
| Prediction endpoint | `POST /predict` (multipart) and `POST /predict/base64` → label + probabilities | `curl -F "file=@…jpg" localhost:8000/predict` |
| requirements.txt | Runtime pins in `requirements.txt`, dev pins in `requirements-dev.txt` | Show the pinned versions |
| Version pinning | Every ML library pinned to an exact version | `cat requirements.txt` |
| Dockerfile | Multi-stage, non-root user, `HEALTHCHECK`, CPU-only torch | `cat Dockerfile` |
| Build & verify locally | `make docker-build && make docker-run` | `curl`/Postman prediction against the container |

Postman: import `docs/postman_collection.json` and run the collection.

## M3 · CI Pipeline (10M)

| Task | Implementation | How to show it |
|---|---|---|
| Pre-processing unit tests | `tests/test_preprocess.py` — validation, resizing, split ratios | `make test` |
| Inference unit tests | `tests/test_serving.py` + `tests/test_api.py` | `make test` |
| Tests run via pytest | 42 tests, coverage reported | Console output |
| CI on push/PR | `.github/workflows/ci.yml` | GitHub Actions run page |
| Checkout → install → test → build | Steps in the `test` and `build-and-push` jobs | Expand the job steps in the UI |
| Artifact publishing | Image pushed to `ghcr.io/sahithisiripuram/mlops-assignment-02` | GitHub → Packages |

## M4 · CD Pipeline & Deployment (10M)

| Task | Implementation | How to show it |
|---|---|---|
| Deployment target | Kubernetes (kind/minikube) and Docker Compose | `kubectl get pods,svc` |
| Infrastructure manifests | `deployment/k8s/deployment.yaml` + `service.yaml`; `deployment/docker-compose.yml` | Show the files |
| Pull image from registry | CD job pulls the exact tag CI published | CD run log |
| Auto-deploy on main | `.github/workflows/cd.yml` triggered by a successful CI run on `main` | CD run page |
| Smoke test | `deployment/smoke_test.py` — health + one prediction, 12 checks | `ALL SMOKE TESTS PASSED` in the log |
| Fail pipeline on smoke failure | Non-zero exit fails the job; a rollback step runs on failure | Show the step configuration |

## M5 · Monitoring, Logs & Final Submission (10M)

| Task | Implementation | How to show it |
|---|---|---|
| Request/response logging | Structured JSON logs per request; image bytes excluded | `docker logs cats-vs-dogs-inference` |
| Request count & latency | `/stats` counters and Prometheus histograms | `curl -s localhost:8000/stats \| jq` |
| Prometheus metrics | `/metrics` + a Prometheus service in Compose | `curl -s localhost:8000/metrics`, Prometheus UI at :9090 |
| Post-deployment tracking | `scripts/monitor_performance.py` replays held-out images with true labels via `/feedback` | `make monitor`, `docs/monitoring_report.json` |

## Deliverables

- [ ] **Zip** — `bash scripts/package_submission.sh` produces
      `AIMLCZG523_Assignment02_SahithiSiripuram.zip` with source, configs
      (DVC, CI/CD, Docker, k8s manifests) and the trained model artifacts.
- [ ] **Screen recording (< 5 min)** — follow `docs/DEMO_SCRIPT.md`: code change →
      tests → image build → CI/CD run → deployed prediction → monitoring.

## Pre-submission verification

```bash
make test                                   # 42 tests pass
make lint                                   # ruff clean
make docker-build && make docker-run
make smoke                                  # all smoke checks pass
make monitor                                # live accuracy report
bash scripts/package_submission.sh
```
