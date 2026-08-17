"""Stage 3 of the pipeline: train, track and export the Cats-vs-Dogs classifier.

Trains three models and logs each as a separate MLflow run:

  1. ``baseline``  -- logistic regression on flattened 64x64 pixels (scikit-learn)
  2. ``cnn``       -- a small CNN trained from scratch (PyTorch)
  3. ``transfer``  -- MobileNetV3-Small, ImageNet weights, retrained head (PyTorch)

Every run logs hyper-parameters, per-epoch metrics, a confusion matrix, ROC curve
and loss/accuracy curves. The model with the best validation accuracy is exported
to `models/` as TorchScript (or joblib for the baseline) together with a
`metadata.json` the inference service reads at startup.

    python -m src.models.train                 # all three models
    python -m src.models.train --models cnn    # a single model
    python -m src.models.train --smoke         # 1 epoch, tiny subset (CI)
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.config import (
    CLASS_NAMES,
    MODELS_DIR,
    NORM_MEAN,
    NORM_STD,
    PROJECT_ROOT,
    REPORTS_DIR,
    ensure_dirs,
    load_params,
    write_json,
)
from src.data.datasets import build_dataloaders, load_flattened_split
from src.models.architectures import build_model, count_parameters
from src.models.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOTS_DIR = REPORTS_DIR / "plots"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """One pass over `loader`. Trains when `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_seen = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    return (
        total_loss / max(total_seen, 1),
        total_correct / max(total_seen, 1),
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def train_torch_model(
    name: str,
    loaders: dict[str, DataLoader],
    params: dict,
    epochs: int,
) -> dict:
    """Train one PyTorch model inside its own MLflow run and return its results."""
    train_params = params["train"]
    model = build_model(name).to(DEVICE)
    total_params, trainable_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_params["learning_rate"]),
        weight_decay=float(train_params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc, best_state, epochs_without_gain = -1.0, None, 0
    patience = int(train_params["early_stopping_patience"])
    started = time.time()

    with mlflow.start_run(run_name=f"{name}-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.set_tags(
            {
                "model_family": "pytorch",
                "architecture": name,
                "dataset": "cats-vs-dogs",
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "device": str(DEVICE),
            }
        )
        mlflow.log_params(
            {
                "model": name,
                "epochs": epochs,
                "batch_size": train_params["batch_size"],
                "learning_rate": train_params["learning_rate"],
                "weight_decay": train_params["weight_decay"],
                "optimizer": "Adam",
                "image_size": params["preprocess"]["image_size"],
                "seed": train_params["seed"],
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                **{f"aug_{k}": v for k, v in train_params["augmentation"].items()},
            }
        )

        for epoch in range(1, epochs + 1):
            train_loss, train_acc, *_ = _run_epoch(model, loaders["train"], criterion, optimizer)
            val_loss, val_acc, val_labels, val_preds, val_probs = _run_epoch(
                model, loaders["val"], criterion, None
            )
            scheduler.step(val_acc)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
            print(
                f"[train:{name}] epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= patience:
                    print(f"[train:{name}] early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final held-out evaluation.
        _, test_acc, test_labels, test_preds, test_probs = _run_epoch(
            model, loaders["test"], criterion, None
        )
        test_metrics = compute_metrics(test_labels, test_preds, test_probs)
        _, _, val_labels, val_preds, val_probs = _run_epoch(model, loaders["val"], criterion, None)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)

        training_seconds = time.time() - started
        mlflow.log_metrics(
            {
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "best_val_accuracy": best_val_acc,
                "training_seconds": training_seconds,
            }
        )

        ensure_dirs(PLOTS_DIR)
        artifacts = [
            plot_training_curves(history, PLOTS_DIR / f"{name}_curves.png", f"{name}: training"),
            plot_confusion_matrix(
                test_labels, test_preds, CLASS_NAMES, PLOTS_DIR / f"{name}_confusion_matrix.png",
                f"{name}: test confusion matrix",
            ),
            plot_roc_curve(test_labels, test_probs, PLOTS_DIR / f"{name}_roc.png", f"{name}: test ROC"),
        ]
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="plots")

        history_path = PLOTS_DIR / f"{name}_history.json"
        write_json(history_path, {"history": history, "test_metrics": test_metrics})
        mlflow.log_artifact(str(history_path), artifact_path="plots")

        # Log the model itself so the run is self-contained and reproducible.
        example = next(iter(loaders["val"]))[0][:1]
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=example.numpy(),
        )

        print(f"[train:{name}] test metrics: {json.dumps(test_metrics, indent=2)}")
        return {
            "name": name,
            "family": "pytorch",
            "model": model,
            "val_accuracy": val_metrics["accuracy"],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "history": history,
            "training_seconds": training_seconds,
            "run_id": run.info.run_id,
            "total_parameters": total_params,
        }


def train_baseline(params: dict, subset: int | None = None) -> dict:
    """Logistic regression on flattened pixels -- the classical baseline run."""
    baseline_params = params["train"]["baseline"]
    image_size = int(baseline_params["image_size"])

    print("[train:baseline] loading flattened pixel arrays ...")
    x_train, y_train = load_flattened_split("train", image_size)
    x_val, y_val = load_flattened_split("val", image_size)
    x_test, y_test = load_flattened_split("test", image_size)

    if subset:
        x_train, y_train = x_train[:subset], y_train[:subset]

    started = time.time()
    with mlflow.start_run(run_name=f"baseline-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.set_tags(
            {
                "model_family": "sklearn",
                "architecture": "logistic_regression",
                "dataset": "cats-vs-dogs",
            }
        )
        mlflow.log_params(
            {
                "model": "baseline_logreg",
                "image_size": image_size,
                "n_features": int(x_train.shape[1]),
                "max_iter": baseline_params["max_iter"],
                "C": baseline_params["C"],
                "scaler": "StandardScaler",
                "seed": params["train"]["seed"],
            }
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(baseline_params["C"]),
                        max_iter=int(baseline_params["max_iter"]),
                        random_state=int(params["train"]["seed"]),
                    ),
                ),
            ]
        )
        pipeline.fit(x_train, y_train)
        training_seconds = time.time() - started

        val_metrics = compute_metrics(
            y_val, pipeline.predict(x_val), pipeline.predict_proba(x_val)[:, 1]
        )
        test_preds = pipeline.predict(x_test)
        test_probs = pipeline.predict_proba(x_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_preds, test_probs)

        mlflow.log_metrics(
            {
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                "training_seconds": training_seconds,
            }
        )

        ensure_dirs(PLOTS_DIR)
        for artifact in [
            plot_confusion_matrix(
                y_test, test_preds, CLASS_NAMES, PLOTS_DIR / "baseline_confusion_matrix.png",
                "baseline: test confusion matrix",
            ),
            plot_roc_curve(y_test, test_probs, PLOTS_DIR / "baseline_roc.png", "baseline: test ROC"),
        ]:
            mlflow.log_artifact(str(artifact), artifact_path="plots")

        mlflow.sklearn.log_model(pipeline, name="model", input_example=x_val[:1])
        print(f"[train:baseline] test metrics: {json.dumps(test_metrics, indent=2)}")

        return {
            "name": "baseline",
            "family": "sklearn",
            "model": pipeline,
            "val_accuracy": val_metrics["accuracy"],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "history": {},
            "training_seconds": training_seconds,
            "run_id": run.info.run_id,
            "baseline_image_size": image_size,
        }


def export_best(result: dict, params: dict, all_results: list[dict]) -> Path:
    """Serialize the winning model plus the metadata the API needs at startup."""
    ensure_dirs(MODELS_DIR)
    image_size = int(params["preprocess"]["image_size"])

    metadata = {
        "model_name": result["name"],
        "model_family": result["family"],
        "class_names": CLASS_NAMES,
        "image_size": image_size,
        "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
        "val_metrics": result["val_metrics"],
        "test_metrics": result["test_metrics"],
        "mlflow_run_id": result["run_id"],
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
        "selection_metric": "val_accuracy",
        "leaderboard": [
            {
                "model": r["name"],
                "val_accuracy": round(r["val_accuracy"], 4),
                "test_accuracy": round(r["test_metrics"]["accuracy"], 4),
                "test_f1": round(r["test_metrics"]["f1"], 4),
                "test_roc_auc": round(r["test_metrics"]["roc_auc"], 4),
                "training_seconds": round(r["training_seconds"], 1),
            }
            for r in sorted(all_results, key=lambda r: r["val_accuracy"], reverse=True)
        ],
    }

    if result["family"] == "pytorch":
        model = result["model"].to("cpu").eval()
        example = torch.zeros(1, 3, image_size, image_size)
        scripted = torch.jit.trace(model, example)
        scripted = torch.jit.freeze(scripted)
        model_path = MODELS_DIR / "model.pt"
        scripted.save(str(model_path))
        # Keep the raw state dict too, for resuming or fine-tuning later.
        torch.save(result["model"].state_dict(), MODELS_DIR / "model_state_dict.pt")
        metadata["artifact"] = "model.pt"
        metadata["artifact_format"] = "torchscript"
    else:
        model_path = MODELS_DIR / "model.pkl"
        joblib.dump(result["model"], model_path)
        metadata["artifact"] = "model.pkl"
        metadata["artifact_format"] = "joblib"
        metadata["baseline_image_size"] = result["baseline_image_size"]

    write_json(MODELS_DIR / "metadata.json", metadata)
    write_json(REPORTS_DIR / "model_leaderboard.json", {"leaderboard": metadata["leaderboard"]})
    print(f"[train] exported best model '{result['name']}' -> {model_path}")
    return model_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and track the Cats-vs-Dogs models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["baseline", "cnn", "transfer"],
        choices=["baseline", "cnn", "transfer"],
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override params.yaml epochs.")
    parser.add_argument("--smoke", action="store_true", help="1 epoch on a tiny subset (for CI).")
    args = parser.parse_args()

    params = load_params()
    set_seed(int(params["train"]["seed"]))

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    experiment_name = params["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=str(PROJECT_ROOT / params["mlflow"].get("artifact_location", "mlruns")),
        )
    mlflow.set_experiment(experiment_name)

    epochs = args.epochs or int(params["train"]["epochs"])
    if args.smoke:
        epochs = 1

    loaders, _ = build_dataloaders()
    if args.smoke:
        # Keep CI fast: a couple of batches per split is enough to prove the loop runs.
        # ImageFolder orders samples by class, so stride-sample to keep both classes.
        from torch.utils.data import Subset

        for split, loader in loaders.items():
            size = len(loader.dataset)
            stride = max(1, size // 64)
            subset = Subset(loader.dataset, list(range(0, size, stride))[:64])
            loaders[split] = DataLoader(subset, batch_size=16, shuffle=(split == "train"))

    results: list[dict] = []
    for model_name in args.models:
        if model_name == "baseline":
            results.append(train_baseline(params, subset=64 if args.smoke else None))
        else:
            results.append(train_torch_model(model_name, loaders, params, epochs))

    best = max(results, key=lambda r: r["val_accuracy"])
    export_best(best, params, results)

    print("\n[train] leaderboard (sorted by validation accuracy)")
    print(f"{'model':<12}{'val_acc':>10}{'test_acc':>10}{'test_f1':>10}{'test_auc':>10}")
    for r in sorted(results, key=lambda r: r["val_accuracy"], reverse=True):
        t = r["test_metrics"]
        print(
            f"{r['name']:<12}{r['val_accuracy']:>10.4f}{t['accuracy']:>10.4f}"
            f"{t['f1']:>10.4f}{t['roc_auc']:>10.4f}"
        )
    print(f"\n[train] MLflow runs stored at {PROJECT_ROOT / 'mlruns'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
