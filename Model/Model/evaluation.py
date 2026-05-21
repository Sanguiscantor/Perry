from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def format_metric_bar(value: float, width: int = 24) -> str:
    filled = int(round(value * width))
    empty = width - filled
    return f"[{'#' * filled + ' ' * empty}] {value:.4f}"


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Sequence[float],
) -> Dict[str, Any]:
    """Compute evaluation metrics and preserve predictions and probabilities."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "predictions": list(map(int, y_pred)),
        "probabilities": [float(x) for x in y_proba],
    }


def extract_feature_weights(pipeline: Any, feature_names: Sequence[str]) -> Dict[str, float]:
    """Extract logistic regression weights from the trained pipeline."""
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        model = pipeline.named_steps["model"]
    else:
        model = pipeline

    coef = np.ravel(getattr(model, "coef_", np.asarray([])))
    if coef.size == 0:
        return {name: 0.0 for name in feature_names}

    return {name: float(weight) for name, weight in zip(feature_names, coef)}


def analyze_diagnostics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect overfitting, unstable performance, and train/test accuracy divergence."""
    notes = []
    train_accuracies = [split["train_metrics"]["accuracy"] for split in results]
    test_accuracies = [split["test_metrics"]["accuracy"] for split in results]
    train_test_gaps = [
        split["train_metrics"]["accuracy"] - split["test_metrics"]["accuracy"]
        for split in results
    ]

    if train_test_gaps:
        max_gap = float(max(train_test_gaps))
        if max_gap > 0.10:
            notes.append(
                f"Potential overfitting detected: maximum train/test accuracy gap is {max_gap:.3f}."
            )

    if len(test_accuracies) > 1:
        test_std = float(np.std(test_accuracies, ddof=0))
        if test_std > 0.05:
            notes.append(
                f"Unstable performance: test accuracy standard deviation is {test_std:.3f}."
            )
        test_shifts = np.abs(np.diff(test_accuracies))
        if test_shifts.max() > 0.10:
            notes.append(
                f"Large test accuracy swing detected: max one-step change is {float(test_shifts.max()):.3f}."
            )

    if any(gap > 0.15 for gap in train_test_gaps):
        notes.append(
            "The model may be memorizing training history rather than generalizing to future unseen windows."
        )

    return {
        "train_accuracy_mean": float(np.mean(train_accuracies)) if train_accuracies else 0.0,
        "test_accuracy_mean": float(np.mean(test_accuracies)) if test_accuracies else 0.0,
        "train_test_gap_max": float(max(train_test_gaps)) if train_test_gaps else 0.0,
        "test_accuracy_std": float(np.std(test_accuracies, ddof=0))
        if len(test_accuracies) > 1
        else 0.0,
        "notes": notes,
    }


def print_pipeline_report(report: Dict[str, Any]) -> None:
    """Print a concise summary of walk-forward evaluation results."""
    print("\n===== Walk-forward evaluation summary =====")
    for split in report["results"]:
        print(
            f"Split {split['split']}: train {split['train_start']} -> {split['train_end']}, "
            f"test {split['test_start']} -> {split['test_end']}"
        )
        print(f"  train accuracy {format_metric_bar(split['train_metrics']['accuracy'])}")
        print(f"  test  accuracy {format_metric_bar(split['test_metrics']['accuracy'])}")
        print(
            f"  precision={split['test_metrics']['precision']:.4f}, "
            f"recall={split['test_metrics']['recall']:.4f}"
        )
        print(f"  confusion_matrix={split['test_metrics']['confusion_matrix']}")
        top_weights = sorted(
            split["feature_weights"].items(),
            key=lambda item: -abs(item[1]),
        )[:5]
        print(f"  top weights={top_weights}\n")

    print("Diagnostics:")
    print(f"  mean train accuracy: {report['diagnostics']['train_accuracy_mean']:.4f}")
    print(f"  mean test accuracy : {report['diagnostics']['test_accuracy_mean']:.4f}")
    print(f"  max train/test gap : {report['diagnostics']['train_test_gap_max']:.4f}")
    print(f"  test accuracy std  : {report['diagnostics']['test_accuracy_std']:.4f}")
    if report["diagnostics"]["notes"]:
        for note in report["diagnostics"]["notes"]:
            print(f"  - {note}")
    else:
        print("  No strong overfitting or instability signals detected.")
