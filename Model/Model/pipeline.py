from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .data_loader import prepare_features_targets
    from .evaluation import (
        analyze_diagnostics,
        extract_feature_weights,
        evaluate_predictions,
    )
except ImportError:
    from data_loader import prepare_features_targets
    from evaluation import (
        analyze_diagnostics,
        extract_feature_weights,
        evaluate_predictions,
    )


def build_logistic_pipeline(random_state: int = 42) -> Pipeline:
    """Build a lightweight logistic regression pipeline with scaling."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=random_state,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )


def _normalize_offset(freq: str) -> pd.DateOffset:
    if freq.endswith("M") and not freq.endswith("ME"):
        freq = freq[:-1] + "ME"
    return pd.tseries.frequencies.to_offset(freq)


def generate_rolling_time_windows(
    df: pd.DataFrame,
    datetime_col: str = "Datetime",
    train_window: str = "3M",
    test_window: str = "1M",
    step_window: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate chronological walk-forward train/test windows."""
    if step_window is None:
        step_window = test_window

    df = df.copy().reset_index(drop=True)
    time_index = pd.DatetimeIndex(df[datetime_col])
    start = time_index.min()
    end = time_index.max()

    train_offset = _normalize_offset(train_window)
    test_offset = _normalize_offset(test_window)
    step_offset = _normalize_offset(step_window)

    splits: List[Dict[str, Any]] = []
    train_start = start

    while True:
        train_end = train_start + train_offset
        test_start = train_end
        test_end = test_start + test_offset

        if test_end > end + pd.Timedelta(seconds=1):
            break

        train_mask = (time_index >= train_start) & (time_index < train_end)
        test_mask = (time_index >= test_start) & (time_index < test_end)

        train_idx = df.index[train_mask].tolist()
        test_idx = df.index[test_mask].tolist()

        if train_idx and test_idx:
            splits.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )

        train_start += step_offset

    if not splits:
        raise ValueError(
            "Unable to generate walk-forward windows. Check the dataset date range or window sizes."
        )

    return splits


def run_walk_forward_pipeline(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    target_col: str = "target",
    datetime_col: str = "Datetime",
    train_window: str = "3M",
    test_window: str = "1M",
    step_window: Optional[str] = None,
    model: Optional[Pipeline] = None,
) -> Dict[str, Any]:
    """Run walk-forward training and inference on chronological windows."""
    if model is None:
        model = build_logistic_pipeline()

    splits = generate_rolling_time_windows(
        df,
        datetime_col=datetime_col,
        train_window=train_window,
        test_window=test_window,
        step_window=step_window,
    )

    results: List[Dict[str, Any]] = []

    for split_index, window in enumerate(splits, start=1):
        train_df = df.loc[window["train_idx"]].reset_index(drop=True)
        test_df = df.loc[window["test_idx"]].reset_index(drop=True)

        X_train, y_train = prepare_features_targets(
            train_df,
            feature_cols=feature_cols,
            target_col=target_col,
        )
        X_test, y_test = prepare_features_targets(
            test_df,
            feature_cols=feature_cols,
            target_col=target_col,
        )

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]

        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        train_metrics = evaluate_predictions(y_train, y_train_pred, y_train_proba)
        test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba)

        results.append(
            {
                "split": split_index,
                "train_start": window["train_start"].isoformat(),
                "train_end": window["train_end"].isoformat(),
                "test_start": window["test_start"].isoformat(),
                "test_end": window["test_end"].isoformat(),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "feature_weights": extract_feature_weights(model, X_train.columns),
            }
        )

    return {"results": results, "diagnostics": analyze_diagnostics(results)}
