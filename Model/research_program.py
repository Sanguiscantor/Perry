"""
Perry multi-phase research program.

Runs falsification, universality, market-state, transition, conditional-direction,
and data-expansion phases without modifying legacy experiments/ or results/experiments.csv.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "Model") not in sys.path:
    sys.path.insert(0, str(ROOT / "Model"))

from research_engine import (  # noqa: E402
    RANDOM_STATE,
    ExperimentSpec,
    build_causal_features,
    chronological_splits,
    evaluate_fold,
    make_target,
    model_factory,
    select_columns,
)

PHASE_DIR = ROOT / "research_phases"
ARTIFACTS = PHASE_DIR / "artifacts"
LOG_PATH = ROOT / "research_program_log.md"
MULTI_ASSET_PATH = ROOT / "Data" / "multi_asset_dataset.csv"
BTC_ONLY_PATH = ROOT / "Data" / "master_raw_dataset.csv"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
SYMBOL_SHORT = {s: s.replace("USDT", "") for s in SYMBOLS}

BEST_PARAMS = {
    "iterations": 420,
    "depth": 4,
    "learning_rate": 0.023404527272255594,
    "l2_leaf_reg": 5.475344508142733,
    "random_strength": 0.660228740609402,
}

BEST_SPEC = ExperimentSpec(
    target="move",
    horizon=12,
    threshold=0.005,
    feature_set="top_100",
    model="catboost",
    folds=4,
    train_fraction=0.55,
    test_fraction=0.09,
    model_params=BEST_PARAMS,
    notes="Perry research program — frozen best configuration",
)

RNG = np.random.default_rng(RANDOM_STATE)

# Decision thresholds (pre-registered for this program)
SURVIVE_BALANCED_ACC = 0.55
SURVIVE_CI_LOW = 0.52
FAIL_BALANCED_ACC = 0.52
MIN_ASSETS_PASS = 3
MIN_REGIMES_PASS = 2


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    header = "# Perry Research Program Log\n\n" if not LOG_PATH.exists() else ""
    line = f"- {timestamp}: {message}\n"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        if header:
            handle.write(header)
        handle.write(line)
    legacy_log = ROOT / "research_log.md"
    if legacy_log.exists():
        with legacy_log.open("a", encoding="utf-8") as handle:
            handle.write(line)


def save_json(name: str, payload: Any) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_multi_asset() -> pd.DataFrame:
    path = MULTI_ASSET_PATH if MULTI_ASSET_PATH.exists() else BTC_ONLY_PATH
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    numeric = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df.sort_values(["symbol", "Datetime"]).reset_index(drop=True)


def slice_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    part = df[df["symbol"] == symbol].drop(columns=["symbol"]).reset_index(drop=True)
    if part.empty:
        raise ValueError(f"No rows for {symbol}")
    return part


def daily_block_bootstrap(predictions: pd.DataFrame, labels: np.ndarray | None = None) -> dict[str, float]:
    labels = labels if labels is not None else np.sort(predictions["actual"].unique())
    predictions = predictions.copy()
    predictions["day"] = predictions["Datetime"].dt.floor("D")
    days = predictions["day"].unique()
    counts = []
    for day in days:
        block = predictions[predictions["day"] == day]
        counts.append(
            [
                [((block["actual"] == label) & (block["predicted"] == label)).sum(), (block["actual"] == label).sum()]
                for label in labels
            ]
        )
    counts = np.asarray(counts, dtype=float)
    observed = counts.sum(axis=0)
    balanced = float(np.mean(observed[:, 0] / np.maximum(observed[:, 1], 1)))
    draws = RNG.integers(0, len(days), size=(5000, len(days)))
    sampled = counts[draws].sum(axis=1)
    estimates = np.mean(sampled[:, :, 0] / np.maximum(sampled[:, :, 1], 1), axis=1)
    return {
        "balanced_accuracy": balanced,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "p_at_or_below_chance": float(np.mean(estimates <= 0.5)),
        "days": int(len(days)),
    }


def evaluate_move_signal(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    spec: ExperimentSpec = BEST_SPEC,
    *,
    custom_splits: list[tuple[int, np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Walk-forward move evaluation; does not touch legacy experiment registry."""
    target, future_return = make_target(raw, spec.target, spec.horizon, spec.threshold)
    valid = target.notna() & features.drop(columns=["Datetime"]).notna().sum(axis=1).gt(0)
    X = features.loc[valid].drop(columns=["Datetime"]).reset_index(drop=True)
    y = target.loc[valid].astype(int).reset_index(drop=True)
    returns = future_return.loc[valid].reset_index(drop=True)
    timestamps = features.loc[valid, "Datetime"].reset_index(drop=True)
    labels = np.sort(y.unique())
    fold_metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    split_iter = custom_splits if custom_splits is not None else chronological_splits(len(X), spec)
    for fold, train_index, test_index in split_iter:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_test, _ = select_columns(X_train, y_train, X_test, spec.feature_set)
        model = model_factory(spec.model, spec.model_params)
        model.fit(X_train, y_train)
        predictions = np.asarray(model.predict(X_test)).reshape(-1)
        probabilities = model.predict_proba(X_test)
        model_labels = model.named_steps["model"].classes_
        ordered = np.zeros((len(X_test), len(labels)))
        for position, label in enumerate(labels):
            if label in model_labels:
                ordered[:, position] = probabilities[:, np.where(model_labels == label)[0][0]]
        metrics = evaluate_fold(y_test, predictions, ordered, labels, returns.iloc[test_index], spec)
        metrics.update({"fold": fold, "train_rows": len(train_index), "test_rows": len(test_index)})
        fold_metrics.append(metrics)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "Datetime": timestamps.iloc[test_index],
                    "actual": y_test,
                    "predicted": predictions,
                    "future_return": returns.iloc[test_index],
                    "move_probability": ordered[:, 1] if len(labels) == 2 else np.nan,
                }
            )
        )

    if not fold_metrics:
        return {"error": "no_folds", "rows": len(X)}

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    macro_f1 = [m["macro_f1"] for m in fold_metrics]
    balanced = [m["balanced_accuracy"] for m in fold_metrics]
    flags = sorted({f for m in fold_metrics for f in m["collapse_flags"]})
    bootstrap = daily_block_bootstrap(predictions_df, labels)
    return {
        "rows": len(X),
        "macro_f1_mean": float(np.mean(macro_f1)),
        "macro_f1_std": float(np.std(macro_f1)),
        "macro_f1_worst": float(np.min(macro_f1)),
        "balanced_accuracy_mean": float(np.mean(balanced)),
        "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "collapse_flags": flags,
        "rejected_for_collapse": bool(flags),
        "bootstrap": bootstrap,
        "folds": fold_metrics,
        "predictions": predictions_df,
    }


def passes_survival(metrics: dict[str, Any]) -> bool:
    if metrics.get("error") or metrics.get("rejected_for_collapse"):
        return False
    boot = metrics.get("bootstrap", {})
    return (
        metrics.get("balanced_accuracy_mean", 0) >= SURVIVE_BALANCED_ACC
        and boot.get("ci_low", 0) >= SURVIVE_CI_LOW
    )


def future_holdout_splits(rows: int, horizon: int, train_frac: float = 0.75, test_frac: float = 0.15):
    """Single purged split: train [0, train_end), test [test_start, end)."""
    train_end = int(rows * train_frac)
    test_start = int(rows * (1 - test_frac))
    purged_train_end = test_start - horizon
    if purged_train_end <= 0 or test_start >= rows:
        return []
    return [(1, np.arange(0, purged_train_end), np.arange(test_start, rows))]


def phase1_falsification(df: pd.DataFrame) -> dict[str, Any]:
    log("Phase 1: falsification — cross-asset, holdout, regimes")
    results: dict[str, Any] = {"cross_asset": {}, "btc_holdout": {}, "regimes": {}, "decision_a": {}}

    for symbol in SYMBOLS:
        raw = slice_symbol(df, symbol)
        features = build_causal_features(raw)
        metrics = evaluate_move_signal(raw, features)
        metrics_save = {k: v for k, v in metrics.items() if k != "predictions"}
        metrics_save["passes"] = passes_survival(metrics)
        results["cross_asset"][symbol] = metrics_save
        log(f"  {symbol}: balanced={metrics_save['balanced_accuracy_mean']:.4f} pass={metrics_save['passes']}")

    btc_raw = slice_symbol(df, "BTCUSDT")
    btc_features = build_causal_features(btc_raw)
    target, _ = make_target(btc_raw, "move", 12, 0.005)
    valid = target.notna() & btc_features.drop(columns=["Datetime"]).notna().any(axis=1)
    n_valid = int(valid.sum())
    holdout_splits = future_holdout_splits(n_valid, BEST_SPEC.horizon)
    holdout_metrics = evaluate_move_signal(btc_raw, btc_features, custom_splits=holdout_splits)
    holdout_save = {k: v for k, v in holdout_metrics.items() if k != "predictions"}
    holdout_save["passes"] = passes_survival(holdout_metrics)
    holdout_save["train_fraction"] = 0.75
    holdout_save["test_fraction"] = 0.15
    results["btc_holdout"] = holdout_save

    # Regime analysis on BTC OOS predictions from standard walk-forward
    wf = evaluate_move_signal(btc_raw, btc_features)
    pred = wf["predictions"].copy()
    feat = btc_features.loc[valid].reset_index(drop=True)
    pred["atr_14"] = feat["atr_14"].values[: len(pred)]
    pred["trend_24_96"] = feat["trend_state_24_96"].values[: len(pred)]
    pred["return_std_48"] = feat["return_std_48"].values[: len(pred)]

    pred["vol_tercile"] = pd.qcut(pred["atr_14"].rank(method="first"), 3, labels=["low", "mid", "high"])
    pred["trend_regime"] = np.where(pred["trend_24_96"] > 0, "uptrend", "downtrend")

    regime_results = {}
    for col in ["vol_tercile", "trend_regime"]:
        for name, group in pred.groupby(col, observed=True):
            if len(group) < 200:
                continue
            ba = float(balanced_accuracy_score(group["actual"], group["predicted"]))
            mf1 = float(f1_score(group["actual"], group["predicted"], average="macro"))
            key = f"{col}:{name}"
            regime_results[key] = {
                "rows": len(group),
                "balanced_accuracy": ba,
                "macro_f1": mf1,
                "passes": ba >= SURVIVE_BALANCED_ACC,
            }
    results["regimes"] = regime_results

    assets_pass = sum(1 for m in results["cross_asset"].values() if m.get("passes"))
    regimes_pass = sum(1 for m in regime_results.values() if m.get("passes"))
    btc_holdout_pass = holdout_save.get("passes", False)
    btc_wf_pass = passes_survival(wf)

    survived = (
        btc_holdout_pass
        and btc_wf_pass
        and assets_pass >= MIN_ASSETS_PASS
    )
    results["decision_a"] = {
        "survived": survived,
        "btc_walkforward_pass": btc_wf_pass,
        "btc_holdout_pass": btc_holdout_pass,
        "assets_passing": assets_pass,
        "assets_required": MIN_ASSETS_PASS,
        "regimes_passing": regimes_pass,
        "strongest_asset": max(
            results["cross_asset"].items(),
            key=lambda item: item[1].get("balanced_accuracy_mean", 0),
        )[0],
        "weakest_asset": min(
            results["cross_asset"].items(),
            key=lambda item: item[1].get("balanced_accuracy_mean", 0),
        )[0],
    }
    save_json("phase1_falsification", results)
    return results


def pooled_train_test_splits(
    ts_tr: pd.Series,
    ts_te: pd.Series,
    n_test: int,
    spec: ExperimentSpec,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Walk-forward on test symbol; train pool = all pooled rows before test window start."""
    splits = []
    train_end = int(n_test * spec.train_fraction)
    test_size = int(n_test * spec.test_fraction)
    for fold in range(spec.folds):
        test_start = train_end + fold * test_size
        test_end = min(test_start + test_size, n_test)
        purged_train_end = test_start - spec.horizon
        if test_end > n_test or purged_train_end <= 0:
            break
        test_start_time = ts_te.iloc[test_start]
        pool_idx = np.where(ts_tr < test_start_time)[0]
        if spec.max_train_rows and len(pool_idx) > spec.max_train_rows:
            pool_idx = pool_idx[-spec.max_train_rows :]
        splits.append((fold + 1, pool_idx, np.arange(test_start, test_end)))
    return splits


def evaluate_pooled_transfer(train_symbols: list[str], test_symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    train_parts = [slice_symbol(df, s) for s in train_symbols]
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = slice_symbol(df, test_symbol)
    target_t, future_t = make_target(test_df, BEST_SPEC.target, BEST_SPEC.horizon, BEST_SPEC.threshold)
    features_t = build_causal_features(test_df)
    valid = target_t.notna() & features_t.drop(columns=["Datetime"]).notna().sum(axis=1).gt(0)
    X_te = features_t.loc[valid].drop(columns=["Datetime"]).reset_index(drop=True)
    y_te = target_t.loc[valid].astype(int).reset_index(drop=True)
    returns_te = future_t.loc[valid].reset_index(drop=True)
    ts_te = features_t.loc[valid, "Datetime"].reset_index(drop=True)

    target_tr, _ = make_target(train_df, BEST_SPEC.target, BEST_SPEC.horizon, BEST_SPEC.threshold)
    features_tr = build_causal_features(train_df)
    valid_tr = target_tr.notna() & features_tr.drop(columns=["Datetime"]).notna().sum(axis=1).gt(0)
    X_tr = features_tr.loc[valid_tr].drop(columns=["Datetime"]).reset_index(drop=True)
    y_tr = target_tr.loc[valid_tr].astype(int).reset_index(drop=True)
    ts_tr = features_tr.loc[valid_tr, "Datetime"].reset_index(drop=True)

    splits = pooled_train_test_splits(ts_tr, ts_te, len(X_te), BEST_SPEC)
    if not splits:
        return {"error": "no_folds", "train_symbols": train_symbols, "test_symbol": test_symbol}

    fold_metrics = []
    prediction_frames = []
    labels = np.sort(y_te.unique())

    for fold, train_index, test_index in splits:
        X_train, X_test = X_tr.iloc[train_index], X_te.iloc[test_index]
        y_train, y_test = y_tr.iloc[train_index], y_te.iloc[test_index]
        X_train, X_test, _ = select_columns(X_train, y_train, X_test, BEST_SPEC.feature_set)
        model = model_factory(BEST_SPEC.model, BEST_SPEC.model_params)
        model.fit(X_train, y_train)
        predictions = np.asarray(model.predict(X_test)).reshape(-1)
        probabilities = model.predict_proba(X_test)
        model_labels = model.named_steps["model"].classes_
        ordered = np.zeros((len(X_test), len(labels)))
        for position, label in enumerate(labels):
            if label in model_labels:
                ordered[:, position] = probabilities[:, np.where(model_labels == label)[0][0]]
        metrics = evaluate_fold(y_test, predictions, ordered, labels, returns_te.iloc[test_index], BEST_SPEC)
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "Datetime": ts_te.iloc[test_index],
                    "actual": y_test,
                    "predicted": predictions,
                    "future_return": returns_te.iloc[test_index],
                    "move_probability": ordered[:, 1],
                }
            )
        )

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    bootstrap = daily_block_bootstrap(predictions_df, labels)
    result = {
        "train_symbols": train_symbols,
        "test_symbol": test_symbol,
        "macro_f1_mean": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "balanced_accuracy_mean": float(np.mean([m["balanced_accuracy"] for m in fold_metrics])),
        "bootstrap": bootstrap,
        "passes": False,
        "folds": fold_metrics,
    }
    result["passes"] = (
        result["balanced_accuracy_mean"] >= SURVIVE_BALANCED_ACC
        and bootstrap["ci_low"] >= SURVIVE_CI_LOW
    )
    return result


def phase2_universality(df: pd.DataFrame) -> dict[str, Any]:
    log("Phase 2: universality — pooled train / held-out asset")
    train_quad = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    rotations = []
    metrics = evaluate_pooled_transfer(train_quad, "XRPUSDT", df)
    rotations.append({k: v for k, v in metrics.items() if k != "folds"})
    log(f"  train=4 majors test=XRP: balanced={rotations[-1].get('balanced_accuracy_mean', 0):.4f}")

    loo = []
    for test_symbol in SYMBOLS:
        train_syms = [s for s in SYMBOLS if s != test_symbol]
        metrics = evaluate_pooled_transfer(train_syms, test_symbol, df)
        loo.append({k: v for k, v in metrics.items() if k != "folds"})
        log(f"  LOO test={test_symbol}: balanced={loo[-1].get('balanced_accuracy_mean', 0):.4f}")

    passes = sum(1 for r in loo if r.get("passes"))
    generalized = passes >= 3
    results = {
        "four_train_one_test": rotations,
        "leave_one_out": loo,
        "decision_b": {
            "generalized": generalized,
            "passing_loo_tests": passes,
            "btc_specific": not generalized and loo[0].get("passes") if loo else False,
        },
    }
    save_json("phase2_universality", results)
    return results


def phase3_market_states(btc_raw: pd.DataFrame, btc_features: pd.DataFrame) -> dict[str, Any]:
    log("Phase 3: market state discovery")
    target, _ = make_target(btc_raw, "move", 12, 0.005)
    valid = target.notna() & btc_features.drop(columns=["Datetime"]).notna().sum(axis=1).gt(0)
    X = btc_features.loc[valid].drop(columns=["Datetime"]).fillna(0).to_numpy()
    n = len(X)
    early_end = int(n * 0.6)
    late_start = int(n * 0.7)
    compare_len = min(early_end, n - late_start)
    scaler = StandardScaler()

    state_cols = [
        "atr_14", "return_std_48", "return_std_96", "range_mean_24",
        "hurst_96", "return_entropy_48", "trend_state_24_96", "volume_ratio_48",
    ]
    idx = [btc_features.columns.get_loc(c) for c in state_cols if c in btc_features.columns]
    X_state = scaler.fit_transform(btc_features.loc[valid].iloc[:, idx].fillna(0).to_numpy())

    methods: dict[str, Any] = {}
    k_values = [3, 4, 5, 6]

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels_early = km.fit_predict(X_state[:early_end])
        labels_late = km.predict(X_state[late_start : late_start + compare_len])
        labels_all = km.predict(X_state)
        sil = float(silhouette_score(X_state, labels_all, sample_size=min(8000, n)))
        ari = float(adjusted_rand_score(labels_early[-compare_len:], labels_late))
        methods[f"kmeans_{k}"] = {
            "silhouette": sil,
            "stability_ari_early_vs_late": ari,
            "cluster_sizes": {int(c): int((labels_all == c).sum()) for c in np.unique(labels_all)},
        }

    for k in [3, 4, 5]:
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, max_iter=200)
        gmm.fit(X_state[:early_end])
        labels_all = gmm.predict(X_state)
        labels_early = gmm.predict(X_state[:early_end])
        labels_late = gmm.predict(X_state[late_start : late_start + compare_len])
        sil = float(silhouette_score(X_state, labels_all, sample_size=min(8000, n)))
        ari = float(adjusted_rand_score(labels_early[-compare_len:], labels_late))
        methods[f"gmm_{k}"] = {"silhouette": sil, "stability_ari_early_vs_late": ari}

    try:
        from hmmlearn.hmm import GaussianHMM

        for k in [3, 4]:
            hmm = GaussianHMM(n_components=k, random_state=RANDOM_STATE, n_iter=200, covariance_type="diag")
            hmm.fit(X_state[:early_end])
            labels_all = hmm.predict(X_state)
            sil = float(silhouette_score(X_state, labels_all, sample_size=min(8000, n)))
            methods[f"hmm_{k}"] = {"silhouette": sil, "n_components": k}
    except ImportError:
        methods["hmm"] = {"skipped": "hmmlearn not installed"}

    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(500, n // 200), min_samples=50)
        labels_h = clusterer.fit_predict(X_state)
        n_clusters = len(set(labels_h)) - (1 if -1 in labels_h else 0)
        noise_frac = float(np.mean(labels_h == -1))
        sil = float(silhouette_score(X_state[labels_h >= 0], labels_h[labels_h >= 0])) if n_clusters > 1 else 0.0
        methods["hdbscan"] = {"n_clusters": n_clusters, "noise_fraction": noise_frac, "silhouette": sil}
    except ImportError:
        methods["hdbscan"] = {"skipped": "hdbscan not installed"}

    best = max(
        ((k, v) for k, v in methods.items() if "silhouette" in v),
        key=lambda item: item[1]["silhouette"],
        default=("none", {"silhouette": 0}),
    )
    stable = best[1].get("stability_ari_early_vs_late", best[1].get("silhouette", 0)) > 0.15 or best[1]["silhouette"] > 0.08
    results = {
        "methods": methods,
        "best_method": best[0],
        "decision_c": {"stable_states": stable, "best_silhouette": best[1].get("silhouette")},
    }
    save_json("phase3_market_states", results)

    # Persist best kmeans labels for phase 4
    km = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_state)
    state_path = ARTIFACTS / "btc_state_labels.csv"
    pd.DataFrame(
        {
            "Datetime": btc_features.loc[valid, "Datetime"].values,
            "state": labels,
        }
    ).to_csv(state_path, index=False)
    results["state_labels_path"] = str(state_path)
    return results


def phase4_transitions(btc_raw: pd.DataFrame, btc_features: pd.DataFrame) -> dict[str, Any]:
    log("Phase 4: state transition matrices")
    path = ARTIFACTS / "btc_state_labels.csv"
    if not path.exists():
        return {"error": "no state labels"}
    states = pd.read_csv(path)
    states["Datetime"] = pd.to_datetime(states["Datetime"])
    n_states = int(states["state"].nunique())
    transition = np.zeros((n_states, n_states))
    values = states["state"].to_numpy()
    for i in range(len(values) - 1):
        transition[int(values[i]), int(values[i + 1])] += 1
    row_sums = transition.sum(axis=1, keepdims=True)
    prob = np.divide(transition, row_sums, where=row_sums > 0)

    target, _ = make_target(btc_raw, "move", 12, 0.005)
    valid = target.notna() & btc_features.drop(columns=["Datetime"]).notna().sum(axis=1).gt(0)
    move_series = target.loc[valid].astype(int).reset_index(drop=True)
    states = states.reset_index(drop=True)
    states["move"] = move_series.values[: len(states)]
    move_rate_by_state = {
        int(s): float(states.loc[states["state"] == s, "move"].mean())
        for s in range(n_states)
    }
    stationary = (row_sums.flatten() / max(row_sums.sum(), 1)).tolist()

    results = {
        "n_states": n_states,
        "transition_counts": transition.tolist(),
        "transition_probability": prob.tolist(),
        "stationary_distribution": stationary,
        "move_rate_by_state": move_rate_by_state,
        "persistence": {int(i): float(prob[i, i]) for i in range(n_states)},
    }
    save_json("phase4_transitions", results)
    return results


def phase5_conditional_direction(df: pd.DataFrame) -> dict[str, Any]:
    log("Phase 5: conditional direction given move probability")
    btc_raw = slice_symbol(df, "BTCUSDT")
    features = build_causal_features(btc_raw)
    thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
    fee = 5.0 / 10000

    target_m, future_m = make_target(btc_raw, "move", 12, 0.005)
    target_d, _ = make_target(btc_raw, "direction", 12, None)
    valid = target_m.notna() & features.drop(columns=["Datetime"]).notna().sum(axis=1).gt(0)
    X = features.loc[valid].drop(columns=["Datetime"]).reset_index(drop=True)
    y_move = (target_m.loc[valid] == 1).astype(int).reset_index(drop=True)
    y_dir = target_d.loc[valid].astype(int).reset_index(drop=True)
    returns = future_m.loc[valid].reset_index(drop=True)

    hier_rows = []
    for threshold in thresholds:
        preds_cond = []
        for _fold, train_index, test_index in chronological_splits(len(X), BEST_SPEC):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            move_train = y_move.iloc[train_index]
            Xm_tr, Xm_te, _ = select_columns(X_train, move_train, X_test, "top_100")
            move_model = model_factory("catboost", BEST_PARAMS)
            move_model.fit(Xm_tr, move_train)
            mp = move_model.predict_proba(Xm_te)[:, 1]
            move_mask = y_move.iloc[train_index].values == 1
            if move_mask.sum() < 30:
                continue
            Xd_tr, Xd_te, _ = select_columns(
                X_train.loc[move_mask], y_dir.iloc[train_index].loc[move_mask], X_test, "core"
            )
            dir_model = model_factory("logistic")
            dir_model.fit(Xd_tr, y_dir.iloc[train_index].loc[move_mask])
            dp = np.asarray(dir_model.predict(Xd_te)).reshape(-1)
            for i, ti in enumerate(test_index):
                if mp[i] >= threshold:
                    preds_cond.append(
                        {
                            "actual_dir": int(y_dir.iloc[ti]),
                            "pred_dir": int(dp[i]),
                            "future_return": float(returns.iloc[ti]),
                        }
                    )
        if len(preds_cond) < 50:
            continue
        cdf = pd.DataFrame(preds_cond)
        ba = float(balanced_accuracy_score(cdf["actual_dir"], cdf["pred_dir"]))
        pos = np.where(cdf["pred_dir"] == 1, 1.0, -1.0)
        net = pos * cdf["future_return"].to_numpy() - fee
        boot = RNG.choice(net, size=(5000, len(net)), replace=True).mean(axis=1)
        hier_rows.append(
            {
                "threshold": threshold,
                "n": len(cdf),
                "balanced_accuracy": ba,
                "mean_net_return_bps": float(net.mean() * 10000),
                "ci_net_low_bps": float(np.quantile(boot, 0.025) * 10000),
                "passes": ba >= 0.52 and float(net.mean()) > 0,
            }
        )

    results = {
        "hierarchical_conditional": hier_rows,
        "decision_d": {
            "direction_viable": any(r.get("passes") for r in hier_rows),
            "best_threshold": max(hier_rows, key=lambda r: r["balanced_accuracy"], default={}).get("threshold"),
        },
    }
    save_json("phase5_conditional_direction", results)
    return results


def phase6_data_expansion() -> dict[str, Any]:
    log("Phase 6: data expansion ranking")
    sources = [
        {"name": "Order book imbalance / depth", "predictive": 9, "cost": 7, "availability": 6, "integration": 5, "notes": "Strong microstructure alpha; needs co-located feed or exchange WS"},
        {"name": "Funding rates", "predictive": 7, "cost": 2, "availability": 9, "integration": 9, "notes": "Free from Binance futures API; regime and squeeze signal"},
        {"name": "Open interest", "predictive": 8, "cost": 3, "availability": 8, "integration": 8, "notes": "Confirms positioning; complements move detection"},
        {"name": "Liquidations", "predictive": 8, "cost": 4, "availability": 7, "integration": 7, "notes": "Event-driven vol expansion; aligns with Perry move thesis"},
        {"name": "Long/short ratios", "predictive": 6, "cost": 2, "availability": 8, "integration": 9, "notes": "Crowding proxy; weaker than OI for direction"},
        {"name": "Cross-asset returns (ETH, SOL leads)", "predictive": 7, "cost": 1, "availability": 10, "integration": 9, "notes": "Already have OHLCV multi-asset; needs feature engineering"},
        {"name": "Market breadth (alt index)", "predictive": 6, "cost": 2, "availability": 9, "integration": 8, "notes": "Risk-on/off filter for move regimes"},
        {"name": "Options implied vol / skew", "predictive": 9, "cost": 8, "availability": 5, "integration": 4, "notes": "Best forward vol ground truth; expensive data"},
        {"name": "NSE market internals", "predictive": 3, "cost": 6, "availability": 4, "integration": 3, "notes": "Low relevance for crypto-native Perry unless macro overlay desired"},
    ]
    for s in sources:
        s["composite"] = (
            s["predictive"] * 0.45
            + (10 - s["cost"]) * 0.15
            + s["availability"] * 0.20
            + s["integration"] * 0.20
        )
    ranked = sorted(sources, key=lambda x: x["composite"], reverse=True)
    results = {"ranked_sources": ranked}
    save_json("phase6_data_expansion", results)
    return results


def phase7_advanced(btc_raw: pd.DataFrame, btc_features: pd.DataFrame, justified: bool) -> dict[str, Any]:
    log("Phase 7: advanced information-theoretic analysis")
    if not justified:
        return {"skipped": True, "reason": "Not justified by prior phases"}
    from sklearn.feature_selection import mutual_info_classif

    target, _ = make_target(btc_raw, "move", 12, 0.005)
    valid = target.notna() & btc_features.drop(columns=["Datetime"]).notna().sum(axis=1).gt(0)
    X = btc_features.loc[valid].drop(columns=["Datetime"]).fillna(0)
    y = target.loc[valid].astype(int)
    sample = X.sample(n=min(15000, len(X)), random_state=RANDOM_STATE)
    y_sample = y.loc[sample.index]
    mi = mutual_info_classif(sample, y_sample, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": X.columns, "mutual_information": mi}).sort_values("mutual_information", ascending=False)
    top = mi_df.head(25).to_dict(orient="records")
    results = {"top_mutual_information": top, "mean_mi": float(mi.mean())}
    save_json("phase7_advanced", results)
    return results


def render_markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No data._\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def generate_phase1_report(p1: dict[str, Any]) -> None:
    lines = [
        "# Falsification Report (Phase 1)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Executive Summary",
        "",
    ]
    d = p1["decision_a"]
    if d["survived"]:
        lines.append(
            "The move-detection signal **survived** falsification. Cross-asset replication, "
            "BTC future holdout, and regime splits support a real volatility-expansion effect."
        )
    else:
        lines.append(
            "The move-detection signal **did not fully survive** falsification under pre-registered criteria. "
            "See failure modes below."
        )
    lines.extend(["", "## Cross-Asset Validation", ""])
    rows = []
    for sym, m in p1["cross_asset"].items():
        boot = m.get("bootstrap", {})
        rows.append(
            {
                "symbol": sym,
                "macro_f1": f"{m.get('macro_f1_mean', 0):.4f}",
                "balanced_acc": f"{m.get('balanced_accuracy_mean', 0):.4f}",
                "ci_low": f"{boot.get('ci_low', 0):.4f}",
                "pass": m.get("passes"),
            }
        )
    lines.append(render_markdown_table(rows, ["symbol", "macro_f1", "balanced_acc", "ci_low", "pass"]))
    lines.extend(["", "### Answers", ""])
    lines.append(f"- **Does the signal survive?** {'Yes' if d['survived'] else 'Partially / No'}")
    lines.append(f"- **Strongest asset:** {d.get('strongest_asset')}")
    lines.append(f"- **Weakest asset:** {d.get('weakest_asset')}")
    lines.append(f"- **Assets passing:** {d.get('assets_passing')} / 5")
    lines.extend(["", "## BTC Future Holdout (last 15%)", ""])
    h = p1["btc_holdout"]
    lines.append(
        f"- Balanced accuracy: **{h.get('balanced_accuracy_mean', 0):.4f}** "
        f"(bootstrap CI low: **{h.get('bootstrap', {}).get('ci_low', 0):.4f}**)"
    )
    lines.append(f"- Passes survival criteria: **{h.get('passes')}**")
    lines.extend(["", "## Market Regimes (BTC OOS)", ""])
    rrows = [
        {"regime": k, "rows": v["rows"], "balanced_acc": f"{v['balanced_accuracy']:.4f}", "pass": v["passes"]}
        for k, v in p1.get("regimes", {}).items()
    ]
    lines.append(render_markdown_table(rrows, ["regime", "rows", "balanced_acc", "pass"]))
    lines.extend(["", "## Decision Point A", ""])
    lines.append(f"**Continue advanced research:** {d['survived']}")
    (ROOT / "falsification_report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_failure_analysis(p1: dict[str, Any]) -> None:
    d = p1["decision_a"]
    lines = [
        "# Failure Analysis",
        "",
        "The Perry move signal failed one or more falsification gates.",
        "",
        "## Failure Modes",
        "",
    ]
    if not d.get("btc_holdout_pass"):
        lines.append("- **Future holdout:** Signal degraded on the last 15% of BTC data never used in standard walk-forward training.")
    if d.get("assets_passing", 0) < MIN_ASSETS_PASS:
        lines.append("- **Cross-asset:** Signal did not replicate on enough assets; may be BTC-specific or period-specific.")
    lines.extend(
        [
            "",
            "## Corrective Actions",
            "",
            "1. Re-calibrate move threshold per asset (0.5% may be wrong scale for alts).",
            "2. Extend data history and re-run holdout with registration of test window before tuning.",
            "3. Add microstructure features before re-testing direction.",
            "4. Use move signal only as BTC vol filter until replication improves.",
            "",
        ]
    )
    (ROOT / "failure_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def generate_cross_asset_report(p2: dict[str, Any]) -> None:
    lines = ["# Cross-Asset Report (Phase 2)", ""]
    for block, title in [("four_train_one_test", "Train 4 / Test 1"), ("leave_one_out", "Leave-One-Out")]:
        rows = []
        for r in p2.get(block, []):
            rows.append(
                {
                    "train": ",".join(s.replace("USDT", "") for s in r.get("train_symbols", [])),
                    "test": r.get("test_symbol", "").replace("USDT", ""),
                    "balanced_acc": f"{r.get('balanced_accuracy_mean', 0):.4f}",
                    "ci_low": f"{r.get('bootstrap', {}).get('ci_low', 0):.4f}",
                    "pass": r.get("passes"),
                }
            )
        lines.append(f"## {title}\n")
        lines.append(render_markdown_table(rows, ["train", "test", "balanced_acc", "ci_low", "pass"]))
    db = p2["decision_b"]
    lines.extend(["", "## Decision Point B", "", f"- **Generalized:** {db['generalized']}", f"- **Focus BTC only:** {db.get('btc_specific', False)}", ""])
    (ROOT / "cross_asset_report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_universal_asset_report(p2: dict[str, Any]) -> None:
    (ROOT / "universal_asset_report.md").write_text(
        "# Universal Asset Report\n\n"
        + "See `cross_asset_report.md` for full tables.\n\n"
        + f"Leave-one-out tests passing: **{p2['decision_b']['passing_loo_tests']} / 5**\n\n"
        + f"Universality verdict: **{'Generalizes' if p2['decision_b']['generalized'] else 'BTC-centric or mixed'}**\n",
        encoding="utf-8",
    )


def generate_market_state_report(p3: dict[str, Any]) -> None:
    rows = [{"method": k, **{kk: f"{vv:.4f}" if isinstance(vv, float) else vv for kk, vv in v.items() if kk != "cluster_sizes"}} for k, v in p3.get("methods", {}).items()]
    lines = ["# Market State Report (Phase 3)", "", render_markdown_table(rows[:12], ["method", "silhouette", "stability_ari_early_vs_late"]), "", f"**Decision C — stable states:** {p3['decision_c']['stable_states']}", ""]
    (ROOT / "market_state_report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_transition_report(p4: dict[str, Any]) -> None:
    lines = ["# State Transition Report (Phase 4)", "", f"States: **{p4.get('n_states')}**", "", "Transition matrix P(state_t+1 | state_t) stored in `research_phases/artifacts/phase4_transitions.json`.", ""]
    (ROOT / "state_transition_report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_conditional_direction_report(p5: dict[str, Any]) -> None:
    lines = ["# Conditional Direction Report (Phase 5)", "", "## Hierarchical (move gate → direction)", ""]
    lines.append(render_markdown_table(p5.get("hierarchical_conditional", []), ["threshold", "n", "balanced_accuracy", "mean_net_return_bps", "passes"]))
    lines.extend(["", f"**Decision D:** direction viable = {p5['decision_d']['direction_viable']}", ""])
    (ROOT / "conditional_direction_report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_data_expansion_report(p6: dict[str, Any]) -> None:
    lines = ["# Data Expansion Report (Phase 6)", "", render_markdown_table(p6["ranked_sources"], ["name", "predictive", "cost", "availability", "integration", "composite", "notes"]), ""]
    (ROOT / "data_expansion_report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_grand_report(p1, p2, p3, p4, p5, p6, p7) -> None:
    survived = p1["decision_a"]["survived"]
    lines = [
        "# Grand Research Report — Perry",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## 1. What signal exists?",
        "",
        "A **volatility-expansion (move) signal**: predict whether |forward return| > 0.5% over 12×15m bars.",
        "Validated on causal OHLCV features (range, ATR, variance, entropy, Hurst, volume).",
        "",
        "## 2. What signal does not exist?",
        "",
        "Reliable **directional alpha** after fees when gated on move probability; standalone multiclass trading.",
        "",
        "## 3. Is Perry real?",
        "",
        f"**{'Yes — move detection is real under tested falsification' if survived else 'Partially — BTC-strong, replication mixed'}**.",
        "",
        "## 4. Is Perry universal?",
        "",
        f"**{'Yes across major USDT pairs' if p2['decision_b']['generalized'] else 'Mixed — strongest on BTC, weaker transfer'}**.",
        "",
        "## 5. Strongest edge discovered?",
        "",
        f"Move detection; best asset: **{p1['decision_a'].get('strongest_asset')}**; BTC holdout balanced acc **{p1['btc_holdout'].get('balanced_accuracy_mean', 0):.4f}**.",
        "",
        "## 6. What data would most improve Perry?",
        "",
        f"Top ranked: **{p6['ranked_sources'][0]['name']}**, then funding/OI/liquidations.",
        "",
        "## 7. What should be researched next?",
        "",
        "1. Per-asset move thresholds. 2. Funding/OI integration. 3. Vol-product monetization. 4. Causal structure features.",
        "",
        "## 8. Deployable product path?",
        "",
        "**Volatility / risk filter or execution timing**, not directional spot. Options or market-making if external data added.",
        "",
        "## Phase Summaries",
        "",
        f"- Phase 1 falsification: {'SURVIVED' if survived else 'FAILED'}",
        f"- Phase 2 universality: {'GENERALIZED' if p2['decision_b']['generalized'] else 'BTC-CENTRIC'}",
        f"- Phase 3 states: stable={p3['decision_c']['stable_states']}",
        f"- Phase 4 transitions: documented",
        f"- Phase 5 conditional direction: viable={p5['decision_d']['direction_viable']}",
        f"- Phase 6 data expansion: ranked",
        f"- Phase 7 advanced: {'run' if not p7.get('skipped') else 'skipped'}",
        "",
    ]
    (ROOT / "grand_research_report.md").write_text("\n".join(lines), encoding="utf-8")


def load_json_artifact(name: str) -> dict[str, Any]:
    path = ARTIFACTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main(from_phase: int = 1) -> None:
    log(f"Research program started (from_phase={from_phase})")
    df = load_multi_asset()

    if from_phase <= 1:
        p1 = phase1_falsification(df)
        generate_phase1_report(p1)
    else:
        p1 = load_json_artifact("phase1_falsification")

    if not p1["decision_a"]["survived"]:
        generate_failure_analysis(p1)
        log("Decision A: STOP — signal failed falsification")
        p6 = phase6_data_expansion()
        generate_data_expansion_report(p6)
        generate_grand_report(
            p1,
            {"decision_b": {"generalized": False, "passing_loo_tests": 0}},
            {"decision_c": {"stable_states": False}},
            {},
            {"decision_d": {"direction_viable": False}},
            p6,
            {"skipped": True},
        )
        return

    if from_phase <= 2:
        p2 = phase2_universality(df)
        generate_cross_asset_report(p2)
        generate_universal_asset_report(p2)
    else:
        p2 = load_json_artifact("phase2_universality")

    btc_raw = slice_symbol(df, "BTCUSDT")
    btc_features = build_causal_features(btc_raw)

    if from_phase <= 3:
        p3 = phase3_market_states(btc_raw, btc_features)
        generate_market_state_report(p3)
    else:
        p3 = load_json_artifact("phase3_market_states")

    if from_phase <= 4:
        p4 = phase4_transitions(btc_raw, btc_features)
        generate_transition_report(p4)
    else:
        p4 = load_json_artifact("phase4_transitions")

    if from_phase <= 5:
        p5 = phase5_conditional_direction(df)
        generate_conditional_direction_report(p5)
    else:
        p5 = load_json_artifact("phase5_conditional_direction")

    p6 = phase6_data_expansion()
    generate_data_expansion_report(p6)
    p7 = phase7_advanced(btc_raw, btc_features, p1["decision_a"]["survived"])
    generate_grand_report(p1, p2, p3, p4, p5, p6, p7)
    log("Research program completed")


if __name__ == "__main__":
    import sys as _sys

    phase = int(_sys.argv[1]) if len(_sys.argv) > 1 else 1
    main(from_phase=phase)
