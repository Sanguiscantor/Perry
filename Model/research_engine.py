"""Leakage-safe chronological research engine for BTCUSDT 15-minute candles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data" / "master_raw_dataset.csv"
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
LOG_PATH = ROOT / "research_log.md"
SUMMARY_PATH = RESULTS_DIR / "experiments.csv"

HORIZONS = [12, 24, 48, 72, 96, 144, 192]
THRESHOLDS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015]
WINDOWS = [4, 8, 12, 24, 48, 96, 192]
RANDOM_STATE = 42


@dataclass(frozen=True)
class ExperimentSpec:
    target: str
    horizon: int
    threshold: float | None
    feature_set: str
    model: str
    folds: int = 4
    train_fraction: float = 0.55
    test_fraction: float = 0.09
    fee_bps: float = 5.0
    max_train_rows: int | None = 45000
    model_params: dict[str, Any] | None = None
    notes: str = ""


class TemporalStackingClassifier(BaseEstimator, ClassifierMixin):
    """Binary stacker whose meta-model is trained only on later chronological rows."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    def _base_models(self):
        from catboost import CatBoostClassifier
        return [
            ExtraTreesClassifier(n_estimators=180, min_samples_leaf=12, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=self.random_state),
            CatBoostClassifier(iterations=180, depth=5, learning_rate=0.05, verbose=False, allow_writing_files=False, auto_class_weights="Balanced", random_seed=self.random_state),
        ]

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        split = int(len(X) * 0.72)
        early_X, early_y = X[:split], y[:split]
        meta_X, meta_y = X[split:], y[split:]
        early_models = self._base_models()
        for model in early_models:
            model.fit(early_X, early_y)
        meta_features = np.column_stack([model.predict_proba(meta_X)[:, 1] for model in early_models])
        self.meta_model_ = LogisticRegression(class_weight="balanced", random_state=self.random_state).fit(meta_features, meta_y)
        self.base_models_ = self._base_models()
        for model in self.base_models_:
            model.fit(X, y)
        self.classes_ = np.array([0, 1])
        self.feature_importances_ = np.mean([
            getattr(model, "feature_importances_", np.zeros(X.shape[1])) for model in self.base_models_
        ], axis=0)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        meta_features = np.column_stack([model.predict_proba(X)[:, 1] for model in self.base_models_])
        return self.meta_model_.predict_proba(meta_features)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"- {timestamp}: {message}\n")


def ensure_directories() -> None:
    for directory in [EXPERIMENTS_DIR, RESULTS_DIR, REPORTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "# Research Log\n\n"
            "Chronological predictive-signal research. All engineered features are causal at candle close. "
            "Training folds are purged by the forecast horizon.\n\n",
            encoding="utf-8",
        )


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="raise")
    df = df.sort_values(["symbol", "Datetime"]).reset_index(drop=True)
    numeric = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    if df["symbol"].nunique() != 1:
        raise ValueError("Research engine currently expects one symbol; split symbols before evaluation.")
    return df


def _rolling_entropy(series: pd.Series, window: int = 48, bins: int = 8) -> pd.Series:
    def entropy(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) < max(8, window // 3) or np.all(values == values[0]):
            return 0.0
        counts = np.histogram(values, bins=bins)[0]
        probabilities = counts[counts > 0] / counts.sum()
        return float(-(probabilities * np.log(probabilities)).sum())

    return series.rolling(window, min_periods=max(8, window // 3)).apply(entropy, raw=True)


def _rolling_hurst(series: pd.Series, window: int = 96) -> pd.Series:
    def hurst(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) < 32:
            return np.nan
        lags = np.arange(2, min(20, len(values) // 2))
        tau = np.array([np.std(values[lag:] - values[:-lag]) for lag in lags])
        valid = tau > 0
        if valid.sum() < 4:
            return np.nan
        return float(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0])

    return series.rolling(window, min_periods=window).apply(hurst, raw=True)


def _rolling_fractal_dimension(series: pd.Series, window: int = 48) -> pd.Series:
    def katz(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) < 8:
            return np.nan
        path = np.abs(np.diff(values)).sum()
        diameter = np.abs(values - values[0]).max()
        if path <= 0 or diameter <= 0:
            return 1.0
        n = len(values) - 1
        return float(np.log10(n) / (np.log10(n) + np.log10(diameter / path)))

    return series.rolling(window, min_periods=window).apply(katz, raw=True)


def build_causal_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    close, high, low, open_, volume = (df[c] for c in ["Close", "High", "Low", "Open", "Volume"])
    ret = close.pct_change()
    log_ret = np.log(close).diff()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)

    features = pd.DataFrame(index=df.index)
    features["return_1"] = ret
    features["log_return_1"] = log_ret
    features["range_pct"] = (high - low) / close
    features["body_pct"] = (close - open_) / open_
    features["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    features["lower_wick_pct"] = (np.minimum(open_, close) - low) / close
    features["close_location"] = (close - low) / (high - low).replace(0, np.nan)
    features["volume_change"] = volume.pct_change()
    features["atr_14"] = true_range.rolling(14, min_periods=7).mean() / close

    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    losses = -delta.clip(upper=0).rolling(14, min_periods=7).mean()
    features["rsi_14"] = 100 - (100 / (1 + gains / losses.replace(0, np.nan)))

    signed_volume = np.sign(delta).fillna(0) * volume
    obv = signed_volume.cumsum()
    features["obv_z_48"] = (obv - obv.rolling(48).mean()) / obv.rolling(48).std()
    typical = (high + low + close) / 3
    features["vwap_distance_24"] = close / (
        (typical * volume).rolling(24).sum() / volume.rolling(24).sum()
    ) - 1

    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    features["macd_pct"] = macd / close
    features["macd_signal_pct"] = macd.ewm(span=9, adjust=False).mean() / close
    features["macd_hist_pct"] = features["macd_pct"] - features["macd_signal_pct"]

    up_move, down_move = high.diff(), -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = true_range.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr
    minus_di = 100 * minus_dm.rolling(14).mean() / atr
    features["adx_14"] = (
        100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    ).rolling(14).mean()
    features["di_spread"] = (plus_di - minus_di) / 100

    for window in WINDOWS:
        rolling_close = close.rolling(window, min_periods=max(3, window // 2))
        rolling_ret = ret.rolling(window, min_periods=max(3, window // 2))
        rolling_vol = volume.rolling(window, min_periods=max(3, window // 2))
        mean = rolling_close.mean()
        std = rolling_close.std()
        features[f"roc_{window}"] = close.pct_change(window)
        features[f"momentum_{window}"] = close / mean - 1
        features[f"close_z_{window}"] = (close - mean) / std
        features[f"return_mean_{window}"] = rolling_ret.mean()
        features[f"return_std_{window}"] = rolling_ret.std()
        features[f"return_var_{window}"] = rolling_ret.var()
        features[f"return_skew_{window}"] = rolling_ret.skew()
        features[f"return_kurt_{window}"] = rolling_ret.kurt()
        features[f"volume_z_{window}"] = (volume - rolling_vol.mean()) / rolling_vol.std()
        features[f"volume_ratio_{window}"] = volume / rolling_vol.mean()
        features[f"range_mean_{window}"] = features["range_pct"].rolling(window).mean()
        features[f"up_fraction_{window}"] = (ret > 0).rolling(window).mean()
        previous_high = high.rolling(window).max().shift(1)
        previous_low = low.rolling(window).min().shift(1)
        features[f"breakout_distance_{window}"] = close / previous_high - 1
        features[f"breakdown_distance_{window}"] = close / previous_low - 1
        features[f"donchian_position_{window}"] = (close - previous_low) / (previous_high - previous_low)

    middle = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features["bollinger_position"] = (close - middle) / (2 * std20)
    features["bollinger_width"] = 4 * std20 / middle
    ema20 = close.ewm(span=20, adjust=False).mean()
    features["keltner_position"] = (close - ema20) / (2 * true_range.rolling(20).mean())
    features["trend_state_24_96"] = close.rolling(24).mean() / close.rolling(96).mean() - 1
    features["higher_high"] = (high > high.shift(1)).astype(float)
    features["lower_low"] = (low < low.shift(1)).astype(float)
    features["confirmed_swing_high"] = (high.shift(2) > high.shift(3)) & (high.shift(2) > high.shift(1))
    features["confirmed_swing_low"] = (low.shift(2) < low.shift(3)) & (low.shift(2) < low.shift(1))
    features["confirmed_swing_high"] = features["confirmed_swing_high"].astype(float)
    features["confirmed_swing_low"] = features["confirmed_swing_low"].astype(float)
    features["return_entropy_48"] = _rolling_entropy(ret, 48)
    features["hurst_96"] = _rolling_hurst(log_ret, 96)
    features["fractal_dimension_48"] = _rolling_fractal_dimension(close, 48)
    features["return_autocorr_48"] = ret.rolling(48).apply(
        lambda values: pd.Series(values).autocorr(lag=1), raw=False
    )
    features["trend_tstat_48"] = close.rolling(48).apply(
        lambda values: np.polyfit(np.arange(len(values)), values, 1)[0] / (np.std(values) + 1e-12),
        raw=True,
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    features.insert(0, "Datetime", df["Datetime"])
    return features


def make_target(raw: pd.DataFrame, target: str, horizon: int, threshold: float | None) -> tuple[pd.Series, pd.Series]:
    future_return = raw["Close"].shift(-horizon) / raw["Close"] - 1
    if target == "multiclass":
        if threshold is None:
            raise ValueError("multiclass requires threshold")
        labels = pd.Series(np.select([future_return < -threshold, future_return > threshold], [0, 2], default=1), index=raw.index)
    elif target == "direction":
        labels = (future_return > 0).astype(int)
    elif target == "move":
        if threshold is None:
            raise ValueError("move requires threshold")
        labels = (future_return.abs() > threshold).astype(int)
    elif target == "volatility":
        rolling_volatility = future_return.abs()
        labels = pd.qcut(rolling_volatility.rank(method="first"), q=3, labels=[0, 1, 2]).astype(float)
    else:
        raise ValueError(f"Unknown target {target}")
    labels = labels.astype(float)
    labels[future_return.isna()] = np.nan
    return labels, future_return


def model_factory(name: str, params: dict[str, Any] | None = None):
    params = params or {}
    if name == "logistic":
        config = {"max_iter": 500, "class_weight": "balanced", "random_state": RANDOM_STATE} | params
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**config)),
        ])
    if name == "sgd":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1500, random_state=RANDOM_STATE, **params)),
        ])
    if name == "extra_trees":
        config = {"n_estimators": 160, "min_samples_leaf": 12, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": -1, "random_state": RANDOM_STATE} | params
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(**config)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=160, min_samples_leaf=12, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE, **params)),
        ])
    if name == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=RANDOM_STATE, **params)),
        ])
    if name == "xgboost":
        from xgboost import XGBClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.75, colsample_bytree=0.7, n_jobs=-1, random_state=RANDOM_STATE, eval_metric="mlogloss", **params)),
        ])
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(n_estimators=180, num_leaves=24, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1, **params)),
        ])
    if name == "catboost":
        from catboost import CatBoostClassifier
        config = {"iterations": 180, "depth": 5, "learning_rate": 0.05, "verbose": False, "allow_writing_files": False, "auto_class_weights": "Balanced", "random_seed": RANDOM_STATE} | params
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostClassifier(**config)),
        ])
    if name == "svm":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", cache_size=2048, random_state=RANDOM_STATE, **params)),
        ])
    if name == "voting":
        from catboost import CatBoostClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", VotingClassifier(estimators=[
                ("logistic", LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)),
                ("extra_trees", ExtraTreesClassifier(n_estimators=160, min_samples_leaf=12, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
                ("catboost", CatBoostClassifier(iterations=180, depth=5, learning_rate=0.05, verbose=False, allow_writing_files=False, auto_class_weights="Balanced", random_seed=RANDOM_STATE)),
            ], voting="soft")),
        ])
    if name == "temporal_stacking":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", TemporalStackingClassifier()),
        ])
    raise ValueError(f"Unknown model {name}")


def chronological_splits(rows: int, spec: ExperimentSpec):
    train_end = int(rows * spec.train_fraction)
    test_size = int(rows * spec.test_fraction)
    for fold in range(spec.folds):
        test_start = train_end + fold * test_size
        test_end = min(test_start + test_size, rows)
        purged_train_end = test_start - spec.horizon
        if test_end > rows or purged_train_end <= 0:
            break
        train_start = 0 if spec.max_train_rows is None else max(0, purged_train_end - spec.max_train_rows)
        yield fold + 1, np.arange(train_start, purged_train_end), np.arange(test_start, test_end)


def select_columns(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, feature_set: str):
    observed_columns = X_train.columns[X_train.notna().any(axis=0)].tolist()
    X_train = X_train[observed_columns]
    X_test = X_test[observed_columns]
    if feature_set == "all":
        return X_train, X_test, list(X_train.columns)
    if feature_set == "core":
        core = [column for column in X_train if any(token in column for token in [
            "return_", "roc_", "momentum_", "rsi", "atr", "macd", "bollinger", "adx", "di_spread", "trend_state"
        ])]
        return X_train[core], X_test[core], core
    if feature_set.startswith("top_"):
        k = min(int(feature_set.split("_")[1]), X_train.shape[1])
        imputed = SimpleImputer(strategy="median").fit_transform(X_train)
        selector = SelectKBest(score_func=f_classif, k=k).fit(imputed, y_train)
        columns = X_train.columns[selector.get_support()].tolist()
        return X_train[columns], X_test[columns], columns
    raise ValueError(f"Unknown feature set {feature_set}")


def feature_importance(model, columns: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_))
        values = values.mean(axis=0) if values.ndim > 1 else values
    else:
        values = np.zeros(len(columns))
    return pd.DataFrame({"feature": columns, "importance": values}).sort_values("importance", ascending=False)


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    try:
        if len(labels) == 2:
            positive = probabilities[:, 1]
            return float(roc_auc_score(y_true, positive)), float(average_precision_score(y_true, positive))
        binary = label_binarize(y_true, classes=labels)
        return (
            float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")),
            float(average_precision_score(binary, probabilities, average="macro")),
        )
    except ValueError:
        return None, None


def economic_metrics(target: str, predictions: np.ndarray, future_return: np.ndarray, fee_bps: float) -> dict[str, float]:
    fee = fee_bps / 10000
    if target == "multiclass":
        position = np.select([predictions == 0, predictions == 2], [-1.0, 1.0], default=0.0)
    elif target == "direction":
        position = np.where(predictions == 1, 1.0, -1.0)
    else:
        position = np.zeros(len(predictions))
    trade = position != 0
    net = position * future_return - trade.astype(float) * fee
    return {
        "trade_rate": float(trade.mean()),
        "mean_net_return": float(np.nanmean(net)),
        "median_net_return": float(np.nanmedian(net)),
        "positive_net_rate": float(np.mean(net > 0)),
    }


def save_calibration_plot(calibration: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    for fold, frame in calibration.groupby("fold"):
        axis.plot(frame["mean_predicted"], frame["fraction_positive"], marker="o", label=f"fold {fold}")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title="Calibration")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def evaluate_fold(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray, labels: np.ndarray, future_return: pd.Series, spec: ExperimentSpec) -> dict[str, Any]:
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=labels, zero_division=0
    )
    roc_auc, pr_auc = probability_metrics(y_true.to_numpy(), probabilities, labels)
    distribution = pd.Series(predictions).value_counts(normalize=True).reindex(labels, fill_value=0)
    flags = []
    if precision.min() < 0.10:
        flags.append("class_precision_below_10pct")
    if recall.min() < 0.10:
        flags.append("class_recall_below_10pct")
    if distribution.max() > 0.90:
        flags.append("predicted_class_concentration_above_90pct")
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_per_class": dict(zip(map(str, labels), map(float, precision))),
        "recall_per_class": dict(zip(map(str, labels), map(float, recall))),
        "f1_per_class": dict(zip(map(str, labels), map(float, per_class_f1))),
        "prediction_distribution": dict(zip(map(str, labels), map(float, distribution))),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "collapse_flags": flags,
        **economic_metrics(spec.target, predictions, future_return.to_numpy(), spec.fee_bps),
    }


def experiment_id(spec: ExperimentSpec) -> str:
    encoded = json.dumps(asdict(spec), sort_keys=True).encode()
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + hashlib.sha1(encoded).hexdigest()[:10]


def run_experiment(raw: pd.DataFrame, features: pd.DataFrame, spec: ExperimentSpec, save_model: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    rss_before = psutil.Process().memory_info().rss
    target, future_return = make_target(raw, spec.target, spec.horizon, spec.threshold)
    valid = target.notna() & features.drop(columns=["Datetime"]).notna().sum(axis=1).gt(0)
    X = features.loc[valid].drop(columns=["Datetime"]).reset_index(drop=True)
    y = target.loc[valid].astype(int).reset_index(drop=True)
    returns = future_return.loc[valid].reset_index(drop=True)
    timestamps = features.loc[valid, "Datetime"].reset_index(drop=True)
    labels = np.sort(y.unique())
    exp_id = experiment_id(spec)
    exp_dir = EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir(parents=True)
    fold_metrics, confusion_frames, importance_frames, prediction_frames, calibration_frames = [], [], [], [], []
    last_model = None

    for fold, train_index, test_index in chronological_splits(len(X), spec):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_test, columns = select_columns(X_train, y_train, X_test, spec.feature_set)
        model = model_factory(spec.model, spec.model_params)
        model.fit(X_train, y_train)
        predictions = np.asarray(model.predict(X_test)).reshape(-1)
        probabilities = model.predict_proba(X_test)
        model_labels = model.named_steps["model"].classes_
        ordered_probabilities = np.zeros((len(X_test), len(labels)))
        for position, label in enumerate(labels):
            if label in model_labels:
                ordered_probabilities[:, position] = probabilities[:, np.where(model_labels == label)[0][0]]
        metrics = evaluate_fold(y_test, predictions, ordered_probabilities, labels, returns.iloc[test_index], spec)
        metrics.update({
            "fold": fold,
            "train_rows": len(train_index),
            "test_rows": len(test_index),
            "train_end": timestamps.iloc[train_index[-1]].isoformat(),
            "test_start": timestamps.iloc[test_index[0]].isoformat(),
            "test_end": timestamps.iloc[test_index[-1]].isoformat(),
        })
        fold_metrics.append(metrics)
        matrix = pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels)
        matrix.insert(0, "actual", labels)
        matrix.insert(0, "fold", fold)
        confusion_frames.append(matrix)
        importance = feature_importance(model, columns).head(100)
        importance.insert(0, "fold", fold)
        importance_frames.append(importance)
        prediction_frame = pd.DataFrame({
            "fold": fold, "Datetime": timestamps.iloc[test_index], "actual": y_test,
            "predicted": predictions, "future_return": returns.iloc[test_index],
        })
        prediction_frames.append(prediction_frame)
        if len(labels) == 2:
            fraction_positive, mean_predicted = calibration_curve(y_test, ordered_probabilities[:, 1], n_bins=10, strategy="quantile")
            calibration_frames.append(pd.DataFrame({"fold": fold, "mean_predicted": mean_predicted, "fraction_positive": fraction_positive}))
        last_model = model

    if not fold_metrics:
        raise ValueError("No chronological folds were generated")
    elapsed = time.perf_counter() - started
    macro_f1 = [fold["macro_f1"] for fold in fold_metrics]
    balanced = [fold["balanced_accuracy"] for fold in fold_metrics]
    accuracy = [fold["accuracy"] for fold in fold_metrics]
    flags = sorted({flag for fold in fold_metrics for flag in fold["collapse_flags"]})
    aggregate = {
        "experiment_id": exp_id,
        **asdict(spec),
        "rows": len(X),
        "feature_count": X.shape[1],
        "macro_f1_mean": float(np.mean(macro_f1)),
        "macro_f1_std": float(np.std(macro_f1)),
        "macro_f1_worst": float(np.min(macro_f1)),
        "macro_f1_best": float(np.max(macro_f1)),
        "balanced_accuracy_mean": float(np.mean(balanced)),
        "accuracy_mean": float(np.mean(accuracy)),
        "collapse_flags": "|".join(flags),
        "rejected_for_collapse": bool(flags),
        "execution_seconds": elapsed,
        "rss_delta_mb": (psutil.Process().memory_info().rss - rss_before) / (1024 ** 2),
        "platform": platform.platform(),
    }
    (exp_dir / "config.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
    (exp_dir / "metrics.json").write_text(json.dumps({"aggregate": aggregate, "folds": fold_metrics}, indent=2), encoding="utf-8")
    pd.concat(confusion_frames).to_csv(exp_dir / "confusion_matrix.csv", index=False)
    pd.concat(importance_frames).to_csv(exp_dir / "feature_importance.csv", index=False)
    pd.concat(prediction_frames).to_csv(exp_dir / "predictions.csv", index=False)
    if calibration_frames:
        calibration = pd.concat(calibration_frames)
        calibration.to_csv(exp_dir / "calibration.csv", index=False)
        save_calibration_plot(calibration, exp_dir / "calibration_plot.png")
    if save_model and last_model is not None:
        joblib.dump(last_model, MODELS_DIR / f"{exp_id}.joblib", compress=3)
    pd.DataFrame([aggregate]).to_csv(SUMMARY_PATH, mode="a", header=not SUMMARY_PATH.exists(), index=False)
    print(f"{exp_id} {spec.target} h={spec.horizon} t={spec.threshold} {spec.feature_set} {spec.model}: macro_f1={aggregate['macro_f1_mean']:.4f} balanced={aggregate['balanced_accuracy_mean']:.4f} flags={aggregate['collapse_flags'] or '-'}")
    return aggregate


def screen_targets(raw: pd.DataFrame, features: pd.DataFrame, folds: int) -> None:
    specs = []
    for horizon in HORIZONS:
        specs.append(ExperimentSpec("direction", horizon, None, "core", "logistic", folds=folds, notes="broad target screen"))
        specs.append(ExperimentSpec("volatility", horizon, None, "core", "logistic", folds=folds, notes="broad target screen"))
        for threshold in THRESHOLDS:
            specs.append(ExperimentSpec("multiclass", horizon, threshold, "core", "logistic", folds=folds, notes="broad target screen"))
            specs.append(ExperimentSpec("move", horizon, threshold, "core", "logistic", folds=folds, notes="broad target screen"))
    log(f"Starting broad target screen with {len(specs)} configurations and {folds} folds.")
    for spec in specs:
        run_experiment(raw, features, spec)
    log("Completed broad target screen.")


def compare_models(raw: pd.DataFrame, features: pd.DataFrame, target: str, horizon: int, threshold: float | None, folds: int) -> None:
    models = ["logistic", "sgd", "extra_trees", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"]
    feature_sets = ["core", "all", "top_50", "top_100"]
    specs = [ExperimentSpec(target, horizon, threshold, feature_set, model, folds=folds, notes="model and feature-set comparison") for feature_set in feature_sets for model in models]
    log(f"Starting model comparison with {len(specs)} configurations for {target} h={horizon} threshold={threshold}.")
    for spec in specs:
        run_experiment(raw, features, spec, save_model=True)
    log("Completed model comparison.")


def tune_move(raw: pd.DataFrame, features: pd.DataFrame, folds: int, trials: int) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    specs = []
    for trial in range(trials):
        params = {
            "iterations": int(rng.choice([140, 180, 240, 320, 420])),
            "depth": int(rng.choice([4, 5, 6, 7])),
            "learning_rate": float(rng.choice([0.025, 0.04, 0.06, 0.09])),
            "l2_leaf_reg": float(rng.choice([1.0, 3.0, 5.0, 9.0])),
            "random_strength": float(rng.choice([0.0, 0.5, 1.0])),
        }
        specs.append(ExperimentSpec(
            "move", 12, 0.005, "all" if trial % 2 == 0 else "top_100", "catboost",
            folds=folds, model_params=params, notes="randomized CatBoost move-target search",
        ))
    log(f"Starting randomized CatBoost move-target search with {len(specs)} trials.")
    for spec in specs:
        run_experiment(raw, features, spec, save_model=True)
    log("Completed randomized CatBoost move-target search.")


def optuna_move(raw: pd.DataFrame, features: pd.DataFrame, folds: int, trials: int) -> None:
    import optuna

    log(f"Starting Optuna Bayesian CatBoost move-target search with {trials} trials.")

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 180, 460, step=40),
            "depth": trial.suggest_int("depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.5),
        }
        spec = ExperimentSpec(
            "move", 12, 0.005, "top_100", "catboost", folds=folds,
            model_params=params, notes="Optuna Bayesian CatBoost move-target search",
        )
        result = run_experiment(raw, features, spec, save_model=True)
        trial.set_user_attr("experiment_id", result["experiment_id"])
        trial.set_user_attr("balanced_accuracy_mean", result["balanced_accuracy_mean"])
        return result["macro_f1_mean"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=trials)
    study.trials_dataframe().to_csv(RESULTS_DIR / "optuna_move_study.csv", index=False)
    log(f"Completed Optuna search. Best macro F1: {study.best_value:.6f}; params: {study.best_params}.")


def run_hierarchy(raw: pd.DataFrame, features: pd.DataFrame, folds: int) -> None:
    for probability_threshold in [0.40, 0.50, 0.60, 0.70]:
        started = time.perf_counter()
        spec = ExperimentSpec(
            "multiclass", 12, 0.005, "top_100+core", "catboost+logistic",
            folds=folds, notes=f"hierarchical move then direction; move probability {probability_threshold}",
        )
        target, future_return = make_target(raw, "multiclass", 12, 0.005)
        valid = target.notna() & features.drop(columns=["Datetime"]).notna().sum(axis=1).gt(0)
        X = features.loc[valid].drop(columns=["Datetime"]).reset_index(drop=True)
        y = target.loc[valid].astype(int).reset_index(drop=True)
        returns = future_return.loc[valid].reset_index(drop=True)
        timestamps = features.loc[valid, "Datetime"].reset_index(drop=True)
        labels = np.array([0, 1, 2])
        exp_id = experiment_id(spec) + f"-p{int(probability_threshold * 100)}"
        exp_dir = EXPERIMENTS_DIR / exp_id
        exp_dir.mkdir(parents=True)
        fold_metrics, confusion_frames, importance_frames, prediction_frames, calibration_frames = [], [], [], [], []
        last_models = None

        for fold, train_index, test_index in chronological_splits(len(X), spec):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            move_train = (y_train != 1).astype(int)
            move_X_train, move_X_test, move_columns = select_columns(X_train, move_train, X_test, "top_100")
            move_model = model_factory("catboost")
            move_model.fit(move_X_train, move_train)
            direction_mask = y_train != 1
            direction_train = (y_train.loc[direction_mask] == 2).astype(int)
            direction_X_train, direction_X_test, direction_columns = select_columns(
                X_train.loc[direction_mask], direction_train, X_test, "core"
            )
            direction_model = model_factory("logistic")
            direction_model.fit(direction_X_train, direction_train)
            move_probability = move_model.predict_proba(move_X_test)[:, 1]
            direction_prediction = np.asarray(direction_model.predict(direction_X_test)).reshape(-1)
            predictions = np.ones(len(X_test), dtype=int)
            predicted_move = move_probability >= probability_threshold
            predictions[predicted_move] = np.where(direction_prediction[predicted_move] == 1, 2, 0)
            probabilities = np.zeros((len(X_test), 3))
            direction_probability = direction_model.predict_proba(direction_X_test)[:, 1]
            probabilities[:, 1] = 1 - move_probability
            probabilities[:, 0] = move_probability * (1 - direction_probability)
            probabilities[:, 2] = move_probability * direction_probability
            metrics = evaluate_fold(y_test, predictions, probabilities, labels, returns.iloc[test_index], spec)
            metrics.update({
                "fold": fold, "train_rows": len(train_index), "test_rows": len(test_index),
                "train_end": timestamps.iloc[train_index[-1]].isoformat(),
                "test_start": timestamps.iloc[test_index[0]].isoformat(),
                "test_end": timestamps.iloc[test_index[-1]].isoformat(),
            })
            fold_metrics.append(metrics)
            matrix = pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels)
            matrix.insert(0, "actual", labels)
            matrix.insert(0, "fold", fold)
            confusion_frames.append(matrix)
            importance = feature_importance(move_model, move_columns).head(100)
            importance.insert(0, "fold", fold)
            importance_frames.append(importance)
            prediction_frames.append(pd.DataFrame({
                "fold": fold, "Datetime": timestamps.iloc[test_index], "actual": y_test,
                "predicted": predictions, "future_return": returns.iloc[test_index],
            }))
            move_test = (y_test != 1).astype(int)
            fraction_positive, mean_predicted = calibration_curve(move_test, move_probability, n_bins=10, strategy="quantile")
            calibration_frames.append(pd.DataFrame({"fold": fold, "mean_predicted": mean_predicted, "fraction_positive": fraction_positive}))
            last_models = {"move": move_model, "direction": direction_model}

        macro_f1 = [item["macro_f1"] for item in fold_metrics]
        balanced = [item["balanced_accuracy"] for item in fold_metrics]
        accuracy = [item["accuracy"] for item in fold_metrics]
        flags = sorted({flag for item in fold_metrics for flag in item["collapse_flags"]})
        aggregate = {
            "experiment_id": exp_id, **asdict(spec),
            "rows": len(X), "feature_count": X.shape[1], "macro_f1_mean": float(np.mean(macro_f1)),
            "macro_f1_std": float(np.std(macro_f1)), "macro_f1_worst": float(np.min(macro_f1)),
            "macro_f1_best": float(np.max(macro_f1)), "balanced_accuracy_mean": float(np.mean(balanced)),
            "accuracy_mean": float(np.mean(accuracy)), "collapse_flags": "|".join(flags),
            "rejected_for_collapse": bool(flags), "execution_seconds": time.perf_counter() - started,
            "rss_delta_mb": 0.0, "platform": platform.platform(),
        }
        (exp_dir / "config.json").write_text(json.dumps({**asdict(spec), "move_probability_threshold": probability_threshold}, indent=2), encoding="utf-8")
        (exp_dir / "metrics.json").write_text(json.dumps({"aggregate": aggregate, "folds": fold_metrics}, indent=2), encoding="utf-8")
        pd.concat(confusion_frames).to_csv(exp_dir / "confusion_matrix.csv", index=False)
        pd.concat(importance_frames).to_csv(exp_dir / "feature_importance.csv", index=False)
        pd.concat(prediction_frames).to_csv(exp_dir / "predictions.csv", index=False)
        calibration = pd.concat(calibration_frames)
        calibration.to_csv(exp_dir / "calibration.csv", index=False)
        save_calibration_plot(calibration, exp_dir / "calibration_plot.png")
        joblib.dump(last_models, MODELS_DIR / f"{exp_id}.joblib", compress=3)
        pd.DataFrame([aggregate]).to_csv(SUMMARY_PATH, mode="a", header=not SUMMARY_PATH.exists(), index=False)
        print(f"{exp_id} hierarchical p={probability_threshold}: macro_f1={aggregate['macro_f1_mean']:.4f} balanced={aggregate['balanced_accuracy_mean']:.4f} flags={aggregate['collapse_flags'] or '-'}")
    log("Completed hierarchical move-then-direction evaluation.")


def run_advanced_models(raw: pd.DataFrame, features: pd.DataFrame, folds: int) -> None:
    specs = []
    for feature_set in ["core", "all", "top_100"]:
        specs.append(ExperimentSpec("move", 12, 0.005, feature_set, "svm", folds=folds, max_train_rows=20000, notes="advanced model comparison"))
        specs.append(ExperimentSpec("move", 12, 0.005, feature_set, "voting", folds=folds, notes="soft voting comparison"))
        specs.append(ExperimentSpec("move", 12, 0.005, feature_set, "temporal_stacking", folds=folds, notes="chronological inner-holdout stacking"))
    log(f"Starting advanced-model comparison with {len(specs)} configurations.")
    for spec in specs:
        run_experiment(raw, features, spec, save_model=True)
    log("Completed advanced-model comparison.")


def run_tsfresh_screen(raw: pd.DataFrame, folds: int) -> None:
    tsfresh_path = ROOT / "Model" / "features_data" / "tsfresh_benchmark_features.csv"
    log("Loading causal rolling-window TSFresh matrix for fold-local subset screen.")
    tsfresh = pd.read_csv(tsfresh_path)
    row_index = tsfresh["window_end_idx"].astype(int).to_numpy()
    tsfresh_raw = raw.iloc[row_index].reset_index(drop=True)
    drop_columns = ["series_id", "window_end_idx", "target_d"]
    tsfresh_features = tsfresh.drop(columns=drop_columns).rename(columns={"window_end_time": "Datetime"})
    tsfresh_features["Datetime"] = pd.to_datetime(tsfresh_features["Datetime"])
    for feature_set in ["top_50", "top_100", "top_250", "top_500", "top_1000", "all"]:
        spec = ExperimentSpec("move", 12, 0.005, feature_set, "extra_trees", folds=folds, max_train_rows=24000, notes="TSFresh causal windows with fold-local feature selection")
        run_experiment(tsfresh_raw, tsfresh_features, spec)
    log("Completed fold-local TSFresh feature screen.")


def summarize(limit: int = 50) -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("Run experiments first")
    results = pd.read_csv(SUMMARY_PATH)
    ranked = results.sort_values(["rejected_for_collapse", "macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean"], ascending=[True, False, False, False])
    ranked.head(limit).to_csv(REPORTS_DIR / "top_50_configurations.csv", index=False)
    print(ranked.head(limit)[["experiment_id", "target", "horizon", "threshold", "feature_set", "model", "macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean", "collapse_flags"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build-features", "screen-targets", "compare-models", "tune-move", "optuna-move", "run-hierarchy", "advanced-models", "tsfresh-screen", "summarize"])
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--target", default="multiclass", choices=["multiclass", "direction", "move", "volatility"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.004)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    ensure_directories()
    cache_path = RESULTS_DIR / "causal_features.pkl"
    if args.command == "build-features" or not cache_path.exists():
        raw = load_raw()
        features = build_causal_features(raw)
        features.to_pickle(cache_path)
        log(f"Built causal feature cache with {len(features)} rows and {features.shape[1] - 1} features.")
        print(f"Saved {cache_path}: {features.shape}")
        if args.command == "build-features":
            return
    raw = load_raw()
    features = pd.read_pickle(cache_path)
    if args.command == "screen-targets":
        screen_targets(raw, features, args.folds)
    elif args.command == "compare-models":
        threshold = None if args.target in {"direction", "volatility"} else args.threshold
        compare_models(raw, features, args.target, args.horizon, threshold, args.folds)
    elif args.command == "tune-move":
        tune_move(raw, features, args.folds, args.trials)
    elif args.command == "optuna-move":
        optuna_move(raw, features, args.folds, args.trials)
    elif args.command == "run-hierarchy":
        run_hierarchy(raw, features, args.folds)
    elif args.command == "advanced-models":
        run_advanced_models(raw, features, args.folds)
    elif args.command == "tsfresh-screen":
        run_tsfresh_screen(raw, args.folds)
    elif args.command == "summarize":
        summarize()


if __name__ == "__main__":
    main()
