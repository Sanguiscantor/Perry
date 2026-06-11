from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable
import warnings

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from Model.research_engine import build_causal_features
from Model.features_data.market_state_discovery import build_market_state_features
from Model.features_data.market_structure import add_anchored_structure_features
from Model.features_data.state_space import build_state_space_features


ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "Data" / "datasets" / "raw"
PROCESSED_DATA_DIR = ROOT / "Data" / "datasets" / "processed"
MODELS_DIR = ROOT / "models"
DEFAULT_ASSET = "BTCUSDT"
LOOKBACK_ROWS = 1500
REPORT_PATH = ROOT / "prototype_report.md"

logger = logging.getLogger(__name__)


@dataclass
class DataRefreshResult:
    status: str
    source: str
    data_path: Path | None = None
    error: str | None = None


@dataclass
class PerryResult:
    asset: str
    data_source: str
    data_refresh_status: str
    last_market_timestamp: str
    market_state: str
    position: str
    confluence: str
    move_probability: int
    directional_bias: str
    bullish_score: int
    bearish_score: int
    expected_move_low: float
    expected_move_high: float
    directional_confidence: str
    top_drivers: list[str]
    interpretation: list[str]
    diagnostics: dict[str, float | str]
    model_path: str
    feature_path: str
    state_feature_path: str


def refresh_data(asset: str = DEFAULT_ASSET) -> DataRefreshResult:
    runtime_path = RAW_DATA_DIR / "futures_klines_15m.csv"

    try:
        from Data.download_extended_klines import fetch_klines

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        fresh = fetch_klines(asset)
        if fresh.empty:
            raise ValueError(f"Binance refresh returned no rows for {asset}")

        if runtime_path.exists():
            cached = pd.read_csv(runtime_path, parse_dates=["Datetime"])
            if "symbol" in cached.columns:
                cached = cached[cached["symbol"] != asset]
                updated = pd.concat([cached, fresh], ignore_index=True)
            else:
                updated = fresh
        else:
            updated = fresh

        updated = updated.sort_values(["symbol", "Datetime"]).reset_index(drop=True)
        updated.to_csv(runtime_path, index=False)
        return DataRefreshResult(status="Success", source="Binance", data_path=runtime_path)
    except Exception as exc:
        logger.warning("Market data refresh failed; falling back to local cache: %s", exc)
        print(f"WARNING: Market data refresh failed. Using local cache. ({exc})")
        return DataRefreshResult(status="Fallback", source="Local Cache", data_path=None, error=str(exc))


def _csv_has_columns(path: Path, required: Iterable[str]) -> bool:
    try:
        sample = pd.read_csv(path, nrows=3)
    except Exception:
        return False
    return all(column in sample.columns for column in required)


def _latest_existing_path(candidates: Iterable[Path], required_columns: Iterable[str] | None = None) -> Path:
    existing = [path for path in candidates if path.exists()]
    if required_columns is not None:
        existing = [path for path in existing if _csv_has_columns(path, required_columns)]
    if not existing:
        names = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No usable data source found. Checked: {names}")
    return max(existing, key=lambda path: path.stat().st_mtime)


def discover_raw_datasets() -> list[Path]:
    required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    candidates = list(RAW_DATA_DIR.rglob("*.csv")) if RAW_DATA_DIR.exists() else []
    candidates.extend(
        [
            ROOT / "Data" / "master_raw_dataset.csv",
            RAW_DATA_DIR / "master_raw_dataset.csv",
            RAW_DATA_DIR / "futures_klines_15m.csv",
        ]
    )
    unique = []
    for path in candidates:
        if path not in unique and path.exists() and _csv_has_columns(path, required):
            unique.append(path)
    return sorted(unique, key=lambda path: path.stat().st_mtime, reverse=True)


def ensure_raw_dataset(asset: str = DEFAULT_ASSET) -> DataRefreshResult:
    discovered = discover_raw_datasets()
    if discovered:
        return DataRefreshResult(status="Cached", source="Discovered Raw Dataset", data_path=discovered[0])

    result = refresh_data(asset)
    if result.data_path and result.data_path.exists():
        return result

    discovered = discover_raw_datasets()
    if discovered:
        return DataRefreshResult(
            status="Fallback",
            source="Discovered Raw Dataset",
            data_path=discovered[0],
            error=result.error,
        )
    raise FileNotFoundError(
        "Raw dataset is missing and automatic ingestion failed. "
        f"Expected a CSV with OHLCV columns under {RAW_DATA_DIR}."
    )


def load_latest_data(
    asset: str = DEFAULT_ASSET,
    lookback_rows: int | None = LOOKBACK_ROWS,
    raw_path: Path | None = None,
) -> pd.DataFrame:
    data_path = raw_path or _latest_existing_path(
        discover_raw_datasets(),
        required_columns=["Datetime", "Open", "High", "Low", "Close", "Volume"],
    )

    df = pd.read_csv(data_path, parse_dates=["Datetime"])
    if "symbol" in df.columns:
        df = df[df["symbol"] == asset].copy()
    else:
        df["symbol"] = asset

    required = ["Datetime", "Open", "High", "Low", "Close", "Volume", "symbol"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Data source {data_path} is missing columns: {missing}")

    df = df[required].sort_values("Datetime").dropna()
    if df.empty:
        raise ValueError(f"No rows found for asset {asset} in {data_path}")

    if lookback_rows is not None:
        df = df.tail(lookback_rows)
    return df.reset_index(drop=True)


def generate_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1"] = out["Close"].pct_change()
    out["return_4"] = out["Close"].pct_change(4)
    out["return_16"] = out["Close"].pct_change(16)
    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["volume_z_96"] = (
        (out["Volume"] - out["Volume"].rolling(96, min_periods=24).mean())
        / out["Volume"].rolling(96, min_periods=24).std()
    )
    return out


def _last_number(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    if pd.isna(value) or value in (np.inf, -np.inf):
        return default
    return float(value)


def _bounded_percent(value: float) -> int:
    return int(round(float(np.clip(value, 0.0, 100.0))))


def _label_confluence(score: float) -> str:
    if score >= 0.50:
        return "High"
    if score >= 0.25:
        return "Moderate"
    return "Low"


def _label_position(position_score: float) -> str:
    if position_score >= 0.68:
        return "Upper Structure"
    if position_score <= 0.32:
        return "Lower Structure"
    return "Equilibrium"


def _label_market_state(compression_ratio: float, trend_strength: float, entropy: float) -> str:
    if compression_ratio < 0.85:
        return "Compression"
    if trend_strength > 0.55:
        return "Directional Expansion"
    if entropy > 0.80:
        return "Transition"
    return "Balance"


def _confidence_label(edge: float, confluence_score: float) -> str:
    if edge >= 0.28 and confluence_score >= 0.35:
        return "High"
    if edge >= 0.12:
        return "Moderate"
    return "Low"


def _driver_scores(row: pd.Series) -> dict[str, float]:
    compression_energy = max(_last_number(row, "compression_energy_50"), 0.0)
    equilibrium_distance = abs(_last_number(row, "equilibrium_distance_pct_50"))
    support_confluence = float(bool(row.get("confluence_to_support_50", False)))
    resistance_confluence = float(bool(row.get("confluence_to_resistance_50", False)))
    trend_geometry = abs(_last_number(row, "trend_persistence_50"))
    structure_position = abs(_last_number(row, "position_score_50", 0.5) - 0.5) * 2.0

    return {
        "Compression Energy": compression_energy,
        "Equilibrium Distance": equilibrium_distance * 16.0,
        "Support Confluence": support_confluence,
        "Resistance Confluence": resistance_confluence,
        "Trend Geometry": trend_geometry,
        "Structure Position": structure_position,
    }


def _human_drivers(row: pd.Series, bias: str) -> list[str]:
    drivers: list[str] = []
    equilibrium_distance = _last_number(row, "equilibrium_distance_pct_50")
    compression_energy = _last_number(row, "compression_energy_50")
    trend_slope = _last_number(row, "trend_slope_50")
    support_confluence = bool(row.get("confluence_to_support_50", False))
    resistance_confluence = bool(row.get("confluence_to_resistance_50", False))
    support_distance = _last_number(row, "distance_to_support_50")
    resistance_distance = _last_number(row, "distance_to_resistance_50")

    if equilibrium_distance > 0:
        drivers.append("Above equilibrium")
    elif equilibrium_distance < 0:
        drivers.append("Below equilibrium")

    if compression_energy > 0.05:
        drivers.append("Compression release")
    if support_confluence or support_distance > resistance_distance:
        drivers.append("Structure support intact")
    if resistance_confluence:
        drivers.append("Resistance overhead")
    if trend_slope > 0:
        drivers.append("Positive trend geometry")
    elif trend_slope < 0:
        drivers.append("Negative trend geometry")

    fallback = [
        "State-space regime active",
        "Geometry features aligned",
        f"{bias} directional pressure",
    ]
    for item in fallback:
        if len(drivers) >= 4:
            break
        if item not in drivers:
            drivers.append(item)
    return drivers[:4]


def determine_market_state(feature_df: pd.DataFrame) -> tuple[str, str, str, dict[str, float | str]]:
    row = feature_df.iloc[-1]
    compression_ratio = _last_number(row, "compression_ratio_50", 1.0)
    position_score = _last_number(row, "position_score_50", 0.5)
    confluence_score = _last_number(row, "confluence_agreement_normalized", 0.0)
    entropy = _last_number(row, "state_entropy", 0.0)
    trend_persistence = abs(_last_number(row, "trend_persistence_50", 0.0))

    market_state = _label_market_state(compression_ratio, trend_persistence, entropy)
    position = _label_position(position_score)
    confluence = _label_confluence(confluence_score)

    diagnostics = {
        "latest_datetime": str(row.get("Datetime", "")),
        "latest_close": _last_number(row, "Close"),
        "compression_ratio_50": compression_ratio,
        "position_score_50": position_score,
        "confluence_score": confluence_score,
        "state_entropy": entropy,
        "trend_persistence_50": trend_persistence,
    }
    return market_state, position, confluence, diagnostics


def run_move_prediction(feature_df: pd.DataFrame, market_state: str) -> int:
    row = feature_df.iloc[-1]
    compression_energy = max(_last_number(row, "compression_energy_50"), 0.0)
    confluence_score = _last_number(row, "confluence_agreement_normalized", 0.0)
    position_extreme = abs(_last_number(row, "position_score_50", 0.5) - 0.5) * 2.0
    range_pct = _last_number(row, "range_pct", 0.0)
    recent_volatility = feature_df["return_1"].tail(96).std(skipna=True)
    recent_volatility = 0.0 if pd.isna(recent_volatility) else float(recent_volatility)

    score = 45.0
    score += np.clip(compression_energy, 0.0, 1.0) * 28.0
    score += np.clip(confluence_score, 0.0, 1.0) * 12.0
    score += np.clip(position_extreme, 0.0, 1.0) * 10.0
    score += np.clip((range_pct + recent_volatility) * 120.0, 0.0, 8.0)
    if market_state == "Compression":
        score += 6.0

    return _bounded_percent(score)


def discover_feature_path(asset: str = DEFAULT_ASSET) -> Path:
    return PROCESSED_DATA_DIR / f"{asset.lower()}_causal_features.pkl"


def discover_state_feature_path(asset: str = DEFAULT_ASSET) -> Path:
    return PROCESSED_DATA_DIR / f"{asset.lower()}_state_space_features.csv"


def ensure_feature_dataset(raw: pd.DataFrame, asset: str = DEFAULT_ASSET) -> tuple[pd.DataFrame, Path, str]:
    path = discover_feature_path(asset)
    if path.exists():
        features = pd.read_pickle(path)
        return features, path, "cached"

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        features = build_causal_features(raw)
    features.to_pickle(path)
    features.to_csv(path.with_suffix(".csv"), index=False)
    return features, path, "built"


def build_runtime_feature_frame(raw: pd.DataFrame, asset: str = DEFAULT_ASSET) -> tuple[pd.DataFrame, Path, str]:
    path = discover_state_feature_path(asset)
    if path.exists():
        feature_df = pd.read_csv(path, parse_dates=["Datetime"])
        return feature_df, path, "cached"

    base = generate_base_features(raw)
    state_space = build_state_space_features(base).add_prefix("state_space_")
    structure = add_anchored_structure_features(base)
    market_state_features = build_market_state_features(base, include_latent=True)
    feature_df = pd.concat(
        [
            base,
            state_space,
            structure[
                [
                    "anchored_support",
                    "anchored_resistance",
                    "distance_to_anchored_support",
                    "distance_to_anchored_resistance",
                    "anchored_breakout",
                    "anchored_breakdown",
                ]
            ],
            market_state_features.drop(columns=["Datetime", "Close", "High", "Low", "Volume"], errors="ignore"),
        ],
        axis=1,
    )
    feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(path, index=False)
    return feature_df, path, "built"


def _model_score(row: pd.Series) -> tuple[bool, float, float]:
    rejected = bool(row.get("rejected_for_collapse", False))
    macro = float(row.get("macro_f1_mean", 0.0) or 0.0)
    balanced = float(row.get("balanced_accuracy_mean", 0.0) or 0.0)
    return (not rejected, macro, balanced)


def discover_model() -> tuple[Any | None, Path | None, dict[str, Any]]:
    summary_path = ROOT / "results" / "experiments.csv"
    candidates: list[tuple[Path, dict[str, Any], tuple[bool, float, float]]] = []
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        for _, row in summary.iterrows():
            model_path = MODELS_DIR / f"{row['experiment_id']}.joblib"
            config_path = ROOT / "experiments" / str(row["experiment_id"]) / "config.json"
            if not model_path.exists():
                continue
            config = {}
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
            candidates.append((model_path, config, _model_score(row)))

    if not candidates and MODELS_DIR.exists():
        for model_path in MODELS_DIR.glob("*.joblib"):
            candidates.append((model_path, {}, (True, 0.0, model_path.stat().st_mtime)))

    if not candidates:
        return None, None, {}

    model_path, config, _ = max(candidates, key=lambda item: item[2])
    return joblib.load(model_path), model_path, config


def predict_move_with_model(
    model: Any | None,
    causal_features: pd.DataFrame,
    fallback_probability: int,
) -> tuple[int, str]:
    if model is None:
        return fallback_probability, "heuristic"
    columns = list(getattr(model, "feature_names_in_", []))
    if not columns:
        return fallback_probability, "heuristic_no_feature_contract"
    missing = [column for column in columns if column not in causal_features.columns]
    if missing:
        return fallback_probability, f"heuristic_missing_model_features:{len(missing)}"
    row = causal_features[columns].tail(1)
    try:
        probabilities = model.predict_proba(row)
        classes = list(getattr(model.named_steps.get("model"), "classes_", []))
        if 1 in classes:
            probability = probabilities[0, classes.index(1)] * 100.0
        else:
            probability = float(np.max(probabilities[0]) * 100.0)
        return _bounded_percent(probability), "trained_model"
    except Exception as exc:
        logger.warning("Model prediction failed; using heuristic probability: %s", exc)
        return fallback_probability, f"heuristic_model_error:{exc}"


def estimate_expected_move(
    causal_features: pd.DataFrame,
    move_probability: int,
    bias: str,
    horizon: int = 12,
) -> tuple[float, float]:
    returns = causal_features["return_1"].tail(192).dropna()
    realized_volatility = float(returns.std()) if not returns.empty else 0.005
    expected_abs_move = max(realized_volatility * np.sqrt(horizon) * (0.65 + move_probability / 100.0), 0.003)
    low = expected_abs_move * 0.70 * 100.0
    high = expected_abs_move * 1.25 * 100.0
    sign = -1.0 if bias == "Bearish" else 1.0
    return sign * low, sign * high


def estimate_directional_bias(feature_df: pd.DataFrame) -> tuple[int, int, str, str]:
    row = feature_df.iloc[-1]
    bullish = 50.0

    position_score = _last_number(row, "position_score_50", 0.5)
    equilibrium_distance = _last_number(row, "equilibrium_distance_pct_50", 0.0)
    trend_slope = _last_number(row, "trend_slope_50", 0.0)
    trend_persistence = _last_number(row, "trend_persistence_50", 0.0)
    support_distance = _last_number(row, "distance_to_support_50", 0.0)
    resistance_distance = _last_number(row, "distance_to_resistance_50", 0.0)
    support_confluence = bool(row.get("confluence_to_support_50", False))
    resistance_confluence = bool(row.get("confluence_to_resistance_50", False))
    compression_energy = max(_last_number(row, "compression_energy_50"), 0.0)

    if trend_slope > 0:
        bullish += 10.0 * max(trend_persistence, 0.2)
    elif trend_slope < 0:
        bullish -= 10.0 * max(trend_persistence, 0.2)

    if position_score < 0.35:
        bullish += 8.0
    elif position_score > 0.65:
        bullish += 4.0 if trend_slope > 0 else -6.0

    if support_confluence:
        bullish += 7.0
    if resistance_confluence:
        bullish -= 7.0

    if support_distance > resistance_distance:
        bullish += 4.0
    elif resistance_distance > support_distance:
        bullish -= 4.0

    bullish += np.clip(equilibrium_distance * 240.0, -8.0, 8.0)
    bullish += np.clip(compression_energy, 0.0, 1.0) * (3.0 if trend_slope >= 0 else -3.0)

    bullish_score = _bounded_percent(bullish)
    bearish_score = 100 - bullish_score
    edge = abs(bullish_score - bearish_score) / 100.0
    confluence_score = _last_number(row, "confluence_agreement_normalized", 0.0)
    confidence = _confidence_label(edge, confluence_score)

    if bullish_score > bearish_score + 3:
        bias = "Bullish"
    elif bearish_score > bullish_score + 3:
        bias = "Bearish"
    else:
        bias = "Neutral"

    return bullish_score, bearish_score, confidence, bias


def generate_report(result: PerryResult) -> str:
    drivers = "\n".join(f"{i}. {driver}" for i, driver in enumerate(result.top_drivers, start=1))
    interpretation = "\n".join(result.interpretation)
    diagnostics = "\n".join(f"- {key}: {value}" for key, value in result.diagnostics.items())

    return f"""# Perry Prototype Report

## Current State

- Asset: {result.asset}
- Data Source: {result.data_source}
- Last Market Timestamp: {result.last_market_timestamp}
- Data Refresh Status: {result.data_refresh_status}
- Market State: {result.market_state}
- Position: {result.position}
- Confluence: {result.confluence}
- Move Probability: {result.move_probability}%
- Directional Bias: {result.directional_bias}
- Bullish Score: {result.bullish_score}%
- Bearish Score: {result.bearish_score}%
- Expected Move: {result.expected_move_low:+.1f}% to {result.expected_move_high:+.1f}%
- Directional Confidence: {result.directional_confidence}
- Model: {result.model_path}
- Feature Dataset: {result.feature_path}
- State Feature Dataset: {result.state_feature_path}

## Top Drivers

{drivers}

## Interpretation

{interpretation}

## Runtime Diagnostics

{diagnostics}

## Method

`python app.py` loads the latest available BTCUSDT data, generates base features,
state-space features, anchored structure features, market-state features,
geometry features, confluence features, current market state, move probability,
directional bias, and this markdown report. The prototype estimates directional
bias only; it is not a hard BUY/SELL predictor and does not claim certainty.
"""


def print_terminal_report(result: PerryResult) -> None:
    print("Market State:")
    print(result.market_state)
    print()
    print("Directional Bias:")
    print(f"{result.directional_bias} {result.bullish_score if result.directional_bias != 'Bearish' else result.bearish_score}%")
    print()
    print("Expected Move:")
    print(f"{result.expected_move_low:+.1f}% to {result.expected_move_high:+.1f}%")
    print()
    print("Confidence:")
    print(f"{result.move_probability}%")
    print()
    print("Top Drivers:")
    for driver in result.top_drivers:
        print(f"- {driver}")
    print()
    print("Runtime:")
    print(f"- Asset: {result.asset}")
    print(f"- Data Source: {result.data_source}")
    print(f"- Last Market Timestamp: {result.last_market_timestamp}")
    print(f"- Data Status: {result.data_refresh_status}")
    print(f"- Model: {result.model_path}")
    print(f"- Report: {REPORT_PATH}")
    print()


def build_proton_result(
    asset: str = DEFAULT_ASSET,
    refresh_result: DataRefreshResult | None = None,
) -> PerryResult:
    refresh_result = refresh_result or ensure_raw_dataset(asset)
    raw = load_latest_data(asset=asset, lookback_rows=None, raw_path=refresh_result.data_path)
    last_market_timestamp = str(raw["Datetime"].iloc[-1])
    causal_features, causal_path, causal_status = ensure_feature_dataset(raw, asset)
    runtime_raw = raw.tail(LOOKBACK_ROWS).reset_index(drop=True)
    feature_df, state_path, state_status = build_runtime_feature_frame(runtime_raw, asset)

    market_state, position, confluence, diagnostics = determine_market_state(feature_df)
    heuristic_probability = run_move_prediction(feature_df, market_state)
    model, model_path, model_config = discover_model()
    move_probability, model_status = predict_move_with_model(model, causal_features, heuristic_probability)
    bullish_score, bearish_score, confidence, bias = estimate_directional_bias(feature_df)
    expected_low, expected_high = estimate_expected_move(
        causal_features,
        move_probability,
        bias,
        horizon=int(model_config.get("horizon", 12) or 12),
    )

    top_drivers = _human_drivers(feature_df.iloc[-1], bias)

    interpretation = []
    interpretation.append("Large move likely." if move_probability >= 70 else "Large move possible.")
    if bias == "Bullish":
        interpretation.append("Current state favors upside.")
    elif bias == "Bearish":
        interpretation.append("Current state favors downside.")
    else:
        interpretation.append("Current state is directionally balanced.")
    interpretation.append(f"Confidence remains {confidence.lower()}.")
    diagnostics["causal_feature_status"] = causal_status
    diagnostics["state_feature_status"] = state_status
    diagnostics["model_status"] = model_status

    return PerryResult(
        asset=asset,
        data_source=refresh_result.source,
        data_refresh_status=refresh_result.status,
        last_market_timestamp=last_market_timestamp,
        market_state=market_state,
        position=position,
        confluence=confluence,
        move_probability=move_probability,
        directional_bias=bias,
        bullish_score=bullish_score,
        bearish_score=bearish_score,
        expected_move_low=expected_low,
        expected_move_high=expected_high,
        directional_confidence=confidence,
        top_drivers=top_drivers,
        interpretation=interpretation,
        diagnostics=diagnostics,
        model_path=str(model_path) if model_path else "No trained model found; heuristic fallback",
        feature_path=str(causal_path),
        state_feature_path=str(state_path),
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    refresh_result = ensure_raw_dataset()
    result = build_proton_result(refresh_result=refresh_result)
    REPORT_PATH.write_text(generate_report(result), encoding="utf-8")
    print_terminal_report(result)


if __name__ == "__main__":
    main()
