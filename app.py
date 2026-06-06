from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from Model.features_data.market_state_discovery import build_market_state_features
from Model.features_data.market_structure import add_anchored_structure_features
from Model.features_data.state_space import build_state_space_features


ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "Data" / "datasets" / "raw"
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
    directional_confidence: str
    top_drivers: list[str]
    interpretation: list[str]
    diagnostics: dict[str, float | str]


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


def _latest_existing_path(candidates: Iterable[Path]) -> Path:
    existing = [path for path in candidates if path.exists()]
    if not existing:
        names = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No usable data source found. Checked: {names}")
    return max(existing, key=lambda path: path.stat().st_mtime)


def load_latest_data(asset: str = DEFAULT_ASSET, lookback_rows: int = LOOKBACK_ROWS) -> pd.DataFrame:
    data_path = _latest_existing_path(
        [
            RAW_DATA_DIR / "futures_klines_15m.csv",
            RAW_DATA_DIR / "master_raw_dataset.csv",
            ROOT / "Data" / "master_raw_dataset.csv",
        ]
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

    return df.tail(lookback_rows).reset_index(drop=True)


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
- Directional Confidence: {result.directional_confidence}

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
    print("=================================")
    print("PERRY MARKET STATE ENGINE")
    print("=========================")
    print()
    print("Asset:")
    print(result.asset)
    print()
    print("Data Source:")
    print(result.data_source)
    print()
    print("Last Market Timestamp:")
    print(result.last_market_timestamp)
    print()
    print("Data Refresh Status:")
    print(result.data_refresh_status)
    print()
    print("Market State:")
    print(result.market_state)
    print()
    print("Position:")
    print(result.position)
    print()
    print("Confluence:")
    print(result.confluence)
    print()
    print("Move Probability:")
    print(f"{result.move_probability}%")
    print()
    print("Directional Bias:")
    print(result.directional_bias)
    print()
    print("Bullish Score:")
    print(f"{result.bullish_score}%")
    print()
    print("Bearish Score:")
    print(f"{result.bearish_score}%")
    print()
    print("Directional Confidence:")
    print(result.directional_confidence)
    print()
    print("Top Drivers:")
    print()
    for i, driver in enumerate(result.top_drivers, start=1):
        print(f"{i}. {driver}")
    print()
    print("Interpretation:")
    print()
    for line in result.interpretation:
        print(line)
    print()
    print("=================================")
    print(f"\nMarkdown report written to: {REPORT_PATH}")


def build_proton_result(
    asset: str = DEFAULT_ASSET,
    refresh_result: DataRefreshResult | None = None,
) -> PerryResult:
    raw = load_latest_data(asset=asset)
    last_market_timestamp = str(raw["Datetime"].iloc[-1])
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

    market_state, position, confluence, diagnostics = determine_market_state(feature_df)
    move_probability = run_move_prediction(feature_df, market_state)
    bullish_score, bearish_score, confidence, bias = estimate_directional_bias(feature_df)

    driver_scores = _driver_scores(feature_df.iloc[-1])
    top_drivers = [
        name for name, _ in sorted(driver_scores.items(), key=lambda item: item[1], reverse=True)[:4]
    ]

    interpretation = []
    interpretation.append("Large move likely." if move_probability >= 70 else "Large move possible.")
    if bias == "Bullish":
        interpretation.append("Current state favors upside.")
    elif bias == "Bearish":
        interpretation.append("Current state favors downside.")
    else:
        interpretation.append("Current state is directionally balanced.")
    interpretation.append(f"Confidence remains {confidence.lower()}.")

    return PerryResult(
        asset=asset,
        data_source=refresh_result.source if refresh_result else "Local Cache",
        data_refresh_status=refresh_result.status if refresh_result else "Fallback",
        last_market_timestamp=last_market_timestamp,
        market_state=market_state,
        position=position,
        confluence=confluence,
        move_probability=move_probability,
        directional_bias=bias,
        bullish_score=bullish_score,
        bearish_score=bearish_score,
        directional_confidence=confidence,
        top_drivers=top_drivers,
        interpretation=interpretation,
        diagnostics=diagnostics,
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    refresh_result = refresh_data()
    result = build_proton_result(refresh_result=refresh_result)
    REPORT_PATH.write_text(generate_report(result), encoding="utf-8")
    print_terminal_report(result)


if __name__ == "__main__":
    main()
