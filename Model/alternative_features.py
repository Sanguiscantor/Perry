"""Causal features from non-OHLCV alternative data sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIV_DIR = ROOT / "Data" / "datasets" / "raw" / "derivatives"
KLINES_PATH = ROOT / "Data" / "datasets" / "raw" / "futures_klines_15m.csv"


def _load_symbol_csv(path: Path, symbol: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Datetime"])
    sym = df[df["symbol"] == symbol].sort_values("Datetime")
    return sym if len(sym) else None


def build_taker_flow_features(symbol: str = "BTCUSDT") -> pd.DataFrame | None:
    if not KLINES_PATH.exists():
        return None
    df = _load_symbol_csv(KLINES_PATH, symbol)
    if df is None:
        return None
    out = df.set_index("Datetime")
    for window in [4, 12, 24, 48, 96]:
        out[f"taker_buy_ratio_mean_{window}"] = out["taker_buy_ratio"].rolling(window).mean()
        out[f"taker_imbalance_mean_{window}"] = out["taker_imbalance"].rolling(window).mean()
        out[f"trades_z_{window}"] = (out["trades"] - out["trades"].rolling(window).mean()) / out["trades"].rolling(window).std()
        out[f"quote_vol_z_{window}"] = (out["quote_volume"] - out["quote_volume"].rolling(window).mean()) / out["quote_volume"].rolling(window).std()
    out["taker_buy_ratio_lag_1"] = out["taker_buy_ratio"].shift(1)
    out["taker_imbalance_lag_1"] = out["taker_imbalance"].shift(1)
    out["taker_imbalance_lag_4"] = out["taker_imbalance"].shift(4)
    base_cols = set(df.columns) | {"index"}
    engineered = [c for c in out.columns if c not in base_cols]
    return out.reset_index()[["Datetime"] + engineered]


def build_sentiment_features(symbol: str = "BTCUSDT", prefix: str = "") -> pd.DataFrame | None:
    frames = []
    specs = [
        ("global_long_short_15m.csv", ["long_short_ratio", "long_account_pct"], "gls"),
        ("top_trader_long_short_15m.csv", ["long_short_ratio", "long_account_pct"], "top"),
        ("taker_buy_sell_15m.csv", ["taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol"], "agg"),
    ]
    for filename, cols, tag in specs:
        path = DERIV_DIR / filename
        sym = _load_symbol_csv(path, symbol)
        if sym is None:
            continue
        s = sym.set_index("Datetime")[cols]
        s.columns = [f"{prefix}{tag}_{c}" for c in cols]
        for window in [12, 48, 96]:
            for c in s.columns:
                s[f"{c}_z_{window}"] = (s[c] - s[c].rolling(window).mean()) / s[c].rolling(window).std()
                s[f"{c}_chg_{window}"] = s[c].pct_change(window)
        frames.append(s)
    if not frames:
        return None
    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged.reset_index()


def build_funding_oi_features(symbol: str = "BTCUSDT") -> pd.DataFrame | None:
    from enhanced_features import build_derivatives_features

    return build_derivatives_features(symbol)


def build_all_alternative(symbol: str = "BTCUSDT") -> pd.DataFrame | None:
    parts = [
        build_taker_flow_features(symbol),
        build_sentiment_features(symbol),
        build_funding_oi_features(symbol),
    ]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="Datetime", how="outer")
    cols = [c for c in merged.columns if c != "Datetime"]
    merged[cols] = merged[cols].replace([np.inf, -np.inf], np.nan)
    return merged.sort_values("Datetime")
