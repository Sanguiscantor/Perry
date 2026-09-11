"""Causal cross-asset and derivatives feature builders for Perry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MULTI_PATH = ROOT / "Data" / "multi_asset_dataset.csv"
DERIV_DIR = ROOT / "Data" / "derivatives"

ALTS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
LAGS = [1, 2, 4, 8, 12]


def _close_panel(multi: pd.DataFrame) -> pd.DataFrame:
    panel = multi.pivot_table(index="Datetime", columns="symbol", values="Close", aggfunc="last").sort_index()
    return panel.ffill()


def _ret_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pct_change()


def build_cross_asset_features(multi: pd.DataFrame, anchor: str = "BTCUSDT") -> pd.DataFrame:
    """Causal features from aligned multi-asset closes; all lags are backward-looking."""
    panel = _close_panel(multi)
    rets = _ret_panel(panel)
    anchor_ret = rets[anchor]
    out = pd.DataFrame(index=panel.index)
    out["Datetime"] = out.index

    for lag in LAGS:
        out[f"btc_ret_lag_{lag}"] = anchor_ret.shift(lag)
        out[f"btc_abs_ret_lag_{lag}"] = anchor_ret.abs().shift(lag)

    for sym in ALTS:
        if sym not in rets.columns:
            continue
        short = sym.replace("USDT", "").lower()
        out[f"{short}_ret_lag_1"] = rets[sym].shift(1)
        out[f"{short}_rel_strength_24"] = (
            panel[sym] / panel[sym].rolling(24).mean() - panel[anchor] / panel[anchor].rolling(24).mean()
        )
        out[f"{short}_vol_ratio_48"] = rets[sym].rolling(48).std() / anchor_ret.rolling(48).std()
        out[f"{short}_corr_btc_48"] = rets[sym].rolling(48).corr(anchor_ret)

    alt_rets = rets[[c for c in ALTS if c in rets.columns]]
    if len(alt_rets.columns):
        out["alt_index_ret_1"] = alt_rets.mean(axis=1).shift(1)
        out["alt_dispersion_48"] = alt_rets.rolling(48).std().mean(axis=1)
        out["alt_breadth_up_12"] = (alt_rets > 0).rolling(12).mean().mean(axis=1)

    btc_range = (multi[multi["symbol"] == anchor].set_index("Datetime")["High"] - multi[multi["symbol"] == anchor].set_index("Datetime")["Low"]) / panel[anchor]
    out["btc_range_pct_lag_1"] = btc_range.shift(1)
    out["btc_range_pct_lag_4"] = btc_range.shift(4)

    return out.reset_index(drop=True)


from perry_config import filename_with_timeframe, get_primary_timeframe, timeframe_to_pandas_offset

def build_derivatives_features(anchor: str = "BTCUSDT") -> pd.DataFrame | None:
    funding_path = DERIV_DIR / "funding_rates.csv"
    oi_path = DERIV_DIR / filename_with_timeframe("open_interest", get_primary_timeframe())
    if not funding_path.exists() and not oi_path.exists():
        return None

    frames = []
    if funding_path.exists():
        funding = pd.read_csv(funding_path, parse_dates=["Datetime"])
        sym = funding[funding["symbol"] == anchor].sort_values("Datetime")
        f = sym.set_index("Datetime")[["funding_rate"]].resample(timeframe_to_pandas_offset()).ffill()
        f["funding_rate_z_96"] = (f["funding_rate"] - f["funding_rate"].rolling(96, min_periods=24).mean()) / f["funding_rate"].rolling(96, min_periods=24).std()
        f["funding_rate_change_8h"] = f["funding_rate"].diff(32)
        frames.append(f)

    if oi_path.exists():
        oi = pd.read_csv(oi_path, parse_dates=["Datetime"])
        sym = oi[oi["symbol"] == anchor].sort_values("Datetime")
        o = sym.set_index("Datetime")[["open_interest", "oi_value"]].sort_index()
        o["oi_pct_change_12"] = o["open_interest"].pct_change(12)
        o["oi_pct_change_48"] = o["open_interest"].pct_change(48)
        o["oi_z_96"] = (o["open_interest"] - o["open_interest"].rolling(96, min_periods=24).mean()) / o["open_interest"].rolling(96, min_periods=24).std()
        frames.append(o)

    if not frames:
        return None
    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.reset_index()
    return merged


def merge_features(
    base: pd.DataFrame,
    cross: pd.DataFrame | None = None,
    deriv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = base.copy()
    for extra in (cross, deriv):
        if extra is None:
            continue
        merged = merged.merge(extra, on="Datetime", how="left")
    feature_cols = [c for c in merged.columns if c != "Datetime"]
    merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], np.nan)
    return merged
