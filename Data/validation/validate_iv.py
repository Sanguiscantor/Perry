"""Validate aggregated Deribit IV snapshots as added features.

Aggregates any `deribit_iv_snapshot_*.csv` or `deribit_iv_panel.csv` into 15m features
(`iv_mean`, `iv_std`, `iv_count`) aligned on `Datetime`, merges with causal base features
and runs the leakage-safe `evaluate_move_signal` from the research engine.

Produces `artifacts/validation/iv_vs_baseline.json` with baseline and iv-augmented metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "Model") not in sys.path:
    sys.path.insert(0, str(ROOT / "Model"))

from research_program import load_multi_asset, slice_symbol, save_json, evaluate_move_signal
from research_engine import build_causal_features
from enhanced_features import merge_features

import pandas as pd

ART_DIR = ROOT / "artifacts" / "validation"
ART_DIR.mkdir(parents=True, exist_ok=True)

IV_PANEL = ROOT / "artifacts" / "data" / "microstructure" / "deribit_iv_panel.csv"


def load_iv_aggregated():
    if not IV_PANEL.exists():
        # try snapshots
        dirp = ROOT / "artifacts" / "data" / "microstructure"
        snaps = list(dirp.glob("deribit_iv_snapshot_*.csv"))
        if not snaps:
            return None
        frames = [pd.read_csv(p) for p in snaps]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(IV_PANEL)

    if df.empty:
        return None

    # Normalize timestamp column and ensure tz-naive Datetime for merges
    if "timestamp_dt" in df.columns:
        df["Datetime"] = pd.to_datetime(df["timestamp_dt"]) 
    elif "timestamp" in df.columns:
        df["Datetime"] = pd.to_datetime(df["timestamp"])
    else:
        df["Datetime"] = pd.to_datetime(df.get("timestamp", pd.NaT))

    # Make Datetime tz-naive to match base features
    try:
        df["Datetime"] = df["Datetime"].dt.tz_convert(None)
    except Exception:
        try:
            df["Datetime"] = df["Datetime"].dt.tz_localize(None)
        except Exception:
            df["Datetime"] = pd.to_datetime(df["Datetime"])  

    # Coerce any p_ prefixed columns to numeric and treat them as IV-like candidates
    pcols = [c for c in df.columns if c.startswith("p_")]
    iv_cols = []
    for c in pcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().sum() > 0:
            iv_cols.append(c)

    # Fallback: coerce other numeric-looking columns if none found
    if not iv_cols:
        cand = [c for c in df.columns if c not in ("instrument_name", "timestamp", "timestamp_dt", "Datetime", "option_type", "strike", "expiration_timestamp")]
        for c in cand:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].notna().sum() > 0:
                iv_cols.append(c)

    if not iv_cols:
        return None

    # Aggregate to 15m: mean, std, count
    df = df.set_index("Datetime")
    agg = df[iv_cols].resample("15min").agg(["mean", "std", "count"]).ffill()
    # flatten columns
    agg.columns = [f"iv_{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()
    return agg


def main():
    multi = load_multi_asset()
    btc_raw = slice_symbol(multi, "BTCUSDT")
    base = build_causal_features(btc_raw)

    baseline_metrics = evaluate_move_signal(btc_raw, base)

    iv = load_iv_aggregated()
    if iv is None or iv.empty:
        print("No IV panel data available; skipping IV-augmented run.")
        save_json("validation/iv_vs_baseline", {"baseline": baseline_metrics, "iv_augmented": "no_data"})
        return

    merged = merge_features(base, deriv=iv)
    iv_metrics = evaluate_move_signal(btc_raw, merged)

    payload = {"baseline": baseline_metrics, "iv_augmented": iv_metrics}
    save_json("validation/iv_vs_baseline", payload)
    print("Saved validation results to artifacts/validation/iv_vs_baseline.json")


if __name__ == "__main__":
    main()
