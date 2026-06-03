"""Acquire free microstructure datasets and write validated artifacts.

Actions:
- Fetch Deribit options instruments and recent trades (BTC, ETH)
- Attempt to paginate Binance liquidation orders
- Compute futures/spot basis from existing CSVs if present
- Write outputs to `artifacts/data/microstructure/` and generate a manifest
- Basic validation checks for produced CSVs

This script is conservative: it will not perform heavy full-history downloads by default; configure run arguments to expand windows.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "data" / "microstructure"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DERIBIT_API = "https://www.deribit.com/api/v2"
BINANCE_FAPI = "https://fapi.binance.com"

# Small defaults to avoid long runs
DERIBIT_COUNT = 1000
BINANCE_LIMIT = 500


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    path = ARTIFACT_DIR / name
    df.to_csv(path, index=False)
    return path


def fetch_deribit_instruments(currency: str = "BTC") -> pd.DataFrame:
    """Fetch active Deribit option instruments for a currency."""
    print("Fetching Deribit instruments...")
    res = requests.get(f"{DERIBIT_API}/public/get_instruments", params={"currency": currency, "kind": "option", "expired": False}, timeout=30)
    res.raise_for_status()
    instruments = res.json()["result"]
    df = pd.DataFrame(instruments)
    path = _write_csv(df, f"deribit_instruments_{currency}.csv")
    print(f"Saved {path} ({len(df):,} rows)")
    return df


def fetch_deribit_trades(instrument_name: str, count: int = DERIBIT_COUNT) -> pd.DataFrame:
    print(f"Fetching Deribit trades {instrument_name}...")
    res = requests.get(f"{DERIBIT_API}/public/get_last_trades_by_instrument", params={"instrument_name": instrument_name, "count": count, "include_old": True}, timeout=30)
    res.raise_for_status()
    trades = res.json()["result"]["trades"]
    df = pd.DataFrame(trades)
    if not df.empty:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    path = _write_csv(df, f"deribit_trades_{instrument_name}.csv")
    print(f"Saved {path} ({len(df):,} rows)")
    return df


def attempt_binance_liquidations(symbols: List[str]) -> pd.DataFrame:
    """Attempt to paginate Binance allForceOrders; returns concatenated DataFrame.

    Note: Binance may restrict historical range. This function is defensive.
    """
    rows = []
    for symbol in symbols:
        print(f"Attempting liquidations for {symbol}...")
        end = int(datetime.now(timezone.utc).timestamp() * 1000)
        for _ in range(200):
            params = {"symbol": symbol, "endTime": end, "limit": BINANCE_LIMIT}
            r = requests.get(f"{BINANCE_FAPI}/fapi/v1/allForceOrders", params=params, timeout=30)
            if r.status_code != 200:
                print(f"  Binance chunk stop {r.status_code}: {r.text[:120]}")
                break
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            # batch is ordered newest-first; set end to oldest timestamp - 1
            oldest = int(batch[-1].get("time", batch[-1].get("timestamp", 0)))
            if oldest <= 0:
                break
            end = oldest - 1
            time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    path = _write_csv(df, "binance_liquidations.csv")
    print(f"Saved {path} ({len(df):,} rows)")
    return df


def compute_basis() -> pd.DataFrame:
    """Compute futures - spot basis using existing local CSVs if available."""
    fut_path = ROOT / "datasets" / "raw" / "futures_klines_15m.csv"
    spot_path = ROOT / "datasets" / "raw" / "master_raw_dataset.csv"
    if not fut_path.exists() or not spot_path.exists():
        print("Required kline files missing; skipping basis computation.")
        return pd.DataFrame()
    print("Loading klines for basis computation...")
    fut = pd.read_csv(fut_path, parse_dates=["Datetime"])  # futures_klines have Datetime and symbol
    spot = pd.read_csv(spot_path, parse_dates=["Datetime"])  # master_raw_dataset has Datetime, Close, symbol
    # normalize column names for merging
    if "Close" in fut.columns:
        fut_close_col = "Close"
    elif "close" in fut.columns:
        fut_close_col = "close"
    else:
        fut_close_col = None
    if fut_close_col is None or "Close" not in spot.columns:
        print("Unexpected kline schemas; skipping basis.")
        return pd.DataFrame()
    fut = fut[["Datetime", "symbol", fut_close_col]].rename(columns={fut_close_col: "future_close"})
    spot = spot[["Datetime", "symbol", "Close"]].rename(columns={"Close": "spot_close"})
    merged = pd.merge_asof(fut.sort_values("Datetime"), spot.sort_values("Datetime"), on="Datetime", by="symbol", direction="backward")
    merged["basis"] = merged["future_close"] - merged["spot_close"]
    path = _write_csv(merged, "basis_15m.csv")
    print(f"Saved {path} ({len(merged):,} rows)")
    return merged


def validate_csv(path: Path, min_rows: int = 1) -> bool:
    try:
        df = pd.read_csv(path)
        valid = len(df) >= min_rows
        print(f"Validate {path.name}: rows={len(df):,} valid={valid}")
        return valid
    except Exception as exc:
        print(f"Validate failed {path}: {exc}")
        return False


def write_manifest(files: List[Path]) -> Path:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": [{"path": str(p), "rows": int(pd.read_csv(p).shape[0]) if p.suffix=='.csv' else None} for p in files],
    }
    path = ARTIFACT_DIR / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest {path}")
    return path


def main():
    files = []

    # 1) Deribit instruments and a small trade sample for BTC and ETH
    try:
        for c in ["BTC", "ETH"]:
            inst = fetch_deribit_instruments(currency=c)
            files.append(ARTIFACT_DIR / f"deribit_instruments_{c}.csv")
            # take top N instruments by liquidity (filter: option instruments only)
            sample = inst.sort_values("tick_size", na_position="last").head(5)["instrument_name"].tolist()
            for instr in sample:
                trades = fetch_deribit_trades(instr, count=500)
                files.append(ARTIFACT_DIR / f"deribit_trades_{instr}.csv")
    except Exception as exc:
        print(f"Deribit fetch failed: {exc}")

    # 2) Binance liquidations (best-effort)
    try:
        bins = attempt_binance_liquidations(["BTCUSDT", "ETHUSDT"])  # conservative
        if not bins.empty:
            files.append(ARTIFACT_DIR / "binance_liquidations.csv")
    except Exception as exc:
        print(f"Binance liquidation fetch failed: {exc}")

    # 3) Basis computation
    try:
        basis = compute_basis()
        if not basis.empty:
            files.append(ARTIFACT_DIR / "basis_15m.csv")
    except Exception as exc:
        print(f"Basis computation failed: {exc}")

    # Validation and manifest
    valid_files = [p for p in files if p.exists() and validate_csv(p, min_rows=1)]
    write_manifest(valid_files)


if __name__ == "__main__":
    main()
