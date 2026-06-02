"""Retry Deribit acquisition with corrected string parameters."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT.parent / "artifacts" / "data" / "microstructure"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DERIBIT_API = "https://www.deribit.com/api/v2"


def fetch_instruments(currency: str = "BTC") -> pd.DataFrame:
    print("Fetching Deribit instruments (retry)...")
    res = requests.get(f"{DERIBIT_API}/public/get_instruments", params={"currency": currency, "kind": "option", "expired": "false"}, timeout=30)
    res.raise_for_status()
    instruments = res.json().get("result", [])
    df = pd.DataFrame(instruments)
    path = ARTIFACT_DIR / f"deribit_instruments_{currency}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df):,} rows)")
    return df


def fetch_trades(instrument_name: str, count: int = 1000) -> pd.DataFrame:
    print(f"Fetching Deribit trades {instrument_name} (retry)...")
    res = requests.get(f"{DERIBIT_API}/public/get_last_trades_by_instrument", params={"instrument_name": instrument_name, "count": count, "include_old": "true"}, timeout=30)
    res.raise_for_status()
    trades = res.json().get("result", {}).get("trades", [])
    df = pd.DataFrame(trades)
    if not df.empty:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    path = ARTIFACT_DIR / f"deribit_trades_{instrument_name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df):,} rows)")
    return df


def main():
    try:
        inst = fetch_instruments("BTC")
        sample = inst.sort_values("tick_size", na_position="last").head(5)["instrument_name"].tolist()
        for instr in sample:
            fetch_trades(instr, count=500)
    except Exception as exc:
        print(f"Deribit retry failed: {exc}")


if __name__ == "__main__":
    main()
