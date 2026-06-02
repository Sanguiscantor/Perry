"""Acquire current Deribit implied vol snapshots across option chain.

Attempts several public endpoints to extract implied vol fields. Saves a timestamped
CSV to `artifacts/data/microstructure/` and appends to a rolling panel file.

This is best-effort: Deribit exposes different fields depending on endpoint; the
script records whatever IV-like fields it finds and provides a simple aggregated
15m-ready CSV for downstream validation.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
import sys
import json

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts" / "data" / "microstructure"
ART_DIR.mkdir(parents=True, exist_ok=True)

DERIBIT_API = "https://www.deribit.com/api/v2"


def fetch_instruments(currency: str = "BTC") -> pd.DataFrame:
    res = requests.get(f"{DERIBIT_API}/public/get_instruments", params={"currency": currency, "kind": "option", "expired": "false"}, timeout=30)
    res.raise_for_status()
    instruments = res.json().get("result", [])
    return pd.DataFrame(instruments)


def probe_endpoints(instrument_name: str) -> dict:
    """Try a sequence of Deribit public endpoints and return any fields that look like IVs."""
    candidates = {}
    endpoints = [
        ("ticker", f"/public/ticker", {"instrument_name": instrument_name}),
        ("book_summary", f"/public/get_book_summary_by_instrument", {"instrument_name": instrument_name}),
        ("order_book", f"/public/get_order_book", {"instrument_name": instrument_name}),
        ("last_trades", f"/public/get_last_trades_by_instrument", {"instrument_name": instrument_name, "count": 10, "include_old": "true"}),
    ]

    for name, path, params in endpoints:
        try:
            res = requests.get(DERIBIT_API + path, params=params, timeout=30)
            if res.status_code != 200:
                continue
            payload = res.json().get("result") or {}
            # Look for common iv keys
            for key in ("iv", "mark_iv", "implied_volatility", "last_iv", "root_iv"):
                if key in payload:
                    candidates[key] = payload.get(key)
            # Sometimes ticker/book contains nested greeks/iv fields
            for k, v in payload.items():
                if isinstance(v, (int, float)) and 0 < v < 10:  # plausible iv ([0,1000%]) stored as 0-5
                    # Heuristic: anything small could be IV expressed as decimal
                    candidates[k] = v
            # If we found something plausible, also capture common useful fields
            for f in ("best_bid_price", "best_ask_price", "underlying_price", "index_price", "last_price"):
                if f in payload:
                    candidates[f] = payload[f]
            # stop early if we gathered a sensible iv
            if any(k in candidates for k in ("iv", "mark_iv", "implied_volatility")):
                return candidates
        except Exception:
            continue
        time.sleep(0.1)
    return candidates


def snapshot(currency: str = "BTC") -> pd.DataFrame:
    inst = fetch_instruments(currency)
    rows = []
    now = datetime.now(timezone.utc)
    for _, r in inst.iterrows():
        instrument = r.get("instrument_name")
        if not instrument:
            continue
        data = probe_endpoints(instrument)
        if not data:
            continue
        row = {
            "timestamp": now.isoformat(),
            "instrument_name": instrument,
            "strike": r.get("strike"),
            "option_type": r.get("option_type"),
            "expiration_timestamp": r.get("expiration_timestamp"),
        }
        # merge in probed fields
        row.update({f"p_{k}": v for k, v in data.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"]) 
    return df


def save_snapshot(df: pd.DataFrame):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ART_DIR / f"deribit_iv_snapshot_{ts}.csv"
    df.to_csv(out, index=False)
    panel = ART_DIR / "deribit_iv_panel.csv"
    if panel.exists():
        old = pd.read_csv(panel)
        combined = pd.concat([old, df.drop(columns=[c for c in df.columns if c.startswith("p_") and df[c].isna().all() if c])], ignore_index=True)
        combined.to_csv(panel, index=False)
    else:
        df.to_csv(panel, index=False)
    print(f"Saved snapshot {out} and updated panel {panel}")


def main():
    try:
        df = snapshot("BTC")
        if df.empty:
            print("No IV-like fields discovered in probes. Consider running again later or inspecting endpoints manually.")
            return
        save_snapshot(df)
    except Exception as exc:
        print(f"Deribit IV snapshot failed: {exc}")


if __name__ == "__main__":
    main()
