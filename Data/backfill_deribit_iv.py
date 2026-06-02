"""Best-effort backfill of Deribit IV panel for a target window.

This script tries to discover historical IV-like snapshots from public Deribit
endpoints. Public Deribit APIs generally expose current state only, so this
attempt will usually fail to produce a full 6-month backfill. The script:

- loads the current instrument list
- inspects existing snapshots to see coverage
- attempts to query endpoints for older data where supported
- writes any discovered snapshots to the rolling panel

Use this to test feasibility; if no historical data is available, the script
will print guidance recommending vendor data or reconstructing IVs from
historical option prices (if you can obtain them).
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
import time
import sys
import json

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts" / "data" / "microstructure"
ART_DIR.mkdir(parents=True, exist_ok=True)

DERIBIT_API = "https://www.deribit.com/api/v2"


def load_instruments():
    f = ART_DIR / "deribit_instruments_BTC.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)


def existing_coverage(panel: Path):
    if not panel.exists():
        return None
    df = pd.read_csv(panel)
    if "timestamp_dt" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"])
        return df["timestamp_dt"].min(), df["timestamp_dt"].max()
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        return df["timestamp_dt"].min(), df["timestamp_dt"].max()
    return None


def probe_instrument_for_time(instrument: str, ts_millis: int):
    """Attempt to retrieve an endpoint response that might contain historical
    fields for a given instrument at or before ts_millis. This is optimistic —
    most public endpoints ignore timestamp args. Returns dict or None.
    """
    endpoints = [
        ("get_book_summary_by_instrument", f"/public/get_book_summary_by_instrument", {"instrument_name": instrument}),
        ("ticker", "/public/ticker", {"instrument_name": instrument}),
        ("order_book", "/public/get_order_book", {"instrument_name": instrument}),
    ]
    # Some endpoints accept timestamp-like params; try adding end_timestamp
    params_extra = {"end_timestamp": ts_millis}

    for name, path, base_params in endpoints:
        params = dict(base_params)
        params.update(params_extra)
        try:
            res = requests.get(DERIBIT_API + path, params=params, timeout=20)
            if res.status_code != 200:
                continue
            payload = res.json().get("result") or {}
            # Heuristics: choose fields that look like IV / mark_iv
            candidates = {}
            for key in ("iv", "mark_iv", "implied_volatility", "last_iv"):
                if key in payload:
                    candidates[key] = payload.get(key)
            for k, v in payload.items():
                if isinstance(v, (int, float)) and 0 < v < 1000:
                    # plausible numeric value
                    candidates[k] = v
            if candidates:
                return candidates
        except Exception:
            continue
        time.sleep(0.05)
    return None


def main(months: int = 6):
    panel = ART_DIR / "deribit_iv_panel.csv"
    cov = existing_coverage(panel)
    now = datetime.now(timezone.utc)
    start_target = now - timedelta(days=30 * months)

    print(f"Existing panel coverage: {cov}")
    print(f"Target backfill window start: {start_target.isoformat()}")

    inst = load_instruments()
    if inst is None:
        print("No instrument list found (deribit_instruments_BTC.csv). Run Data/retry_deribit.py first.")
        return

    # If panel already covers the window, nothing to do
    if cov and cov[0] <= pd.Timestamp(start_target):
        print("Panel already covers target window; no backfill needed.")
        return

    # Best-effort: iterate timestamps monthly and probe instruments for historical payloads
    found_rows = []
    for month_offset in range(months, 0, -1):
        target_ts = now - timedelta(days=30 * month_offset)
        ts_millis = int(target_ts.timestamp() * 1000)
        print(f"Probing around {target_ts.date()} (ts={ts_millis})")
        for _, r in inst.iterrows():
            instr = r.get("instrument_name")
            if not instr:
                continue
            data = probe_instrument_for_time(instr, ts_millis)
            if not data:
                continue
            row = {
                "timestamp": target_ts.isoformat(),
                "instrument_name": instr,
                "strike": r.get("strike"),
                "option_type": r.get("option_type"),
                "expiration_timestamp": r.get("expiration_timestamp"),
            }
            row.update({f"p_{k}": v for k, v in data.items()})
            found_rows.append(row)

    if not found_rows:
        print("No historical IV-like data available via public endpoints."
              " To backfill 6 months you will likely need vendor data or"
              " reconstruct IVs from archived option price feeds.")
        return

    df = pd.DataFrame(found_rows)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"]) 
    # append to panel
    if panel.exists():
        old = pd.read_csv(panel)
        combined = pd.concat([old, df], ignore_index=True)
        combined.to_csv(panel, index=False)
    else:
        df.to_csv(panel, index=False)
    print(f"Appended {len(df)} historical snapshots to {panel}")


if __name__ == "__main__":
    main(6)
