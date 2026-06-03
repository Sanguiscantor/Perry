"""Download Binance USDT-M futures funding rate and open interest (public API)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "raw" / "derivatives"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
FAPI = "https://fapi.binance.com"


def _paginate(url: str, params: dict, time_key: str = "timestamp", limit: int = 1000) -> list:
    rows: list = []
    start = params.get("startTime", START_MS)
    while True:
        p = {**params, "startTime": start, "limit": limit}
        response = requests.get(url, params=p, timeout=60)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        start = int(batch[-1][time_key]) + 1
        time.sleep(0.15)
    return rows


def fetch_funding(symbol: str) -> pd.DataFrame:
    print(f"Funding {symbol}...")
    raw = _paginate(
        f"{FAPI}/fapi/v1/fundingRate",
        {"symbol": symbol, "startTime": START_MS},
        time_key="fundingTime",
    )
    df = pd.DataFrame(raw)
    df["Datetime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_convert(None)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"])
    df["symbol"] = symbol
    return df[["Datetime", "symbol", "funding_rate"]]


def fetch_open_interest(symbol: str, period: str = "15m") -> pd.DataFrame:
    """Binance OI history is capped (~30 days per request window); walk backward in time."""
    print(f"Open interest {symbol} ({period})...")
    window_ms = 29 * 24 * 60 * 60 * 1000
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows: list = []
    while end > START_MS:
        start = max(START_MS, end - window_ms)
        params = {"symbol": symbol, "period": period, "endTime": end, "limit": 500}
        if end - START_MS > window_ms:
            params["startTime"] = start
        response = requests.get(f"{FAPI}/futures/data/openInterestHist", params=params, timeout=60)
        if response.status_code != 200:
            print(f"  OI chunk failed {symbol} {response.status_code}: {response.text[:200]}")
            break
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        end = start - 1
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"])
    df["Datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"])
    df["oi_value"] = pd.to_numeric(df["sumOpenInterestValue"])
    df["symbol"] = symbol
    return df[["Datetime", "symbol", "open_interest", "oi_value"]].sort_values("Datetime")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    funding_frames, oi_frames = [], []
    for symbol in SYMBOLS:
        try:
            funding_frames.append(fetch_funding(symbol))
        except Exception as exc:
            print(f"  funding failed {symbol}: {exc}")
        try:
            oi_frames.append(fetch_open_interest(symbol))
        except Exception as exc:
            print(f"  OI failed {symbol}: {exc}")

    if funding_frames:
        funding = pd.concat(funding_frames, ignore_index=True)
        funding.to_csv(OUT_DIR / "funding_rates.csv", index=False)
        print(f"Saved funding_rates.csv ({len(funding):,} rows)")
    if oi_frames:
        oi = pd.concat(oi_frames, ignore_index=True)
        oi.to_csv(OUT_DIR / "open_interest_15m.csv", index=False)
        print(f"Saved open_interest_15m.csv ({len(oi):,} rows)")


if __name__ == "__main__":
    main()
