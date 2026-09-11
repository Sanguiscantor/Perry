"""Download Binance USDT-M futures klines with taker-flow and trade-count fields."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
from perry_config import binance_interval, filename_with_timeframe, get_primary_timeframe

OUT_DIR = ROOT / "datasets" / "raw"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
FAPI = "https://fapi.binance.com"
START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
LIMIT = 1500


def fetch_klines(symbol: str, start_time_ms: int | None = None) -> pd.DataFrame:
    print(f"Klines {symbol}...")
    rows: list = []
    start = start_time_ms if start_time_ms is not None else START_MS
    end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if start is not None and start > end_time_ms:
        return pd.DataFrame(columns=["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"])

    while True:
        interval = binance_interval()
        params = {"symbol": symbol, "interval": interval, "startTime": start, "limit": LIMIT}
        response = requests.get(f"{FAPI}/fapi/v1/klines", params=params, timeout=60)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        if last_open_time >= end_time_ms:
            break
        start = last_open_time + 1
        if len(batch) < LIMIT:
            break
        time.sleep(0.12)
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )
    for col in ["Open", "High", "Low", "Close", "Volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col])
    df["trades"] = pd.to_numeric(df["trades"])
    df["Datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df["symbol"] = symbol
    df["taker_sell_base"] = df["Volume"] - df["taker_buy_base"]
    df["taker_buy_ratio"] = df["taker_buy_base"] / df["Volume"].replace(0, np.nan)
    df["taker_imbalance"] = (df["taker_buy_base"] - df["taker_sell_base"]) / df["Volume"].replace(0, np.nan)
    return df[
        [
            "Datetime", "symbol", "Open", "High", "Low", "Close", "Volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
            "taker_buy_ratio", "taker_imbalance",
        ]
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [fetch_klines(s) for s in SYMBOLS]
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "Datetime"])
    output_path = OUT_DIR / filename_with_timeframe("futures_klines", get_primary_timeframe())
    out.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
