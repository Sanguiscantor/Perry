"""Download Binance USDT-M futures klines with taker-flow and trade-count fields."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "futures_klines_15m.csv"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
FAPI = "https://fapi.binance.com"
START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
INTERVAL = "15m"
LIMIT = 1500


def fetch_klines(symbol: str) -> pd.DataFrame:
    print(f"Klines {symbol}...")
    rows: list = []
    start = START_MS
    while True:
        params = {"symbol": symbol, "interval": INTERVAL, "startTime": start, "limit": LIMIT}
        response = requests.get(f"{FAPI}/fapi/v1/klines", params=params, timeout=60)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        start = int(batch[-1][0]) + 1
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
    frames = [fetch_klines(s) for s in SYMBOLS]
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "Datetime"])
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
