"""Paginate Binance futures sentiment metrics backward (OI, L/S, taker ratio)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "derivatives"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
FAPI = "https://fapi.binance.com/futures/data"
START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

from perry_config import filename_with_timeframe, get_primary_timeframe

LIMIT = 500
MAX_CHUNKS = 200


def paginate_metric(endpoint: str, symbol: str, parser) -> pd.DataFrame:
    print(f"  {endpoint} {symbol}...")
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows: list = []
    for _ in range(MAX_CHUNKS):
        response = requests.get(
            f"{FAPI}/{endpoint}",
            params={"symbol": symbol, "period": get_primary_timeframe(), "limit": LIMIT, "endTime": end},
            timeout=60,
        )
        if response.status_code != 200:
            print(f"    stop {response.status_code}: {response.text[:120]}")
            break
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        oldest = int(batch[0]["timestamp"])
        if oldest <= START_MS or oldest <= 0:
            break
        end = oldest - 1
        if end <= START_MS:
            break
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    return parser(rows, symbol)


def parse_oi(raw: list, symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(raw).drop_duplicates(subset=["timestamp"])
    df["Datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"])
    df["oi_value"] = pd.to_numeric(df["sumOpenInterestValue"])
    df["symbol"] = symbol
    return df[["Datetime", "symbol", "open_interest", "oi_value"]]


def parse_ls(raw: list, symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(raw).drop_duplicates(subset=["timestamp"])
    df["Datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df["long_short_ratio"] = pd.to_numeric(df["longShortRatio"])
    df["long_account_pct"] = pd.to_numeric(df["longAccount"])
    df["symbol"] = symbol
    return df[["Datetime", "symbol", "long_short_ratio", "long_account_pct"]]


def parse_taker(raw: list, symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(raw).drop_duplicates(subset=["timestamp"])
    df["Datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df["taker_buy_sell_ratio"] = pd.to_numeric(df["buySellRatio"])
    df["taker_buy_vol"] = pd.to_numeric(df["buyVol"])
    df["taker_sell_vol"] = pd.to_numeric(df["sellVol"])
    df["symbol"] = symbol
    return df[["Datetime", "symbol", "taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oi_frames, global_ls, top_ls, taker = [], [], [], []
    for symbol in SYMBOLS:
        oi_frames.append(paginate_metric("openInterestHist", symbol, parse_oi))
        global_ls.append(paginate_metric("globalLongShortAccountRatio", symbol, parse_ls))
        top_ls.append(paginate_metric("topLongShortPositionRatio", symbol, parse_ls))
        taker.append(paginate_metric("takerlongshortRatio", symbol, parse_taker))

    period = get_primary_timeframe()
    for name, frames in [
        (filename_with_timeframe("open_interest", period), oi_frames),
        (filename_with_timeframe("global_long_short", period), global_ls),
        (filename_with_timeframe("top_trader_long_short", period), top_ls),
        (filename_with_timeframe("taker_buy_sell", period), taker),
    ]:
        valid = [f for f in frames if len(f)]
        if valid:
            out = pd.concat(valid, ignore_index=True).sort_values(["symbol", "Datetime"])
            path = OUT_DIR / name
            out.to_csv(path, index=False)
            print(f"Saved {path} ({len(out):,} rows, {out.Datetime.min()} to {out.Datetime.max()})")


if __name__ == "__main__":
    main()
