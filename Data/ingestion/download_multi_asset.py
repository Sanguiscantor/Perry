"""Download 15m Binance klines for Perry cross-asset research. Does not overwrite BTC-only master."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from binance.client import Client

ROOT = Path(__file__).resolve().parents[1]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
START_DATE = "1 Jan, 2024"
OUTPUT = ROOT / "datasets" / "raw" / "multi_asset_dataset.csv"


def fetch_symbol(client: Client, symbol: str) -> pd.DataFrame:
    print(f"Downloading {symbol}...")
    klines = client.get_historical_klines(symbol, INTERVAL, START_DATE)
    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["Datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col])
    df["symbol"] = symbol
    return df[["Datetime", "Open", "High", "Low", "Close", "Volume", "symbol"]]


def main() -> None:
    client = Client()
    frames = [fetch_symbol(client, symbol) for symbol in SYMBOLS]
    master = pd.concat(frames, ignore_index=True)
    master = master.sort_values(["symbol", "Datetime"]).reset_index(drop=True)
    master.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT} ({len(master):,} rows)")
    for symbol in SYMBOLS:
        count = (master["symbol"] == symbol).sum()
        print(f"  {symbol}: {count:,}")


if __name__ == "__main__":
    main()
