import tkinter as tk
import winsound

def show_completion_popup(message):

    winsound.MessageBeep()

    root = tk.Tk()

    root.title("Perry")

    root.geometry("400x120")

    label = tk.Label(
        root,
        text=message,
        font=("Arial", 12),
        pady=20,
    )

    label.pack()

    root.mainloop()

from binance.client import Client
import pandas as pd
from pathlib import Path

client = Client()

SYMBOLS = [
    "BTCUSDT",
    #"ETHUSDT",
    #"SOLUSDT",
    #"BNBUSDT",
    #"XRPUSDT",
]

INTERVAL = Client.KLINE_INTERVAL_15MINUTE

START_DATE = "1 Jan, 2024"

def fetch_symbol(symbol):

    print(f"\nDownloading {symbol}")

    klines = client.get_historical_klines(
        symbol,
        INTERVAL,
        START_DATE,
    )

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df["Datetime"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
    )

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        df[col] = pd.to_numeric(df[col])

    df["symbol"] = symbol

    return df[
        [
            "Datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "symbol",
        ]
    ]


def main():

    frames = []

    for symbol in SYMBOLS:
        frames.append(
            fetch_symbol(symbol)
        )

    master = pd.concat(
        frames,
        ignore_index=True,
    )

    output_path = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "raw"
        / "master_raw_dataset.csv"
    )

    master.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved")
    print(output_path)
    print(len(master))
    show_completion_popup(
    f"Download Complete\n\nRows: {len(master):,}"
    )

if __name__ == "__main__":
    main()