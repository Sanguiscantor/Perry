import yfinance as yf
import pandas as pd
from pathlib import Path


# ============================================
# CONFIG
# ============================================

CONFIG = {
    "tickers": [
        "RELIANCE.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "TCS.NS",
        "INFY.NS",
        "LT.NS",
        "TATASTEEL.NS",
        "ADANIENT.NS",
        "AXISBANK.NS",
    ],
    "interval": "15m",
    "period": "1mo",
}


# ============================================
# DOWNLOAD DATA
# ============================================

def fetch_symbol_data(ticker):

    print(f"\nDownloading {ticker}...")

    df = yf.download(
        tickers=ticker,
        interval=CONFIG["interval"],
        period=CONFIG["period"],
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print(f"  Skipped {ticker}: no data returned.")
        return None

    df = df.reset_index()

    # Flatten multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep required columns
    df = df[[
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]]

    # Timezone conversion
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    if df["Datetime"].dt.tz is not None:
        df["Datetime"] = (
            df["Datetime"]
            .dt.tz_convert("Asia/Kolkata")
        )

    # Round prices
    price_cols = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    df[price_cols] = (
        df[price_cols]
        .round(2)
    )

    df["symbol"] = ticker

    df = df.sort_values("Datetime").reset_index(drop=True)

    print(f"  {ticker}: {len(df)} rows")

    return df


def fetch_market_data():

    print("\nDownloading market data...\n")

    frames = []
    failed = []

    for ticker in CONFIG["tickers"]:
        try:
            df = fetch_symbol_data(ticker)

            if df is not None:
                frames.append(df)
            else:
                failed.append(ticker)

        except Exception as exc:
            print(f"  Skipped {ticker}: {exc}")
            failed.append(ticker)

    if not frames:
        raise ValueError("No data downloaded for any symbol.")

    master = pd.concat(frames, ignore_index=True)

    master = master.sort_values(
        ["symbol", "Datetime"]
    ).reset_index(drop=True)

    master = master[[
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "symbol",
    ]]

    if failed:
        print(f"\nFailed symbols ({len(failed)}): {', '.join(failed)}")

    return master


# ============================================
# SAVE DATASET
# ============================================

def save_dataset(df):

    output_path = (
        Path(__file__).resolve().parent
        / "master_raw_dataset.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("\nDataset saved successfully.")
    print(f"\nSaved to: {output_path}")
    print(f"\nTotal rows: {len(df)}")
    print(f"\nSymbols: {df['symbol'].nunique()}")


# ============================================
# MAIN
# ============================================

def main():

    df = fetch_market_data()

    save_dataset(df)

    print("\nRaw dataset generation complete.")


if __name__ == "__main__":
    main()
