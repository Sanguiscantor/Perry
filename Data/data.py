import yfinance as yf
import pandas as pd
from pathlib import Path


# ============================================
# CONFIG
# ============================================

CONFIG = {
    "ticker": "RELIANCE.NS",
    "interval": "5m",
    "period": "60d"
}


# ============================================
# DOWNLOAD DATA
# ============================================

def fetch_market_data():

    print("\nDownloading market data...\n")

    df = yf.download(
        tickers=CONFIG["ticker"],
        interval=CONFIG["interval"],
        period=CONFIG["period"],
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("No data downloaded.")

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
        "Volume"
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
        "Close"
    ]

    df[price_cols] = (
        df[price_cols]
        .round(2)
    )

    return df


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
        index=False
    )

    print("\nDataset saved successfully.")
    print(f"\nSaved to: {output_path}")
    print(f"\nTotal rows: {len(df)}")


# ============================================
# MAIN
# ============================================

def main():

    df = fetch_market_data()

    save_dataset(df)

    print("\nRaw dataset generation complete.")


if __name__ == "__main__":
    main()