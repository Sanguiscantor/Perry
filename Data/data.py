from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
import yfinance.cache as yf_cache

try:
    from IPython.display import display
except ImportError:
    display = None


pd.options.display.float_format = "{:.2f}".format

DATA_DIR = Path(__file__).resolve().parent
YFINANCE_CACHE_DIR = DATA_DIR / ".yfinance_cache"
YFINANCE_CACHE_DIR.mkdir(exist_ok=True)
yf_cache.set_cache_location(str(YFINANCE_CACHE_DIR))
yf_cache.set_tz_cache_location(str(YFINANCE_CACHE_DIR))


# =========================
# OPTIONS
# =========================

intervals = {
    1: "1m",
    2: "2m",
    3: "5m",
    4: "15m",
    5: "30m",
    6: "60m",
    7: "90m",
    8: "1d",
    9: "5d",
    10: "1wk",
    11: "1mo",
    12: "3mo",
}

stocks = {
    "GSFC": "GSFC.NS",
    "RELIANCE": "RELIANCE.NS",
    "COFFEEDAY": "COFFEEDAY.NS",
    "TRIDENT": "TRIDENT.NS",
    "ITC": "ITC.NS",
}


# =========================
# PERIOD MAP
# =========================

period_map = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "90m": "60d",
    "1d": "max",
    "5d": "max",
    "1wk": "max",
    "1mo": "max",
    "3mo": "max",
}


# =========================
# FETCH MARKET DATA
# =========================

def fetch_market_data(
    ticker: str,
    interval: str,
    period: Optional[str] = None,
    start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    df = yf.download(
        ticker,
        interval=interval,
        period=period,
        start=start,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Datetime" not in df.columns:
        df = df.rename(columns={"Date": "Datetime"})

    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        utc=True,
        errors="coerce",
    ).dt.tz_convert("Asia/Kolkata")

    df = df[
        [
            "Datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    df[price_columns] = df[price_columns].round(2)
    df["body_mid"] = ((df["Open"] + df["Close"]) / 2).round(2)

    return df.dropna(subset=["Datetime"]).reset_index(drop=True)


# =========================
# CSV HANDLING
# =========================

def dataset_path(file_name: str) -> Path:
    return DATA_DIR / file_name


def normalize_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        utc=True,
        errors="coerce",
    ).dt.tz_convert("Asia/Kolkata")

    return df.dropna(subset=["Datetime"])


def load_existing(file_path: Path) -> pd.DataFrame:
    if file_path.exists():
        df = pd.read_csv(file_path)
        return normalize_datetime_column(df)

    return pd.DataFrame()


def update_dataset(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    if old_df.empty:
        combined = new_df.copy()
    else:
        combined = pd.concat(
            [old_df, new_df],
            ignore_index=True,
        )

    combined = normalize_datetime_column(combined)

    return (
        combined.drop_duplicates(
            subset=["Datetime"],
            keep="last",
        )
        .sort_values(by="Datetime")
        .reset_index(drop=True)
    )


def save_dataset(
    df: pd.DataFrame,
    file_path: Path,
) -> None:
    temp_file = file_path.with_name(f"{file_path.stem}_temp{file_path.suffix}")

    df.to_csv(
        temp_file,
        index=False,
        float_format="%.2f",
    )

    os.replace(
        temp_file,
        file_path,
    )


# =========================
# PIPELINE
# =========================

def run_pipeline(
    name: str,
    ticker: str,
    interval: str,
) -> None:
    period = period_map.get(
        interval,
        "max",
    )

    file_name = f"{name}_{interval}_{period}_ohlcv.csv"
    file_path = dataset_path(file_name)

    existing_df = load_existing(file_path)

    if existing_df.empty:
        new_df = fetch_market_data(
            ticker=ticker,
            interval=interval,
            period=period,
        )
    else:
        last_time = existing_df["Datetime"].max()

        buffer_map = {
            "1m": 1,
            "2m": 2,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "60m": 60,
            "90m": 90,
        }

        buffer = buffer_map.get(
            interval,
            1440,
        )

        safe_start = last_time - timedelta(minutes=buffer)

        new_df = fetch_market_data(
            ticker=ticker,
            interval=interval,
            start=safe_start,
        )

    if new_df.empty:
        print(f"{name} | {interval} -> No new data")
        return

    updated_df = update_dataset(
        existing_df,
        new_df,
    )

    save_dataset(
        updated_df,
        file_path,
    )

    print(f"\n{name} (NSE) | {interval} | {period}")
    print(f"Rows : {len(updated_df)}")
    print(f"File : {file_path}")

    show_dataframe(updated_df)


def show_dataframe(df: pd.DataFrame) -> None:
    if display is not None:
        display(df)
    else:
        print(df)


# =========================
# USER INPUT
# =========================

def parse_interval_selection(value: str) -> List[str]:
    selected_intervals = []

    for item in value.split(","):
        item = item.strip()

        if item.isdigit():
            key = int(item)

            if key in intervals:
                selected_intervals.append(intervals[key])

    return selected_intervals


def parse_stock_list(value: str) -> List[str]:
    names = [
        item.strip().upper()
        for item in value.split(",")
        if item.strip()
    ]

    invalid_names = [
        name
        for name in names
        if name not in stocks
    ]

    if invalid_names:
        invalid = ", ".join(invalid_names)
        available = ", ".join(stocks)
        raise ValueError(f"Unknown stocks: {invalid}. Available stocks: {available}")

    return names


def prompt_for_stock_intervals() -> Dict[str, Tuple[str, List[str]]]:
    print("\nAvailable Intervals:\n")

    for key, value in intervals.items():
        print(f"{key}. {value}")

    print("\nEnter intervals for each stock")
    print("Example : 1,3,8")
    print("Use 0 to skip\n")

    stock_interval_map = {}

    for name, ticker in stocks.items():
        try:
            user_input = input(f"{name}: ").strip()
        except EOFError:
            print("\nNo interactive input received.")
            return {}

        if user_input in ["0", ""]:
            continue

        selected_intervals = parse_interval_selection(user_input)

        if selected_intervals:
            stock_interval_map[name] = (
                ticker,
                selected_intervals,
            )

    return stock_interval_map


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download NSE market data from yfinance into the Data folder."
    )
    parser.add_argument(
        "--stocks",
        default="",
        help="Comma-separated stock names to fetch, e.g. GSFC,ITC. Omit for interactive mode.",
    )
    parser.add_argument(
        "--intervals",
        default="",
        help="Comma-separated interval numbers, e.g. 3,8. Omit for interactive mode.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.stocks.strip() and args.intervals.strip():
        selected_intervals = parse_interval_selection(args.intervals)
        selected_stocks = parse_stock_list(args.stocks)

        stock_interval_map = {
            name: (
                stocks[name],
                selected_intervals,
            )
            for name in selected_stocks
        }
    else:
        stock_interval_map = prompt_for_stock_intervals()

    for name, (ticker, interval_list) in stock_interval_map.items():
        for interval in interval_list:
            run_pipeline(
                name=name,
                ticker=ticker,
                interval=interval,
            )


if __name__ == "__main__":
    main()
