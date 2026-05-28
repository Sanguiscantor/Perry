import json
import math
import webbrowser
from pathlib import Path

import pandas as pd


SELECTED_SYMBOL = "RELIANCE"
DEFAULT_CANDLE_LIMIT = 500
CANDLE_COUNT_OPTIONS = [100, 200, 300, 500, 1000]


VISUALIZATION_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = VISUALIZATION_DIR / "frontend"
DATA_FILE = FRONTEND_DIR / "data" / "marketStructureData.js"
INDEX_FILE = FRONTEND_DIR / "index.html"


REQUIRED_COLUMNS = [
    "Datetime",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "symbol",
    "swing_high",
    "swing_low",
    "anchored_support",
    "anchored_resistance",
    "anchored_breakout",
    "anchored_breakdown",
    "support_anchor_1_position",
    "support_anchor_1_price",
    "support_anchor_2_position",
    "support_anchor_2_price",
    "resistance_anchor_1_position",
    "resistance_anchor_1_price",
    "resistance_anchor_2_position",
    "resistance_anchor_2_price",
]


def load_dataset():
    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "features_data"
        / "master_feature_dataset.csv"
    )

    df = pd.read_csv(dataset_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="mixed")

    return df


def validate_columns(df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "Missing required columns for market structure visualization: "
            + ", ".join(missing_columns)
        )


def normalize_symbol(symbol):
    return str(symbol).upper().split(".")[0]


def default_symbol(symbols):
    selected_key = SELECTED_SYMBOL.upper()

    for symbol in symbols:
        if str(symbol).upper() == selected_key or normalize_symbol(symbol) == selected_key:
            return symbol

    return symbols[0]


def clean_number(value):
    if pd.isna(value):
        return None

    number = float(value)

    if math.isnan(number) or math.isinf(number):
        return None

    return number


def clean_int(value):
    if pd.isna(value):
        return None

    return int(value)


def to_lightweight_time(timestamp):
    return int(pd.Timestamp(timestamp).timestamp())


def build_row(row, position):
    return {
        "time": to_lightweight_time(row["Datetime"]),
        "open": clean_number(row["Open"]),
        "high": clean_number(row["High"]),
        "low": clean_number(row["Low"]),
        "close": clean_number(row["Close"]),
        "volume": clean_number(row["Volume"]),
        "position": position,
        "swingHigh": int(row["swing_high"]) == 1,
        "swingLow": int(row["swing_low"]) == 1,
        "anchoredBreakout": int(row["anchored_breakout"]) == 1,
        "anchoredBreakdown": int(row["anchored_breakdown"]) == 1,
        "supportAnchor1Position": clean_int(row["support_anchor_1_position"]),
        "supportAnchor1Price": clean_number(row["support_anchor_1_price"]),
        "supportAnchor2Position": clean_int(row["support_anchor_2_position"]),
        "supportAnchor2Price": clean_number(row["support_anchor_2_price"]),
        "resistanceAnchor1Position": clean_int(row["resistance_anchor_1_position"]),
        "resistanceAnchor1Price": clean_number(row["resistance_anchor_1_price"]),
        "resistanceAnchor2Position": clean_int(row["resistance_anchor_2_position"]),
        "resistanceAnchor2Price": clean_number(row["resistance_anchor_2_price"]),
    }


def build_visualization_payload(df):
    validate_columns(df)

    symbols = sorted(df["symbol"].dropna().astype(str).unique())

    if not symbols:
        raise ValueError("No symbols found in the feature dataset.")

    payload = {
        "symbols": symbols,
        "defaultSymbol": default_symbol(symbols),
        "defaultCandleCount": DEFAULT_CANDLE_LIMIT,
        "candleCountOptions": CANDLE_COUNT_OPTIONS,
        "series": {},
    }

    for symbol in symbols:
        symbol_df = (
            df[df["symbol"].astype(str) == symbol]
            .sort_values("Datetime")
            .reset_index(drop=True)
        )

        payload["series"][symbol] = [
            build_row(row, position)
            for position, (_, row) in enumerate(symbol_df.iterrows())
        ]

    return payload


def write_data_file(payload):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, separators=(",", ":"))
    DATA_FILE.write_text(
        f"window.PERRY_MARKET_DATA = {data_json};\n",
        encoding="utf-8",
    )


def open_chart():
    webbrowser.open(INDEX_FILE.as_uri())


def main():
    df = load_dataset()
    payload = build_visualization_payload(df)
    write_data_file(payload)
    open_chart()
    print(f"Opened TradingView Lightweight Charts inspector: {INDEX_FILE}")


if __name__ == "__main__":
    main()
