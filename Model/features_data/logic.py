from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_PATH = Path(__file__).with_name("master_feature_dataset.csv")

REQUIRED_COLUMNS = [
    "Datetime",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

FEATURE_COLUMNS = [
    "Datetime",
    "Close",
    "momentum",
    "return_1",
    "volume_spike",
    "volatility",
    "target",
    "directional_volume",
]

ROLLING_WINDOW = 20
TARGET_CANDLE_OFFSET = 10


def find_input_csvs(data_dir: Path = DATA_DIR) -> list[Path]:
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    return csv_files


def load_market_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    df = df.dropna(subset=numeric_columns)

    return df.sort_values(by="Datetime").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["momentum"] = df["Close"] - df["Close"].shift(1)
    df["return_1"] = df["Close"].pct_change()

    df["avg_volume"] = df["Volume"].rolling(window=ROLLING_WINDOW).mean()
    df["volume_spike"] = df["Volume"] / df["avg_volume"]

    df["volatility"] = df["return_1"].rolling(window=ROLLING_WINDOW).std()

    df["directional_volume"] = df["Volume"] * (
        (df["Close"] > df["Close"].shift(1)).astype(int) * 2 - 1
    )

    future_close = df["Close"].shift(-TARGET_CANDLE_OFFSET)

    df["target"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[future_close.notna(), "target"] = (
        future_close.loc[future_close.notna()]
        > df.loc[future_close.notna(), "Close"]
    ).astype(int)

    required_feature_columns = [
        column
        for column in FEATURE_COLUMNS
        if column != "target"
    ]
    feature_df = df.dropna(subset=required_feature_columns).reset_index(drop=True)

    return feature_df


def build_master_feature_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    feature_frames = []

    for csv_path in find_input_csvs(data_dir):
        source_df = load_market_csv(csv_path)
        feature_df = build_features(source_df)

        if feature_df.empty:
            print(f"Skipped {csv_path.name}: not enough rows for rolling features")
            continue

        feature_frames.append(feature_df)

    if not feature_frames:
        raise ValueError("No feature rows were created from the input CSV files")

    return (
        pd.concat(feature_frames, ignore_index=True)
        .sort_values(by="Datetime")
        .reset_index(drop=True)
    )


def save_feature_dataset(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    try:
        df.to_csv(
            output_path,
            index=False,
        )
    except PermissionError as exc:
        raise PermissionError(
            f"Unable to write {output_path}. Close the file if it is open "
            "in Excel or another program, then run this script again."
        ) from exc


def main() -> None:
    df = build_master_feature_dataset()

    print(df[FEATURE_COLUMNS])

    save_feature_dataset(df)

    print(f"\nSaved feature dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
