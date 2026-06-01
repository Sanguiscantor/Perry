import tkinter as tk
import winsound

import numpy as np
import pandas as pd
from pathlib import Path
from tsfresh import extract_features, select_features


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


def main():

    # ============================================
    # LOAD RAW DATASET
    # ============================================

    base_path = Path(__file__).resolve().parents[2]

    raw_path = (
        base_path
        / "Data"
        / "master_raw_dataset.csv"
    )

    print(f"\nLoading raw dataset from: {raw_path}")

    df = pd.read_csv(raw_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="mixed")
    df["series_id"] = df["symbol"]

    window = 96
    horizon = 72

    # ============================================
    # FEATURE EXTRACTION
    # ============================================

    print("\nExtracting features with tsfresh...")
    window_df = df.iloc[:window].copy()
    window_df["series_id"] = 0

    tsfresh_df = window_df[
        ["Datetime", "series_id", "Open", "High", "Low", "Close", "Volume"]
    ].copy()

    # n_jobs=0 disables multiprocessing (required on Windows spawn)
    features = extract_features(
        tsfresh_df,
        column_id="series_id",
        column_sort="Datetime",
        n_jobs=0,
    )

    print(features.shape)
    pd.Series(features.columns).to_csv(
    "tsfresh_feature_names.csv",
    index=False
    )

    # ============================================
    # TARGETS
    # ============================================

    future_close = (
        df.groupby("symbol")["Close"]
        .shift(-horizon)
    )

    future_return = (
        future_close - df["Close"]
    ) / df["Close"]

    bullish_threshold = 0.0075
    bearish_threshold = -0.0075

    bullish = future_return > bullish_threshold
    bearish = future_return < bearish_threshold

    df["target_d"] = np.select(
        [bullish, bearish],
        [2, 0],
        default=1,
    )

    df["target_p"] = future_return

    print("\nNew Target Distribution:")
    print(df["target_d"].value_counts())

    # ============================================
    # CLEAN DATA
    # ============================================

    df = df.dropna(subset=["target_p"])

    # ============================================
    # SAVE FEATURE DATASET
    # ============================================

    output_path = (
        Path(__file__).resolve().parent
        / "master_feature_dataset.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nFeature dataset saved successfully.")
    print(f"\nSaved to: {output_path}")
    print(f"\nFinal rows: {len(df)}")

    show_completion_popup(
        f"Feature Compilation Complete\n\nRows: {len(df):,}"
    )


if __name__ == "__main__":
    main()
