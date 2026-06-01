import time
from pathlib import Path

import numpy as np
import pandas as pd
from tsfresh import extract_features, select_features


WINDOW_SIZE = 96
SAMPLE_LIMIT = 40000
HORIZON = 72
N_JOBS = 12

VALUE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
META_COLUMNS = ["window_end_idx", "window_end_time", "target_d"]


def load_sample_data():

    raw_path = (
        Path(__file__).resolve().parents[2]
        / "Data"
        / "master_raw_dataset.csv"
    )

    df = pd.read_csv(raw_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="mixed")

    return df.iloc[:SAMPLE_LIMIT].copy()


def add_target_d(df):

    if "symbol" in df.columns:
        future_close = (
            df.groupby("symbol")["Close"]
            .shift(-HORIZON)
        )
    else:
        future_close = df["Close"].shift(-HORIZON)

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

    return df


def build_rolled_dataframe(df):

    window_frames = []
    meta_rows = []

    for end_idx in range(WINDOW_SIZE - 1, len(df)):
        window_df = (
            df.iloc[end_idx - WINDOW_SIZE + 1 : end_idx + 1][
                ["Datetime"] + VALUE_COLUMNS
            ]
            .copy()
        )
        window_df["series_id"] = end_idx
        window_frames.append(window_df)

        meta_rows.append({
            "series_id": end_idx,
            "window_end_idx": end_idx,
            "window_end_time": df.iloc[end_idx]["Datetime"],
            "target_d": df.iloc[end_idx]["target_d"],
        })

    long_df = pd.concat(window_frames, ignore_index=True)
    meta = pd.DataFrame(meta_rows)

    return long_df, meta


def merge_extracted_features(features, meta):

    features = features.copy()

    if features.index.name != "series_id":
        features.index.name = "series_id"

    features = features.reset_index()

    if "index" in features.columns and "series_id" not in features.columns:
        features = features.rename(columns={"index": "series_id"})

    if "id" in features.columns and "series_id" not in features.columns:
        features = features.rename(columns={"id": "series_id"})

    return meta.merge(
        features,
        on="series_id",
        how="inner",
    )


def memory_mb(frame):

    return frame.memory_usage(deep=True).sum() / (1024 ** 2)


def run_benchmark():

    output_dir = Path(__file__).resolve().parent

    print("\nTSFresh rolling-window benchmark (batched extraction)")
    print(f"  Samples: {SAMPLE_LIMIT}")
    print(f"  Window size: {WINDOW_SIZE}")
    print(f"  n_jobs: {N_JOBS}")

    df = add_target_d(load_sample_data())
    print(f"\nLoaded rows: {len(df)}")

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"Need at least {WINDOW_SIZE} rows, got {len(df)}"
        )

    build_start = time.perf_counter()
    long_df, meta = build_rolled_dataframe(df)
    build_elapsed = time.perf_counter() - build_start

    window_count = len(meta)
    print(f"\nBuilt long dataframe: {long_df.shape[0]} rows, {window_count} windows")
    print(f"Window assembly time (seconds): {build_elapsed:.2f}")

    extract_start = time.perf_counter()

    features = extract_features(
        long_df,
        column_id="series_id",
        column_sort="Datetime",
        n_jobs=N_JOBS,
    )

    extract_elapsed = time.perf_counter() - extract_start

    results = merge_extracted_features(features, meta).copy()
    feature_cols = [
        col for col in results.columns
        if col not in META_COLUMNS and col != "series_id"
    ]

    features_path = output_dir / "tsfresh_benchmark_features.csv"
    feature_names_path = output_dir / "tsfresh_feature_names.csv"

    results.to_csv(features_path, index=False)
    pd.Series(feature_cols, name="feature").to_csv(
        feature_names_path,
        index=False,
    )

    print("\n--- Extraction results ---")
    print(f"Runtime (seconds): {extract_elapsed:.2f}")
    print(f"Windows processed: {window_count}")
    print(f"Feature matrix shape: {results[feature_cols].shape}")
    print(f"Avg seconds per window: {extract_elapsed / window_count:.3f}")
    print(f"\nSaved feature matrix: {features_path}")
    print(f"Saved feature names: {feature_names_path}")

    print("\nFirst 10 rows of feature matrix:")
    print(results[feature_cols].head(10))

    print("\nFirst 50 feature names:")
    for name in feature_cols[:50]:
        print(name)

    selection_df = results.dropna(subset=["target_d"]).copy()
    X_full = (
        selection_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .copy()
    )
    y = selection_df["target_d"]

    original_count = len(feature_cols)

    print("\nRunning TSFresh feature selection against target_d...")
    selection_start = time.perf_counter()

    X_selected = select_features(
        X_full,
        y,
        multiclass=True,
        n_jobs=N_JOBS,
    )

    selection_elapsed = time.perf_counter() - selection_start
    selected_cols = X_selected.columns.tolist()
    selected_count = len(selected_cols)
    reduction_pct = (
        (1 - selected_count / original_count) * 100
        if original_count > 0
        else 0.0
    )

    selected_features = selection_df[
        META_COLUMNS + selected_cols
    ].copy()

    selected_path = output_dir / "tsfresh_selected_features.csv"
    selected_names_path = (
        output_dir / "tsfresh_selected_feature_names.csv"
    )

    selected_features.to_csv(selected_path, index=False)
    pd.Series(selected_cols, name="feature").to_csv(
        selected_names_path,
        index=False,
    )

    print("\n--- Feature selection results ---")
    print(f"Selection runtime (seconds): {selection_elapsed:.2f}")
    print(f"Original feature count: {original_count}")
    print(f"Selected feature count: {selected_count}")
    print(f"Percentage reduction: {reduction_pct:.2f}%")
    print(f"\nSaved selected features: {selected_path}")
    print(f"Saved selected feature names: {selected_names_path}")

    print("\nFirst 50 selected feature names:")
    for name in selected_cols[:50]:
        print(name)

    print("\n--- Memory usage ---")
    print(
        f"Original feature matrix: "
        f"{memory_mb(X_full):.2f} MB"
    )
    print(
        f"Selected feature matrix: "
        f"{memory_mb(X_selected):.2f} MB"
    )
    print(
        f"Full results (with metadata): "
        f"{memory_mb(results):.2f} MB"
    )


if __name__ == "__main__":
    run_benchmark()
