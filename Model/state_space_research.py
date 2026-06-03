"""Exploratory driver for state-space discovery and latent market state analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from Model.features_data.state_space import build_pca_state_features, build_state_space_features


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "Data" / "datasets" / "raw" / "master_raw_dataset.csv"


def load_market_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"])
    if "Close" not in df.columns:
        raise ValueError("Expected Close column in raw dataset")
    return df.sort_values("Datetime").reset_index(drop=True)


def run_sample_analysis() -> pd.DataFrame:
    df = load_market_data()
    latent = build_pca_state_features(df, columns=["Close"], n_components=4)
    state_space = build_state_space_features(df)
    summary = pd.concat([df["Datetime"].reset_index(drop=True), latent, state_space], axis=1)
    return summary


def main() -> None:
    summary = run_sample_analysis()
    print("State-space feature sample")
    print(summary.head(10).to_string(index=False))
    output_dir = ROOT / "artifacts" / "data" / "latent_state"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "state_space_features.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved latent state sample to: {output_path}")


if __name__ == "__main__":
    main()
