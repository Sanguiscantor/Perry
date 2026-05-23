import pandas as pd
from pathlib import Path


FEATURE_COLUMNS = [
    "body_mid",
    "momentum",
    "return_1",
    "avg_volume",
    "volume_spike",
    "volatility",
    "directional_volume"
]

TARGET_COLUMN = "target"


def load_dataset():

    dataset_path = (
        Path(__file__).resolve().parent
        / "features_data"
        / "master_feature_dataset.csv"
    )

    print(f"\nLoading dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    return df


def prepare_features_targets():

    df = load_dataset()

    print("\nAvailable Columns:")
    print(df.columns.tolist())

    df = df.dropna(subset=[TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]

    return X, y, df