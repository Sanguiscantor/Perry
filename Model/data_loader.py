import pandas as pd
from pathlib import Path


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

TARGET_COLUMN_D = "target_d"
TARGET_COLUMN_P = "target_p"


def load_dataset():

    dataset_path = (
        Path(__file__).resolve().parent
        / "features_data"
        / "master_feature_dataset.csv"
    )

    df = pd.read_csv(dataset_path)

    return df


def prepare_features_targets():

    df = load_dataset()

    print("\nAvailable Columns:")
    print(df.columns.tolist())

    print("\nTarget Class Distribution:")
    print(df["target_d"].value_counts())
    

    df = df.dropna(subset=[TARGET_COLUMN_D, TARGET_COLUMN_P])

    X = df[FEATURE_COLUMNS]

    y_d = df[TARGET_COLUMN_D]
    y_p = df[TARGET_COLUMN_P]

    return X, y_d, y_p, df
