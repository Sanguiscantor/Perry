from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd


def load_chronological_dataset(
    path: str,
    datetime_col: str = "Datetime",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load the dataset and preserve strict chronological order."""
    if parse_dates:
        df = pd.read_csv(path, parse_dates=[datetime_col])
    else:
        df = pd.read_csv(path)
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    df = df.sort_values(by=datetime_col, kind="stable").reset_index(drop=True)
    return df


def prepare_features_targets(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    target_col: str = "target",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Select feature columns and target column for model training and inference."""
    df = df.dropna(subset=[target_col]).copy()

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in {target_col, "Datetime"}]

    X = df.loc[:, feature_cols].astype(float)
    y = df[target_col].astype(int)
    return X, y
