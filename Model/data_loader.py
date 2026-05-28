import pandas as pd
from pathlib import Path


FEATURE_COLUMNS = [

    # Price action
    "body_mid",
    "momentum",
    "return_1",

    # Volume
    "avg_volume",
    "volume_spike",
    "directional_volume",

    # Volatility / regime
    "volatility",
    "atr_14",
    "atr_ratio",
    "adx_14",
    "high_volatility_market",
    "low_volatility_market",
    "trending_market",
    "volatility_expansion",

    # Momentum structure
    "rolling_avg_momentum",
    "directional_persistence",

    # Compression / structure
    "compression_width",
    "swing_high",
    "swing_low",

    # Support / resistance
    "anchored_support",
    "anchored_resistance",
    "distance_to_anchored_support",
    "distance_to_anchored_resistance",
    "pivot_mean",
    "pivot_distance",

    # Wick structure
    "lower_wick",
    "upper_wick",
    "wick_ratio",

    # Sequence structure
    "bearish_sequence_strength",
    "bullish_sequence_strength",

    # Breakout logic
    "anchored_breakout",
    "anchored_breakdown",

    # Trend logic
    "trend_slope_50",
    "trend_slope_100",

    # Moving averages
    "ma_20",
    "ma_50",
    "ma_100",

    # MA cross structure
    "bullish_cross_20_50",
    "bearish_cross_20_50",
    "ma_cross_distance_20_50",
    "ma_cross_distance_50_100",

    # Structural continuation
    "rolling_higher_highs",
    "rolling_lower_lows",

    # Equilibrium structure
    "equilibrium_trend_distance",

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
