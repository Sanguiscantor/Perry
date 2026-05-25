import numpy as np
import pandas as pd
from pathlib import Path


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


# ============================================
# DATETIME
# ============================================

df["Datetime"] = pd.to_datetime(
    df["Datetime"],
    format="mixed"
)


# ============================================
# BASIC FEATURES
# ============================================

# Candle midpoint

df["body_mid"] = (
    df["Open"] + df["Close"]
) / 2


# Momentum

df["momentum"] = (
    df["Close"] - df["Close"].shift(10)
)


# Return

df["return_1"] = (
    df["Close"].pct_change()
)


# Average volume

df["avg_volume"] = (
    df["Volume"]
    .rolling(window=20)
    .mean()
)


# Volume spike

df["volume_spike"] = (
    df["Volume"]
    / df["avg_volume"]
)


# Volatility

df["volatility"] = (
    df["return_1"]
    .rolling(window=10)
    .std()
)


# Directional volume

df["directional_volume"] = (
    (df["Close"] - df["Open"])
    * df["Volume"]
)


# ============================================
# EXTENDED FEATURES
# ============================================

# Sustained directional pressure

df["rolling_avg_momentum"] = (
    df["momentum"]
    .rolling(window=10)
    .mean()
)


# Volatility contraction / breakout setup

df["compression_width"] = (
    df["High"].rolling(window=20).max()
    - df["Low"].rolling(window=20).min()
)


# Nearby support zone

df["local_support"] = (
    df["Low"]
    .rolling(window=20)
    .min()
)


# Nearby resistance zone

df["local_resistance"] = (
    df["High"]
    .rolling(window=20)
    .max()
)


# Structural levels

df["structural_support"] = (
    df["Low"]
    .rolling(window=200)
    .min()
)

df["structural_resistance"] = (
    df["High"]
    .rolling(window=200)
    .max()
)


def rolling_slope(series, window=20):

    x = np.arange(window)

    def slope(values):
        return np.polyfit(x, values, 1)[0]

    return series.rolling(window).apply(slope, raw=True)


# Local level slopes

df["support_slope"] = rolling_slope(df["local_support"])
df["resistance_slope"] = rolling_slope(df["local_resistance"])


# Multi-scale compression

df["structural_compression"] = (
    df["structural_resistance"]
    - df["structural_support"]
)

df["compression_ratio"] = (
    df["compression_width"]
    / df["structural_compression"]
)

df["compression_tightness"] = (
    1 / df["compression_ratio"]
)


# Candle sequence strength

df["bearish_sequence_strength"] = (
    (df["Close"] < df["Open"])
    .astype(int)
    .rolling(window=5)
    .sum()
)

df["bullish_sequence_strength"] = (
    (df["Close"] > df["Open"])
    .astype(int)
    .rolling(window=5)
    .sum()
)


# Post-compression volatility expansion

df["volatility_expansion"] = (
    df["volatility"]
    / df["volatility"].rolling(window=20).mean()
)


# Distance to support / resistance

df["support_distance"] = (
    (df["Close"] - df["local_support"])
    / df["Close"]
)

df["resistance_distance"] = (
    (df["local_resistance"] - df["Close"])
    / df["Close"]
)


# Pivot mean and distance

df["pivot_mean"] = (
    df["Close"]
    .rolling(window=20)
    .mean()
)

df["pivot_distance"] = (
    (df["Close"] - df["pivot_mean"])
    / df["Close"]
)

df["equilibrium_distance"] = (
    (df["Close"] - df["pivot_mean"])
    / df["Close"]
)


# ============================================
# TARGETS
# ============================================

future_return_10 = (
    df["Close"].shift(-10)
    - df["Close"]
) / df["Close"]

future_return_10 = future_return_10.clip(-0.1, 0.1)


# Regression target: approximate future movement magnitude

df["target_p"] = future_return_10


# 3-way classification target

threshold = 0.003


def classify_target(x):

    if x > threshold:
        return 2

    elif x < -threshold:
        return 0

    else:
        return 1


df["target_d"] = future_return_10.apply(classify_target)


# ============================================
# CLEAN DATA
# ============================================

df = df.dropna()


# ============================================
# SAVE FEATURE DATASET
# ============================================

output_path = (
    Path(__file__).resolve().parent
    / "master_feature_dataset.csv"
)


df.to_csv(
    output_path,
    index=False
)


print("\nFeature dataset saved successfully.")
print(f"\nSaved to: {output_path}")
print(f"\nFinal rows: {len(df)}")
