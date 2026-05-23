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
# 3-WAY TARGET
# ============================================

future_return = (
    df["Close"].shift(-10)
    - df["Close"]
) / df["Close"]


threshold = 0.003


def classify_target(x):

    if x > threshold:
        return 2

    elif x < -threshold:
        return 0

    else:
        return 1


df["target"] = future_return.apply(classify_target)


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