import numpy as np
import pandas as pd
from pathlib import Path

from market_structure import add_anchored_structure_features


PIVOT_WINDOW = 3
BREAK_CONFIRMATION_CANDLES = 2
MIN_PIVOT_DISTANCE = 10
MAX_ALLOWED_SLOPE = 0.0022
PIVOT_STRENGTH = 0.85
MINIMUM_TRENDLINE_SPAN = 28


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
    df["Close"] - df["Close"].shift(8)
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
# MARKET REGIME (ATR / ADX)
# ============================================
# These are context features to help the model distinguish:
# trending vs ranging regimes and high vs low volatility conditions.

# ATR (Average True Range) captures realized volatility using True Range (TR).
prev_close = df["Close"].shift(1)

tr = pd.concat(
    [
        (df["High"] - df["Low"]),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ],
    axis=1,
).max(axis=1)

df["atr_14"] = (
    tr
    .rolling(window=14)
    .mean()
)

# Normalize volatility relative to price.
df["atr_ratio"] = (
    df["atr_14"]
    / df["Close"]
)


# Simplified ADX (Average Directional Index) for trend strength.
up_move = df["High"].diff()
down_move = -df["Low"].diff()

plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

tr_14_sum = tr.rolling(window=14).sum()
plus_dm_14 = pd.Series(plus_dm, index=df.index).rolling(window=14).sum()
minus_dm_14 = pd.Series(minus_dm, index=df.index).rolling(window=14).sum()

plus_di = 100 * (plus_dm_14 / (tr_14_sum + 1e-9))
minus_di = 100 * (minus_dm_14 / (tr_14_sum + 1e-9))

dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)

df["adx_14"] = (
    dx
    .rolling(window=14)
    .mean()
)


# Regime detection logic:
# - ADX high implies trending, ADX low implies ranging
# - ATR ratio relative to its rolling median implies high/low volatility regime
df["trending_market"] = (df["adx_14"] > 25).astype(int)
df["ranging_market"] = (df["adx_14"] < 20).astype(int)

atr_ratio_median = df["atr_ratio"].rolling(window=100).median()
df["high_volatility_market"] = (df["atr_ratio"] > atr_ratio_median).astype(int)
df["low_volatility_market"] = (df["atr_ratio"] < atr_ratio_median).astype(int)


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


def calculate_rolling_trend_slope(series, window=50):

    x = np.arange(window)

    def slope(values):
        return np.polyfit(x, values, 1)[0]

    return series.rolling(window).apply(slope, raw=True)


def add_symbol_anchored_structure(symbol_df):
    if "symbol" not in symbol_df.columns:
        symbol_df = symbol_df.assign(symbol=symbol_df.name)

    return add_anchored_structure_features(
        symbol_df,
        pivot_window=PIVOT_WINDOW,
        confirmation_candles=BREAK_CONFIRMATION_CANDLES,
        min_pivot_distance=MIN_PIVOT_DISTANCE,
        max_allowed_slope=MAX_ALLOWED_SLOPE,
        pivot_strength=PIVOT_STRENGTH,
        minimum_trendline_span=MINIMUM_TRENDLINE_SPAN,
    )


# Wick rejection features

df["lower_wick"] = (
    np.minimum(df["Open"], df["Close"])
    - df["Low"]
)

df["upper_wick"] = (
    df["High"]
    - np.maximum(df["Open"], df["Close"])
)

df["wick_ratio"] = (
    df["lower_wick"]
    / (df["upper_wick"] + 1e-9)
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

# Anchored swing-point market structure

df = (
    df
    .sort_values(["symbol", "Datetime"])
    .groupby("symbol", group_keys=False)
    .apply(add_symbol_anchored_structure)
    .reset_index(drop=True)
)


# Failed breakdown (intracandle breach with close recovery)

df["failed_breakdown"] = (
    (df["Low"] < df["anchored_support"])
    & (df["Close"] > df["anchored_support"])
).astype(int)


# ============================================
# MACRO TREND & DIRECTIONAL CONTEXT
# ============================================

# Trend slopes

df["trend_slope_50"] = (
    df
    .groupby("symbol")["Close"]
    .transform(lambda close: calculate_rolling_trend_slope(close, window=50))
)
df["trend_slope_100"] = (
    df
    .groupby("symbol")["Close"]
    .transform(lambda close: calculate_rolling_trend_slope(close, window=100))
)


# Moving averages

df["ma_20"] = (
    df["Close"]
    .rolling(window=20)
    .mean()
)

df["ma_50"] = (
    df["Close"]
    .rolling(window=50)
    .mean()
)

df["ma_100"] = (
    df["Close"]
    .rolling(window=100)
    .mean()
)


# Moving average crossovers

bullish_20_50_prev = df["ma_20"].shift(1) <= df["ma_50"].shift(1)
bullish_20_50_now = df["ma_20"] > df["ma_50"]
df["bullish_cross_20_50"] = (
    bullish_20_50_prev
    & bullish_20_50_now
).astype(int)

bearish_20_50_prev = df["ma_20"].shift(1) >= df["ma_50"].shift(1)
bearish_20_50_now = df["ma_20"] < df["ma_50"]
df["bearish_cross_20_50"] = (
    bearish_20_50_prev
    & bearish_20_50_now
).astype(int)

bullish_50_100_prev = df["ma_50"].shift(1) <= df["ma_100"].shift(1)
bullish_50_100_now = df["ma_50"] > df["ma_100"]
df["bullish_cross_50_100"] = (
    bullish_50_100_prev
    & bullish_50_100_now
).astype(int)

bearish_50_100_prev = df["ma_50"].shift(1) >= df["ma_100"].shift(1)
bearish_50_100_now = df["ma_50"] < df["ma_100"]
df["bearish_cross_50_100"] = (
    bearish_50_100_prev
    & bearish_50_100_now
).astype(int)


# Moving average crossover strength

df["ma_cross_distance_20_50"] = (
    (df["ma_20"] - df["ma_50"])
    / df["Close"]
)

df["ma_cross_distance_50_100"] = (
    (df["ma_50"] - df["ma_100"])
    / df["Close"]
)


# Higher high / lower low structure

higher_high = df["High"] > df["High"].shift(1)
lower_low = df["Low"] < df["Low"].shift(1)

df["rolling_higher_highs"] = (
    higher_high
    .astype(int)
    .rolling(window=20)
    .sum()
)

df["rolling_lower_lows"] = (
    lower_low
    .astype(int)
    .rolling(window=20)
    .sum()
)


# Directional persistence

df["directional_persistence"] = (
    np.sign(df["return_1"])
    .rolling(window=20)
    .mean()
)


# Equilibrium trend interaction

df["equilibrium_trend_distance"] = (
    df["pivot_distance"]
    * df["directional_persistence"]
)


# ============================================
# TARGETS
# ============================================

# Sustained move targets from next 12 candles:
# use future max High and future min Low windows (excluding current candle).
horizon = 12
threshold = 0.0045

future_max_high = (
    df["High"]
    .shift(-1)
    .rolling(window=horizon, min_periods=horizon)
    .max()
    .shift(-(horizon - 1))
)

future_min_low = (
    df["Low"]
    .shift(-1)
    .rolling(window=horizon, min_periods=horizon)
    .min()
    .shift(-(horizon - 1))
)

future_max_return = (
    future_max_high
    - df["Close"]
) / df["Close"]

future_min_return = (
    future_min_low
    - df["Close"]
) / df["Close"]


# Dominant directional excursion labeling (opportunity-based).
# Instead of treating "both sides hit" as sideways, we select the direction
# with the stronger tradeable excursion over the next 12 candles.
bearish_magnitude = future_min_return.abs()

bullish = (
    (future_max_return > bearish_magnitude)
    & (future_max_return > threshold)
)

bearish = (
    (bearish_magnitude > future_max_return)
    & (bearish_magnitude > threshold)
)

df["target_d"] = np.select(
    [bullish, bearish],
    [2, 0],
    default=1,
)


# Regression target: signed dominant excursion (0 for sideways).
df["target_p"] = np.select(
    [bullish, bearish],
    [future_max_return, future_min_return],
    default=0.0,
)


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
