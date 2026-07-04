"""Market state discovery: multi-scale structure, geometry, confluence, and local dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ============================================================================
# PHASE 1: MULTI-SCALE MARKET STRUCTURE
# ============================================================================


def detect_moving_averages(df: pd.DataFrame, windows: list[int] = [20, 50, 100, 200]) -> pd.DataFrame:
    """Compute simple moving averages at multiple horizons."""
    out = pd.DataFrame(index=df.index)
    for w in windows:
        out[f"sma_{w}"] = df["Close"].rolling(w, min_periods=1).mean()
    return out


def detect_support_resistance_simple(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Detect local support and resistance levels."""
    rolling_high = df["High"].rolling(window, center=False).max()
    rolling_low = df["Low"].rolling(window, center=False).min()

    out = pd.DataFrame(index=df.index)
    out[f"support_{window}"] = rolling_low
    out[f"resistance_{window}"] = rolling_high
    out[f"support_distance_{window}"] = (df["Close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
    out[f"resistance_distance_{window}"] = (rolling_high - df["Close"]) / (rolling_high - rolling_low + 1e-9)
    return out


def detect_multi_scale_structure(df: pd.DataFrame, scales: list[int] = [20, 50, 100, 200]) -> pd.DataFrame:
    """Build multi-scale structure features: support, resistance, compression."""
    frames = []
    for scale in scales:
        rolling_high = df["High"].rolling(scale, min_periods=1).max()
        rolling_low = df["Low"].rolling(scale, min_periods=1).min()
        mid = (rolling_high + rolling_low) / 2

        s = pd.DataFrame(index=df.index)
        s[f"support_{scale}"] = rolling_low
        s[f"resistance_{scale}"] = rolling_high
        s[f"mid_{scale}"] = mid
        s[f"compression_width_{scale}"] = rolling_high - rolling_low
        s[f"compression_width_pct_{scale}"] = (rolling_high - rolling_low) / df["Close"]

        # Distance to anchored levels
        s[f"distance_to_support_{scale}"] = df["Close"] - rolling_low
        s[f"distance_to_resistance_{scale}"] = rolling_high - df["Close"]

        # Percentile location within structure
        s[f"structure_position_{scale}"] = (
            (df["Close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
        )

        frames.append(s)

    return pd.concat(frames, axis=1)


# ============================================================================
# PHASE 2: MARKET POSITION
# ============================================================================


def detect_moving_average_crosses(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Distance from moving average crosses."""
    ma_fast = df["Close"].rolling(fast, min_periods=1).mean()
    ma_slow = df["Close"].rolling(slow, min_periods=1).mean()

    out = pd.DataFrame(index=df.index)
    out[f"ma_{fast}_level"] = ma_fast
    out[f"ma_{slow}_level"] = ma_slow
    out[f"ma_cross_distance_{fast}_{slow}"] = ma_fast - ma_slow
    out[f"ma_cross_distance_pct_{fast}_{slow}"] = (ma_fast - ma_slow) / (ma_slow + 1e-9)
    out[f"ma_above_below_{fast}_{slow}"] = np.sign(ma_fast - ma_slow)
    return out


def detect_equilibrium_distance(df: pd.DataFrame, scales: list[int] = [20, 50, 100, 200]) -> pd.DataFrame:
    """Distance to equilibrium / mid-point of structure."""
    frames = []
    for scale in scales:
        rolling_high = df["High"].rolling(scale, min_periods=1).max()
        rolling_low = df["Low"].rolling(scale, min_periods=1).min()
        equilibrium = (rolling_high + rolling_low) / 2

        s = pd.DataFrame(index=df.index)
        s[f"equilibrium_{scale}"] = equilibrium
        s[f"equilibrium_distance_{scale}"] = df["Close"] - equilibrium
        s[f"equilibrium_distance_pct_{scale}"] = (df["Close"] - equilibrium) / (equilibrium + 1e-9)
        frames.append(s)

    return pd.concat(frames, axis=1)


def market_position_composite(df: pd.DataFrame, scales: list[int] = [20, 50, 100, 200]) -> pd.DataFrame:
    """Composite market position: where is price relative to structure?"""
    frames = []

    for scale in scales:
        rolling_high = df["High"].rolling(scale, min_periods=1).max()
        rolling_low = df["Low"].rolling(scale, min_periods=1).min()
        mid = (rolling_high + rolling_low) / 2

        s = pd.DataFrame(index=df.index)
        # Position score: 0 at support, 1 at resistance, 0.5 at equilibrium
        s[f"position_score_{scale}"] = (df["Close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
        # Proximity to extremes
        s[f"proximity_to_support_{scale}"] = (df["Close"] - rolling_low) / (mid - rolling_low + 1e-9)
        s[f"proximity_to_resistance_{scale}"] = (rolling_high - df["Close"]) / (rolling_high - mid + 1e-9)
        frames.append(s)

    return pd.concat(frames, axis=1)


# ============================================================================
# PHASE 3: CONFLUENCE
# ============================================================================


def detect_confluence_density(df: pd.DataFrame, scales: list[int] = [20, 50, 100, 200]) -> pd.DataFrame:
    """Confluence: how many independent structures agree at current price?"""
    frames = []

    for scale in scales:
        rolling_high = df["High"].rolling(scale, min_periods=1).max()
        rolling_low = df["Low"].rolling(scale, min_periods=1).min()

        # Count how many levels are "near" the current price
        tolerance = 0.005  # 0.5% tolerance
        tolerance_value = df["Close"] * tolerance

        s = pd.DataFrame(index=df.index)
        s[f"confluence_density_{scale}"] = 0.0
        s[f"confluence_to_support_{scale}"] = (df["Close"] - rolling_low).abs() <= tolerance_value
        s[f"confluence_to_resistance_{scale}"] = (rolling_high - df["Close"]).abs() <= tolerance_value
        frames.append(s)

    confluence = pd.concat(frames, axis=1)
    # Sum confluence signals
    confluence_cols = [c for c in confluence.columns if "confluence_to_" in c]
    confluence["confluence_agreement_count"] = confluence[confluence_cols].sum(axis=1)
    confluence["confluence_agreement_normalized"] = (
        confluence["confluence_agreement_count"] / len(confluence_cols)
    )

    return confluence


def detect_compression_zones(df: pd.DataFrame, scales: list[int] = [20, 50, 100]) -> pd.DataFrame:
    """Identify compression: when width is below median."""
    frames = []
    for scale in scales:
        rolling_high = df["High"].rolling(scale, min_periods=1).max()
        rolling_low = df["Low"].rolling(scale, min_periods=1).min()
        width = rolling_high - rolling_low
        median_width = width.rolling(scale * 2, min_periods=1).median()

        s = pd.DataFrame(index=df.index)
        s[f"is_compression_{scale}"] = (width < median_width).astype(int)
        s[f"compression_ratio_{scale}"] = width / (median_width + 1e-9)
        s[f"compression_energy_{scale}"] = (median_width - width) / median_width
        frames.append(s)

    return pd.concat(frames, axis=1)


# ============================================================================
# PHASE 4: MARKET GEOMETRY
# ============================================================================


def detect_local_geometry(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Local price-path geometry: curvature, acceleration, slope."""
    prices = df["Close"].astype(float)
    returns = prices.pct_change().fillna(0.0)

    # First and second derivatives
    velocity = returns  # instantaneous change
    acceleration = velocity.diff().fillna(0.0)

    # Curvature: how much the path is bending
    curvature = acceleration.abs() / (velocity.abs() + 1e-9)

    # Angle: direction of movement relative to recent history
    angle = np.arctan2(acceleration, velocity.replace(0, 1e-9)).fillna(0.0)

    out = pd.DataFrame(index=df.index)
    out["geometry_velocity"] = velocity
    out["geometry_acceleration"] = acceleration
    out["geometry_curvature"] = curvature
    out["geometry_angle"] = angle
    out["geometry_curvature_smooth"] = curvature.rolling(window, min_periods=1).mean()
    out["geometry_angle_smooth"] = angle.rolling(window, min_periods=1).mean()

    return out


def detect_trend_geometry(df: pd.DataFrame, scales: list[int] = [20, 50, 100]) -> pd.DataFrame:
    """Trend geometry: persistence, direction, strength."""
    frames = []
    for scale in scales:
        # Linear regression slope
        prices = df["Close"].values
        slope_vals = []
        for i in range(len(prices)):
            if i < scale:
                x = np.arange(i + 1)
                y = prices[: i + 1]
            else:
                x = np.arange(scale)
                y = prices[i - scale + 1 : i + 1]
            if len(x) > 1:
                coeffs = np.polyfit(x, y, 1)
                slope_vals.append(coeffs[0])
            else:
                slope_vals.append(0.0)

        slope = pd.Series(np.array(slope_vals), index=df.index)

        s = pd.DataFrame(index=df.index)
        s[f"trend_slope_{scale}"] = slope
        s[f"trend_slope_sign_{scale}"] = np.sign(slope)
        s[f"trend_persistence_{scale}"] = (
            (np.sign(slope) == np.sign(slope.shift(1)))
            .rolling(scale // 2, min_periods=1)
            .mean()
        )
        frames.append(s)

    return pd.concat(frames, axis=1)


def detect_structural_acceleration(df: pd.DataFrame, scales: list[int] = [20, 50, 100]) -> pd.DataFrame:
    """Structural acceleration: is the trend speeding up or slowing down?"""
    frames = []
    for scale in scales:
        prices = df["Close"].astype(float).values

        # Trend acceleration
        acceleration_vals = []
        for i in range(len(prices)):
            if i < scale:
                x = np.arange(i + 1)
                y = prices[: i + 1]
            else:
                x = np.arange(scale)
                y = prices[i - scale + 1 : i + 1]
            if len(x) > 2:
                coeffs = np.polyfit(x, y, 2)
                acceleration_vals.append(coeffs[0])  # quadratic term
            else:
                acceleration_vals.append(0.0)

        acceleration = np.array(acceleration_vals)

        s = pd.DataFrame(index=df.index)
        s[f"structural_acceleration_{scale}"] = acceleration
        s[f"structural_acceleration_sign_{scale}"] = np.sign(acceleration)
        frames.append(s)

    return pd.concat(frames, axis=1)


def detect_pressure_metrics(df: pd.DataFrame, scales: list[int] = [20, 50, 100]) -> pd.DataFrame:
    """Market pressure: breakout pressure, support/resistance density."""
    frames = []
    for scale in scales:
        rolling_high = df["High"].rolling(scale, min_periods=1).max()
        rolling_low = df["Low"].rolling(scale, min_periods=1).min()

        # How close is price to recent highs/lows? (pressure)
        high_distance = (df["High"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
        low_distance = (df["Low"] - rolling_low) / (rolling_high - rolling_low + 1e-9)

        s = pd.DataFrame(index=df.index)
        s[f"breakout_pressure_up_{scale}"] = high_distance
        s[f"breakout_pressure_down_{scale}"] = 1.0 - low_distance
        s[f"recent_breakout_attempt_{scale}"] = (high_distance > 0.9).astype(int) | (low_distance < 0.1).astype(int)
        frames.append(s)

    return pd.concat(frames, axis=1)


# ============================================================================
# PHASE 5: STATE SPACE INTEGRATION
# ============================================================================


def build_latent_state_features(
    df: pd.DataFrame,
    columns: list[str] = ["Close"],
    n_components: int = 3,
) -> pd.DataFrame:
    """PCA-based latent state extraction."""
    returns = df[columns].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(returns)

    pca = PCA(n_components=min(n_components, X.shape[1]))
    components = pca.fit_transform(X)

    out = pd.DataFrame(index=df.index)
    for i in range(components.shape[1]):
        out[f"latent_state_{i + 1}"] = components[:, i]
    out["latent_variance_explained"] = np.sum(pca.explained_variance_ratio_)
    return out


def build_gmm_soft_states(
    df: pd.DataFrame,
    n_states: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Soft Gaussian mixture state memberships."""
    state_cols = [c for c in df.columns if c.startswith("latent_state_")]
    if not state_cols:
        return pd.DataFrame(index=df.index)

    X = df[state_cols].fillna(0.0).values
    out = pd.DataFrame(index=df.index)
    if len(X) < 2:
        out["state_label"] = 0
        out["state_prob_0"] = 1.0
        out["state_entropy"] = 0.0
        return out

    try:
        gmm = GaussianMixture(n_components=min(n_states, len(X)), covariance_type="full", random_state=random_state)
        labels = gmm.fit_predict(X)
        probs = gmm.predict_proba(X)
    except ValueError:
        out["state_label"] = 0
        out["state_prob_0"] = 1.0
        out["state_entropy"] = 0.0
        return out

    out["state_label"] = labels
    for i in range(probs.shape[1]):
        out[f"state_prob_{i}"] = probs[:, i]
    out["state_entropy"] = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return out


# ============================================================================
# PHASE 6: LOCAL DYNAMICS
# ============================================================================


def detect_local_regime_behavior(df: pd.DataFrame, state_col: str = "state_label") -> pd.DataFrame:
    """For each regime, compute local statistics."""
    if state_col not in df.columns:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=df.index)
    states = df[state_col].unique()

    for state in states:
        mask = df[state_col] == state
        local_returns = df.loc[mask, "Close"].pct_change()

        out[f"regime_{state}_return_mean"] = 0.0
        out[f"regime_{state}_return_vol"] = 0.0
        out[f"regime_{state}_return_skew"] = 0.0

        if mask.sum() > 0:
            out.loc[mask, f"regime_{state}_return_mean"] = local_returns.mean()
            out.loc[mask, f"regime_{state}_return_vol"] = local_returns.std()
            out.loc[mask, f"regime_{state}_return_skew"] = local_returns.skew()

    return out


def detect_compression_behavior(df: pd.DataFrame, compression_col: str = "is_compression_50") -> pd.DataFrame:
    """Behavior differences in compression vs expansion."""
    if compression_col not in df.columns:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=df.index)

    # Compression statistics
    compressed = df[compression_col] == 1
    expanded = df[compression_col] == 0

    compression_returns = df.loc[compressed, "Close"].pct_change()
    expansion_returns = df.loc[expanded, "Close"].pct_change()

    out["compression_return_mean"] = 0.0
    out["expansion_return_mean"] = 0.0
    out["compression_return_vol"] = 0.0
    out["expansion_return_vol"] = 0.0

    out.loc[compressed, "compression_return_mean"] = compression_returns.mean()
    out.loc[expanded, "expansion_return_mean"] = expansion_returns.mean()
    out.loc[compressed, "compression_return_vol"] = compression_returns.std()
    out.loc[expanded, "expansion_return_vol"] = expansion_returns.std()

    return out


# ============================================================================
# MASTER STATE DISCOVERY BUILDER
# ============================================================================


def build_market_state_features(
    df: pd.DataFrame,
    scales: list[int] = [20, 50, 100, 200],
    include_latent: bool = True,
) -> pd.DataFrame:
    """Build comprehensive multi-phase market state features."""
    frames = [df[["Datetime", "Close", "High", "Low", "Volume"]].copy()]

    # Phase 1: Multi-scale structure
    frames.append(detect_multi_scale_structure(df, scales=scales))
    frames.append(detect_moving_averages(df, windows=scales))

    # Phase 2: Market position
    for fast, slow in [(20, 50), (50, 100)]:
        frames.append(detect_moving_average_crosses(df, fast=fast, slow=slow))
    frames.append(detect_equilibrium_distance(df, scales=scales))
    frames.append(market_position_composite(df, scales=scales))

    # Phase 3: Confluence
    frames.append(detect_confluence_density(df, scales=scales))
    frames.append(detect_compression_zones(df, scales=scales))

    # Phase 4: Geometry
    frames.append(detect_local_geometry(df, window=8))
    frames.append(detect_trend_geometry(df, scales=scales))
    frames.append(detect_structural_acceleration(df, scales=scales))
    frames.append(detect_pressure_metrics(df, scales=scales))

    # Phase 5: Latent states
    if include_latent:
        latent = build_latent_state_features(df, n_components=3)
        frames.append(latent)
        gmm = build_gmm_soft_states(latent, n_states=3)
        frames.append(gmm)

        # Phase 6: Local dynamics
        frames.append(detect_local_regime_behavior(latent))
        frames.append(detect_compression_behavior(df))

    return pd.concat(frames, axis=1)
