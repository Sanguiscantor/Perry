"""State-space and latent market representation helpers for Perry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def delay_embedding(df: pd.DataFrame, columns: list[str], lag: int = 1, dim: int = 5) -> pd.DataFrame:
    """Build delay-embedded representations from price or return series."""
    if dim < 1 or lag < 1:
        raise ValueError("dim and lag must be positive integers")

    frames = []
    for i in range(dim):
        shifted = df[columns].shift(i * lag)
        shifted.columns = [f"{c}_lag_{i * lag}" for c in columns]
        frames.append(shifted)

    embedded = pd.concat(frames, axis=1)
    return embedded


def build_pca_state_features(
    df: pd.DataFrame,
    columns: list[str] = ["Close"],
    n_components: int = 4,
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """Extract compact latent state features from price-derived variables."""
    values = df[columns].copy()
    if values.isna().any(axis=None):
        values = values.fillna(method="ffill").fillna(method="bfill")

    returns = values.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = scaler or StandardScaler()
    X = scaler.fit_transform(returns)

    pca = PCA(n_components=min(n_components, X.shape[1]))
    components = pca.fit_transform(X)
    names = [f"pca_state_{i + 1}" for i in range(components.shape[1])]
    out = pd.DataFrame(components, columns=names, index=df.index)
    out["pca_explained_variance_ratio"] = np.sum(pca.explained_variance_ratio_)
    return out


def build_local_geometry_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute price-path curvature and local instability measures."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain Close column")

    prices = df["Close"].astype(float)
    velocity = prices.diff()
    acceleration = velocity.diff()
    curvature = acceleration.abs() / (velocity.abs() + 1e-9)
    angle = np.arctan2(acceleration, velocity.replace(0, 1e-9)).fillna(0.0)

    features = pd.DataFrame(
        {
            "geometry_velocity": velocity,
            "geometry_acceleration": acceleration,
            "geometry_curvature": curvature,
            "geometry_angle": angle,
        },
        index=df.index,
    )
    features["geometry_curvature_smooth"] = features["geometry_curvature"].rolling(window, min_periods=1).mean()
    features["geometry_angle_smooth"] = features["geometry_angle"].rolling(window, min_periods=1).mean()
    return features


def build_gmm_state_features(
    df: pd.DataFrame,
    n_states: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build soft regime features from a latent representation using Gaussian mixture."""
    embed_cols = [c for c in df.columns if c.startswith("pca_state_")]
    if not embed_cols:
        raise ValueError("DataFrame must contain PCA latent state columns")

    values = df[embed_cols].fillna(0.0)
    out = pd.DataFrame(index=df.index)
    if len(values) < 2:
        out["state_label"] = 0
        out["state_prob_0"] = 1.0
        out["state_entropy"] = 0.0
        return out

    try:
        gmm = GaussianMixture(n_components=min(n_states, len(values)), covariance_type="full", random_state=random_state)
        labels = gmm.fit_predict(values)
        probs = gmm.predict_proba(values)
    except ValueError:
        out["state_label"] = 0
        out["state_prob_0"] = 1.0
        out["state_entropy"] = 0.0
        return out

    out["state_label"] = labels
    for i in range(probs.shape[1]):
        out[f"state_prob_{i}"] = probs[:, i]
    out[f"state_entropy"] = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return out


def build_state_space_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact set of candidate state-space features for Perry."""
    pca = build_pca_state_features(df, columns=["Close"], n_components=3)
    geom = build_local_geometry_features(df, window=8)
    gmm = build_gmm_state_features(pca, n_states=3)
    return pd.concat([pca, geom, gmm], axis=1)
