"""Phase 8: Information analysis — measure feature importance, redundancy, uniqueness."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler


def compute_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compute mutual information between features and target."""
    mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)
    return pd.DataFrame({
        "feature": X.columns,
        "mutual_information": mi_scores,
    }).sort_values("mutual_information", ascending=False)


def compute_f_statistic(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compute F-statistic (ANOVA) between features and target."""
    f_scores, p_values = f_classif(X.fillna(0), y)
    return pd.DataFrame({
        "feature": X.columns,
        "f_statistic": f_scores,
        "p_value": p_values,
    }).sort_values("f_statistic", ascending=False)


def compute_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model: Optional[RandomForestClassifier] = None,
) -> pd.DataFrame:
    """Compute permutation-based feature importance."""
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X.fillna(0), y)

    importance = model.feature_importances_
    return pd.DataFrame({
        "feature": X.columns,
        "importance": importance,
    }).sort_values("importance", ascending=False)


def compute_feature_redundancy(X: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise feature correlation to identify redundancy."""
    corr = X.fillna(0).corr().abs()
    # Upper triangle to avoid duplicates
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    redundancy = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if mask[i, j]:
                redundancy.append({
                    "feature_1": corr.columns[i],
                    "feature_2": corr.columns[j],
                    "correlation": corr.iloc[i, j],
                })
    return pd.DataFrame(redundancy).sort_values("correlation", ascending=False)


def compute_incremental_contribution(
    X: pd.DataFrame,
    y: pd.Series,
    baseline_features: list[str] = None,
) -> pd.DataFrame:
    """Measure incremental importance: does adding this feature improve baseline?"""
    if baseline_features is None:
        baseline_features = X.columns.tolist()

    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    baseline_model.fit(X[baseline_features].fillna(0), y)
    baseline_score = baseline_model.score(X[baseline_features].fillna(0), y)

    incremental = []
    for col in X.columns:
        if col in baseline_features:
            continue

        test_features = baseline_features + [col]
        test_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        test_model.fit(X[test_features].fillna(0), y)
        test_score = test_model.score(X[test_features].fillna(0), y)

        incremental.append({
            "feature": col,
            "incremental_improvement": test_score - baseline_score,
            "new_score": test_score,
        })

    return pd.DataFrame(incremental).sort_values("incremental_improvement", ascending=False)


def compute_feature_category_importance(
    X: pd.DataFrame,
    y: pd.Series,
    category_mapping: dict[str, str],
) -> pd.DataFrame:
    """Group features by category and compute category-level importance."""
    X_copy = X.fillna(0)
    
    category_features = {}
    for feature, category in category_mapping.items():
        if feature in X_copy.columns:
            if category not in category_features:
                category_features[category] = []
            category_features[category].append(feature)

    category_importance = []
    for category, features in category_features.items():
        X_category = X_copy[features]
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_category, y)
        score = model.score(X_category, y)
        category_importance.append({
            "category": category,
            "accuracy": score,
            "n_features": len(features),
        })

    return pd.DataFrame(category_importance).sort_values("accuracy", ascending=False)


def analyze_information_content(
    X: pd.DataFrame,
    y: pd.Series,
    category_mapping: dict[str, str] = None,
) -> dict:
    """Run comprehensive information analysis."""
    results = {
        "mutual_information": compute_mutual_information(X, y),
        "f_statistic": compute_f_statistic(X, y),
        "feature_importance": compute_feature_importance(X, y),
        "redundancy": compute_feature_redundancy(X),
    }

    if category_mapping:
        results["category_importance"] = compute_feature_category_importance(X, y, category_mapping)

    return results


def summarize_analysis(analysis: dict) -> dict:
    """Extract key findings from analysis."""
    mi = analysis["mutual_information"]
    importance = analysis["feature_importance"]
    redundancy = analysis["redundancy"]

    top_mi = mi.head(10)
    top_importance = importance.head(10)

    # Find highly redundant features (correlation > 0.8)
    high_redundancy = redundancy[redundancy["correlation"] > 0.8]

    summary = {
        "top_10_mutual_information": top_mi,
        "top_10_feature_importance": top_importance,
        "high_redundancy_pairs": high_redundancy,
        "total_features": len(mi),
        "mean_mutual_information": mi["mutual_information"].mean(),
        "max_correlation": redundancy["correlation"].max() if len(redundancy) > 0 else 0.0,
    }

    return summary
