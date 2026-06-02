import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import build_binary_model


META_COLUMNS = ["window_end_idx", "window_end_time", "target_d"]
TRADE_PROBABILITY_THRESHOLD = 0.65


def predict_hierarchical(
    trade_classifier,
    direction_classifier,
    X,
    threshold=TRADE_PROBABILITY_THRESHOLD,
):

    trade_probabilities = trade_classifier.predict_proba(X)
    predicted_trade = (
        trade_probabilities[:, 1] > threshold
    ).astype(int)

    predictions = np.ones(len(X), dtype=int)
    trade_indices = np.where(predicted_trade == 1)[0]

    if len(trade_indices) > 0:
        direction_predictions = direction_classifier.predict(
            X.iloc[trade_indices]
        ).astype(int)

        predictions[trade_indices] = np.where(
            direction_predictions == 1,
            2,
            0,
        )

    return predictions


def main():

    data_path = (
        Path(__file__).resolve().parent
        / "tsfresh_selected_features.csv"
    )

    df = pd.read_csv(data_path)
    print(f"\nLoaded windows: {len(df)}")

    feature_cols = [c for c in df.columns if c not in META_COLUMNS]

    X = df[feature_cols]
    y = df["target_d"]

    print("\nTarget Distribution:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    y_trade_train = (y_train != 1).astype(int)
    trade_classifier = build_binary_model()
    trade_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_trade_train,
    )

    trade_classifier.fit(
        X_train,
        y_trade_train,
        sample_weight=trade_weights,
    )

    trade_mask = y_trade_train == 1
    direction_y_train = (y_train[trade_mask] == 2).astype(int)
    direction_classifier = build_binary_model()
    direction_weights = compute_sample_weight(
        class_weight="balanced",
        y=direction_y_train,
    )

    direction_classifier.fit(
        X_train[trade_mask],
        direction_y_train,
        sample_weight=direction_weights,
    )

    predictions = predict_hierarchical(
        trade_classifier,
        direction_classifier,
        X_test,
    )
    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": trade_classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 50 trade classifier feature importances:")
    for _, row in importance.head(50).iterrows():
        print(f"{row['importance']:.6f}  {row['feature']}")


if __name__ == "__main__":
    main()
