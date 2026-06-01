import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import build_model


META_COLUMNS = ["window_end_idx", "window_end_time", "target_d"]


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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    classifier = build_model()
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
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

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 50 feature importances:")
    for _, row in importance.head(50).iterrows():
        print(f"{row['importance']:.6f}  {row['feature']}")


if __name__ == "__main__":
    main()
