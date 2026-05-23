import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from data_loader import prepare_features_targets
from model import build_model


def run_pipeline():

    print("\nPreparing features...")

    X, y, df = prepare_features_targets()

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    print("\nBuilding model...")

    model = build_model()

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("\nGenerating predictions...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    confidence = probabilities.max(axis=1)

    print("\nEvaluating model...")

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nAccuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    results = pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions,
        "confidence": confidence
    })

    print("\nSample Predictions:")
    print(results.head(20))

    return model, results