from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_true, predictions):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    print("\nAccuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )