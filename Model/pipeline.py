import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from data_loader import prepare_features_targets
from model import build_model


def build_regressor():

    return XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def run_pipeline():

    X, y_d, y_p, df = prepare_features_targets()

    print("\nSplitting dataset...")

    X_train, X_test, y_d_train, y_d_test = train_test_split(
        X,
        y_d,
        test_size=0.2,
        shuffle=False,
    )

    _, _, y_p_train, y_p_test = train_test_split(
        X,
        y_p,
        test_size=0.2,
        shuffle=False,
    )

    classifier = build_model()

    classifier.fit(X_train, y_d_train)

    regressor = build_regressor()

    regressor.fit(X_train, y_p_train)

    print("\nGenerating predictions...")

    predicted_d = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    confidence = probabilities.max(axis=1)

    predicted_p = regressor.predict(X_test)

    accuracy = accuracy_score(
        y_d_test,
        predicted_d,
    )

    print("\nAccuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            y_d_test,
            predicted_d,
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_d_test,
            predicted_d,
        )
    )

    mae = mean_absolute_error(y_p_test, predicted_p)
    rmse = np.sqrt(mean_squared_error(y_p_test, predicted_p))
    r2 = r2_score(y_p_test, predicted_p)

    print("\nMAE:")
    print(mae)

    print("\nRMSE:")
    print(rmse)

    print("\nR2 Score:")
    print(r2)

    results = pd.DataFrame({
        "actual_d": y_d_test.values,
        "predicted_d": predicted_d,
        "confidence": confidence,
        "actual_p": y_p_test.values,
        "predicted_p": predicted_p,
    })

    print("\nSample Predictions:")
    print(results.head(20))

    models = {
        "classifier": classifier,
        "regressor": regressor,
    }

    return models, results
