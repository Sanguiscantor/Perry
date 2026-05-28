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
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor

from data_loader import prepare_features_targets
from model import build_model


def build_regressor():

    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.4,
        colsample_bytree=0.4,
        random_state=42,
    )


def run_pipeline():

    X, y_d, y_p, df = prepare_features_targets()

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

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_d_train,
    )

    classifier.fit(X_train, y_d_train, sample_weight=sample_weight)

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

    feature_importance_df =pd.DataFrame({
        "feature": X.columns,
        "importance": classifier.feature_importances_,
    })

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            by="importance",
            ascending=False,
        )
    )

    print("\nAll Feature Importances:")
    print(feature_importance_df.to_string(index=False))

    print("\nTop 20 Classifier Feature Importances:")
    print(feature_importance.head(20))

    regressor = build_regressor()

    regressor.fit(X_train, y_p_train)


    predicted_d = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    confidence = probabilities.max(axis=1)

    predicted_p = regressor.predict(X_test)
    train_predicted_p = regressor.predict(X_train)
    train_predicted_d = classifier.predict(X_train)

    accuracy = accuracy_score(
        y_d_test,
        predicted_d,
    )
    train_accuracy = accuracy_score(
        y_d_train,
        train_predicted_d,
    )

    print("\nTraining Accuracy:")
    print(train_accuracy)
    print("\nTraining Classification Report:")

    print(
        classification_report(
            y_d_train,
            train_predicted_d,
            zero_division=0,
        )
    )

    print("\nAccuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            y_d_test,
            predicted_d,
            zero_division=0,
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
    train_mae = mean_absolute_error(y_p_train, train_predicted_p)
    rmse = np.sqrt(mean_squared_error(y_p_test, predicted_p))
    train_rmse = np.sqrt(mean_squared_error(y_p_train, train_predicted_p))
    r2 = r2_score(y_p_test, predicted_p)
    train_r2 = r2_score(y_p_train, train_predicted_p)

    print("\nMAE:")
    print(mae)
    print("\nTraining MAE:")
    print(train_mae)

    print("\nRMSE:")
    print(rmse)
    print("\nTraining RMSE:")
    print(train_rmse)

    print("\nR2 Score:")
    print(r2)
    print("\nTraining R2 Score:")
    print(train_r2)

    results = pd.DataFrame({
        "actual_d": y_d_test.values,
        "predicted_d": predicted_d,
        "confidence": confidence,
        "actual_p": y_p_test.values,
        "predicted_p": predicted_p,
    })

    print("\nSample Predictions:")
    print(results.head(20))

    print("\nDiagnosis:")

    accuracy_gap = train_accuracy - accuracy
    r2_gap = train_r2 - r2

    if accuracy_gap > 0.20 or r2_gap > 0.50:
        print("Overfit")
    elif train_accuracy < 0.50 and accuracy < 0.50:
        print("Underfit")
    else:
        print("Seems healthy")
    
    models = {
        "classifier": classifier,
        "regressor": regressor,
    }

    return models, results
