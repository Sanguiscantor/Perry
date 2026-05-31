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
from model import build_binary_model


TRADE_PROBABILITY_THRESHOLD = 0.65


def build_regressor():

    return XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.5,
        colsample_bytree=0.5,
        random_state=42,
    )


def _predict_hierarchical(
    trade_classifier,
    direction_classifier,
    X,
    threshold=TRADE_PROBABILITY_THRESHOLD,
):

    trade_probabilities = trade_classifier.predict_proba(X)
    predicted_trade = (
        trade_probabilities[:, 1] > threshold
    ).astype(int)

    predicted_d = np.ones(len(X), dtype=int)

    trade_prediction_mask = predicted_trade == 1
    direction_predictions = np.array([], dtype=int)

    if trade_prediction_mask.sum() > 0:
        trade_indices = np.where(trade_prediction_mask)[0]

        direction_predictions = direction_classifier.predict(
            X.iloc[trade_indices]
        ).astype(int)

        predicted_d[trade_indices] = np.where(
            direction_predictions == 1,
            2,
            0,
        )

    confidence = trade_probabilities.max(axis=1)

    return predicted_d, predicted_trade, direction_predictions, confidence


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

    y_trade_train = (y_d_train != 1).astype(int)
    y_trade_test = (y_d_test != 1).astype(int)

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

    direction_classifier = build_binary_model()

    direction_y_train = (y_d_train[trade_mask] == 2).astype(int)

    direction_weights = compute_sample_weight(
        class_weight="balanced",
        y=direction_y_train,
    )

    direction_classifier.fit(
        X_train[trade_mask],
        direction_y_train,
        sample_weight=direction_weights,
    )

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": trade_classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

    feature_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": trade_classifier.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    print("\nAll Feature Importances:")
    print(feature_importance_df.to_string(index=False))

    print("\nTop 20 Classifier Feature Importances:")
    print(feature_importance.head(20))

    regressor = build_regressor()

    regressor.fit(
        X_train[trade_mask],
        y_p_train[trade_mask],
    )

    predicted_d, predicted_trade, direction_predictions, confidence = (
        _predict_hierarchical(
            trade_classifier,
            direction_classifier,
            X_test,
        )
    )

    predicted_p = regressor.predict(X_test)

    train_predicted_d, train_predicted_trade, train_direction_predictions, _ = (
        _predict_hierarchical(
            trade_classifier,
            direction_classifier,
            X_train,
        )
    )

    train_predicted_p = regressor.predict(X_train)

    accuracy = accuracy_score(y_d_test, predicted_d)
    train_accuracy = accuracy_score(y_d_train, train_predicted_d)

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
    print(confusion_matrix(y_d_test, predicted_d))

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

    print("\nTrade Predictions:")
    print(pd.Series(predicted_trade).value_counts())
    print("\nActual Trade Labels:")
    print(pd.Series(y_trade_test).value_counts())

    print("\nDirection Predictions:")
    print(pd.Series(direction_predictions).value_counts())

    print("\nTrade Training Distribution:")
    print(pd.Series(y_trade_train).value_counts())

    trade_probabilities = trade_classifier.predict_proba(X_test)

    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
        predictions = (
            trade_probabilities[:, 1] > threshold
        ).astype(int)

        print(
            threshold,
            pd.Series(predictions).value_counts().to_dict(),
        )

    print("\nTrade Probability Statistics:")
    print(pd.Series(trade_probabilities[:, 1]).describe())

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

    if accuracy_gap > 0.10 or r2_gap > 0.50:
        print("Overfit")
    elif train_accuracy < 0.50 and accuracy < 0.50:
        print("Underfit")
    else:
        print("Seems healthy")

    models = {
        "trade_classifier": trade_classifier,
        "direction_classifier": direction_classifier,
        "regressor": regressor,
    }

    return models, results
