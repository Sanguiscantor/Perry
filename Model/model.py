from xgboost import XGBClassifier


def build_model():

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.4,
        colsample_bytree=0.4,
        random_state=42,
        eval_metric="mlogloss"
    )

    return model