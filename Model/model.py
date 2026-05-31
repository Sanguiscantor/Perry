from xgboost import XGBClassifier


def build_model():

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=50,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.5,
        colsample_bytree=0.5,
        random_state=42,
        eval_metric="mlogloss"
    )

    return model