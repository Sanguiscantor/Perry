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


def build_binary_model():

    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,
        max_depth=4,
        subsample=0.5,
        colsample_bytree=0.5,
        random_state=42,
        eval_metric="logloss",
    )
