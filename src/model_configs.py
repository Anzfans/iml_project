from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_model(model_name):
    if model_name == "logistic_baseline":
        return LogisticRegression( max_iter=1000)
    elif model_name == "XG":
   
        return XGBClassifier(
            n_estimators=2000,
            learning_rate=0.005,
            max_depth=7,
            eval_metric='logloss',
            subsample=0.8,
            colsample_bytree=0.6,
        )