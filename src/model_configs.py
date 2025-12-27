from sklearn.linear_model import LogisticRegression

def get_model(model_name):
    if model_name == "logistic_baseline":
        return LogisticRegression( max_iter=1000)