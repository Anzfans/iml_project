from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_model(model_name):
    if model_name == "logistic_baseline":
        return LogisticRegression( max_iter=1000)
    elif model_name == "XG":
        return XGBClassifier(
            n_estimators=2000,
            learning_rate=0.004,
            max_depth=6,
            eval_metric='logloss',
            subsample=0.8,
            colsample_bytree=0.6,
        )
    elif model_name == "SVM":
        return SVC(kernel='rbf',# 必须使用核函数处理非线性
            cache_size=4000,    # 增加缓存，提升训练速度              
            C=0.8,             # 增加惩罚系数。默认是1.0。
                                # C越大，模型越会为了那剩下几个百分点的正确率去“死磕”
            gamma='scale',      # 自动决定核函数的辐射范围
            probability=True,   # 必须开启，方便后续做模型融合
# 自动处理类别不平衡，让模型更重视那少数的成功案例
            random_state=42)