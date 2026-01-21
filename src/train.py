from src.model_configs import get_model
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils import visualize_lr_weights, visualize_xgb_importance_direct

def train_and_save_model(model_name, train_df, data_file):
    # 1. 加载 preprocess 处理好的数值数据
    X = train_df.drop('target', axis=1)
    y = train_df['target']

    if model_name == 'logistic_baseline':
        model = get_model(model_name)
    
    # 3. 定义 Pipeline：将缩放器和模型“焊”在一起
    # 这样你就不用在 preprocess.py 里做标准化了
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

    # 4. 训练
        print(f"正在训练模型: {model_name+'_'+data_file}...")
        pipeline.fit(X, y)
        visualize_lr_weights(pipeline)

    # 5. 保存训练好的 Pipeline (内含 Scaler 参数)
        model_path = f'results/{model_name+'_'+data_file}.pkl'
        joblib.dump(pipeline, model_path)
        print(f"模型已保存至: {model_path}")

    elif model_name == 'XG':
        model = get_model(model_name)
        assert model is not None, "Model should not be None"
        print(f"正在训练模型: {model_name+'_'+data_file}...")
        model.fit(X, y)
        visualize_xgb_importance_direct(model, X.columns.tolist())
        train_acc = model.score(X, y)
        print(f"训练集准确率 (Training Accuracy): {train_acc:.4f}")
        model_path = f'results/{model_name+'_'+data_file}.pkl'
        joblib.dump(model, model_path)
        print(f"模型已保存至: {model_path}")
        
    elif model_name == 'SVM':
        model = get_model(model_name)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        print(f"正在训练模型: {model_name+'_'+data_file}...")
        pipeline.fit(X, y)
        train_acc = pipeline.score(X, y)
        print(f"训练集准确率 (Training Accuracy): {train_acc:.4f}")
        model_path = f'results/{model_name+'_'+data_file}.pkl'
        joblib.dump(pipeline, model_path)
        print(f"模型已保存至: {model_path}")
