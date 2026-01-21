# src/core/trainer.py
import joblib
from os import path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.model_configs import get_model

class ModelTrainer:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir

    def train_and_save(self, model_name, train_df, save_id):
        # 1. 准备数据 (复用你原来的逻辑)
        X = train_df.drop('target', axis=1)
        y = train_df['target']
        
        # 2. 获取模型
        model = get_model(model_name)
        
        # 3. 逻辑分发 (保持你原有的 if-else 结构，但更整洁)
        if model_name in ['logistic', 'svm']:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
        else: # xgboost 等
            pipeline = model

        # 4. 训练与保存
        print(f"开始训练: {model_name}...")
        if pipeline is not None:
            pipeline.fit(X, y)
        else:
         # 处理 pipeline 为空的情况，比如报错或打印提示
            print("Error: Pipeline was not initialized.")
        
        save_path = path.join(self.output_dir, f'{save_id}.pkl')
        joblib.dump(pipeline, save_path)
        print(f"模型已保存至: {save_path}")
        return pipeline