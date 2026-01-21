# src/core/predict.py
import joblib
import pandas as pd
import os

class ModelPredictor:
    def __init__(self, model_file_name):
        model_path = os.path.join('results', f'{model_file_name}.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型: {model_path}")
        self.model = joblib.load(model_path)
        print(f"模型 {model_file_name} 加载成功")

    def predict(self, test_data, output_name):
        # 复用你原来的预测逻辑
        predictions = self.model.predict(test_data)
        
        reverse_map = {1: 'yes', 0: 'no'}
        submission_df = pd.DataFrame({
            'id': test_data.index,
            'subscribe': [reverse_map[p] for p in predictions]
        })
        
        save_path = f'results/{output_name}_submission.csv'
        submission_df.to_csv(save_path, index=False)
        print(f"预测结果已保存: {save_path}")