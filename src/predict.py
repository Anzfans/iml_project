import pandas as pd
import joblib
import os

def pridict_and_save(model_name, test_data):
    # 1. 加载保存的 Pipeline 模型
    model_path = f'results/{model_name}.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    pipeline = joblib.load(model_path)
    
    # 2. 确保输入数据是 DataFrame 格式
    if not isinstance(test_data, pd.DataFrame):
        raise ValueError("输入数据必须是 pandas DataFrame 格式")
    
    # 3. 使用 Pipeline 进行预测
    predictions = pipeline.predict(test_data)

    reverse_map = {1: 'yes', 0: 'no'}
    subscribe_labels = [reverse_map[p] for p in predictions]
    
    # 4. 构建符合 submission.csv 格式的 DataFrame
    submission_df = pd.DataFrame({
        'id': test_data.index,
        'subscribe': subscribe_labels
    })

    # 5. 保存预测结果到 CSV 文件
    submission_path = f'results/{model_name}_submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"预测结果已保存至: {submission_path}")    
    