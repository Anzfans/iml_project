import joblib
import numpy as np
from src.preprocess import  preprocess_for_model

def load_trained_model(model_name):
    # 假设你的模型保存在 'results/' 或 'models/' 目录下
    # 请根据你实际的保存路径修改这里
    path = f'results/{model_name+'_submission.csv'}.pkl' 
    return joblib.load(path)

def ensemble_predict(model_names, X_test_raw, mapping):
    """
    手动投票逻辑：
    1. 遍历每个模型
    2. 针对该模型做特定的预处理 (preprocess_for_model)
    3. 预测概率
    4. 取平均值
    """
    all_probs = []
    
    for name in model_names:
        print(f"Loading and predicting with: {name}...")
        model = load_trained_model(name)
        
        # 关键点：每个模型可能需要特定的特征处理（比如不同的列筛选）
        # 我们传入 原始经过basic处理的df，让 preprocess_for_model 处理成该模型需要的格式
        df_model_input = preprocess_for_model(X_test_raw, name, mapping)
        
        # 获取概率 (Soft Voting), 如果模型不支持 predict_proba，则用 predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_model_input)[:, 1] # 取正类的概率
        else:
            probs = model.predict(df_model_input) # 只有0/1
        print(f"Model {name} output samples: {probs[:12]}")
        all_probs.append(probs)
    
    # 计算概率
    weights = [1, 2]
    avg_probs = np.average(all_probs, axis=0, weights=weights)
    
    # 转换为最终类别 (阈值 0.5)
    final_preds = (avg_probs >= 0.5).astype(int)
    
    return final_preds