import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from preprocess import basic_preprocess, XG_preprocess, get_target_mapping
from train import train_and_save_model
from preprocess import preprocess_for_model



def analyze_xgboost_errors():
    # 1. 模拟数据加载
    data_path = 'data/raw/train.csv'
    if not os.path.exists(data_path):
        print(f"找不到文件: {data_path}")
        return
    
    raw_df = pd.read_csv(data_path)
    
    # 2. 划分验证集
    train_raw, val_raw = train_test_split(raw_df, test_size=0.2, random_state=42)
    
    # 【核心修复】：必须立即重置索引，否则 basic_preprocess 里的 map 和 concat 会产生 NaN
    train_raw = train_raw.reset_index(drop=True)
    val_raw = val_raw.reset_index(drop=True)
    
    # 3. 执行基础预处理
    train_base = basic_preprocess(train_raw)
    val_base = basic_preprocess(val_raw)
    
    # 4. 获取 Mapping 并预处理训练集
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'day_of_week']
    mapping = get_target_mapping(train_base, cat_cols)
    
    df_train_processed = preprocess_for_model(train_base, 'XG', mapping)
    
    # 【双重保险】：强制转换标签类型并剔除可能存在的残留 NaN
    df_train_processed = df_train_processed.dropna(subset=['target'])
    df_train_processed['target'] = df_train_processed['target'].astype(int)
        
    # 训练并保存
    train_and_save_model('XG', df_train_processed, 'train_processed.csv')

    # 5. 处理验证集
    df_val_processed = preprocess_for_model(val_base, 'XG', mapping)

    # 统一使用处理后的 target 列，不要直接用 val_raw['subscribe']，防止类型不匹配
    X_val = df_val_processed.drop('target', axis=1)
    y_val = df_val_processed['target'].astype(int).values

    # 6. 加载模型
    model_path = 'results/XG_train_processed.csv.pkl'
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}")
        return
    
    model = joblib.load(model_path)
    print("模型加载成功。开始预测验证集...")
    
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    
    # 计算准确率
    val_acc = (val_preds == y_val).mean()
    print(f"验证集准确率 (Validation Accuracy): {val_acc:.4f}")

    # 7. 分析错误案例
    # 此时 val_raw, val_preds, y_val 的索引和长度是完美对齐的
    error_mask = (val_preds != y_val)
    errors = val_raw[error_mask].copy()
    
    print(f"总错误案例数: {len(errors)}")
    if len(errors) > 0:
        # 直接使用 mask 赋值，避免使用 .index 导致潜在的匹配错误
        errors['predicted'] = val_preds[error_mask]
        errors['predicted_prob'] = val_probs[error_mask]
        errors['true_label'] = y_val[error_mask]
        
        print("部分错误案例预览:")
        print(errors[['job', 'education', 'duration', 'true_label', 'predicted']].head())
        
        errors.to_csv('results/xgboost_error_analysis.csv', index=False)
        print("错误案例已保存至 results/xgboost_error_analysis.csv")
    else:
        print("没有错误案例，模型表现完美！")

if __name__ == '__main__':
    analyze_xgboost_errors()    
    

   