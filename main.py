from src.train import train_and_save_model  
from src.preprocess import preprocess_for_model, get_target_mapping,basic_preprocess
from src.predict import predict_and_save
from src.utils import ensemble_predict, load_trained_model
import pandas as pd   
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    # 原有参数
    parser.add_argument('--model_name', type=str, default='logistic_baseline')
    parser.add_argument('--file_name', type=str, default='submission.csv')
    
    # 新增参数：vote_models
    # 用法示例：python main.py --vote_models svm,random_forest,logistic_baseline
    
    args = parser.parse_args()

    # --- 第一步：通用数据准备 (获取 Mapping) ---
    # 无论训练还是投票，我们都需要从 Train 集获取 Mapping 信息
    print("Processing training data to generate mappings...")
    df_train_raw = pd.read_csv('data/raw/train.csv')
    df_train_basic = basic_preprocess(df_train_raw)
    
    target_enc_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'day_of_week']
    # 这里我们只获取 mapping，不一定非要生成 df_train_processed，除非是训练模式
    mapping = get_target_mapping(df_train_basic, target_enc_cols)

    

        
        # 1. 完成训练集的特定模型处理
    df_train_processed = preprocess_for_model(df_train_basic, args.model_name, mapping)
        
        # 2. 训练并保存
    train_and_save_model(args.model_name, df_train_processed, args.file_name)
        
        # 3. 处理测试集并预测 (单模型)
    df_test_raw = pd.read_csv('data/raw/test.csv')
    df_test_basic = basic_preprocess(df_test_raw)
    df_test_processed = preprocess_for_model(df_test_basic, args.model_name, mapping)
        
    predict_and_save(args.model_name, df_test_processed, args.file_name)

if __name__ == '__main__':
    main()
