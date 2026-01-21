import pandas as pd 
import argparse
import os
import itertools  # 用于生成两两对比的组合
from src.train import train_and_save_model  
from src.preprocess import preprocess_for_model, get_target_mapping, basic_preprocess
from src.predict import predict_and_save
from src.utils import ensemble_predict, load_trained_model
from src.KfoldTraininer import execute_pipeline, parse_args as parse_kfold_args
from src.paired_ttest import paired_ttest, get_results

def main():
    parser = argparse.ArgumentParser()
    # 默认执行这三个模型
    parser.add_argument('--models', type=str, default='logistic,xgboost,svm', 
                        help='Comma-separated list of models to train')
    parser.add_argument('--file_name', type=str, default='submission.csv')
    parser.add_argument('--balance', action='store_true', default=True, help='Use balance in Kfold')
    
    args = parser.parse_args()

    # --- 第一步：通用数据准备 ---
    print("\n[1/4] Preparing Data...")
    df_train_raw = pd.read_csv('data/raw/train.csv')
    df_test_raw = pd.read_csv('data/raw/test.csv')
    df_train_basic = basic_preprocess(df_train_raw)
    df_test_basic = basic_preprocess(df_test_raw)
    
    target_enc_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'day_of_week']
    mapping = get_target_mapping(df_train_basic, target_enc_cols)

    # --- 第二步：循环训练与预测 (单模型产出) ---
    print("\n[2/4] Training & Predicting Individual Models...")
    model_list = [m.strip() for m in args.models.split(',')]

    for model_name in model_list:
        model_name = model_name.strip()
        print(f"--- Processing: {model_name} ---")
        
        # 预处理...
        df_train_processed = preprocess_for_model(df_train_basic, model_name, mapping)
        df_test_processed = preprocess_for_model(df_test_basic, model_name, mapping)

        # 重点：构造一个包含模型名称的文件标识符
        # 这样 save_id 就会是 'logistic_baseline' 或 'xgboost'
        save_id = f"{model_name}" 

        # 训练
        train_and_save_model(model_name, df_train_processed, save_id)

        # 预测
        # 注意：这里我们修改传递给 predict_and_save 的参数
        predict_and_save(model_name, df_test_processed, save_id)

    # --- 第三步：执行 K-Fold (保存评估分数) ---
    print("\n[3/4] Running K-Fold Cross Validation for all models...")
    # 这里 '--model all' 会根据你的 Kfold.py 逻辑依次评估模型
    kfold_args = parse_kfold_args(['--model', 'all', '--balance' if args.balance else '', '--save']) 
    execute_pipeline(kfold_args)

    # --- 第四步：执行 Paired T-Test (直接 Load 第三步的结果) ---
    print("\n[4/4] Running Paired T-Tests (Loading CV results)...")
    
    # 自动生成两两对比组合，例如: (logistic, xgboost), (xgboost, svm) 等
    comparisons = list(itertools.combinations(model_list, 2))
    
    for m1, m2 in comparisons:
        print(f"\nComparing {m1} vs {m2}:")
        # 核心：load_saved=True 确保直接读取刚刚 Kfold 保存的文件
        res1 = get_results(m1, args.balance, load_saved=True, folds=5, seed=42)
        res2 = get_results(m2, args.balance, load_saved=True, folds=5, seed=42)

        if res1 and res2:
            for metric in ['accuracy', 'recall']:
                paired_ttest(
                    res1['fold_metrics'][metric], 
                    res2['fold_metrics'][metric], 
                    m1.upper(), 
                    m2.upper(), 
                    metric
                )
        else:
            print(f"Skipping {m1} vs {m2} due to missing result files.")

if __name__ == '__main__':
    main()