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
    parser.add_argument('--vote_models', type=str, default=None, 
                        help='Comma-separated list of models to vote with (e.g., "svm,rf"). If set, skips training.')
    
    args = parser.parse_args()

    # --- 第一步：通用数据准备 (获取 Mapping) ---
    # 无论训练还是投票，我们都需要从 Train 集获取 Mapping 信息
    print("Processing training data to generate mappings...")
    df_train_raw = pd.read_csv('data/raw/train.csv')
    df_train_basic = basic_preprocess(df_train_raw)
    
    target_enc_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'day_of_week']
    # 这里我们只获取 mapping，不一定非要生成 df_train_processed，除非是训练模式
    mapping = get_target_mapping(df_train_basic, target_enc_cols)

    # --- 第二步：逻辑分支 ---
    
    if args.vote_models:
        # ================= 投票模式 (Vote Mode) =================
        model_list = args.vote_models.split(',')
        print(f"=== Vote Mode Activated === Models: {model_list}")
        
        # 1. 读取并进行基础预处理 Test 数据
        df_test_raw = pd.read_csv('data/raw/test.csv')
        df_test_basic = basic_preprocess(df_test_raw)
        
        # 2. 调用上面的手动投票函数
        # 注意：我们在函数内部针对每个模型再次调用 preprocess_for_model
        # 这样能保证即使不同模型用了不同的特征子集，也能正确处理
        predictions = ensemble_predict(model_list, df_test_basic, mapping)
        
        # 3. 保存结果
        # 这里假设你有一个 save_submission(predictions, filename) 的函数
        # 如果没有，可以用 pd.DataFrame 简单构建
        reverse_map = {1: 'yes', 0: 'no'}
        subscribe_labels = [reverse_map[p] for p in predictions]
        output_df = pd.DataFrame({'id': df_test_raw.index, 'subscribe': subscribe_labels}) 
        output_df.to_csv(f'results/{'voting_'+args.file_name}', index=False)
        print(f"Vote result saved to results/{'voting_'+args.file_name}")

    else:
        # ================= 训练模式 (Train Mode - 原有逻辑) =================
        print(f"=== Train Mode Activated === Model: {args.model_name}")
        
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
