from src.train import train_and_save_model  
from src.preprocess import preprocess_for_model, get_target_mapping,basic_preprocess
from src.predict import predict_and_save
import pandas as pd   
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="the name of the model to train and the data paths")
    parser.add_argument('--model_name', type=str, default='logistic_baseline', help='Name of the model to train')
    parser.add_argument('--file_name', type=str, default='train_processed.csv', help='Name of the processed training data CSV file')
    args = parser.parse_args()  
    model_name = args.model_name
    processed_data_path = os.path.join("data", "processed", args.file_name)
    # 1. 读取数据
    df = basic_preprocess(pd.read_csv('data/raw/train.csv'))
    target_enc_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome','day_of_week']
    mapping= get_target_mapping(df, target_enc_cols)
    # 2. 预处理数据
    df_processed = preprocess_for_model(df, model_name,mapping)
    
    # 3. 训练模型并保存
    train_and_save_model(model_name, df_processed,args.file_name)

    test_data_processed = preprocess_for_model(basic_preprocess(pd.read_csv('data/raw/test.csv')), model_name,mapping)

    predict_and_save(model_name, test_data_processed,args.file_name)
    


if __name__ == "__main__":
    main()
