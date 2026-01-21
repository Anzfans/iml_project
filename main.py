import argparse
import pandas as pd
from src.ModelTrainer import ModelTrainer
from src.ModelPredictor import ModelPredictor
from src.preprocess import basic_preprocess

def main():
    parser = argparse.ArgumentParser(description="Bank Marketing Pipeline")
    subparsers = parser.add_subparsers(dest="mode", help="运行模式: train 或 predict")

    # 训练模式子命令
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--model", type=str, required=True, choices=['xgboost', 'svm', 'logistic'])
    train_p.add_argument("--save_id", type=str, default="best_model")

    # 预测模式子命令
    pred_p = subparsers.add_parser("predict")
    pred_p.add_argument("--model_file", type=str, required=True, help="模型文件名(不带.pkl)")
    pred_p.add_argument("--out", type=str, default="submission")

    args = parser.parse_args()

    # 统一数据加载与基础处理
    if args.mode:
        data_path = 'data/raw/train.csv' if args.mode == 'train' else 'data/raw/test.csv'
        df = basic_preprocess(pd.read_csv(data_path))

        if args.mode == "train":
            ModelTrainer(output_dir='results').train_and_save(args.model, df, args.save_id)
        elif args.mode == "predict":
            ModelPredictor(args.model_file).predict(df, args.out)

if __name__ == "__main__":
    main()