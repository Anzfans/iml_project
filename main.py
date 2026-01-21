import argparse
import pandas as pd

from src.ModelTrainer import ModelTrainer
from src.ModelPredictor import ModelPredictor
from src.KfoldTraininer import KFoldTrainer
from src.ModelEvaluator import ModelEvaluator
from src.preprocess import basic_preprocess
from src.model_configs import get_model

def main():
    parser = argparse.ArgumentParser(description="Bank Marketing Project CLI")
    subparsers = parser.add_subparsers(dest="mode", help="执行模式")

    # 1. 训练模式
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--model", type=str, required=True, choices=['xgboost', 'svm', 'logistic'])
    train_p.add_argument("--save_id", type=str, default="best_model")

    # 2. 预测模式
    pred_p = subparsers.add_parser("predict")
    pred_p.add_argument("--model_file", type=str, required=True)
    pred_p.add_argument("--out", type=str, default="submission")

    # 3. K-Fold 验证模式
    kfold_p = subparsers.add_parser("kfold")
    kfold_p.add_argument("--model", type=str, required=True, choices=['xgboost', 'svm', 'logistic'])
    kfold_p.add_argument("--splits", type=int, default=5)

    # 4. 模型对比模式 (T-Test)
    compare_p = subparsers.add_parser("compare")
    compare_p.add_argument("--m1", type=str, required=True)
    compare_p.add_argument("--m2", type=str, required=True)

    args = parser.parse_args()

    # --- 逻辑执行 ---
    if args.mode in ["train", "kfold"]:
        # 加载训练数据
        df = basic_preprocess(pd.read_csv('data/raw/train.csv'))
        
        if args.mode == "train":
            ModelTrainer(output_dir='results').train_and_save(args.model, df, args.save_id)
        
        elif args.mode == "kfold":
            # 这里利用你封装的 KFoldTrainer
            config = {'n_splits': args.splits, 'seed': 42}
            trainer = KFoldTrainer(model_factory=get_model, nsplits=args.splits)
            X, y = df.drop('target', axis=1), df['target']
            trainer.run(X, y, args.model)

    elif args.mode == "predict":
        df_test = basic_preprocess(pd.read_csv('data/raw/test.csv'))
        ModelPredictor(args.model_file).predict(df_test, args.out)

    elif args.mode == "compare":
        # 假设对比逻辑需要两组预先跑好的结果，或者现场跑
        # 这里演示现场跑 K-Fold 并对比
        df = basic_preprocess(pd.read_csv('data/raw/train.csv'))
        X, y = df.drop('target', axis=1), df['target']
        
        trainer = KFoldTrainer(model_factory=get_model, nsplits=5)
        
        print(f"正在评估 {args.m1}...")
        res1 = trainer.run(X, y, args.m1)
        print(f"正在评估 {args.m2}...")
        res2 = trainer.run(X, y, args.m2)
        
        evaluator = ModelEvaluator(alpha=0.05)
        evaluator.compare_models({'fold_metrics': res1}, {'fold_metrics': res2}, args.m1, args.m2)

if __name__ == "__main__":
    main()