from src.train import train_and_save_model  
from src.preprocess import basic_preprocess
from src.predict import pridict_and_save
import pandas as pd   
def main():
    # 1. 读取数据
    df = pd.read_csv('data/raw/train.csv')
    
    # 2. 预处理数据
    df_processed = basic_preprocess(df)
    
    # 3. 训练模型并保存
    train_and_save_model("logistic_baseline", df_processed)

    test_data_processed = basic_preprocess(pd.read_csv('data/raw/test.csv'))

    pridict_and_save("logistic_baseline", test_data_processed)
    


if __name__ == "__main__":
    main()
