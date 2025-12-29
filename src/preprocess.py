import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_pdays(df):
    """
    专门针对 pdays 进行三分类处理，并为逻辑回归生成 One-Hot 编码。
    分类逻辑：
    - 0-7天: Recent (高转化区)
    - 8-20天: Follow-up (中期区)
    - >20天: Cold/New (低转化/新客区)
    """
    
    # 1. 创建分类标签
    # 我们直接在函数内定义逻辑，避免影响原数据
    conditions = [
        (df['pdays'] <= 7),
        (df['pdays'] > 7) & (df['pdays'] <= 20),
        (df['pdays'] > 20)
    ]
    choices = ['pdays_recent', 'pdays_followup', 'pdays_cold']
    
    # 创建一个临时列
    temp_cat = np.select(conditions, choices, default='pdays_cold')
    
    # 2. 转换为 One-Hot 编码 (哑变量)
    # 对于逻辑回归，建议 drop_first=True 以避免多重共线性（哑变量陷阱）
    # 但如果是为了特征解释性，也可以保留全部，这里演示保留关键特征的做法
    pdays_dummies = pd.get_dummies(temp_cat, prefix='', drop_first=True)
    
    # 3. 合并回原 dataframe 并删除原始 pdays 列（逻辑回归对 999 这种异常值很敏感）
    df = pd.concat([df, pdays_dummies], axis=1)
    df.drop('pdays', axis=1, inplace=True)
    
    return df


# 综合预处理函数, 把处理后的数据保存为新的 CSV 文件在
def basic_preprocess(df, save_path):
    df = df.copy()
    
    # 1. 目标变量转换
    if 'subscribe' in df.columns:
        df['target'] = df['subscribe'].map({'yes': 1, 'no': 0})
        df.drop('subscribe', axis=1, inplace=True)
    
    # 2. pdays 特征工程 (你之前的逻辑)
    #df = preprocess_pdays(df)
    
    # 3. 选取分类变量进行 One-Hot (包括 pdays_cat, job, loan 等)
    # drop_first=True 对逻辑回归至关重要
    
    df = pd.get_dummies(df, drop_first=True)
    
    # 4. 数值变量处理 (Duration 取对数)
    df['duration'] = np.log1p(df['duration'])
    
    # 移除 ID 等无关列
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
        
    df.to_csv(save_path, index=False)

    return df




# 使用示例:
# train_df = pd.read_csv('train.csv')
# train_df = preprocess_pdays(train_df)