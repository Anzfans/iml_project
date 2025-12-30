import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
_GLOBAL_MAPPING = None

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
def basic_preprocess(df):
    df = df.copy()
    
    # 1. 目标变量转换
    if 'subscribe' in df.columns:
        df['target'] = df['subscribe'].map({'yes': 1, 'no': 0})
        df.drop('subscribe', axis=1, inplace=True)
    
    # 2. pdays 特征工程 (你之前的逻辑)
    df = preprocess_pdays(df)
    
    
    # 移除 ID 等无关列
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    return df

def Lg_preprocess(df,):
    df = pd.get_dummies(df, drop_first=True)
    df['duration'] = np.log1p(df['duration'])
    return df

def XG_preprocess(df, mapping=None):
    df['duration'] = np.log1p(df['duration'])

    if mapping is not None:
        target_cols = [c for c in mapping.keys() if c != '_global_mean']
        for col in target_cols:
            df[col] = df[col].map(mapping[col])
            df[col].fillna(mapping['_global_mean'])

    for col in df.select_dtypes('object'):
        df[col] = pd.factorize(df[col])[0]
    return df

def preprocess_for_model(df, model_name, mapping=None):
    if model_name == 'logistic_baseline':
        return Lg_preprocess(df)
    elif model_name == 'XG':
        return XG_preprocess(df, mapping)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_target_mapping(train_df, cols):
    """从训练集中学习平滑后的目标映射"""
    mapping = {}
    global_mean = train_df['target'].mean()
    m = 50  # 平滑系数
    
    for col in cols:
        stats = train_df.groupby(col)['target'].agg(['count', 'mean'])
        # 平滑公式
        smooth = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
        mapping[col] = smooth.to_dict()
        
    mapping['_global_mean'] = global_mean
    return mapping

# 使用示例:
# train_df = pd.read_csv('train.csv')
# train_df = preprocess_pdays(train_df)