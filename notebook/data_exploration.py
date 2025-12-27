import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

PLOT_DIR = Path('notebook/plots')
PLOT_DIR.mkdir(parents=True, exist_ok=True)
# 加载数据
df = pd.read_csv('data\\raw\\train.csv')

# 1. 打印数值统计
#print("pdays 数值统计：")
#print(df['pdays'].describe())

# 2. 绘制直方图查看分布
#plt.figure(figsize=(10, 6))
#sns.histplot(x=df['pdays'], bins=100, kde=False)
#plt.title('Distribution of pdays')
#plt.xlabel('Days since last contact')
#plt.ylabel('Frequency')
#plt.savefig(PLOT_DIR / "distribution_of_pdays.png",dpi=300)
#plt.show()
## 过滤掉 999 后的分布
#real_contacts = df[df['pdays'] < 949]['pdays']
#sns.histplot(x=real_contacts, bins=30, kde=True, color='green')
#plt.title('Distribution of pdays (Only normal customers)')
#plt.xlabel('Days since last contact')
#plt.ylabel('Frequency')
#plt.savefig(PLOT_DIR / "distribution_of_pdays_only_normal_customers.png",dpi=300)
#plt.show()
#
## 查看 pdays 最大的前 10 个数值及其出现次数
#print("pdays 高频数值统计：")
#print(real_contacts.value_counts().sort_index(ascending=False).head(10))

#def bin_pdays(x):
    #if x <= 7: return '0-7d (Recent)'
    #elif x <= 20: return '8-20d (Medium)'
    #elif x <= 949: return '21-949d (Long)'
    #else: return '>949d (Never)'

#df['pdays_group'] = df['pdays'].apply(bin_pdays)

## 2. 计算每个组的购买比例 (将 subscribe 转为 0/1)
#df['subscribe_num'] = df['subscribe'].map({'yes': 1, 'no': 0})
#analysis = df.groupby('pdays_group')['subscribe_num'].agg(['count', 'mean']).reset_index()
#analysis.columns = ['pdays_group', 'total_users', 'purchase_rate']

#print(analysis.sort_values('purchase_rate', ascending=False))


df['subscribe_num'] = df['subscribe'].map({'yes': 1, 'no': 0})

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 6))

# --- 图 1: 通话时长的密度分布图 (KDE Plot) ---
plt.subplot(1, 2, 1)
sns.kdeplot(data=df, x="duration", hue="subscribe", fill=True, common_norm=False, palette="viridis", alpha=.5)
plt.title('Duration Density Distribution by Subscribe Status')
plt.xlabel('Call Duration (seconds)')
plt.ylabel('Density')
# 限制一下x轴范围，方便观察（大部分通话在2000秒内）
plt.xlim(0, 2500)

# --- 图 2: 通话时长分桶后的转化率趋势 ---
plt.subplot(1, 2, 2)
# 对 duration 进行等频分桶（分成10组）
df['duration_bin'] = pd.qcut(df['duration'], q=10, duplicates='drop')
conversion_rate = df.groupby('duration_bin', observed=True)['subscribe_num'].mean()

conversion_rate.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Conversion Rate by Duration Deciles')
plt.xlabel('Duration Intervals (seconds)')
plt.ylabel('Conversion Rate (Purchase %)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# --- 补充：查看对数化后的效果 ---
plt.figure(figsize=(8, 5))
sns.histplot(np.log1p(df['duration']), kde=True, color='orange')
plt.title('Distribution of Log-Transformed Duration')
plt.xlabel('Log(Duration + 1)')
plt.show()