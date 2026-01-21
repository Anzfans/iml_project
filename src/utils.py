import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocess import  preprocess_for_model
import seaborn as sns
plt.rcParams.update({'font.size': 18})

def load_trained_model(model_name):
    # 假设你的模型保存在 'results/' 或 'models/' 目录下
    # 请根据你实际的保存路径修改这里
    path = f'results/{model_name+'_submission.csv'}.pkl' 
    return joblib.load(path)

def ensemble_predict(model_names, X_test_raw, mapping):
    """
    手动投票逻辑：
    1. 遍历每个模型
    2. 针对该模型做特定的预处理 (preprocess_for_model)
    3. 预测概率
    4. 取平均值
    """
    all_probs = []
    
    for name in model_names:
        print(f"Loading and predicting with: {name}...")
        model = load_trained_model(name)
        
        # 关键点：每个模型可能需要特定的特征处理（比如不同的列筛选）
        # 我们传入 原始经过basic处理的df，让 preprocess_for_model 处理成该模型需要的格式
        df_model_input = preprocess_for_model(X_test_raw, name, mapping)
        
        # 获取概率 (Soft Voting), 如果模型不支持 predict_proba，则用 predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_model_input)[:, 1] # 取正类的概率
        else:
            probs = model.predict(df_model_input) # 只有0/1
        print(f"Model {name} output samples: {probs[:12]}")
        all_probs.append(probs)
    
    # 计算概率
    weights = [1, 2]
    avg_probs = np.average(all_probs, axis=0, weights=weights)
    
    # 转换为最终类别 (阈值 0.5)
    final_preds = (avg_probs >= 0.5).astype(int)
    
    return final_preds

def visualize_lr_weights(pipeline):
    # 1. 提取模型和特征名
    model = pipeline.named_steps['classifier']
    
    try:
        feature_names = pipeline.named_steps['scaler'].get_feature_names_out()
    except:
        if 'preprocessor' in pipeline.named_steps:
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        else:
            raise ValueError("无法自动获取特征名，请检查 Pipeline 步骤名称")

    # 2. 提取权重
    weights = model.coef_.flatten()

    # 3. 整理成 DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': weights
    })
    importance_df['AbsWeight'] = importance_df['Weight'].abs()

    # --- 逻辑分歧点 ---
    # 获取前 15 个最重要的 (Top 15 Important)
    top_15_df = importance_df.sort_values(by='AbsWeight', ascending=True).tail(15)
    
    # 获取倒数 15 个最不重要的 (Least 15 Important)
    bottom_15_df = importance_df.sort_values(by='AbsWeight', ascending=False).tail(15)

    # 4. 绘图：Top 15 重要特征
    plt.figure(figsize=(10, 6))
    colors_top = ['#ff7675' if w < 0 else '#74b9ff' for w in top_15_df['Weight']]
    plt.barh(top_15_df['Feature'], top_15_df['Weight'], color=colors_top)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.title('Top 15 Most Influential Features', fontsize=14)
    plt.xlabel('Coefficient Value')
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 5. 绘图：倒数 15 个最不重要特征
    plt.figure(figsize=(10, 6))
    colors_bottom = ['#ff7675' if w < 0 else '#74b9ff' for w in bottom_15_df['Weight']]
    plt.barh(bottom_15_df['Feature'], bottom_15_df['Weight'], color=colors_bottom)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    # 注意：这里的横坐标范围通常会非常小，因为权重接近 0
    plt.title('Least 15 Influential Features (Near Zero)', fontsize=14)
    plt.xlabel('Coefficient Value')
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_xgb_importance_direct(model, feature_names, top_n=15):
    """
    可视化 XGBoost 的最强 15 个和最弱 15 个特征
    """
    # 1. 获取重要性得分
    importance_scores = model.get_booster().get_score(importance_type='gain')
    
    # 2. 映射特征名称
    feat_importances = []
    for i, name in enumerate(feature_names):
        # 尝试匹配 f+index 或者原始名称
        score = importance_scores.get(f'f{i}', 0) 
        if name in importance_scores:
            score = importance_scores[name]
        feat_importances.append({'Feature': name, 'Importance': score})
    
    full_df = pd.DataFrame(feat_importances).sort_values(by='Importance', ascending=False)

    # 3. 提取两极特征
    top_df = full_df.head(top_n)
    # 如果总特征数不足 top_n，则取全部；否则取末尾 n 个
    bottom_df = full_df.tail(top_n).sort_values(by='Importance', ascending=True)

    # 4. 创建画布 (1行2列)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 左侧图：Top 15
    sns.barplot(data=top_df, x='Importance', y='Feature', hue='Feature', 
                palette='magma', legend=False, ax=ax1)
    ax1.set_title(f'Top {top_n} Strongest Features (Gain)', fontsize=14)
    ax1.set_xlabel('Average Gain')

    # 右侧图：Bottom 15
    sns.barplot(data=bottom_df, x='Importance', y='Feature', hue='Feature', 
                palette='viridis', legend=False, ax=ax2)
    ax2.set_title(f'Bottom {top_n} Weakest Features (Gain)', fontsize=14)
    ax2.set_xlabel('Average Gain')

    plt.tight_layout()
    plt.show()

    # 返回那些贡献度为 0 的特征列表，方便你直接删除
    zero_importance = full_df[full_df['Importance'] == 0]['Feature'].tolist()
    return zero_importance