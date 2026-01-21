import numpy as np
from scipy import stats

class ModelEvaluator:
    def __init__(self, alpha=0.05):
        """
        :param alpha: 显著性水平，默认 0.05
        """
        self.alpha = alpha

    def compare_models(self, res1, res2, name1, name2):
        """
        对比两个模型的多个指标
        :param res1: 模型1的返回结果字典 (包含 'fold_metrics')
        :param res2: 模型2的返回结果字典
        """
        print(f"\n{'='*20} 配对 T 检验结果 {'='*20}")
        
        metrics = res1['fold_metrics'].keys() # 动态获取指标列表，如 accuracy, recall
        
        comparison_summary = {}
        
        for metric in metrics:
            scores1 = res1['fold_metrics'][metric]
            scores2 = res2['fold_metrics'][metric]
            
            # 执行核心 T-Test 逻辑
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            mean_diff = np.mean(scores1) - np.mean(scores2)
            
            is_significant = p_value < self.alpha
            winner = "Draw (无显著差异)"
            if is_significant:
                winner = name1 if mean_diff > 0 else name2

            # 打印格式化结果
            print(f"\n指标: {metric.upper()}")
            print(f"  {name1}: {np.mean(scores1):.4f} ± {np.std(scores1):.4f}")
            print(f"  {name2}: {np.mean(scores2):.4f} ± {np.std(scores2):.4f}")
            print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
            print(f"  结论: {'*** 显著差异' if is_significant else '无显著差异'} (α={self.alpha})")
            if is_significant:
                print(f"  → 优胜者: {winner}")
            
            comparison_summary[metric] = {
                'p_value': p_value,
                'is_significant': is_significant,
                'winner': winner
            }
            
        return comparison_summary