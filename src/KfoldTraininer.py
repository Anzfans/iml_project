import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

class KFoldTrainer:
    def __init__(self, model_factory, config):
        """
        :param model_factory: 一个函数，根据名称返回模型对象 (来自你的 model_configs)
        :param config: 全局配置字典 (来自 yaml)
        """
        self.model_factory = model_factory
        self.config = config
        self.results = {}

    def run(self, X, y, model_name, class_balance=True):
        print(f"\n{'='*20} 正在开始 {model_name} 的 K-Fold 验证 {'='*20}")
        
        n_splits = self.config.get('n_splits', 5)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.get('seed', 42))
        
        metrics = {'accuracy': [], 'recall': [], 'time': []}
        all_y_real = []
        all_y_pred = []

        # 将 X 转换为 numpy 方便索引
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        y_values = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_values, y_values)):
            start_time = time.time()
            
            # 1. 划分数据
            X_train, X_val = X_values[train_idx], X_values[val_idx]
            y_train, y_val = y_values[train_idx], y_values[val_idx]

            # 2. 预处理 (在 Fold 内部进行，防止数据泄露)
            if self.config.get('use_scaler', True):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            # 3. 获取并训练模型
            model = self.model_factory(model_name, class_balance)
            model.fit(X_train, y_train)

            # 4. 预测与评估
            y_pred = model.predict(X_val)
            elapsed = time.time() - start_time
            
            # 存入指标
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['recall'].append(recall_score(y_val, y_pred))
            metrics['time'].append(elapsed)
            all_y_real.extend(y_val)
            all_y_pred.extend(y_pred)

            print(f"Fold {fold+1}: Accuracy={metrics['accuracy'][-1]:.4f}, Recall={metrics['recall'][-1]:.4f}, Time={elapsed:.2f}s")

        self._print_summary(model_name, metrics, all_y_real, all_y_pred)
        return metrics

    def _print_summary(self, name, metrics, y_real, y_pred):
        print(f"\n总结 [{name}]:")
        print(f"  Accuracy: {np.mean(metrics['accuracy']):.4f} (+/- {np.std(metrics['accuracy']):.4f})")
        print(f"  Recall:   {np.mean(metrics['recall']):.4f} (+/- {np.std(metrics['recall']):.4f})")
        print(f"  Avg Time: {np.mean(metrics['time']):.2f}s")
        print("\n混淆矩阵:")
        print(confusion_matrix(y_real, y_pred))
        print("\n分类报告:")
        print(classification_report(y_real, y_pred))