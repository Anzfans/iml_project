# CS182 银行产品销售预测项目 (cs182-project)

这是一个基于机器学习的分类预测项目，旨在预测客户是否会订阅银行产品。本项目采用 **uv** 进行依赖管理，并遵循标准的数据科学项目结构。


## 📂 目录结构说明

- **data/**: 数据存放目录
  - **raw/**: 存放原始数据（如 `train.csv`, `test.csv`）。
  - **processed/**: 存放经过预处理后的中间特征数据（可选）。
- **src/**: 源代码核心逻辑
  - `preprocess.py`: **数据预处理**。包含 `basic_preprocess` 及 `preprocess_for_model`，负责清洗、One-Hot 编码及特征对齐。
  - `ModelTrainer.py`: **训练**。封装 `ModelTrainer` 类，支持 Pipeline 构建（集成 Scaler）与模型持久化。
  - `ModelPredictor.py`: **推理**。封装 `ModelPredictor` 类，负责加载 `.pkl` 模型并生成最终预测标签。
  - `KfoldTraininer.py`: **评估工具**。封装 `KFoldTrainer` 类，执行分层交叉验证并输出详细的分类报告。
  - `ModelEvaluator.py`: **统计对比**。封装 `ModelEvaluator` 类，利用配对 T 检验分析不同模型间的显著性差异。
  - `model_configs.py`: 统一管理所有模型的超参数配置及实例化逻辑。
  - `utils.py`: **通用工具**。包含可视化（特征重要性、权重图）及voting模型逻辑。
- **results/**: 存放训练好的模型 (`.pkl`)、评估报告及最终预测结果 (`.csv`)。
- **notebook/**: 存放 EDA（数据探索性分析）过程中的实验记录及可视化图表。
- **main.py**: **项目总调度入口**。利用 `argparse` 子命令模式实现训练、预测、评估的全流程调度。
- **pyproject.toml / uv.lock**: 项目依赖管理文件（基于 `uv` 工具）。
---

## 🚀 快速开始

### 1. 环境初始化
本项目推荐使用 [uv](https://docs.astral.sh/uv/)。在项目根目录下运行以下命令，它会自动安装所有依赖并创建虚拟环境：

```bash
uv sync
```


## 2. 运行项目
运行项目main.py, 相关命令如下

```Bash
uv run main.py train --model<model_name> --save_id<id>
```

```Bash
uv run main.py predict --model_file<file_name> --out<file_name>
```

```Bash
uv run main.py kfold --model<model_name> --splits<n>
```

```Bash
uv run main.py compare --m1<model_name_1> --m2<model_name_2>
```



## 添加新依赖
```Bash
uv add <库名称>
```
