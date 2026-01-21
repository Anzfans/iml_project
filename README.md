# CS182 银行产品销售预测项目 (cs182-project)

这是一个基于机器学习的分类预测项目，旨在预测客户是否会订阅银行产品。本项目采用 **uv** 进行依赖管理，并遵循标准的数据科学项目结构。

## 📂 目录结构说明

- **data/**: 数据存放目录 (已在 .gitignore 中忽略，需手动创建)
  - **raw/**: 存放从天池下载的原始数据文件（如 `train.csv`, `test.csv`）。
  - **processed/**: 存放经过 `preprocess.py` 处理后的中间特征数据。
- **src/**: 源代码核心逻辑
  - `preprocess.py`: 数据清洗、特征工程逻辑。
  - `train.py`: 模型训练与 Pipeline 构建逻辑。
  - `predict.py`: 加载模型并生成预测结果。
  - `model_configs.py`: 模型定义与超参数配置。
- **notebook**:记录项目过程中的一些探索和实验记录，以及相关图片
- **results/**: 存放训练好的模型文件 (`.pkl`) 和最终生成的提交文件 (`.csv`)。
- **main.py**: 项目总入口，负责串联预处理、训练和预测流程。
- **pyproject.toml / uv.lock**: uv 环境配置文件。

---

## 🚀 快速开始

### 1. 环境初始化
本项目推荐使用 [uv](https://docs.astral.sh/uv/)。在项目根目录下运行以下命令，它会自动安装所有依赖并创建虚拟环境：

```bash
uv sync
```


## 2. 运行项目
只需运行 main.py，程序将自动完成任务：

```Bash
uv run main.py
```




## 添加新依赖
```Bash
uv add <库名称>
```
