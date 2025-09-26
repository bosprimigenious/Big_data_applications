# Human vs AI Generated Essays 数据分析项目

## 项目简介

本项目旨在构建能够准确区分人类撰写文本和AI生成文本的分类模型，分析人类和AI在写作风格、语言模式、词汇使用等方面的差异，为AI内容检测提供技术支撑。

## 数据集

- **数据来源**: Kaggle - Human vs AI Generated Essays
- **数据规模**: 2,750篇文章（平衡数据集）
- **文本长度**: 平均300-800字

## 项目结构

```
Big_data_applications/
├── data/                              # 数据目录
│   └── balanced_ai_human_prompts.csv  # 原始数据集
├── models/                            # 模型保存目录
├── 01_data_exploration.py            # 数据探索与可视化
├── 02_feature_engineering.py         # 特征工程(待写)
├── 03_model_training.py              # 模型训练与比较(待写)
├── 04_model_evaluation.py            # 模型评估与深度分析(待写)
├── main.py                           # 主程序入口
├── requirements.txt                  # 依赖管理
├── tasks.md                          # 任务分工方案
├── ideas.md                          # 项目思路
└── README.md                         # 项目说明
```

## 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 或者手动安装
pip install pandas numpy matplotlib seaborn scikit-learn

# 可选：安装NLP相关依赖
pip install nltk textstat
```

## 团队分工


## 寻求帮助
联系开发者：[bosprimigenious@foxmail.com](mailto:bosprimigenious@foxmail.com)


## 许可证

本项目仅用于学术研究和教育目的。

## 更新日志

- **v1.0.0**: 初始版本，包含完整的分析流程
- 支持多种机器学习算法
- 完整的特征工程流程
- 详细的模型评估和分析
