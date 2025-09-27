# AI文本检测分析项目

## 项目简介

本项目成功构建了高精度的AI文本检测模型，能够准确区分人类撰写文本和AI生成文本。通过多层次特征工程和多种机器学习算法，实现了**92.56%**的检测准确率，为AI内容检测提供技术支撑。

## 核心成果

- **模型准确率**: 92.56%
- **AUC值**: 0.9478  
- **特征数量**: 42个基础特征 + 5000维TF-IDF特征
- **算法对比**: 5种主流机器学习算法
- **数据集**: 2,750篇平衡文本（人类/AI各50%）

## 项目结构

```
Big_data_applications/
├── data/                              # 数据目录
│   ├── balanced_ai_human_prompts.csv  # 原始数据集
│   ├── processed_data.csv             # 预处理后数据
│   ├── features_data.csv              # 特征数据
│   └── tfidf_matrix.npz              # TF-IDF矩阵
├── models/                            # 训练好的模型
│   ├── best_model.pkl                 # 最佳模型
│   ├── gradient_boosting_model.pkl    # 梯度提升模型
│   └── ...                           # 其他模型
├── photos/                            # 可视化图表
│   ├── model_comparison.png           # 模型性能对比
│   ├── roc_curves.png                # ROC曲线
│   └── feature_importance.png         # 特征重要性
├── report/                            # 研究报告
│   ├── 实验报告_AI文本检测分析.md      # 完整实验报告
│   └── executive_summary.md           # 执行摘要
├── results/                           # 实验结果
├── src/                               # 源代码
│   ├── 01_data_exploration.py         # 数据探索
│   ├── 02_feature_engineering.py      # 特征工程
│   ├── 03_model_training.py           # 模型训练
│   ├── 04_model_evaluation.py         # 模型评估
│   ├── main.py                        # 主程序入口
│   └── run.py                         # 简化启动脚本
├── teams/                             # 团队协作
│   ├── tasks.md                       # 任务分工
│   └── ideas.md                       # 项目思路
├── requirements.txt                   # 依赖管理
└── README.md                          # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
# 完整安装（推荐）
pip install -r requirements.txt

# 最小安装
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. 运行项目

```bash
# 运行完整流程
python src/run.py

# 或使用主程序
python src/main.py

# 运行单个步骤
python src/main.py --step data_exploration
```

### 3. 查看结果

- **可视化图表**: `photos/` 目录
- **实验报告**: `report/` 目录  
- **模型文件**: `models/` 目录
- **详细数据**: `results/` 目录


## 团队协作

本项目由6人团队协作完成，采用明确的分工和有效的协作机制，确保项目高质量完成。

## 联系信息

- **项目负责人**: bosprimigenious
- **邮箱**: [bosprimigenious@foxmail.com](mailto:bosprimigenious@foxmail.com)
- **GitHub**: [Big_data_applications](https://github.com/bosprimigenious/Big_data_applications)

## 许可证

本项目仅用于学术研究和教育目的。

---

*最后更新: 2025年9月27日*
