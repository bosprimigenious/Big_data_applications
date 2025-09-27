"""
Human vs AI Generated Essays 数据分析项目
03 - 模型训练与比较

作者：bosprimigenious
生成日期：2025-9-26-13:44
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# 导入图表样式配置
from plot_style import setup_plot_style, create_beautiful_plot, save_beautiful_plot

# 设置中文字体与图像参数
setup_plot_style()

def load_feature_data():
    """加载特征数据"""
    print("正在加载特征数据...")
    
    # 加载特征数据
    features_path = os.path.join('data', 'features_data.csv')
    if not os.path.exists(features_path):
        print(f"特征数据文件不存在: {features_path}")
        print("请先运行 02_feature_engineering.py")
        return None, None, None
    
    df = pd.read_csv(features_path, encoding='utf-8')
    print(f"特征数据加载完成，形状: {df.shape}")
    
    # 加载TF-IDF矩阵
    tfidf_path = os.path.join('data', 'tfidf_matrix.npz')
    if not os.path.exists(tfidf_path):
        print(f"TF-IDF矩阵文件不存在: {tfidf_path}")
        return df, None, None
    
    tfidf_data = np.load(tfidf_path)
    tfidf_matrix = tfidf_data['matrix']
    print(f"TF-IDF矩阵加载完成，形状: {tfidf_matrix.shape}")
    
    # 加载特征名称
    feature_names_path = os.path.join('data', 'feature_names.pkl')
    if not os.path.exists(feature_names_path):
        print(f"特征名称文件不存在: {feature_names_path}")
        return df, tfidf_matrix, None
    
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    print(f"特征名称加载完成，数量: {len(feature_names)}")
    
    return df, tfidf_matrix, feature_names

def prepare_data(df, tfidf_matrix):
    """准备训练数据"""
    print("\n" + "="*50)
    print("准备训练数据")
    print("="*50)
    
    # 选择数值特征（排除文本列和标签列）
    exclude_cols = ['text', 'cleaned_text', 'advanced_cleaned_text', 'generated']
    numeric_features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    print(f"选择的数值特征数量: {len(numeric_features)}")
    print(f"特征列表: {numeric_features[:10]}...")  # 显示前10个特征
    
    # 准备特征矩阵
    X_numeric = df[numeric_features].fillna(0)
    X_tfidf = tfidf_matrix
    y = df['generated']
    
    print(f"数值特征矩阵形状: {X_numeric.shape}")
    print(f"TF-IDF特征矩阵形状: {X_tfidf.shape}")
    print(f"标签分布: {y.value_counts().to_dict()}")
    
    return X_numeric, X_tfidf, y, numeric_features

def create_models():
    """创建模型管道"""
    print("\n" + "="*50)
    print("创建模型管道")
    print("="*50)
    
    models = {
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ]),
        
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        
        'SVM': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', SVC(kernel='linear', random_state=42, probability=True))
        ]),
        
        'Gradient Boosting': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
    }
    
    print(f"创建了 {len(models)} 个模型:")
    for name in models.keys():
        print(f"  - {name}")
    
    return models

def train_and_evaluate_models(models, X_text, y):
    """训练和评估模型"""
    print("\n" + "="*50)
    print("训练和评估模型")
    print("="*50)
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        try:
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 评估指标
            results[name] = {
                'model': model,
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision': precision_score(y_test, y_pred),
                'test_recall': recall_score(y_test, y_pred),
                'test_f1': f1_score(y_test, y_pred),
                'test_auc': roc_auc_score(y_test, y_pred_proba),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  测试集准确率: {results[name]['test_accuracy']:.4f}")
            print(f"  测试集F1分数: {results[name]['test_f1']:.4f}")
            print(f"  测试集AUC: {results[name]['test_auc']:.4f}")
            
        except Exception as e:
            print(f"  训练失败: {e}")
            continue
    
    return results, X_test, y_test

def hyperparameter_tuning(best_model_name, models, X_text, y):
    """超参数调优"""
    print(f"\n" + "="*50)
    print(f"对最佳模型 {best_model_name} 进行超参数调优")
    print("="*50)
    
    # 定义参数网格
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    }
    
    if best_model_name not in param_grids:
        print(f"模型 {best_model_name} 没有预定义的参数网格")
        return models[best_model_name]
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 网格搜索
    grid_search = GridSearchCV(
        models[best_model_name],
        param_grids[best_model_name],
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("开始网格搜索...")
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def visualize_results(results):
    """可视化结果"""
    print("\n" + "="*50)
    print("生成结果可视化")
    print("="*50)
    
    # 准备数据
    model_names = list(results.keys())
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. 模型性能对比
    ax = axes[0]
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[model][metric] for model in model_names]
        ax.bar(x + i * width, values, width, label=name)
    
    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('模型性能对比')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 交叉验证结果
    ax = axes[1]
    cv_means = [results[model]['cv_accuracy_mean'] for model in model_names]
    cv_stds = [results[model]['cv_accuracy_std'] for model in model_names]
    
    ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
    ax.set_ylabel('交叉验证准确率')
    ax.set_title('交叉验证结果')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 3-6. 各模型的混淆矩阵
    for i, (model_name, result) in enumerate(results.items()):
        if i >= 4:  # 最多显示4个模型的混淆矩阵
            break
        
        ax = axes[i + 2]
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name} 混淆矩阵')
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
    
    plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()
    
    # ROC曲线 - 优化版本
    plt.figure(figsize=(12, 10))
    
    # 定义更好的颜色和线型
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # 绘制每个模型的ROC曲线
    for i, (model_name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        auc = result['test_auc']
        
        # 使用不同的颜色、线型和标记
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        
        # 每隔几个点显示一个标记，避免过于密集
        step = max(1, len(fpr) // 20)
        plt.plot(fpr[::step], tpr[::step], 
                color=color, linestyle=linestyle, marker=marker, 
                markersize=6, linewidth=2.5, alpha=0.8,
                label=f'{model_name} (AUC = {auc:.3f})')
        
        # 绘制完整的曲线（不带标记）
        plt.plot(fpr, tpr, color=color, linestyle=linestyle, 
                linewidth=1.5, alpha=0.6)
    
    # 随机分类器基准线
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='随机分类器 (AUC = 0.500)')
    
    # 设置坐标轴
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)', fontsize=12, fontweight='bold')
    plt.ylabel('真正率 (True Positive Rate)', fontsize=12, fontweight='bold')
    plt.title('ROC曲线对比 - 模型性能评估', fontsize=14, fontweight='bold', pad=20)
    
    # 改进图例
    plt.legend(loc='lower right', fontsize=10, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.9)
    
    # 改进网格
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.1, linestyle='--', linewidth=0.3, which='minor')
    
    # 添加对角线区域说明
    plt.text(0.6, 0.2, '对角线以上：模型优于随机\n对角线以下：模型劣于随机', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7),
             fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()

def save_models(results, best_model):
    """保存模型"""
    print("\n" + "="*50)
    print("保存模型")
    print("="*50)
    
    # 创建模型保存目录
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存所有模型
    for name, result in results.items():
        model_path = os.path.join(model_dir, f'{name.lower().replace(" ", "_")}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        print(f"模型 {name} 已保存到: {model_path}")
    
    # 保存最佳模型
    if best_model is not None:
        best_model_path = os.path.join(model_dir, 'best_model.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"最佳模型已保存到: {best_model_path}")
    
    # 保存结果摘要
    results_summary = {}
    for name, result in results.items():
        results_summary[name] = {
            'cv_accuracy_mean': result['cv_accuracy_mean'],
            'cv_accuracy_std': result['cv_accuracy_std'],
            'test_accuracy': result['test_accuracy'],
            'test_precision': result['test_precision'],
            'test_recall': result['test_recall'],
            'test_f1': result['test_f1'],
            'test_auc': result['test_auc']
        }
    
    summary_path = os.path.join(model_dir, 'results_summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(results_summary, f)
    print(f"结果摘要已保存到: {summary_path}")

def print_results_summary(results):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("模型训练结果摘要")
    print("="*80)
    
    # 创建结果表格
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            '模型': name,
            '交叉验证准确率': f"{result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}",
            '测试准确率': f"{result['test_accuracy']:.4f}",
            '精确率': f"{result['test_precision']:.4f}",
            '召回率': f"{result['test_recall']:.4f}",
            'F1分数': f"{result['test_f1']:.4f}",
            'AUC': f"{result['test_auc']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    # 保存到报告
    try:
        with open('report.txt', 'a', encoding='utf-8') as f:
            f.write('\n==== 模型训练结果摘要 ====' + '\n')
            f.write(summary_df.to_string(index=False) + '\n')
    except Exception:
        pass
    
    # 找出最佳模型
    best_model_by_auc = max(results.items(), key=lambda x: x[1]['test_auc'])
    best_model_by_f1 = max(results.items(), key=lambda x: x[1]['test_f1'])
    
    print(f"\n最佳AUC模型: {best_model_by_auc[0]} (AUC: {best_model_by_auc[1]['test_auc']:.4f})")
    print(f"最佳F1模型: {best_model_by_f1[0]} (F1: {best_model_by_f1[1]['test_f1']:.4f})")

def main():
    """主函数"""
    print("Human vs AI Generated Essays 数据分析项目")
    print("03 - 模型训练与比较")
    print("="*50)
    
    try:
        # 加载数据
        df, tfidf_matrix, feature_names = load_feature_data()
        if df is None:
            return False
        
        # 准备数据
        X_numeric, X_tfidf, y, numeric_features = prepare_data(df, tfidf_matrix)
        
        # 使用TF-IDF特征进行文本分类
        X_text = df['advanced_cleaned_text'].fillna('')
        
        # 创建模型
        models = create_models()
        
        # 训练和评估模型
        results, X_test, y_test = train_and_evaluate_models(models, X_text, y)
        
        if not results:
            print("没有模型训练成功")
            return False
        
        # 打印结果摘要
        print_results_summary(results)
        
        # 可视化结果
        visualize_results(results)
        
        # 超参数调优（选择最佳模型）
        best_model_name = max(results.items(), key=lambda x: x[1]['test_auc'])[0]
        print(f"\n选择 {best_model_name} 进行超参数调优...")
        
        best_model = hyperparameter_tuning(best_model_name, models, X_text, y)
        
        # 保存模型
        save_models(results, best_model)
        
        print("\n" + "="*50)
        print("模型训练完成！")
        print("="*50)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
