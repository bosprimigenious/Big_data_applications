"""
Human vs AI Generated Essays 数据分析项目
04 - 模型评估与深度分析

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
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 导入图表样式配置
from plot_style import setup_plot_style, create_beautiful_plot, save_beautiful_plot

# 设置中文字体
setup_plot_style()

def load_models_and_data():
    """加载模型和数据"""
    print("正在加载模型和数据...")
    
    # 加载特征数据
    features_path = os.path.join('data', 'features_data.csv')
    if not os.path.exists(features_path):
        print(f"特征数据文件不存在: {features_path}")
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
    
    # 加载训练好的模型
    models = {}
    model_dir = 'models'
    
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
            model_path = os.path.join(model_dir, model_file)
            try:
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"模型 {model_name} 加载成功")
            except Exception as e:
                print(f"加载模型 {model_name} 失败: {e}")
    else:
        print("模型目录不存在，请先运行 03_model_training.py")
        return df, tfidf_matrix, feature_names
    
    return df, tfidf_matrix, feature_names, models

def prepare_test_data(df, tfidf_matrix):
    """准备测试数据"""
    print("\n" + "="*50)
    print("准备测试数据")
    print("="*50)
    
    # 使用清理后的文本
    X_text = df['advanced_cleaned_text'].fillna('')
    y = df['generated']
    
    # 分割数据（与训练时保持一致）
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"测试集标签分布: {y_test.value_counts().to_dict()}")
    
    return X_test, y_test

def evaluate_models(models, X_test, y_test):
    """评估所有模型"""
    print("\n" + "="*50)
    print("评估所有模型")
    print("="*50)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n评估模型: {name}")
        
        try:
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # 分类报告
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'avg_precision': avg_precision,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': report
            }
            
            print(f"  准确率: {accuracy:.4f}")
            print(f"  精确率: {precision:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  F1分数: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  平均精确率: {avg_precision:.4f}")

            # 追加写入报告
            try:
                with open('report.txt', 'a', encoding='utf-8') as f:
                    f.write(f"\n==== 模型评估: {name} ====\n")
                    f.write(f"准确率: {accuracy:.4f}\n精确率: {precision:.4f}\n召回率: {recall:.4f}\nF1分数: {f1:.4f}\nAUC: {auc:.4f}\n平均精确率: {avg_precision:.4f}\n")
            except Exception:
                pass
            
        except Exception as e:
            print(f"  评估失败: {e}")
            continue
    
    return results

def analyze_feature_importance(results, X_test, feature_names):
    """分析特征重要性"""
    print("\n" + "="*50)
    print("分析特征重要性")
    print("="*50)
    
    # 选择最佳模型（基于AUC）
    best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
    best_model = results[best_model_name]['model']
    
    print(f"使用最佳模型 {best_model_name} 进行特征重要性分析")
    
    # 提取特征重要性
    if hasattr(best_model.named_steps['classifier'], 'coef_'):
        # 逻辑回归或SVM
        coefficients = best_model.named_steps['classifier'].coef_[0]
        feature_importance = list(zip(feature_names, coefficients))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n{best_model_name} 特征重要性分析:")
        print("\nAI文本特征 (正系数，前20个):")
        # 修复稀疏矩阵比较问题
        ai_features = []
        for feature, coef in feature_importance:
            # 处理稀疏矩阵
            if hasattr(coef, 'toarray'):  # 稀疏矩阵
                coef_val = float(coef.toarray()[0, 0])
            elif hasattr(coef, '__len__'):  # 如果是数组
                try:
                    coef_val = float(coef[0])
                except (TypeError, IndexError):
                    coef_val = float(coef)
            else:  # 如果是标量
                coef_val = float(coef)
            if coef_val > 0:
                ai_features.append((feature, coef_val))
                if len(ai_features) >= 20:
                    break
        
        for feature, coef in ai_features:
            print(f"  {feature}: {coef:.4f}")
        
        print("\n人类文本特征 (负系数，前20个):")
        human_features = []
        for feature, coef in feature_importance:
            # 处理稀疏矩阵
            if hasattr(coef, 'toarray'):  # 稀疏矩阵
                coef_val = float(coef.toarray()[0, 0])
            elif hasattr(coef, '__len__'):  # 如果是数组
                try:
                    coef_val = float(coef[0])
                except (TypeError, IndexError):
                    coef_val = float(coef)
            else:  # 如果是标量
                coef_val = float(coef)
            if coef_val < 0:
                human_features.append((feature, coef_val))
                if len(human_features) >= 20:
                    break
        
        for feature, coef in human_features:
            print(f"  {feature}: {coef:.4f}")
        
        # 可视化特征重要性 - 优化版本
        # 直接设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建更大的图形，分为两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # 处理前15个特征
        top_features = feature_importance[:15]
        
        # 修复稀疏矩阵处理问题
        features = []
        coeffs = []
        for feature, coef in top_features:
            # 处理稀疏矩阵
            if hasattr(coef, 'toarray'):  # 稀疏矩阵
                coef_val = float(coef.toarray()[0, 0])
            elif hasattr(coef, '__len__'):  # 如果是数组
                try:
                    coef_val = float(coef[0])
                except (TypeError, IndexError):
                    coef_val = float(coef)
            else:  # 如果是标量
                coef_val = float(coef)
            features.append(feature)
            coeffs.append(coef_val)
        
        # 为AI特征（正系数）和人类特征（负系数）使用不同颜色
        colors = ['#FF6B6B' if c > 0 else '#4ECDC4' for c in coeffs]
        abs_coeffs = [abs(c) for c in coeffs]
        
        # 左图：水平条形图
        bars1 = ax1.barh(range(len(features)), abs_coeffs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, fontsize=10)
        ax1.set_xlabel('特征重要性 (|系数|)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{best_model_name} - Top 15 最重要特征', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars1, abs_coeffs)):
            ax1.text(bar.get_width() + max(abs_coeffs)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # 右图：饼图显示AI vs 人类特征比例
        ai_features = sum(1 for c in coeffs if c > 0)
        human_features = len(coeffs) - ai_features
        
        labels = ['AI特征 (正系数)', '人类特征 (负系数)']
        sizes = [ai_features, human_features]
        colors_pie = ['#FF6B6B', '#4ECDC4']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('特征类型分布', fontsize=14, fontweight='bold')
        
        # 添加图例说明
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#FF6B6B', alpha=0.8, label='AI特征 (正系数)'),
            plt.Rectangle((0,0),1,1, facecolor='#4ECDC4', alpha=0.8, label='人类特征 (负系数)')
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # 添加统计信息
        total_importance = sum(abs_coeffs)
        ai_importance = sum(abs(c) for c in coeffs if c > 0)
        human_importance = sum(abs(c) for c in coeffs if c < 0)
        
        info_text = f'总重要性: {total_importance:.3f}\nAI特征重要性: {ai_importance:.3f}\n人类特征重要性: {human_importance:.3f}'
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.show()
        
    elif hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        # 随机森林或梯度提升
        importances = best_model.named_steps['classifier'].feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{best_model_name} 特征重要性分析 (前20个):")
        for feature, importance in feature_importance[:20]:
            print(f"  {feature}: {importance:.4f}")
        
        # 可视化特征重要性 - 优化版本
        # 直接设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建更大的图形，分为两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # 处理前20个特征
        top_features = feature_importance[:20]
        features, importances = zip(*top_features)
        
        # 使用渐变色
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        
        # 左图：水平条形图
        bars1 = ax1.barh(range(len(features)), importances, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, fontsize=10)
        ax1.set_xlabel('特征重要性', fontsize=12, fontweight='bold')
        ax1.set_title(f'{best_model_name} - Top 20 最重要特征', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars1, importances)):
            ax1.text(bar.get_width() + max(importances)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # 右图：累积重要性图
        cumulative_importance = np.cumsum(importances)
        total_importance = sum(importances)
        cumulative_percentage = (cumulative_importance / total_importance) * 100
        
        ax2.plot(range(1, len(features)+1), cumulative_percentage, 'o-', linewidth=2, markersize=6, color='#2E86AB')
        ax2.fill_between(range(1, len(features)+1), cumulative_percentage, alpha=0.3, color='#2E86AB')
        ax2.set_xlabel('特征数量', fontsize=12, fontweight='bold')
        ax2.set_ylabel('累积重要性 (%)', fontsize=12, fontweight='bold')
        ax2.set_title('累积特征重要性', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # 添加重要阈值线
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80%重要性阈值')
        ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90%重要性阈值')
        ax2.legend()
        
        # 添加统计信息
        top_5_importance = sum(importances[:5])
        top_10_importance = sum(importances[:10])
        
        info_text = f'前5个特征重要性: {top_5_importance:.3f}\n前10个特征重要性: {top_10_importance:.3f}\n总重要性: {total_importance:.3f}'
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.show()
    
    return feature_importance

def analyze_text_differences(df):
    """分析人类和AI文本的差异"""
    print("\n" + "="*50)
    print("分析人类和AI文本差异")
    print("="*50)
    
    human_texts = df[df['generated'] == 0]
    ai_texts = df[df['generated'] == 1]
    
    # 统计特征对比
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col not in ['generated']]
    
    comparison_data = []
    for feature in numeric_features:
        human_mean = human_texts[feature].mean()
        ai_mean = ai_texts[feature].mean()
        human_std = human_texts[feature].std()
        ai_std = ai_texts[feature].std()
        
        comparison_data.append({
            '特征': feature,
            '人类均值': human_mean,
            'AI均值': ai_mean,
            '人类标准差': human_std,
            'AI标准差': ai_std,
            '差异': abs(human_mean - ai_mean),
            '相对差异': abs(human_mean - ai_mean) / max(abs(human_mean), abs(ai_mean), 1e-8)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('相对差异', ascending=False)
    
    print("人类与AI文本特征差异分析 (前20个):")
    print(comparison_df.head(20).to_string(index=False))
    
    # 可视化差异最大的特征
    plt.figure(figsize=(15, 10))
    top_features = comparison_df.head(10)['特征'].tolist()
    
    for i, feature in enumerate(top_features):
        plt.subplot(2, 5, i + 1)
        plt.hist(human_texts[feature], bins=30, alpha=0.7, label='人类文本', color='lightblue')
        plt.hist(ai_texts[feature], bins=30, alpha=0.7, label='AI文本', color='lightcoral')
        plt.xlabel(feature)
        plt.ylabel('频次')
        plt.title(f'{feature} 分布对比')
        plt.legend()
    
    plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])
    plt.savefig('text_differences_analysis.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()
    
    return comparison_df

def analyze_error_cases(results, X_test, y_test, df):
    """分析错误案例"""
    print("\n" + "="*50)
    print("分析错误案例")
    print("="*50)
    
    # 选择最佳模型
    best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
    best_result = results[best_model_name]
    
    print(f"使用最佳模型 {best_model_name} 分析错误案例")
    
    y_pred = best_result['y_pred']
    y_pred_proba = best_result['y_pred_proba']
    
    # 获取测试集原始数据
    test_indices = X_test.index
    test_data = df.loc[test_indices].copy()
    test_data['predicted'] = y_pred
    test_data['prediction_prob'] = y_pred_proba
    
    # 误分类案例
    false_positives = test_data[(test_data['generated'] == 0) & (test_data['predicted'] == 1)]
    false_negatives = test_data[(test_data['generated'] == 1) & (test_data['predicted'] == 0)]
    
    print(f"假阳性案例数量: {len(false_positives)}")
    print(f"假阴性案例数量: {len(false_negatives)}")
    
    # 分析高置信度错误预测
    high_conf_errors = test_data[
        ((test_data['generated'] != test_data['predicted']) & 
         (test_data['prediction_prob'] > 0.8)) |
        ((test_data['generated'] != test_data['predicted']) & 
         (test_data['prediction_prob'] < 0.2))
    ]
    
    print(f"高置信度错误预测数量: {len(high_conf_errors)}")
    
    # 展示一些错误案例
    if len(high_conf_errors) > 0:
        print("\n高置信度错误预测案例:")
        for idx, row in high_conf_errors.head(5).iterrows():
            print(f"\n案例 {idx}:")
            print(f"真实标签: {'AI' if row['generated'] == 1 else '人类'}")
            print(f"预测标签: {'AI' if row['predicted'] == 1 else '人类'}")
            print(f"预测置信度: {row['prediction_prob']:.3f}")
            print(f"文本长度: {row['text_length']} 字符")
            print(f"文本前200字符: {row['text'][:200]}...")
            print("-" * 80)
    
    # 分析错误案例的特征分布
    if len(false_positives) > 0 and len(false_negatives) > 0:
        plt.figure(figsize=(15, 5))
        
        # 假阳性案例特征分析
        plt.subplot(1, 3, 1)
        fp_features = ['text_length', 'word_count', 'sentence_count']
        fp_data = false_positives[fp_features].mean()
        plt.bar(fp_features, fp_data.values)
        plt.title('假阳性案例特征均值')
        plt.xticks(rotation=45)
        
        # 假阴性案例特征分析
        plt.subplot(1, 3, 2)
        fn_data = false_negatives[fp_features].mean()
        plt.bar(fp_features, fn_data.values)
        plt.title('假阴性案例特征均值')
        plt.xticks(rotation=45)
        
        # 对比
        plt.subplot(1, 3, 3)
        x = np.arange(len(fp_features))
        width = 0.35
        plt.bar(x - width/2, fp_data.values, width, label='假阳性')
        plt.bar(x + width/2, fn_data.values, width, label='假阴性')
        plt.xlabel('特征')
        plt.ylabel('均值')
        plt.title('错误案例特征对比')
        plt.xticks(x, fp_features, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('error_case_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return false_positives, false_negatives, high_conf_errors

def generate_comprehensive_report(results, feature_importance, comparison_df, error_cases):
    """生成综合评估报告"""
    print("\n" + "="*80)
    print("综合评估报告")
    print("="*80)
    
    # 清空/创建报告文件头
    with open('report.txt', 'w', encoding='utf-8') as f:
        f.write('Human vs AI Generated Essays 项目报告\n')
        f.write('===============================\n')

    # 模型性能排名
    print("1. 模型性能排名:")
    performance_ranking = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    for i, (name, result) in enumerate(performance_ranking, 1):
        print(f"   {i}. {name}: AUC={result['auc']:.4f}, F1={result['f1']:.4f}, 准确率={result['accuracy']:.4f}")
    
    # 最佳模型详细分析
    best_model_name, best_result = performance_ranking[0]
    print(f"\n2. 最佳模型 {best_model_name} 详细分析:")
    print(f"   - 准确率: {best_result['accuracy']:.4f}")
    print(f"   - 精确率: {best_result['precision']:.4f}")
    print(f"   - 召回率: {best_result['recall']:.4f}")
    print(f"   - F1分数: {best_result['f1']:.4f}")
    print(f"   - AUC: {best_result['auc']:.4f}")
    print(f"   - 平均精确率: {best_result['avg_precision']:.4f}")
    
    # 关键发现
    print("\n3. 关键发现:")
    print("   - 人类和AI文本在以下特征上存在显著差异:")
    top_differences = comparison_df.head(5)
    for _, row in top_differences.iterrows():
        print(f"     * {row['特征']}: 相对差异 {row['相对差异']:.3f}")
    
    # 特征重要性
    if feature_importance:
        print("\n   - 最重要的区分特征:")
        for feature, importance in feature_importance[:5]:
            # 处理稀疏矩阵格式化问题
            if hasattr(importance, 'toarray'):  # 稀疏矩阵
                importance_val = float(importance.toarray()[0, 0])
            else:
                importance_val = float(importance)
            print(f"     * {feature}: {importance_val:.4f}")
    
    # 错误分析
    false_positives, false_negatives, high_conf_errors = error_cases
    print(f"\n   - 错误分析:")
    print(f"     * 假阳性率: {len(false_positives)/len(results[best_model_name]['y_pred'])*100:.2f}%")
    print(f"     * 假阴性率: {len(false_negatives)/len(results[best_model_name]['y_pred'])*100:.2f}%")
    print(f"     * 高置信度错误: {len(high_conf_errors)} 个")
    
    # 应用建议
    print("\n4. 应用建议:")
    print("   - 该模型可用于教育领域的AI文本检测")
    print("   - 建议结合人工审核处理高置信度错误案例")
    print("   - 可考虑集成多个模型提高检测准确性")
    print("   - 需要定期更新模型以适应新的AI生成技术")

def main():
    """主函数"""
    print("Human vs AI Generated Essays 数据分析项目")
    print("04 - 模型评估与深度分析")
    print("="*50)
    
    try:
        # 加载模型和数据
        data = load_models_and_data()
        if len(data) == 3:
            df, tfidf_matrix, feature_names = data
            models = {}
        else:
            df, tfidf_matrix, feature_names, models = data
        
        if not models:
            print("没有找到训练好的模型，请先运行 03_model_training.py")
            return False
        
        # 准备测试数据
        X_test, y_test = prepare_test_data(df, tfidf_matrix)
        
        # 评估模型
        results = evaluate_models(models, X_test, y_test)
        
        if not results:
            print("没有模型评估成功")
            return False
        
        # 分析特征重要性
        feature_importance = analyze_feature_importance(results, X_test, feature_names)
        
        # 分析文本差异
        comparison_df = analyze_text_differences(df)
        
        # 分析错误案例
        error_cases = analyze_error_cases(results, X_test, y_test, df)
        
        # 生成综合报告
        generate_comprehensive_report(results, feature_importance, comparison_df, error_cases)
        
        print("\n" + "="*50)
        print("模型评估完成！")
        print("="*50)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
