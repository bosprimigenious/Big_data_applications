"""
结果导出模块 - 将所有数字结果导出到CSV和TXT文件
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

class ResultsExporter:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_dir = f"results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def export_data_exploration_results(self, df):
        """导出数据探索结果"""
        print(f"正在导出数据探索结果到 {self.results_dir}/...")
        
        # 1. 基础统计信息
        basic_stats = {
            '数据形状': df.shape,
            '列名': list(df.columns),
            '数据类型': df.dtypes.to_dict(),
            '缺失值统计': df.isnull().sum().to_dict(),
            '标签分布': df['generated'].value_counts().to_dict()
        }
        
        # 保存基础统计信息
        with open(f"{self.results_dir}/basic_statistics.txt", 'w', encoding='utf-8') as f:
            f.write("=== 数据探索基础统计信息 ===\n\n")
            for key, value in basic_stats.items():
                f.write(f"{key}: {value}\n")
        
        # 2. 文本统计特征
        if 'text_length' in df.columns:
            text_stats = df.groupby('generated').agg({
                'text_length': ['mean', 'std', 'min', 'max', 'median'],
                'word_count': ['mean', 'std', 'min', 'max', 'median'],
                'sentence_count': ['mean', 'std', 'min', 'max', 'median'],
                'avg_word_length': ['mean', 'std', 'min', 'max', 'median']
            }).round(4)
            
            # 保存到CSV
            text_stats.to_csv(f"{self.results_dir}/text_statistics.csv", encoding='utf-8')
            
            # 保存到TXT
            with open(f"{self.results_dir}/text_statistics.txt", 'w', encoding='utf-8') as f:
                f.write("=== 文本统计特征分析 ===\n\n")
                f.write(text_stats.to_string())
        
        # 3. 详细数据导出
        df.to_csv(f"{self.results_dir}/processed_data.csv", index=False, encoding='utf-8')
        
        print(f"✓ 数据探索结果已导出到 {self.results_dir}/")
    
    def export_feature_engineering_results(self, df, correlations):
        """导出特征工程结果"""
        print(f"正在导出特征工程结果到 {self.results_dir}/...")
        
        # 1. 特征相关性分析
        if correlations is not None:
            correlations_df = pd.DataFrame({
                '特征名称': correlations.index,
                '相关性': correlations.values,
                '绝对相关性': correlations.abs().values
            }).sort_values('绝对相关性', ascending=False)
            
            correlations_df.to_csv(f"{self.results_dir}/feature_correlations.csv", 
                                 index=False, encoding='utf-8')
        
        # 2. 特征统计信息
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col not in ['generated']]
        
        feature_stats = df[numeric_features].describe().round(4)
        feature_stats.to_csv(f"{self.results_dir}/feature_statistics.csv", encoding='utf-8')
        
        # 3. 按标签分组的特征统计
        if 'generated' in df.columns:
            grouped_stats = df.groupby('generated')[numeric_features].agg(['mean', 'std', 'min', 'max']).round(4)
            grouped_stats.to_csv(f"{self.results_dir}/grouped_feature_statistics.csv", encoding='utf-8')
        
        # 4. 特征重要性排序
        if correlations is not None:
            importance_ranking = pd.DataFrame({
                '排名': range(1, len(correlations) + 1),
                '特征名称': correlations.index,
                '重要性得分': correlations.abs().values,
                '相关性': correlations.values
            }).sort_values('重要性得分', ascending=False)
            
            importance_ranking.to_csv(f"{self.results_dir}/feature_importance_ranking.csv", 
                                    index=False, encoding='utf-8')
        
        print(f"✓ 特征工程结果已导出到 {self.results_dir}/")
    
    def export_model_training_results(self, results):
        """导出模型训练结果"""
        print(f"正在导出模型训练结果到 {self.results_dir}/...")
        
        # 1. 模型性能对比表
        performance_data = []
        for model_name, result in results.items():
            performance_data.append({
                '模型名称': model_name,
                '交叉验证准确率均值': result.get('cv_accuracy_mean', 0),
                '交叉验证准确率标准差': result.get('cv_accuracy_std', 0),
                '测试集准确率': result.get('test_accuracy', 0),
                '测试集精确率': result.get('test_precision', 0),
                '测试集召回率': result.get('test_recall', 0),
                '测试集F1分数': result.get('test_f1', 0),
                '测试集AUC': result.get('test_auc', 0)
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(f"{self.results_dir}/model_performance_comparison.csv", 
                            index=False, encoding='utf-8')
        
        # 2. 模型性能排名
        performance_df_sorted = performance_df.sort_values('测试集AUC', ascending=False)
        performance_df_sorted['排名'] = range(1, len(performance_df_sorted) + 1)
        performance_df_sorted.to_csv(f"{self.results_dir}/model_performance_ranking.csv", 
                                   index=False, encoding='utf-8')
        
        # 3. 详细性能报告
        with open(f"{self.results_dir}/model_performance_report.txt", 'w', encoding='utf-8') as f:
            f.write("=== 模型训练性能报告 ===\n\n")
            f.write(f"训练时间: {self.timestamp}\n")
            f.write(f"模型数量: {len(results)}\n\n")
            
            for i, (model_name, result) in enumerate(results.items(), 1):
                f.write(f"模型 {i}: {model_name}\n")
                f.write("-" * 50 + "\n")
                f.write(f"交叉验证准确率: {result.get('cv_accuracy_mean', 0):.4f} ± {result.get('cv_accuracy_std', 0):.4f}\n")
                f.write(f"测试集准确率: {result.get('test_accuracy', 0):.4f}\n")
                f.write(f"测试集精确率: {result.get('test_precision', 0):.4f}\n")
                f.write(f"测试集召回率: {result.get('test_recall', 0):.4f}\n")
                f.write(f"测试集F1分数: {result.get('test_f1', 0):.4f}\n")
                f.write(f"测试集AUC: {result.get('test_auc', 0):.4f}\n\n")
        
        # 4. 最佳模型信息
        best_model = max(results.items(), key=lambda x: x[1].get('test_auc', 0))
        best_model_info = {
            '最佳模型名称': best_model[0],
            'AUC得分': best_model[1].get('test_auc', 0),
            '准确率': best_model[1].get('test_accuracy', 0),
            'F1分数': best_model[1].get('test_f1', 0)
        }
        
        with open(f"{self.results_dir}/best_model_info.txt", 'w', encoding='utf-8') as f:
            f.write("=== 最佳模型信息 ===\n\n")
            for key, value in best_model_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ 模型训练结果已导出到 {self.results_dir}/")
    
    def export_model_evaluation_results(self, results, feature_importance, comparison_df, error_cases):
        """导出模型评估结果"""
        print(f"正在导出模型评估结果到 {self.results_dir}/...")
        
        # 1. 详细评估指标
        evaluation_data = []
        for model_name, result in results.items():
            evaluation_data.append({
                '模型名称': model_name,
                '准确率': result.get('accuracy', 0),
                '精确率': result.get('precision', 0),
                '召回率': result.get('recall', 0),
                'F1分数': result.get('f1', 0),
                'AUC': result.get('auc', 0),
                '平均精确率': result.get('avg_precision', 0)
            })
        
        evaluation_df = pd.DataFrame(evaluation_data)
        evaluation_df.to_csv(f"{self.results_dir}/model_evaluation_metrics.csv", 
                           index=False, encoding='utf-8')
        
        # 2. 特征重要性分析
        if feature_importance:
            if isinstance(feature_importance, list) and len(feature_importance) > 0:
                if isinstance(feature_importance[0], tuple):
                    # 处理 (feature, importance) 格式
                    importance_data = []
                    for i, (feature, importance) in enumerate(feature_importance):
                        importance_data.append({
                            '排名': i + 1,
                            '特征名称': feature,
                            '重要性得分': importance
                        })
                    
                    importance_df = pd.DataFrame(importance_data)
                    importance_df.to_csv(f"{self.results_dir}/feature_importance_analysis.csv", 
                                       index=False, encoding='utf-8')
        
        # 3. 文本差异分析
        if comparison_df is not None:
            comparison_df.to_csv(f"{self.results_dir}/text_differences_analysis.csv", 
                               index=False, encoding='utf-8')
        
        # 4. 错误案例分析
        if error_cases:
            false_positives, false_negatives, high_conf_errors = error_cases
            
            # 错误统计
            error_stats = {
                '假阳性数量': len(false_positives),
                '假阴性数量': len(false_negatives),
                '高置信度错误数量': len(high_conf_errors),
                '总错误数量': len(false_positives) + len(false_negatives)
            }
            
            with open(f"{self.results_dir}/error_analysis.txt", 'w', encoding='utf-8') as f:
                f.write("=== 错误案例分析 ===\n\n")
                for key, value in error_stats.items():
                    f.write(f"{key}: {value}\n")
                
                if len(high_conf_errors) > 0:
                    f.write(f"\n高置信度错误案例详情:\n")
                    f.write("-" * 50 + "\n")
                    for i, (idx, row) in enumerate(high_conf_errors.head(10).iterrows()):
                        f.write(f"案例 {i+1}:\n")
                        f.write(f"真实标签: {'AI' if row['generated'] == 1 else '人类'}\n")
                        f.write(f"预测标签: {'AI' if row['predicted'] == 1 else '人类'}\n")
                        f.write(f"预测置信度: {row['prediction_prob']:.3f}\n")
                        f.write(f"文本长度: {row['text_length']} 字符\n")
                        f.write(f"文本前100字符: {row['text'][:100]}...\n\n")
        
        # 5. 综合评估报告
        with open(f"{self.results_dir}/comprehensive_evaluation_report.txt", 'w', encoding='utf-8') as f:
            f.write("=== 综合评估报告 ===\n\n")
            f.write(f"评估时间: {self.timestamp}\n")
            f.write(f"评估模型数量: {len(results)}\n\n")
            
            # 模型性能排名
            f.write("模型性能排名 (按AUC排序):\n")
            f.write("-" * 50 + "\n")
            sorted_results = sorted(results.items(), key=lambda x: x[1].get('auc', 0), reverse=True)
            for i, (model_name, result) in enumerate(sorted_results, 1):
                f.write(f"{i}. {model_name}: AUC={result.get('auc', 0):.4f}, "
                       f"F1={result.get('f1', 0):.4f}, 准确率={result.get('accuracy', 0):.4f}\n")
            
            # 关键发现
            f.write(f"\n关键发现:\n")
            f.write("-" * 50 + "\n")
            if comparison_df is not None and len(comparison_df) > 0:
                f.write("人类与AI文本的主要差异特征:\n")
                for _, row in comparison_df.head(5).iterrows():
                    f.write(f"- {row['特征']}: 相对差异 {row['相对差异']:.3f}\n")
            
            if feature_importance and len(feature_importance) > 0:
                f.write(f"\n最重要的区分特征:\n")
                for feature, importance in feature_importance[:5]:
                    f.write(f"- {feature}: {importance:.4f}\n")
        
        print(f"✓ 模型评估结果已导出到 {self.results_dir}/")
    
    def export_summary_statistics(self, df, results):
        """导出汇总统计信息"""
        print(f"正在导出汇总统计信息到 {self.results_dir}/...")
        
        # 1. 项目概览
        project_overview = {
            '项目名称': 'Human vs AI Generated Essays 数据分析项目',
            '分析时间': self.timestamp,
            '数据规模': f"{df.shape[0]} 个样本, {df.shape[1]} 个特征",
            '标签分布': df['generated'].value_counts().to_dict(),
            '模型数量': len(results) if results else 0,
            '最佳模型': max(results.items(), key=lambda x: x[1].get('test_auc', 0))[0] if results else 'N/A'
        }
        
        # 保存项目概览
        with open(f"{self.results_dir}/project_overview.txt", 'w', encoding='utf-8') as f:
            f.write("=== 项目概览 ===\n\n")
            for key, value in project_overview.items():
                f.write(f"{key}: {value}\n")
        
        # 2. 关键指标汇总
        if results:
            key_metrics = {
                '最高准确率': max([r.get('test_accuracy', 0) for r in results.values()]),
                '最高AUC': max([r.get('test_auc', 0) for r in results.values()]),
                '最高F1分数': max([r.get('test_f1', 0) for r in results.values()]),
                '平均准确率': np.mean([r.get('test_accuracy', 0) for r in results.values()]),
                '平均AUC': np.mean([r.get('test_auc', 0) for r in results.values()])
            }
            
            with open(f"{self.results_dir}/key_metrics_summary.txt", 'w', encoding='utf-8') as f:
                f.write("=== 关键指标汇总 ===\n\n")
                for key, value in key_metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
        
        # 3. 数据质量报告
        data_quality = {
            '数据完整性': f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%",
            '标签平衡性': f"人类文本: {df['generated'].value_counts()[0]} ({df['generated'].value_counts()[0]/len(df)*100:.1f}%), "
                         f"AI文本: {df['generated'].value_counts()[1]} ({df['generated'].value_counts()[1]/len(df)*100:.1f}%)",
            '文本长度范围': f"{df['text_length'].min()} - {df['text_length'].max()} 字符",
            '平均文本长度': f"{df['text_length'].mean():.1f} 字符"
        }
        
        with open(f"{self.results_dir}/data_quality_report.txt", 'w', encoding='utf-8') as f:
            f.write("=== 数据质量报告 ===\n\n")
            for key, value in data_quality.items():
                f.write(f"{key}: {value}\n")
        
        # 4. 创建结果索引文件
        self.create_results_index()
        
        print(f"✓ 汇总统计信息已导出到 {self.results_dir}/")
    
    def create_results_index(self):
        """创建结果文件索引"""
        files = []
        for file in os.listdir(self.results_dir):
            if file.endswith(('.csv', '.txt')):
                file_path = os.path.join(self.results_dir, file)
                file_size = os.path.getsize(file_path)
                files.append({
                    '文件名': file,
                    '文件大小': f"{file_size} 字节",
                    '文件类型': 'CSV数据文件' if file.endswith('.csv') else 'TXT报告文件'
                })
        
        files_df = pd.DataFrame(files)
        files_df.to_csv(f"{self.results_dir}/results_index.csv", index=False, encoding='utf-8')
        
        # 创建README文件
        with open(f"{self.results_dir}/README.txt", 'w', encoding='utf-8') as f:
            f.write("=== 结果文件说明 ===\n\n")
            f.write("本目录包含Human vs AI Generated Essays项目的所有分析结果\n\n")
            f.write("文件类型说明:\n")
            f.write("- CSV文件: 包含数值数据和统计结果，可用Excel打开\n")
            f.write("- TXT文件: 包含分析报告和文字说明\n\n")
            f.write("主要文件说明:\n")
            f.write("- basic_statistics.txt: 数据基础统计信息\n")
            f.write("- text_statistics.csv: 文本特征统计结果\n")
            f.write("- feature_correlations.csv: 特征相关性分析\n")
            f.write("- model_performance_comparison.csv: 模型性能对比\n")
            f.write("- model_evaluation_metrics.csv: 模型评估指标\n")
            f.write("- feature_importance_analysis.csv: 特征重要性分析\n")
            f.write("- comprehensive_evaluation_report.txt: 综合评估报告\n")
            f.write("- project_overview.txt: 项目概览\n")
            f.write("- key_metrics_summary.txt: 关键指标汇总\n\n")
            f.write(f"生成时间: {self.timestamp}\n")
    
    def export_all_results(self, df, correlations=None, training_results=None, 
                          evaluation_results=None, feature_importance=None, 
                          comparison_df=None, error_cases=None):
        """导出所有结果"""
        print(f"\n开始导出所有分析结果...")
        print(f"结果将保存到目录: {self.results_dir}")
        
        # 导出各个阶段的结果
        self.export_data_exploration_results(df)
        self.export_feature_engineering_results(df, correlations)
        
        if training_results:
            self.export_model_training_results(training_results)
        
        if evaluation_results:
            self.export_model_evaluation_results(evaluation_results, feature_importance, 
                                               comparison_df, error_cases)
        
        # 导出汇总信息
        self.export_summary_statistics(df, training_results or evaluation_results)
        
        print(f"\n✓ 所有结果已成功导出到 {self.results_dir}/")
        print(f"共生成 {len([f for f in os.listdir(self.results_dir) if f.endswith(('.csv', '.txt'))])} 个结果文件")
        
        return self.results_dir

