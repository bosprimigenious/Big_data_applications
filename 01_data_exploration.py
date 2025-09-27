"""
Human vs AI Generated Essays 数据分析项目
01 - 数据探索与可视化

作者：bosprimigenious
生成日期：2025-9-26-13:44
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 导入图表样式配置和增强可视化
from plot_style import setup_plot_style, create_beautiful_plot, save_beautiful_plot, apply_theme
from enhanced_visualization import create_comprehensive_analysis_charts
from results_exporter import ResultsExporter

# 设置中文字体和图表样式
setup_plot_style()
apply_theme('professional')

def load_data():
    """加载数据集"""
    data_path = os.path.join('data', 'balanced_ai_human_prompts.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    print(f"正在加载数据: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"数据加载完成，形状: {df.shape}")
    
    return df

def basic_data_exploration(df):
    """基础数据探索"""
    print("\n" + "="*50)
    print("基础数据探索")
    print("="*50)
    
    # 基本信息
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 数据类型
    print("\n数据类型:")
    print(df.dtypes)
    
    # 缺失值检查
    print("\n缺失值统计:")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("无缺失值")
    
    # 标签分布
    print("\n标签分布:")
    label_counts = df['generated'].value_counts()
    print(label_counts)
    print(f"人类文本 (0): {label_counts[0]} ({label_counts[0]/len(df)*100:.1f}%)")
    print(f"AI文本 (1): {label_counts[1]} ({label_counts[1]/len(df)*100:.1f}%)")
    
    return label_counts

def text_statistics(df):
    """文本统计分析"""
    print("\n" + "="*50)
    print("文本统计分析")
    print("="*50)
    
    # 计算文本长度统计
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['sentence_count'] = df['text'].str.split('.').str.len()
    df['avg_word_length'] = df['text'].str.replace(' ', '').str.len() / df['word_count']
    
    # 处理可能的除零错误
    df['avg_word_length'] = df['avg_word_length'].fillna(0)
    
    # 按标签分组统计
    stats_by_label = df.groupby('generated').agg({
        'text_length': ['mean', 'std', 'min', 'max'],
        'word_count': ['mean', 'std', 'min', 'max'],
        'sentence_count': ['mean', 'std', 'min', 'max'],
        'avg_word_length': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    print("按标签分组的文本统计:")
    print(stats_by_label)
    
    return df

def visualize_text_features(df):
    """可视化文本特征分布 - 使用简单有效的中文字体设置方法"""
    print("\n" + "="*50)
    print("生成文本特征可视化")
    print("="*50)
    
    # 直接设置中文字体 - 使用和chinese_font_test.png相同的方法
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✓ 直接设置中文字体: Microsoft YaHei")
    
    # 创建美观的图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 15))
    fig.suptitle('文本特征分析 - 人类 vs AI 文本对比', fontsize=18, fontweight='bold', y=0.98)
    
    # 特征列表
    features = ['text_length', 'word_count', 'sentence_count', 'avg_word_length']
    feature_names = ['文本长度(字符)', '词数', '句数', '平均词长']
    
    # 创建子图
    for i, (feature, name) in enumerate(zip(features, feature_names), 1):
        ax = axes[(i-1)//3, (i-1)%3]
        
        # 箱线图
        sns.boxplot(data=df, x='generated', y=feature, ax=ax)
        ax.set_title(f'{name} 分布对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('文本类型 (0=人类, 1=AI)', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        
        # 添加统计信息
        human_mean = df[df['generated'] == 0][feature].mean()
        ai_mean = df[df['generated'] == 1][feature].mean()
        ax.text(0.5, 0.95, f'人类均值: {human_mean:.1f}\nAI均值: {ai_mean:.1f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 标签分布饼图
    ax = axes[1, 1]
    label_counts = df['generated'].value_counts()
    labels = ['人类文本', 'AI文本']
    colors = ['#2E86AB', '#A23B72']
    wedges, texts, autotexts = ax.pie(label_counts.values, labels=labels, colors=colors, 
                                     autopct='%1.1f%%', startangle=90)
    ax.set_title('数据集标签分布', fontsize=14, fontweight='bold')
    
    # 美化饼图
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 文本长度分布直方图
    ax = axes[1, 2]
    ax.hist(df[df['generated'] == 0]['text_length'], bins=30, alpha=0.7, 
            label='人类文本', color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax.hist(df[df['generated'] == 1]['text_length'], bins=30, alpha=0.7, 
            label='AI文本', color='#A23B72', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('文本长度(字符)', fontsize=12)
    ax.set_ylabel('频次', fontsize=12)
    ax.set_title('文本长度分布对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    # 美化所有子图
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    
    # 直接保存图表 - 不使用save_beautiful_plot函数
    plt.savefig('text_features_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ 文本特征分析图表已保存: text_features_analysis.png")
    
    plt.close()  # 关闭图表以释放内存

def sample_text_analysis(df):
    """样本文本分析"""
    print("\n" + "="*50)
    print("样本文本分析")
    print("="*50)
    
    # 随机选择几个样本
    np.random.seed(42)
    human_samples = df[df['generated'] == 0].sample(3)
    ai_samples = df[df['generated'] == 1].sample(3)
    
    print("人类文本样本:")
    for i, (idx, row) in enumerate(human_samples.iterrows(), 1):
        print(f"\n样本 {i} (长度: {row['text_length']} 字符):")
        print(f"内容: {row['text'][:200]}...")
    
    print("\n" + "-"*50)
    print("AI文本样本:")
    for i, (idx, row) in enumerate(ai_samples.iterrows(), 1):
        print(f"\n样本 {i} (长度: {row['text_length']} 字符):")
        print(f"内容: {row['text'][:200]}...")

def save_processed_data(df):
    """保存处理后的数据"""
    output_path = os.path.join('data', 'processed_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\n处理后的数据已保存到: {output_path}")

def main():
    """主函数"""
    print("Human vs AI Generated Essays 数据分析项目")
    print("01 - 数据探索与可视化")
    print("="*50)
    
    try:
        # 加载数据
        df = load_data()
        
        # 基础数据探索
        label_counts = basic_data_exploration(df)
        
        # 文本统计分析
        df = text_statistics(df)
        
        # 可视化文本特征
        visualize_text_features(df)
        
        # 样本文本分析
        sample_text_analysis(df)
        
        # 生成增强可视化图表
        print("\n" + "="*50)
        print("生成增强可视化图表")
        print("="*50)
        create_comprehensive_analysis_charts(df)
        
        # 导出结果
        print("\n" + "="*50)
        print("导出分析结果")
        print("="*50)
        exporter = ResultsExporter()
        exporter.export_data_exploration_results(df)
        
        # 保存处理后的数据
        save_processed_data(df)
        
        print("\n" + "="*50)
        print("数据探索完成！")
        print("="*50)
        
    except Exception as e:
        print(f"错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
