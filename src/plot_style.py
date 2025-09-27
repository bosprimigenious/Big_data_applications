"""
图表样式配置 - 改进美观度和中文字体支持
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_fonts():
    """设置中文字体 - 使用简单有效的方法"""
    # 直接设置中文字体 - 这是经过验证的有效方法
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("✓ 使用简单方法设置中文字体: Microsoft YaHei")
    return 'Microsoft YaHei'

def setup_plot_style():
    """设置图表样式 - 使用简单有效的方法"""
    # 直接设置中文字体 - 简单有效
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置其他样式参数
    plt.rcParams.update({
        'font.size': 12,
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'figure.autolayout': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': True,
        'legend.shadow': True,
    })
    
    print("✓ 图表样式设置完成，使用中文字体: Microsoft YaHei")
    return 'Microsoft YaHei'
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 设置颜色主题
    colors = {
        'primary': '#2E86AB',      # 蓝色
        'secondary': '#A23B72',    # 紫红色
        'success': '#F18F01',      # 橙色
        'warning': '#C73E1D',      # 红色
        'info': '#6A994E',         # 绿色
        'light': '#F8F9FA',        # 浅灰色
        'dark': '#212529',         # 深灰色
    }
    
    return colors

def create_beautiful_plot(figsize=(12, 8), title="", xlabel="", ylabel=""):
    """创建美观的图表"""
    colors = setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置标题
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 设置坐标轴标签
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    return fig, ax, colors

def save_beautiful_plot(fig, filename, dpi=300, bbox_inches='tight'):
    """保存美观的图表"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # 保存图表
    fig.savefig(
        filename, 
        dpi=dpi, 
        bbox_inches=bbox_inches, 
        pad_inches=0.3,
        facecolor='white',
        edgecolor='none'
    )
    print(f"✓ 图表已保存: {filename}")

def create_comparison_plot(data, title="对比分析", figsize=(15, 10)):
    """创建对比分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    colors = setup_plot_style()
    
    # 美化所有子图
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
    
    return fig, axes, colors

def add_watermark(fig, text="AI vs Human Text Analysis", alpha=0.1):
    """添加水印"""
    fig.text(0.5, 0.5, text, 
             fontsize=50, color='gray', alpha=alpha,
             ha='center', va='center', rotation=30,
             transform=fig.transFigure)

# 预定义的图表主题
THEMES = {
    'professional': {
        'colors': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
        'style': 'whitegrid',
        'palette': 'husl'
    },
    'colorful': {
        'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
        'style': 'darkgrid',
        'palette': 'Set2'
    },
    'minimal': {
        'colors': ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12'],
        'style': 'white',
        'palette': 'muted'
    }
}

def apply_theme(theme_name='professional'):
    """应用预定义主题"""
    if theme_name in THEMES:
        theme = THEMES[theme_name]
        sns.set_style(theme['style'])
        sns.set_palette(theme['palette'])
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=theme['colors'])
        print(f"✓ 应用主题: {theme_name}")
    else:
        print(f"⚠ 未知主题: {theme_name}")

if __name__ == "__main__":
    # 测试字体和样式设置
    setup_plot_style()
    print("图表样式设置完成！")
