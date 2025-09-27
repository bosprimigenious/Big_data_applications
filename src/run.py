"""
AI文本检测分析项目 - 简化启动脚本
"""
import sys
import os

# 设置中文字体（在导入matplotlib之前）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 强制设置中文字体
chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
available_fonts = [f.name for f in fm.fontManager.ttflist]

selected_font = None
for font in chinese_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams.update({
        'font.sans-serif': [selected_font, 'DejaVu Sans', 'Arial'],
        'font.family': 'sans-serif',
        'axes.unicode_minus': False,
    })
    print(f"✓ 设置中文字体: {selected_font}")

# 设置编码
try:
    if os.name == 'nt':
        os.system('chcp 65001 > nul')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    print("✓ 设置中文编码")
except:
    pass

# 导入主程序
from main import main

if __name__ == "__main__":
    print("AI文本检测分析项目")
    print("="*50)
    print("功能：")
    print("1. 数据探索与可视化")
    print("2. 特征工程")
    print("3. 模型训练与比较")
    print("4. 模型评估与深度分析")
    print("5. 生成研究报告")
    print("="*50)
    
    # 运行主程序
    success = main()
    
    if success:
        print("\n🎉 分析完成！")
        print("生成的文件：")
        print("- *.png: 分析图表")
        print("- results_*/: 详细结果数据")
        print("- research_report.md: 研究报告")
        print("- executive_summary.md: 执行摘要")
    else:
        print("\n❌ 分析过程中出现错误")
        sys.exit(1)
