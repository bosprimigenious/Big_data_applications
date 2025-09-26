"""
测试脚本 - 验证项目设置
"""

import os
import sys

def test_file_structure():
    """测试文件结构"""
    print("检查项目文件结构...")
    
    required_files = [
        '01_data_exploration.py',
        '02_feature_engineering.py', 
        '03_model_training.py',
        '04_model_evaluation.py',
        'main.py',
        'requirements.txt',
        'README.md',
        'tasks.md',
        'ideas.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file}")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_data_directory():
    """测试数据目录"""
    print("\n检查数据目录...")
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"  ✗ 数据目录不存在: {data_dir}")
        print("  请创建数据目录并放入 balanced_ai_human_prompts.csv 文件")
        return False
    
    print(f"  ✓ 数据目录存在: {data_dir}")
    
    # 检查数据文件
    data_file = os.path.join(data_dir, 'balanced_ai_human_prompts.csv')
    if os.path.exists(data_file):
        print(f"  ✓ 数据文件存在: {data_file}")
        return True
    else:
        print(f"  ✗ 数据文件不存在: {data_file}")
        print("  请将数据集文件放入 data/ 目录中")
        return False

def test_imports():
    """测试导入"""
    print("\n测试Python模块导入...")
    
    # 测试标准库
    try:
        import os, sys, pickle, warnings
        print("  ✓ 标准库导入成功")
    except ImportError as e:
        print(f"  ✗ 标准库导入失败: {e}")
        return False
    
    # 测试核心依赖
    core_deps = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn']
    missing_deps = []
    
    for dep in core_deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} (需要安装)")
            missing_deps.append(dep)
    
    # 测试可选依赖
    optional_deps = ['nltk', 'textstat', 'tqdm']
    missing_optional = []
    
    print("\n测试可选依赖...")
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ⚠ {dep} (可选，未安装)")
            missing_optional.append(dep)
    
    if missing_deps:
        print(f"\n缺少核心依赖: {', '.join(missing_deps)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n可选依赖未安装: {', '.join(missing_optional)}")
        print("某些功能可能不可用，建议安装: pip install nltk textstat tqdm")
    
    return True

def main():
    """主函数"""
    print("Human vs AI Generated Essays 项目设置测试")
    print("=" * 50)
    
    # 测试文件结构
    files_ok = test_file_structure()
    
    # 测试数据目录
    data_ok = test_data_directory()
    
    # 测试导入
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    print("测试结果:")
    print(f"  文件结构: {'✓' if files_ok else '✗'}")
    print(f"  数据目录: {'✓' if data_ok else '✗'}")
    print(f"  依赖导入: {'✓' if imports_ok else '✗'}")
    
    if files_ok and data_ok and imports_ok:
        print("\n🎉 项目设置完成，可以开始运行！")
        print("\n运行命令:")
        print("  python main.py                    # 完整流程")
        print("  python main.py --check-only       # 环境检查")
        print("  python main.py --step data_exploration  # 单步骤运行")
        return True
    else:
        print("\n❌ 项目设置不完整，请解决上述问题")
        return False

if __name__ == "__main__":
    main()
