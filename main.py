"""
Human vs AI Generated Essays 数据分析项目
主程序入口

作者：bosprimigenious
生成日期：2025-9-26-13:44
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

def check_dependencies():
    """检查依赖库"""
    print("检查依赖库...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn'
    ]
    
    optional_packages = [
        'nltk', 'textstat'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"  ✗ {package} (必需)")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"  ✗ {package} (可选)")
    
    if missing_required:
        print(f"\n错误: 缺少必需的依赖库: {', '.join(missing_required)}")
        print("请使用以下命令安装:")
        print(f"pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n警告: 缺少可选的依赖库: {', '.join(missing_optional)}")
        print("某些功能可能不可用，建议安装:")
        print(f"pip install {' '.join(missing_optional)}")
    
    return True

def check_data_files():
    """检查数据文件"""
    print("\n检查数据文件...")
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"  ✗ 数据目录不存在: {data_dir}")
        return False
    
    required_files = [
        'balanced_ai_human_prompts.csv'
    ]
    
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (必需)")
            return False
    
    return True

def run_data_exploration():
    """运行数据探索"""
    print("\n" + "="*60)
    print("步骤 1: 数据探索与可视化")
    print("="*60)
    
    try:
        from 01_data_exploration import main as explore_main
        return explore_main()
    except Exception as e:
        print(f"数据探索失败: {e}")
        return False

def run_feature_engineering():
    """运行特征工程"""
    print("\n" + "="*60)
    print("步骤 2: 特征工程")
    print("="*60)
    
    try:
        from 02_feature_engineering import main as feature_main
        return feature_main()
    except Exception as e:
        print(f"特征工程失败: {e}")
        return False

def run_model_training():
    """运行模型训练"""
    print("\n" + "="*60)
    print("步骤 3: 模型训练与比较")
    print("="*60)
    
    try:
        from 03_model_training import main as training_main
        return training_main()
    except Exception as e:
        print(f"模型训练失败: {e}")
        return False

def run_model_evaluation():
    """运行模型评估"""
    print("\n" + "="*60)
    print("步骤 4: 模型评估与深度分析")
    print("="*60)
    
    try:
        from 04_model_evaluation import main as evaluation_main
        return evaluation_main()
    except Exception as e:
        print(f"模型评估失败: {e}")
        return False

def run_complete_pipeline():
    """运行完整流程"""
    print("Human vs AI Generated Essays 数据分析项目")
    print("完整流程执行")
    print("="*60)
    
    # 检查环境
    if not check_dependencies():
        return False
    
    if not check_data_files():
        print("\n请确保数据文件存在于 data/ 目录中")
        return False
    
    # 执行各个步骤
    steps = [
        ("数据探索", run_data_exploration),
        ("特征工程", run_feature_engineering),
        ("模型训练", run_model_training),
        ("模型评估", run_model_evaluation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n开始执行: {step_name}")
        if not step_func():
            print(f"步骤 '{step_name}' 执行失败，程序终止")
            return False
        print(f"步骤 '{step_name}' 执行成功")
    
    print("\n" + "="*60)
    print("🎉 完整流程执行成功！")
    print("="*60)
    print("\n生成的文件:")
    print("  - data/processed_data.csv: 处理后的数据")
    print("  - data/features_data.csv: 特征数据")
    print("  - data/tfidf_matrix.npz: TF-IDF矩阵")
    print("  - data/feature_names.pkl: 特征名称")
    print("  - models/: 训练好的模型")
    print("  - *.png: 可视化图表")
    
    return True

def run_single_step(step):
    """运行单个步骤"""
    print(f"Human vs AI Generated Essays 数据分析项目")
    print(f"执行步骤: {step}")
    print("="*60)
    
    # 检查环境
    if not check_dependencies():
        return False
    
    if step in ['feature_engineering', 'model_training', 'model_evaluation']:
        if not check_data_files():
            print("\n请确保数据文件存在于 data/ 目录中")
            return False
    
    # 执行指定步骤
    step_functions = {
        'data_exploration': run_data_exploration,
        'feature_engineering': run_feature_engineering,
        'model_training': run_model_training,
        'model_evaluation': run_model_evaluation
    }
    
    if step not in step_functions:
        print(f"未知步骤: {step}")
        print(f"可用步骤: {', '.join(step_functions.keys())}")
        return False
    
    return step_functions[step]()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Human vs AI Generated Essays 数据分析项目',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py                    # 运行完整流程
  python main.py --step data_exploration    # 只运行数据探索
  python main.py --step model_training      # 只运行模型训练
  python main.py --help             # 显示帮助信息

可用步骤:
  data_exploration     - 数据探索与可视化
  feature_engineering  - 特征工程
  model_training       - 模型训练与比较
  model_evaluation     - 模型评估与深度分析
        """
    )
    
    parser.add_argument(
        '--step', 
        choices=['data_exploration', 'feature_engineering', 'model_training', 'model_evaluation'],
        help='指定要执行的单个步骤'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='只检查环境和依赖，不执行任何步骤'
    )
    
    args = parser.parse_args()
    
    # 只检查环境
    if args.check_only:
        print("Human vs AI Generated Essays 数据分析项目")
        print("环境检查")
        print("="*60)
        
        deps_ok = check_dependencies()
        data_ok = check_data_files()
        
        if deps_ok and data_ok:
            print("\n✓ 环境检查通过，可以运行项目")
            return True
        else:
            print("\n✗ 环境检查失败，请解决上述问题后重试")
            return False
    
    # 执行步骤
    if args.step:
        return run_single_step(args.step)
    else:
        return run_complete_pipeline()

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
