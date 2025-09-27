"""
设置终端编码以支持中文显示
"""
import os
import sys
import locale

def setup_chinese_encoding():
    """设置中文编码支持"""
    try:
        # Windows系统编码设置
        if os.name == 'nt':
            # 设置控制台代码页为UTF-8
            os.system('chcp 65001 > nul')
            
            # 设置环境变量
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            os.environ['LANG'] = 'zh_CN.UTF-8'
            os.environ['LC_ALL'] = 'zh_CN.UTF-8'
            
            print("✓ Windows中文编码设置完成")
        
        # 设置Python默认编码
        if hasattr(sys, 'setdefaultencoding'):
            sys.setdefaultencoding('utf-8')
        
        # 设置locale
        try:
            locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
        except:
            try:
                locale.setlocale(locale.LC_ALL, 'Chinese_China.utf8')
            except:
                pass
        
        print("✓ Python编码设置完成")
        return True
        
    except Exception as e:
        print(f"编码设置失败: {e}")
        return False

if __name__ == "__main__":
    setup_chinese_encoding()
