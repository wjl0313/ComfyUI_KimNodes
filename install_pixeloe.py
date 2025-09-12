#!/usr/bin/env python3
"""
ComfyUI_KimNodes - pixeloe 依赖安装脚本
"""

import subprocess
import sys
import os

def install_pixeloe():
    """安装 pixeloe 包"""
    print("正在安装 pixeloe 包...")
    
    try:
        # 首先尝试卸载可能存在的旧版本
        print("尝试卸载已有的 pixeloe...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "pixeloe", "-y"], 
                      capture_output=True, text=True)
        
        # 安装最新版本
        print("安装最新版本的 pixeloe...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "pixeloe"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ pixeloe 安装成功！")
            return True
        else:
            print(f"✗ 安装失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ 安装过程中出错: {e}")
        return False

def test_import():
    """测试导入"""
    print("\n测试 pixeloe 导入...")
    
    try:
        import pixeloe
        print("✓ pixeloe 基础导入成功")
        
        # 测试 torch 版本
        try:
            from pixeloe.torch.pixelize import pixelize
            print("✓ pixeloe.torch.pixelize 导入成功")
        except ImportError as e:
            print(f"✗ torch 版本导入失败: {e}")
        
        # 测试 legacy 版本
        try:
            from pixeloe.legacy.pixelize import pixelize
            print("✓ pixeloe.legacy.pixelize 导入成功")
        except ImportError as e:
            print(f"✗ legacy 版本导入失败: {e}")
            
        return True
        
    except ImportError as e:
        print(f"✗ pixeloe 导入失败: {e}")
        return False

def main():
    print("=" * 50)
    print("ComfyUI_KimNodes - pixeloe 依赖安装器")
    print("=" * 50)
    
    # 显示当前 Python 环境信息
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 安装 pixeloe
    if install_pixeloe():
        # 测试导入
        if test_import():
            print("\n" + "=" * 50)
            print("✓ 安装成功！现在您可以使用 Pixelate_Filter 节点了。")
            print("请重启 ComfyUI 以确保更改生效。")
        else:
            print("\n" + "=" * 50)
            print("⚠️  安装完成但导入测试失败。")
            print("请检查您的 Python 环境。")
    else:
        print("\n" + "=" * 50)
        print("✗ 安装失败。请手动运行:")
        print("pip install pixeloe")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 