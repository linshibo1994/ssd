#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据分析与预测系统测试脚本
"""

import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """测试基本功能"""
    print("=== 双色球数据分析与预测系统测试 ===\n")
    
    # 测试数据文件是否存在
    data_files = ["data/ssq_data.csv", "data/ssq_data_all.csv"]
    existing_files = [f for f in data_files if os.path.exists(f)]
    
    if not existing_files:
        print("未找到数据文件，请先运行以下命令获取数据：")
        print("python src/main.py crawl")
        print("或")
        print("python src/main.py crawl --all")
        return False
    
    print(f"找到数据文件: {', '.join(existing_files)}")
    
    # 测试导入模块
    try:
        from analyzer import SSQAnalyzer
        from advanced_analyzer import SSQAdvancedAnalyzer
        from utils import format_ssq_numbers, generate_random_numbers
        print("✓ 所有模块导入成功")
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False
    
    # 测试基础分析器
    try:
        data_file = existing_files[0]
        analyzer = SSQAnalyzer(data_file=data_file, output_dir="data")
        if analyzer.load_data():
            print(f"✓ 基础分析器加载成功，数据量: {len(analyzer.data)}期")
        else:
            print("✗ 基础分析器加载失败")
            return False
    except Exception as e:
        print(f"✗ 基础分析器测试失败: {e}")
        return False
    
    # 测试高级分析器
    try:
        advanced_analyzer = SSQAdvancedAnalyzer(data_file)
        if advanced_analyzer.load_data():
            print(f"✓ 高级分析器加载成功，数据量: {len(advanced_analyzer.data)}期")
        else:
            print("✗ 高级分析器加载失败")
            return False
    except Exception as e:
        print(f"✗ 高级分析器测试失败: {e}")
        return False
    
    # 测试号码生成
    try:
        red_balls, blue_ball = generate_random_numbers()
        formatted = format_ssq_numbers(red_balls, blue_ball)
        print(f"✓ 随机号码生成成功: {formatted}")
    except Exception as e:
        print(f"✗ 号码生成测试失败: {e}")
        return False
    
    # 测试马尔可夫链预测
    try:
        predictions = advanced_analyzer.predict_multiple_by_markov_chain(count=2, explain=False)
        print(f"✓ 马尔可夫链预测成功，生成{len(predictions)}注号码")
        for i, (reds, blue) in enumerate(predictions):
            formatted = format_ssq_numbers(reds, blue)
            print(f"  第{i+1}注: {formatted}")
    except Exception as e:
        print(f"✗ 马尔可夫链预测测试失败: {e}")
        return False
    
    print("\n✓ 所有基本功能测试通过！")
    return True

def show_usage_examples():
    """显示使用示例"""
    print("\n=== 使用示例 ===")
    print("\n1. 数据爬取:")
    print("   python src/main.py crawl                    # 爬取最近300期")
    print("   python src/main.py crawl --all              # 爬取所有历史数据")
    
    print("\n2. 数据分析:")
    print("   python src/main.py analyze                  # 基础分析")
    print("   python src/main.py advanced --method all    # 高级分析")
    
    print("\n3. 智能预测:")
    print("   python src/main.py markov_predict --count 5 # 马尔可夫链预测5注")
    print("   python src/main.py predict --method ensemble # 集成方法预测")
    
    print("\n4. 其他功能:")
    print("   python src/main.py latest                   # 查看最新开奖")
    print("   python src/main.py generate                 # 生成号码")

if __name__ == "__main__":
    success = test_basic_functionality()
    show_usage_examples()
    
    if success:
        print("\n🎉 项目测试完成，系统运行正常！")
    else:
        print("\n❌ 项目测试失败，请检查环境配置。")
