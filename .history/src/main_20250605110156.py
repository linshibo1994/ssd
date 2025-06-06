#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球项目主程序
整合爬虫、分析和高级分析功能，提供命令行界面
"""

import os
import sys
import argparse
from datetime import datetime

# 导入项目模块
from cwl_crawler import SSQCWLCrawler
from analyzer import SSQAnalyzer
from utils import (
    validate_ssq_data,
    generate_random_numbers,
    generate_smart_numbers,
    format_ssq_numbers,
    get_latest_draw
)

# 导入高级分析模块
try:
    # 尝试使用绝对导入
    from src.advanced_analyzer import AdvancedSSQAnalyzer
    ADVANCED_ANALYZER_AVAILABLE = True
except ImportError:
    try:
        # 尝试使用相对导入
        from .advanced_analyzer import AdvancedSSQAnalyzer
        ADVANCED_ANALYZER_AVAILABLE = True
    except ImportError:
        try:
            # 尝试直接导入（如果在同一目录下）
            from advanced_analyzer import AdvancedSSQAnalyzer
            ADVANCED_ANALYZER_AVAILABLE = True
        except ImportError:
            ADVANCED_ANALYZER_AVAILABLE = False
            print("警告: 高级分析模块未安装，高级分析功能将不可用")
            print("请安装所需依赖: pip install -r requirements.txt")


def get_project_root():
    """获取项目根目录"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录为上一级目录
    return os.path.dirname(current_dir)


def get_data_dir():
    """获取数据目录"""
    return os.path.join(get_project_root(), "data")


def get_data_file():
    """获取数据文件路径"""
    return os.path.join(get_data_dir(), "ssq_data.csv")


def crawl_cwl_data(args):
    """从中国福利彩票官方网站爬取数据"""
    print("开始从中国福利彩票官方网站爬取双色球历史数据...")
    
    # 创建爬虫实例
    crawler = SSQCWLCrawler(data_dir=get_data_dir())
    
    # 爬取历史数据
    count = args.count if args.count else 300
    results = crawler.get_history_data(count=count)
    
    # 保存数据
    if results:
        print(f"共获取{len(results)}期双色球开奖结果")
        crawler.save_to_csv(results)
    else:
        print("未获取到数据")


def analyze_data(args):
    """分析数据"""
    data_file = get_data_file()
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    # 验证数据
    if not validate_ssq_data(data_file):
        print("数据验证失败，请重新获取数据")
        return
    
    print("开始分析双色球历史数据...")
    
    # 创建分析器实例
    analyzer = SSQAnalyzer(data_file=data_file, output_dir=get_data_dir())
    
    # 运行分析
    if analyzer.run_analysis():
        print(f"分析结果已保存到 {get_data_dir()} 目录")
    else:
        print("分析失败")


def generate_numbers(args):
    """生成号码"""
    data_file = get_data_file()
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}，将使用纯随机方式生成号码")
        method = "random"
    else:
        method = args.method
    
    print(f"使用 {method} 方法生成双色球号码...")
    
    # 生成号码
    if method == "random" or not os.path.exists(data_file):
        red_balls, blue_ball = generate_random_numbers()
    else:
        red_balls, blue_ball = generate_smart_numbers(data_file, method=method)
    
    # 打印号码
    print("\n生成的双色球号码:")
    print(format_ssq_numbers(red_balls, blue_ball))
    print()
    
    # 获取最新一期开奖结果进行对比
    if os.path.exists(data_file):
        issue, date, winning_reds, winning_blue = get_latest_draw(data_file)
        if issue:
            from utils import calculate_prize
            
            prize_level = calculate_prize(red_balls, blue_ball, winning_reds, winning_blue)
            
            print(f"最新开奖结果: {format_ssq_numbers(winning_reds, winning_blue)}")
            
            if prize_level:
                print(f"恭喜！中得{prize_level}等奖！")
            else:
                print("很遗憾，未中奖")


def show_latest(args):
    """显示最新开奖结果"""
    data_file = get_data_file()
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    # 获取最新一期开奖结果
    issue, date, red_balls, blue_ball = get_latest_draw(data_file)
    
    if issue:
        print(f"\n最新一期({issue})开奖结果:")
        print(f"开奖日期: {date}")
        print(format_ssq_numbers(red_balls, blue_ball))
    else:
        print("获取最新开奖结果失败")


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='双色球数据爬取、分析和号码生成工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 爬取命令
    crawl_parser = subparsers.add_parser('crawl', help='爬取双色球历史数据')
    crawl_parser.add_argument('--count', type=int, help="爬取的数据条数，默认300期")
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析双色球历史数据')
    
    # 高级分析命令
    advanced_parser = subparsers.add_parser('advanced', help='使用高级统计和机器学习方法分析双色球数据')
    advanced_parser.add_argument('--method', choices=['all', 'stats', 'probability', 'frequency', 'decision_tree', 'cycle', 'bayes'], 
                               default='all', help='高级分析方法')
    advanced_parser.add_argument('--periods', type=int, default=300, help='分析期数，默认为300期')
    advanced_parser.add_argument('--save_model', action='store_true', help='是否保存训练好的模型')
    
    # 智能预测命令
    predict_parser = subparsers.add_parser('predict', help='使用高级分析模型预测双色球号码')
    predict_parser.add_argument('--method', choices=['stats', 'probability', 'decision_tree', 'bayes', 'ensemble'], 
                              default='ensemble', help='预测方法')
    predict_parser.add_argument('--count', type=int, default=1, help='生成注数，默认为1注')
    predict_parser.add_argument('--explain', action='store_true', help='是否解释预测结果')
    
    # 生成命令
    generate_parser = subparsers.add_parser('generate', help='生成双色球号码')
    generate_parser.add_argument('--method', choices=['random', 'frequency', 'trend', 'hybrid'], 
                                default='hybrid', help='生成方法，默认为hybrid')
    
    # 最新开奖命令
    latest_parser = subparsers.add_parser('latest', help='显示最新开奖结果')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有指定子命令，显示帮助信息
    if not args.command:
        parser.print_help()
        return
    
    # 执行对应的子命令
    if args.command == "crawl":
        crawl_cwl_data(args)
    elif args.command == "analyze":
        analyze_data(args)
    elif args.command == "advanced":
        if not ADVANCED_ANALYZER_AVAILABLE:
            print("错误: 高级分析模块不可用，请确保已安装所需依赖")
            return
        
        method = args.method
        periods = args.periods
        save_model = args.save_model
        
        advanced_analyzer = AdvancedSSQAnalyzer(data_file)
        if method == "all":
            advanced_analyzer.run_all_analysis(periods=periods, save_model=save_model)
        elif method == "stats":
            advanced_analyzer.run_statistical_analysis(periods=periods)
        elif method == "probability":
            advanced_analyzer.run_probability_analysis(periods=periods)
        elif method == "frequency":
            advanced_analyzer.run_frequency_analysis(periods=periods)
        elif method == "decision_tree":
            advanced_analyzer.run_decision_tree_analysis(periods=periods, save_model=save_model)
        elif method == "cycle":
            advanced_analyzer.run_cycle_analysis(periods=periods)
        elif method == "bayes":
            advanced_analyzer.run_bayesian_analysis(periods=periods, save_model=save_model)
    elif args.command == 'predict':
        if not ADVANCED_ANALYZER_AVAILABLE:
            print("错误: 高级分析模块不可用，请确保已安装所需依赖")
            return
        
        method = args.method
        count = args.count
        explain = args.explain
        
        advanced_analyzer = AdvancedSSQAnalyzer(data_file)
        
        # 预测号码
        numbers = advanced_analyzer.predict_numbers(method=method, explain=explain)
        
        # 格式化预测号码
        red_balls = numbers[:6]
        blue_ball = numbers[6]
        formatted_numbers = format_ssq_numbers(red_balls, blue_ball)
        
        print(f"\n预测号码: {formatted_numbers}")
        
        # 如果需要生成多注
        if count > 1:
            print(f"\n额外预测{count-1}注:")
            for i in range(count-1):
                extra_numbers = advanced_analyzer.predict_numbers(method=method, explain=False)
                extra_red_balls = extra_numbers[:6]
                extra_blue_ball = extra_numbers[6]
                extra_formatted = format_ssq_numbers(extra_red_balls, extra_blue_ball)
                print(f"第{i+2}注: {extra_formatted}")
        
        # 获取最新开奖结果并比对
        latest_draw = get_latest_draw(data_file)
        if latest_draw:
            issue, date, winning_reds, winning_blue = latest_draw
            latest_numbers = winning_reds + [winning_blue]
            latest_formatted = format_ssq_numbers(winning_reds, winning_blue)
            
            # 计算中奖情况
            from utils import calculate_prize
            prize_level = calculate_prize(numbers[:6], numbers[6], winning_reds, winning_blue)
            
            print(f"最新开奖结果: {latest_formatted}")
            
            if prize_level:
                print(f"恭喜！中得{prize_level}等奖！")
            else:
                print("很遗憾，未中奖")
    elif args.command == "generate":
        generate_numbers(args)
    elif args.command == "latest":
        show_latest(args)


if __name__ == "__main__":
    main()