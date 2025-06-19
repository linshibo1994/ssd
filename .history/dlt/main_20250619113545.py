#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大乐透分析工具主程序

提供命令行接口，用于分析大乐透数据、生成号码、显示最新开奖结果等功能
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from basic_analyzer import BasicAnalyzer
from advanced_analyzer import DLTAdvancedAnalyzer
from utils import generate_random_numbers, generate_smart_numbers, get_latest_draw, calculate_prize as check_prize_level
from cwl_crawler import DLTCWLCrawler


def check_data_file(data_file):
    """检查数据文件是否存在，如果不存在则尝试爬取数据

    Args:
        data_file: 数据文件路径

    Returns:
        bool: 数据文件是否存在
    """
    if os.path.exists(data_file):
        return True
    
    # 如果数据文件不存在，尝试爬取数据
    print(f"数据文件 {data_file} 不存在，尝试爬取数据...")
    data_dir = os.path.dirname(data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    try:
        # 创建爬虫实例并获取数据
        crawler = DLTCWLCrawler()
        results = crawler.get_history_data(count=300)
        if results:
            crawler.save_to_csv(results, filename=os.path.basename(data_file))
            return True
        else:
            raise Exception("未获取到数据")
    except Exception as e:
        print(f"爬取数据失败: {e}")
        return False


def analyze(args):
    """分析大乐透数据

    Args:
        args: 命令行参数
    """
    data_file = args.data_file
    output_dir = args.output_dir
    periods = args.periods
    advanced = args.advanced
    
    # 检查数据文件
    if not check_data_file(data_file):
        print("无法获取数据，分析终止")
        return
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 根据参数选择分析器
    if advanced:
        print("使用高级分析器...")
        analyzer = DLTAdvancedAnalyzer(data_file, output_dir, periods)
        analyzer.run_advanced_analysis()
    else:
        print("使用基础分析器...")
        analyzer = BasicAnalyzer(data_file, output_dir)
        analyzer.run_basic_analysis()
    
    print(f"分析完成，结果保存在 {output_dir}")


def generate(args):
    """生成大乐透号码

    Args:
        args: 命令行参数
    """
    data_file = args.data_file
    count = args.count
    strategy = args.strategy
    periods = args.periods
    
    # 检查数据文件
    if strategy != "random" and not check_data_file(data_file):
        print("无法获取数据，使用随机策略生成号码")
        strategy = "random"
    
    # 生成号码
    print(f"使用 {strategy} 策略生成 {count} 注大乐透号码...")
    
    if strategy == "random":
        # 随机生成
        for i in range(count):
            front_balls, back_balls = generate_random_numbers()
            print(f"[{i+1}] 前区: {front_balls}, 后区: {back_balls}")
    else:
        # 智能生成
         for i in range(count):
            front_balls, back_balls = generate_smart_numbers(data_file, strategy)
            print(f"[{i+1}] 前区: {front_balls}, 后区: {back_balls}")


def latest(args):
    """显示最新开奖结果

    Args:
        args: 命令行参数
    """
    data_file = args.data_file
    compare = args.compare
    
    # 获取最新开奖结果
    try:
        latest_result = get_latest_draw(data_file)
        if latest_result:
            issue, date, front_balls, back_balls = latest_result
            
            print(f"\n最新开奖结果 (期号: {issue}, 日期: {date})")
            print(f"前区号码: {front_balls}")
            print(f"后区号码: {back_balls}")
            
            # 如果需要比对
            if compare:
                compare_with_latest(front_balls, back_balls)
        else:
            print("无法获取最新开奖结果")
    except Exception as e:
        print(f"获取最新开奖结果失败: {e}")


def compare_with_latest(front_balls_latest, back_balls_latest):
    """将用户输入的号码与最新开奖结果进行比对

    Args:
        latest_front: 最新开奖前区号码
        latest_back: 最新开奖后区号码
    """
    try:
        # 获取用户输入
        print("\n请输入您的大乐透号码进行比对:")
        print("前区号码 (用空格分隔5个号码，范围1-35): ")
        front_input = input().strip()
        front_balls = [int(x) for x in front_input.split()]
        
        print("后区号码 (用空格分隔2个号码，范围1-12): ")
        back_input = input().strip()
        back_balls = [int(x) for x in back_input.split()]
        
        # 验证输入
        if len(front_balls) != 5 or len(back_balls) != 2:
            print("输入号码数量不正确")
            return
        
        if not all(1 <= x <= 35 for x in front_balls) or not all(1 <= x <= 12 for x in back_balls):
            print("输入号码范围不正确")
            return
        
        # 排序
        front_balls.sort()
        back_balls.sort()
        
        # 比对
        front_match = len(set(front_balls) & set(front_balls_latest))
        back_match = len(set(back_balls) & set(back_balls_latest))
        
        print(f"\n您的号码: 前区 {front_balls}, 后区 {back_balls}")
        print(f"开奖号码: 前区 {front_balls_latest}, 后区 {back_balls_latest}")
        print(f"匹配结果: 前区匹配 {front_match} 个, 后区匹配 {back_match} 个")
        
        # 判断中奖等级
        prize_level = check_prize_level(front_balls, back_balls, front_balls_latest, back_balls_latest)
        if prize_level > 0:
            print(f"恭喜您中得 {prize_level} 等奖！")
        else:
            print("很遗憾，您未中奖")
            
    except Exception as e:
        print(f"比对失败: {e}")


def markov(args):
    """使用马尔可夫链分析和预测

    Args:
        args: 命令行参数
    """
    data_file = args.data_file
    output_dir = args.output_dir
    periods = args.periods
    count = args.count
    
    # 检查数据文件
    if not check_data_file(data_file):
        print("无法获取数据，分析终止")
        return
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载数据
    df = pd.read_csv(data_file)
    if periods > 0:
        df = df.head(periods)
    
    # 创建高级分析器并进行马尔可夫链分析
    analyzer = DLTAdvancedAnalyzer(data_file, output_dir, periods)
    markov_results = analyzer.analyze_markov_chain()
    
    # 生成预测号码
    print(f"\n基于马尔可夫链分析生成 {count} 注预测号码:")
    for i in range(count):
        front_balls, back_balls = analyzer.predict_by_markov_chain(explain=True)
        print(f"\n[{i+1}] 前区: {','.join([str(b).zfill(2) for b in front_balls])}, 后区: {','.join([str(b).zfill(2) for b in back_balls])}")
    
    # 获取最新开奖结果进行比对
    try:
        latest_result = get_latest_draw(data_file)
        if latest_result:
            issue, date, front_balls, back_balls = latest_result
            
            print(f"\n最新开奖结果 (期号: {issue}, 日期: {date})")
            print(f"前区号码: {front_balls}")
            print(f"后区号码: {back_balls}")
            
            # 生成一注预测号码并与最新结果比对
            pred_front, pred_back = analyzer.predict_by_markov_chain()
            front_match = len(set(pred_front) & set(front_balls))
            back_match = len(set(pred_back) & set(back_balls))
            
            print(f"\n预测号码: 前区 {','.join([str(b).zfill(2) for b in pred_front])}, 后区 {','.join([str(b).zfill(2) for b in pred_back])}")
            print(f"匹配结果: 前区匹配 {front_match} 个, 后区匹配 {back_match} 个")
            
            # 判断中奖等级
            prize_level = check_prize_level(pred_front, pred_back, front_balls, back_balls)
            if prize_level > 0:
                print(f"预测结果中得 {prize_level} 等奖！")
            else:
                print("预测结果未中奖")
    except Exception as e:
        print(f"获取最新开奖结果失败: {e}")


def bayesian(args):
    """使用贝叶斯分析和预测

    Args:
        args: 命令行参数
    """
    data_file = args.data_file
    output_dir = args.output_dir
    periods = args.periods
    count = args.count
    
    # 检查数据文件
    if not check_data_file(data_file):
        print("无法获取数据，分析终止")
        return
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载数据
    df = pd.read_csv(data_file)
    if periods > 0:
        df = df.head(periods)
    
    # 创建高级分析器并进行贝叶斯分析
    analyzer = DLTAdvancedAnalyzer(data_file, output_dir, periods)
    bayesian_results = analyzer.analyze_bayesian()
    
    # 生成预测号码
    print(f"\n基于贝叶斯分析生成 {count} 注预测号码:")
    for i in range(count):
        front_balls, back_balls = analyzer.predict_by_bayes(explain=True)
        print(f"\n[{i+1}] 前区: {','.join([str(b).zfill(2) for b in front_balls])}, 后区: {','.join([str(b).zfill(2) for b in back_balls])}")
    
    # 获取最新开奖结果进行比对
    try:
        latest_result = get_latest_draw(data_file)
        if latest_result:
            issue, date, front_balls, back_balls = latest_result
            
            print(f"\n最新开奖结果 (期号: {issue}, 日期: {date})")
            print(f"前区号码: {front_balls}")
            print(f"后区号码: {back_balls}")
            
            # 生成一注预测号码并与最新结果比对
            pred_front, pred_back = analyzer.predict_by_bayes()
            front_match = len(set(pred_front) & set(front_balls))
            back_match = len(set(pred_back) & set(back_balls))
            
            print(f"\n预测号码: 前区 {','.join([str(b).zfill(2) for b in pred_front])}, 后区 {','.join([str(b).zfill(2) for b in pred_back])}")
            print(f"匹配结果: 前区匹配 {front_match} 个, 后区匹配 {back_match} 个")
            
            # 判断中奖等级
            prize_level = check_prize_level(pred_front, pred_back, front_balls, back_balls)
            if prize_level > 0:
                print(f"预测结果中得 {prize_level} 等奖！")
            else:
                print("预测结果未中奖")
    except Exception as e:
        print(f"获取最新开奖结果失败: {e}")


def compare(args):
    """比较用户输入的号码与历史数据

    Args:
        args: 命令行参数
    """
    data_file = args.data_file
    periods = args.periods
    
    # 检查数据文件
    if not check_data_file(data_file):
        print("无法获取数据，分析终止")
        return
    
    try:
        # 获取用户输入
        print("请输入您的大乐透号码进行历史比对:")
        print("前区号码 (用空格分隔5个号码，范围1-35): ")
        front_input = input().strip()
        front_balls = [int(x) for x in front_input.split()]
        
        print("后区号码 (用空格分隔2个号码，范围1-12): ")
        back_input = input().strip()
        back_balls = [int(x) for x in back_input.split()]
        
        # 验证输入
        if len(front_balls) != 5 or len(back_balls) != 2:
            print("输入号码数量不正确")
            return
        
        if not all(1 <= x <= 35 for x in front_balls) or not all(1 <= x <= 12 for x in back_balls):
            print("输入号码范围不正确")
            return
        
        # 排序
        front_balls.sort()
        back_balls.sort()
        
        # 创建高级分析器并进行比对分析
        analyzer = DLTAdvancedAnalyzer(data_file, "", periods)
        analyzer.compare_with_history(front_balls, back_balls)
        
    except Exception as e:
        print(f"比对失败: {e}")


def crawl(args):
    """爬取大乐透历史数据

    Args:
        args: 命令行参数
    """
    data_file = args.data_file
    append = args.append
    
    # 创建数据目录
    data_dir = os.path.dirname(data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 爬取数据
    try:
        # 创建爬虫实例
        crawler = DLTCWLCrawler()
        
        # 获取历史数据
        results = crawler.get_history_data(count=300)
        
        # 保存数据
        if results:
            # 从data_file中提取文件名
            filename = os.path.basename(data_file)
            saved_path = crawler.save_to_csv(results, filename=filename)
            print(f"数据爬取完成，保存到 {saved_path}")
        else:
            print("未获取到数据")
    except Exception as e:
        print(f"爬取数据失败: {e}")


def markov_predict(args):
    """使用马尔可夫链分析历史数据并预测下一期号码"""
    # 检查高级分析模块是否可用
    try:
        from advanced_analyzer import DLTAdvancedAnalyzer
        ADVANCED_ANALYZER_AVAILABLE = True
    except ImportError:
        print("错误: 高级分析模块不可用，请确保已安装所需依赖")
        return
    
    # 确定数据文件路径
    data_file = args.data_file
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    # 确定分析期数
    periods = args.periods
    print(f"将使用近{periods}期数据进行马尔可夫链分析")
    
    print("开始马尔可夫链分析...")
    
    # 创建高级分析器实例
    advanced_analyzer = DLTAdvancedAnalyzer(data_file)
    
    # 加载数据
    if not advanced_analyzer.load_data():
        print("加载数据失败")
        return
    
    # 只保留最近periods期数据进行分析
    if len(advanced_analyzer.data) > periods:
        # 数据是按日期降序排列的，所以取前periods行
        advanced_analyzer.data = advanced_analyzer.data.head(periods).reset_index(drop=True)
        print(f"已筛选最近{len(advanced_analyzer.data)}期数据进行分析")
    else:
        print(f"警告: 数据总期数({len(advanced_analyzer.data)})小于指定期数({periods})，将使用全部可用数据")
    
    # 执行马尔可夫链分析
    advanced_analyzer.analyze_markov_chain()
    
    # 预测下一期号码
    print("\n预测下一期号码:")
    front_balls, back_balls = advanced_analyzer.predict_by_markov_chain(explain=args.explain)
    formatted_numbers = format_dlt_numbers(front_balls, back_balls)
    print(f"\n马尔可夫链预测号码: {formatted_numbers}")
    
    # 如果需要生成多注
    if args.count > 1:
        print(f"\n额外预测{args.count-1}注:")
        for i in range(args.count-1):
            front_balls, back_balls = advanced_analyzer.predict_by_markov_chain(explain=False)
            formatted_numbers = format_dlt_numbers(front_balls, back_balls)
            print(f"第{i+2}注: {formatted_numbers}")
    
    # 与最新开奖结果比对
    if args.check_latest:
        try:
            issue, date, winning_fronts, winning_backs = get_latest_draw(data_file, real_time=True)
            if issue:
                from utils import calculate_prize
                prize_level = calculate_prize(front_balls, back_balls, winning_fronts, winning_backs)
                
                latest_formatted = format_dlt_numbers(winning_fronts, winning_backs)
                print(f"\n最新开奖结果({issue}期): {latest_formatted}")
                print(f"开奖日期: {date}")
                
                if prize_level > 0:
                    print(f"恭喜！中得{prize_level}等奖！")
                else:
                    print("很遗憾，未中奖")
        except Exception as e:
            print(f"获取最新开奖结果失败: {e}")


def bayesian_predict(args):
    """使用贝叶斯分析历史数据并预测下一期号码"""
    # 检查高级分析模块是否可用
    try:
        from advanced_analyzer import DLTAdvancedAnalyzer
        ADVANCED_ANALYZER_AVAILABLE = True
    except ImportError:
        print("错误: 高级分析模块不可用，请确保已安装所需依赖")
        return
    
    # 确定数据文件路径
    data_file = args.data_file
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    print("开始贝叶斯分析...")
    
    # 创建高级分析器实例
    advanced_analyzer = DLTAdvancedAnalyzer(data_file)
    
    # 加载数据
    if not advanced_analyzer.load_data():
        print("加载数据失败")
        return
    
    # 执行贝叶斯分析
    advanced_analyzer.analyze_bayesian()
    
    # 预测下一期号码
    print("\n预测下一期号码:")
    front_balls, back_balls = advanced_analyzer.predict_by_bayes(explain=args.explain)
    formatted_numbers = format_dlt_numbers(front_balls, back_balls)
    print(f"\n贝叶斯预测号码: {formatted_numbers}")
    
    # 如果需要生成多注
    if args.count > 1:
        print(f"\n额外预测{args.count-1}注:")
        for i in range(args.count-1):
            front_balls, back_balls = advanced_analyzer.predict_by_bayes(explain=False)
            formatted_numbers = format_dlt_numbers(front_balls, back_balls)
            print(f"第{i+2}注: {formatted_numbers}")
    
    # 与最新开奖结果比对
    if args.check_latest:
        try:
            issue, date, winning_fronts, winning_backs = get_latest_draw(data_file, real_time=True)
            if issue:
                from utils import calculate_prize
                prize_level = calculate_prize(front_balls, back_balls, winning_fronts, winning_backs)
                
                latest_formatted = format_dlt_numbers(winning_fronts, winning_backs)
                print(f"\n最新开奖结果({issue}期): {latest_formatted}")
                print(f"开奖日期: {date}")
                
                if prize_level > 0:
                    print(f"恭喜！中得{prize_level}等奖！")
                else:
                    print("很遗憾，未中奖")
        except Exception as e:
            print(f"获取最新开奖结果失败: {e}")


def main():
    """主函数"""
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description="大乐透分析工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 分析子命令
    analyze_parser = subparsers.add_parser("analyze", help="分析大乐透数据")
    analyze_parser.add_argument("-d", "--data-file", default="../data/dlt_data.csv", help="数据文件路径")
    analyze_parser.add_argument("-o", "--output-dir", default="../output", help="输出目录")
    analyze_parser.add_argument("-p", "--periods", type=int, default=0, help="分析期数，0表示全部")
    analyze_parser.add_argument("-a", "--advanced", action="store_true", help="使用高级分析")
    
    # 生成子命令
    generate_parser = subparsers.add_parser("generate", help="生成大乐透号码")
    generate_parser.add_argument("-d", "--data-file", default="../data/dlt_data.csv", help="数据文件路径")
    generate_parser.add_argument("-c", "--count", type=int, default=5, help="生成号码注数")
    generate_parser.add_argument("-s", "--strategy", choices=["random", "frequency", "trend", "mixed"], default="random", help="生成策略")
    generate_parser.add_argument("-p", "--periods", type=int, default=0, help="参考期数，0表示全部")
    
    # 最新开奖子命令
    latest_parser = subparsers.add_parser("latest", help="显示最新开奖结果")
    latest_parser.add_argument("-d", "--data-file", default="../data/dlt_data.csv", help="数据文件路径")
    latest_parser.add_argument("-c", "--compare", action="store_true", help="与自选号码比对")
    
    # 马尔可夫链分析子命令
    markov_parser = subparsers.add_parser("markov", help="使用马尔可夫链分析和预测")
    markov_parser.add_argument("-d", "--data-file", default="../data/dlt_data.csv", help="数据文件路径")
    markov_parser.add_argument("-o", "--output-dir", default="../output/advanced", help="输出目录")
    markov_parser.add_argument("-p", "--periods", type=int, default=100, help="分析期数，0表示全部")
    markov_parser.add_argument("-c", "--count", type=int, default=5, help="生成预测号码注数")
    markov_parser.add_argument("--explain", action="store_true", help="解释预测结果")
    markov_parser.add_argument("--check-latest", action="store_true", help="检查与最新一期的匹配情况")
    
    # 贝叶斯分析子命令
    bayesian_parser = subparsers.add_parser("bayesian", help="使用贝叶斯分析和预测")
    bayesian_parser.add_argument("-d", "--data-file", default="../data/dlt_data.csv", help="数据文件路径")
    bayesian_parser.add_argument("-o", "--output-dir", default="../output/advanced", help="输出目录")
    bayesian_parser.add_argument("-p", "--periods", type=int, default=100, help="分析期数，0表示全部")
    bayesian_parser.add_argument("-c", "--count", type=int, default=5, help="生成预测号码注数")
    bayesian_parser.add_argument("--explain", action="store_true", help="解释预测结果")
    bayesian_parser.add_argument("--check-latest", action="store_true", help="检查与最新一期的匹配情况")
    
    # 比较子命令
    compare_parser = subparsers.add_parser("compare", help="比较用户输入的号码与历史数据")
    compare_parser.add_argument("-d", "--data-file", default="../data/dlt_data.csv", help="数据文件路径")
    compare_parser.add_argument("-p", "--periods", type=int, default=0, help="比较期数，0表示全部")
    
    # 爬取数据子命令
    crawl_parser = subparsers.add_parser("crawl", help="爬取大乐透历史数据")
    crawl_parser.add_argument("-d", "--data-file", default="../data/dlt_data.csv", help="数据文件路径")
    crawl_parser.add_argument("-a", "--append", action="store_true", help="追加到现有文件")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据子命令调用相应的函数
    if args.command == "analyze":
        analyze(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "latest":
        latest(args)
    elif args.command == "markov":
        markov_predict(args)
    elif args.command == "bayesian":
        bayesian_predict(args)
    elif args.command == "compare":
        compare(args)
    elif args.command == "crawl":
        crawl(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()