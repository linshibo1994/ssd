#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球项目主程序
整合爬虫和分析功能，提供命令行界面
"""

import os
import sys
import argparse
from datetime import datetime

# 导入项目模块
from crawler import SSQCrawler
from cwl_crawler import SSQCWLCrawler
from analyzer import SSQAnalyzer
from utils import (
    validate_ssq_data,
    generate_random_numbers,
    generate_smart_numbers,
    format_ssq_numbers,
    get_latest_draw
)


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


def crawl_data(args):
    """爬取数据"""
    print("开始爬取双色球历史数据...")
    
    # 创建爬虫实例
    crawler = SSQCrawler(data_dir=get_data_dir())
    
    # 爬取历史数据
    page_count = args.pages if args.pages else 15
    results = crawler.crawl_history_data(page_count=page_count)
    
    # 保存数据
    if results:
        print(f"共获取{len(results)}期双色球开奖结果")
        crawler.save_to_csv(results)
    else:
        print("未获取到数据")


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
            
            print(f"最新一期({issue})开奖结果:")
            print(format_ssq_numbers(winning_reds, winning_blue))
            print()
            
            if prize_level > 0:
                print(f"恭喜！如果您使用这注号码，将获得{prize_level}等奖！")
            else:
                print("很遗憾，这注号码与最新一期开奖结果不匹配")


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
    parser = argparse.ArgumentParser(description="双色球数据爬取与分析工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 爬取数据子命令
    crawl_parser = subparsers.add_parser("crawl", help="爬取双色球历史数据")
    crawl_parser.add_argument("--pages", type=int, help="爬取的页面数量，每页20条，默认15页约300期")
    
    # 从中国福利彩票官方网站爬取数据子命令
    cwl_parser = subparsers.add_parser("crawl_cwl", help="从中国福利彩票官方网站爬取双色球历史数据")
    cwl_parser.add_argument("--count", type=int, help="爬取的数据条数，默认300期")
    
    # 分析数据子命令
    analyze_parser = subparsers.add_parser("analyze", help="分析双色球历史数据")
    
    # 生成号码子命令
    generate_parser = subparsers.add_parser("generate", help="生成双色球号码")
    generate_parser.add_argument("--method", choices=["random", "frequency", "trend", "hybrid"],
                               default="hybrid", help="生成方法，默认为hybrid")
    
    # 显示最新开奖结果子命令
    latest_parser = subparsers.add_parser("latest", help="显示最新开奖结果")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有指定子命令，显示帮助信息
    if not args.command:
        parser.print_help()
        return
    
    # 执行对应的子命令
    if args.command == "crawl":
        crawl_data(args)
    elif args.command == "crawl_cwl":
        crawl_cwl_data(args)
    elif args.command == "analyze":
        analyze_data(args)
    elif args.command == "generate":
        generate_numbers(args)
    elif args.command == "latest":
        show_latest(args)


if __name__ == "__main__":
    main()