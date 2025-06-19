#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球项目主程序
整合爬虫、分析和高级分析功能，提供命令行界面
"""

import os
import sys
import argparse
import requests
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
    from src.advanced_analyzer import SSQAdvancedAnalyzer as AdvancedSSQAnalyzer
    ADVANCED_ANALYZER_AVAILABLE = True
except ImportError:
    try:
        # 尝试使用相对导入
        from .advanced_analyzer import SSQAdvancedAnalyzer as AdvancedSSQAnalyzer
        ADVANCED_ANALYZER_AVAILABLE = True
    except ImportError:
        try:
            # 尝试直接导入（如果在同一目录下）
            from advanced_analyzer import SSQAdvancedAnalyzer as AdvancedSSQAnalyzer
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


def get_all_data_file():
    """获取完整数据文件路径"""
    return os.path.join(get_data_dir(), "ssq_data_all.csv")


def crawl_cwl_data(args):
    """从中国福利彩票官方网站爬取数据"""
    # 创建爬虫实例
    crawler = SSQCWLCrawler(data_dir=get_data_dir())
    
    # 获取爬取数量
    count = args.count if args.count else None
    all_periods = args.all if hasattr(args, 'all') else False
    
    # 如果指定了--all参数，则忽略count参数
    if all_periods:
        count = None
        filename = "ssq_data_all.csv"
        print("将爬取所有期数的双色球历史数据...")
    else:
        count = count or 300  # 如果没有指定count且没有指定all，则默认为300
        filename = "ssq_data.csv"
        print(f"将爬取最近{count}期双色球历史数据...")
    
    # 获取历史数据
    results = crawler.get_history_data(count=count)
    
    # 保存数据
    if results:
        crawler.save_to_csv(results, filename=filename)
        print(f"成功爬取{len(results)}期双色球历史数据，保存到{filename}")
    else:
        print("爬取数据失败")


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
    
    # 获取最新一期开奖结果进行对比（使用实时获取）
    try:
        issue, date, winning_reds, winning_blue = get_latest_draw(data_file, real_time=True)
        if issue:
            from utils import calculate_prize
            prize_level = calculate_prize(red_balls, blue_ball, winning_reds, winning_blue)
            
            print(f"最新开奖结果({issue}期): {format_ssq_numbers(winning_reds, winning_blue)}")
            print(f"开奖日期: {date}")
            
            if prize_level:
                print(f"恭喜！中得{prize_level}等奖！")
            else:
                print("很遗憾，未中奖")
    except Exception as e:
        print(f"获取最新开奖结果失败: {e}")


def show_latest(args):
    """显示最新开奖结果"""
    data_file = get_data_file()
    
    # 默认使用实时获取
    real_time = True
    
    # 如果数据文件不存在，提示用户但仍然尝试实时获取
    if not os.path.exists(data_file):
        print(f"警告: 本地数据文件不存在: {data_file}")
        print("将尝试从网络实时获取最新开奖结果...")
    
    # 获取最新一期开奖结果，优先使用实时获取
    issue, date, red_balls, blue_ball = get_latest_draw(data_file, real_time=real_time)
    
    if issue:
        print(f"\n最新一期({issue})开奖结果:")
        print(f"开奖日期: {date}")
        print(format_ssq_numbers(red_balls, blue_ball))
    else:
        print("获取最新开奖结果失败")


def markov_predict(args):
    """使用马尔可夫链分析历史数据并预测下一期号码"""
    # 检查高级分析模块是否可用
    if not ADVANCED_ANALYZER_AVAILABLE:
        print("错误: 高级分析模块不可用，请确保已安装所需依赖")
        return
    
    # 确定数据文件路径
    if args.use_all_data:
        data_file = get_all_data_file()
        print("将使用所有历史数据进行马尔可夫链分析")
    else:
        data_file = get_data_file()
        print("将使用最近300期数据进行马尔可夫链分析")
    
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
    advanced_analyzer = AdvancedSSQAnalyzer(data_file)
    
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
    red_balls, blue_ball = advanced_analyzer._predict_by_markov_chain(explain=args.explain)
    formatted_numbers = format_ssq_numbers(red_balls, blue_ball)
    print(f"\n马尔可夫链预测号码: {formatted_numbers}")
    
    # 如果需要生成多注
    if args.count > 1:
        print(f"\n额外预测{args.count-1}注:")
        for i in range(args.count-1):
            red_balls, blue_ball = advanced_analyzer._predict_by_markov_chain(explain=False)
            formatted_numbers = format_ssq_numbers(red_balls, blue_ball)
            print(f"第{i+2}注: {formatted_numbers}")
    
    # 与最新开奖结果比对
    if args.check_latest:
        try:
            issue, date, winning_reds, winning_blue = get_latest_draw(data_file, real_time=True)
            if issue:
                from utils import calculate_prize
                prize_level = calculate_prize(red_balls, blue_ball, winning_reds, winning_blue)
                
                latest_formatted = format_ssq_numbers(winning_reds, winning_blue)
                print(f"\n最新开奖结果({issue}期): {latest_formatted}")
                print(f"开奖日期: {date}")
                
                if prize_level:
                    print(f"恭喜！中得{prize_level}等奖！")
                else:
                    print("很遗憾，未中奖")
        except Exception as e:
            print(f"获取最新开奖结果失败: {e}")


def fetch_latest(args):
    """获取最新一期开奖结果并保存到数据文件"""
    data_file = get_data_file()
    data_dir = get_data_dir()
    
    print("正在爬取最新一期双色球开奖结果...")
    
    # 创建爬虫实例
    crawler = SSQCWLCrawler(data_dir=data_dir)
    
    try:
        # 从中国福利彩票官方网站获取最新一期数据
        api_url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        
        # 请求头
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.cwl.gov.cn/kjxx/ssq/kjgg/",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://www.cwl.gov.cn"
        }
        
        # 设置请求参数 - 只获取最新一期
        params = {
            "name": "ssq",  # 双色球
            "pageNo": 1,     # 第一页
            "pageSize": 1,   # 只获取一条
            "systemType": "PC"  # 系统类型
        }
        
        # 发送请求
        response = requests.get(api_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        # 解析JSON数据
        data = response.json()
        
        # 检查是否有结果数据
        if "result" in data and isinstance(data["result"], list) and len(data["result"]) > 0:
            # 提取开奖结果
            item = data["result"][0]  # 获取最新一期
            
            issue = item["code"]  # 期号
            date = item["date"]  # 开奖日期
            
            # 获取红球号码（格式为 "01,02,03,04,05,06"）
            red_str = item["red"]
            red_balls = red_str.split(",")
            
            # 获取蓝球号码
            blue_ball = item["blue"]
            
            # 确保所有号码都是两位数格式
            red_balls = [ball.zfill(2) for ball in red_balls]
            blue_ball = blue_ball.zfill(2)
            
            result = {
                "issue": issue,
                "date": date,
                "red_balls": ",".join(red_balls),
                "blue_ball": blue_ball
            }
            
            # 显示最新开奖结果
            print(f"\n最新一期({issue})开奖结果:")
            print(f"开奖日期: {date}")
            print(format_ssq_numbers([int(ball) for ball in red_balls], int(blue_ball)))
            
            # 检查数据文件是否存在
            if os.path.exists(data_file):
                # 读取现有数据
                try:
                    import pandas as pd
                    df = pd.read_csv(data_file)
                    
                    # 检查最新期号是否已存在
                    if issue in df["issue"].values:
                        print(f"期号 {issue} 已存在于数据文件中，无需更新")
                    else:
                        # 将新数据添加到DataFrame
                        new_df = pd.DataFrame([result])
                        df = pd.concat([new_df, df]).reset_index(drop=True)
                        
                        # 保存更新后的数据
                        df.to_csv(data_file, index=False)
                        print(f"已将最新一期({issue})数据添加到 {data_file}")
                except Exception as e:
                    print(f"更新数据文件失败: {e}")
            else:
                # 如果数据文件不存在，创建新文件
                try:
                    import pandas as pd
                    df = pd.DataFrame([result])
                    df.to_csv(data_file, index=False)
                    print(f"已创建数据文件 {data_file} 并保存最新一期({issue})数据")
                except Exception as e:
                    print(f"创建数据文件失败: {e}")
            
            return True
        else:
            print("未获取到最新开奖结果")
            return False
    except Exception as e:
        print(f"获取最新一期开奖结果失败: {e}")
        return False


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='双色球数据爬取、分析和号码生成工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 爬取命令
    crawl_parser = subparsers.add_parser('crawl', help='爬取双色球历史数据')
    crawl_parser.add_argument('--count', type=int, help="爬取的数据条数，默认300期")
    crawl_parser.add_argument('--all', action='store_true', help="爬取所有期数的双色球历史数据")
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析双色球历史数据')
    
    # 高级分析命令
    advanced_parser = subparsers.add_parser('advanced', help='使用高级统计和机器学习方法分析双色球数据')
    advanced_parser.add_argument('--method', choices=['all', 'stats', 'probability', 'frequency', 'decision_tree', 'cycle', 'bayes', 'correlation', 'issue_correlation'], 
                               default='all', help='高级分析方法')
    advanced_parser.add_argument('--periods', type=int, default=300, help='分析期数，默认为300期')
    advanced_parser.add_argument('--save_model', action='store_true', help='是否保存训练好的模型')
    advanced_parser.add_argument('--correlation_periods', type=str, default='5,10,50,100', help='分析历史关联性的期数间隔，用逗号分隔，默认为5,10,50,100')
    
    # 智能预测命令
    predict_parser = subparsers.add_parser('predict', help='使用高级分析模型预测双色球号码')
    predict_parser.add_argument('--method', choices=['stats', 'probability', 'decision_tree', 'bayes', 'ensemble', 'pattern', 'markov'], 
                              default='ensemble', help='预测方法')
    predict_parser.add_argument('--count', type=int, default=1, help='生成注数，默认为1注')
    predict_parser.add_argument('--explain', action='store_true', help='是否解释预测结果')
    predict_parser.add_argument('--compare', action='store_true', help='是否与历史数据进行对比分析')
    predict_parser.add_argument('--compare_periods', type=int, default=300, help='与历史数据对比的期数，默认为300期')
    predict_parser.add_argument('--check-latest', action='store_true', help='检查与最新一期的匹配情况')
    
    # 生成命令
    generate_parser = subparsers.add_parser('generate', help='生成双色球号码')
    generate_parser.add_argument('--method', choices=['random', 'frequency', 'trend', 'hybrid'], 
                                default='hybrid', help='生成方法，默认为hybrid')
    
    # 最新开奖命令
    latest_parser = subparsers.add_parser('latest', help='显示最新开奖结果')
    
    # 获取最新一期开奖结果命令
    fetch_latest_parser = subparsers.add_parser('fetch-latest', help='获取最新一期开奖结果并保存到数据文件')
    
    # 马尔可夫链预测命令
    markov_parser = subparsers.add_parser('markov_predict', help='使用马尔可夫链分析历史数据并预测下一期号码')
    markov_parser.add_argument('--periods', type=int, default=300, help='使用近多少期数据进行分析，默认300期')
    markov_parser.add_argument('--use-all-data', action='store_true', help='使用所有历史数据进行分析')
    markov_parser.add_argument('--explain', action='store_true', help='解释预测结果')
    markov_parser.add_argument('--count', type=int, default=1, help='生成注数，默认为1注')
    markov_parser.add_argument('--check-latest', action='store_true', help='检查与最新一期的匹配情况')
    
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
    elif args.command == "fetch-latest":
        fetch_latest(args)
    elif args.command == "advanced":
        if not ADVANCED_ANALYZER_AVAILABLE:
            print("错误: 高级分析模块不可用，请确保已安装所需依赖")
            return
        
        method = args.method
        periods = args.periods
        save_model = args.save_model
        
        # 获取数据文件路径
        data_file = get_data_file()
        
        advanced_analyzer = AdvancedSSQAnalyzer(data_file)
        # 加载数据
        if not advanced_analyzer.load_data():
            print("加载数据失败")
            return
            
        if method == "all":
            # 运行所有分析
            advanced_analyzer.run_advanced_analysis()
        elif method == "stats":
            # 运行统计特性分析
            advanced_analyzer.analyze_statistical_features()
        elif method == "probability":
            # 运行概率分布分析
            advanced_analyzer.analyze_probability_distribution()
        elif method == "frequency":
            # 运行频率模式分析
            advanced_analyzer.analyze_frequency_patterns()
        elif method == "decision_tree":
            # 运行决策树分析
            advanced_analyzer.analyze_decision_tree()
        elif method == "cycle":
            # 运行周期分析
            advanced_analyzer.analyze_cycle_patterns()
        elif method == "bayes":
            # 运行贝叶斯分析
            if PYMC_AVAILABLE:
                advanced_analyzer.analyze_bayesian()
            else:
                print("PyMC未安装，无法进行贝叶斯分析")
                return
        elif method == "correlation":
            # 运行历史关联性分析
            correlation_periods = [int(p) for p in args.correlation_periods.split(',')]
            advanced_analyzer.analyze_historical_correlation(periods_list=correlation_periods)
        elif method == "issue_correlation":
            # 运行期号关联性分析
            advanced_analyzer.analyze_issue_number_correlation()
    elif args.command == 'predict':
        if not ADVANCED_ANALYZER_AVAILABLE:
            print("错误: 高级分析模块不可用，请确保已安装所需依赖")
            return
        
        method = args.method
        count = args.count
        explain = args.explain
        compare = args.compare
        compare_periods = args.compare_periods
        
        # 获取数据文件路径
        data_file = get_data_file()
        
        advanced_analyzer = AdvancedSSQAnalyzer(data_file)
        
        # 预测号码
        if method == 'pattern':
            red_balls, blue_ball, explanation = advanced_analyzer.predict_based_on_patterns(explain=explain)
            numbers = list(red_balls) + [blue_ball]
        else:
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
        
        # 与历史数据进行对比分析
        if compare:
            print("\n与历史数据对比分析:")
            advanced_analyzer.compare_with_historical_data((numbers[:6], numbers[6]), periods=compare_periods)
        
        # 获取最新开奖结果并比对（使用实时获取）
        latest_draw = get_latest_draw(data_file, real_time=True)
        if latest_draw:
            issue, date, winning_reds, winning_blue = latest_draw
            latest_numbers = winning_reds + [winning_blue]
            latest_formatted = format_ssq_numbers(winning_reds, winning_blue)
            
            # 计算中奖情况
            from utils import calculate_prize
            prize_level = calculate_prize(numbers[:6], numbers[6], winning_reds, winning_blue)
            
            print(f"最新开奖结果({issue}期): {latest_formatted}")
            print(f"开奖日期: {date}")
            
            if prize_level:
                print(f"恭喜！中得{prize_level}等奖！")
            else:
                print("很遗憾，未中奖")
    elif args.command == "generate":
        generate_numbers(args)
    elif args.command == "latest":
        show_latest(args)
    elif args.command == "markov_predict":
        markov_predict(args)


if __name__ == "__main__":
    main()