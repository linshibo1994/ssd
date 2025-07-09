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
    predict_parser.add_argument('--method', choices=['stats', 'probability', 'decision_tree', 'bayes', 'ensemble', 'pattern'],
                              default='ensemble', help='预测方法')
    predict_parser.add_argument('--count', type=int, default=1, help='生成注数，默认为1注')
    predict_parser.add_argument('--explain', action='store_true', help='是否解释预测结果')
    predict_parser.add_argument('--compare', action='store_true', help='是否与历史数据进行对比分析')
    predict_parser.add_argument('--compare_periods', type=int, default=300, help='与历史数据对比的期数，默认为300期')

    # LSTM深度学习预测命令
    lstm_parser = subparsers.add_parser('lstm', help='LSTM深度学习预测')
    lstm_parser.add_argument('--train', action='store_true', help='训练LSTM模型')
    lstm_parser.add_argument('--retrain', action='store_true', help='重新训练LSTM模型')
    lstm_parser.add_argument('-p', '--periods', type=int, default=500, help='训练期数，默认500期')
    lstm_parser.add_argument('--red_epochs', type=int, default=50, help='红球训练轮数，默认50')
    lstm_parser.add_argument('--blue_epochs', type=int, default=50, help='蓝球训练轮数，默认50')
    lstm_parser.add_argument('-n', '--num', type=int, default=1, help='预测注数，默认1注')

    # 集成学习预测命令
    ensemble_parser = subparsers.add_parser('ensemble', help='集成学习预测')
    ensemble_parser.add_argument('--train', action='store_true', help='训练集成学习模型')
    ensemble_parser.add_argument('-p', '--periods', type=int, default=500, help='训练期数，默认500期')
    ensemble_parser.add_argument('-n', '--num', type=int, default=1, help='预测注数，默认1注')

    # 蒙特卡洛模拟预测命令
    monte_carlo_parser = subparsers.add_parser('monte_carlo', help='蒙特卡洛模拟预测')
    monte_carlo_parser.add_argument('-n', '--num', type=int, default=1, help='预测注数，默认1注')
    monte_carlo_parser.add_argument('-s', '--simulations', type=int, default=10000, help='模拟次数，默认10000次')
    monte_carlo_parser.add_argument('--analyze', action='store_true', help='进行模式分析')
    monte_carlo_parser.add_argument('--save', action='store_true', help='保存分析结果')

    # 聚类分析预测命令
    clustering_parser = subparsers.add_parser('clustering', help='聚类分析预测')
    clustering_parser.add_argument('-n', '--num', type=int, default=1, help='预测注数，默认1注')
    clustering_parser.add_argument('-k', '--clusters', type=int, help='聚类数，默认自动确定')
    clustering_parser.add_argument('--visualize', action='store_true', help='生成聚类可视化图')
    clustering_parser.add_argument('--save', action='store_true', help='保存分析结果')

    # 超级预测器命令
    super_parser = subparsers.add_parser('super', help='超级预测器（集成所有方法）')
    super_parser.add_argument('-m', '--mode', choices=['ensemble', 'quick', 'all', 'compare'],
                             default='ensemble', help='预测模式，默认为ensemble')
    super_parser.add_argument('-n', '--num', type=int, default=1, help='预测注数，默认1注')
    super_parser.add_argument('--train', action='store_true', help='训练所有模型')
    super_parser.add_argument('-p', '--periods', type=int, default=500, help='训练期数，默认500期')
    super_parser.add_argument('--save', action='store_true', help='保存预测结果')
    
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
            advanced_analyzer.compare_with_historical_data(numbers[:6], numbers[6], periods=compare_periods)
        
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
    elif args.command == "lstm":
        # LSTM深度学习预测
        try:
            from lstm_predictor import SSQLSTMPredictor

            data_file = get_data_file()
            model_dir = os.path.join(get_project_root(), "data", "models")

            predictor = SSQLSTMPredictor(data_file=data_file, model_dir=model_dir)

            if args.train or args.retrain:
                success = predictor.train_models(
                    periods=args.periods,
                    red_epochs=args.red_epochs,
                    blue_epochs=args.blue_epochs
                )
                if success:
                    print("LSTM模型训练成功！")
                else:
                    print("LSTM模型训练失败！")
            else:
                print("🧠 LSTM深度学习预测")
                print("=" * 40)

                predictions = predictor.predict(num_predictions=args.num)
                if predictions:
                    for i, (red_balls, blue_ball) in enumerate(predictions, 1):
                        formatted = format_ssq_numbers(red_balls, blue_ball)
                        print(f"第 {i} 注: {formatted}")
                else:
                    print("预测失败，请先训练模型")
        except ImportError:
            print("LSTM预测器不可用，请安装TensorFlow: pip install tensorflow")

    elif args.command == "ensemble":
        # 集成学习预测
        try:
            from ensemble_predictor import SSQEnsemblePredictor

            data_file = get_data_file()
            model_dir = os.path.join(get_project_root(), "data", "models")

            predictor = SSQEnsemblePredictor(data_file=data_file, model_dir=model_dir)

            if args.train:
                success = predictor.train_models(periods=args.periods)
                if success:
                    print("集成学习模型训练成功！")
                else:
                    print("集成学习模型训练失败！")
            else:
                print("🤖 集成学习预测")
                print("=" * 40)

                predictions = predictor.predict(num_predictions=args.num)
                if predictions:
                    for i, (red_balls, blue_ball) in enumerate(predictions, 1):
                        formatted = format_ssq_numbers(red_balls, blue_ball)
                        print(f"第 {i} 注: {formatted}")
                else:
                    print("预测失败，请先训练模型")
        except ImportError:
            print("集成学习预测器不可用，请安装依赖: pip install xgboost lightgbm")

    elif args.command == "monte_carlo":
        # 蒙特卡洛模拟预测
        try:
            from monte_carlo_predictor import SSQMonteCarloPredictor

            data_file = get_data_file()
            output_dir = os.path.join(get_project_root(), "data", "monte_carlo")

            predictor = SSQMonteCarloPredictor(data_file=data_file, output_dir=output_dir)

            print("🎲 蒙特卡洛模拟预测")
            print("=" * 40)
            print(f"模拟次数: {args.simulations:,}")

            predictions = predictor.predict(
                num_predictions=args.num,
                num_simulations=args.simulations
            )

            if predictions:
                for i, (red_balls, blue_ball, confidence) in enumerate(predictions, 1):
                    formatted = format_ssq_numbers(red_balls, blue_ball)
                    print(f"第 {i} 注: {formatted} (置信度: {confidence:.1%})")

                if args.analyze:
                    pattern_analysis = predictor.analyze_patterns()
                    print("\n📊 模式分析结果:")
                    print(f"红球热号: {', '.join([f'{ball:02d}' for ball in pattern_analysis['red_hot_numbers'][:5]])}")
                    print(f"蓝球热号: {', '.join([f'{ball:02d}' for ball in pattern_analysis['blue_hot_numbers'][:3]])}")

                if args.save:
                    if not args.analyze:
                        pattern_analysis = predictor.analyze_patterns()
                    predictor.save_analysis_results(predictions, pattern_analysis)
            else:
                print("预测失败")
        except ImportError:
            print("蒙特卡洛预测器不可用，请安装SciPy: pip install scipy")

    elif args.command == "clustering":
        # 聚类分析预测
        try:
            from clustering_predictor import SSQClusteringPredictor

            data_file = get_data_file()
            output_dir = os.path.join(get_project_root(), "data", "clustering")

            predictor = SSQClusteringPredictor(data_file=data_file, output_dir=output_dir)

            print("🔍 K-Means聚类分析预测")
            print("=" * 40)

            predictions = predictor.predict(num_predictions=args.num, k=args.clusters)

            if predictions:
                for i, (red_balls, blue_ball) in enumerate(predictions, 1):
                    formatted = format_ssq_numbers(red_balls, blue_ball)
                    print(f"第 {i} 注: {formatted}")

                if args.visualize or args.save:
                    features_df = predictor.extract_clustering_features()
                    clustering_results = predictor.perform_clustering(features_df, k=args.clusters)
                    cluster_patterns = predictor.analyze_cluster_patterns(clustering_results)

                    if args.visualize:
                        predictor.visualize_clusters(clustering_results)

                    if args.save:
                        predictor.save_clustering_results(clustering_results, cluster_patterns, predictions)
            else:
                print("预测失败")
        except ImportError:
            print("聚类分析预测器不可用，请安装依赖: pip install scikit-learn matplotlib")

    elif args.command == "super":
        # 超级预测器
        try:
            from super_predictor import SSQSuperPredictor

            data_file = get_data_file()
            model_dir = os.path.join(get_project_root(), "data", "models")
            output_dir = os.path.join(get_project_root(), "data", "super")

            predictor = SSQSuperPredictor(
                data_file=data_file,
                model_dir=model_dir,
                output_dir=output_dir
            )

            if args.train:
                predictor.train_all_models(periods=args.periods)
            else:
                print("🌟 超级预测器")
                print("=" * 80)

                if args.mode == 'ensemble':
                    print(f"🏆 集成预测模式 - {args.num}注")
                elif args.mode == 'quick':
                    print(f"⚡ 快速预测模式 - {args.num}注")
                elif args.mode == 'all':
                    print(f"🌟 全方法预测模式 - {args.num}注")
                elif args.mode == 'compare':
                    print(f"📊 方法对比模式 - {args.num}注")

                print("=" * 80)

                results = predictor.predict(mode=args.mode, num_predictions=args.num)

                if results:
                    if args.mode == 'ensemble':
                        from super_predictor import print_ensemble_results
                        print_ensemble_results(results, predictor)
                    elif args.mode == 'quick':
                        # 使用utils中的format_ssq_numbers
                        print("⚡ 快速预测结果:")
                        for i, (red_balls, blue_ball) in enumerate(results, 1):
                            formatted = format_ssq_numbers(red_balls, blue_ball)
                            print(f"第 {i} 注: {formatted}")
                    elif args.mode == 'all':
                        from super_predictor import print_all_results
                        print_all_results(results)
                    elif args.mode == 'compare':
                        from super_predictor import print_compare_results
                        print_compare_results(results)

                    if args.save:
                        predictor.save_prediction_results(results, args.mode)
                else:
                    print("预测失败")
        except ImportError as e:
            print(f"超级预测器不可用: {e}")

    elif args.command == "generate":
        generate_numbers(args)
    elif args.command == "latest":
        show_latest(args)


if __name__ == "__main__":
    main()