#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蒙特卡洛模拟预测器
基于Callam7/LottoPipeline项目的蒙特卡洛方法
使用概率分布建模 + 随机采样 + 自助法
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 科学计算相关导入
try:
    from scipy import stats
    from scipy.stats import bootstrap
    import matplotlib.pyplot as plt
    import seaborn as sns
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: SciPy未安装，蒙特卡洛模拟预测功能将不可用")
    print("请安装SciPy: pip install scipy matplotlib seaborn")


class SSQMonteCarloPredictor:
    """双色球蒙特卡洛模拟预测器"""
    
    def __init__(self, data_file="../data/ssq_data.csv", output_dir="../data/monte_carlo"):
        """
        初始化蒙特卡洛预测器
        
        Args:
            data_file: 数据文件路径
            output_dir: 输出目录
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.data = None
        
        # 双色球参数
        self.red_range = (1, 33)  # 红球范围1-33
        self.blue_range = (1, 16)  # 蓝球范围1-16
        
        # 蒙特卡洛参数
        self.default_simulations = 10000  # 默认模拟次数
        self.confidence_level = 0.95  # 置信水平
        
        # 概率分布参数
        self.red_probs = {}  # 红球概率分布
        self.blue_probs = {}  # 蓝球概率分布
        self.time_decay_factor = 0.95  # 时间衰减因子
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """加载和预处理数据"""
        try:
            self.data = pd.read_csv(self.data_file)
            
            # 处理日期列
            self.data['date'] = pd.to_datetime(self.data['date'], format='%Y-%m-%d', errors='coerce')
            
            # 拆分红球列为单独的列
            red_balls = self.data['red_balls'].str.split(',', expand=True)
            for i in range(6):
                self.data[f'red_{i+1}'] = red_balls[i].astype(int)
            
            # 转换蓝球为整数
            self.data['blue_ball'] = self.data['blue_ball'].astype(int)
            
            # 按日期排序（最新的在前）
            self.data = self.data.sort_values('date', ascending=False).reset_index(drop=True)
            
            print(f"成功加载{len(self.data)}条数据")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def calculate_probability_distribution(self, periods=500):
        """
        计算概率分布（带时间衰减权重）
        
        Args:
            periods: 分析期数
        """
        print("计算概率分布...")
        
        # 限制分析期数
        data = self.data.head(periods)
        
        # 初始化计数器
        red_counts = {i: 0 for i in range(1, 34)}
        blue_counts = {i: 0 for i in range(1, 17)}
        
        # 计算带时间衰减的权重
        total_weight = 0
        
        for idx, row in data.iterrows():
            # 时间衰减权重（越近期权重越高）
            weight = self.time_decay_factor ** idx
            total_weight += weight
            
            # 统计红球
            for i in range(1, 7):
                red_ball = row[f'red_{i}']
                red_counts[red_ball] += weight
            
            # 统计蓝球
            blue_ball = row['blue_ball']
            blue_counts[blue_ball] += weight
        
        # 计算概率
        total_red_weight = total_weight * 6  # 每期6个红球
        
        for ball in range(1, 34):
            self.red_probs[ball] = red_counts[ball] / total_red_weight
        
        for ball in range(1, 17):
            self.blue_probs[ball] = blue_counts[ball] / total_weight
        
        print("概率分布计算完成")
    
    def monte_carlo_simulation(self, num_simulations=10000):
        """
        蒙特卡洛模拟
        
        Args:
            num_simulations: 模拟次数
            
        Returns:
            模拟结果
        """
        print(f"开始蒙特卡洛模拟，模拟次数: {num_simulations}")
        
        # 准备概率数组
        red_numbers = list(range(1, 34))
        red_probabilities = [self.red_probs[i] for i in red_numbers]
        
        blue_numbers = list(range(1, 17))
        blue_probabilities = [self.blue_probs[i] for i in blue_numbers]
        
        # 模拟结果存储
        simulation_results = []
        
        for i in range(num_simulations):
            # 模拟红球（不重复抽取6个）
            red_balls = np.random.choice(
                red_numbers, 
                size=6, 
                replace=False, 
                p=red_probabilities
            )
            red_balls = sorted(red_balls.tolist())
            
            # 模拟蓝球
            blue_ball = np.random.choice(
                blue_numbers, 
                p=blue_probabilities
            )
            
            simulation_results.append((red_balls, blue_ball))
        
        print("蒙特卡洛模拟完成")
        return simulation_results
    
    def calculate_confidence_interval(self, simulation_results):
        """
        计算置信区间
        
        Args:
            simulation_results: 模拟结果
            
        Returns:
            置信区间信息
        """
        print("计算置信区间...")
        
        # 统计每个号码在模拟中的出现频次
        red_counts = {i: 0 for i in range(1, 34)}
        blue_counts = {i: 0 for i in range(1, 17)}
        
        for red_balls, blue_ball in simulation_results:
            for ball in red_balls:
                red_counts[ball] += 1
            blue_counts[blue_ball] += 1
        
        # 计算置信区间
        n_simulations = len(simulation_results)
        
        # 红球置信区间
        red_confidence = {}
        for ball in range(1, 34):
            count = red_counts[ball]
            prob = count / (n_simulations * 6)  # 每次模拟6个红球
            red_confidence[ball] = {
                'probability': prob,
                'count': count
            }
        
        # 蓝球置信区间
        blue_confidence = {}
        for ball in range(1, 17):
            count = blue_counts[ball]
            prob = count / n_simulations
            blue_confidence[ball] = {
                'probability': prob,
                'count': count
            }
        
        print("置信区间计算完成")
        return red_confidence, blue_confidence
    
    def select_best_combination(self, simulation_results, red_confidence, blue_confidence):
        """
        选择最佳组合
        
        Args:
            simulation_results: 模拟结果
            red_confidence: 红球置信区间
            blue_confidence: 蓝球置信区间
            
        Returns:
            最佳组合
        """
        # 根据概率选择红球
        red_scores = {}
        for ball in range(1, 34):
            prob = red_confidence[ball]['probability']
            red_scores[ball] = prob
        
        # 选择得分最高的6个红球
        sorted_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        selected_reds = sorted([ball for ball, score in sorted_reds[:6]])
        
        # 选择概率最高的蓝球
        blue_scores = {}
        for ball in range(1, 17):
            prob = blue_confidence[ball]['probability']
            blue_scores[ball] = prob
        
        selected_blue = max(blue_scores.items(), key=lambda x: x[1])[0]
        
        # 计算整体置信度
        red_confidence_avg = np.mean([red_confidence[ball]['probability'] for ball in selected_reds])
        blue_confidence_val = blue_confidence[selected_blue]['probability']
        overall_confidence = (red_confidence_avg * 6 + blue_confidence_val) / 7
        
        return selected_reds, selected_blue, overall_confidence
    
    def predict(self, num_predictions=1, num_simulations=None):
        """
        预测双色球号码
        
        Args:
            num_predictions: 预测注数
            num_simulations: 模拟次数
            
        Returns:
            预测结果列表
        """
        if self.data is None or self.data.empty:
            if not self.load_data():
                return None
        
        if num_simulations is None:
            num_simulations = self.default_simulations
        
        # 计算概率分布
        self.calculate_probability_distribution()
        
        predictions = []
        
        for i in range(num_predictions):
            print(f"生成第{i+1}注预测...")
            
            # 蒙特卡洛模拟
            simulation_results = self.monte_carlo_simulation(num_simulations)
            
            # 计算置信区间
            red_confidence, blue_confidence = self.calculate_confidence_interval(simulation_results)
            
            # 选择最佳组合
            red_balls, blue_ball, confidence = self.select_best_combination(
                simulation_results, red_confidence, blue_confidence
            )
            
            predictions.append((red_balls, blue_ball, confidence))
        
        return predictions


def format_ssq_numbers(red_balls, blue_ball):
    """格式化双色球号码显示"""
    red_str = ' '.join([f'{ball:02d}' for ball in red_balls])
    return f"前区 {red_str} | 后区 {blue_ball:02d}"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='双色球蒙特卡洛模拟预测器')
    parser.add_argument('-n', '--num', type=int, default=1, help='预测注数，默认1注')
    parser.add_argument('-s', '--simulations', type=int, default=10000, help='模拟次数，默认10000次')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 设置默认路径
    data_file = os.path.join(project_root, "data", "ssq_data.csv")
    output_dir = os.path.join(project_root, "data", "monte_carlo")
    
    # 创建预测器实例
    predictor = SSQMonteCarloPredictor(data_file=data_file, output_dir=output_dir)
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    # 预测号码
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
    else:
        print("预测失败，请检查数据文件")


if __name__ == "__main__":
    main()
