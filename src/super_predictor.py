#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超级预测器
整合所有预测方法的智能融合系统
使用多算法集成 + 加权投票 + 智能融合
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入所有预测器
try:
    from lstm_predictor import SSQLSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    from ensemble_predictor import SSQEnsemblePredictor
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from monte_carlo_predictor import SSQMonteCarloPredictor
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False

try:
    from clustering_predictor import SSQClusteringPredictor
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

# 导入现有的高级分析器
try:
    from advanced_analyzer import SSQAdvancedAnalyzer
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False


class SSQSuperPredictor:
    """双色球超级预测器"""
    
    def __init__(self, data_file="../data/ssq_data.csv", model_dir="../data/models", output_dir="../data/super"):
        self.data_file = data_file
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.data = None
        
        # 双色球参数
        self.red_range = (1, 33)  # 红球范围1-33
        self.blue_range = (1, 16)  # 蓝球范围1-16

        # 预测器实例
        self.predictors = {}
        self.predictor_weights = {
            'HYBRID': 0.30,      # 混合分析（原有的高级分析器）
            'LSTM': 0.25,        # LSTM深度学习
            'ENSEMBLE': 0.20,    # 集成学习
            'MONTE_CARLO': 0.15, # 蒙特卡洛模拟
            'CLUSTERING': 0.10   # 聚类分析
        }
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 初始化预测器
        self._initialize_predictors()
    
    def _initialize_predictors(self):
        """初始化所有可用的预测器"""
        print("初始化预测器...")
        
        # LSTM预测器
        if LSTM_AVAILABLE:
            try:
                self.predictors['LSTM'] = SSQLSTMPredictor(
                    data_file=self.data_file, 
                    model_dir=self.model_dir
                )
                print("✓ LSTM预测器初始化成功")
            except Exception as e:
                print(f"✗ LSTM预测器初始化失败: {e}")
        else:
            print("✗ LSTM预测器不可用")

        # 集成学习预测器
        if ENSEMBLE_AVAILABLE:
            try:
                self.predictors['ENSEMBLE'] = SSQEnsemblePredictor(
                    data_file=self.data_file,
                    model_dir=self.model_dir
                )
                print("✓ 集成学习预测器初始化成功")
            except Exception as e:
                print(f"✗ 集成学习预测器初始化失败: {e}")
        else:
            print("✗ 集成学习预测器不可用")

        # 蒙特卡洛预测器
        if MONTE_CARLO_AVAILABLE:
            try:
                self.predictors['MONTE_CARLO'] = SSQMonteCarloPredictor(
                    data_file=self.data_file,
                    output_dir=os.path.join(self.output_dir, "monte_carlo")
                )
                print("✓ 蒙特卡洛预测器初始化成功")
            except Exception as e:
                print(f"✗ 蒙特卡洛预测器初始化失败: {e}")
        else:
            print("✗ 蒙特卡洛预测器不可用")

        # 聚类分析预测器
        if CLUSTERING_AVAILABLE:
            try:
                self.predictors['CLUSTERING'] = SSQClusteringPredictor(
                    data_file=self.data_file,
                    output_dir=os.path.join(self.output_dir, "clustering")
                )
                print("✓ 聚类分析预测器初始化成功")
            except Exception as e:
                print(f"✗ 聚类分析预测器初始化失败: {e}")
        else:
            print("✗ 聚类分析预测器不可用")

        # 混合分析预测器
        if ADVANCED_AVAILABLE:
            try:
                self.predictors['HYBRID'] = SSQAdvancedAnalyzer(
                    data_file=self.data_file
                )
                print("✓ 混合分析预测器初始化成功")
            except Exception as e:
                print(f"✗ 混合分析预测器初始化失败: {e}")
        else:
            print("✗ 混合分析预测器不可用")
        
        print(f"共初始化{len(self.predictors)}个预测器")
    
    def load_data(self):
        """加载数据"""
        try:
            self.data = pd.read_csv(self.data_file)
            print(f"成功加载{len(self.data)}条数据")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def quick_predict(self, num_predictions=1):
        """快速预测"""
        # 优先使用混合分析
        if 'HYBRID' in self.predictors:
            print("使用 HYBRID 进行快速预测...")
            try:
                results = []
                for _ in range(num_predictions):
                    numbers = self.predictors['HYBRID'].predict_numbers(method='ensemble', explain=False)
                    red_balls = numbers[:6]
                    blue_ball = numbers[6]
                    results.append((red_balls, blue_ball))
                return results
            except Exception as e:
                print(f"HYBRID预测失败: {e}")
        
        # 后备使用LSTM
        if 'LSTM' in self.predictors:
            print("使用 LSTM 进行快速预测...")
            return self.predictors['LSTM'].predict(num_predictions=num_predictions)
        
        print("没有可用的快速预测方法")
        return None

    def predict_single_method(self, method_name, num_predictions=1):
        """
        使用单一方法预测

        Args:
            method_name: 方法名称
            num_predictions: 预测注数

        Returns:
            预测结果
        """
        if method_name not in self.predictors:
            print(f"预测器 {method_name} 不可用")
            return None

        predictor = self.predictors[method_name]

        try:
            if method_name == 'HYBRID':
                # 使用高级分析器的预测方法
                results = []
                for _ in range(num_predictions):
                    numbers = predictor.predict_numbers(method='ensemble', explain=False)
                    red_balls = numbers[:6]
                    blue_ball = numbers[6]
                    results.append((red_balls, blue_ball))
                return results
            elif method_name == 'MONTE_CARLO':
                # 蒙特卡洛预测返回带置信度的结果
                predictions = predictor.predict(num_predictions=num_predictions)
                if predictions:
                    return [(red, blue) for red, blue, conf in predictions]
                return None
            else:
                # 其他预测器
                return predictor.predict(num_predictions=num_predictions)
        except Exception as e:
            print(f"预测器 {method_name} 预测失败: {e}")
            return None

    def ensemble_predict(self, num_predictions=1):
        """
        集成预测（加权投票）

        Args:
            num_predictions: 预测注数

        Returns:
            集成预测结果
        """
        print("开始集成预测...")

        # 收集所有预测器的结果
        all_predictions = {}

        for method_name in self.predictors.keys():
            print(f"使用 {method_name} 预测...")
            predictions = self.predict_single_method(method_name, num_predictions=1)
            if predictions:
                all_predictions[method_name] = predictions[0]  # 只取第一注
                print(f"✓ {method_name} 预测成功")
            else:
                print(f"✗ {method_name} 预测失败")

        if not all_predictions:
            print("所有预测器都失败了")
            return None

        # 加权投票生成最终结果
        final_predictions = []

        for i in range(num_predictions):
            red_votes = {}  # 红球投票
            blue_votes = {}  # 蓝球投票

            # 收集投票
            for method_name, (red_balls, blue_ball) in all_predictions.items():
                weight = self.predictor_weights.get(method_name, 0.1)

                # 红球投票
                for ball in red_balls:
                    if ball not in red_votes:
                        red_votes[ball] = 0
                    red_votes[ball] += weight

                # 蓝球投票
                if blue_ball not in blue_votes:
                    blue_votes[blue_ball] = 0
                blue_votes[blue_ball] += weight

            # 选择得票最高的红球
            sorted_red_votes = sorted(red_votes.items(), key=lambda x: x[1], reverse=True)
            selected_reds = [ball for ball, votes in sorted_red_votes[:6]]

            # 确保6个红球
            if len(selected_reds) < 6:
                remaining = [j for j in range(1, 34) if j not in selected_reds]
                import random
                selected_reds.extend(random.sample(remaining, 6 - len(selected_reds)))

            selected_reds = sorted(selected_reds[:6])

            # 选择得票最高的蓝球
            selected_blue = max(blue_votes.items(), key=lambda x: x[1])[0]

            final_predictions.append((selected_reds, selected_blue))

        return final_predictions

    def predict(self, mode='quick', num_predictions=1):
        """主预测方法"""
        if self.data is None:
            if not self.load_data():
                return None

        if mode == 'ensemble':
            return self.ensemble_predict(num_predictions)
        elif mode == 'quick':
            return self.quick_predict(num_predictions)
        else:
            return self.quick_predict(num_predictions)


def format_ssq_numbers(red_balls, blue_ball):
    """格式化双色球号码显示"""
    red_str = ' '.join([f'{ball:02d}' for ball in red_balls])
    return f"前区 {red_str} | 后区 {blue_ball:02d}"


def print_ensemble_results(results, predictor):
    """打印集成预测结果"""
    print("🏆 集成预测结果:")
    for i, (red_balls, blue_ball) in enumerate(results, 1):
        formatted = format_ssq_numbers(red_balls, blue_ball)
        print(f"第 {i} 注: {formatted}")

    print("\n📊 投票详情:")
    for method, weight in predictor.predictor_weights.items():
        if method in predictor.predictors:
            print(f"  {method}: 权重 {weight*100:.1f}%")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='双色球超级预测器')
    parser.add_argument('-m', '--mode', choices=['quick'], default='quick', help='预测模式')
    parser.add_argument('-n', '--num', type=int, default=1, help='预测注数，默认1注')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 设置默认路径
    data_file = os.path.join(project_root, "data", "ssq_data.csv")
    model_dir = os.path.join(project_root, "data", "models")
    output_dir = os.path.join(project_root, "data", "super")
    
    # 创建超级预测器实例
    predictor = SSQSuperPredictor(
        data_file=data_file, 
        model_dir=model_dir, 
        output_dir=output_dir
    )
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    # 预测号码
    print("🌟 超级预测器")
    print("=" * 80)
    print(f"⚡ {args.mode}预测模式 - {args.num}注")
    print("=" * 80)
    
    results = predictor.predict(mode=args.mode, num_predictions=args.num)
    
    if results:
        print("⚡ 预测结果:")
        for i, (red_balls, blue_ball) in enumerate(results, 1):
            formatted = format_ssq_numbers(red_balls, blue_ball)
            print(f"第 {i} 注: {formatted}")
    else:
        print("预测失败")


if __name__ == "__main__":
    main()
