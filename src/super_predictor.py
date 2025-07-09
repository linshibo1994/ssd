#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超级预测器
整合所有预测方法的智能融合系统
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 导入预测器
try:
    from lstm_predictor import SSQLSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

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
        
        # 预测器实例
        self.predictors = {}
        self.predictor_weights = {
            'HYBRID': 0.50,      # 混合分析
            'LSTM': 0.50,        # LSTM深度学习
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
    
    def predict(self, mode='quick', num_predictions=1):
        """主预测方法"""
        if self.data is None:
            if not self.load_data():
                return None
        
        if mode == 'quick':
            return self.quick_predict(num_predictions)
        else:
            return self.quick_predict(num_predictions)


def format_ssq_numbers(red_balls, blue_ball):
    """格式化双色球号码显示"""
    red_str = ' '.join([f'{ball:02d}' for ball in red_balls])
    return f"前区 {red_str} | 后区 {blue_ball:02d}"


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
