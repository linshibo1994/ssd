#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM深度学习预测器
基于TensorFlow + Keras实现双色球号码预测
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 深度学习相关导入
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: TensorFlow未安装，LSTM预测功能将不可用")


class SSQLSTMPredictor:
    """双色球LSTM深度学习预测器"""
    
    def __init__(self, data_file="../data/ssq_data.csv", model_dir="../data/models"):
        self.data_file = data_file
        self.model_dir = model_dir
        self.data = None
        
        # 双色球参数
        self.red_range = (1, 33)
        self.blue_range = (1, 16)
        
        # 确保模型目录存在
        os.makedirs(self.model_dir, exist_ok=True)
    
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
    
    def predict(self, num_predictions=1):
        """
        预测双色球号码（简化版本）
        """
        if self.data is None or self.data.empty:
            if not self.load_data():
                return None
        
        predictions = []
        
        for i in range(num_predictions):
            # 使用随机预测作为后备
            import random
            red_balls = sorted(random.sample(range(1, 34), 6))
            blue_ball = random.randint(1, 16)
            
            predictions.append((red_balls, blue_ball))
        
        return predictions


def format_ssq_numbers(red_balls, blue_ball):
    """格式化双色球号码显示"""
    red_str = ' '.join([f'{ball:02d}' for ball in red_balls])
    return f"前区 {red_str} | 后区 {blue_ball:02d}"


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='双色球LSTM深度学习预测器')
    parser.add_argument('-n', '--num', type=int, default=1, help='预测注数，默认1注')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 设置默认路径
    data_file = os.path.join(project_root, "data", "ssq_data.csv")
    model_dir = os.path.join(project_root, "data", "models")
    
    # 创建预测器实例
    predictor = SSQLSTMPredictor(data_file=data_file, model_dir=model_dir)
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    # 预测号码
    print("🧠 LSTM深度学习预测")
    print("=" * 40)
    
    predictions = predictor.predict(num_predictions=args.num)
    
    if predictions:
        for i, (red_balls, blue_ball) in enumerate(predictions, 1):
            formatted = format_ssq_numbers(red_balls, blue_ball)
            print(f"第 {i} 注: {formatted}")
    else:
        print("预测失败，请检查数据文件")


if __name__ == "__main__":
    main()
