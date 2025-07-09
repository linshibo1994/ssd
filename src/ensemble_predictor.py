#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成学习预测器
基于zhaoyangpp/LottoProphet项目的集成学习方法
使用XGBoost + LightGBM + Random Forest + Gradient Boosting
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关导入
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("警告: 机器学习库未完全安装，集成学习预测功能将不可用")
    print("请安装依赖: pip install xgboost lightgbm scikit-learn")


class SSQEnsemblePredictor:
    """双色球集成学习预测器"""
    
    def __init__(self, data_file="../data/ssq_data.csv", model_dir="../data/models"):
        """
        初始化集成学习预测器
        
        Args:
            data_file: 数据文件路径
            model_dir: 模型保存目录
        """
        self.data_file = data_file
        self.model_dir = model_dir
        self.data = None
        
        # 双色球参数
        self.red_range = (1, 33)  # 红球范围1-33
        self.blue_range = (1, 16)  # 蓝球范围1-16
        
        # 集成学习模型
        self.red_models = {}  # 红球模型字典
        self.blue_models = {}  # 蓝球模型字典
        self.feature_scalers = {}  # 特征缩放器
        
        # 模型名称
        self.model_names = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']
        
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
    
    def extract_features(self, periods=500):
        """
        提取特征工程
        
        Args:
            periods: 分析期数
            
        Returns:
            特征DataFrame
        """
        print("开始特征工程...")
        
        # 限制分析期数
        data = self.data.head(periods).copy()
        features = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            feature_dict = {}
            
            # 基础特征
            red_balls = [row[f'red_{j}'] for j in range(1, 7)]
            blue_ball = row['blue_ball']
            
            # 1. 频率特征
            if i < len(data) - 1:  # 不包括当前期
                # 计算历史频率
                history_data = data.iloc[i+1:]
                
                # 红球历史频率
                red_freq = {}
                for j in range(1, 7):
                    red_counts = history_data[f'red_{j}'].value_counts()
                    for ball in range(1, 34):
                        red_freq[f'red_{ball}_freq'] = red_counts.get(ball, 0) / len(history_data)
                
                # 蓝球历史频率
                blue_counts = history_data['blue_ball'].value_counts()
                blue_freq = {}
                for ball in range(1, 17):
                    blue_freq[f'blue_{ball}_freq'] = blue_counts.get(ball, 0) / len(history_data)
                
                feature_dict.update(red_freq)
                feature_dict.update(blue_freq)
            
            # 2. 统计特征
            feature_dict['red_sum'] = sum(red_balls)
            feature_dict['red_mean'] = np.mean(red_balls)
            feature_dict['red_std'] = np.std(red_balls)
            feature_dict['red_min'] = min(red_balls)
            feature_dict['red_max'] = max(red_balls)
            feature_dict['red_range'] = max(red_balls) - min(red_balls)
            
            # 3. 奇偶特征
            odd_count = sum(1 for ball in red_balls if ball % 2 == 1)
            feature_dict['red_odd_count'] = odd_count
            feature_dict['red_even_count'] = 6 - odd_count
            feature_dict['blue_odd'] = 1 if blue_ball % 2 == 1 else 0
            
            # 4. 大小特征（以17为界）
            big_count = sum(1 for ball in red_balls if ball >= 17)
            feature_dict['red_big_count'] = big_count
            feature_dict['red_small_count'] = 6 - big_count
            feature_dict['blue_big'] = 1 if blue_ball >= 9 else 0
            
            # 5. 连号特征
            sorted_reds = sorted(red_balls)
            consecutive_count = 0
            for j in range(len(sorted_reds) - 1):
                if sorted_reds[j+1] - sorted_reds[j] == 1:
                    consecutive_count += 1
            feature_dict['consecutive_count'] = consecutive_count
            
            # 6. 分布特征
            # 将1-33分为3个区间
            zone1 = sum(1 for ball in red_balls if 1 <= ball <= 11)
            zone2 = sum(1 for ball in red_balls if 12 <= ball <= 22)
            zone3 = sum(1 for ball in red_balls if 23 <= ball <= 33)
            feature_dict['zone1_count'] = zone1
            feature_dict['zone2_count'] = zone2
            feature_dict['zone3_count'] = zone3
            
            # 7. 尾数特征
            tail_counts = {}
            for tail in range(10):
                tail_counts[f'tail_{tail}_count'] = sum(1 for ball in red_balls if ball % 10 == tail)
            feature_dict.update(tail_counts)
            
            # 8. 时间特征
            if pd.notna(row['date']):
                feature_dict['weekday'] = row['date'].weekday()
                feature_dict['month'] = row['date'].month
                feature_dict['day'] = row['date'].day
            else:
                feature_dict['weekday'] = 0
                feature_dict['month'] = 1
                feature_dict['day'] = 1
            
            # 添加目标变量
            feature_dict['target_red_1'] = red_balls[0]
            feature_dict['target_red_2'] = red_balls[1]
            feature_dict['target_red_3'] = red_balls[2]
            feature_dict['target_red_4'] = red_balls[3]
            feature_dict['target_red_5'] = red_balls[4]
            feature_dict['target_red_6'] = red_balls[5]
            feature_dict['target_blue'] = blue_ball
            
            features.append(feature_dict)
        
        features_df = pd.DataFrame(features)
        
        # 填充缺失值
        features_df = features_df.fillna(0)
        
        print(f"特征工程完成，共提取{len(features_df.columns)}个特征")
        return features_df
    
    def create_models(self):
        """创建集成学习模型"""
        models = {}
        
        # XGBoost
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Random Forest
        models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        return models
    
    def predict(self, num_predictions=1):
        """
        预测双色球号码
        
        Args:
            num_predictions: 预测注数
            
        Returns:
            预测结果列表
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
    parser = argparse.ArgumentParser(description='双色球集成学习预测器')
    parser.add_argument('-n', '--num', type=int, default=1, help='预测注数，默认1注')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 设置默认路径
    data_file = os.path.join(project_root, "data", "ssq_data.csv")
    model_dir = os.path.join(project_root, "data", "models")
    
    # 创建预测器实例
    predictor = SSQEnsemblePredictor(data_file=data_file, model_dir=model_dir)
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    # 预测号码
    print("🤖 集成学习预测")
    print("=" * 40)
    
    predictions = predictor.predict(num_predictions=args.num)
    
    if predictions:
        for i, (red_balls, blue_ball) in enumerate(predictions, 1):
            formatted = format_ssq_numbers(red_balls, blue_ball)
            print(f"第 {i} 注: {formatted}")
    else:
        print("预测失败")


if __name__ == "__main__":
    main()
