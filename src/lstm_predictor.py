#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM深度学习预测器
基于KittenCN/predict_Lottery_ticket项目的LSTM模型
使用TensorFlow + Keras实现双色球号码预测
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

# 深度学习相关导入
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: TensorFlow未安装，LSTM预测功能将不可用")
    print("请安装TensorFlow: pip install tensorflow")


class SSQLSTMPredictor:
    """双色球LSTM深度学习预测器"""

    def __init__(self, data_file="../data/ssq_data.csv", model_dir="../data/models"):
        """
        初始化LSTM预测器

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

        # LSTM模型参数
        self.window_sizes = [3, 5, 7]  # 多窗口尺寸
        self.red_models = {}  # 红球模型字典
        self.blue_models = {}  # 蓝球模型字典
        self.red_scalers = {}  # 红球数据缩放器
        self.blue_scalers = {}  # 蓝球数据缩放器

        # 确保模型目录存在
        os.makedirs(self.model_dir, exist_ok=True)

        # 检查GPU可用性
        self.use_gpu = len(tf.config.list_physical_devices('GPU')) > 0 if TENSORFLOW_AVAILABLE else False
        if self.use_gpu:
            print("检测到GPU，将使用GPU加速训练")
        else:
            print("未检测到GPU，将使用CPU训练")

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

    def prepare_sequences(self, data, window_size):
        """
        准备LSTM训练序列

        Args:
            data: 输入数据
            window_size: 窗口大小

        Returns:
            (X, y): 特征序列和目标值
        """
        X, y = [], []

        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    def create_lstm_model(self, input_shape, output_dim, use_bidirectional=False):
        """
        创建LSTM模型

        Args:
            input_shape: 输入形状
            output_dim: 输出维度
            use_bidirectional: 是否使用双向LSTM

        Returns:
            编译好的LSTM模型
        """
        model = Sequential()

        if use_bidirectional:
            # 双向LSTM（用于复杂模型）
            model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(64, return_sequences=False)))
            model.add(Dropout(0.2))
        else:
            # 单向LSTM（用于简单模型）
            model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dropout(0.2))

        # 输出层
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(output_dim, activation='sigmoid'))

        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def load_models(self):
        """加载已训练的模型"""
        print("加载LSTM模型...")

        for window_size in self.window_sizes:
            # 加载红球模型
            red_model_path = os.path.join(self.model_dir, f'red_lstm_w{window_size}.h5')
            red_scaler_path = os.path.join(self.model_dir, f'red_scaler_w{window_size}.pkl')

            if os.path.exists(red_model_path) and os.path.exists(red_scaler_path):
                self.red_models[window_size] = load_model(red_model_path)
                with open(red_scaler_path, 'rb') as f:
                    self.red_scalers[window_size] = pickle.load(f)
                print(f"加载红球窗口{window_size}模型成功")
            else:
                print(f"红球窗口{window_size}模型不存在，需要先训练")

            # 加载蓝球模型
            blue_model_path = os.path.join(self.model_dir, f'blue_lstm_w{window_size}.h5')
            blue_scaler_path = os.path.join(self.model_dir, f'blue_scaler_w{window_size}.pkl')

            if os.path.exists(blue_model_path) and os.path.exists(blue_scaler_path):
                self.blue_models[window_size] = load_model(blue_model_path)
                with open(blue_scaler_path, 'rb') as f:
                    self.blue_scalers[window_size] = pickle.load(f)
                print(f"加载蓝球窗口{window_size}模型成功")
            else:
                print(f"蓝球窗口{window_size}模型不存在，需要先训练")

    def predict_red_balls(self):
        """预测红球号码"""
        if not self.red_models:
            print("红球模型未加载，无法预测")
            return None

        predictions = []

        for window_size in self.window_sizes:
            if window_size not in self.red_models:
                continue

            model = self.red_models[window_size]
            scaler = self.red_scalers[window_size]

            # 准备最近的数据
            recent_data = []
            for i in range(window_size):
                if i < len(self.data):
                    red_balls = [self.data.iloc[i][f'red_{j}'] for j in range(1, 7)]
                    recent_data.append(red_balls)

            if len(recent_data) < window_size:
                continue

            recent_data = np.array(recent_data)
            scaled_data = scaler.transform(recent_data)

            # 预测
            X = scaled_data.reshape(1, window_size, 6)
            pred_scaled = model.predict(X, verbose=0)
            pred = scaler.inverse_transform(pred_scaled)[0]

            # 转换为整数并确保在有效范围内
            pred_ints = []
            for p in pred:
                p_int = max(1, min(33, round(p)))
                pred_ints.append(p_int)

            predictions.append(pred_ints)

        if not predictions:
            return None

        # 集成多个窗口的预测结果
        final_pred = self._ensemble_predictions(predictions, is_red=True)
        return sorted(list(set(final_pred)))[:6]  # 确保6个不重复的红球

    def predict_blue_ball(self):
        """预测蓝球号码"""
        if not self.blue_models:
            print("蓝球模型未加载，无法预测")
            return None

        predictions = []

        for window_size in self.window_sizes:
            if window_size not in self.blue_models:
                continue

            model = self.blue_models[window_size]
            scaler = self.blue_scalers[window_size]

            # 准备最近的数据
            recent_data = []
            for i in range(window_size):
                if i < len(self.data):
                    recent_data.append([self.data.iloc[i]['blue_ball']])

            if len(recent_data) < window_size:
                continue

            recent_data = np.array(recent_data)
            scaled_data = scaler.transform(recent_data)

            # 预测
            X = scaled_data.reshape(1, window_size, 1)
            pred_scaled = model.predict(X, verbose=0)
            pred = scaler.inverse_transform(pred_scaled)[0][0]

            # 转换为整数并确保在有效范围内
            pred_int = max(1, min(16, round(pred)))
            predictions.append(pred_int)

        if not predictions:
            return None

        # 返回最常见的预测结果
        from collections import Counter
        return Counter(predictions).most_common(1)[0][0]

    def _ensemble_predictions(self, predictions, is_red=True):
        """集成多个预测结果"""
        from collections import Counter

        if is_red:
            # 红球集成：统计每个号码的出现频次
            all_numbers = []
            for pred in predictions:
                all_numbers.extend(pred)

            # 获取最常见的6个号码
            counter = Counter(all_numbers)
            most_common = counter.most_common(6)

            result = [num for num, _ in most_common]

            # 如果不足6个，随机补充
            if len(result) < 6:
                remaining = [i for i in range(1, 34) if i not in result]
                import random
                result.extend(random.sample(remaining, 6 - len(result)))

            return result[:6]
        else:
            # 蓝球集成：返回最常见的预测
            return Counter(predictions).most_common(1)[0][0]

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

        # 加载模型
        if not self.red_models or not self.blue_models:
            self.load_models()

        # 如果没有可用的模型，使用随机预测
        if not self.red_models or not self.blue_models:
            print("警告: 没有可用的LSTM模型，使用随机预测")

        predictions = []

        for i in range(num_predictions):
            # 预测红球
            red_balls = self.predict_red_balls()
            if red_balls is None:
                # 使用随机预测作为后备
                import random
                red_balls = sorted(random.sample(range(1, 34), 6))

            # 确保红球数量正确且不重复
            if len(red_balls) < 6:
                remaining = [j for j in range(1, 34) if j not in red_balls]
                import random
                red_balls.extend(random.sample(remaining, 6 - len(red_balls)))

            red_balls = sorted(red_balls[:6])

            # 预测蓝球
            blue_ball = self.predict_blue_ball()
            if blue_ball is None:
                import random
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
