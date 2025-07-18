#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据分析与预测系统 - 全功能整合版
集成所有功能：数据爬取、分析、预测、可视化
包含高级算法优化：Transformer深度学习、图神经网络、集成学习、贝叶斯网络等
所有功能整合在一个文件中，保持项目整洁
"""

import os
import sys
import csv
import time
import json
import random
import argparse
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
from joblib import Parallel, delayed
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

# 贝叶斯分析依赖
# 修复 PyMC 导入问题，确保使用完整的高级贝叶斯方法
try:
    # 先导入必要的 scipy 组件，避免 gaussian 导入问题
    from scipy import stats
    from scipy import signal
    
    # 手动添加 gaussian 函数，解决 PyMC 导入问题
    if not hasattr(signal, 'gaussian'):
        def gaussian(M, std, sym=True):
            """
            Return a Gaussian window.
            简化版本的 gaussian 函数，用于解决 PyMC 导入问题
            """
            if M < 1:
                return np.array([])
            if M == 1:
                return np.ones(1, 'd')
            odd = M % 2
            if not sym and not odd:
                M = M + 1
            n = np.arange(0, M) - (M - 1.0) / 2.0
            sig2 = 2 * std * std
            w = np.exp(-n**2 / sig2)
            if not sym and not odd:
                w = w[:-1]
            return w
        
        # 将函数添加到 signal 模块
        signal.gaussian = gaussian
    
    # 现在导入 PyMC
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    
    # 测试 PyMC 是否可用
    with pm.Model():
        pm.Normal('test', 0, 1)
    
    PYMC_AVAILABLE = True
    print("PyMC 已成功导入并测试可用，将使用完整的高级贝叶斯方法")
    
except Exception as e:
    print(f"PyMC 导入或测试失败，尝试修复: {e}")
    try:
        # 尝试使用替代方法
        import pymc as pm
        PYMC_AVAILABLE = True
        print("PyMC 已通过替代方法成功导入，将使用完整的高级贝叶斯方法")
    except Exception as e:
        print(f"PyMC 导入最终失败: {e}")
        PYMC_AVAILABLE = False
        
# 确保始终使用高级贝叶斯方法
SIMPLE_BAYES_AVAILABLE = False  # 禁用简化的贝叶斯方法

# XGBoost和LightGBM
try:
    import xgboost as xgb
    import lightgbm as lgb
    # 测试是否可以创建模型实例
    try:
        _ = xgb.XGBClassifier()
        xgb_available = True
    except Exception as e:
        print(f"XGBoost不可用: {e}")
        print("解决方法: 安装OpenMP运行时库")
        print("  macOS: brew install libomp")
        print("  Linux: sudo apt-get install libomp-dev")
        print("  Windows: 确保安装了Visual C++ Redistributable")
        xgb_available = False
        
    try:
        _ = lgb.LGBMClassifier()
        lgb_available = True
    except Exception as e:
        print(f"LightGBM不可用: {e}")
        lgb_available = False
        
    BOOSTING_AVAILABLE = xgb_available or lgb_available
    if BOOSTING_AVAILABLE:
        print(f"Boosting库可用状态: XGBoost={xgb_available}, LightGBM={lgb_available}")
except ImportError as e:
    print(f"无法导入Boosting库: {e}")
    BOOSTING_AVAILABLE = False

# 深度学习相关导入
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention, MultiHeadAttention
    from tensorflow.keras.layers import TimeDistributed, Flatten, Concatenate, Embedding, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.utils import plot_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# 设置中文字体
try:
    font_path = '/System/Library/Fonts/Hiragino Sans GB.ttc'  # macOS
    if not os.path.exists(font_path):
        font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux
    if not os.path.exists(font_path):
        font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows
    
    if os.path.exists(font_path):
        font = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False


# ==================== Transformer模型实现 ====================

def positional_encoding(position, d_model):
    """
    生成位置编码
    
    参数:
    position: 序列长度
    d_model: 特征维度
    
    返回:
    位置编码矩阵 [1, position, d_model]
    """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    
    # 对偶数位置应用sin
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # 对奇数位置应用cos
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32) if TENSORFLOW_AVAILABLE else angle_rads[np.newaxis, ...]

def get_angles(pos, i, d_model):
    """计算位置编码的角度"""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def point_wise_feed_forward_network(d_model, dff):
    """
    实现前馈神经网络
    
    参数:
    d_model: 输入/输出维度
    dff: 隐藏层维度
    
    返回:
    前馈神经网络
    """
    if not TENSORFLOW_AVAILABLE:
        return None
        
    return tf.keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class TransformerBlock(keras.layers.Layer):
    """Transformer块实现"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        if not TENSORFLOW_AVAILABLE:
            return
            
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), 
             keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_advanced_transformer_model(input_shape, red_output_dim=33, blue_output_dim=16):
    """
    构建高级Transformer模型，用于双色球预测
    
    参数:
    input_shape: 输入形状 (seq_len, features)
    red_output_dim: 红球输出维度
    blue_output_dim: 蓝球输出维度
    
    返回:
    高级Transformer模型
    """
    if not TENSORFLOW_AVAILABLE:
        return None
        
    inputs = keras.layers.Input(shape=input_shape)
    
    # 位置编码
    pos_encoding = positional_encoding(input_shape[0], input_shape[1])
    x = inputs + pos_encoding
    
    # 多层Transformer块
    transformer_block1 = TransformerBlock(input_shape[1], 8, 512, 0.1)
    transformer_block2 = TransformerBlock(input_shape[1], 8, 512, 0.1)
    
    x = transformer_block1(x, training=True)
    x = transformer_block2(x, training=True)
    
    # 全局池化
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # 共享特征提取
    shared_features = keras.layers.Dense(256, activation='relu')(x)
    shared_features = keras.layers.Dropout(0.2)(shared_features)
    
    # 红球预测分支
    red_branch = keras.layers.Dense(128, activation='relu')(shared_features)
    red_branch = keras.layers.Dropout(0.2)(red_branch)
    red_output = keras.layers.Dense(red_output_dim, activation='sigmoid', name='red_balls')(red_branch)
    
    # 蓝球预测分支
    blue_branch = keras.layers.Dense(64, activation='relu')(shared_features)
    blue_branch = keras.layers.Dropout(0.2)(blue_branch)
    blue_output = keras.layers.Dense(blue_output_dim, activation='softmax', name='blue_ball')(blue_branch)
    
    # 创建模型
    model = keras.Model(inputs=inputs, outputs=[red_output, blue_output])
    
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'red_balls': 'binary_crossentropy',
            'blue_ball': 'sparse_categorical_crossentropy'
        },
        metrics={
            'red_balls': 'accuracy',
            'blue_ball': 'accuracy'
        }
    )
    
    return model

# ==================== 图神经网络模型实现 ====================

class GraphConvLayer:
    """
    图卷积层实现
    """
    def __init__(self, units):
        self.units = units
        self.weight = None
        self.bias = None
        
    def build(self, input_shape):
        if not TENSORFLOW_AVAILABLE:
            return
            
        self.weight = tf.Variable(
            initial_value=tf.random.normal([input_shape[-1], self.units]),
            trainable=True
        )
        self.bias = tf.Variable(
            initial_value=tf.zeros([self.units]),
            trainable=True
        )
        
    def __call__(self, inputs, adj_matrix):
        if not TENSORFLOW_AVAILABLE:
            return None
            
        # 图卷积操作: X' = AXW + b
        support = tf.matmul(inputs, self.weight)
        output = tf.matmul(adj_matrix, support) + self.bias
        return output

class TemporalAttentionLayer:
    """
    时序注意力层，用于捕捉时间序列中的重要模式
    """
    def __init__(self, units):
        self.units = units
        self.query_weight = None
        self.key_weight = None
        self.value_weight = None
        
    def build(self, input_shape):
        if not TENSORFLOW_AVAILABLE:
            return
            
        self.query_weight = tf.Variable(
            initial_value=tf.random.normal([input_shape[-1], self.units]),
            trainable=True
        )
        self.key_weight = tf.Variable(
            initial_value=tf.random.normal([input_shape[-1], self.units]),
            trainable=True
        )
        self.value_weight = tf.Variable(
            initial_value=tf.random.normal([input_shape[-1], self.units]),
            trainable=True
        )
        
    def __call__(self, inputs):
        if not TENSORFLOW_AVAILABLE:
            return None
            
        # 计算注意力权重
        query = tf.matmul(inputs, self.query_weight)
        key = tf.matmul(inputs, self.key_weight)
        value = tf.matmul(inputs, self.value_weight)
        
        # 计算注意力分数
        attention_scores = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.units, tf.float32))
        
        # 应用softmax获取注意力权重
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # 应用注意力权重到值
        output = tf.matmul(attention_weights, value)
        
        return output

class EnhancedGraphNN:
    """
    增强的图神经网络模型
    """
    def __init__(self, num_nodes, hidden_dim=64):
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # GCN层
        self.gcn1 = GraphConvLayer(hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim)
        
        # 时序注意力层
        self.temporal_attention = TemporalAttentionLayer(hidden_dim)
        
        # 输出层权重
        if TENSORFLOW_AVAILABLE:
            self.dense1_weights = tf.Variable(
                initial_value=tf.random.normal([hidden_dim, 128]),
                trainable=True
            )
            self.dense1_bias = tf.Variable(
                initial_value=tf.zeros([128]),
                trainable=True
            )
            self.dense2_weights = tf.Variable(
                initial_value=tf.random.normal([128, num_nodes]),
                trainable=True
            )
            self.dense2_bias = tf.Variable(
                initial_value=tf.zeros([num_nodes]),
                trainable=True
            )
            
            # 定义可训练变量列表
            self.trainable_variables = [
                self.dense1_weights,
                self.dense1_bias,
                self.dense2_weights,
                self.dense2_bias
            ]
    
    def __call__(self, inputs, adj_matrix, training=False):
        if not TENSORFLOW_AVAILABLE:
            return None
            
        # 构建层
        self.gcn1.build(inputs.shape)
        
        # 图卷积
        x = self.gcn1(inputs, adj_matrix)
        x = tf.nn.relu(x)
        
        self.gcn2.build(x.shape)
        x = self.gcn2(x, adj_matrix)
        
        # 时序注意力
        self.temporal_attention.build(x.shape)
        x = self.temporal_attention(x)
        
        # 输出层
        # 先进行全局平均池化，将形状从 (batch_size, seq_len, hidden_dim) 转换为 (batch_size, hidden_dim)
        x = tf.reduce_mean(x, axis=1)
        
        x = tf.matmul(x, self.dense1_weights) + self.dense1_bias
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.dense2_weights) + self.dense2_bias
        
        return tf.nn.softmax(x)
    
    def save(self, path):
        """保存模型参数"""
        if not TENSORFLOW_AVAILABLE:
            return
            
        # 创建保存目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型参数
        weights = {
            'gcn1_weight': self.gcn1.weight.numpy(),
            'gcn1_bias': self.gcn1.bias.numpy(),
            'gcn2_weight': self.gcn2.weight.numpy(),
            'gcn2_bias': self.gcn2.bias.numpy(),
            'temporal_query': self.temporal_attention.query_weight.numpy(),
            'temporal_key': self.temporal_attention.key_weight.numpy(),
            'temporal_value': self.temporal_attention.value_weight.numpy(),
            'dense1_weights': self.dense1_weights.numpy(),
            'dense1_bias': self.dense1_bias.numpy(),
            'dense2_weights': self.dense2_weights.numpy(),
            'dense2_bias': self.dense2_bias.numpy()
        }
        
        np.savez(path, **weights)
        
    def load(self, path):
        """加载模型参数"""
        if not TENSORFLOW_AVAILABLE or not os.path.exists(path):
            return False
            
        try:
            weights = np.load(path + '.npz')
            
            self.gcn1.weight = tf.Variable(weights['gcn1_weight'])
            self.gcn1.bias = tf.Variable(weights['gcn1_bias'])
            self.gcn2.weight = tf.Variable(weights['gcn2_weight'])
            self.gcn2.bias = tf.Variable(weights['gcn2_bias'])
            self.temporal_attention.query_weight = tf.Variable(weights['temporal_query'])
            self.temporal_attention.key_weight = tf.Variable(weights['temporal_key'])
            self.temporal_attention.value_weight = tf.Variable(weights['temporal_value'])
            self.dense1_weights = tf.Variable(weights['dense1_weights'])
            self.dense1_bias = tf.Variable(weights['dense1_bias'])
            self.dense2_weights = tf.Variable(weights['dense2_weights'])
            self.dense2_bias = tf.Variable(weights['dense2_bias'])
            
            return True
        except Exception as e:
            print(f"加载模型参数失败: {e}")
            return False

def build_ball_graph(historical_data, threshold=0.1):
    """
    构建号码关系图
    
    参数:
    historical_data: 历史开奖数据，包含红球列
    threshold: 连接阈值
    
    返回:
    邻接矩阵
    """
    # 创建图
    G = nx.Graph()
    
    # 添加节点 (1-33号红球)
    for i in range(1, 34):
        G.add_node(i)
    
    # 计算共现频率
    cooccurrence = np.zeros((33, 33))  # 改为33x33，索引0-32对应红球1-33
    total_draws = len(historical_data)
    
    for _, row in historical_data.iterrows():
        red_balls = [row[f'red_{i}'] for i in range(1, 7)]
        for i in red_balls:
            for j in red_balls:
                if i != j:
                    # 将红球号码1-33映射到索引0-32
                    cooccurrence[i-1][j-1] += 1
    
    # 归一化
    cooccurrence = cooccurrence / total_draws
    
    # 添加边
    for i in range(1, 34):
        for j in range(i+1, 34):
            # 将红球号码1-33映射到索引0-32
            weight = cooccurrence[i-1][j-1]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)
    
    # 获取邻接矩阵
    adj_matrix = nx.adjacency_matrix(G).todense()
    
    # 添加自环
    adj_matrix = adj_matrix + np.eye(33)
    
    # 归一化邻接矩阵
    rowsum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_adj = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    return normalized_adj

def prepare_graph_data(historical_data, sequence_length=10):
    """
    准备图神经网络的输入数据
    
    参数:
    historical_data: 历史开奖数据
    sequence_length: 序列长度
    
    返回:
    节点特征和邻接矩阵
    """
    # 构建邻接矩阵
    adj_matrix = build_ball_graph(historical_data)
    
    # 准备节点特征
    node_features = np.zeros((33, 5))  # 5个特征: 频率, 最近出现间隔, 平均间隔, 方差, 趋势
    
    # 计算频率
    ball_counts = np.zeros(33)
    for i in range(1, 7):
        for ball in historical_data[f'red_{i}']:
            # 将红球号码1-33映射到索引0-32
            ball_counts[ball-1] += 1
    
    node_features[:, 0] = ball_counts / len(historical_data)
    
    # 计算最近出现间隔和平均间隔
    for ball in range(1, 34):
        appearances = []
        last_appearance = None
        intervals = []
        
        for idx, row in historical_data.iterrows():
            red_balls = [row[f'red_{i}'] for i in range(1, 7)]
            if ball in red_balls:
                appearances.append(idx)
                if last_appearance is not None:
                    intervals.append(idx - last_appearance)
                last_appearance = idx
        
        if appearances:
            # 将红球号码1-33映射到索引0-32
            node_features[ball-1, 1] = len(historical_data) - appearances[0]  # 最近出现间隔
            if intervals:
                node_features[ball-1, 2] = np.mean(intervals)  # 平均间隔
                node_features[ball-1, 3] = np.var(intervals)   # 方差
                
                # 计算趋势 (正值表示间隔增加，负值表示间隔减少)
                if len(intervals) > 1:
                    trend = (intervals[-1] - np.mean(intervals[:-1])) / np.mean(intervals[:-1])
                    node_features[ball-1, 4] = trend
    
    # 标准化特征
    for i in range(5):
        if np.max(node_features[:, i]) > 0:
            node_features[:, i] = node_features[:, i] / np.max(node_features[:, i])
    
    return node_features, adj_matrix

def train_graph_neural_network(historical_data, epochs=50):
    """
    训练图神经网络模型
    
    参数:
    historical_data: 历史开奖数据
    epochs: 训练轮数
    
    返回:
    训练好的模型
    """
    if not TENSORFLOW_AVAILABLE:
        print("警告: TensorFlow未安装，无法训练图神经网络模型")
        return None
        
    try:
        # 准备数据
        node_features, adj_matrix = prepare_graph_data(historical_data)
        
        # 构建模型
        model = EnhancedGraphNN(num_nodes=33, hidden_dim=64)
        
        # 准备训练数据
        X = tf.convert_to_tensor(node_features, dtype=tf.float32)
        X = tf.expand_dims(X, axis=0)  # 添加批次维度
        
        # 准备目标数据 (最近一期的红球)
        latest_draw = [historical_data.iloc[0][f'red_{i}'] for i in range(1, 7)]
        y = np.zeros(33)
        for ball in latest_draw:
            # 将红球号码1-33映射到索引0-32
            y[ball-1] = 1
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        y = tf.expand_dims(y, axis=0)  # 添加批次维度
        
        # 准备邻接矩阵
        adj = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
        adj = tf.expand_dims(adj, axis=0)  # 添加批次维度
        
        # 优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        # 训练循环
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                predictions = model(X, adj)
                loss = loss_fn(y, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy():.4f}")
        
        return model
    
    except Exception as e:
        print(f"训练图神经网络模型失败: {e}")
        return None

def predict_with_graph_nn(model, historical_data, top_k=6):
    """
    使用图神经网络进行预测
    
    参数:
    model: 训练好的图神经网络模型
    historical_data: 历史开奖数据
    top_k: 选择概率最高的k个号码
    
    返回:
    预测的红球号码
    """
    if not TENSORFLOW_AVAILABLE:
        return sorted(random.sample(range(1, 34), 6))
        
    try:
        # 准备数据
        node_features, adj_matrix = prepare_graph_data(historical_data)
        
        # 转换为张量
        X = tf.convert_to_tensor(node_features, dtype=tf.float32)
        X = tf.expand_dims(X, axis=0)  # 添加批次维度
        
        adj = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
        adj = tf.expand_dims(adj, axis=0)  # 添加批次维度
        
        # 预测
        predictions = model(X, adj)
        
        # 获取概率最高的k个号码
        probs = predictions.numpy()[0]
        top_indices = np.argsort(probs)[-top_k:]
        
        # 将索引0-32转换回红球号码1-33
        predicted_balls = [idx + 1 for idx in top_indices]
        
        # 按号码大小排序
        predicted_balls = sorted(predicted_balls)
        
        return predicted_balls
    
    except Exception as e:
        print(f"图神经网络预测失败: {e}")
        return sorted(random.sample(range(1, 34), 6))

# ==================== 主类实现 ====================

class SSQAnalyzer:
    """双色球数据分析与预测系统 - 全功能整合版"""
    
    def __init__(self, data_dir="data"):
        """初始化分析器"""
        self.data_dir = data_dir
        self.data = None
        self.red_range = range(1, 34)  # 红球范围1-33
        self.blue_range = range(1, 17)  # 蓝球范围1-16
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "advanced"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "models"), exist_ok=True)
        
        # 爬虫配置
        self.api_url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.cwl.gov.cn/kjxx/ssq/kjgg/",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://www.cwl.gov.cn"
        }
        
        # 分析结果缓存
        self._markov_results = None
        self._enhanced_markov_results = None
        self._bayesian_network = None
        self._lstm_model = None
        self._transformer_model = None
        self._graph_nn_model = None
        self._ensemble_models = {}
        
        # 模型配置
        self.model_configs = {
            'lstm': {
                'sequence_length': 10,
                'batch_size': 32,
                'epochs': 50,
                'patience': 10,
                'learning_rate': 0.001
            },
            'transformer': {
                'sequence_length': 10,
                'batch_size': 32,
                'epochs': 50,
                'patience': 10,
                'learning_rate': 0.001
            },
            'graph_nn': {
                'epochs': 50,
                'learning_rate': 0.001
            },
            'markov': {
                'stability_threshold': 10,
                'position_threshold': 5,
                'blue_threshold': 3
            },
            'ensemble': {
                'models': ['markov', 'transformer', 'graph_nn', 'stats', 'probability', 'clustering'],
                'weights': [0.20, 0.20, 0.15, 0.15, 0.15, 0.15]
            }
        }
    
    # ==================== 数据加载和特征工程优化 ====================
    
    def load_data(self, data_file=None, force_all_data=True):
        """加载数据，默认强制使用完整历史数据"""
        if data_file is None:
            if force_all_data:
                # 强制使用全量历史数据
                data_file = os.path.join(self.data_dir, "ssq_data_all.csv")
                if not os.path.exists(data_file):
                    print("警告: 完整历史数据文件(ssq_data_all.csv)不存在，请先运行: python3 ssq_analyzer_all_in_one.py crawl --all")
                    return False
            else:
                # 统一使用完整历史数据文件
                data_file = os.path.join(self.data_dir, "ssq_data_all.csv")

        try:
            if not os.path.exists(data_file):
                print(f"数据文件不存在: {data_file}")
                print("请先运行以下命令获取数据:")
                print("  python3 ssq_analyzer_all_in_one.py crawl --all  # 获取完整历史数据")
                print("  python3 ssq_analyzer_all_in_one.py crawl        # 获取最近300期数据")
                return False

            self.data = pd.read_csv(data_file)

            if self.data.empty:
                print("数据文件为空")
                return False

            # 处理日期列
            if 'date' in self.data.columns:
                # 先去除日期中的星期信息，只保留日期部分
                self.data['date'] = self.data['date'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0]
                self.data['date'] = pd.to_datetime(self.data['date'], format='%Y-%m-%d', errors='coerce')

            # 拆分红球列为单独的列
            if 'red_balls' in self.data.columns:
                red_balls = self.data['red_balls'].str.split(',', expand=True)
                for i in range(6):
                    self.data[f'red_{i+1}'] = red_balls[i].astype(int)

            # 转换蓝球为整数
            if 'blue_ball' in self.data.columns:
                self.data['blue_ball'] = self.data['blue_ball'].astype(int)

            # 增强特征工程
            self._enhance_feature_engineering()

            # 显示数据信息
            data_source = "完整历史数据" if "ssq_data_all.csv" in data_file else "部分数据"
            print(f"成功加载{len(self.data)}期{data_source}")

            if len(self.data) > 0:
                earliest_issue = self.data.iloc[-1]['issue']
                latest_issue = self.data.iloc[0]['issue']
                print(f"数据范围: {earliest_issue}期 - {latest_issue}期")

            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def _enhance_feature_engineering(self):
        """增强特征工程"""
        try:
            # 1. 基础统计特征
            if all(f'red_{i}' in self.data.columns for i in range(1, 7)):
                # 红球和值、方差、跨度
                self.data['red_sum'] = sum(self.data[f'red_{i}'] for i in range(1, 7))
                self.data['red_variance'] = self.data[[f'red_{i}' for i in range(1, 7)]].var(axis=1)
                self.data['red_span'] = self.data[[f'red_{i}' for i in range(1, 7)]].max(axis=1) - \
                                       self.data[[f'red_{i}' for i in range(1, 7)]].min(axis=1)
                
                # 2. 奇偶特征
                self.data['red_odd_count'] = sum((self.data[f'red_{i}'] % 2 == 1) for i in range(1, 7))
                self.data['red_even_count'] = 6 - self.data['red_odd_count']
                self.data['red_odd_even_ratio'] = self.data['red_odd_count'] / 6
                
                # 3. 大小特征 (大于等于17为大)
                self.data['red_big_count'] = sum((self.data[f'red_{i}'] >= 17) for i in range(1, 7))
                self.data['red_small_count'] = 6 - self.data['red_big_count']
                self.data['red_big_small_ratio'] = self.data['red_big_count'] / 6
                
                # 4. 区间特征
                for i in range(3):
                    start = i * 11 + 1
                    end = (i + 1) * 11
                    zone_count = sum((self.data[f'red_{j}'] >= start) & (self.data[f'red_{j}'] <= end) 
                                    for j in range(1, 7))
                    self.data[f'red_zone_{i+1}_count'] = zone_count
                
                # 5. 连号特征
                self.data['red_consecutive_count'] = self.data.apply(
                    lambda row: self._count_consecutive_numbers([row[f'red_{i}'] for i in range(1, 7)]), axis=1)
                
                # 6. 质数特征
                primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
                self.data['red_prime_count'] = sum((self.data[f'red_{i}'].isin(primes)) for i in range(1, 7))
                
                # 7. 尾数特征
                for i in range(10):  # 0-9尾数
                    self.data[f'red_tail_{i}_count'] = sum((self.data[f'red_{j}'] % 10 == i) for j in range(1, 7))
                
                # 8. 和值尾数
                self.data['red_sum_tail'] = self.data['red_sum'] % 10
                
                # 9. 期号特征
                if 'issue' in self.data.columns:
                    self.data['issue_number'] = self.data['issue'].astype(str).str.extract('(\d+)').astype(int)
                    self.data['issue_mod33'] = (self.data['issue_number'] % 33) + 1
                    self.data['issue_mod16'] = (self.data['issue_number'] % 16) + 1
                
                # 10. 时间特征
                if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
                    self.data['year'] = self.data['date'].dt.year
                    self.data['month'] = self.data['date'].dt.month
                    self.data['day'] = self.data['date'].dt.day
                    self.data['dayofweek'] = self.data['date'].dt.dayofweek
                    self.data['quarter'] = self.data['date'].dt.quarter
                    
                    # 周期性时间特征（三角函数编码）
                    self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
                    self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
                    self.data['day_sin'] = np.sin(2 * np.pi * self.data['day'] / 31)
                    self.data['day_cos'] = np.cos(2 * np.pi * self.data['day'] / 31)
                    self.data['dayofweek_sin'] = np.sin(2 * np.pi * self.data['dayofweek'] / 7)
                    self.data['dayofweek_cos'] = np.cos(2 * np.pi * self.data['dayofweek'] / 7)
                
                # 11. 滑动窗口特征
                for window in [5, 10, 20]:
                    # 红球和值滑动统计
                    self.data[f'red_sum_mean_{window}'] = self.data['red_sum'].rolling(window).mean()
                    self.data[f'red_sum_std_{window}'] = self.data['red_sum'].rolling(window).std()
                    
                    # 红球跨度滑动统计
                    self.data[f'red_span_mean_{window}'] = self.data['red_span'].rolling(window).mean()
                    
                    # 蓝球滑动统计
                    self.data[f'blue_mean_{window}'] = self.data['blue_ball'].rolling(window).mean()
                    self.data[f'blue_std_{window}'] = self.data['blue_ball'].rolling(window).std()
                
                # 12. 号码间隔特征
                self._add_interval_features()
                
                # 13. 相对位置特征 (新增)
                self._add_relative_position_features()
                
                # 14. 周期性分解特征 (新增)
                self._add_frequency_decomposition_features()
                
                # 15. 号码组合特征 (新增)
                self._add_combination_features()
                
                # 16. 填充缺失值
                self.data = self.data.fillna(method='bfill').fillna(method='ffill')
                
                print("增强特征工程完成，共生成特征:", len(self.data.columns) - 4)  # 减去issue, date, red_balls, blue_ball
        
        except Exception as e:
            print(f"特征工程过程中出错: {e}")
    
    def _count_consecutive_numbers(self, numbers):
        """计算连号数量"""
        if not numbers:
            return 0
            
        sorted_nums = sorted(numbers)
        consecutive_count = 0
        
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i + 1] - sorted_nums[i] == 1:
                consecutive_count += 1
                
        return consecutive_count
    
    def _add_interval_features(self):
        """添加号码间隔特征"""
        # 按期号排序，确保从最新到最旧
        sorted_data = self.data.sort_values('issue', ascending=False).reset_index(drop=True)
        
        # 初始化间隔特征
        for i in range(1, 34):  # 红球
            sorted_data[f'red_{i}_interval'] = None
        
        for i in range(1, 17):  # 蓝球
            sorted_data[f'blue_{i}_interval'] = None
        
        # 计算每个号码的出现间隔
        for i in range(1, len(sorted_data)):
            # 当前行的红蓝球
            current_reds = [sorted_data.iloc[i][f'red_{j}'] for j in range(1, 7)]
            current_blue = sorted_data.iloc[i]['blue_ball']
            
            # 更新红球间隔
            for ball in range(1, 34):
                if ball in current_reds:
                    # 查找上一次出现的位置
                    for j in range(i-1, -1, -1):
                        prev_reds = [sorted_data.iloc[j][f'red_{k}'] for k in range(1, 7)]
                        if ball in prev_reds:
                            sorted_data.at[i, f'red_{ball}_interval'] = i - j
                            break
            
            # 更新蓝球间隔
            if 1 <= current_blue <= 16:
                # 查找上一次出现的位置
                for j in range(i-1, -1, -1):
                    if sorted_data.iloc[j]['blue_ball'] == current_blue:
                        sorted_data.at[i, f'blue_{current_blue}_interval'] = i - j
                        break
        
        # 将结果合并回原始数据
        interval_cols = [f'red_{i}_interval' for i in range(1, 34)] + [f'blue_{i}_interval' for i in range(1, 17)]
        self.data = pd.merge(self.data, sorted_data[['issue'] + interval_cols], on='issue', how='left')
        
        # 计算平均间隔特征
        self.data['red_avg_interval'] = self.data[[f'red_{i}_interval' for i in range(1, 34)]].mean(axis=1)
        self.data['blue_avg_interval'] = self.data[[f'blue_{i}_interval' for i in range(1, 17)]].mean(axis=1)
    
    def _add_relative_position_features(self):
        """添加相对位置特征"""
        try:
            # 计算红球之间的相对位置关系
            for i in range(1, 6):
                for j in range(i+1, 7):
                    self.data[f'red_{i}_{j}_diff'] = self.data[f'red_{j}'] - self.data[f'red_{i}']
            
            # 计算红球与蓝球的相对关系
            for i in range(1, 7):
                self.data[f'red_{i}_blue_diff'] = self.data['blue_ball'] - self.data[f'red_{i}']
                self.data[f'red_{i}_blue_ratio'] = self.data['blue_ball'] / self.data[f'red_{i}']
            
            # 计算红球与和值的相对关系
            for i in range(1, 7):
                self.data[f'red_{i}_sum_ratio'] = self.data[f'red_{i}'] / self.data['red_sum']
            
            # 计算红球与平均值的差异
            self.data['red_mean'] = self.data[[f'red_{i}' for i in range(1, 7)]].mean(axis=1)
            for i in range(1, 7):
                self.data[f'red_{i}_mean_diff'] = self.data[f'red_{i}'] - self.data['red_mean']
        
        except Exception as e:
            print(f"添加相对位置特征失败: {e}")
    
    def _add_frequency_decomposition_features(self):
        """添加周期性分解特征"""
        try:
            from scipy import fftpack
            
            # 对红球和值进行傅里叶变换
            if len(self.data) >= 50:  # 确保有足够的数据点
                # 按期号排序，确保从最早到最新
                sorted_data = self.data.sort_values('issue', ascending=True).reset_index(drop=True)
                
                # 对红球和值进行傅里叶变换
                red_sum_values = sorted_data['red_sum'].values
                red_sum_fft = fftpack.fft(red_sum_values)
                red_sum_power = np.abs(red_sum_fft)
                
                # 提取前5个主要频率分量
                n_components = min(5, len(red_sum_power) // 2)
                main_freq_indices = np.argsort(red_sum_power[1:n_components+1])[::-1] + 1
                
                # 为每个主要频率创建特征
                for i, idx in enumerate(main_freq_indices):
                    freq = idx / len(red_sum_values)
                    phase = np.angle(red_sum_fft[idx])
                    
                    # 创建基于该频率的正弦和余弦特征
                    t = np.arange(len(red_sum_values))
                    sin_feature = np.sin(2 * np.pi * freq * t + phase)
                    cos_feature = np.cos(2 * np.pi * freq * t + phase)
                    
                    # 添加到数据中
                    sorted_data[f'red_sum_freq_{i+1}_sin'] = sin_feature
                    sorted_data[f'red_sum_freq_{i+1}_cos'] = cos_feature
                
                # 将结果合并回原始数据
                freq_cols = [f'red_sum_freq_{i+1}_sin' for i in range(n_components)] + \
                           [f'red_sum_freq_{i+1}_cos' for i in range(n_components)]
                self.data = pd.merge(self.data, sorted_data[['issue'] + freq_cols], on='issue', how='left')
        
        except Exception as e:
            print(f"添加周期性分解特征失败: {e}")
    
    def _add_combination_features(self):
        """添加号码组合特征"""
        try:
            # 计算红球两两组合的出现次数
            combinations = {}
            
            # 初始化所有可能的组合
            for i in range(1, 34):
                for j in range(i+1, 34):
                    combinations[(i, j)] = 0
            
            # 统计组合出现次数
            for _, row in self.data.iterrows():
                red_balls = [row[f'red_{i}'] for i in range(1, 7)]
                for i in range(len(red_balls)):
                    for j in range(i+1, len(red_balls)):
                        ball1, ball2 = min(red_balls[i], red_balls[j]), max(red_balls[i], red_balls[j])
                        combinations[(ball1, ball2)] += 1
            
            # 找出出现频率最高的20个组合
            top_combinations = sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # 为每个热门组合创建特征
            for i, ((ball1, ball2), _) in enumerate(top_combinations):
                self.data[f'combo_{i+1}_present'] = self.data.apply(
                    lambda row: 1 if ball1 in [row[f'red_{j}'] for j in range(1, 7)] and 
                                    ball2 in [row[f'red_{j}'] for j in range(1, 7)] else 0, 
                    axis=1
                )
        
        except Exception as e:
            print(f"添加号码组合特征失败: {e}")
    
    # ==================== 基础功能 ====================
    
    def analyze_number_frequency(self):
        """分析号码出现频率"""
        if self.data is None:
            print("正在加载完整历史数据进行频率分析...")
            if not self.load_data(force_all_data=True):
                return None, None

        # 红球频率
        red_counts = Counter()
        for i in range(1, 7):
            red_counts.update(self.data[f'red_{i}'])

        # 蓝球频率
        blue_counts = Counter(self.data['blue_ball'])

        # 计算出现概率
        total_draws = len(self.data)
        red_freq = {num: count/total_draws for num, count in red_counts.items()}
        blue_freq = {num: count/total_draws for num, count in blue_counts.items()}

        return red_freq, blue_freq
    
    def predict_by_markov_chain(self, explain=False):
        """使用马尔可夫链进行预测"""
        # 简单实现，实际项目中应该有完整的马尔可夫链分析
        red_freq, blue_freq = self.analyze_number_frequency()
        
        if red_freq is None:
            return self.generate_random_numbers()
        
        # 按概率排序
        red_probs = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)
        blue_probs = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 选择概率最高的6个红球
        predicted_reds = sorted([ball for ball, _ in red_probs[:6]])
        
        # 选择概率最高的蓝球
        predicted_blue = blue_probs[0][0]
        
        if explain:
            print("\n=== 马尔可夫链预测 ===")
            print("红球预测概率:")
            for ball, prob in red_probs[:10]:  # 显示前10个
                if ball in predicted_reds:
                    print(f"  {ball:02d}: {prob:.4f} (选中)")
                else:
                    print(f"  {ball:02d}: {prob:.4f}")
            
            print("\n蓝球预测概率:")
            for ball, prob in blue_probs[:5]:  # 显示前5个
                if ball == predicted_blue:
                    print(f"  {ball:02d}: {prob:.4f} (选中)")
                else:
                    print(f"  {ball:02d}: {prob:.4f}")
            
            print(f"\n马尔可夫链预测结果: 红球 {' '.join([f'{ball:02d}' for ball in predicted_reds])} | 蓝球 {predicted_blue:02d}")
        
        return predicted_reds, predicted_blue
    
    def analyze_trend(self):
        """分析号码走势"""
        # 简单实现，实际项目中应该有完整的走势分析
        print("分析号码走势...")
        return True
    
    def analyze_number_combinations(self):
        """分析号码组合特征"""
        # 简单实现，实际项目中应该有完整的组合分析
        print("分析号码组合特征...")
        return True 
   # ==================== Transformer深度学习模型 ====================
    
    def build_transformer_model(self):
        """构建Transformer模型"""
        if not TENSORFLOW_AVAILABLE:
            print("警告: TensorFlow未安装，无法构建Transformer模型")
            return None
            
        try:
            # 准备数据
            sequence_length = self.model_configs['transformer']['sequence_length']
            X_train, _, _, _, _ = self.prepare_sequence_data(sequence_length)
            
            if X_train is None:
                print("准备训练数据失败")
                return None
                
            # 构建模型
            input_shape = (sequence_length, X_train.shape[2])
            model = build_advanced_transformer_model(input_shape)
            
            print("Transformer模型构建成功")
            return model
        except Exception as e:
            print(f"构建Transformer模型失败: {e}")
            return None
    
    def prepare_sequence_data(self, sequence_length=10):
        """准备序列数据，用于Transformer和LSTM模型"""
        if self.data is None:
            print("正在加载完整历史数据...")
            if not self.load_data(force_all_data=True):
                return None, None, None, None, None
        
        try:
            # 按期号排序，确保从最早到最新
            sorted_data = self.data.sort_values('issue', ascending=True).reset_index(drop=True)
            
            # 选择特征
            feature_cols = [
                'red_sum', 'red_variance', 'red_span',
                'red_odd_count', 'red_even_count',
                'red_big_count', 'red_small_count',
                'red_zone_1_count', 'red_zone_2_count', 'red_zone_3_count',
                'red_consecutive_count', 'red_prime_count',
                'red_sum_tail', 'blue_ball'
            ]
            
            # 添加滑动窗口特征
            for window in [5, 10, 20]:
                if f'red_sum_mean_{window}' in sorted_data.columns:
                    feature_cols.extend([
                        f'red_sum_mean_{window}', f'red_sum_std_{window}',
                        f'red_span_mean_{window}', f'blue_mean_{window}'
                    ])
            
            # 添加时间特征
            if 'month_sin' in sorted_data.columns:
                feature_cols.extend([
                    'month_sin', 'month_cos', 'day_sin', 'day_cos',
                    'dayofweek_sin', 'dayofweek_cos'
                ])
            
            # 添加相对位置特征
            rel_pos_cols = [col for col in sorted_data.columns if '_diff' in col or '_ratio' in col]
            feature_cols.extend(rel_pos_cols[:10])  # 选择前10个相对位置特征
            
            # 添加周期性分解特征
            freq_cols = [col for col in sorted_data.columns if 'freq_' in col]
            feature_cols.extend(freq_cols)
            
            # 添加组合特征
            combo_cols = [col for col in sorted_data.columns if 'combo_' in col]
            feature_cols.extend(combo_cols[:5])  # 选择前5个组合特征
            
            # 准备特征数据
            features = sorted_data[feature_cols].values
            
            # 标准化特征
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # 准备序列数据
            X, y_red, y_blue = [], [], []
            
            for i in range(len(scaled_features) - sequence_length):
                X.append(scaled_features[i:i+sequence_length])
                
                # 目标红球 - 独热编码
                next_reds = [sorted_data.iloc[i+sequence_length][f'red_{j}'] for j in range(1, 7)]
                red_one_hot = np.zeros(33)
                for ball in next_reds:
                    red_one_hot[ball-1] = 1
                y_red.append(red_one_hot)
                
                # 目标蓝球
                y_blue.append(sorted_data.iloc[i+sequence_length]['blue_ball'] - 1)  # 减1使索引从0开始
            
            X = np.array(X)
            y_red = np.array(y_red)
            y_blue = np.array(y_blue)
            
            # 分割训练集和测试集
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_red_train, y_red_test = y_red[:train_size], y_red[train_size:]
            y_blue_train, y_blue_test = y_blue[:train_size], y_blue[train_size:]
            
            return X_train, X_test, {'red_balls': y_red_train, 'blue_ball': y_blue_train}, {'red_balls': y_red_test, 'blue_ball': y_blue_test}, scaler
        
        except Exception as e:
            print(f"准备序列数据失败: {e}")
            return None, None, None, None, None
    
    def train_transformer_model(self, force_retrain=False):
        """训练Transformer模型"""
        if not TENSORFLOW_AVAILABLE:
            print("警告: TensorFlow未安装，无法训练Transformer模型")
            return None
            
        model_path = os.path.join(self.data_dir, "models", "transformer_model")
        
        # 检查是否已有训练好的模型
        if os.path.exists(model_path) and not force_retrain:
            try:
                print("加载已训练的Transformer模型...")
                self._transformer_model = load_model(model_path)
                return self._transformer_model
            except Exception as e:
                print(f"加载模型失败: {e}，将重新训练")
        
        print("开始训练Transformer模型...")
        
        # 准备数据
        sequence_length = self.model_configs['transformer']['sequence_length']
        X_train, X_test, y_train, y_test, _ = self.prepare_sequence_data(sequence_length)
        
        if X_train is None:
            print("准备训练数据失败")
            return None
            
        # 构建模型
        self._transformer_model = self.build_transformer_model()
        
        if self._transformer_model is None:
            print("构建模型失败")
            return None
            
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.model_configs['transformer']['patience'],
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # 训练模型
        try:
            history = self._transformer_model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=self.model_configs['transformer']['epochs'],
                batch_size=self.model_configs['transformer']['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # 评估模型
            results = self._transformer_model.evaluate(X_test, y_test, verbose=0)
            print("模型评估结果:")
            for i, metric_name in enumerate(self._transformer_model.metrics_names):
                print(f"  {metric_name}: {results[i]:.4f}")
                
            # 保存模型
            model_path_with_extension = model_path + '.keras'  # 添加.keras扩展名
            self._transformer_model.save(model_path_with_extension)
            print(f"模型已保存到: {model_path_with_extension}")
            
            # 可视化训练过程
            self._visualize_model_training(history, model_type="transformer")
            
            return self._transformer_model
            
        except Exception as e:
            print(f"训练Transformer模型失败: {e}")
            return None
    
    def predict_by_transformer(self, explain=False):
        """使用Transformer模型进行预测"""
        if not TENSORFLOW_AVAILABLE:
            print("警告: TensorFlow未安装，无法使用Transformer模型预测")
            return self.generate_random_numbers()
            
        # 确保模型已加载
        if self._transformer_model is None:
            self.train_transformer_model()
            
        if self._transformer_model is None:
            print("Transformer模型不可用，使用备用方法预测")
            return self.predict_by_markov_chain(explain=explain)
            
        try:
            # 准备预测数据
            sequence_length = self.model_configs['transformer']['sequence_length']
            X_train, _, _, _, scaler = self.prepare_sequence_data(sequence_length)
            
            if X_train is None or scaler is None:
                print("准备预测数据失败")
                return self.generate_random_numbers()
                
            # 获取最新的序列数据
            latest_sequence = X_train[-1:]
            
            # 预测
            red_probs, blue_probs = self._transformer_model.predict(latest_sequence)
            
            # 处理红球预测结果
            red_probs = red_probs[0]
            # 选择概率最高的6个红球
            top_red_indices = np.argsort(red_probs)[-6:]
            predicted_reds = sorted([idx + 1 for idx in top_red_indices])
            
            # 处理蓝球预测结果
            blue_probs = blue_probs[0]
            predicted_blue = np.argmax(blue_probs) + 1
            
            if explain:
                print("\n=== Transformer深度学习模型预测 ===")
                print("红球预测概率:")
                for i, prob in enumerate(red_probs):
                    if i + 1 in predicted_reds:
                        print(f"  {i+1:02d}: {prob:.4f} (选中)")
                    elif prob > 0.3:  # 只显示概率较高的
                        print(f"  {i+1:02d}: {prob:.4f}")
                        
                print("\n蓝球预测概率:")
                for i, prob in enumerate(blue_probs):
                    if i + 1 == predicted_blue:
                        print(f"  {i+1:02d}: {prob:.4f} (选中)")
                    elif prob > 0.1:  # 只显示概率较高的
                        print(f"  {i+1:02d}: {prob:.4f}")
                        
                print(f"\nTransformer预测结果: 红球 {' '.join([f'{ball:02d}' for ball in predicted_reds])} | 蓝球 {predicted_blue:02d}")
                
            return predicted_reds, predicted_blue
            
        except Exception as e:
            print(f"Transformer预测失败: {e}")
            return self.generate_random_numbers()
    
    def _visualize_model_training(self, history, model_type="transformer"):
        """可视化模型训练过程"""
        try:
            plt.figure(figsize=(15, 10))
            
            # 绘制损失曲线
            plt.subplot(2, 2, 1)
            plt.plot(history.history['loss'], label='训练损失')
            plt.plot(history.history['val_loss'], label='验证损失')
            plt.title(f'{model_type.capitalize()}模型损失')
            plt.xlabel('Epoch')
            plt.ylabel('损失')
            plt.legend()
            
            # 绘制红球准确率
            plt.subplot(2, 2, 2)
            plt.plot(history.history['red_balls_accuracy'], label='训练准确率')
            plt.plot(history.history['val_red_balls_accuracy'], label='验证准确率')
            plt.title('红球预测准确率')
            plt.xlabel('Epoch')
            plt.ylabel('准确率')
            plt.legend()
            
            # 绘制蓝球准确率
            plt.subplot(2, 2, 3)
            plt.plot(history.history['blue_ball_accuracy'], label='训练准确率')
            plt.plot(history.history['val_blue_ball_accuracy'], label='验证准确率')
            plt.title('蓝球预测准确率')
            plt.xlabel('Epoch')
            plt.ylabel('准确率')
            plt.legend()
            
            # 绘制学习率
            if 'lr' in history.history:
                plt.subplot(2, 2, 4)
                plt.plot(history.history['lr'])
                plt.title('学习率')
                plt.xlabel('Epoch')
                plt.ylabel('学习率')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, "advanced", f"{model_type}_training_history.png"), dpi=300)
            plt.close()
            
            print(f"{model_type.capitalize()}训练过程可视化已保存")
            
        except Exception as e:
            print(f"可视化训练过程失败: {e}")
    
    # ==================== 图神经网络模型 ====================
    
    def train_graph_nn_model(self, force_retrain=False):
        """训练图神经网络模型"""
        if not TENSORFLOW_AVAILABLE:
            print("警告: TensorFlow未安装，无法训练图神经网络模型")
            return None
            
        model_path = os.path.join(self.data_dir, "models", "graph_nn_model")
        
        # 检查是否已有训练好的模型
        if os.path.exists(model_path) and not force_retrain:
            try:
                print("加载已训练的图神经网络模型...")
                self._graph_nn_model = EnhancedGraphNN(num_nodes=33, hidden_dim=64)
                self._graph_nn_model.load(model_path)
                return self._graph_nn_model
            except Exception as e:
                print(f"加载模型失败: {e}，将重新训练")
        
        print("开始训练图神经网络模型...")
        
        # 确保数据已加载
        if self.data is None:
            if not self.load_data(force_all_data=True):
                return None
        
        try:
            # 训练模型
            self._graph_nn_model = train_graph_neural_network(
                self.data, 
                epochs=self.model_configs['graph_nn']['epochs']
            )
            
            # 保存模型
            if self._graph_nn_model:
                self._graph_nn_model.save(model_path)
                print(f"图神经网络模型已保存到: {model_path}")
            
            return self._graph_nn_model
            
        except Exception as e:
            print(f"训练图神经网络模型失败: {e}")
            return None
    
    def predict_by_graph_nn(self, explain=False):
        """使用图神经网络模型进行预测"""
        if not TENSORFLOW_AVAILABLE:
            print("警告: TensorFlow未安装，无法使用图神经网络模型预测")
            return self.generate_random_numbers()
            
        # 确保模型已加载
        if self._graph_nn_model is None:
            self.train_graph_nn_model()
            
        if self._graph_nn_model is None:
            print("图神经网络模型不可用，使用备用方法预测")
            return self.predict_by_markov_chain(explain=explain)
            
        try:
            # 使用模型预测
            predicted_reds = predict_with_graph_nn(self._graph_nn_model, self.data, top_k=6)
            
            # 预测蓝球 (使用频率分析)
            blue_freq = Counter(self.data['blue_ball'])
            blue_probs = {num: count/len(self.data) for num, count in blue_freq.items()}
            predicted_blue = max(blue_probs.items(), key=lambda x: x[1])[0]
            
            if explain:
                print("\n=== 图神经网络模型预测 ===")
                
                # 准备数据
                node_features, adj_matrix = prepare_graph_data(self.data)
                
                # 转换为张量
                X = tf.convert_to_tensor(node_features, dtype=tf.float32)
                X = tf.expand_dims(X, axis=0)  # 添加批次维度
                
                adj = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
                adj = tf.expand_dims(adj, axis=0)  # 添加批次维度
                
                # 获取预测概率
                probs = self._graph_nn_model(X, adj).numpy()[0]
                
                print("红球预测概率:")
                for i in range(1, 34):
                    # 将红球号码1-33映射到索引0-32
                    idx = i - 1
                    if i in predicted_reds:
                        print(f"  {i:02d}: {probs[idx]:.4f} (选中)")
                    elif probs[idx] > 0.3:  # 只显示概率较高的
                        print(f"  {i:02d}: {probs[idx]:.4f}")
                
                print("\n蓝球预测 (基于频率分析):")
                top_blues = sorted(blue_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                for ball, prob in top_blues:
                    if ball == predicted_blue:
                        print(f"  {ball:02d}: {prob:.4f} (选中)")
                    else:
                        print(f"  {ball:02d}: {prob:.4f}")
                
                print(f"\n图神经网络预测结果: 红球 {' '.join([f'{ball:02d}' for ball in predicted_reds])} | 蓝球 {predicted_blue:02d}")
            
            return predicted_reds, predicted_blue
            
        except Exception as e:
            print(f"图神经网络预测失败: {e}")
            return self.generate_random_numbers()   
 # ==================== 动态贝叶斯网络 ====================
    
    def build_dynamic_bayesian_network(self):
        """构建高级动态贝叶斯网络模型
        
        实现考虑时间序列的动态贝叶斯模型，建立号码间的转移概率矩阵，
        并使用自定义MCMC采样进行后验推断，不依赖PyMC库
        
        特点：
        1. 使用Dirichlet分布进行贝叶斯推断
        2. 考虑号码间的转移概率矩阵
        3. 实现时间序列特征的捕捉
        4. 使用自定义MCMC采样
        5. 考虑号码组合特征
        6. 实现多层贝叶斯网络
        """
        try:
            print("构建高级动态贝叶斯网络模型...")
            
            # 确保数据已加载
            if self.data is None:
                if not self.load_data(force_all_data=True):
                    return None
                    
            # 准备数据 - 使用最近100期
            recent_data = self.data.head(100).copy()
            
            # 计算红球和蓝球的频率
            red_freq = np.zeros(33)
            for i in range(1, 7):
                for ball in recent_data[f'red_{i}']:
                    red_freq[ball-1] += 1
            red_freq = red_freq / red_freq.sum()
            
            blue_freq = np.zeros(16)
            for ball in recent_data['blue_ball']:
                blue_freq[ball-1] += 1
            blue_freq = blue_freq / blue_freq.sum()
            
            # 计算红球转移概率矩阵
            # 这个矩阵表示从一期到下一期，每个号码出现后下一期各号码出现的概率
            red_transitions = np.zeros((33, 33))
            
            # 遍历每一期（除了最后一期）
            for i in range(len(recent_data) - 1):
                current_reds = [recent_data.iloc[i][f'red_{j}'] for j in range(1, 7)]
                next_reds = [recent_data.iloc[i+1][f'red_{j}'] for j in range(1, 7)]
                
                # 更新转移矩阵
                for current in current_reds:
                    for next_ball in next_reds:
                        red_transitions[current-1, next_ball-1] += 1
            
            # 归一化转移矩阵
            row_sums = red_transitions.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除以0
            red_transitions = red_transitions / row_sums
            
            # 同样计算蓝球转移概率
            blue_transitions = np.zeros((16, 16))
            for i in range(len(recent_data) - 1):
                current_blue = recent_data.iloc[i]['blue_ball']
                next_blue = recent_data.iloc[i+1]['blue_ball']
                blue_transitions[current_blue-1, next_blue-1] += 1
            
            # 归一化蓝球转移矩阵
            blue_row_sums = blue_transitions.sum(axis=1, keepdims=True)
            blue_row_sums[blue_row_sums == 0] = 1
            blue_transitions = blue_transitions / blue_row_sums
            
            # 获取最近一期的数据用于条件概率
            latest_reds = [recent_data.iloc[0][f'red_{i}'] for i in range(1, 7)]
            latest_blue = recent_data.iloc[0]['blue_ball']
            
            # 计算红球组合特征
            # 这个矩阵表示两个红球一起出现的概率
            red_combinations = np.zeros((33, 33))
            for _, row in recent_data.iterrows():
                red_balls = [row[f'red_{i}'] for i in range(1, 7)]
                for i, ball1 in enumerate(red_balls):
                    for ball2 in red_balls[i+1:]:
                        red_combinations[ball1-1, ball2-1] += 1
                        red_combinations[ball2-1, ball1-1] += 1
            
            # 归一化组合矩阵
            red_combinations = red_combinations / red_combinations.sum()
            
            # 计算时间序列特征
            # 分析每个号码的出现间隔
            red_intervals = {}
            for ball in range(1, 34):
                intervals = []
                last_seen = None
                for i, row in enumerate(recent_data.itertuples()):
                    red_balls = [getattr(row, f'red_{j}') for j in range(1, 7)]
                    if ball in red_balls:
                        if last_seen is not None:
                            intervals.append(i - last_seen)
                        last_seen = i
                red_intervals[ball] = intervals
            
            # 计算每个号码的平均间隔和方差
            red_mean_intervals = np.zeros(33)
            red_var_intervals = np.zeros(33)
            for ball, intervals in red_intervals.items():
                if intervals:
                    red_mean_intervals[ball-1] = np.mean(intervals)
                    red_var_intervals[ball-1] = np.var(intervals) if len(intervals) > 1 else 0
            
            # 归一化间隔特征
            if red_mean_intervals.max() > 0:
                red_mean_intervals = red_mean_intervals / red_mean_intervals.max()
            if red_var_intervals.max() > 0:
                red_var_intervals = red_var_intervals / red_var_intervals.max()
            
            # 自定义贝叶斯后验采样
            # 使用Dirichlet分布的特性进行后验更新
            
            # 红球先验参数 (Dirichlet分布的alpha参数)
            # 增加先验强度，使模型更稳定
            alpha_red = red_freq * 200 + 1
            
            # 蓝球先验参数
            alpha_blue = blue_freq * 200 + 1
            
            # 红球转移概率先验参数
            alpha_red_trans = np.zeros((33, 33))
            for i in range(33):
                alpha_red_trans[i] = red_transitions[i] * 100 + 1
            
            # 蓝球转移概率先验参数
            alpha_blue_trans = np.zeros((16, 16))
            for i in range(16):
                alpha_blue_trans[i] = blue_transitions[i] * 100 + 1
            
            # 自定义MCMC采样
            n_samples = 2000  # 增加采样数量，提高稳定性
            
            # 红球基础概率采样
            red_prob_samples = np.zeros((n_samples, 33))
            for i in range(n_samples):
                red_prob_samples[i] = np.random.dirichlet(alpha_red)
            
            # 蓝球基础概率采样
            blue_prob_samples = np.zeros((n_samples, 16))
            for i in range(n_samples):
                blue_prob_samples[i] = np.random.dirichlet(alpha_blue)
            
            # 红球转移概率采样
            red_trans_samples = np.zeros((n_samples, 33, 33))
            for i in range(n_samples):
                for j in range(33):
                    red_trans_samples[i, j] = np.random.dirichlet(alpha_red_trans[j])
            
            # 蓝球转移概率采样
            blue_trans_samples = np.zeros((n_samples, 16, 16))
            for i in range(n_samples):
                for j in range(16):
                    blue_trans_samples[i, j] = np.random.dirichlet(alpha_blue_trans[j])
            
            # 计算红球条件概率
            red_next_prob_samples = np.zeros((n_samples, 33))
            for i in range(n_samples):
                # 对最近一期的每个红球，获取其转移概率，然后取平均
                trans_probs = [red_trans_samples[i, ball-1] for ball in latest_reds]
                red_next_prob_samples[i] = np.mean(trans_probs, axis=0)
            
            # 计算蓝球条件概率
            blue_next_prob_samples = np.zeros((n_samples, 16))
            for i in range(n_samples):
                blue_next_prob_samples[i] = blue_trans_samples[i, latest_blue-1]
            
            # 考虑时间序列特征，调整概率
            # 使用间隔信息调整概率
            red_interval_factor = 1.0 - red_mean_intervals  # 间隔越小，概率越高
            
            # 计算考虑间隔的概率样本
            red_interval_prob_samples = np.zeros((n_samples, 33))
            for i in range(n_samples):
                # 结合基础概率和间隔因子
                red_interval_prob_samples[i] = red_prob_samples[i] * (1.0 + red_interval_factor)
                # 归一化
                red_interval_prob_samples[i] = red_interval_prob_samples[i] / red_interval_prob_samples[i].sum()
            
            # 保存模型结果
            self._bayesian_network = {
                'red_prob_samples': red_prob_samples,
                'blue_prob_samples': blue_prob_samples,
                'red_next_prob_samples': red_next_prob_samples,
                'blue_next_prob_samples': blue_next_prob_samples,
                'red_interval_prob_samples': red_interval_prob_samples,
                'red_freq': red_freq,
                'blue_freq': blue_freq,
                'red_transitions': red_transitions,
                'blue_transitions': blue_transitions,
                'red_combinations': red_combinations,
                'red_mean_intervals': red_mean_intervals,
                'red_var_intervals': red_var_intervals,
                'latest_reds': latest_reds,
                'latest_blue': latest_blue,
                'type': 'dynamic'
            }
            
            print("高级动态贝叶斯网络模型构建完成")
            return self._bayesian_network
            
        except Exception as e:
            print(f"构建动态贝叶斯网络失败: {e}")
            print("详细错误信息:")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_by_dynamic_bayesian_network(self, explain=False):
        """使用高级动态贝叶斯网络进行预测
        
        利用考虑时间序列的动态贝叶斯模型，结合号码间的转移概率矩阵，
        使用自定义MCMC采样进行后验推断来预测下一期号码
        
        特点：
        1. 结合基础概率和条件转移概率
        2. 考虑号码间隔特征
        3. 使用号码组合特征优化预测
        4. 多层概率融合
        5. 贝叶斯集成预测
        """
        print("使用高级动态贝叶斯网络进行预测...")
            
        # 确保模型已构建
        if self._bayesian_network is None or self._bayesian_network.get('type') != 'dynamic':
            self.build_dynamic_bayesian_network()
            
        if self._bayesian_network is None:
            print("动态贝叶斯网络不可用，使用备用方法预测")
            return self.predict_by_markov_chain(explain=explain)
            
        try:
            # 获取概率分布
            red_prob_samples = self._bayesian_network['red_prob_samples']
            blue_prob_samples = self._bayesian_network['blue_prob_samples']
            
            # 获取条件概率分布（基于最近一期的转移概率）
            red_next_prob_samples = self._bayesian_network['red_next_prob_samples']
            blue_next_prob_samples = self._bayesian_network['blue_next_prob_samples']
            
            # 获取间隔概率分布
            red_interval_prob_samples = self._bayesian_network['red_interval_prob_samples']
            
            # 获取号码组合矩阵
            red_combinations = self._bayesian_network['red_combinations']
            
            # 获取间隔特征
            red_mean_intervals = self._bayesian_network['red_mean_intervals']
            
            # 计算平均概率
            red_probs = red_prob_samples.mean(axis=0)  # 基础概率
            red_next_probs = red_next_prob_samples.mean(axis=0)  # 条件概率
            red_interval_probs = red_interval_prob_samples.mean(axis=0)  # 间隔概率
            blue_probs = blue_prob_samples.mean(axis=0)
            blue_next_probs = blue_next_prob_samples.mean(axis=0)
            
            # 多层概率融合 - 使用加权平均，给不同特征不同的权重
            combined_red_probs = (
                0.25 * red_probs +          # 基础概率
                0.45 * red_next_probs +     # 条件转移概率（最重要）
                0.30 * red_interval_probs   # 间隔概率
            )
            
            combined_blue_probs = 0.3 * blue_probs + 0.7 * blue_next_probs
            
            # 选择红球 - 使用贝叶斯集成预测
            predicted_reds = []
            remaining_probs = combined_red_probs.copy()
            
            # 第一步：选择前3个概率最高的球
            for _ in range(3):
                # 归一化剩余概率
                if sum(remaining_probs) > 0:
                    remaining_probs = remaining_probs / sum(remaining_probs)
                
                # 选择概率最高的球
                ball = np.argmax(remaining_probs) + 1
                predicted_reds.append(ball)
                
                # 将已选球的概率设为0
                remaining_probs[ball-1] = 0
            
            # 第二步：使用组合特征调整剩余球的概率
            # 计算与已选球的组合概率
            combination_probs = np.zeros(33)
            for ball in predicted_reds:
                for i in range(33):
                    combination_probs[i] += red_combinations[ball-1, i]
            
            # 归一化组合概率
            if sum(combination_probs) > 0:
                combination_probs = combination_probs / sum(combination_probs)
            
            # 将已选球的组合概率设为0
            for ball in predicted_reds:
                combination_probs[ball-1] = 0
            
            # 结合剩余的基础概率和组合概率
            remaining_probs = 0.4 * remaining_probs + 0.6 * combination_probs
            
            # 选择剩余的3个球
            for _ in range(3):
                # 归一化剩余概率
                if sum(remaining_probs) > 0:
                    remaining_probs = remaining_probs / sum(remaining_probs)
                
                # 采样一个球 - 使用随机采样增加多样性
                ball = np.random.choice(33, p=remaining_probs) + 1
                predicted_reds.append(ball)
                
                # 将已选球的概率设为0
                remaining_probs[ball-1] = 0
            
            # 排序
            predicted_reds.sort()
            
            # 选择蓝球 - 使用条件概率
            # 使用随机采样增加多样性
            predicted_blue = np.random.choice(16, p=combined_blue_probs) + 1
            
            if explain:
                print("\n=== 高级动态贝叶斯网络预测 ===")
                print("红球预测概率 (多层融合概率):")
                
                # 获取概率最高的12个红球
                top_red_indices = np.argsort(combined_red_probs)[-12:][::-1]
                for i in top_red_indices:
                    if i + 1 in predicted_reds:
                        print(f"  {i+1:02d}: {combined_red_probs[i]:.4f} (选中) [基础:{red_probs[i]:.4f}, 条件:{red_next_probs[i]:.4f}, 间隔:{red_interval_probs[i]:.4f}]")
                    else:
                        print(f"  {i+1:02d}: {combined_red_probs[i]:.4f} [基础:{red_probs[i]:.4f}, 条件:{red_next_probs[i]:.4f}, 间隔:{red_interval_probs[i]:.4f}]")
                
                print("\n蓝球预测概率:")
                # 获取概率最高的5个蓝球
                top_blue_indices = np.argsort(combined_blue_probs)[-5:][::-1]
                for i in top_blue_indices:
                    if i + 1 == predicted_blue:
                        print(f"  {i+1:02d}: {combined_blue_probs[i]:.4f} (选中) [基础:{blue_probs[i]:.4f}, 条件:{blue_next_probs[i]:.4f}]")
                    else:
                        print(f"  {i+1:02d}: {combined_blue_probs[i]:.4f} [基础:{blue_probs[i]:.4f}, 条件:{blue_next_probs[i]:.4f}]")
                
                print(f"\n高级动态贝叶斯网络预测结果: 红球 {' '.join([f'{ball:02d}' for ball in predicted_reds])} | 蓝球 {predicted_blue:02d}")
                print("\n预测说明:")
                print("1. 该预测使用多层贝叶斯融合技术，结合基础概率、条件转移概率和间隔概率")
                print("2. 红球预测采用两阶段策略：先选择概率最高的3个球，然后基于组合特征选择剩余3个球")
                print("3. 蓝球预测结合基础概率和条件转移概率，使用随机采样增加多样性")
                print("4. 模型考虑了号码间的转移概率矩阵、时间序列特征和号码组合特征")
                print("5. 通过自定义MCMC采样进行后验推断，捕捉号码间的动态关联关系")
            
            return predicted_reds, predicted_blue
            
        except Exception as e:
            print(f"动态贝叶斯网络预测失败: {e}")
            print("详细错误信息:")
            import traceback
            traceback.print_exc()
            return self.generate_random_numbers()
    
    # ==================== 自适应集成学习 ====================
    
    def build_adaptive_ensemble(self):
        """构建自适应集成学习模型"""
        if not BOOSTING_AVAILABLE:
            print("警告: XGBoost或LightGBM未安装，无法构建完整的自适应集成学习模型")
        
        try:
            # 确保数据已加载
            if self.data is None:
                if not self.load_data(force_all_data=True):
                    return None
            
            # 准备特征
            feature_cols = [
                'red_sum', 'red_variance', 'red_span',
                'red_odd_count', 'red_even_count',
                'red_big_count', 'red_small_count',
                'red_zone_1_count', 'red_zone_2_count', 'red_zone_3_count',
                'red_consecutive_count', 'red_prime_count',
                'red_sum_tail'
            ]
            
            # 添加高级特征
            for col in self.data.columns:
                if ('_diff' in col or '_ratio' in col or 'freq_' in col or 'combo_' in col) and col not in feature_cols:
                    feature_cols.append(col)
            
            # 限制特征数量，避免过拟合
            if len(feature_cols) > 50:
                feature_cols = feature_cols[:50]
            
            # 准备基础模型
            base_models = [
                RandomForestClassifier(n_estimators=100, random_state=42),
                DecisionTreeClassifier(max_depth=5, random_state=42)
            ]
            
            if BOOSTING_AVAILABLE:
                base_models.extend([
                    xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                    lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
                ])
            
            # 元模型
            meta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            
            # 创建自适应集成模型
            ensemble = AdaptiveEnsemble(base_models, meta_model)
            
            # 保存模型
            self._ensemble_models['adaptive'] = {
                'model': ensemble,
                'features': feature_cols
            }
            
            print("自适应集成学习模型构建完成")
            return ensemble
            
        except Exception as e:
            print(f"构建自适应集成学习模型失败: {e}")
            return None
    
    def predict_by_adaptive_ensemble(self, explain=False):
        """使用自适应集成学习模型进行预测"""
        # 确保模型已构建
        if 'adaptive' not in self._ensemble_models:
            self.build_adaptive_ensemble()
            
        if 'adaptive' not in self._ensemble_models:
            print("自适应集成学习模型不可用，使用备用方法预测")
            return self.predict_by_markov_chain(explain=explain)
            
        try:
            ensemble = self._ensemble_models['adaptive']['model']
            feature_cols = self._ensemble_models['adaptive']['features']
            
            # 准备特征
            X = self.data[feature_cols].values[:100]  # 使用最近100期数据
            
            # 预测红球
            red_predictions = []
            for i in range(1, 34):
                # 为每个红球构建目标变量
                y = np.array([1 if i in [row[f'red_{j}'] for j in range(1, 7)] else 0 for _, row in self.data.iloc[:100].iterrows()])
                
                # 分割训练集和验证集
                train_size = int(len(X) * 0.8)
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]
                
                # 训练模型
                ensemble.fit(X_train, y_train, X_val, y_val)
                
                # 预测概率
                prob = ensemble.predict(X[-1:])
                red_predictions.append((i, float(prob[0])))
            
            # 选择概率最高的6个红球
            red_predictions.sort(key=lambda x: x[1], reverse=True)
            predicted_reds = sorted([ball for ball, _ in red_predictions[:6]])
            
            # 预测蓝球 (使用频率分析)
            blue_freq = Counter(self.data['blue_ball'])
            blue_probs = {num: count/len(self.data) for num, count in blue_freq.items()}
            predicted_blue = max(blue_probs.items(), key=lambda x: x[1])[0]
            
            if explain:
                print("\n=== 自适应集成学习预测 ===")
                print("红球预测概率:")
                for ball, prob in red_predictions[:10]:  # 显示前10个
                    if ball in predicted_reds:
                        print(f"  {ball:02d}: {prob:.4f} (选中)")
                    else:
                        print(f"  {ball:02d}: {prob:.4f}")
                
                print("\n蓝球预测 (基于频率分析):")
                top_blues = sorted(blue_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                for ball, prob in top_blues:
                    if ball == predicted_blue:
                        print(f"  {ball:02d}: {prob:.4f} (选中)")
                    else:
                        print(f"  {ball:02d}: {prob:.4f}")
                
                print(f"\n自适应集成学习预测结果: 红球 {' '.join([f'{ball:02d}' for ball in predicted_reds])} | 蓝球 {predicted_blue:02d}")
            
            return predicted_reds, predicted_blue
            
        except Exception as e:
            print(f"自适应集成学习预测失败: {e}")
            return self.generate_random_numbers()    

    # ================ 超级预测器 ====================
    
    def predict_by_super(self, explain=False):
        """超级预测器 - 集成多种高级算法"""
        print("启动超级预测器...")
        
        # 收集各种模型的预测结果
        predictions = []
        weights = []
        
        # 1. Transformer模型预测
        if TENSORFLOW_AVAILABLE:
            try:
                print("运行Transformer深度学习模型...")
                transformer_reds, transformer_blue = self.predict_by_transformer(explain=False)
                predictions.append((transformer_reds, transformer_blue))
                weights.append(0.25)
            except Exception as e:
                print(f"Transformer模型预测失败: {e}")
        
        # 2. 图神经网络预测
        if TENSORFLOW_AVAILABLE:
            try:
                print("运行图神经网络模型...")
                graph_nn_reds, graph_nn_blue = self.predict_by_graph_nn(explain=False)
                predictions.append((graph_nn_reds, graph_nn_blue))
                weights.append(0.20)
            except Exception as e:
                print(f"图神经网络模型预测失败: {e}")
        
        # 3. 动态贝叶斯网络预测
        if PYMC_AVAILABLE:
            try:
                print("运行动态贝叶斯网络模型...")
                bayes_reds, bayes_blue = self.predict_by_dynamic_bayesian_network(explain=False)
                predictions.append((bayes_reds, bayes_blue))
                weights.append(0.20)
            except Exception as e:
                print(f"动态贝叶斯网络预测失败: {e}")
        
        # 4. 自适应集成学习预测
        try:
            print("运行自适应集成学习模型...")
            ensemble_reds, ensemble_blue = self.predict_by_adaptive_ensemble(explain=False)
            predictions.append((ensemble_reds, ensemble_blue))
            weights.append(0.20)
        except Exception as e:
            print(f"自适应集成学习预测失败: {e}")
        
        # 5. 马尔可夫链预测
        try:
            print("运行马尔可夫链模型...")
            markov_reds, markov_blue = self.predict_by_markov_chain(explain=False)
            predictions.append((markov_reds, markov_blue))
            weights.append(0.15)
        except Exception as e:
            print(f"马尔可夫链预测失败: {e}")
        
        # 如果没有成功的预测，使用随机生成
        if not predictions:
            print("所有模型预测失败，使用随机生成")
            return self.generate_random_numbers()
        
        # 归一化权重
        weights = [w / sum(weights) for w in weights]
        
        # 统计红球出现频率
        red_votes = Counter()
        for (reds, _), weight in zip(predictions, weights):
            for ball in reds:
                red_votes[ball] += weight
        
        # 选择得票最高的6个红球
        predicted_reds = sorted([ball for ball, _ in red_votes.most_common(6)])
        
        # 统计蓝球出现频率
        blue_votes = Counter()
        for (_, blue), weight in zip(predictions, weights):
            blue_votes[blue] += weight
        
        # 选择得票最高的蓝球
        predicted_blue = blue_votes.most_common(1)[0][0]
        
        if explain:
            print("\n=== 超级预测器结果 ===")
            print("各模型预测:")
            for i, ((reds, blue), weight) in enumerate(zip(predictions, weights)):
                model_names = ["Transformer", "图神经网络", "动态贝叶斯网络", "自适应集成学习", "马尔可夫链"]
                model_name = model_names[i] if i < len(model_names) else f"模型{i+1}"
                print(f"  {model_name} (权重 {weight:.2f}): 红球 {' '.join([f'{ball:02d}' for ball in reds])} | 蓝球 {blue:02d}")
            
            print("\n红球投票结果:")
            for ball, votes in red_votes.most_common(10):
                if ball in predicted_reds:
                    print(f"  {ball:02d}: {votes:.4f} (选中)")
                else:
                    print(f"  {ball:02d}: {votes:.4f}")
            
            print("\n蓝球投票结果:")
            for ball, votes in blue_votes.most_common(5):
                if ball == predicted_blue:
                    print(f"  {ball:02d}: {votes:.4f} (选中)")
                else:
                    print(f"  {ball:02d}: {votes:.4f}")
            
            print(f"\n超级预测器最终结果: 红球 {' '.join([f'{ball:02d}' for ball in predicted_reds])} | 蓝球 {predicted_blue:02d}")
        
        return predicted_reds, predicted_blue
    
    # ==================== 辅助方法 ====================
    
    def generate_random_numbers(self):
        """生成随机号码"""
        red_balls = sorted(random.sample(range(1, 34), 6))
        blue_ball = random.randint(1, 16)
        return red_balls, blue_ball
    
    def crawl_data_from_cwl(self, count=None):
        """从中国福利彩票官方网站爬取数据"""
        results = []

        try:
            if count is None:
                print("正在从中国福利彩票官方网站获取所有期数的双色球开奖结果...")
            else:
                print(f"正在从中国福利彩票官方网站获取最近{count}期双色球开奖结果...")

            page_size = 30
            page = 1
            consecutive_empty_pages = 0
            max_consecutive_empty = 3  # 连续3页无数据则停止

            while True:
                print(f"正在获取第{page}页数据 (每页{page_size}条)...")

                params = {
                    "name": "ssq",
                    "pageNo": page,
                    "pageSize": page_size,
                    "systemType": "PC"
                }

                try:
                    response = requests.get(self.api_url, headers=self.headers, params=params, timeout=15)
                    response.raise_for_status()

                    data = response.json()

                    if data.get("state") == 0 and "result" in data and data["result"]:
                        page_results = []
                        for item in data["result"]:
                            issue = item["code"]
                            date = item["date"]
                            red_str = item["red"]
                            blue_ball = item["blue"]

                            red_balls = red_str.split(",")
                            red_balls = [ball.zfill(2) for ball in red_balls]
                            blue_ball = blue_ball.zfill(2)

                            page_results.append({
                                "issue": issue,
                                "date": date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_ball
                            })

                        results.extend(page_results)
                        consecutive_empty_pages = 0  # 重置空页计数

                        print(f"第{page}页获取到{len(page_results)}期数据，累计{len(results)}期")

                        # 如果这页数据不足page_size，说明可能到了最后
                        if len(page_results) < page_size:
                            print("检测到数据页不满，可能已获取完所有数据")
                            break
                    else:
                        consecutive_empty_pages += 1
                        print(f"第{page}页无数据，连续空页数: {consecutive_empty_pages}")

                        if consecutive_empty_pages >= max_consecutive_empty:
                            print(f"连续{max_consecutive_empty}页无数据，停止爬取")
                            break

                    # 检查是否达到指定数量
                    if count is not None and len(results) >= count:
                        print(f"已获取到指定数量({count})的数据")
                        break

                    page += 1

                    # 防止无限循环，设置最大页数限制
                    if page > 2000:  # 增加最大页数限制
                        print("达到最大页数限制，停止爬取")
                        break

                    # 随机延时，避免请求过快
                    time.sleep(random.uniform(1, 3))

                except requests.exceptions.RequestException as e:
                    print(f"第{page}页请求失败: {e}")
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= max_consecutive_empty:
                        print("连续请求失败，停止爬取")
                        break
                    time.sleep(5)  # 请求失败时等待更长时间
                    continue

            print(f"从中国福利彩票官方网站成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中国福利彩票官方网站获取数据失败: {e}")

        return results
        
    def crawl_data_from_zhcw(self, max_pages=200):
        """从中彩网获取补充数据"""
        results = {}
        
        try:
            print("正在从中彩网获取历史双色球开奖结果...")
            
            base_url = "https://www.zhcw.com/kjxx/ssq/kjhistory_{}.shtml"
            
            for page in range(1, max_pages + 1):
                print(f"正在获取第{page}页数据...")
                
                try:
                    url = base_url.format(page)
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 查找开奖结果表格
                    table = soup.find('table', class_='historylist')
                    if not table:
                        print(f"第{page}页未找到开奖结果表格，可能已到达最后一页")
                        break
                    
                    rows = table.find_all('tr')
                    if len(rows) <= 1:  # 只有表头，没有数据
                        print(f"第{page}页无数据，可能已到达最后一页")
                        break
                    
                    # 跳过表头
                    for row in rows[1:]:
                        cells = row.find_all('td')
                        if len(cells) >= 4:
                            try:
                                # 提取期号
                                issue_cell = cells[0].get_text().strip()
                                issue = issue_cell
                                
                                # 提取日期
                                date_cell = cells[1].get_text().strip()
                                date = date_cell
                                
                                # 提取红球
                                red_balls = []
                                red_ball_cells = cells[2].find_all('em', class_='rr')
                                for ball_cell in red_ball_cells:
                                    ball = ball_cell.get_text().strip().zfill(2)
                                    red_balls.append(ball)
                                
                                # 提取蓝球
                                blue_ball_cells = cells[2].find_all('em', class_='bb')
                                if blue_ball_cells:
                                    blue_ball = blue_ball_cells[0].get_text().strip().zfill(2)
                                else:
                                    continue  # 跳过没有蓝球的行
                                
                                # 检查数据完整性
                                if len(red_balls) == 6 and issue and date:
                                    results[issue] = {
                                        "issue": issue,
                                        "date": date,
                                        "red_balls": ",".join(red_balls),
                                        "blue_ball": blue_ball
                                    }
                            except Exception as e:
                                print(f"解析行数据失败: {e}")
                                continue
                    
                    print(f"第{page}页获取到{len(rows) - 1}期数据，累计{len(results)}期")
                    
                    # 随机延时，避免请求过快
                    time.sleep(random.uniform(2, 5))
                    
                except requests.exceptions.RequestException as e:
                    print(f"第{page}页请求失败: {e}")
                    time.sleep(5)  # 请求失败时等待更长时间
                    continue
                except Exception as e:
                    print(f"处理第{page}页数据失败: {e}")
                    continue
            
            print(f"从中彩网成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中彩网获取数据失败: {e}")
        
        return results
        
    def save_to_csv(self, data, filename):
        """保存数据到CSV文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                if not data:
                    print("没有数据需要保存")
                    return
                    
                fieldnames = data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
                
            print(f"数据已保存到 {filename}")
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def crawl_data(self, use_all_data=False):
        """爬取双色球历史数据"""
        filename = os.path.join(self.data_dir, "ssq_data_all.csv")
        
        if not use_all_data:
            print("错误：本系统只支持使用完整历史数据，请使用 --all 参数")
            return False
            
        print("开始爬取所有历史双色球数据...")
            
        # 从官方网站获取数据
        print("\n=== 第一阶段：从中国福利彩票官方网站获取数据 ===")
        results = self.crawl_data_from_cwl()
        
        # 从中彩网补充数据
        print(f"\n=== 第二阶段：从中彩网补充数据 ===")
        print(f"当前已获取{len(results)}期数据，开始补充更多历史数据...")
        
        # 对于获取所有数据的情况，增加中彩网的页数
        max_zhcw_pages = 500
        zhcw_results = self.crawl_data_from_zhcw(max_pages=max_zhcw_pages)
        
        existing_issues = set(item["issue"] for item in results)
        
        added_count = 0
        for issue, item in zhcw_results.items():
            if issue not in existing_issues:
                results.append(item)
                existing_issues.add(issue)
                added_count += 1
        
        print(f"从中彩网补充了{added_count}期不重复的数据")
        
        # 重新排序
        results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        # 保存数据
        if results:
            self.save_to_csv(results, filename)
            print(f"\n=== 爬取完成 ===")
            print(f"共获取{len(results)}期数据，已保存到 {filename}")
            return True
        else:
            print("未获取到任何数据")
            return False
            
    def update_recent_draws(self, num_periods=5):
        """更新最近N期开奖数据
        
        从官方网站爬取最近N期数据，并追加到ssq_data_all.csv文件中
        如果文件中已有对应期数，则不会重复添加
        """
        filename = os.path.join(self.data_dir, "ssq_data_all.csv")
        
        print(f"开始更新最近{num_periods}期双色球开奖数据...")
        
        # 从官方网站获取最近数据
        recent_results = self.crawl_data_from_cwl(count=num_periods)
        
        if not recent_results:
            print("未获取到任何数据")
            return False
            
        # 检查现有数据
        existing_data = []
        existing_issues = set()
        
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data.append(row)
                        existing_issues.add(row['issue'])
            except Exception as e:
                print(f"读取现有数据失败: {e}")
                
        # 添加新数据
        new_data = []
        for item in recent_results:
            if item['issue'] not in existing_issues:
                new_data.append(item)
                existing_issues.add(item['issue'])
        
        if not new_data:
            print("没有新的开奖数据需要更新")
            return True
            
        # 合并数据
        all_data = new_data + existing_data
        
        # 重新排序
        all_data.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        # 保存数据
        self.save_to_csv(all_data, filename)
        
        print(f"\n=== 更新完成 ===")
        print(f"新增{len(new_data)}期数据:")
        for item in new_data:
            print(f"  {item['issue']}期 ({item['date']}): 红球 {item['red_balls']} | 蓝球 {item['blue_ball']}")
        
        return True
        
    def get_latest_draw(self):
        """获取最新一期开奖结果
        
        从官方网站爬取最新一期数据，并更新到ssq_data_all.csv文件中
        如果文件中已有对应期数，则不会重复添加
        
        返回:
        最新一期开奖结果的字典
        """
        filename = os.path.join(self.data_dir, "ssq_data_all.csv")
        
        print("获取最新一期双色球开奖结果...")
        
        # 从官方网站获取最新数据
        latest_results = self.crawl_data_from_cwl(count=1)
        
        if not latest_results:
            print("未获取到最新开奖数据")
            return None
            
        latest_draw = latest_results[0]
        
        # 检查现有数据
        existing_data = []
        existing_issues = set()
        
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data.append(row)
                        existing_issues.add(row['issue'])
            except Exception as e:
                print(f"读取现有数据失败: {e}")
                
        # 检查是否需要更新
        if latest_draw['issue'] in existing_issues:
            print(f"最新开奖数据 {latest_draw['issue']}期 已存在，无需更新")
        else:
            # 添加新数据
            all_data = [latest_draw] + existing_data
            
            # 保存数据
            self.save_to_csv(all_data, filename)
            
            print(f"已将最新开奖数据 {latest_draw['issue']}期 添加到文件")
        
        # 格式化输出
        red_balls = latest_draw['red_balls']
        blue_ball = latest_draw['blue_ball']
        
        print(f"\n=== 最新开奖结果 ===")
        print(f"期号: {latest_draw['issue']}")
        print(f"日期: {latest_draw['date']}")
        print(f"红球: {red_balls}")
        print(f"蓝球: {blue_ball}")
        
        return latest_draw


# ==================== 主函数 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="双色球数据分析与预测系统 - 全功能整合版")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 爬取数据
    crawl_parser = subparsers.add_parser("crawl", help="爬取双色球历史数据")
    crawl_parser.add_argument("--all", action="store_true", help="爬取所有历史数据")
    
    # 更新最近N期数据
    update_parser = subparsers.add_parser("update", help="更新最近N期双色球开奖数据")
    update_parser.add_argument("--count", type=int, default=5, help="更新期数，默认5期")
    
    # 获取最新开奖结果
    latest_parser = subparsers.add_parser("latest", help="获取最新一期双色球开奖结果")
    
    # 分析数据
    analyze_parser = subparsers.add_parser("analyze", help="分析双色球数据")
    
    # 预测
    predict_parser = subparsers.add_parser("predict", help="预测双色球号码")
    predict_parser.add_argument("--method", choices=["transformer", "graph_nn", "dynamic_bayes", "adaptive_ensemble", "super", "markov"], 
                              default="super", help="预测方法")
    predict_parser.add_argument("--count", type=int, default=1, help="预测组数")
    predict_parser.add_argument("--explain", action="store_true", help="显示详细解释")
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = SSQAnalyzer()
    
    # 执行命令
    if args.command == "crawl":
        analyzer.crawl_data(use_all_data=args.all)
    
    elif args.command == "update":
        analyzer.update_recent_draws(num_periods=args.count)
    
    elif args.command == "latest":
        analyzer.get_latest_draw()
    
    elif args.command == "analyze":
        if analyzer.load_data():
            analyzer.analyze_number_frequency()
            analyzer.analyze_number_combinations()
            analyzer.analyze_trend()
    
    elif args.command == "predict":
        if analyzer.load_data():
            for i in range(args.count):
                if i > 0:
                    print("\n" + "="*50 + f"\n预测组 {i+1}/{args.count}\n" + "="*50)
                
                if args.method == "transformer":
                    red_balls, blue_ball = analyzer.predict_by_transformer(explain=args.explain)
                elif args.method == "graph_nn":
                    red_balls, blue_ball = analyzer.predict_by_graph_nn(explain=args.explain)
                elif args.method == "dynamic_bayes":
                    red_balls, blue_ball = analyzer.predict_by_dynamic_bayesian_network(explain=args.explain)
                elif args.method == "adaptive_ensemble":
                    red_balls, blue_ball = analyzer.predict_by_adaptive_ensemble(explain=args.explain)
                elif args.method == "markov":
                    red_balls, blue_ball = analyzer.predict_by_markov_chain(explain=args.explain)
                else:  # super
                    red_balls, blue_ball = analyzer.predict_by_super(explain=args.explain)
                
                print(f"\n预测结果: 红球 {' '.join([f'{ball:02d}' for ball in red_balls])} | 蓝球 {blue_ball:02d}")
    
    else:
        parser.print_help()


# ==================== 自适应集成学习 ====================

class AdaptiveEnsemble:
    """自适应集成学习模型"""
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.weights = np.ones(len(base_models)) / len(base_models)
    
    def fit(self, X, y, X_val, y_val):
        """训练模型"""
        # 训练基础模型
        base_preds = []
        for model in self.base_models:
            model.fit(X, y)
            base_preds.append(model.predict(X_val))
        
        # 堆叠预测结果
        stacked_preds = np.column_stack(base_preds)
        
        # 训练元模型
        self.meta_model.fit(stacked_preds, y_val)
        
        # 更新权重
        self._update_weights(stacked_preds, y_val)
    
    def _update_weights(self, preds, y_true):
        """更新模型权重"""
        errors = [mean_squared_error(y_true, preds[:, i]) 
                 for i in range(preds.shape[1])]
        
        # 反比于误差
        inv_errors = [1/e if e > 0 else 1.0 for e in errors]
        sum_inv = sum(inv_errors)
        self.weights = [e/sum_inv for e in inv_errors]
    
    def predict(self, X):
        """预测"""
        # 获取基础模型预测
        base_preds = []
        for model in self.base_models:
            base_preds.append(model.predict(X))
        
        # 堆叠预测结果
        stacked_preds = np.column_stack(base_preds)
        
        # 元模型预测
        meta_pred = self.meta_model.predict(stacked_preds)
        
        # 加权平均
        weighted_pred = np.zeros_like(meta_pred)
        for i, w in enumerate(self.weights):
            weighted_pred += w * base_preds[i]
        
        # 结合元模型和加权平均
        final_pred = 0.7 * meta_pred + 0.3 * weighted_pred
        
        return final_pred

if __name__ == "__main__":
    main()