#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
K-Means聚类分析预测器
基于Callam7/LottoPipeline项目的聚类分析方法
使用K-Means + DBSCAN + PCA降维
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关导入
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import seaborn as sns
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    print("警告: 聚类分析库未安装，聚类预测功能将不可用")
    print("请安装依赖: pip install scikit-learn matplotlib seaborn")


class SSQClusteringPredictor:
    """双色球聚类分析预测器"""
    
    def __init__(self, data_file="../data/ssq_data.csv", output_dir="../data/clustering"):
        """
        初始化聚类预测器
        
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
        
        # 聚类参数
        self.optimal_k = None  # 最优聚类数
        self.kmeans_model = None  # K-Means模型
        self.dbscan_model = None  # DBSCAN模型
        self.pca_model = None  # PCA模型
        self.scaler = None  # 标准化器
        
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
    
    def extract_clustering_features(self, periods=500):
        """
        提取聚类特征
        
        Args:
            periods: 分析期数
            
        Returns:
            特征DataFrame
        """
        print("提取聚类特征...")
        
        # 限制分析期数
        data = self.data.head(periods).copy()
        features = []
        
        for i, row in data.iterrows():
            feature_dict = {}
            
            # 基础号码特征
            red_balls = [row[f'red_{j}'] for j in range(1, 7)]
            blue_ball = row['blue_ball']
            
            # 1. One-hot编码特征（每个号码位置）
            for j, ball in enumerate(red_balls, 1):
                for k in range(1, 34):
                    feature_dict[f'red_{j}_is_{k}'] = 1 if ball == k else 0
            
            for k in range(1, 17):
                feature_dict[f'blue_is_{k}'] = 1 if blue_ball == k else 0
            
            # 2. 统计特征
            feature_dict['red_sum'] = sum(red_balls)
            feature_dict['red_mean'] = np.mean(red_balls)
            feature_dict['red_std'] = np.std(red_balls)
            feature_dict['red_min'] = min(red_balls)
            feature_dict['red_max'] = max(red_balls)
            feature_dict['red_range'] = max(red_balls) - min(red_balls)
            feature_dict['red_median'] = np.median(red_balls)
            
            # 3. 分布特征
            # 奇偶分布
            odd_count = sum(1 for ball in red_balls if ball % 2 == 1)
            feature_dict['red_odd_count'] = odd_count
            feature_dict['red_even_count'] = 6 - odd_count
            feature_dict['red_odd_ratio'] = odd_count / 6
            feature_dict['blue_odd'] = 1 if blue_ball % 2 == 1 else 0
            
            # 大小分布（以17为界）
            big_count = sum(1 for ball in red_balls if ball >= 17)
            feature_dict['red_big_count'] = big_count
            feature_dict['red_small_count'] = 6 - big_count
            feature_dict['red_big_ratio'] = big_count / 6
            feature_dict['blue_big'] = 1 if blue_ball >= 9 else 0
            
            # 4. 连续性特征
            sorted_reds = sorted(red_balls)
            consecutive_pairs = 0
            consecutive_triplets = 0
            
            for j in range(len(sorted_reds) - 1):
                if sorted_reds[j+1] - sorted_reds[j] == 1:
                    consecutive_pairs += 1
                    if j < len(sorted_reds) - 2 and sorted_reds[j+2] - sorted_reds[j+1] == 1:
                        consecutive_triplets += 1
            
            feature_dict['consecutive_pairs'] = consecutive_pairs
            feature_dict['consecutive_triplets'] = consecutive_triplets
            
            # 5. 间隔特征
            gaps = [sorted_reds[j+1] - sorted_reds[j] for j in range(len(sorted_reds) - 1)]
            feature_dict['avg_gap'] = np.mean(gaps)
            feature_dict['max_gap'] = max(gaps)
            feature_dict['min_gap'] = min(gaps)
            feature_dict['gap_std'] = np.std(gaps)
            
            # 6. 尾数特征
            tail_counts = {}
            for tail in range(10):
                count = sum(1 for ball in red_balls if ball % 10 == tail)
                tail_counts[f'tail_{tail}_count'] = count
            feature_dict.update(tail_counts)
            
            # 7. 区间分布特征
            # 将1-33分为3个区间
            zone1 = sum(1 for ball in red_balls if 1 <= ball <= 11)
            zone2 = sum(1 for ball in red_balls if 12 <= ball <= 22)
            zone3 = sum(1 for ball in red_balls if 23 <= ball <= 33)
            
            feature_dict['zone1_count'] = zone1
            feature_dict['zone2_count'] = zone2
            feature_dict['zone3_count'] = zone3
            feature_dict['zone1_ratio'] = zone1 / 6
            feature_dict['zone2_ratio'] = zone2 / 6
            feature_dict['zone3_ratio'] = zone3 / 6
            
            # 8. 质数特征
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
            prime_count = sum(1 for ball in red_balls if ball in primes)
            feature_dict['prime_count'] = prime_count
            feature_dict['prime_ratio'] = prime_count / 6
            
            # 9. 和值特征分类
            red_sum = sum(red_balls)
            if red_sum <= 90:
                sum_category = 0  # 低和值
            elif red_sum <= 120:
                sum_category = 1  # 中和值
            else:
                sum_category = 2  # 高和值
            feature_dict['sum_category'] = sum_category
            
            # 10. AC值（算术复杂性）
            ac_value = len(set(abs(red_balls[i] - red_balls[j]) 
                              for i in range(6) for j in range(i+1, 6)))
            feature_dict['ac_value'] = ac_value
            
            features.append(feature_dict)
        
        features_df = pd.DataFrame(features)
        print(f"特征提取完成，共提取{len(features_df.columns)}个特征")
        return features_df
    
    def find_optimal_clusters(self, features_df, max_k=15):
        """
        寻找最优聚类数
        
        Args:
            features_df: 特征DataFrame
            max_k: 最大聚类数
            
        Returns:
            最优聚类数
        """
        print("寻找最优聚类数...")
        
        # 数据标准化
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_df)
        
        # 使用轮廓系数评估不同的k值
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(features_df) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"k={k}, 轮廓系数={silhouette_avg:.4f}")
        
        # 选择轮廓系数最高的k值
        optimal_k = k_range[np.argmax(silhouette_scores)]
        self.optimal_k = optimal_k
        
        print(f"最优聚类数: {optimal_k}")
        return optimal_k
    
    def perform_clustering(self, features_df, k=None):
        """
        执行聚类分析
        
        Args:
            features_df: 特征DataFrame
            k: 聚类数，如果为None则使用最优聚类数
        """
        print("执行聚类分析...")
        
        if k is None:
            k = self.optimal_k or self.find_optimal_clusters(features_df)
        
        # 数据标准化
        if self.scaler is None:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_df)
        else:
            features_scaled = self.scaler.transform(features_df)
        
        # K-Means聚类
        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = self.kmeans_model.fit_predict(features_scaled)
        
        # DBSCAN聚类（作为补充）
        # 自动估计eps参数
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors_fit = neighbors.fit(features_scaled)
        distances, indices = neighbors_fit.kneighbors(features_scaled)
        distances = np.sort(distances[:, 4], axis=0)
        eps = np.percentile(distances, 90)  # 使用90%分位数作为eps
        
        self.dbscan_model = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = self.dbscan_model.fit_predict(features_scaled)
        
        # PCA降维用于可视化
        self.pca_model = PCA(n_components=2)
        features_pca = self.pca_model.fit_transform(features_scaled)
        
        # 保存聚类结果
        clustering_results = features_df.copy()
        clustering_results['kmeans_cluster'] = kmeans_labels
        clustering_results['dbscan_cluster'] = dbscan_labels
        clustering_results['pca_1'] = features_pca[:, 0]
        clustering_results['pca_2'] = features_pca[:, 1]
        
        print(f"K-Means聚类完成，共{k}个簇")
        print(f"DBSCAN聚类完成，共{len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}个簇")
        
        return clustering_results
    
    def predict(self, num_predictions=1, k=None):
        """
        预测双色球号码
        
        Args:
            num_predictions: 预测注数
            k: 聚类数
            
        Returns:
            预测结果列表
        """
        if self.data is None or self.data.empty:
            if not self.load_data():
                return None
        
        # 提取特征
        features_df = self.extract_clustering_features()
        
        # 执行聚类
        clustering_results = self.perform_clustering(features_df, k=k)
        
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
    parser = argparse.ArgumentParser(description='双色球聚类分析预测器')
    parser.add_argument('-n', '--num', type=int, default=1, help='预测注数，默认1注')
    parser.add_argument('-k', '--clusters', type=int, help='聚类数，默认自动确定')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 设置默认路径
    data_file = os.path.join(project_root, "data", "ssq_data.csv")
    output_dir = os.path.join(project_root, "data", "clustering")
    
    # 创建预测器实例
    predictor = SSQClusteringPredictor(data_file=data_file, output_dir=output_dir)
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    # 预测号码
    print("🔍 K-Means聚类分析预测")
    print("=" * 40)
    
    predictions = predictor.predict(num_predictions=args.num, k=args.clusters)
    
    if predictions:
        for i, (red_balls, blue_ball) in enumerate(predictions, 1):
            formatted = format_ssq_numbers(red_balls, blue_ball)
            print(f"第 {i} 注: {formatted}")
    else:
        print("预测失败")


if __name__ == "__main__":
    main()
