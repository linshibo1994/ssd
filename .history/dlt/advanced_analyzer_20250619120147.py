#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级分析器模块
提供大乐透数据的高级分析功能
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
import json
import networkx as nx
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

# 设置中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except Exception as e:
    print(f"设置中文显示失败: {e}")

# 忽略警告
warnings.filterwarnings("ignore")


class DLTAdvancedAnalyzer:
    """大乐透高级分析器"""
    
    def __init__(self, data_file, output_dir="./output/advanced"):
        """初始化分析器

        Args:
            data_file: 数据文件路径
            output_dir: 输出目录
        """
        self.data_file = data_file
        self.output_dir = output_dir
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 读取数据
        self.df = pd.read_csv(data_file)
        
        # 解析前区和后区号码
        self._parse_ball_numbers()
    
    def _parse_ball_numbers(self):
        """解析前区和后区号码"""
        # 解析前区号码
        self.front_balls_lists = []
        for _, row in self.df.iterrows():
            front_balls = [int(ball) for ball in row["front_balls"].split(",")]
            self.front_balls_lists.append(front_balls)
        
        # 解析后区号码
        self.back_balls_lists = []
        for _, row in self.df.iterrows():
            back_balls = [int(ball) for ball in row["back_balls"].split(",")]
            self.back_balls_lists.append(back_balls)
    
    def analyze_statistical_features(self, save_result=True):
        """分析统计学特征

        Args:
            save_result: 是否保存结果

        Returns:
            统计特征结果字典
        """
        print("分析统计学特征...")
        
        # 计算前区号码和值、方差、跨度等统计特征
        front_sums = []
        front_means = []
        front_variances = []
        front_spans = []
        front_odds = []
        front_evens = []
        
        for front_list in self.front_balls_lists:
            # 和值
            front_sum = sum(front_list)
            front_sums.append(front_sum)
            
            # 均值
            front_mean = np.mean(front_list)
            front_means.append(front_mean)
            
            # 方差
            front_var = np.var(front_list)
            front_variances.append(front_var)
            
            # 跨度（最大值-最小值）
            front_span = max(front_list) - min(front_list)
            front_spans.append(front_span)
            
            # 奇偶比例
            front_odd = sum(1 for ball in front_list if ball % 2 == 1)
            front_even = sum(1 for ball in front_list if ball % 2 == 0)
            front_odds.append(front_odd)
            front_evens.append(front_even)
        
        # 计算后区号码和值、方差、跨度等统计特征
        back_sums = []
        back_means = []
        back_variances = []
        back_spans = []
        back_odds = []
        back_evens = []
        
        for back_list in self.back_balls_lists:
            # 和值
            back_sum = sum(back_list)
            back_sums.append(back_sum)
            
            # 均值
            back_mean = np.mean(back_list)
            back_means.append(back_mean)
            
            # 方差
            back_var = np.var(back_list)
            back_variances.append(back_var)
            
            # 跨度（最大值-最小值）
            back_span = max(back_list) - min(back_list)
            back_spans.append(back_span)
            
            # 奇偶比例
            back_odd = sum(1 for ball in back_list if ball % 2 == 1)
            back_even = sum(1 for ball in back_list if ball % 2 == 0)
            back_odds.append(back_odd)
            back_evens.append(back_even)
        
        # 绘制前区和值分布图
        plt.figure(figsize=(12, 6))
        plt.hist(front_sums, bins=30, alpha=0.7, color="blue")
        plt.title("大乐透前区和值分布")
        plt.xlabel("和值")
        plt.ylabel("频数")
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_sum_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 绘制前区和值时间序列图
        plt.figure(figsize=(15, 6))
        plt.plot(front_sums, marker="o", markersize=3, linestyle="-", alpha=0.7)
        plt.title("大乐透前区和值时间序列")
        plt.xlabel("期数")
        plt.ylabel("和值")
        plt.grid(True, alpha=0.3)
        
        # 添加均值线
        plt.axhline(y=np.mean(front_sums), color="r", linestyle="--", label=f"均值: {np.mean(front_sums):.2f}")
        plt.legend()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_sum_time_series.png"), dpi=300, bbox_inches="tight")
        
        # 绘制前区方差分布图
        plt.figure(figsize=(12, 6))
        plt.hist(front_variances, bins=30, alpha=0.7, color="green")
        plt.title("大乐透前区方差分布")
        plt.xlabel("方差")
        plt.ylabel("频数")
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_variance_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 绘制前区跨度分布图
        plt.figure(figsize=(12, 6))
        plt.hist(front_spans, bins=30, alpha=0.7, color="purple")
        plt.title("大乐透前区跨度分布")
        plt.xlabel("跨度")
        plt.ylabel("频数")
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_span_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 绘制前区奇偶比例分布图
        plt.figure(figsize=(10, 6))
        odd_even_counts = Counter([(odd, even) for odd, even in zip(front_odds, front_evens)])
        labels = [f"{odd}奇{even}偶" for odd, even in odd_even_counts.keys()]
        values = list(odd_even_counts.values())
        
        # 按奇数数量排序
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        plt.bar(sorted_labels, sorted_values, color="orange")
        plt.title("大乐透前区奇偶比例分布")
        plt.xlabel("奇偶比例")
        plt.ylabel("频数")
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_odd_even_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区和值分布图
        plt.figure(figsize=(12, 6))
        plt.hist(back_sums, bins=20, alpha=0.7, color="blue")
        plt.title("大乐透后区和值分布")
        plt.xlabel("和值")
        plt.ylabel("频数")
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_sum_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区和值时间序列图
        plt.figure(figsize=(15, 6))
        plt.plot(back_sums, marker="o", markersize=3, linestyle="-", alpha=0.7)
        plt.title("大乐透后区和值时间序列")
        plt.xlabel("期数")
        plt.ylabel("和值")
        plt.grid(True, alpha=0.3)
        
        # 添加均值线
        plt.axhline(y=np.mean(back_sums), color="r", linestyle="--", label=f"均值: {np.mean(back_sums):.2f}")
        plt.legend()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_sum_time_series.png"), dpi=300, bbox_inches="tight")
        
        # 保存统计特征数据到CSV
        if save_result:
            # 创建前区统计特征数据框
            front_stats_df = pd.DataFrame({
                "issue": self.df["issue"].values,
                "date": self.df["date"].values,
                "front_sum": front_sums,
                "front_mean": front_means,
                "front_variance": front_variances,
                "front_span": front_spans,
                "front_odd": front_odds,
                "front_even": front_evens
            })
            front_stats_df.to_csv(os.path.join(self.output_dir, "front_statistical_features.csv"), index=False)
            
            # 创建后区统计特征数据框
            back_stats_df = pd.DataFrame({
                "issue": self.df["issue"].values,
                "date": self.df["date"].values,
                "back_sum": back_sums,
                "back_mean": back_means,
                "back_variance": back_variances,
                "back_span": back_spans,
                "back_odd": back_odds,
                "back_even": back_evens
            })
            back_stats_df.to_csv(os.path.join(self.output_dir, "back_statistical_features.csv"), index=False)
        
        # 返回统计特征结果
        stats_results = {
            "front": {
                "sum": {
                    "mean": np.mean(front_sums),
                    "std": np.std(front_sums),
                    "min": np.min(front_sums),
                    "max": np.max(front_sums),
                    "distribution": Counter(front_sums)
                },
                "variance": {
                    "mean": np.mean(front_variances),
                    "std": np.std(front_variances),
                    "min": np.min(front_variances),
                    "max": np.max(front_variances)
                },
                "span": {
                    "mean": np.mean(front_spans),
                    "std": np.std(front_spans),
                    "min": np.min(front_spans),
                    "max": np.max(front_spans),
                    "distribution": Counter(front_spans)
                },
                "odd_even": odd_even_counts
            },
            "back": {
                "sum": {
                    "mean": np.mean(back_sums),
                    "std": np.std(back_sums),
                    "min": np.min(back_sums),
                    "max": np.max(back_sums),
                    "distribution": Counter(back_sums)
                },
                "variance": {
                    "mean": np.mean(back_variances),
                    "std": np.std(back_variances),
                    "min": np.min(back_variances),
                    "max": np.max(back_variances)
                },
                "span": {
                    "mean": np.mean(back_spans),
                    "std": np.std(back_spans),
                    "min": np.min(back_spans),
                    "max": np.max(back_spans),
                    "distribution": Counter(back_spans)
                }
            }
        }
        
        return stats_results
    
    def analyze_probability_distribution(self, save_result=True):
        """分析概率分布

        Args:
            save_result: 是否保存结果

        Returns:
            概率分布结果字典
        """
        print("分析概率分布...")
        
        # 统计前区号码频率
        front_balls_flat = [ball for sublist in self.front_balls_lists for ball in sublist]
        front_counter = Counter(front_balls_flat)
        
        # 确保所有可能的前区号码都在字典中
        for i in range(1, 36):
            if i not in front_counter:
                front_counter[i] = 0
        
        # 计算前区号码概率分布
        total_front_draws = len(self.front_balls_lists) * 5  # 总前区号码数量
        front_prob = {k: v / total_front_draws for k, v in front_counter.items()}
        
        # 统计后区号码频率
        back_balls_flat = [ball for sublist in self.back_balls_lists for ball in sublist]
        back_counter = Counter(back_balls_flat)
        
        # 确保所有可能的后区号码都在字典中
        for i in range(1, 13):
            if i not in back_counter:
                back_counter[i] = 0
        
        # 计算后区号码概率分布
        total_back_draws = len(self.back_balls_lists) * 2  # 总后区号码数量
        back_prob = {k: v / total_back_draws for k, v in back_counter.items()}
        
        # 绘制前区号码概率分布图
        plt.figure(figsize=(15, 6))
        plt.bar(front_prob.keys(), front_prob.values())
        plt.title("大乐透前区号码概率分布")
        plt.xlabel("号码")
        plt.ylabel("概率")
        plt.xticks(range(1, 36))
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 添加理论均匀分布线
        plt.axhline(y=1/35, color="r", linestyle="--", label=f"理论概率: {1/35:.4f}")
        plt.legend()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_probability_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区号码概率分布图
        plt.figure(figsize=(12, 6))
        plt.bar(back_prob.keys(), back_prob.values())
        plt.title("大乐透后区号码概率分布")
        plt.xlabel("号码")
        plt.ylabel("概率")
        plt.xticks(range(1, 13))
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 添加理论均匀分布线
        plt.axhline(y=1/12, color="r", linestyle="--", label=f"理论概率: {1/12:.4f}")
        plt.legend()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_probability_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 进行卡方检验，检验号码分布是否符合均匀分布
        front_observed = np.array(list(front_counter.values()))
        front_expected = np.ones(35) * total_front_draws / 35
        front_chi2, front_p = stats.chisquare(front_observed, front_expected)
        
        back_observed = np.array(list(back_counter.values()))
        back_expected = np.ones(12) * total_back_draws / 12
        back_chi2, back_p = stats.chisquare(back_observed, back_expected)
        
        # 保存概率分布数据到CSV
        if save_result:
            # 创建前区概率分布数据框
            front_prob_df = pd.DataFrame({
                "number": list(range(1, 36)),
                "frequency": [front_counter[i] for i in range(1, 36)],
                "probability": [front_prob[i] for i in range(1, 36)],
                "expected_probability": [1/35] * 35,
                "deviation": [front_prob[i] - 1/35 for i in range(1, 36)]
            })
            front_prob_df.to_csv(os.path.join(self.output_dir, "front_probability_distribution.csv"), index=False)
            
            # 创建后区概率分布数据框
            back_prob_df = pd.DataFrame({
                "number": list(range(1, 13)),
                "frequency": [back_counter[i] for i in range(1, 13)],
                "probability": [back_prob[i] for i in range(1, 13)],
                "expected_probability": [1/12] * 12,
                "deviation": [back_prob[i] - 1/12 for i in range(1, 13)]
            })
            back_prob_df.to_csv(os.path.join(self.output_dir, "back_probability_distribution.csv"), index=False)
        
        # 返回概率分布结果
        prob_results = {
            "front": {
                "probability": front_prob,
                "chi2_test": {
                    "chi2": front_chi2,
                    "p_value": front_p,
                    "is_uniform": front_p > 0.05
                }
            },
            "back": {
                "probability": back_prob,
                "chi2_test": {
                    "chi2": back_chi2,
                    "p_value": back_p,
                    "is_uniform": back_p > 0.05
                }
            }
        }
        
        return prob_results
    
    def analyze_frequency_patterns(self, save_result=True):
        """分析频率模式

        Args:
            save_result: 是否保存结果

        Returns:
            频率模式结果字典
        """
        print("分析频率模式...")
        
        # 分析前区号码的频率模式
        front_patterns = {}
        
        # 分析前区号码的大小比例
        front_big_small_ratio = []
        for front_list in self.front_balls_lists:
            big_count = sum(1 for ball in front_list if ball > 18)
            small_count = sum(1 for ball in front_list if ball <= 18)
            front_big_small_ratio.append((big_count, small_count))
        
        front_big_small_counter = Counter(front_big_small_ratio)
        
        # 分析前区号码的奇偶比例
        front_odd_even_ratio = []
        for front_list in self.front_balls_lists:
            odd_count = sum(1 for ball in front_list if ball % 2 == 1)
            even_count = sum(1 for ball in front_list if ball % 2 == 0)
            front_odd_even_ratio.append((odd_count, even_count))
        
        front_odd_even_counter = Counter(front_odd_even_ratio)
        
        # 分析前区号码的区间分布
        front_zone_distribution = []
        for front_list in self.front_balls_lists:
            zone1 = sum(1 for ball in front_list if 1 <= ball <= 7)
            zone2 = sum(1 for ball in front_list if 8 <= ball <= 14)
            zone3 = sum(1 for ball in front_list if 15 <= ball <= 21)
            zone4 = sum(1 for ball in front_list if 22 <= ball <= 28)
            zone5 = sum(1 for ball in front_list if 29 <= ball <= 35)
            front_zone_distribution.append((zone1, zone2, zone3, zone4, zone5))
        
        front_zone_counter = Counter(front_zone_distribution)
        
        # 分析后区号码的频率模式
        back_patterns = {}
        
        # 分析后区号码的大小比例
        back_big_small_ratio = []
        for back_list in self.back_balls_lists:
            big_count = sum(1 for ball in back_list if ball > 6)
            small_count = sum(1 for ball in back_list if ball <= 6)
            back_big_small_ratio.append((big_count, small_count))
        
        back_big_small_counter = Counter(back_big_small_ratio)
        
        # 分析后区号码的奇偶比例
        back_odd_even_ratio = []
        for back_list in self.back_balls_lists:
            odd_count = sum(1 for ball in back_list if ball % 2 == 1)
            even_count = sum(1 for ball in back_list if ball % 2 == 0)
            back_odd_even_ratio.append((odd_count, even_count))
        
        back_odd_even_counter = Counter(back_odd_even_ratio)
        
        # 绘制前区大小比例分布图
        plt.figure(figsize=(10, 6))
        labels = [f"{big}大{small}小" for big, small in front_big_small_counter.keys()]
        values = list(front_big_small_counter.values())
        
        # 按大号数量排序
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        plt.bar(sorted_labels, sorted_values, color="blue")
        plt.title("大乐透前区大小比例分布")
        plt.xlabel("大小比例")
        plt.ylabel("频数")
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_big_small_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 绘制前区奇偶比例分布图
        plt.figure(figsize=(10, 6))
        labels = [f"{odd}奇{even}偶" for odd, even in front_odd_even_counter.keys()]
        values = list(front_odd_even_counter.values())
        
        # 按奇数数量排序
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        plt.bar(sorted_labels, sorted_values, color="green")
        plt.title("大乐透前区奇偶比例分布")
        plt.xlabel("奇偶比例")
        plt.ylabel("频数")
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_odd_even_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区大小比例分布图
        plt.figure(figsize=(10, 6))
        labels = [f"{big}大{small}小" for big, small in back_big_small_counter.keys()]
        values = list(back_big_small_counter.values())
        
        # 按大号数量排序
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        plt.bar(sorted_labels, sorted_values, color="blue")
        plt.title("大乐透后区大小比例分布")
        plt.xlabel("大小比例")
        plt.ylabel("频数")
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_big_small_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区奇偶比例分布图
        plt.figure(figsize=(10, 6))
        labels = [f"{odd}奇{even}偶" for odd, even in back_odd_even_counter.keys()]
        values = list(back_odd_even_counter.values())
        
        # 按奇数数量排序
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        plt.bar(sorted_labels, sorted_values, color="green")
        plt.title("大乐透后区奇偶比例分布")
        plt.xlabel("奇偶比例")
        plt.ylabel("频数")
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_odd_even_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 保存频率模式数据到CSV
        if save_result:
            # 创建前区大小比例数据框
            front_big_small_df = pd.DataFrame({
                "big_count": [k[0] for k in front_big_small_counter.keys()],
                "small_count": [k[1] for k in front_big_small_counter.keys()],
                "frequency": list(front_big_small_counter.values()),
                "percentage": [v / len(self.front_balls_lists) * 100 for v in front_big_small_counter.values()]
            })
            front_big_small_df.to_csv(os.path.join(self.output_dir, "front_big_small_ratio.csv"), index=False)
            
            # 创建前区奇偶比例数据框
            front_odd_even_df = pd.DataFrame({
                "odd_count": [k[0] for k in front_odd_even_counter.keys()],
                "even_count": [k[1] for k in front_odd_even_counter.keys()],
                "frequency": list(front_odd_even_counter.values()),
                "percentage": [v / len(self.front_balls_lists) * 100 for v in front_odd_even_counter.values()]
            })
            front_odd_even_df.to_csv(os.path.join(self.output_dir, "front_odd_even_ratio.csv"), index=False)
            
            # 创建后区大小比例数据框
            back_big_small_df = pd.DataFrame({
                "big_count": [k[0] for k in back_big_small_counter.keys()],
                "small_count": [k[1] for k in back_big_small_counter.keys()],
                "frequency": list(back_big_small_counter.values()),
                "percentage": [v / len(self.back_balls_lists) * 100 for v in back_big_small_counter.values()]
            })
            back_big_small_df.to_csv(os.path.join(self.output_dir, "back_big_small_ratio.csv"), index=False)
            
            # 创建后区奇偶比例数据框
            back_odd_even_df = pd.DataFrame({
                "odd_count": [k[0] for k in back_odd_even_counter.keys()],
                "even_count": [k[1] for k in back_odd_even_counter.keys()],
                "frequency": list(back_odd_even_counter.values()),
                "percentage": [v / len(self.back_balls_lists) * 100 for v in back_odd_even_counter.values()]
            })
            back_odd_even_df.to_csv(os.path.join(self.output_dir, "back_odd_even_ratio.csv"), index=False)
        
        # 返回频率模式结果
        pattern_results = {
            "front": {
                "big_small_ratio": {str(k): v for k, v in front_big_small_counter.items()},
                "odd_even_ratio": {str(k): v for k, v in front_odd_even_counter.items()},
                "zone_distribution": {str(k): v for k, v in front_zone_counter.items()}
            },
            "back": {
                "big_small_ratio": {str(k): v for k, v in back_big_small_counter.items()},
                "odd_even_ratio": {str(k): v for k, v in back_odd_even_counter.items()}
            }
        }
        
        return pattern_results
    
    def analyze_markov_chain(self, save_result=True):
        """分析马尔可夫链

        Args:
            save_result: 是否保存结果

        Returns:
            马尔可夫链分析结果字典
        """
        print("分析马尔可夫链...")
        
        # 分析前区号码的马尔可夫链
        front_transitions = {}
        for i in range(1, 36):
            front_transitions[i] = {j: 0 for j in range(1, 36)}
        
        # 计算前区号码的转移概率
        for i in range(len(self.front_balls_lists) - 1):
            current_draw = self.front_balls_lists[i]
            next_draw = self.front_balls_lists[i + 1]
            
            for current_ball in current_draw:
                for next_ball in next_draw:
                    front_transitions[current_ball][next_ball] += 1
        
        # 归一化前区转移概率
        front_transition_probs = {}
        for i in range(1, 36):
            total = sum(front_transitions[i].values())
            if total > 0:
                front_transition_probs[i] = {j: count / total for j, count in front_transitions[i].items()}
            else:
                front_transition_probs[i] = {j: 0 for j in range(1, 36)}
        
        # 分析后区号码的马尔可夫链
        back_transitions = {}
        for i in range(1, 13):
            back_transitions[i] = {j: 0 for j in range(1, 13)}
        
        # 计算后区号码的转移概率
        for i in range(len(self.back_balls_lists) - 1):
            current_draw = self.back_balls_lists[i]
            next_draw = self.back_balls_lists[i + 1]
            
            for current_ball in current_draw:
                for next_ball in next_draw:
                    back_transitions[current_ball][next_ball] += 1
        
        # 归一化后区转移概率
        back_transition_probs = {}
        for i in range(1, 13):
            total = sum(back_transitions[i].values())
            if total > 0:
                back_transition_probs[i] = {j: count / total for j, count in back_transitions[i].items()}
            else:
                back_transition_probs[i] = {j: 0 for j in range(1, 13)}
        
        # 保存马尔可夫链分析结果
        if save_result:
            # 保存为JSON文件
            markov_results = {
                "front_transition_probs": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in front_transition_probs.items()},
                "back_transition_probs": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in back_transition_probs.items()}
            }
            
            with open(os.path.join(self.output_dir, "markov_chain_analysis.json"), "w", encoding="utf-8") as f:
                json.dump(markov_results, f, ensure_ascii=False, indent=4)
        
        # 可视化后区号码的转移概率网络图
        plt.figure(figsize=(12, 10))
        G = nx.DiGraph()
        
        # 添加节点
        for i in range(1, 13):
            G.add_node(i)
        
        # 添加边
        for i in range(1, 13):
            for j in range(1, 13):
                if back_transition_probs[i][j] > 0.1:  # 只显示概率大于0.1的边
                    G.add_edge(i, j, weight=back_transition_probs[i][j])
        
        # 设置节点位置
        pos = nx.spring_layout(G, seed=42)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
        
        # 绘制边
        edges = G.edges(data=True)
        weights = [d["weight"] * 3 for _, _, d in edges]  # 边的宽度与权重成正比
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color="gray", alpha=0.6, arrows=True, arrowsize=15)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
        
        # 绘制边标签（转移概率）
        edge_labels = {(i, j): f"{back_transition_probs[i][j]:.2f}" for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("大乐透后区号码转移概率网络图")
        plt.axis("off")
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_transition_network.png"), dpi=300, bbox_inches="tight")
        
        # 可视化前区号码的转移概率热力图
        plt.figure(figsize=(15, 12))
        
        # 创建转移概率矩阵
        front_transition_matrix = np.zeros((35, 35))
        for i in range(1, 36):
            for j in range(1, 36):
                front_transition_matrix[i-1, j-1] = front_transition_probs[i][j]
        
        # 绘制热力图
        ax = sns.heatmap(front_transition_matrix, cmap="YlGnBu", xticklabels=range(1, 36), yticklabels=range(1, 36))
        plt.title("大乐透前区号码转移概率热力图")
        plt.xlabel("下一期号码")
        plt.ylabel("当前期号码")
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_transition_heatmap.png"), dpi=300, bbox_inches="tight")
        
        # 返回马尔可夫链分析结果
        markov_chain_results = {
            "front_transition_probs": front_transition_probs,
            "back_transition_probs": back_transition_probs
        }
        
        return markov_chain_results
    
    def predict_by_markov_chain(self, explain=False, num_draws=1):
        """使用马尔可夫链预测下一期号码

        Args:
            explain: 是否解释预测结果
            num_draws: 生成的号码组数

        Returns:
            预测的前区号码列表和后区号码列表
        """
        print("使用马尔可夫链预测下一期号码...")
        
        # 获取马尔可夫链分析结果
        markov_results = self.analyze_markov_chain(save_result=False)
        front_transition_probs = markov_results["front_transition_probs"]
        back_transition_probs = markov_results["back_transition_probs"]
        
        # 获取最近一期的号码
        latest_front = self.front_balls_lists[0]
        latest_back = self.back_balls_lists[0]
        
        if explain:
            print(f"\n最近一期号码: 前区 {','.join([str(b).zfill(2) for b in sorted(latest_front)])}, 后区 {','.join([str(b).zfill(2) for b in sorted(latest_back)])}")
            print("\n基于马尔可夫链状态转移概率预测:")
        
        # 预测前区号码
        front_candidates = {}
        for current_ball in latest_front:
            for next_ball in range(1, 36):
                if next_ball not in front_candidates:
                    front_candidates[next_ball] = 0
                front_candidates[next_ball] += front_transition_probs[current_ball][next_ball]
        
        # 按概率排序
        sorted_front = sorted(front_candidates.items(), key=lambda x: x[1], reverse=True)
        
        if explain:
            print("\n前区号码预测:")
            print("基于上期号码的转移概率，候选号码排名(前10):")
            for i, (ball, prob) in enumerate(sorted_front[:10]):
                print(f"  {ball:02d}: 概率 {prob:.4f}")
        
        # 选择概率最高的5个不重复的号码
        predicted_front = []
        for ball, prob in sorted_front:
            if ball not in predicted_front:
                predicted_front.append(ball)
                if len(predicted_front) == 5:
                    break
        
        # 如果不足5个，随机补充
        if len(predicted_front) < 5:
            remaining = [ball for ball in range(1, 36) if ball not in predicted_front]
            additional = random.sample(remaining, 5 - len(predicted_front))
            if explain:
                print(f"\n前区号码不足5个，随机补充: {', '.join([str(b).zfill(2) for b in additional])}")
            predicted_front.extend(additional)
        
        predicted_front.sort()
        
        # 预测后区号码
        back_candidates = {}
        for current_ball in latest_back:
            for next_ball in range(1, 13):
                if next_ball not in back_candidates:
                    back_candidates[next_ball] = 0
                back_candidates[next_ball] += back_transition_probs[current_ball][next_ball]
        
        # 按概率排序
        sorted_back = sorted(back_candidates.items(), key=lambda x: x[1], reverse=True)
        
        if explain:
            print("\n后区号码预测:")
            print("基于上期号码的转移概率，候选号码排名:")
            for i, (ball, prob) in enumerate(sorted_back):
                print(f"  {ball:02d}: 概率 {prob:.4f}")
        
        # 选择概率最高的2个不重复的号码
        predicted_back = []
        for ball, prob in sorted_back:
            if ball not in predicted_back:
                predicted_back.append(ball)
                if len(predicted_back) == 2:
                    break
        
        # 如果不足2个，随机补充
        if len(predicted_back) < 2:
            remaining = [ball for ball in range(1, 13) if ball not in predicted_back]
            additional = random.sample(remaining, 2 - len(predicted_back))
            if explain:
                print(f"\n后区号码不足2个，随机补充: {', '.join([str(b).zfill(2) for b in additional])}")
            predicted_back.extend(additional)
        
        predicted_back.sort()
        
        if explain:
            print(f"\n最终预测号码: 前区 {','.join([str(b).zfill(2) for b in predicted_front])}, 后区 {','.join([str(b).zfill(2) for b in predicted_back])}")
        
        return predicted_front, predicted_back
    
    def predict_with_markov_chain(self, num_draws=5):
        """使用马尔可夫链预测下一期号码（向后兼容方法）

        Args:
            num_draws: 生成的号码组数

        Returns:
            预测号码列表，每个元素为(前区号码列表, 后区号码列表, 解释字符串)
        """
        print("使用马尔可夫链预测下一期号码...")
        
        # 生成多组预测号码
        predictions = []
        for _ in range(num_draws):
            front_balls, back_balls = self.predict_by_markov_chain(explain=False)
            # 为了保持向后兼容，添加一个空的解释字符串
            predictions.append((front_balls, back_balls, ""))
        
        return predictions
    
    def analyze_bayesian(self, save_result=True):
        """贝叶斯分析

        Args:
            save_result: 是否保存结果

        Returns:
            贝叶斯分析结果字典
        """
        print("进行贝叶斯分析...")
        
        # 统计前区号码频率
        front_balls_flat = [ball for sublist in self.front_balls_lists for ball in sublist]
        front_counter = Counter(front_balls_flat)
        
        # 确保所有可能的前区号码都在字典中
        for i in range(1, 36):
            if i not in front_counter:
                front_counter[i] = 0
        
        # 计算前区号码先验概率
        total_front_draws = len(self.front_balls_lists) * 5  # 总前区号码数量
        front_prior = {k: (v + 1) / (total_front_draws + 35) for k, v in front_counter.items()}  # 拉普拉斯平滑
        
        # 统计后区号码频率
        back_balls_flat = [ball for sublist in self.back_balls_lists for ball in sublist]
        back_counter = Counter(back_balls_flat)
        
        # 确保所有可能的后区号码都在字典中
        for i in range(1, 13):
            if i not in back_counter:
                back_counter[i] = 0
        
        # 计算后区号码先验概率
        total_back_draws = len(self.back_balls_lists) * 2  # 总后区号码数量
        back_prior = {k: (v + 1) / (total_back_draws + 12) for k, v in back_counter.items()}  # 拉普拉斯平滑
        
        # 计算条件概率（号码之间的关联性）
        front_conditional = {}
        for i in range(1, 36):
            front_conditional[i] = {}
            for j in range(1, 36):
                if i != j:
                    # 计算号码i和j同时出现的次数
                    co_occurrence = sum(1 for front_list in self.front_balls_lists if i in front_list and j in front_list)
                    # 计算号码i出现的次数
                    i_occurrence = sum(1 for front_list in self.front_balls_lists if i in front_list)
                    # 计算条件概率P(j|i)，即在i出现的情况下j也出现的概率
                    if i_occurrence > 0:
                        front_conditional[i][j] = (co_occurrence + 1) / (i_occurrence + 35)  # 拉普拉斯平滑
                    else:
                        front_conditional[i][j] = 1 / 35  # 如果i从未出现，使用均匀分布
        
        # 计算后区条件概率
        back_conditional = {}
        for i in range(1, 13):
            back_conditional[i] = {}
            for j in range(1, 13):
                if i != j:
                    # 计算号码i和j同时出现的次数
                    co_occurrence = sum(1 for back_list in self.back_balls_lists if i in back_list and j in back_list)
                    # 计算号码i出现的次数
                    i_occurrence = sum(1 for back_list in self.back_balls_lists if i in back_list)
                    # 计算条件概率P(j|i)，即在i出现的情况下j也出现的概率
                    if i_occurrence > 0:
                        back_conditional[i][j] = (co_occurrence + 1) / (i_occurrence + 12)  # 拉普拉斯平滑
                    else:
                        back_conditional[i][j] = 1 / 12  # 如果i从未出现，使用均匀分布
        
        # 保存贝叶斯分析结果
        if save_result:
            # 保存为JSON文件
            bayesian_results = {
                "front_prior": front_prior,
                "back_prior": back_prior,
                "front_conditional": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in front_conditional.items()},
                "back_conditional": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in back_conditional.items()}
            }
            
            with open(os.path.join(self.output_dir, "bayesian_analysis.json"), "w", encoding="utf-8") as f:
                json.dump(bayesian_results, f, ensure_ascii=False, indent=4)
        
        # 可视化前区号码先验概率
        plt.figure(figsize=(15, 6))
        plt.bar(range(1, 36), [front_prior[i] for i in range(1, 36)])
        plt.title("大乐透前区号码先验概率分布")
        plt.xlabel("号码")
        plt.ylabel("概率")
        plt.xticks(range(1, 36))
        plt.grid(axis="y", alpha=0.3)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_prior_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 可视化后区号码先验概率
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, 13), [back_prior[i] for i in range(1, 13)])
        plt.title("大乐透后区号码先验概率分布")
        plt.xlabel("号码")
        plt.ylabel("概率")
        plt.xticks(range(1, 13))
        plt.grid(axis="y", alpha=0.3)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_prior_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 返回贝叶斯分析结果
        bayesian_results = {
            "front_prior": front_prior,
            "back_prior": back_prior,
            "front_conditional": front_conditional,
            "back_conditional": back_conditional
        }
        
        return bayesian_results
             back_counter[i] = 0
        
        # 计算后区号码先验概率
        total_back_draws = len(self.back_balls_lists) * 2  # 总后区号码数量
        back_prior = {k: (v + 1) / (total_back_draws + 12) for k, v in back_counter.items()}  # 拉普拉斯平滑
        
        # 计算前区号码的条件概率
        front_conditional = {}
        for i in range(1, 36):
            front_conditional[i] = {}
            for j in range(1, 36):
                if i != j:
                    # 计算号码i和j一起出现的次数
                    count = 0
                    for front_list in self.front_balls_lists:
                        if i in front_list and j in front_list:
                            count += 1
                    
                    # 计算条件概率 P(j|i) = P(i,j) / P(i)
                    front_conditional[i][j] = (count + 1) / (front_counter[i] + 1)  # 拉普拉斯平滑
        
        # 计算后区号码的条件概率
        back_conditional = {}
        for i in range(1, 13):
            back_conditional[i] = {}
            for j in range(1, 13):
                if i != j:
                    # 计算号码i和j一起出现的次数
                    count = 0
                    for back_list in self.back_balls_lists:
                        if i in back_list and j in back_list:
                            count += 1
                    
                    # 计算条件概率 P(j|i) = P(i,j) / P(i)
                    back_conditional[i][j] = (count + 1) / (back_counter[i] + 1)  # 拉普拉斯平滑
        
        # 保存贝叶斯分析结果
        if save_result:
            # 保存为JSON文件
            bayesian_results = {
                "front_prior": front_prior,
                "back_prior": back_prior,
                "front_conditional": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in front_conditional.items()},
                "back_conditional": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in back_conditional.items()}
            }
            
            with open(os.path.join(self.output_dir, "bayesian_analysis.json"), "w", encoding="utf-8") as f:
                json.dump(bayesian_results, f, ensure_ascii=False, indent=4)
        
        # 绘制前区号码先验概率图
        plt.figure(figsize=(15, 6))
        plt.bar(front_prior.keys(), front_prior.values())
        plt.title("大乐透前区号码先验概率分布")
        plt.xlabel("号码")
        plt.ylabel("先验概率")
        plt.xticks(range(1, 36))
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_prior_probability.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区号码先验概率图
        plt.figure(figsize=(12, 6))
        plt.bar(back_prior.keys(), back_prior.values())
        plt.title("大乐透后区号码先验概率分布")
        plt.xlabel("号码")
        plt.ylabel("先验概率")
        plt.xticks(range(1, 13))
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_prior_probability.png"), dpi=300, bbox_inches="tight")
        
        # 返回贝叶斯分析结果
        bayesian_results = {
            "front_prior": front_prior,
            "back_prior": back_prior,
            "front_conditional": front_conditional,
            "back_conditional": back_conditional
        }
        
        return bayesian_results
    
    def predict_with_bayesian(self, num_draws=5):
        """使用贝叶斯分析预测下一期号码（向后兼容方法）

        Args:
            num_draws: 生成的号码组数

        Returns:
            预测号码列表，每个元素为(前区号码列表, 后区号码列表, 解释字符串)
        """
        print("使用贝叶斯分析预测下一期号码...")
        
        # 生成多组预测号码
        predictions = []
        for _ in range(num_draws):
            front_balls, back_balls = self.predict_by_bayes(explain=False)
            # 为了保持向后兼容，添加一个空的解释字符串
            predictions.append((front_balls, back_balls, ""))
        
        return predictions
    
    def predict_by_bayes(self, explain=False, num_draws=1):
        """使用贝叶斯分析预测下一期号码

        Args:
            explain: 是否解释预测结果
            num_draws: 生成的号码组数

        Returns:
            预测的前区号码列表和后区号码列表
        """
        print("使用贝叶斯分析预测下一期号码...")
        
        # 获取贝叶斯分析结果
        bayesian_results = self.analyze_bayesian(save_result=False)
        front_prior = bayesian_results["front_prior"]
        back_prior = bayesian_results["back_prior"]
        front_conditional = bayesian_results["front_conditional"]
        back_conditional = bayesian_results["back_conditional"]
        
        # 获取统计特征分析结果
        stats_results = self.analyze_statistical_features(save_result=False)
        
        if explain:
            print("\n基于贝叶斯后验概率预测:")
        
        # 预测前区号码
        # 先选择一个号码作为起点（基于先验概率）
        first_ball = np.random.choice(list(range(1, 36)), p=[front_prior[i] for i in range(1, 36)])
        predicted_front = [first_ball]
        
        if explain:
            print(f"\n前区号码预测:")
            print(f"首先基于先验概率选择起始号码: {first_ball:02d}")
        
        # 基于条件概率选择剩余号码
        for i in range(4):
            # 计算每个候选号码的后验概率
            posterior = {}
            for ball in range(1, 36):
                if ball not in predicted_front:
                    # 基于已选号码计算后验概率
                    prob = front_prior[ball]
                    for selected in predicted_front:
                        if ball in front_conditional[selected]:
                            prob *= front_conditional[selected][ball]
                    posterior[ball] = prob
            
            # 归一化后验概率
            total = sum(posterior.values())
            if total > 0:
                posterior = {k: v / total for k, v in posterior.items()}
                
                # 按后验概率选择下一个号码
                next_ball = np.random.choice(list(posterior.keys()), p=list(posterior.values()))
                predicted_front.append(next_ball)
                
                if explain:
                    print(f"第{i+2}个号码基于后验概率选择: {next_ball:02d}")
            else:
                # 如果后验概率全为0，随机选择
                remaining = [ball for ball in range(1, 36) if ball not in predicted_front]
                next_ball = random.choice(remaining)
                predicted_front.append(next_ball)
                
                if explain:
                    print(f"第{i+2}个号码随机选择: {next_ball:02d}")
        
        predicted_front.sort()
        
        # 预测后区号码（直接基于先验概率）
        back_probs = [back_prior[i] for i in range(1, 13)]
        back_probs_normalized = [p / sum(back_probs) for p in back_probs]
        predicted_back = np.random.choice(list(range(1, 13)), size=2, replace=False, p=back_probs_normalized)
        predicted_back = sorted(predicted_back)
        
        if explain:
            print("\n后区号码预测:")
            print("基于先验概率选择后区号码:")
            for i, ball in enumerate(predicted_back):
                print(f"  第{i+1}个号码: {ball:02d}, 概率: {back_prior[ball]:.4f}")
            
            print(f"\n最终预测号码: 前区 {','.join([str(b).zfill(2) for b in predicted_front])}, 后区 {','.join([str(b).zfill(2) for b in predicted_back])}")
        
        return predicted_front, predicted_back
        
    def compare_with_history(self, front_balls, back_balls):
        """将用户输入的号码与历史数据进行对比分析

        Args:
            front_balls: 前区号码列表
            back_balls: 后区号码列表

        Returns:
            对比分析结果字典
        """
        print("将号码与历史数据进行对比分析...")
        
        # 检查号码格式
        if len(front_balls) != 5 or len(back_balls) != 2:
            print("号码格式错误，前区应为5个号码，后区应为2个号码")
            return None
        
        # 统计前区号码在历史数据中的出现次数
        front_counts = {}
        for ball in front_balls:
            count = 0
            for front_list in self.front_balls_lists:
                if ball in front_list:
                    count += 1
            front_counts[ball] = count
        
        # 统计后区号码在历史数据中的出现次数
        back_counts = {}
        for ball in back_balls:
            count = 0
            for back_list in self.back_balls_lists:
                if ball in back_list:
                    count += 1
            back_counts[ball] = count
        
        # 检查是否与历史开奖号码完全匹配
        exact_matches = []
        for i, (front_list, back_list) in enumerate(zip(self.front_balls_lists, self.back_balls_lists)):
            if set(front_balls) == set(front_list) and set(back_balls) == set(back_list):
                exact_matches.append({
                    "issue": self.df.iloc[i]["issue"],
                    "date": self.df.iloc[i]["date"]
                })
        
        # 检查前区号码匹配情况
        front_match_counts = []
        for front_list in self.front_balls_lists:
            match_count = len(set(front_balls) & set(front_list))
            front_match_counts.append(match_count)
        
        # 统计前区匹配数分布
        front_match_distribution = Counter(front_match_counts)
        
        # 检查后区号码匹配情况
        back_match_counts = []
        for back_list in self.back_balls_lists:
            match_count = len(set(back_balls) & set(back_list))
            back_match_counts.append(match_count)
        
        # 统计后区匹配数分布
        back_match_distribution = Counter(back_match_counts)
        
        # 统计中奖情况
        prize_counts = {i: 0 for i in range(9)}  # 0-8等奖
        for i, (front_list, back_list) in enumerate(zip(self.front_balls_lists, self.back_balls_lists)):
            front_match = len(set(front_balls) & set(front_list))
            back_match = len(set(back_balls) & set(back_list))
            
            # 判断中奖等级
            if front_match == 5 and back_match == 2:
                prize = 1  # 一等奖
            elif front_match == 5 and back_match == 1:
                prize = 2  # 二等奖
            elif front_match == 5 and back_match == 0:
                prize = 3  # 三等奖
            elif front_match == 4 and back_match == 2:
                prize = 4  # 四等奖
            elif (front_match == 4 and back_match == 1) or (front_match == 3 and back_match == 2):
                prize = 5  # 五等奖
            elif (front_match == 4 and back_match == 0) or (front_match == 3 and back_match == 1) or (front_match == 2 and back_match == 2):
                prize = 6  # 六等奖
            elif (front_match == 3 and back_match == 0) or (front_match == 2 and back_match == 1) or (front_match == 1 and back_match == 2) or (front_match == 0 and back_match == 2):
                prize = 7  # 七等奖
            elif (front_match == 2 and back_match == 0) or (front_match == 1 and back_match == 1) or (front_match == 0 and back_match == 1):
                prize = 8  # 八等奖
            else:
                prize = 0  # 未中奖
            
            prize_counts[prize] += 1
        
        # 输出对比分析结果
        print("\n号码对比分析结果:")
        print(f"前区号码: {front_balls}")
        print(f"后区号码: {back_balls}")
        print("\n前区号码出现次数:")
        for ball, count in front_counts.items():
            print(f"号码 {ball}: {count} 次 ({count/len(self.front_balls_lists)*100:.2f}%)")
        
        print("\n后区号码出现次数:")
        for ball, count in back_counts.items():
            print(f"号码 {ball}: {count} 次 ({count/len(self.back_balls_lists)*100:.2f}%)")
        
        if exact_matches:
            print("\n完全匹配的历史记录:")
            for match in exact_matches:
                print(f"期号: {match['issue']}, 日期: {match['date']}")
        else:
            print("\n没有完全匹配的历史记录")
        
        print("\n前区匹配数分布:")
        for i in range(6):
            count = front_match_distribution.get(i, 0)
            print(f"{i}个号码匹配: {count} 次 ({count/len(self.front_balls_lists)*100:.2f}%)")
        
        print("\n后区匹配数分布:")
        for i in range(3):
            count = back_match_distribution.get(i, 0)
            print(f"{i}个号码匹配: {count} 次 ({count/len(self.back_balls_lists)*100:.2f}%)")
        
        print("\n历史中奖情况:")
        prize_names = {
            1: "一等奖",
            2: "二等奖",
            3: "三等奖",
            4: "四等奖",
            5: "五等奖",
            6: "六等奖",
            7: "七等奖",
            8: "八等奖",
            0: "未中奖"
        }
        for i in range(1, 9):
            count = prize_counts[i]
            print(f"{prize_names[i]}: {count} 次 ({count/len(self.front_balls_lists)*100:.2f}%)")
        
        # 返回对比分析结果
        compare_results = {
            "front_balls": front_balls,
            "back_balls": back_balls,
            "front_counts": front_counts,
            "back_counts": back_counts,
            "exact_matches": exact_matches,
            "front_match_distribution": front_match_distribution,
            "back_match_distribution": back_match_distribution,
            "prize_counts": prize_counts
        }
        
        return compare_results
    
    def run_advanced_analysis(self):
        """运行所有高级分析

        Returns:
            分析结果字典
        """
        print("开始高级分析...")
        
        # 创建结果字典
        results = {}
        
        # 分析统计学特征
        stats_results = self.analyze_statistical_features()
        results["statistical_features"] = stats_results
        
        # 分析概率分布
        prob_results = self.analyze_probability_distribution()
        results["probability_distribution"] = prob_results
        
        # 分析频率模式
        pattern_results = self.analyze_frequency_patterns()
        results["frequency_patterns"] = pattern_results
        
        # 分析马尔可夫链
        markov_results = self.analyze_markov_chain()
        results["markov_chain"] = markov_results
        
        # 贝叶斯分析
        bayesian_results = self.analyze_bayesian()
        results["bayesian"] = bayesian_results
        
        print("高级分析完成")
        return results


if __name__ == "__main__":
    # 测试高级分析器
    analyzer = DLTAdvancedAnalyzer("../data/dlt_data.csv", "../output/advanced")
    analyzer.run_advanced_analysis()