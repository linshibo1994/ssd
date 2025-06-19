#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础分析器模块
提供大乐透数据的基本分析功能
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import seaborn as sns
from datetime import datetime
import warnings

# 设置中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except Exception as e:
    print(f"设置中文显示失败: {e}")

# 忽略警告
warnings.filterwarnings("ignore")


class BasicAnalyzer:
    """大乐透基础分析器"""
    
    def __init__(self, data_file, output_dir="./output"):
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
    
    def analyze_frequency(self, save_result=True):
        """分析号码出现频率

        Args:
            save_result: 是否保存结果

        Returns:
            (前区号码频率字典, 后区号码频率字典)
        """
        print("分析号码出现频率...")
        
        # 统计前区号码频率
        front_balls_flat = [ball for sublist in self.front_balls_lists for ball in sublist]
        front_counter = Counter(front_balls_flat)
        
        # 确保所有可能的前区号码都在字典中
        for i in range(1, 36):
            if i not in front_counter:
                front_counter[i] = 0
        
        # 统计后区号码频率
        back_balls_flat = [ball for sublist in self.back_balls_lists for ball in sublist]
        back_counter = Counter(back_balls_flat)
        
        # 确保所有可能的后区号码都在字典中
        for i in range(1, 13):
            if i not in back_counter:
                back_counter[i] = 0
        
        # 绘制前区号码频率图
        plt.figure(figsize=(15, 6))
        plt.bar(front_counter.keys(), front_counter.values())
        plt.title("大乐透前区号码出现频率")
        plt.xlabel("号码")
        plt.ylabel("出现次数")
        plt.xticks(range(1, 36))
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_balls_frequency.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区号码频率图
        plt.figure(figsize=(12, 6))
        plt.bar(back_counter.keys(), back_counter.values())
        plt.title("大乐透后区号码出现频率")
        plt.xlabel("号码")
        plt.ylabel("出现次数")
        plt.xticks(range(1, 13))
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_balls_frequency.png"), dpi=300, bbox_inches="tight")
            
            # 保存频率数据到CSV
            front_freq_df = pd.DataFrame({
                "number": list(front_counter.keys()),
                "frequency": list(front_counter.values())
            })
            front_freq_df = front_freq_df.sort_values(by="number")
            front_freq_df.to_csv(os.path.join(self.output_dir, "front_balls_frequency.csv"), index=False)
            
            back_freq_df = pd.DataFrame({
                "number": list(back_counter.keys()),
                "frequency": list(back_counter.values())
            })
            back_freq_df = back_freq_df.sort_values(by="number")
            back_freq_df.to_csv(os.path.join(self.output_dir, "back_balls_frequency.csv"), index=False)
        
        return dict(front_counter), dict(back_counter)
    
    def analyze_missing_values(self, save_result=True):
        """分析号码遗漏值

        Args:
            save_result: 是否保存结果

        Returns:
            (前区号码当前遗漏值字典, 后区号码当前遗漏值字典)
        """
        print("分析号码遗漏值...")
        
        # 计算前区号码遗漏值
        front_missing = {i: 0 for i in range(1, 36)}
        front_current_missing = {i: 0 for i in range(1, 36)}
        
        # 计算每期每个号码的遗漏值
        front_missing_history = []
        
        for i, front_list in enumerate(self.front_balls_lists):
            # 更新当前遗漏值
            for num in range(1, 36):
                if num in front_list:
                    front_current_missing[num] = 0
                else:
                    front_current_missing[num] += 1
            
            # 记录历史遗漏值
            front_missing_history.append(front_current_missing.copy())
        
        # 计算后区号码遗漏值
        back_missing = {i: 0 for i in range(1, 13)}
        back_current_missing = {i: 0 for i in range(1, 13)}
        
        # 计算每期每个号码的遗漏值
        back_missing_history = []
        
        for i, back_list in enumerate(self.back_balls_lists):
            # 更新当前遗漏值
            for num in range(1, 13):
                if num in back_list:
                    back_current_missing[num] = 0
                else:
                    back_current_missing[num] += 1
            
            # 记录历史遗漏值
            back_missing_history.append(back_current_missing.copy())
        
        # 绘制前区号码当前遗漏值图
        plt.figure(figsize=(15, 6))
        plt.bar(front_current_missing.keys(), front_current_missing.values())
        plt.title("大乐透前区号码当前遗漏值")
        plt.xlabel("号码")
        plt.ylabel("遗漏值")
        plt.xticks(range(1, 36))
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_balls_missing.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区号码当前遗漏值图
        plt.figure(figsize=(12, 6))
        plt.bar(back_current_missing.keys(), back_current_missing.values())
        plt.title("大乐透后区号码当前遗漏值")
        plt.xlabel("号码")
        plt.ylabel("遗漏值")
        plt.xticks(range(1, 13))
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_balls_missing.png"), dpi=300, bbox_inches="tight")
            
            # 保存遗漏值数据到CSV
            front_missing_df = pd.DataFrame({
                "number": list(front_current_missing.keys()),
                "missing_value": list(front_current_missing.values())
            })
            front_missing_df = front_missing_df.sort_values(by="number")
            front_missing_df.to_csv(os.path.join(self.output_dir, "front_balls_missing.csv"), index=False)
            
            back_missing_df = pd.DataFrame({
                "number": list(back_current_missing.keys()),
                "missing_value": list(back_current_missing.values())
            })
            back_missing_df = back_missing_df.sort_values(by="number")
            back_missing_df.to_csv(os.path.join(self.output_dir, "back_balls_missing.csv"), index=False)
        
        return front_current_missing, back_current_missing
    
    def analyze_hot_numbers(self, recent_periods=30, save_result=True):
        """分析热门号码

        Args:
            recent_periods: 最近多少期数据
            save_result: 是否保存结果

        Returns:
            (前区热门号码列表, 后区热门号码列表)
        """
        print(f"分析最近{recent_periods}期热门号码...")
        
        # 获取最近N期数据
        recent_front_lists = self.front_balls_lists[:recent_periods]
        recent_back_lists = self.back_balls_lists[:recent_periods]
        
        # 统计前区号码频率
        front_balls_flat = [ball for sublist in recent_front_lists for ball in sublist]
        front_counter = Counter(front_balls_flat)
        
        # 确保所有可能的前区号码都在字典中
        for i in range(1, 36):
            if i not in front_counter:
                front_counter[i] = 0
        
        # 统计后区号码频率
        back_balls_flat = [ball for sublist in recent_back_lists for ball in sublist]
        back_counter = Counter(back_balls_flat)
        
        # 确保所有可能的后区号码都在字典中
        for i in range(1, 13):
            if i not in back_counter:
                back_counter[i] = 0
        
        # 获取前区热门号码（出现频率前10的号码）
        front_hot = [num for num, _ in front_counter.most_common(10)]
        
        # 获取后区热门号码（出现频率前5的号码）
        back_hot = [num for num, _ in back_counter.most_common(5)]
        
        # 绘制前区热门号码图
        plt.figure(figsize=(12, 6))
        hot_front_counts = [front_counter[num] for num in front_hot]
        plt.bar(front_hot, hot_front_counts)
        plt.title(f"大乐透前区热门号码（最近{recent_periods}期）")
        plt.xlabel("号码")
        plt.ylabel("出现次数")
        plt.xticks(front_hot)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 在柱状图上标注出现次数
        for i, count in enumerate(hot_front_counts):
            plt.text(front_hot[i], count + 0.1, str(count), ha="center")
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_balls_hot.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区热门号码图
        plt.figure(figsize=(10, 6))
        hot_back_counts = [back_counter[num] for num in back_hot]
        plt.bar(back_hot, hot_back_counts)
        plt.title(f"大乐透后区热门号码（最近{recent_periods}期）")
        plt.xlabel("号码")
        plt.ylabel("出现次数")
        plt.xticks(back_hot)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 在柱状图上标注出现次数
        for i, count in enumerate(hot_back_counts):
            plt.text(back_hot[i], count + 0.1, str(count), ha="center")
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_balls_hot.png"), dpi=300, bbox_inches="tight")
            
            # 保存热门号码数据到CSV
            hot_numbers_df = pd.DataFrame({
                "front_hot": front_hot + [None] * (len(back_hot) - len(front_hot)) if len(front_hot) < len(back_hot) else front_hot,
                "front_count": hot_front_counts + [None] * (len(back_hot) - len(front_hot)) if len(front_hot) < len(back_hot) else hot_front_counts,
                "back_hot": back_hot + [None] * (len(front_hot) - len(back_hot)) if len(back_hot) < len(front_hot) else back_hot,
                "back_count": hot_back_counts + [None] * (len(front_hot) - len(back_hot)) if len(back_hot) < len(front_hot) else hot_back_counts
            })
            hot_numbers_df.to_csv(os.path.join(self.output_dir, "hot_numbers.csv"), index=False)
        
        return front_hot, back_hot
    
    def analyze_cold_numbers(self, recent_periods=30, save_result=True):
        """分析冷门号码

        Args:
            recent_periods: 最近多少期数据
            save_result: 是否保存结果

        Returns:
            (前区冷门号码列表, 后区冷门号码列表)
        """
        print(f"分析最近{recent_periods}期冷门号码...")
        
        # 获取最近N期数据
        recent_front_lists = self.front_balls_lists[:recent_periods]
        recent_back_lists = self.back_balls_lists[:recent_periods]
        
        # 统计前区号码频率
        front_balls_flat = [ball for sublist in recent_front_lists for ball in sublist]
        front_counter = Counter(front_balls_flat)
        
        # 确保所有可能的前区号码都在字典中
        for i in range(1, 36):
            if i not in front_counter:
                front_counter[i] = 0
        
        # 统计后区号码频率
        back_balls_flat = [ball for sublist in recent_back_lists for ball in sublist]
        back_counter = Counter(back_balls_flat)
        
        # 确保所有可能的后区号码都在字典中
        for i in range(1, 13):
            if i not in back_counter:
                back_counter[i] = 0
        
        # 获取前区冷门号码（出现频率后10的号码）
        front_cold = [num for num, _ in sorted(front_counter.items(), key=lambda x: x[1])[:10]]
        
        # 获取后区冷门号码（出现频率后5的号码）
        back_cold = [num for num, _ in sorted(back_counter.items(), key=lambda x: x[1])[:5]]
        
        # 绘制前区冷门号码图
        plt.figure(figsize=(12, 6))
        cold_front_counts = [front_counter[num] for num in front_cold]
        plt.bar(front_cold, cold_front_counts)
        plt.title(f"大乐透前区冷门号码（最近{recent_periods}期）")
        plt.xlabel("号码")
        plt.ylabel("出现次数")
        plt.xticks(front_cold)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 在柱状图上标注出现次数
        for i, count in enumerate(cold_front_counts):
            plt.text(front_cold[i], count + 0.1, str(count), ha="center")
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_balls_cold.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区冷门号码图
        plt.figure(figsize=(10, 6))
        cold_back_counts = [back_counter[num] for num in back_cold]
        plt.bar(back_cold, cold_back_counts)
        plt.title(f"大乐透后区冷门号码（最近{recent_periods}期）")
        plt.xlabel("号码")
        plt.ylabel("出现次数")
        plt.xticks(back_cold)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 在柱状图上标注出现次数
        for i, count in enumerate(cold_back_counts):
            plt.text(back_cold[i], count + 0.1, str(count), ha="center")
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_balls_cold.png"), dpi=300, bbox_inches="tight")
            
            # 保存冷门号码数据到CSV
            cold_numbers_df = pd.DataFrame({
                "front_cold": front_cold + [None] * (len(back_cold) - len(front_cold)) if len(front_cold) < len(back_cold) else front_cold,
                "front_count": cold_front_counts + [None] * (len(back_cold) - len(front_cold)) if len(front_cold) < len(back_cold) else cold_front_counts,
                "back_cold": back_cold + [None] * (len(front_cold) - len(back_cold)) if len(back_cold) < len(front_cold) else back_cold,
                "back_count": cold_back_counts + [None] * (len(front_cold) - len(back_cold)) if len(back_cold) < len(front_cold) else cold_back_counts
            })
            cold_numbers_df.to_csv(os.path.join(self.output_dir, "cold_numbers.csv"), index=False)
        
        return front_cold, back_cold
    
    def analyze_number_distribution(self, save_result=True):
        """分析号码分布

        Args:
            save_result: 是否保存结果

        Returns:
            None
        """
        print("分析号码分布...")
        
        # 分析前区号码分布
        front_distribution = {}
        for i in range(1, 36):
            front_distribution[i] = 0
        
        for front_list in self.front_balls_lists:
            for ball in front_list:
                front_distribution[ball] += 1
        
        # 计算前区号码分布比例
        total_front_draws = len(self.front_balls_lists) * 5  # 总前区号码数量
        front_distribution_ratio = {k: v / total_front_draws * 100 for k, v in front_distribution.items()}
        
        # 分析后区号码分布
        back_distribution = {}
        for i in range(1, 13):
            back_distribution[i] = 0
        
        for back_list in self.back_balls_lists:
            for ball in back_list:
                back_distribution[ball] += 1
        
        # 计算后区号码分布比例
        total_back_draws = len(self.back_balls_lists) * 2  # 总后区号码数量
        back_distribution_ratio = {k: v / total_back_draws * 100 for k, v in back_distribution.items()}
        
        # 绘制前区号码分布热力图
        plt.figure(figsize=(15, 6))
        
        # 创建5x7的网格（覆盖1-35）
        front_grid = np.zeros((5, 7))
        for i in range(5):
            for j in range(7):
                num = i * 7 + j + 1
                if num <= 35:
                    front_grid[i, j] = front_distribution_ratio.get(num, 0)
                else:
                    front_grid[i, j] = np.nan
        
        # 绘制热力图
        ax = sns.heatmap(front_grid, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "出现比例 (%)"}, linewidths=0.5)
        
        # 设置坐标轴标签
        ax.set_xticks(np.arange(7) + 0.5)
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_xticklabels(range(1, 8))
        ax.set_yticklabels(range(1, 6))
        
        plt.title("大乐透前区号码分布热力图 (%)")
        plt.xlabel("列")
        plt.ylabel("行")
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_balls_distribution.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区号码分布热力图
        plt.figure(figsize=(12, 4))
        
        # 创建2x6的网格（覆盖1-12）
        back_grid = np.zeros((2, 6))
        for i in range(2):
            for j in range(6):
                num = i * 6 + j + 1
                if num <= 12:
                    back_grid[i, j] = back_distribution_ratio.get(num, 0)
                else:
                    back_grid[i, j] = np.nan
        
        # 绘制热力图
        ax = sns.heatmap(back_grid, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "出现比例 (%)"}, linewidths=0.5)
        
        # 设置坐标轴标签
        ax.set_xticks(np.arange(6) + 0.5)
        ax.set_yticks(np.arange(2) + 0.5)
        ax.set_xticklabels(range(1, 7))
        ax.set_yticklabels(range(1, 3))
        
        plt.title("大乐透后区号码分布热力图 (%)")
        plt.xlabel("列")
        plt.ylabel("行")
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_balls_distribution.png"), dpi=300, bbox_inches="tight")
            
            # 保存分布数据到CSV
            distribution_df = pd.DataFrame({
                "number": list(range(1, 36)),
                "front_count": [front_distribution[i] for i in range(1, 36)],
                "front_ratio": [front_distribution_ratio[i] for i in range(1, 36)]
            })
            distribution_df.to_csv(os.path.join(self.output_dir, "front_balls_distribution.csv"), index=False)
            
            back_distribution_df = pd.DataFrame({
                "number": list(range(1, 13)),
                "back_count": [back_distribution[i] for i in range(1, 13)],
                "back_ratio": [back_distribution_ratio[i] for i in range(1, 13)]
            })
            back_distribution_df.to_csv(os.path.join(self.output_dir, "back_balls_distribution.csv"), index=False)
    
    def analyze_number_trend(self, recent_periods=50, save_result=True):
        """分析号码走势

        Args:
            recent_periods: 最近多少期数据
            save_result: 是否保存结果

        Returns:
            None
        """
        print(f"分析最近{recent_periods}期号码走势...")
        
        # 获取最近N期数据
        recent_issues = self.df["issue"].values[:recent_periods]
        recent_front_lists = self.front_balls_lists[:recent_periods]
        recent_back_lists = self.back_balls_lists[:recent_periods]
        
        # 创建走势图数据
        front_trend = np.zeros((recent_periods, 35))
        for i, front_list in enumerate(recent_front_lists):
            for ball in front_list:
                front_trend[i, ball-1] = ball
        
        back_trend = np.zeros((recent_periods, 12))
        for i, back_list in enumerate(recent_back_lists):
            for ball in back_list:
                back_trend[i, ball-1] = ball
        
        # 绘制前区号码走势图
        plt.figure(figsize=(15, 10))
        
        # 绘制网格
        for i in range(36):
            plt.axhline(y=i, color="gray", linestyle="-", alpha=0.2)
        
        for i in range(recent_periods):
            plt.axvline(x=i, color="gray", linestyle="-", alpha=0.2)
        
        # 绘制号码点
        for i in range(recent_periods):
            for j in range(35):
                if front_trend[i, j] > 0:
                    plt.scatter(i, j+1, color="red", s=50)
        
        plt.title(f"大乐透前区号码走势图（最近{recent_periods}期）")
        plt.xlabel("期号")
        plt.ylabel("号码")
        plt.yticks(range(1, 36))
        plt.xticks(range(0, recent_periods, 5), recent_issues[::5], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "front_balls_trend.png"), dpi=300, bbox_inches="tight")
        
        # 绘制后区号码走势图
        plt.figure(figsize=(15, 8))
        
        # 绘制网格
        for i in range(13):
            plt.axhline(y=i, color="gray", linestyle="-", alpha=0.2)
        
        for i in range(recent_periods):
            plt.axvline(x=i, color="gray", linestyle="-", alpha=0.2)
        
        # 绘制号码点
        for i in range(recent_periods):
            for j in range(12):
                if back_trend[i, j] > 0:
                    plt.scatter(i, j+1, color="blue", s=50)
        
        plt.title(f"大乐透后区号码走势图（最近{recent_periods}期）")
        plt.xlabel("期号")
        plt.ylabel("号码")
        plt.yticks(range(1, 13))
        plt.xticks(range(0, recent_periods, 5), recent_issues[::5], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_result:
            plt.savefig(os.path.join(self.output_dir, "back_balls_trend.png"), dpi=300, bbox_inches="tight")
    
    def run_basic_analysis(self):
        """运行所有基础分析

        Returns:
            分析结果字典
        """
        print("开始基础分析...")
        
        # 创建结果字典
        results = {}
        
        # 分析号码出现频率
        front_freq, back_freq = self.analyze_frequency()
        results["frequency"] = {"front": front_freq, "back": back_freq}
        
        # 分析号码遗漏值
        front_missing, back_missing = self.analyze_missing_values()
        results["missing"] = {"front": front_missing, "back": back_missing}
        
        # 分析热门号码
        front_hot, back_hot = self.analyze_hot_numbers()
        results["hot"] = {"front": front_hot, "back": back_hot}
        
        # 分析冷门号码
        front_cold, back_cold = self.analyze_cold_numbers()
        results["cold"] = {"front": front_cold, "back": back_cold}
        
        # 分析号码分布
        self.analyze_number_distribution()
        
        # 分析号码走势
        self.analyze_number_trend()
        
        print("基础分析完成")
        return results


if __name__ == "__main__":
    # 测试基础分析器
    analyzer = BasicAnalyzer("../data/dlt_data.csv", "../output/basic")
    analyzer.run_basic_analysis()