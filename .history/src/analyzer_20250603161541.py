#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据分析模块
分析双色球历史开奖数据，生成统计图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from collections import Counter

# 设置中文字体
try:
    # 尝试使用系统中文字体
    font_path = '/System/Library/Fonts/PingFang.ttc'  # macOS中文字体
    if not os.path.exists(font_path):
        font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux中文字体
    if not os.path.exists(font_path):
        font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows中文字体
    
    if os.path.exists(font_path):
        font = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font.get_name()
    else:
        print("警告：未找到合适的中文字体，图表中文可能显示为乱码")
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except Exception as e:
    print(f"设置中文字体时出错: {e}")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False


class SSQAnalyzer:
    """双色球数据分析类"""

    def __init__(self, data_file="../data/ssq_data.csv", output_dir="../data"):
        """初始化分析器

        Args:
            data_file: 数据文件路径
            output_dir: 输出目录
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.data = None
        self.red_range = range(1, 34)  # 红球范围1-33
        self.blue_range = range(1, 17)  # 蓝球范围1-16
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """加载数据

        Returns:
            成功返回True，失败返回False
        """
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
            
            print(f"成功加载{len(self.data)}条数据")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def analyze_number_frequency(self):
        """分析号码出现频率

        Returns:
            (红球频率字典, 蓝球频率字典)
        """
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

    def plot_number_frequency(self):
        """绘制号码频率图"""
        red_freq, blue_freq = self.analyze_number_frequency()
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 红球频率图
        red_nums = list(self.red_range)
        red_freqs = [red_freq.get(num, 0) for num in red_nums]
        
        bars = ax1.bar(red_nums, red_freqs, color='red', alpha=0.7)
        ax1.set_title('红球出现频率', fontsize=14)
        ax1.set_xlabel('红球号码', fontsize=12)
        ax1.set_ylabel('出现概率', fontsize=12)
        ax1.set_xticks(red_nums)
        ax1.set_xticklabels([str(num) for num in red_nums], rotation=90, fontsize=8)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 为频率最高的红球添加标签
        top_reds = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        for num, freq in top_reds:
            ax1.text(num, freq, f'{freq:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 蓝球频率图
        blue_nums = list(self.blue_range)
        blue_freqs = [blue_freq.get(num, 0) for num in blue_nums]
        
        bars = ax2.bar(blue_nums, blue_freqs, color='blue', alpha=0.7)
        ax2.set_title('蓝球出现频率', fontsize=14)
        ax2.set_xlabel('蓝球号码', fontsize=12)
        ax2.set_ylabel('出现概率', fontsize=12)
        ax2.set_xticks(blue_nums)
        ax2.set_xticklabels([str(num) for num in blue_nums], fontsize=10)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 为频率最高的蓝球添加标签
        top_blues = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        for num, freq in top_blues:
            ax2.text(num, freq, f'{freq:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'number_frequency.png'), dpi=300)
        plt.close()
        
        print("号码频率图已保存")

    def analyze_number_combinations(self):
        """分析号码组合特征"""
        # 计算红球和值
        self.data['red_sum'] = sum(self.data[f'red_{i}'] for i in range(1, 7))
        
        # 计算红球奇偶比
        for i in range(1, 7):
            self.data[f'red_{i}_odd'] = self.data[f'red_{i}'] % 2 != 0
        
        self.data['odd_count'] = sum(self.data[f'red_{i}_odd'] for i in range(1, 7))
        self.data['even_count'] = 6 - self.data['odd_count']
        
        # 计算红球大小比（大于等于17为大）
        for i in range(1, 7):
            self.data[f'red_{i}_big'] = self.data[f'red_{i}'] >= 17
        
        self.data['big_count'] = sum(self.data[f'red_{i}_big'] for i in range(1, 7))
        self.data['small_count'] = 6 - self.data['big_count']
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 红球和值分布
        sns.histplot(self.data['red_sum'], bins=30, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('红球和值分布', fontsize=14)
        axes[0, 0].set_xlabel('和值', fontsize=12)
        axes[0, 0].set_ylabel('频次', fontsize=12)
        
        # 奇偶比分布
        odd_even_counts = self.data.groupby(['odd_count', 'even_count']).size().reset_index(name='count')
        odd_even_counts['ratio'] = odd_even_counts.apply(lambda x: f"{int(x['odd_count'])}:{int(x['even_count'])}", axis=1)
        
        sns.barplot(x='ratio', y='count', data=odd_even_counts, ax=axes[0, 1])
        axes[0, 1].set_title('红球奇偶比分布', fontsize=14)
        axes[0, 1].set_xlabel('奇偶比', fontsize=12)
        axes[0, 1].set_ylabel('频次', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 大小比分布
        big_small_counts = self.data.groupby(['big_count', 'small_count']).size().reset_index(name='count')
        big_small_counts['ratio'] = big_small_counts.apply(lambda x: f"{int(x['big_count'])}:{int(x['small_count'])}", axis=1)
        
        sns.barplot(x='ratio', y='count', data=big_small_counts, ax=axes[1, 0])
        axes[1, 0].set_title('红球大小比分布', fontsize=14)
        axes[1, 0].set_xlabel('大小比', fontsize=12)
        axes[1, 0].set_ylabel('频次', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 蓝球奇偶分布
        self.data['blue_odd'] = self.data['blue_ball'] % 2 != 0
        blue_odd_counts = self.data.groupby('blue_odd').size().reset_index(name='count')
        blue_odd_counts['type'] = blue_odd_counts['blue_odd'].map({True: '奇数', False: '偶数'})
        
        sns.barplot(x='type', y='count', data=blue_odd_counts, ax=axes[1, 1])
        axes[1, 1].set_title('蓝球奇偶分布', fontsize=14)
        axes[1, 1].set_xlabel('类型', fontsize=12)
        axes[1, 1].set_ylabel('频次', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'number_combinations.png'), dpi=300)
        plt.close()
        
        print("号码组合特征图已保存")

    def analyze_trend(self):
        """分析号码走势"""
        # 按日期排序
        self.data = self.data.sort_values('date')
        
        # 取最近50期数据进行走势分析
        recent_data = self.data.tail(50).copy()
        recent_data['issue_index'] = range(len(recent_data))
        
        # 创建红球走势图
        plt.figure(figsize=(15, 10))
        
        # 绘制红球走势线
        for i in range(1, 7):
            plt.scatter(recent_data['issue_index'], recent_data[f'red_{i}'], 
                      color='red', alpha=0.7, s=30)
        
        # 添加水平参考线
        for i in range(1, 34):
            plt.axhline(y=i, color='gray', linestyle='-', alpha=0.2)
        
        plt.title('最近50期红球走势', fontsize=14)
        plt.xlabel('期号', fontsize=12)
        plt.ylabel('号码', fontsize=12)
        plt.yticks(range(1, 34))
        plt.grid(True, axis='both', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'red_ball_trend.png'), dpi=300)
        plt.close()
        
        # 创建蓝球走势图
        plt.figure(figsize=(15, 8))
        
        plt.scatter(recent_data['issue_index'], recent_data['blue_ball'], 
                  color='blue', alpha=0.7, s=50)
        
        # 添加水平参考线
        for i in range(1, 17):
            plt.axhline(y=i, color='gray', linestyle='-', alpha=0.2)
        
        plt.title('最近50期蓝球走势', fontsize=14)
        plt.xlabel('期号', fontsize=12)
        plt.ylabel('号码', fontsize=12)
        plt.yticks(range(1, 17))
        plt.grid(True, axis='both', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'blue_ball_trend.png'), dpi=300)
        plt.close()
        
        print("号码走势图已保存")

    def run_analysis(self):
        """运行所有分析"""
        if not self.load_data():
            return False
        
        print("开始分析数据...")
        self.plot_number_frequency()
        self.analyze_number_combinations()
        self.analyze_trend()
        print("分析完成！")
        
        return True


def main():
    """主函数"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据目录为上一级的data目录
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    data_file = os.path.join(data_dir, "ssq_data.csv")
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    # 创建分析器实例
    analyzer = SSQAnalyzer(data_file=data_file, output_dir=data_dir)
    
    # 运行分析
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
