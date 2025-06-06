#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球高级数据分析模块
使用数学统计、概率论、频率分析、决策树、周期和规律等方法分析双色球开奖结果
并使用PyMC进行贝叶斯分析
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from collections import Counter
from datetime import datetime, timedelta
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

# 尝试导入PyMC (PyMC3已更名为PyMC)
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    print("警告: PyMC未安装，贝叶斯分析功能将不可用")
    print("可以使用 'pip install pymc arviz' 安装所需依赖")
    PYMC_AVAILABLE = False

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体
try:
    # 尝试使用系统中文字体
    font_path = '/System/Library/Fonts/Hiragino Sans GB.ttc'  # macOS中文字体
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


class SSQAdvancedAnalyzer:
    """双色球高级数据分析类"""

    def __init__(self, data_file="../data/ssq_data.csv", output_dir="../data/advanced"):
        """
        初始化高级分析器

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
        """
        加载数据并进行预处理

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
            
            # 按日期排序
            self.data = self.data.sort_values('date')
            
            # 添加期号索引
            self.data['issue_index'] = range(len(self.data))
            
            # 计算红球和值
            self.data['red_sum'] = sum(self.data[f'red_{i+1}'] for i in range(6))
            
            # 计算红球方差
            self.data['red_variance'] = self.data.apply(
                lambda x: np.var([x[f'red_{i+1}'] for i in range(1, 7)]), axis=1
            )
            
            # 计算红球跨度（最大值-最小值）
            self.data['red_span'] = self.data.apply(
                lambda x: max([x[f'red_{i+1}'] for i in range(1, 7)]) - 
                         min([x[f'red_{i+1}'] for i in range(1, 7)]),
                axis=1
            )
            
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
            
            # 计算蓝球奇偶
            self.data['blue_odd'] = self.data['blue_ball'] % 2 != 0
            
            # 计算红球区间分布
            self.data['red_zone_1'] = self.data.apply(
                lambda x: sum(1 for i in range(1, 7) if 1 <= x[f'red_{i}'] <= 11), axis=1
            )
            self.data['red_zone_2'] = self.data.apply(
                lambda x: sum(1 for i in range(1, 7) if 12 <= x[f'red_{i}'] <= 22), axis=1
            )
            self.data['red_zone_3'] = self.data.apply(
                lambda x: sum(1 for i in range(1, 7) if 23 <= x[f'red_{i}'] <= 33), axis=1
            )
            
            # 计算连号个数
            self.data['consecutive_count'] = self.data.apply(self._count_consecutive_numbers, axis=1)
            
            print(f"成功加载并预处理{len(self.data)}条数据")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _count_consecutive_numbers(self, row):
        """计算一组红球中连号的个数"""
        red_balls = sorted([row[f'red_{i}'] for i in range(1, 7)])
        consecutive_count = 0
        for i in range(len(red_balls) - 1):
            if red_balls[i+1] - red_balls[i] == 1:
                consecutive_count += 1
        return consecutive_count

    def analyze_statistical_features(self):
        """
        分析统计学特征
        包括红球和值、方差、跨度的分布及其统计特性
        """
        print("分析统计学特征...")
        
        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # 红球和值分布
        sns.histplot(self.data['red_sum'], bins=30, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('红球和值分布', fontsize=14)
        axes[0, 0].set_xlabel('和值', fontsize=12)
        axes[0, 0].set_ylabel('频次', fontsize=12)
        
        # 红球和值的时间序列
        axes[0, 1].plot(self.data['issue_index'], self.data['red_sum'], marker='o', linestyle='-', alpha=0.7)
        axes[0, 1].set_title('红球和值时间序列', fontsize=14)
        axes[0, 1].set_xlabel('期号', fontsize=12)
        axes[0, 1].set_ylabel('和值', fontsize=12)
        
        # 红球方差分布
        sns.histplot(self.data['red_variance'], bins=30, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('红球方差分布', fontsize=14)
        axes[1, 0].set_xlabel('方差', fontsize=12)
        axes[1, 0].set_ylabel('频次', fontsize=12)
        
        # 红球方差的时间序列
        axes[1, 1].plot(self.data['issue_index'], self.data['red_variance'], marker='o', linestyle='-', alpha=0.7)
        axes[1, 1].set_title('红球方差时间序列', fontsize=14)
        axes[1, 1].set_xlabel('期号', fontsize=12)
        axes[1, 1].set_ylabel('方差', fontsize=12)
        
        # 红球跨度分布
        sns.histplot(self.data['red_span'], bins=30, kde=True, ax=axes[2, 0])
        axes[2, 0].set_title('红球跨度分布', fontsize=14)
        axes[2, 0].set_xlabel('跨度', fontsize=12)
        axes[2, 0].set_ylabel('频次', fontsize=12)
        
        # 红球跨度的时间序列
        axes[2, 1].plot(self.data['issue_index'], self.data['red_span'], marker='o', linestyle='-', alpha=0.7)
        axes[2, 1].set_title('红球跨度时间序列', fontsize=14)
        axes[2, 1].set_xlabel('期号', fontsize=12)
        axes[2, 1].set_ylabel('跨度', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'statistical_features.png'), dpi=300)
        plt.close()
        
        # 计算统计特性
        stats_data = {
            '特征': ['红球和值', '红球方差', '红球跨度'],
            '最小值': [
                self.data['red_sum'].min(),
                self.data['red_variance'].min(),
                self.data['red_span'].min()
            ],
            '最大值': [
                self.data['red_sum'].max(),
                self.data['red_variance'].max(),
                self.data['red_span'].max()
            ],
            '平均值': [
                self.data['red_sum'].mean(),
                self.data['red_variance'].mean(),
                self.data['red_span'].mean()
            ],
            '中位数': [
                self.data['red_sum'].median(),
                self.data['red_variance'].median(),
                self.data['red_span'].median()
            ],
            '标准差': [
                self.data['red_sum'].std(),
                self.data['red_variance'].std(),
                self.data['red_span'].std()
            ],
            '众数': [
                self.data['red_sum'].mode().iloc[0],
                self.data['red_variance'].mode().iloc[0],
                self.data['red_span'].mode().iloc[0]
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(os.path.join(self.output_dir, 'statistical_features.csv'), index=False)
        
        print("统计学特征分析完成")
        return stats_df

    def analyze_probability_distribution(self):
        """
        分析概率分布
        使用概率论方法分析红球和蓝球的概率分布
        """
        print("分析概率分布...")
        
        # 计算红球频率
        red_counts = {}
        for i in range(1, 34):
            red_counts[i] = 0
        
        for i in range(1, 7):
            for _, row in self.data.iterrows():
                red_counts[row[f'red_{i}']] += 1
        
        total_red_draws = len(self.data) * 6  # 总红球数量
        red_probs = {num: count/total_red_draws for num, count in red_counts.items()}
        
        # 计算蓝球频率
        blue_counts = Counter(self.data['blue_ball'])
        total_blue_draws = len(self.data)  # 总蓝球数量
        blue_probs = {num: count/total_blue_draws for num, count in blue_counts.items()}
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 红球概率分布
        red_nums = list(self.red_range)
        red_probabilities = [red_probs.get(num, 0) for num in red_nums]
        
        # 理论均匀分布概率
        uniform_prob = 1/33
        
        bars = axes[0, 0].bar(red_nums, red_probabilities, color='red', alpha=0.7)
        axes[0, 0].axhline(y=uniform_prob, color='black', linestyle='--', alpha=0.7, label='理论均匀分布')
        axes[0, 0].set_title('红球概率分布', fontsize=14)
        axes[0, 0].set_xlabel('红球号码', fontsize=12)
        axes[0, 0].set_ylabel('概率', fontsize=12)
        axes[0, 0].set_xticks(red_nums)
        axes[0, 0].set_xticklabels([str(num) for num in red_nums], rotation=90, fontsize=8)
        axes[0, 0].legend()
        
        # 红球概率偏差
        red_deviation = [prob - uniform_prob for prob in red_probabilities]
        bars = axes[0, 1].bar(red_nums, red_deviation, color=['green' if x >= 0 else 'red' for x in red_deviation], alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.7)
        axes[0, 1].set_title('红球概率偏差 (实际概率 - 理论概率)', fontsize=14)
        axes[0, 1].set_xlabel('红球号码', fontsize=12)
        axes[0, 1].set_ylabel('概率偏差', fontsize=12)
        axes[0, 1].set_xticks(red_nums)
        axes[0, 1].set_xticklabels([str(num) for num in red_nums], rotation=90, fontsize=8)
        
        # 蓝球概率分布
        blue_nums = list(self.blue_range)
        blue_probabilities = [blue_probs.get(num, 0) for num in blue_nums]
        
        # 理论均匀分布概率
        uniform_blue_prob = 1/16
        
        bars = axes[1, 0].bar(blue_nums, blue_probabilities, color='blue', alpha=0.7)
        axes[1, 0].axhline(y=uniform_blue_prob, color='black', linestyle='--', alpha=0.7, label='理论均匀分布')
        axes[1, 0].set_title('蓝球概率分布', fontsize=14)
        axes[1, 0].set_xlabel('蓝球号码', fontsize=12)
        axes[1, 0].set_ylabel('概率', fontsize=12)
        axes[1, 0].set_xticks(blue_nums)
        axes[1, 0].set_xticklabels([str(num) for num in blue_nums], fontsize=10)
        axes[1, 0].legend()
        
        # 蓝球概率偏差
        blue_deviation = [prob - uniform_blue_prob for prob in blue_probabilities]
        bars = axes[1, 1].bar(blue_nums, blue_deviation, color=['green' if x >= 0 else 'red' for x in blue_deviation], alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.7)
        axes[1, 1].set_title('蓝球概率偏差 (实际概率 - 理论概率)', fontsize=14)
        axes[1, 1].set_xlabel('蓝球号码', fontsize=12)
        axes[1, 1].set_ylabel('概率偏差', fontsize=12)
        axes[1, 1].set_xticks(blue_nums)
        axes[1, 1].set_xticklabels([str(num) for num in blue_nums], fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'probability_distribution.png'), dpi=300)
        plt.close()
        
        # 保存概率数据
        prob_data = {
            '红球号码': red_nums,
            '红球概率': red_probabilities,
            '红球偏差': red_deviation,
            '蓝球号码': blue_nums + [None] * (len(red_nums) - len(blue_nums)),
            '蓝球概率': blue_probabilities + [None] * (len(red_nums) - len(blue_nums)),
            '蓝球偏差': blue_deviation + [None] * (len(red_nums) - len(blue_nums))
        }
        
        prob_df = pd.DataFrame(prob_data)
        prob_df.to_csv(os.path.join(self.output_dir, 'probability_distribution.csv'), index=False)
        
        print("概率分布分析完成")
        return red_probs, blue_probs

    def analyze_frequency_patterns(self):
        """
        分析频率模式
        分析红球和蓝球的出现频率模式，包括热门号码和冷门号码
        """
        print("分析频率模式...")
        
        # 计算每个号码的冷热周期
        red_last_appear = {num: -100 for num in range(1, 34)}  # 初始化为一个很小的值
        red_cold_periods = {num: [] for num in range(1, 34)}
        
        blue_last_appear = {num: -100 for num in range(1, 17)}
        blue_cold_periods = {num: [] for num in range(1, 17)}
        
        # 遍历数据计算冷热周期
        for idx, row in self.data.iterrows():
            issue_idx = row['issue_index']
            
            # 红球
            current_reds = [row[f'red_{i}'] for i in range(1, 7)]
            for num in range(1, 34):
                if num in current_reds:
                    if red_last_appear[num] != -100:  # 不是第一次出现
                        cold_period = issue_idx - red_last_appear[num]
                        red_cold_periods[num].append(cold_period)
                    red_last_appear[num] = issue_idx
            
            # 蓝球
            blue = row['blue_ball']
            if blue_last_appear[blue] != -100:  # 不是第一次出现
                cold_period = issue_idx - blue_last_appear[blue]
                blue_cold_periods[blue].append(cold_period)
            blue_last_appear[blue] = issue_idx
        
        # 计算平均冷却周期
        red_avg_cold_period = {num: np.mean(periods) if periods else float('inf') 
                              for num, periods in red_cold_periods.items()}
        blue_avg_cold_period = {num: np.mean(periods) if periods else float('inf') 
                               for num, periods in blue_cold_periods.items()}
        
        # 计算当前冷却期
        last_issue_idx = self.data['issue_index'].max()
        red_current_cold = {num: last_issue_idx - red_last_appear[num] for num in range(1, 34)}
        blue_current_cold = {num: last_issue_idx - blue_last_appear[num] for num in range(1, 17)}
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 红球平均冷却周期
        red_nums = list(range(1, 34))
        red_cold_values = [red_avg_cold_period[num] for num in red_nums]
        
        bars = axes[0, 0].bar(red_nums, red_cold_values, color='red', alpha=0.7)
        axes[0, 0].set_title('红球平均冷却周期', fontsize=14)
        axes[0, 0].set_xlabel('红球号码', fontsize=12)
        axes[0, 0].set_ylabel('平均期数', fontsize=12)
        axes[0, 0].set_xticks(red_nums)
        axes[0, 0].set_xticklabels([str(num) for num in red_nums], rotation=90, fontsize=8)
        
        # 红球当前冷却期
        red_current_values = [red_current_cold[num] for num in red_nums]
        
        bars = axes[0, 1].bar(red_nums, red_current_values, 
                            color=['green' if red_current_cold[num] > red_avg_cold_period[num] else 'red' 
                                  for num in red_nums], alpha=0.7)
        axes[0, 1].set_title('红球当前冷却期', fontsize=14)
        axes[0, 1].set_xlabel('红球号码', fontsize=12)
        axes[0, 1].set_ylabel('期数', fontsize=12)
        axes[0, 1].set_xticks(red_nums)
        axes[0, 1].set_xticklabels([str(num) for num in red_nums], rotation=90, fontsize=8)
        
        # 蓝球平均冷却周期
        blue_nums = list(range(1, 17))
        blue_cold_values = [blue_avg_cold_period[num] for num in blue_nums]
        
        bars = axes[1, 0].bar(blue_nums, blue_cold_values, color='blue', alpha=0.7)
        axes[1, 0].set_title('蓝球平均冷却周期', fontsize=14)
        axes[1, 0].set_xlabel('蓝球号码', fontsize=12)
        axes[1, 0].set_ylabel('平均期数', fontsize=12)
        axes[1, 0].set_xticks(blue_nums)
        axes[1, 0].set_xticklabels([str(num) for num in blue_nums], fontsize=10)
        
        # 蓝球当前冷却期
        blue_current_values = [blue_current_cold[num] for num in blue_nums]
        
        bars = axes[1, 1].bar(blue_nums, blue_current_values, 
                            color=['green' if blue_current_cold[num] > blue_avg_cold_period[num] else 'blue' 
                                  for num in blue_nums], alpha=0.7)
        axes[1, 1].set_title('蓝球当前冷却期', fontsize=14)
        axes[1, 1].set_xlabel('蓝球号码', fontsize=12)
        axes[1, 1].set_ylabel('期数', fontsize=12)
        axes[1, 1].set_xticks(blue_nums)
        axes[1, 1].set_xticklabels([str(num) for num in blue_nums], fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'frequency_patterns.png'), dpi=300)
        plt.close()
        
        # 保存频率数据
        freq_data = {
            '红球号码': red_nums,
            '红球平均冷却期': [red_avg_cold_period[num] for num in red_nums],
            '红球当前冷却期': [red_current_cold[num] for num in red_nums],
            '红球冷热状态': ['热' if red_current_cold[num] <= red_avg_cold_period[num] else '冷' for num in red_nums],
            '蓝球号码': blue_nums + [None] * (len(red_nums) - len(blue_nums)),
            '蓝球平均冷却期': [blue_avg_cold_period[num] for num in blue_nums] + [None] * (len(red_nums) - len(blue_nums)),
            '蓝球当前冷却期': [blue_current_cold[num] for num in blue_nums] + [None] * (len(red_nums) - len(blue_nums)),
            '蓝球冷热状态': ['热' if blue_current_cold[num] <= blue_avg_cold_period[num] else '冷' for num in blue_nums] + [None] * (len(red_nums) - len(blue_nums))
        }
        
        freq_df = pd.DataFrame(freq_data)
        freq_df.to_csv(os.path.join(self.output_dir, 'frequency_patterns.csv'), index=False)
        
        print("频率模式分析完成")
        return red_avg_cold_period, blue_avg_cold_period, red_current_cold, blue_current_cold

    def analyze_decision_tree(self):
        """
        使用决策树分析红球和蓝球的规律
        """
        print("使用决策树分析...")
        
        # 准备特征和目标变量
        # 我们使用前n期的数据预测下一期的蓝球
        n_prev = 5  # 使用前5期数据
        
        features = []
        blue_targets = []
        
        for i in range(n_prev, len(self.data)):
            # 特征：前n期的红球和值、方差、跨度、蓝球
            feature_row = []
            for j in range(n_prev):
                idx = i - n_prev + j
                feature_row.extend([
                    self.data.iloc[idx]['red_sum'],
                    self.data.iloc[idx]['red_variance'],
                    self.data.iloc[idx]['red_span'],
                    self.data.iloc[idx]['blue_ball']
                ])
            features.append(feature_row)
            
            # 目标：当前期的蓝球
            blue_targets.append(self.data.iloc[i]['blue_ball'])
        
        # 转换为numpy数组
        X = np.array(features)
        y_blue = np.array(blue_targets)
        
        # 划分训练集和测试集
        X_train, X_test, y_blue_train, y_blue_test = train_test_split(
            X, y_blue, test_size=0.3, random_state=42
        )
        
        # 训练决策树模型预测蓝球
        blue_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        blue_tree.fit(X_train, y_blue_train)
        
        # 预测并评估
        y_blue_pred = blue_tree.predict(X_test)
        blue_accuracy = accuracy_score(y_blue_test, y_blue_pred)
        
        # 使用随机森林提高准确率
        blue_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        blue_rf.fit(X_train, y_blue_train)
        
        y_blue_rf_pred = blue_rf.predict(X_test)
        blue_rf_accuracy = accuracy_score(y_blue_test, y_blue_rf_pred)
        
        # 创建特征重要性图表
        plt.figure(figsize=(12, 6))
        
        # 获取特征名称
        feature_names = []
        for j in range(n_prev):
            feature_names.extend([
                f'红球和值(t-{n_prev-j})',
                f'红球方差(t-{n_prev-j})',
                f'红球跨度(t-{n_prev-j})',
                f'蓝球(t-{n_prev-j})'
            ])
        
        # 绘制随机森林特征重要性
        importances = blue_rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(X.shape[1]), importances[indices], color='blue', alpha=0.7)
        plt.title('蓝球预测的特征重要性 (随机森林)', fontsize=14)
        plt.xlabel('特征', fontsize=12)
        plt.ylabel('重要性', fontsize=12)
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'decision_tree_feature_importance.png'), dpi=300)
        plt.close()
        
        # 保存模型评估结果
        model_results = {
            '模型': ['决策树 (蓝球)', '随机森林 (蓝球)'],
            '准确率': [blue_accuracy, blue_rf_accuracy],
            '样本数量': [len(y_blue_test), len(y_blue_test)]
        }
        
        model_df = pd.DataFrame(model_results)
        model_df.to_csv(os.path.join(self.output_dir, 'decision_tree_results.csv'), index=False)
        
        # 保存特征重要性
        importance_data = {
            '特征': [feature_names[i] for i in indices],
            '重要性': importances[indices]
        }
        
        importance_df = pd.DataFrame(importance_data)
        importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        
        print(f"决策树分析完成，蓝球预测准确率: {blue_accuracy:.4f}, 随机森林准确率: {blue_rf_accuracy:.4f}")
        return blue_rf, blue_rf_accuracy, importance_df

    def analyze_cycle_patterns(self):
        """
        分析周期和规律
        使用时间序列分析方法寻找潜在的周期性模式
        """
        print("分析周期和规律...")
        
        # 分析红球和值的周期性
        red_sum_series = self.data['red_sum'].values
        
        # 计算自相关系数
        max_lag = min(50, len(red_sum_series) // 2)
        acf_values = [1.0]  # 自相关系数，lag=0时为1
        for lag in range(1, max_lag + 1):
            acf = np.corrcoef(red_sum_series[:-lag], red_sum_series[lag:])[0, 1]
            acf_values.append(acf)
        
        # 创建自相关图
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(acf_values)), acf_values, color='blue', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axhline(y=1.96/np.sqrt(len(red_sum_series)), color='red', linestyle='--', alpha=0.7, label='95% 置信区间')
        plt.axhline(y=-1.96/np.sqrt(len(red_sum_series)), color='red', linestyle='--', alpha=0.7)
        plt.title('红球和值的自相关函数 (ACF)', fontsize=14)
        plt.xlabel('滞后期数', fontsize=12)
        plt.ylabel('自相关系数', fontsize=12)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'red_sum_acf.png'), dpi=300)
        plt.close()
        
        # 分析蓝球的周期性
        # 创建蓝球号码的时间序列
        blue_series = self.data['blue_ball'].values
        
        # 计算自相关系数
        blue_acf_values = [1.0]  # 自相关系数，lag=0时为1
        for lag in range(1, max_lag + 1):
            acf = np.corrcoef(blue_series[:-lag], blue_series[lag:])[0, 1]
            blue_acf_values.append(acf)
        
        # 创建自相关图
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(blue_acf_values)), blue_acf_values, color='blue', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axhline(y=1.96/np.sqrt(len(blue_series)), color='red', linestyle='--', alpha=0.7, label='95% 置信区间')
        plt.axhline(y=-1.96/np.sqrt(len(blue_series)), color='red', linestyle='--', alpha=0.7)
        plt.title('蓝球号码的自相关函数 (ACF)', fontsize=14)
        plt.xlabel('滞后期数', fontsize=12)
        plt.ylabel('自相关系数', fontsize=12)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'blue_ball_acf.png'), dpi=300)
        plt.close()
        
        # 保存自相关系数数据
        acf_data = {
            '滞后期数': list(range(len(acf_values))),
            '红球和值自相关系数': acf_values,
            '蓝球自相关系数': blue_acf_values
        }
        
        acf_df = pd.DataFrame(acf_data)
        acf_df.to_csv(os.path.join(self.output_dir, 'autocorrelation.csv'), index=False)
        
        # 寻找显著的周期
        significant_threshold = 1.96/np.sqrt(len(red_sum_series))
        red_significant_lags = [lag for lag, acf in enumerate(acf_values) if abs(acf) > significant_threshold and lag > 0]
        blue_significant_lags = [lag for lag, acf in enumerate(blue_acf_values) if abs(acf) > significant_threshold and lag > 0]
        
        # 保存显著周期数据
        cycle_data = {
            '红球和值显著周期': red_significant_lags,
            '红球和值周期自相关系数': [acf_values[lag] for lag in red_significant_lags] if red_significant_lags else [],
            '蓝球显著周期': blue_significant_lags,
            '蓝球周期自相关系数': [blue_acf_values[lag] for lag in blue_significant_lags] if blue_significant_lags else []
        }
        
        # 转换为DataFrame并保存
        max_len = max(len(red_significant_lags), len(blue_significant_lags))
        for key in cycle_data:
            if len(cycle_data[key]) < max_len:
                cycle_data[key] = cycle_data[key] + [None] * (max_len - len(cycle_data[key]))
        
        cycle_df = pd.DataFrame(cycle_data)
        cycle_df.to_csv(os.path.join(self.output_dir, 'significant_cycles.csv'), index=False)
        
        print("周期和规律分析完成")
        return acf_df, cycle_df

    def analyze_bayesian(self):
        """
        使用PyMC进行贝叶斯分析
        分析红球和蓝球的概率分布
        """
        if not PYMC_AVAILABLE:
            print("PyMC未安装，跳过贝叶斯分析")
            return None
        
        print("使用PyMC进行贝叶斯分析...")
        
        # 分析蓝球的贝叶斯概率
        blue_counts = Counter(self.data['blue_ball'])
        observed_blues = np.array([blue_counts.get(i, 0) for i in range(1, 17)])
        
        # 创建贝叶斯模型
        with pm.Model() as blue_model:
            # 先验分布 - 假设所有蓝球概率相等
            alpha = np.ones(16)  # 均匀先验
            p = pm.Dirichlet('p', a=alpha)
            
            # 似然函数 - 多项分布
            observed = pm.Multinomial('observed', n=sum(observed_blues), p=p, observed=observed_blues)
            
            # 采样
            trace = pm.sample(2000, tune=1000, return_inferencedata=True)
        
        # 提取后验分布
        posterior_samples = az.extract(trace, var_names=['p'])
        blue_posterior_means = posterior_samples.mean(axis=0).values
        blue_posterior_hdi = az.hdi(trace, var_names=['p']).values
        
        # 创建贝叶斯分析图表
        plt.figure(figsize=(12, 6))
        
        # 绘制后验均值和95%HDI
        blue_nums = np.arange(1, 17)
        plt.bar(blue_nums, blue_posterior_means, color='blue', alpha=0.7)
        
        # 添加95% HDI误差棒
        for i, num in enumerate(blue_nums):
            plt.plot([num, num], [blue_posterior_hdi[0][i], blue_posterior_hdi[1][i]], color='black')
        
        # 添加均匀分布参考线
        plt.axhline(y=1/16, color='red', linestyle='--', alpha=0.7, label='均匀分布概率')
        
        plt.title('蓝球号码的贝叶斯后验概率分布', fontsize=14)
        plt.xlabel('蓝球号码', fontsize=12)
        plt.ylabel('概率', fontsize=12)
        plt.xticks(blue_nums)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'blue_bayesian_analysis.png'), dpi=300)
        plt.close()
        
        # 保存贝叶斯分析结果
        bayes_data = {
            '蓝球号码': blue_nums,
            '后验均值': blue_posterior_means,
            '后验95%HDI下限': blue_posterior_hdi[0],
            '后验95%HDI上限': blue_posterior_hdi[1],
            '均匀分布概率': [1/16] * 16
        }
        
        bayes_df = pd.DataFrame(bayes_data)
        bayes_df.to_csv(os.path.join(self.output_dir, 'bayesian_analysis.csv'), index=False)
        
        print("贝叶斯分析完成")
        return bayes_df

    def predict_numbers(self, method='ensemble', explain=False):
        """
        使用高级分析方法预测双色球号码
        
        Args:
            method: 预测方法，可选值为'stats'（统计学）, 'probability'（概率论）, 
                   'decision_tree'（决策树）, 'bayes'（贝叶斯）, 'ensemble'（集成方法）
            explain: 是否解释预测结果
            
        Returns:
            包含7个数字的列表，前6个为红球，最后1个为蓝球
        """
        # 确保数据已加载
        if self.data is None:
            self.load_data()
            
        print(f"使用{method}方法预测双色球号码...")
        
        # 根据不同方法预测号码
        if method == 'stats':
            red_balls, blue_ball = self._predict_by_stats(explain)
        elif method == 'probability':
            red_balls, blue_ball = self._predict_by_probability(explain)
        elif method == 'decision_tree':
            red_balls, blue_ball = self._predict_by_decision_tree(explain)
        elif method == 'bayes':
            red_balls, blue_ball = self._predict_by_bayes(explain)
        else:  # ensemble
            red_balls, blue_ball = self._predict_by_ensemble(explain)
        
        # 返回完整号码（红球+蓝球）
        return red_balls + [blue_ball]
        
    def _predict_by_stats(self, explain=False):
        """
        基于统计学特征预测双色球号码
        使用红球和值、方差、跨度等统计特性
        
        Args:
            explain: 是否解释预测结果
            
        Returns:
            (红球列表, 蓝球)
        """
        import random
        
        # 获取红球的平均和值、方差和跨度
        if not hasattr(self, '_stats_df'):
            self._stats_df = self.analyze_statistical_features()
        
        avg_red_sum = self._stats_df.loc[self._stats_df['特征'] == '红球和值', '平均值'].values[0]
        avg_red_variance = self._stats_df.loc[self._stats_df['特征'] == '红球方差', '平均值'].values[0]
        avg_red_span = self._stats_df.loc[self._stats_df['特征'] == '红球跨度', '平均值'].values[0]
        
        # 生成候选红球组合
        max_attempts = 100
        best_combination = None
        min_diff = float('inf')
        
        for _ in range(max_attempts):
            # 随机生成6个不重复的红球
            red_balls = sorted(random.sample(range(1, 34), 6))
            
            # 计算统计特性
            red_sum = sum(red_balls)
            red_variance = np.var(red_balls)
            red_span = max(red_balls) - min(red_balls)
            
            # 计算与平均值的差异
            sum_diff = abs(red_sum - avg_red_sum) / avg_red_sum
            var_diff = abs(red_variance - avg_red_variance) / avg_red_variance
            span_diff = abs(red_span - avg_red_span) / avg_red_span
            
            # 综合差异（加权平均）
            total_diff = sum_diff * 0.4 + var_diff * 0.3 + span_diff * 0.3
            
            if total_diff < min_diff:
                min_diff = total_diff
                best_combination = red_balls
        
        # 选择蓝球：分析历史数据中的蓝球分布
        blue_counts = Counter(self.data['blue_ball'])
        total_draws = len(self.data)
        
        # 计算蓝球概率
        blue_probs = {num: count/total_draws for num, count in blue_counts.items()}
        
        # 根据概率选择蓝球（轮盘赌选择法）
        blue_nums = list(range(1, 17))
        blue_weights = [blue_probs.get(num, 1/16) for num in blue_nums]  # 默认概率为1/16
        blue_ball = random.choices(blue_nums, weights=blue_weights, k=1)[0]
        
        if explain:
            print("\n基于统计学特征的预测解释:")
            print(f"1. 选择的红球组合统计特性与历史平均值的差异为{min_diff:.4f}")
            print(f"2. 红球和值: {sum(best_combination)} (历史平均: {avg_red_sum:.2f})")
            print(f"3. 红球方差: {np.var(best_combination):.2f} (历史平均: {avg_red_variance:.2f})")
            print(f"4. 红球跨度: {max(best_combination) - min(best_combination)} (历史平均: {avg_red_span:.2f})")
            print(f"5. 蓝球{blue_ball}的历史概率为{blue_probs.get(blue_ball, 1/16):.4f}")
        
        return best_combination, blue_ball
    
    def _predict_by_probability(self, explain=False):
        """
        基于概率论预测双色球号码
        使用红球和蓝球的历史概率分布
        
        Args:
            explain: 是否解释预测结果
            
        Returns:
            (红球列表, 蓝球)
        """
        import random
        
        # 获取红球和蓝球的概率分布
        if not hasattr(self, '_red_probs') or not hasattr(self, '_blue_probs'):
            self._red_probs, self._blue_probs = self.analyze_probability_distribution()
        
        # 计算红球的选择概率
        red_nums = list(range(1, 34))
        red_weights = [self._red_probs.get(num, 1/33) * 33 for num in red_nums]  # 乘以33使权重更明显
        
        # 使用加权随机抽样选择红球
        selected_reds = []
        remaining_nums = red_nums.copy()
        remaining_weights = red_weights.copy()
        
        for _ in range(6):
            # 归一化权重
            total_weight = sum(remaining_weights)
            norm_weights = [w/total_weight for w in remaining_weights]
            
            # 选择一个红球
            idx = random.choices(range(len(remaining_nums)), weights=norm_weights, k=1)[0]
            selected_reds.append(remaining_nums[idx])
            
            # 移除已选择的红球
            remaining_nums.pop(idx)
            remaining_weights.pop(idx)
        
        selected_reds.sort()
        
        # 选择蓝球
        blue_nums = list(range(1, 17))
        blue_weights = [self._blue_probs.get(num, 1/16) * 16 for num in blue_nums]  # 乘以16使权重更明显
        blue_ball = random.choices(blue_nums, weights=blue_weights, k=1)[0]
        
        if explain:
            print("\n基于概率论的预测解释:")
            print("1. 红球选择基于历史概率分布，概率越高的号码被选中的可能性越大")
            for num in selected_reds:
                print(f"   红球{num:02d}的历史概率为{self._red_probs.get(num, 1/33):.4f}")
            print(f"2. 蓝球{blue_ball}的历史概率为{self._blue_probs.get(blue_ball, 1/16):.4f}")
        
        return selected_reds, blue_ball
    
    def _predict_by_decision_tree(self, explain=False):
        """
        基于决策树预测双色球号码
        使用机器学习模型预测
        
        Args:
            explain: 是否解释预测结果
            
        Returns:
            (红球列表, 蓝球)
        """
        import random
        
        # 如果数据量不足，使用概率方法
        if len(self.data) < 10:
            if explain:
                print("\n数据量不足，无法使用决策树模型，将使用概率方法代替")
            return self._predict_by_probability(explain)
        
        # 训练决策树模型
        if not hasattr(self, '_blue_rf'):
            self._blue_rf, self._blue_rf_accuracy, self._importance_df = self.analyze_decision_tree()
        
        # 准备特征数据（最近5期的数据）
        n_prev = 5
        feature_row = []
        for j in range(min(n_prev, len(self.data))):
            idx = len(self.data) - min(n_prev, len(self.data)) + j
            feature_row.extend([
                self.data.iloc[idx]['red_sum'],
                self.data.iloc[idx]['red_variance'],
                self.data.iloc[idx]['red_span'],
                self.data.iloc[idx]['blue_ball']
            ])
        
        # 如果数据不足5期，用0填充
        if len(self.data) < n_prev:
            padding = (n_prev - len(self.data)) * 4  # 每期4个特征
            feature_row = [0] * padding + feature_row
        
        # 预测蓝球概率
        blue_probs = self._blue_rf.predict_proba([feature_row])[0]
        
        # 根据预测概率选择蓝球
        blue_candidates = [(i+1, prob) for i, prob in enumerate(blue_probs)]
        blue_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 从概率最高的5个中随机选择1个
        top_blues = [num for num, prob in blue_candidates[:5]]
        top_probs = [prob for _, prob in blue_candidates[:5]]
        blue_ball = random.choices(top_blues, weights=top_probs, k=1)[0]
        
        # 对于红球，我们使用频率分析
        if not hasattr(self, '_red_current_cold'):
            _, _, self._red_current_cold, _ = self.analyze_frequency_patterns()
        
        # 根据冷热状态选择红球
        red_candidates = [(num, self._red_current_cold[num]) for num in range(1, 34)]
        red_candidates.sort(key=lambda x: x[1], reverse=True)  # 按冷却期排序
        
        # 从冷球中选择4个，从热球中选择2个
        cold_reds = [num for num, _ in red_candidates[:20]]  # 前20个为冷球
        hot_reds = [num for num, _ in red_candidates[20:]]   # 后13个为热球
        
        selected_reds = random.sample(cold_reds, 4) + random.sample(hot_reds, 2)
        selected_reds.sort()
        
        if explain:
            print("\n基于决策树的预测解释:")
            print(f"1. 决策树模型的蓝球预测准确率为{self._blue_rf_accuracy:.4f}")
            print(f"2. 蓝球{blue_ball}的预测概率为{blue_candidates[top_blues.index(blue_ball)][1]:.4f}")
            print("3. 红球选择基于冷热分析:")
            for num in selected_reds:
                status = "冷" if num in cold_reds else "热"
                print(f"   红球{num:02d}当前冷却期为{self._red_current_cold[num]}期 ({status})")
            
            # 显示特征重要性
            print("4. 蓝球预测的主要影响因素:")
            for i in range(min(3, len(self._importance_df))):
                feature = self._importance_df.iloc[i]['特征']
                importance = self._importance_df.iloc[i]['重要性']
                print(f"   {feature}: {importance:.4f}")
        
        return selected_reds, blue_ball
    
    def _predict_by_bayes(self, explain=False):
        """
        基于贝叶斯分析预测双色球号码
        使用贝叶斯后验概率
        
        Args:
            explain: 是否解释预测结果
            
        Returns:
            (红球列表, 蓝球)
        """
        import random
        
        # 如果PyMC不可用，使用概率方法
        if not PYMC_AVAILABLE:
            if explain:
                print("\nPyMC未安装，无法使用贝叶斯分析，将使用概率方法代替")
            return self._predict_by_probability(explain)
        
        # 进行贝叶斯分析
        if not hasattr(self, '_bayes_df'):
            self._bayes_df = self.analyze_bayesian()
        
        # 使用贝叶斯后验概率选择蓝球
        blue_nums = self._bayes_df['蓝球号码'].values
        blue_probs = self._bayes_df['后验均值'].values
        blue_ball = random.choices(blue_nums, weights=blue_probs, k=1)[0]
        
        # 对于红球，我们使用统计特性和概率相结合的方法
        # 获取红球的平均和值、方差和跨度
        if not hasattr(self, '_stats_df'):
            self._stats_df = self.analyze_statistical_features()
        
        avg_red_sum = self._stats_df.loc[self._stats_df['特征'] == '红球和值', '平均值'].values[0]
        
        # 获取红球概率分布
        if not hasattr(self, '_red_probs'):
            self._red_probs, _ = self.analyze_probability_distribution()
        
        # 生成候选红球组合
        max_attempts = 100
        best_combination = None
        max_score = -float('inf')
        
        for _ in range(max_attempts):
            # 使用加权随机抽样选择红球
            red_nums = list(range(1, 34))
            red_weights = [self._red_probs.get(num, 1/33) * 33 for num in red_nums]
            
            selected_reds = []
            remaining_nums = red_nums.copy()
            remaining_weights = red_weights.copy()
            
            for _ in range(6):
                # 归一化权重
                total_weight = sum(remaining_weights)
                norm_weights = [w/total_weight for w in remaining_weights]
                
                # 选择一个红球
                idx = random.choices(range(len(remaining_nums)), weights=norm_weights, k=1)[0]
                selected_reds.append(remaining_nums[idx])
                
                # 移除已选择的红球
                remaining_nums.pop(idx)
                remaining_weights.pop(idx)
            
            selected_reds.sort()
            
            # 计算和值与平均值的接近程度
            red_sum = sum(selected_reds)
            sum_score = 1 - abs(red_sum - avg_red_sum) / avg_red_sum
            
            # 计算概率得分
            prob_score = sum(self._red_probs.get(num, 1/33) for num in selected_reds) / 6
            
            # 综合得分
            total_score = sum_score * 0.5 + prob_score * 0.5
            
            if total_score > max_score:
                max_score = total_score
                best_combination = selected_reds
        
        if explain:
            print("\n基于贝叶斯分析的预测解释:")
            print(f"1. 蓝球{blue_ball}的贝叶斯后验概率为{blue_probs[list(blue_nums).index(blue_ball)]:.4f}")
            print(f"2. 蓝球{blue_ball}的95%可信区间为[{self._bayes_df.loc[self._bayes_df['蓝球号码'] == blue_ball, '后验95%HDI下限'].values[0]:.4f}, {self._bayes_df.loc[self._bayes_df['蓝球号码'] == blue_ball, '后验95%HDI上限'].values[0]:.4f}]")
            print("3. 红球选择基于概率分布和统计特性:")
            print(f"   红球和值: {sum(best_combination)} (历史平均: {avg_red_sum:.2f})")
            for num in best_combination:
                print(f"   红球{num:02d}的历史概率为{self._red_probs.get(num, 1/33):.4f}")
        
        return best_combination, blue_ball
    
    def _predict_by_ensemble(self, explain=False):
        """
        使用集成方法预测双色球号码
        综合多种预测方法的结果
        
        Args:
            explain: 是否解释预测结果
            
        Returns:
            (红球列表, 蓝球)
        """
        import random
        
        # 使用多种方法预测
        methods = ['stats', 'probability']
        
        # 如果决策树和贝叶斯方法可用，也加入集成
        if len(self.data) >= 10:
            methods.append('decision_tree')
        
        if PYMC_AVAILABLE:
            methods.append('bayes')
        
        # 收集各种方法的预测结果
        predictions = {}
        for method in methods:
            if method == 'stats':
                predictions['stats'] = self._predict_by_stats(False)
            elif method == 'probability':
                predictions['probability'] = self._predict_by_probability(False)
            elif method == 'decision_tree':
                predictions['decision_tree'] = self._predict_by_decision_tree(False)
            elif method == 'bayes':
                predictions['bayes'] = self._predict_by_bayes(False)
        
        # 统计红球出现频率
        red_votes = Counter()
        for method, (reds, _) in predictions.items():
            for num in reds:
                red_votes[num] += 1
        
        # 选择得票最多的红球
        top_reds = [num for num, _ in red_votes.most_common(6)]
        
        # 如果不足6个，从剩余号码中随机补充
        if len(top_reds) < 6:
            remaining = [num for num in range(1, 34) if num not in top_reds]
            top_reds.extend(random.sample(remaining, 6 - len(top_reds)))
        
        top_reds.sort()
        
        # 统计蓝球出现频率
        blue_votes = Counter()
        for method, (_, blue) in predictions.items():
            blue_votes[blue] += 1
        
        # 选择得票最多的蓝球，如果平局则随机选择
        most_votes = max(blue_votes.values())
        top_blues = [num for num, votes in blue_votes.items() if votes == most_votes]
        blue_ball = random.choice(top_blues)
        
        if explain:
            print("\n基于集成方法的预测解释:")
            print(f"1. 使用了{len(methods)}种预测方法: {', '.join(methods)}")
            print("2. 红球投票结果:")
            for num, votes in red_votes.most_common():
                if num in top_reds:
                    print(f"   红球{num:02d}: {votes}票 (入选)")
            print("3. 蓝球投票结果:")
            for num, votes in blue_votes.most_common():
                status = "入选" if num == blue_ball else ""
                print(f"   蓝球{num:02d}: {votes}票 {status}")
        
        return top_reds, blue_ball
        
    def generate_smart_numbers_advanced(self):
        """
        使用高级分析方法生成智能双色球号码
        结合统计特性、概率分布、频率模式、决策树和贝叶斯分析
        
        Returns:
            (红球列表, 蓝球)
        """
        # 调用predict_numbers方法，使用ensemble方法
        numbers = self.predict_numbers(method='ensemble', explain=True)
        return numbers[:6], numbers[6]
        
        # 1. 基于频率模式选择红球
        # 分析红球的冷热状态
        _, _, red_current_cold, _ = self.analyze_frequency_patterns()
        
        # 获取红球的平均和值、方差和跨度
        stats_df = self.analyze_statistical_features()
        avg_red_sum = stats_df.loc[stats_df['特征'] == '红球和值', '平均值'].values[0]
        avg_red_variance = stats_df.loc[stats_df['特征'] == '红球方差', '平均值'].values[0]
        avg_red_span = stats_df.loc[stats_df['特征'] == '红球跨度', '平均值'].values[0]
        
        # 获取红球的概率分布
        red_probs, _ = self.analyze_probability_distribution()
        
        # 根据冷热状态和概率分布选择红球候选集
        red_candidates = []
        for num in range(1, 34):
            # 计算综合得分：冷热指数 + 概率偏差
            cold_score = red_current_cold[num] / max(red_current_cold.values())  # 归一化冷却期
            prob_score = red_probs.get(num, 0) * 33  # 归一化概率（乘以33使其均值为1）
            
            # 综合得分 = 冷热得分 * 0.6 + 概率得分 * 0.4
            combined_score = cold_score * 0.6 + prob_score * 0.4
            red_candidates.append((num, combined_score))
        
        # 按综合得分排序
        red_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 从前15个高分候选中选择4个，从剩余候选中选择2个
        high_score_reds = [num for num, _ in red_candidates[:15]]
        low_score_reds = [num for num, _ in red_candidates[15:]]
        
        selected_reds = random.sample(high_score_reds, 4) + random.sample(low_score_reds, 2)
        selected_reds.sort()
        
        # 2. 使用决策树模型预测蓝球
        # 如果有足够的数据，使用决策树模型预测
        if len(self.data) >= 10:
            try:
                # 训练决策树模型
                blue_rf, _, _ = self.analyze_decision_tree()
                
                # 准备特征数据（最近5期的数据）
                n_prev = 5
                feature_row = []
                for j in range(min(n_prev, len(self.data))):
                    idx = len(self.data) - min(n_prev, len(self.data)) + j
                    feature_row.extend([
                        self.data.iloc[idx]['red_sum'],
                        self.data.iloc[idx]['red_variance'],
                        self.data.iloc[idx]['red_span'],
                        self.data.iloc[idx]['blue_ball']
                    ])
                
                # 如果数据不足5期，用0填充
                if len(self.data) < n_prev:
                    padding = (n_prev - len(self.data)) * 4  # 每期4个特征
                    feature_row = [0] * padding + feature_row
                
                # 预测蓝球
                predicted_blues = blue_rf.predict_proba([feature_row])[0]
                
                # 根据预测概率选择蓝球
                blue_probs = [(i+1, prob) for i, prob in enumerate(predicted_blues)]
                blue_probs.sort(key=lambda x: x[1], reverse=True)
                
                # 从概率最高的5个中随机选择1个
                top_blues = [num for num, _ in blue_probs[:5]]
                selected_blue = random.choice(top_blues)
            except Exception as e:
                print(f"使用决策树预测蓝球失败: {e}，将使用频率方法")
                # 如果决策树预测失败，使用频率方法
                _, _, _, blue_current_cold = self.analyze_frequency_patterns()
                blue_candidates = [(num, blue_current_cold[num]) for num in range(1, 17)]
                blue_candidates.sort(key=lambda x: x[1], reverse=True)
                top_blues = [num for num, _ in blue_candidates[:5]]
                selected_blue = random.choice(top_blues)
        else:
            # 数据不足，使用频率方法
            _, _, _, blue_current_cold = self.analyze_frequency_patterns()
            blue_candidates = [(num, blue_current_cold[num]) for num in range(1, 17)]
            blue_candidates.sort(key=lambda x: x[1], reverse=True)
            top_blues = [num for num, _ in blue_candidates[:5]]
            selected_blue = random.choice(top_blues)
        
        # 3. 验证选择的红球是否满足统计特性
        # 计算所选红球的和值、方差和跨度
        selected_sum = sum(selected_reds)
        selected_variance = np.var(selected_reds)
        selected_span = max(selected_reds) - min(selected_reds)
        
        # 如果统计特性偏离平均值太多，重新选择
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            # 检查和值是否在合理范围内（平均值±20%）
            if abs(selected_sum - avg_red_sum) > 0.2 * avg_red_sum:
                # 重新选择一个红球
                selected_reds.pop(random.randint(0, 5))
                remaining = [num for num in range(1, 34) if num not in selected_reds]
                selected_reds.append(random.choice(remaining))
                selected_reds.sort()
                
                # 重新计算统计特性
                selected_sum = sum(selected_reds)
                selected_variance = np.var(selected_reds)
                selected_span = max(selected_reds) - min(selected_reds)
                
                attempt += 1
            else:
                break
        
        print(f"生成的智能双色球号码: 红球: {' '.join([f'{num:02d}' for num in selected_reds])} | 蓝球: {selected_blue:02d}")
        return selected_reds, selected_blue

    def run_advanced_analysis(self):
        """
        运行所有高级分析
        """
        if not self.load_data():
            return False
        
        print("开始高级分析数据...")
        
        # 运行各种分析方法
        self.analyze_statistical_features()
        self.analyze_probability_distribution()
        self.analyze_frequency_patterns()
        self.analyze_decision_tree()
        self.analyze_cycle_patterns()
        
        # 如果PyMC可用，运行贝叶斯分析
        if PYMC_AVAILABLE:
            self.analyze_bayesian()
        
        print("高级分析完成！")
        return True


def main():
    """主函数"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据目录为上一级的data目录
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    data_file = os.path.join(data_dir, "ssq_data.csv")
    output_dir = os.path.join(data_dir, "advanced")
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据")
        return
    
    # 创建高级分析器实例
    analyzer = SSQAdvancedAnalyzer(data_file=data_file, output_dir=output_dir)
    
    # 运行高级分析
    analyzer.run_advanced_analysis()
    
    # 生成智能号码
    analyzer.generate_smart_numbers_advanced()


if __name__ == "__main__":
    main()