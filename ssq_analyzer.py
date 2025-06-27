#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据分析与预测系统 - 整合版本
集成所有功能：数据爬取、分析、预测、可视化
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
from collections import Counter
from bs4 import BeautifulSoup
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

# 可选依赖
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

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


class SSQAnalyzer:
    """双色球数据分析与预测系统 - 整合版本"""
    
    def __init__(self, data_dir="data"):
        """初始化分析器"""
        self.data_dir = data_dir
        self.data = None
        self.red_range = range(1, 34)  # 红球范围1-33
        self.blue_range = range(1, 17)  # 蓝球范围1-16
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "advanced"), exist_ok=True)
        
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
    
    # ==================== 数据爬取功能 ====================
    
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
            
            if count is not None:
                total_pages = (count + page_size - 1) // page_size
            else:
                total_pages = 1000
            
            while page <= total_pages:
                print(f"正在获取第{page}页数据 (每页{page_size}条)...")
                
                params = {
                    "name": "ssq",
                    "pageNo": page,
                    "pageSize": page_size,
                    "systemType": "PC"
                }
                
                response = requests.get(self.api_url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("state") == 0 and "result" in data:
                    for item in data["result"]:
                        issue = item["code"]
                        date = item["date"]
                        red_str = item["red"]
                        blue_ball = item["blue"]
                        
                        red_balls = red_str.split(",")
                        red_balls = [ball.zfill(2) for ball in red_balls]
                        blue_ball = blue_ball.zfill(2)
                        
                        results.append({
                            "issue": issue,
                            "date": date,
                            "red_balls": ",".join(red_balls),
                            "blue_ball": blue_ball
                        })
                    
                    time.sleep(random.uniform(1, 3))
                else:
                    break
                
                if len(data.get("result", [])) < page_size:
                    break
                
                page += 1
                
                if count is not None and len(results) >= count:
                    break
            
            print(f"从中国福利彩票官方网站成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中国福利彩票官方网站获取数据失败: {e}")
        
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            if count is not None and len(results) > count:
                results = results[:count]
        
        return results
    
    def crawl_data_from_zhcw(self):
        """从中彩网获取补充数据"""
        results = {}
        
        try:
            print("正在从中彩网获取补充数据...")
            
            max_pages = 50
            
            for page in range(1, max_pages + 1):
                try:
                    url = f"http://kaijiang.zhcw.com/zhcw/html/ssq/list_{page}.html"
                    
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Referer": "http://kaijiang.zhcw.com/",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                    }
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    response.encoding = 'utf-8'
                    
                    soup = BeautifulSoup(response.text, "html.parser")
                    table = soup.find('table', attrs={'class': 'wqhgt'})
                    
                    if not table:
                        break
                    
                    rows = table.find_all('tr')
                    if len(rows) <= 1:
                        break
                    
                    has_data = False
                    for row in rows[1:]:
                        try:
                            cells = row.find_all('td')
                            if len(cells) < 3:
                                continue
                            
                            date = cells[0].text.strip()
                            issue = cells[1].text.strip()
                            ball_cell = cells[2]
                            all_balls = ball_cell.find_all('em')
                            
                            if len(all_balls) != 7:
                                continue
                            
                            red_balls = []
                            for i in range(6):
                                red_balls.append(all_balls[i].text.strip().zfill(2))
                            
                            blue_ball = all_balls[6].text.strip().zfill(2)
                            
                            results[issue] = {
                                "issue": issue,
                                "date": date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_ball
                            }
                            has_data = True
                        except Exception as e:
                            continue
                    
                    if not has_data:
                        break
                    
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    continue
            
            print(f"从中彩网成功获取 {len(results)} 期双色球开奖结果")
        except Exception as e:
            print(f"从中彩网获取数据失败: {e}")
        
        return results
    
    def crawl_data(self, count=300, use_all_data=False):
        """爬取双色球历史数据"""
        if use_all_data:
            count = None
            filename = "ssq_data_all.csv"
        else:
            filename = "ssq_data.csv"
        
        # 从官方网站获取数据
        results = self.crawl_data_from_cwl(count)
        
        # 如果数据不足，从中彩网补充
        if count is None or len(results) < count:
            zhcw_results = self.crawl_data_from_zhcw()
            existing_issues = set(item["issue"] for item in results)
            
            added_count = 0
            for issue, item in zhcw_results.items():
                if issue not in existing_issues:
                    results.append(item)
                    existing_issues.add(issue)
                    added_count += 1
            
            print(f"从中彩网补充了{added_count}期不重复的数据")
            
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            if count is not None and len(results) > count:
                results = results[:count]
        
        # 保存数据
        if results:
            self.save_to_csv(results, filename)
            print(f"成功爬取{len(results)}期双色球历史数据，保存到{filename}")
            return True
        else:
            print("爬取数据失败")
            return False
    
    def save_to_csv(self, results, filename):
        """保存数据到CSV文件"""
        if not results:
            return

        file_path = os.path.join(self.data_dir, filename)

        # 检查文件是否已存在
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data[row['issue']] = row
            except Exception:
                pass

        # 合并数据
        all_data = {}
        for issue, row in existing_data.items():
            all_data[issue] = row

        for row in results:
            all_data[row['issue']] = row

        # 排序并保存
        sorted_data = list(all_data.values())
        sorted_data.sort(key=lambda x: int(x['issue']), reverse=True)

        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['issue', 'date', 'red_balls', 'blue_ball'])
            writer.writeheader()
            for row in sorted_data:
                writer.writerow(row)

        print(f"数据已保存到 {file_path}")

    def append_to_csv(self, new_results, filename):
        """追加新数据到CSV文件，按期号倒序排列"""
        if not new_results:
            print("没有新数据需要追加")
            return

        file_path = os.path.join(self.data_dir, filename)

        # 读取现有数据
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data[row['issue']] = row
                print(f"读取现有数据: {len(existing_data)}期")
            except Exception as e:
                print(f"读取现有数据失败: {e}")

        # 检查新数据
        new_count = 0
        updated_count = 0

        for row in new_results:
            issue = row['issue']
            if issue in existing_data:
                # 更新现有数据
                existing_data[issue] = row
                updated_count += 1
            else:
                # 添加新数据
                existing_data[issue] = row
                new_count += 1

        # 按期号倒序排列
        sorted_data = list(existing_data.values())
        sorted_data.sort(key=lambda x: int(x['issue']), reverse=True)

        # 保存到文件
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['issue', 'date', 'red_balls', 'blue_ball'])
            writer.writeheader()
            for row in sorted_data:
                writer.writerow(row)

        print(f"数据追加完成: 新增{new_count}期，更新{updated_count}期，总计{len(sorted_data)}期")

        # 显示最新几期数据
        if new_count > 0:
            print("最新追加的数据:")
            for i, row in enumerate(sorted_data[:min(3, new_count)]):
                print(f"  {row['issue']}期 ({row['date']}): 红球 {row['red_balls']} | 蓝球 {row['blue_ball']}")

    def crawl_specific_periods(self, start_issue=None, end_issue=None, count=None):
        """爬取指定期数的数据"""
        print(f"爬取指定期数数据...")

        if start_issue and end_issue:
            print(f"爬取期数范围: {start_issue} - {end_issue}")
        elif count:
            print(f"爬取最新{count}期数据")

        # 从官方网站获取数据
        results = []

        try:
            page_size = 30
            page = 1
            target_count = count if count else 1000

            while len(results) < target_count and page <= 100:  # 最多100页
                print(f"正在获取第{page}页数据...")

                params = {
                    "name": "ssq",
                    "pageNo": page,
                    "pageSize": page_size,
                    "systemType": "PC"
                }

                response = requests.get(self.api_url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()

                if data.get("state") == 0 and "result" in data:
                    page_results = []
                    for item in data["result"]:
                        issue = item["code"]
                        issue_num = int(issue)

                        # 检查是否在指定范围内
                        if start_issue and end_issue:
                            if not (int(start_issue) <= issue_num <= int(end_issue)):
                                continue

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

                    # 如果这页没有数据或数据不足，可能已经到底了
                    if len(page_results) < page_size:
                        break
                else:
                    break

                page += 1
                time.sleep(random.uniform(1, 2))

                # 如果指定了数量限制
                if count and len(results) >= count:
                    results = results[:count]
                    break

            print(f"成功爬取{len(results)}期数据")
            return results

        except Exception as e:
            print(f"爬取数据失败: {e}")
            return []

    def fetch_and_append_latest(self, filename="ssq_data.csv"):
        """获取最新一期开奖结果并追加到CSV文件"""
        print("获取最新一期开奖结果...")

        try:
            # 从官方API获取最新一期
            params = {
                "name": "ssq",
                "pageNo": 1,
                "pageSize": 1,
                "systemType": "PC"
            }

            response = requests.get(self.api_url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("state") == 0 and "result" in data and data["result"]:
                item = data["result"][0]
                issue = item["code"]
                date = item["date"]
                red_str = item["red"]
                blue_ball = item["blue"]

                red_balls = red_str.split(",")
                red_balls = [ball.zfill(2) for ball in red_balls]
                blue_ball = blue_ball.zfill(2)

                latest_result = [{
                    "issue": issue,
                    "date": date,
                    "red_balls": ",".join(red_balls),
                    "blue_ball": blue_ball
                }]

                print(f"获取到最新开奖: {issue}期 ({date})")
                print(f"开奖号码: 红球 {','.join(red_balls)} | 蓝球 {blue_ball}")

                # 追加到文件
                self.append_to_csv(latest_result, filename)

                return latest_result[0]
            else:
                print("未获取到最新开奖数据")
                return None

        except Exception as e:
            print(f"获取最新开奖失败: {e}")
            return None
    
    # ==================== 数据加载和验证 ====================
    
    def load_data(self, data_file=None):
        """加载数据"""
        if data_file is None:
            # 优先使用全量数据
            data_file = os.path.join(self.data_dir, "ssq_data_all.csv")
            if not os.path.exists(data_file):
                data_file = os.path.join(self.data_dir, "ssq_data.csv")
        
        try:
            if not os.path.exists(data_file):
                print(f"数据文件不存在: {data_file}")
                return False
            
            self.data = pd.read_csv(data_file)
            
            if self.data.empty:
                print("数据文件为空")
                return False
            
            # 处理日期列
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'], format='%Y-%m-%d', errors='coerce')
            
            # 拆分红球列为单独的列
            if 'red_balls' in self.data.columns:
                red_balls = self.data['red_balls'].str.split(',', expand=True)
                for i in range(6):
                    self.data[f'red_{i+1}'] = red_balls[i].astype(int)
            
            # 转换蓝球为整数
            if 'blue_ball' in self.data.columns:
                self.data['blue_ball'] = self.data['blue_ball'].astype(int)
            
            # 计算统计特征
            if all(f'red_{i}' in self.data.columns for i in range(1, 7)):
                self.data['red_sum'] = sum(self.data[f'red_{i}'] for i in range(1, 7))
                self.data['red_variance'] = self.data[[f'red_{i}' for i in range(1, 7)]].var(axis=1)
                self.data['red_span'] = self.data[[f'red_{i}' for i in range(1, 7)]].max(axis=1) - \
                                       self.data[[f'red_{i}' for i in range(1, 7)]].min(axis=1)
            
            print(f"成功加载{len(self.data)}条数据")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def validate_data(self, data_file=None):
        """验证数据文件的完整性"""
        if data_file is None:
            data_file = os.path.join(self.data_dir, "ssq_data.csv")
        
        try:
            if not os.path.exists(data_file):
                print(f"数据文件不存在: {data_file}")
                return False
            
            df = pd.read_csv(data_file)
            
            required_columns = ["issue", "date", "red_balls", "blue_ball"]
            for col in required_columns:
                if col not in df.columns:
                    print(f"数据文件缺少必要的列: {col}")
                    return False
            
            if len(df) == 0:
                print("数据文件为空")
                return False
            
            # 检查红球格式
            for _, row in df.iterrows():
                red_balls = row["red_balls"].split(",")
                if len(red_balls) != 6:
                    print(f"红球数量不正确: {row['issue']}期 {row['red_balls']}")
                    return False
            
            print(f"数据验证成功，共{len(df)}条记录")
            return True
        except Exception as e:
            print(f"数据验证失败: {e}")
            return False

    # ==================== 基础分析功能 ====================

    def analyze_number_frequency(self):
        """分析号码出现频率"""
        if self.data is None:
            if not self.load_data():
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

    def plot_number_frequency(self):
        """绘制号码频率图"""
        red_freq, blue_freq = self.analyze_number_frequency()

        if red_freq is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 红球频率图
        red_numbers = sorted(red_freq.keys())
        red_frequencies = [red_freq[num] for num in red_numbers]

        bars1 = ax1.bar(red_numbers, red_frequencies, color='red', alpha=0.7)
        ax1.set_title('红球号码出现频率', fontsize=14)
        ax1.set_xlabel('号码', fontsize=12)
        ax1.set_ylabel('出现频率', fontsize=12)
        ax1.set_xticks(red_numbers)
        ax1.grid(True, axis='y', alpha=0.3)

        # 添加数值标签
        for bar, freq in zip(bars1, red_frequencies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{freq:.3f}', ha='center', va='bottom', fontsize=8)

        # 蓝球频率图
        blue_numbers = sorted(blue_freq.keys())
        blue_frequencies = [blue_freq[num] for num in blue_numbers]

        bars2 = ax2.bar(blue_numbers, blue_frequencies, color='blue', alpha=0.7)
        ax2.set_title('蓝球号码出现频率', fontsize=14)
        ax2.set_xlabel('号码', fontsize=12)
        ax2.set_ylabel('出现频率', fontsize=12)
        ax2.set_xticks(blue_numbers)
        ax2.grid(True, axis='y', alpha=0.3)

        # 添加数值标签
        for bar, freq in zip(bars2, blue_frequencies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{freq:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'number_frequency.png'), dpi=300)
        plt.close()

        print("号码频率图已保存")

    def analyze_number_combinations(self):
        """分析号码组合特征"""
        if self.data is None:
            if not self.load_data():
                return

        # 计算组合特征
        combinations = []
        for _, row in self.data.iterrows():
            red_balls = [row[f'red_{i}'] for i in range(1, 7)]

            # 奇偶比
            odd_count = sum(1 for ball in red_balls if ball % 2 == 1)
            even_count = 6 - odd_count

            # 大小比（大于等于17为大）
            big_count = sum(1 for ball in red_balls if ball >= 17)
            small_count = 6 - big_count

            # 和值
            sum_value = sum(red_balls)

            # 跨度
            span = max(red_balls) - min(red_balls)

            combinations.append({
                'odd_count': odd_count,
                'even_count': even_count,
                'big_count': big_count,
                'small_count': small_count,
                'sum_value': sum_value,
                'span': span
            })

        # 绘制组合特征图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 奇偶比分布
        odd_counts = [combo['odd_count'] for combo in combinations]
        odd_count_freq = Counter(odd_counts)

        axes[0, 0].bar(odd_count_freq.keys(), odd_count_freq.values(), color='orange', alpha=0.7)
        axes[0, 0].set_title('红球奇数个数分布', fontsize=14)
        axes[0, 0].set_xlabel('奇数个数', fontsize=12)
        axes[0, 0].set_ylabel('出现次数', fontsize=12)
        axes[0, 0].grid(True, axis='y', alpha=0.3)

        # 大小比分布
        big_counts = [combo['big_count'] for combo in combinations]
        big_count_freq = Counter(big_counts)

        axes[0, 1].bar(big_count_freq.keys(), big_count_freq.values(), color='green', alpha=0.7)
        axes[0, 1].set_title('红球大数个数分布', fontsize=14)
        axes[0, 1].set_xlabel('大数个数(≥17)', fontsize=12)
        axes[0, 1].set_ylabel('出现次数', fontsize=12)
        axes[0, 1].grid(True, axis='y', alpha=0.3)

        # 和值分布
        sum_values = [combo['sum_value'] for combo in combinations]
        axes[1, 0].hist(sum_values, bins=30, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('红球和值分布', fontsize=14)
        axes[1, 0].set_xlabel('和值', fontsize=12)
        axes[1, 0].set_ylabel('出现次数', fontsize=12)
        axes[1, 0].grid(True, axis='y', alpha=0.3)

        # 跨度分布
        spans = [combo['span'] for combo in combinations]
        axes[1, 1].hist(spans, bins=20, color='brown', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('红球跨度分布', fontsize=14)
        axes[1, 1].set_xlabel('跨度', fontsize=12)
        axes[1, 1].set_ylabel('出现次数', fontsize=12)
        axes[1, 1].grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'number_combinations.png'), dpi=300)
        plt.close()

        print("号码组合特征图已保存")

        # 打印统计信息
        print("\n=== 号码组合特征统计 ===")
        print(f"和值范围: {min(sum_values)} - {max(sum_values)}")
        print(f"和值平均值: {np.mean(sum_values):.2f}")
        print(f"跨度范围: {min(spans)} - {max(spans)}")
        print(f"跨度平均值: {np.mean(spans):.2f}")

    def analyze_trend(self):
        """分析号码走势"""
        if self.data is None:
            if not self.load_data():
                return

        # 取最近50期数据
        recent_data = self.data.head(50)

        # 绘制红球走势图
        plt.figure(figsize=(16, 10))

        # 红球走势
        plt.subplot(2, 1, 1)
        for i in range(1, 7):
            plt.plot(range(len(recent_data)), recent_data[f'red_{i}'],
                    marker='o', label=f'红球{i}', linewidth=2, markersize=4)

        plt.title('最近50期红球走势', fontsize=14)
        plt.xlabel('期数（从最新到最旧）', fontsize=12)
        plt.ylabel('号码', fontsize=12)
        plt.yticks(range(1, 34))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='both', linestyle='--', alpha=0.3)

        # 蓝球走势
        plt.subplot(2, 1, 2)
        plt.plot(range(len(recent_data)), recent_data['blue_ball'],
                marker='o', color='blue', linewidth=2, markersize=6)

        plt.title('最近50期蓝球走势', fontsize=14)
        plt.xlabel('期数（从最新到最旧）', fontsize=12)
        plt.ylabel('号码', fontsize=12)
        plt.yticks(range(1, 17))
        plt.grid(True, axis='both', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'trend_analysis.png'), dpi=300)
        plt.close()

        print("号码走势图已保存")

    def run_basic_analysis(self):
        """运行基础分析"""
        if not self.load_data():
            return False

        print("开始基础分析...")
        self.plot_number_frequency()
        self.analyze_number_combinations()
        self.analyze_trend()
        print("基础分析完成！")
        return True

    # ==================== 高级分析功能 ====================

    def analyze_statistical_features(self):
        """分析统计特性"""
        if self.data is None:
            if not self.load_data():
                return

        print("分析统计特性...")

        # 计算红球统计特征
        red_sums = self.data['red_sum']
        red_variances = self.data['red_variance']
        red_spans = self.data['red_span']

        stats_results = {
            '红球和值': {
                '平均值': red_sums.mean(),
                '标准差': red_sums.std(),
                '最小值': red_sums.min(),
                '最大值': red_sums.max(),
                '中位数': red_sums.median()
            },
            '红球方差': {
                '平均值': red_variances.mean(),
                '标准差': red_variances.std(),
                '最小值': red_variances.min(),
                '最大值': red_variances.max(),
                '中位数': red_variances.median()
            },
            '红球跨度': {
                '平均值': red_spans.mean(),
                '标准差': red_spans.std(),
                '最小值': red_spans.min(),
                '最大值': red_spans.max(),
                '中位数': red_spans.median()
            }
        }

        # 保存结果
        output_file = os.path.join(self.data_dir, "advanced", "statistical_features.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats_results, f, ensure_ascii=False, indent=4, default=str)

        print(f"统计特性分析结果已保存到 {output_file}")
        return stats_results

    def analyze_probability_distribution(self):
        """分析概率分布"""
        if self.data is None:
            if not self.load_data():
                return

        print("分析概率分布...")

        # 计算红球和蓝球的历史概率分布
        red_freq, blue_freq = self.analyze_number_frequency()

        # 计算概率分布统计
        red_probs = list(red_freq.values())
        blue_probs = list(blue_freq.values())

        prob_results = {
            '红球概率分布': {
                '平均概率': np.mean(red_probs),
                '概率标准差': np.std(red_probs),
                '最高概率': max(red_probs),
                '最低概率': min(red_probs),
                '概率分布': red_freq
            },
            '蓝球概率分布': {
                '平均概率': np.mean(blue_probs),
                '概率标准差': np.std(blue_probs),
                '最高概率': max(blue_probs),
                '最低概率': min(blue_probs),
                '概率分布': blue_freq
            }
        }

        # 保存结果
        output_file = os.path.join(self.data_dir, "advanced", "probability_distribution.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prob_results, f, ensure_ascii=False, indent=4, default=str)

        print(f"概率分布分析结果已保存到 {output_file}")
        return prob_results

    def analyze_frequency_patterns(self):
        """分析频率模式"""
        if self.data is None:
            if not self.load_data():
                return

        print("分析频率模式...")

        red_freq, blue_freq = self.analyze_number_frequency()

        # 计算平均频率
        avg_red_freq = np.mean(list(red_freq.values()))
        avg_blue_freq = np.mean(list(blue_freq.values()))

        # 分类冷热号码
        hot_red = {num: freq for num, freq in red_freq.items() if freq > avg_red_freq}
        cold_red = {num: freq for num, freq in red_freq.items() if freq <= avg_red_freq}

        hot_blue = {num: freq for num, freq in blue_freq.items() if freq > avg_blue_freq}
        cold_blue = {num: freq for num, freq in blue_freq.items() if freq <= avg_blue_freq}

        pattern_results = {
            '红球热号': hot_red,
            '红球冷号': cold_red,
            '蓝球热号': hot_blue,
            '蓝球冷号': cold_blue,
            '平均红球频率': avg_red_freq,
            '平均蓝球频率': avg_blue_freq
        }

        # 保存结果
        output_file = os.path.join(self.data_dir, "advanced", "frequency_patterns.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pattern_results, f, ensure_ascii=False, indent=4, default=str)

        print(f"频率模式分析结果已保存到 {output_file}")
        return pattern_results

    def analyze_decision_tree(self):
        """决策树分析"""
        if self.data is None:
            if not self.load_data():
                return

        print("进行决策树分析...")

        try:
            # 准备特征数据
            features = []
            targets = []

            # 使用前5期的数据作为特征
            n_prev = 5
            for i in range(n_prev, len(self.data)):
                feature_row = []

                # 提取前n期的特征
                for j in range(n_prev):
                    idx = i - n_prev + j
                    feature_row.extend([
                        self.data.iloc[idx]['red_sum'],
                        self.data.iloc[idx]['red_variance'],
                        self.data.iloc[idx]['red_span'],
                        self.data.iloc[idx]['blue_ball']
                    ])

                features.append(feature_row)
                targets.append(self.data.iloc[i]['blue_ball'])

            X = np.array(features)
            y = np.array(targets)

            # 训练随机森林模型
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X, y)

            # 交叉验证评估
            scores = cross_val_score(rf, X, y, cv=5)
            accuracy = scores.mean()

            # 特征重要性
            feature_names = []
            for j in range(n_prev):
                feature_names.extend([
                    f'前{n_prev-j}期红球和值',
                    f'前{n_prev-j}期红球方差',
                    f'前{n_prev-j}期红球跨度',
                    f'前{n_prev-j}期蓝球'
                ])

            feature_importance = dict(zip(feature_names, rf.feature_importances_))

            tree_results = {
                '模型准确率': accuracy,
                '交叉验证分数': scores.tolist(),
                '特征重要性': feature_importance,
                '训练样本数': len(X)
            }

            # 保存结果
            output_file = os.path.join(self.data_dir, "advanced", "decision_tree_analysis.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tree_results, f, ensure_ascii=False, indent=4, default=str)

            print(f"决策树分析结果已保存到 {output_file}")
            print(f"模型准确率: {accuracy:.4f}")

            return tree_results
        except Exception as e:
            print(f"决策树分析失败: {e}")
            return None

    def analyze_cycle_patterns(self):
        """分析周期和规律"""
        if self.data is None:
            if not self.load_data():
                return

        print("分析周期和规律...")

        # 分析红球和值的周期性
        red_sum_series = self.data['red_sum'].values

        # 计算自相关系数
        max_lag = min(50, len(red_sum_series) // 2)
        acf_values = [1.0]  # 自相关系数，lag=0时为1
        for lag in range(1, max_lag + 1):
            if len(red_sum_series) > lag:
                acf = np.corrcoef(red_sum_series[:-lag], red_sum_series[lag:])[0, 1]
                if not np.isnan(acf):
                    acf_values.append(acf)
                else:
                    acf_values.append(0.0)
            else:
                acf_values.append(0.0)

        # 寻找显著的周期性
        significant_lags = []
        threshold = 0.1  # 相关系数阈值
        for i, acf in enumerate(acf_values[1:], 1):
            if abs(acf) > threshold:
                significant_lags.append((i, acf))

        # 分析蓝球的周期性
        blue_series = self.data['blue_ball'].values
        blue_acf_values = [1.0]
        for lag in range(1, min(30, len(blue_series) // 2) + 1):
            if len(blue_series) > lag:
                acf = np.corrcoef(blue_series[:-lag], blue_series[lag:])[0, 1]
                if not np.isnan(acf):
                    blue_acf_values.append(acf)
                else:
                    blue_acf_values.append(0.0)
            else:
                blue_acf_values.append(0.0)

        # 分析号码出现间隔
        interval_stats = {}
        for ball in range(1, 34):  # 红球
            intervals = []
            last_pos = -1
            for i, row in self.data.iterrows():
                if ball in [row[f'red_{j}'] for j in range(1, 7)]:
                    if last_pos != -1:
                        intervals.append(i - last_pos)
                    last_pos = i

            if intervals:
                interval_stats[f'红球{ball}'] = {
                    '平均间隔': np.mean(intervals),
                    '最小间隔': min(intervals),
                    '最大间隔': max(intervals),
                    '间隔标准差': np.std(intervals)
                }

        cycle_results = {
            '红球和值自相关': {f'lag_{i}': acf for i, acf in enumerate(acf_values)},
            '蓝球自相关': {f'lag_{i}': acf for i, acf in enumerate(blue_acf_values)},
            '显著周期': significant_lags,
            '号码间隔统计': interval_stats
        }

        # 保存结果
        output_file = os.path.join(self.data_dir, "advanced", "cycle_patterns.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cycle_results, f, ensure_ascii=False, indent=4, default=str)

        print(f"周期分析结果已保存到 {output_file}")
        if significant_lags:
            print(f"发现{len(significant_lags)}个显著周期")

        return cycle_results

    def analyze_historical_correlation(self, periods_list=[5, 10, 50, 100]):
        """分析历史号码之间的关联性"""
        if self.data is None:
            if not self.load_data():
                return

        print("分析历史号码关联性...")

        # 确保数据按期号排序（从新到旧）
        if 'issue' in self.data.columns:
            self.data = self.data.sort_values('issue', ascending=False).reset_index(drop=True)

        results = []

        # 对每个期数间隔进行分析
        for periods in periods_list:
            if len(self.data) <= periods:
                print(f"数据量不足{periods}期，跳过此间隔分析")
                continue

            red_matches = 0
            blue_matches = 0

            # 分析红球重复
            for i in range(len(self.data) - periods):
                current_reds = set([self.data.iloc[i][f'red_{j}'] for j in range(1, 7)])
                target_reds = set([self.data.iloc[i + periods][f'red_{j}'] for j in range(1, 7)])

                overlap = len(current_reds & target_reds)
                if overlap > 0:
                    red_matches += overlap

            # 分析蓝球重复
            for i in range(len(self.data) - periods):
                if self.data.iloc[i]['blue_ball'] == self.data.iloc[i + periods]['blue_ball']:
                    blue_matches += 1

            total_comparisons = len(self.data) - periods

            results.append({
                '间隔期数': periods,
                '红球重复次数': red_matches,
                '蓝球重复次数': blue_matches,
                '总比较次数': total_comparisons,
                '红球重复率': red_matches / (total_comparisons * 6) if total_comparisons > 0 else 0,
                '蓝球重复率': blue_matches / total_comparisons if total_comparisons > 0 else 0
            })

        correlation_results = {
            '间隔分析': results,
            '分析说明': '分析不同期数间隔的号码重复情况'
        }

        # 保存结果
        output_file = os.path.join(self.data_dir, "advanced", "historical_correlation.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(correlation_results, f, ensure_ascii=False, indent=4, default=str)

        print(f"历史关联性分析结果已保存到 {output_file}")
        return correlation_results

    def analyze_issue_number_correlation(self):
        """分析开奖号码与期号的关联性"""
        if self.data is None:
            if not self.load_data():
                return

        print("分析开奖号码与期号的关联性...")

        # 确保数据包含期号
        if 'issue' not in self.data.columns:
            print("数据中不包含期号信息，无法进行分析")
            return None

        # 提取期号的数字部分
        self.data['issue_number'] = self.data['issue'].astype(str).str.extract('(\d+)').astype(int)

        # 初始化结果列表
        results = []

        # 分析红球与期号的关联
        for i in range(1, 7):
            red_col = f'red_{i}'
            for _, row in self.data.iterrows():
                issue_num = row['issue_number']
                red_ball = row[red_col]

                # 检查红球是否与期号末尾数字相关
                if red_ball == (issue_num % 33) + 1:
                    results.append({
                        '类型': '红球',
                        '号码': red_ball,
                        '期号': row['issue'],
                        '日期': row['date'],
                        '关联描述': f"红球{red_ball}与期号{row['issue']}的计算值{(issue_num % 33) + 1}相同"
                    })

        # 分析蓝球与期号的关联
        for _, row in self.data.iterrows():
            issue_num = row['issue_number']
            blue_ball = row['blue_ball']

            # 检查蓝球是否与期号末尾数字相关
            if blue_ball == (issue_num % 16) + 1:
                results.append({
                    '类型': '蓝球',
                    '号码': blue_ball,
                    '期号': row['issue'],
                    '日期': row['date'],
                    '关联描述': f"蓝球{blue_ball}与期号{row['issue']}的计算值{(issue_num % 16) + 1}相同"
                })

        issue_correlation_results = {
            '关联匹配': results,
            '统计摘要': {
                '总匹配数': len(results),
                '红球匹配数': len([r for r in results if r['类型'] == '红球']),
                '蓝球匹配数': len([r for r in results if r['类型'] == '蓝球'])
            }
        }

        # 保存结果
        output_file = os.path.join(self.data_dir, "advanced", "issue_number_correlation.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(issue_correlation_results, f, ensure_ascii=False, indent=4, default=str)

        print(f"期号关联性分析结果已保存到 {output_file}")
        print(f"发现{len(results)}个与期号相关的匹配")

        return issue_correlation_results

    def analyze_markov_chain(self):
        """马尔可夫链分析"""
        if self.data is None:
            if not self.load_data():
                return

        print("分析双色球号码的转移概率...")
        print(f"分析数据期数: {len(self.data)}期")

        # 按期号排序，确保从最早期到最新期的顺序
        sorted_data = self.data.sort_values('issue', ascending=True).reset_index(drop=True)

        # 初始化结果字典
        results = {
            '红球全局转移概率': {},
            '红球位置转移概率': {},
            '蓝球转移概率': {},
            '红球组合转移概率': {},
            '期号转移统计': {},
            '转移概率演化': {}
        }

        # 1. 分析红球全局转移概率（不考虑位置）
        print("分析红球全局转移概率...")
        red_global_transitions = {}

        for i in range(len(sorted_data) - 1):
            current_reds = [sorted_data.iloc[i][f'red_{j}'] for j in range(1, 7)]
            next_reds = [sorted_data.iloc[i + 1][f'red_{j}'] for j in range(1, 7)]

            # 统计每个红球到下一期所有红球的转移
            for current_ball in current_reds:
                if current_ball not in red_global_transitions:
                    red_global_transitions[current_ball] = {}

                for next_ball in next_reds:
                    if next_ball not in red_global_transitions[current_ball]:
                        red_global_transitions[current_ball][next_ball] = 0
                    red_global_transitions[current_ball][next_ball] += 1

        # 计算全局转移概率
        red_global_probs = {}
        for current, nexts in red_global_transitions.items():
            total = sum(nexts.values())
            red_global_probs[current] = {
                next_ball: count / total for next_ball, count in nexts.items()
            }

        results['红球全局转移概率'] = red_global_probs

        # 2. 分析红球位置转移概率
        print("分析红球位置转移概率...")
        for pos in range(1, 7):
            red_col = f'red_{pos}'
            position_transitions = {}

            for i in range(len(sorted_data) - 1):
                current_ball = sorted_data.iloc[i][red_col]
                next_ball = sorted_data.iloc[i + 1][red_col]

                if current_ball not in position_transitions:
                    position_transitions[current_ball] = {}

                if next_ball not in position_transitions[current_ball]:
                    position_transitions[current_ball][next_ball] = 0

                position_transitions[current_ball][next_ball] += 1

            # 计算位置转移概率
            position_probs = {}
            for current, nexts in position_transitions.items():
                total = sum(nexts.values())
                position_probs[current] = {
                    next_ball: count / total for next_ball, count in nexts.items()
                }

            results['红球位置转移概率'][pos] = position_probs

        # 3. 分析蓝球转移概率
        print("分析蓝球转移概率...")
        blue_transitions = {}

        for i in range(len(sorted_data) - 1):
            current_ball = sorted_data.iloc[i]['blue_ball']
            next_ball = sorted_data.iloc[i + 1]['blue_ball']

            if current_ball not in blue_transitions:
                blue_transitions[current_ball] = {}

            if next_ball not in blue_transitions[current_ball]:
                blue_transitions[current_ball][next_ball] = 0

            blue_transitions[current_ball][next_ball] += 1

        # 计算蓝球转移概率
        blue_probs = {}
        for current, nexts in blue_transitions.items():
            total = sum(nexts.values())
            blue_probs[current] = {
                next_ball: count / total for next_ball, count in nexts.items()
            }

        results['蓝球转移概率'] = blue_probs

        # 4. 分析红球组合模式转移
        print("分析红球组合模式转移...")
        combo_transitions = {}

        for i in range(len(sorted_data) - 1):
            current_reds = sorted([sorted_data.iloc[i][f'red_{j}'] for j in range(1, 7)])
            next_reds = sorted([sorted_data.iloc[i + 1][f'red_{j}'] for j in range(1, 7)])

            # 分析奇偶比转移
            current_odd_count = sum(1 for x in current_reds if x % 2 == 1)
            next_odd_count = sum(1 for x in next_reds if x % 2 == 1)

            odd_key = f"奇偶比_{current_odd_count}_{6-current_odd_count}"
            if odd_key not in combo_transitions:
                combo_transitions[odd_key] = {}

            next_odd_key = f"奇偶比_{next_odd_count}_{6-next_odd_count}"
            if next_odd_key not in combo_transitions[odd_key]:
                combo_transitions[odd_key][next_odd_key] = 0
            combo_transitions[odd_key][next_odd_key] += 1

            # 分析大小比转移（大于等于17为大）
            current_big_count = sum(1 for x in current_reds if x >= 17)
            next_big_count = sum(1 for x in next_reds if x >= 17)

            big_key = f"大小比_{current_big_count}_{6-current_big_count}"
            if big_key not in combo_transitions:
                combo_transitions[big_key] = {}

            next_big_key = f"大小比_{next_big_count}_{6-next_big_count}"
            if next_big_key not in combo_transitions[big_key]:
                combo_transitions[big_key][next_big_key] = 0
            combo_transitions[big_key][next_big_key] += 1

        # 计算组合转移概率
        combo_probs = {}
        for pattern, nexts in combo_transitions.items():
            total = sum(nexts.values())
            combo_probs[pattern] = {
                next_pattern: count / total for next_pattern, count in nexts.items()
            }

        results['红球组合转移概率'] = combo_probs

        # 保存详细的分析结果
        try:
            # 将结果转换为可序列化的格式
            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {str(k): convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            serializable_results = convert_to_serializable(results)

            output_file = os.path.join(self.data_dir, 'advanced', 'enhanced_markov_chain_analysis.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=4)
            print(f"马尔可夫链分析结果已保存到 {output_file}")
        except Exception as e:
            print(f"保存马尔可夫链分析结果时出错: {e}")

        # 创建可视化
        try:
            self._visualize_markov_chain(results)
        except Exception as e:
            print(f"可视化马尔可夫链时出错: {e}")

        # 将结果保存为类属性，供预测使用
        self._enhanced_markov_results = results

        # 打印分析摘要
        print("\n=== 马尔可夫链分析摘要 ===")
        print(f"分析期数: {len(sorted_data)}期")
        print(f"红球全局转移状态数: {len(results['红球全局转移概率'])}")
        print(f"蓝球转移状态数: {len(results['蓝球转移概率'])}")
        print(f"组合模式数: {len(results['红球组合转移概率'])}")

        return results

    def _analyze_markov_chain_simple(self, data):
        """简化的马尔可夫链分析（用于准确性测试）"""
        # 按期号排序，确保从最早期到最新期的顺序
        sorted_data = data.sort_values('issue', ascending=True).reset_index(drop=True)

        # 初始化结果字典
        results = {
            '红球全局转移概率': {},
            '红球位置转移概率': {},
            '蓝球转移概率': {}
        }

        # 1. 分析红球全局转移概率
        red_global_transitions = {}

        for i in range(len(sorted_data) - 1):
            current_reds = [sorted_data.iloc[i][f'red_{j}'] for j in range(1, 7)]
            next_reds = [sorted_data.iloc[i + 1][f'red_{j}'] for j in range(1, 7)]

            for current_ball in current_reds:
                if current_ball not in red_global_transitions:
                    red_global_transitions[current_ball] = {}

                for next_ball in next_reds:
                    if next_ball not in red_global_transitions[current_ball]:
                        red_global_transitions[current_ball][next_ball] = 0
                    red_global_transitions[current_ball][next_ball] += 1

        # 计算全局转移概率
        red_global_probs = {}
        for current, nexts in red_global_transitions.items():
            total = sum(nexts.values())
            red_global_probs[current] = {
                next_ball: count / total for next_ball, count in nexts.items()
            }

        results['红球全局转移概率'] = red_global_probs

        # 2. 分析红球位置转移概率
        for pos in range(1, 7):
            red_col = f'red_{pos}'
            position_transitions = {}

            for i in range(len(sorted_data) - 1):
                current_ball = sorted_data.iloc[i][red_col]
                next_ball = sorted_data.iloc[i + 1][red_col]

                if current_ball not in position_transitions:
                    position_transitions[current_ball] = {}

                if next_ball not in position_transitions[current_ball]:
                    position_transitions[current_ball][next_ball] = 0

                position_transitions[current_ball][next_ball] += 1

            # 计算位置转移概率
            position_probs = {}
            for current, nexts in position_transitions.items():
                total = sum(nexts.values())
                position_probs[current] = {
                    next_ball: count / total for next_ball, count in nexts.items()
                }

            results['红球位置转移概率'][pos] = position_probs

        # 3. 分析蓝球转移概率
        blue_transitions = {}

        for i in range(len(sorted_data) - 1):
            current_ball = sorted_data.iloc[i]['blue_ball']
            next_ball = sorted_data.iloc[i + 1]['blue_ball']

            if current_ball not in blue_transitions:
                blue_transitions[current_ball] = {}

            if next_ball not in blue_transitions[current_ball]:
                blue_transitions[current_ball][next_ball] = 0

            blue_transitions[current_ball][next_ball] += 1

        # 计算蓝球转移概率
        blue_probs = {}
        for current, nexts in blue_transitions.items():
            total = sum(nexts.values())
            blue_probs[current] = {
                next_ball: count / total for next_ball, count in nexts.items()
            }

        results['蓝球转移概率'] = blue_probs

        return results

    def _analyze_markov_chain_stability(self, data):
        """基于指定期数数据的马尔可夫链稳定性分析"""
        print(f"进行马尔可夫链稳定性概率统计...")

        # 按期号排序，确保从最早期到最新期的顺序
        sorted_data = data.sort_values('issue', ascending=True).reset_index(drop=True)

        # 初始化结果字典
        results = {
            '红球稳定性转移概率': {},
            '红球位置稳定性概率': {},
            '蓝球稳定性转移概率': {},
            '稳定性统计': {},
            '概率置信度': {}
        }

        print(f"分析{len(sorted_data)}期数据的转移稳定性...")

        # 1. 分析红球全局稳定性转移概率
        red_global_transitions = {}
        red_transition_counts = {}

        for i in range(len(sorted_data) - 1):
            current_reds = [sorted_data.iloc[i][f'red_{j}'] for j in range(1, 7)]
            next_reds = [sorted_data.iloc[i + 1][f'red_{j}'] for j in range(1, 7)]

            for current_ball in current_reds:
                if current_ball not in red_global_transitions:
                    red_global_transitions[current_ball] = {}
                    red_transition_counts[current_ball] = 0

                red_transition_counts[current_ball] += 1

                for next_ball in next_reds:
                    if next_ball not in red_global_transitions[current_ball]:
                        red_global_transitions[current_ball][next_ball] = 0
                    red_global_transitions[current_ball][next_ball] += 1

        # 计算稳定性概率（考虑样本数量的置信度）
        red_stability_probs = {}
        for current, nexts in red_global_transitions.items():
            total_transitions = red_transition_counts[current]
            total_next_count = sum(nexts.values())

            # 计算稳定性权重（样本数越多，稳定性越高）
            stability_weight = min(1.0, total_transitions / 10.0)  # 10次以上转移认为稳定

            red_stability_probs[current] = {}
            for next_ball, count in nexts.items():
                base_prob = count / total_next_count
                # 稳定性调整概率
                stability_prob = base_prob * stability_weight + (1 - stability_weight) * (1/33)
                red_stability_probs[current][next_ball] = {
                    '概率': stability_prob,
                    '原始概率': base_prob,
                    '出现次数': count,
                    '总转移次数': total_transitions,
                    '稳定性权重': stability_weight
                }

        results['红球稳定性转移概率'] = red_stability_probs

        # 2. 分析红球位置稳定性概率
        for pos in range(1, 7):
            red_col = f'red_{pos}'
            position_transitions = {}
            position_counts = {}

            for i in range(len(sorted_data) - 1):
                current_ball = sorted_data.iloc[i][red_col]
                next_ball = sorted_data.iloc[i + 1][red_col]

                if current_ball not in position_transitions:
                    position_transitions[current_ball] = {}
                    position_counts[current_ball] = 0

                position_counts[current_ball] += 1

                if next_ball not in position_transitions[current_ball]:
                    position_transitions[current_ball][next_ball] = 0

                position_transitions[current_ball][next_ball] += 1

            # 计算位置稳定性概率
            position_stability_probs = {}
            for current, nexts in position_transitions.items():
                total_transitions = position_counts[current]
                total_next_count = sum(nexts.values())

                stability_weight = min(1.0, total_transitions / 5.0)  # 位置转移5次以上认为稳定

                position_stability_probs[current] = {}
                for next_ball, count in nexts.items():
                    base_prob = count / total_next_count
                    stability_prob = base_prob * stability_weight + (1 - stability_weight) * (1/33)
                    position_stability_probs[current][next_ball] = {
                        '概率': stability_prob,
                        '原始概率': base_prob,
                        '出现次数': count,
                        '总转移次数': total_transitions,
                        '稳定性权重': stability_weight
                    }

            results['红球位置稳定性概率'][pos] = position_stability_probs

        # 3. 分析蓝球稳定性转移概率
        blue_transitions = {}
        blue_counts = {}

        for i in range(len(sorted_data) - 1):
            current_ball = sorted_data.iloc[i]['blue_ball']
            next_ball = sorted_data.iloc[i + 1]['blue_ball']

            if current_ball not in blue_transitions:
                blue_transitions[current_ball] = {}
                blue_counts[current_ball] = 0

            blue_counts[current_ball] += 1

            if next_ball not in blue_transitions[current_ball]:
                blue_transitions[current_ball][next_ball] = 0

            blue_transitions[current_ball][next_ball] += 1

        # 计算蓝球稳定性概率
        blue_stability_probs = {}
        for current, nexts in blue_transitions.items():
            total_transitions = blue_counts[current]
            total_next_count = sum(nexts.values())

            stability_weight = min(1.0, total_transitions / 3.0)  # 蓝球3次以上转移认为稳定

            blue_stability_probs[current] = {}
            for next_ball, count in nexts.items():
                base_prob = count / total_next_count
                stability_prob = base_prob * stability_weight + (1 - stability_weight) * (1/16)
                blue_stability_probs[current][next_ball] = {
                    '概率': stability_prob,
                    '原始概率': base_prob,
                    '出现次数': count,
                    '总转移次数': total_transitions,
                    '稳定性权重': stability_weight
                }

        results['蓝球稳定性转移概率'] = blue_stability_probs

        # 4. 计算整体稳定性统计
        total_red_states = len(red_stability_probs)
        stable_red_states = sum(1 for probs in red_stability_probs.values()
                               if any(info['稳定性权重'] >= 0.5 for info in probs.values()))

        total_blue_states = len(blue_stability_probs)
        stable_blue_states = sum(1 for probs in blue_stability_probs.values()
                                if any(info['稳定性权重'] >= 0.5 for info in probs.values()))

        results['稳定性统计'] = {
            '红球总状态数': total_red_states,
            '红球稳定状态数': stable_red_states,
            '红球稳定性比例': stable_red_states / total_red_states if total_red_states > 0 else 0,
            '蓝球总状态数': total_blue_states,
            '蓝球稳定状态数': stable_blue_states,
            '蓝球稳定性比例': stable_blue_states / total_blue_states if total_blue_states > 0 else 0,
            '分析期数': len(sorted_data)
        }

        print(f"稳定性分析完成:")
        print(f"  红球稳定状态: {stable_red_states}/{total_red_states} ({stable_red_states/total_red_states*100:.1f}%)")
        print(f"  蓝球稳定状态: {stable_blue_states}/{total_blue_states} ({stable_blue_states/total_blue_states*100:.1f}%)")

        return results

    def _visualize_markov_chain(self, results):
        """可视化马尔可夫链分析结果"""
        try:
            # 1. 可视化红球全局转移概率热力图
            red_global_probs = results['红球全局转移概率']

            # 创建转移概率矩阵
            transition_matrix = np.zeros((33, 33))
            for from_ball in range(1, 34):
                if from_ball in red_global_probs:
                    for to_ball in range(1, 34):
                        if to_ball in red_global_probs[from_ball]:
                            transition_matrix[from_ball-1, to_ball-1] = red_global_probs[from_ball][to_ball]

            plt.figure(figsize=(15, 12))
            sns.heatmap(transition_matrix,
                       xticklabels=range(1, 34),
                       yticklabels=range(1, 34),
                       cmap='YlOrRd',
                       cbar_kws={'label': '转移概率'})
            plt.title('红球全局转移概率热力图', fontsize=16)
            plt.xlabel('下期红球号码', fontsize=12)
            plt.ylabel('当期红球号码', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'advanced', 'red_ball_global_transition_heatmap.png'), dpi=300)
            plt.close()

            # 2. 可视化蓝球转移概率热力图
            blue_probs = results['蓝球转移概率']

            blue_matrix = np.zeros((16, 16))
            for from_ball in range(1, 17):
                if from_ball in blue_probs:
                    for to_ball in range(1, 17):
                        if to_ball in blue_probs[from_ball]:
                            blue_matrix[from_ball-1, to_ball-1] = blue_probs[from_ball][to_ball]

            plt.figure(figsize=(12, 10))
            sns.heatmap(blue_matrix,
                       xticklabels=range(1, 17),
                       yticklabels=range(1, 17),
                       cmap='Blues',
                       annot=True,
                       fmt='.3f',
                       cbar_kws={'label': '转移概率'})
            plt.title('蓝球转移概率热力图', fontsize=16)
            plt.xlabel('下期蓝球号码', fontsize=12)
            plt.ylabel('当期蓝球号码', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'advanced', 'blue_ball_transition_heatmap.png'), dpi=300)
            plt.close()

            print("马尔可夫链可视化完成")
        except Exception as e:
            print(f"马尔可夫链可视化失败: {e}")

    # ==================== 预测功能 ====================

    def predict_by_markov_chain(self, explain=False, use_max_prob=True):
        """基于马尔可夫链预测双色球号码"""
        # 确保马尔可夫链分析已完成
        if not hasattr(self, '_enhanced_markov_results') or self._enhanced_markov_results is None:
            self._enhanced_markov_results = self.analyze_markov_chain()

        # 获取最近一期的号码
        latest_data = self.data.iloc[0]
        latest_reds = [latest_data[f'red_{i}'] for i in range(1, 7)]
        latest_blue = latest_data['blue_ball']

        if explain:
            print("\n基于马尔可夫链的预测:")
            print(f"最近一期开奖号码: 红球 {' '.join([f'{ball:02d}' for ball in latest_reds])} | 蓝球 {latest_blue:02d}")
            print("使用全量历史数据的转移概率进行预测...")

        # 使用增强的分析结果
        red_global_probs = self._enhanced_markov_results['红球全局转移概率']
        red_position_probs = self._enhanced_markov_results['红球位置转移概率']
        blue_probs = self._enhanced_markov_results['蓝球转移概率']

        # 计算全局红球频率分布（用于没有转移概率记录的情况）
        red_freq = {}
        for i in range(1, 34):
            red_freq[i] = 0

        for _, row in self.data.iterrows():
            for j in range(1, 7):
                red_freq[row[f'red_{j}']] += 1

        total_red_count = sum(red_freq.values())
        red_freq = {ball: count / total_red_count for ball, count in red_freq.items()}

        # 预测红球 - 综合多种策略
        predicted_reds = []
        used_balls = set()

        # 策略1: 基于位置转移概率预测前3个红球
        for pos in range(1, 4):  # 前3个位置
            current_ball = latest_reds[pos - 1]

            if pos in red_position_probs and current_ball in red_position_probs[pos]:
                candidates = list(red_position_probs[pos][current_ball].keys())
                probabilities = list(red_position_probs[pos][current_ball].values())

                # 过滤已使用的球
                filtered_candidates = []
                filtered_probs = []
                for i, candidate in enumerate(candidates):
                    if candidate not in used_balls:
                        filtered_candidates.append(candidate)
                        filtered_probs.append(probabilities[i])

                if filtered_candidates:
                    if use_max_prob:
                        # 选择概率最高的
                        max_idx = filtered_probs.index(max(filtered_probs))
                        next_ball = filtered_candidates[max_idx]
                    else:
                        # 基于概率分布随机选择
                        prob_sum = sum(filtered_probs)
                        normalized_probs = [p / prob_sum for p in filtered_probs]
                        next_ball = np.random.choice(filtered_candidates, p=normalized_probs)

                    predicted_reds.append(next_ball)
                    used_balls.add(next_ball)

                    if explain:
                        print(f"位置{pos}: {current_ball:02d} -> {next_ball:02d} (概率: {red_position_probs[pos][current_ball][next_ball]:.4f})")

        # 策略2: 基于全局转移概率预测剩余红球
        for current_ball in latest_reds:
            if len(predicted_reds) >= 6:
                break

            if current_ball in red_global_probs:
                candidates = list(red_global_probs[current_ball].keys())
                probabilities = list(red_global_probs[current_ball].values())

                # 过滤已使用的球
                filtered_candidates = []
                filtered_probs = []
                for i, candidate in enumerate(candidates):
                    if candidate not in used_balls:
                        filtered_candidates.append(candidate)
                        filtered_probs.append(probabilities[i])

                if filtered_candidates:
                    if use_max_prob:
                        # 选择概率最高的
                        max_idx = filtered_probs.index(max(filtered_probs))
                        next_ball = filtered_candidates[max_idx]
                    else:
                        # 基于概率分布随机选择
                        prob_sum = sum(filtered_probs)
                        normalized_probs = [p / prob_sum for p in filtered_probs]
                        next_ball = np.random.choice(filtered_candidates, p=normalized_probs)

                    if next_ball not in used_balls:
                        predicted_reds.append(next_ball)
                        used_balls.add(next_ball)

                        if explain:
                            print(f"全局转移: {current_ball:02d} -> {next_ball:02d} (概率: {red_global_probs[current_ball][next_ball]:.4f})")

        # 策略3: 如果还不够6个，使用频率分布补充
        while len(predicted_reds) < 6:
            # 按频率排序，选择高频且未使用的球
            available_balls = [(ball, freq) for ball, freq in red_freq.items() if ball not in used_balls]

            if available_balls:
                if use_max_prob:
                    # 选择频率最高的
                    available_balls.sort(key=lambda x: x[1], reverse=True)
                    next_ball = available_balls[0][0]
                else:
                    # 基于频率分布随机选择
                    balls, freqs = zip(*available_balls)
                    next_ball = np.random.choice(balls, p=np.array(freqs) / sum(freqs))

                predicted_reds.append(next_ball)
                used_balls.add(next_ball)

                if explain:
                    print(f"频率补充: {next_ball:02d} (频率: {red_freq[next_ball]:.4f})")
            else:
                # 如果所有球都用完了（理论上不应该发生），随机选择
                remaining_balls = [i for i in range(1, 34) if i not in used_balls]
                if remaining_balls:
                    next_ball = random.choice(remaining_balls)
                    predicted_reds.append(next_ball)
                    used_balls.add(next_ball)

        # 排序红球
        predicted_reds.sort()

        # 预测蓝球 - 使用转移概率
        if latest_blue in blue_probs and blue_probs[latest_blue]:
            candidates = list(blue_probs[latest_blue].keys())
            probabilities = list(blue_probs[latest_blue].values())

            if use_max_prob:
                # 选择概率最高的蓝球
                max_prob_index = probabilities.index(max(probabilities))
                predicted_blue = candidates[max_prob_index]

                if explain:
                    print(f"蓝球转移: {latest_blue:02d} -> {predicted_blue:02d} (概率: {max(probabilities):.4f})")
            else:
                # 基于概率分布随机选择
                predicted_blue = np.random.choice(candidates, p=probabilities)

                if explain:
                    print(f"蓝球转移: {latest_blue:02d} -> {predicted_blue:02d} (随机选择)")
        else:
            # 如果没有转移记录，使用全局蓝球频率分布
            blue_freq = {}
            for i in range(1, 17):
                blue_freq[i] = 0

            for _, row in self.data.iterrows():
                blue_freq[row['blue_ball']] += 1

            total_blue_count = sum(blue_freq.values())
            blue_freq = {ball: count / total_blue_count for ball, count in blue_freq.items()}

            if use_max_prob:
                # 选择频率最高的蓝球
                predicted_blue = max(blue_freq.items(), key=lambda x: x[1])[0]
            else:
                # 基于频率分布随机选择
                balls = list(blue_freq.keys())
                freqs = list(blue_freq.values())
                predicted_blue = np.random.choice(balls, p=freqs)

            if explain:
                print(f"蓝球频率: {predicted_blue:02d} (频率: {blue_freq[predicted_blue]:.4f})")

        return predicted_reds, predicted_blue

    def predict_multiple_by_markov_chain(self, count=1, explain=False):
        """使用马尔可夫链预测多注双色球号码"""
        print(f"使用马尔可夫链预测{count}注双色球号码...")

        # 确保马尔可夫链分析已完成
        if not hasattr(self, '_enhanced_markov_results') or self._enhanced_markov_results is None:
            self._enhanced_markov_results = self.analyze_markov_chain()

        predictions = []

        for i in range(count):
            if explain and count > 1:
                print(f"\n=== 第{i+1}注预测 ===")

            # 第一注使用最大概率，后续注数使用随机选择增加多样性
            use_max_prob = (i == 0)

            red_balls, blue_ball = self.predict_by_markov_chain(
                explain=explain if i == 0 else False,  # 只对第一注详细解释
                use_max_prob=use_max_prob
            )

            predictions.append((red_balls, blue_ball))

            if not explain and count > 1:
                formatted = self.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")

        return predictions

    def predict_by_markov_chain_with_periods(self, periods=None, count=1, explain=True):
        """基于指定期数的马尔可夫链稳定性概率预测"""
        if self.data is None:
            if not self.load_data():
                return []

        print(f"\n{'='*60}")
        print(f"基于马尔可夫链稳定性概率的指定期数预测")
        print(f"{'='*60}")

        # 如果指定了期数，使用指定期数的数据
        if periods:
            if len(self.data) < periods:
                print(f"警告: 可用数据({len(self.data)}期)少于指定期数({periods}期)，将使用全部数据")
                analysis_data = self.data.copy()
                actual_periods = len(analysis_data)
            else:
                analysis_data = self.data.head(periods).copy()
                actual_periods = periods
                print(f"使用最近{periods}期数据进行马尔可夫链稳定性分析")
        else:
            analysis_data = self.data.copy()
            actual_periods = len(analysis_data)
            print(f"使用全部{len(analysis_data)}期数据进行马尔可夫链稳定性分析")

        print(f"分析数据期数: {len(analysis_data)}期")
        print(f"数据范围: {analysis_data.iloc[-1]['issue']}期 - {analysis_data.iloc[0]['issue']}期")

        # 基于指定数据进行深度马尔可夫链稳定性分析
        print(f"\n开始基于{actual_periods}期数据的马尔可夫链稳定性分析...")
        markov_results = self._analyze_markov_chain_stability(analysis_data)

        # 获取最近一期的号码作为预测基础
        latest_data = analysis_data.iloc[0]
        latest_reds = [latest_data[f'red_{i}'] for i in range(1, 7)]
        latest_blue = latest_data['blue_ball']

        print(f"\n预测基础数据:")
        print(f"最近一期: {latest_data['issue']}期 ({latest_data['date']})")
        print(f"开奖号码: 红球 {' '.join([f'{ball:02d}' for ball in latest_reds])} | 蓝球 {latest_blue:02d}")

        # 进行多注预测，每注都基于稳定性概率
        predictions = []

        for i in range(count):
            print(f"\n{'='*50}")
            print(f"第{i+1}注基于{actual_periods}期稳定性概率预测")
            print(f"{'='*50}")

            predicted_reds, predicted_blue = self._predict_with_stability_analysis(
                markov_results, latest_reds, latest_blue, actual_periods, i+1, explain
            )

            predictions.append((predicted_reds, predicted_blue))

            formatted = self.format_numbers(predicted_reds, predicted_blue)
            print(f"\n第{i+1}注最稳定概率预测结果: {formatted}")

        # 显示预测汇总和稳定性分析
        print(f"\n{'='*60}")
        print(f"基于{actual_periods}期数据的稳定性预测汇总")
        print(f"{'='*60}")
        print(f"分析期数: {actual_periods}期")
        print(f"预测注数: {count}注")
        print(f"预测基础: {latest_data['issue']}期")
        print(f"预测方法: 马尔可夫链稳定性概率分析")

        for i, (red_balls, blue_ball) in enumerate(predictions):
            formatted = self.format_numbers(red_balls, blue_ball)
            print(f"第{i+1}注: {formatted}")

        return predictions

    def _predict_with_detailed_process(self, markov_results, latest_reds, latest_blue, use_max_prob, explain):
        """详细预测过程"""
        red_global_probs = markov_results['红球全局转移概率']
        red_position_probs = markov_results['红球位置转移概率']
        blue_probs = markov_results['蓝球转移概率']

        if explain:
            print(f"预测策略: {'最大概率选择' if use_max_prob else '概率分布随机选择'}")
            print(f"转移概率统计: 红球全局{len(red_global_probs)}个状态, 蓝球{len(blue_probs)}个状态")

        # 计算全局红球频率分布
        red_freq = {}
        for i in range(1, 34):
            red_freq[i] = 0

        for _, row in self.data.iterrows():
            for j in range(1, 7):
                red_freq[row[f'red_{j}']] += 1

        total_red_count = sum(red_freq.values())
        red_freq = {ball: count / total_red_count for ball, count in red_freq.items()}

        # 预测红球
        predicted_reds = []
        used_balls = set()
        prediction_details = []

        if explain:
            print(f"\n红球预测过程:")

        # 策略1: 基于位置转移概率预测前3个红球
        for pos in range(1, 4):
            current_ball = latest_reds[pos - 1]

            if pos in red_position_probs and current_ball in red_position_probs[pos]:
                candidates = list(red_position_probs[pos][current_ball].keys())
                probabilities = list(red_position_probs[pos][current_ball].values())

                # 过滤已使用的球
                filtered_candidates = []
                filtered_probs = []
                for i, candidate in enumerate(candidates):
                    if candidate not in used_balls:
                        filtered_candidates.append(candidate)
                        filtered_probs.append(probabilities[i])

                if filtered_candidates:
                    if use_max_prob:
                        max_idx = filtered_probs.index(max(filtered_probs))
                        next_ball = filtered_candidates[max_idx]
                        prob = filtered_probs[max_idx]
                    else:
                        prob_sum = sum(filtered_probs)
                        normalized_probs = [p / prob_sum for p in filtered_probs]
                        next_ball = np.random.choice(filtered_candidates, p=normalized_probs)
                        prob = red_position_probs[pos][current_ball][next_ball]

                    predicted_reds.append(next_ball)
                    used_balls.add(next_ball)

                    detail = f"位置{pos}: {current_ball:02d} -> {next_ball:02d} (概率: {prob:.4f}, 候选数: {len(filtered_candidates)})"
                    prediction_details.append(detail)

                    if explain:
                        print(f"  {detail}")

        # 策略2: 基于全局转移概率预测剩余红球
        if explain and len(predicted_reds) < 6:
            print(f"  继续使用全局转移概率预测剩余{6-len(predicted_reds)}个红球...")

        for current_ball in latest_reds:
            if len(predicted_reds) >= 6:
                break

            if current_ball in red_global_probs:
                candidates = list(red_global_probs[current_ball].keys())
                probabilities = list(red_global_probs[current_ball].values())

                # 过滤已使用的球
                filtered_candidates = []
                filtered_probs = []
                for i, candidate in enumerate(candidates):
                    if candidate not in used_balls:
                        filtered_candidates.append(candidate)
                        filtered_probs.append(probabilities[i])

                if filtered_candidates:
                    if use_max_prob:
                        max_idx = filtered_probs.index(max(filtered_probs))
                        next_ball = filtered_candidates[max_idx]
                        prob = filtered_probs[max_idx]
                    else:
                        prob_sum = sum(filtered_probs)
                        normalized_probs = [p / prob_sum for p in filtered_probs]
                        next_ball = np.random.choice(filtered_candidates, p=normalized_probs)
                        prob = red_global_probs[current_ball][next_ball]

                    if next_ball not in used_balls:
                        predicted_reds.append(next_ball)
                        used_balls.add(next_ball)

                        detail = f"全局转移: {current_ball:02d} -> {next_ball:02d} (概率: {prob:.4f}, 候选数: {len(filtered_candidates)})"
                        prediction_details.append(detail)

                        if explain:
                            print(f"  {detail}")

        # 策略3: 频率分布补充
        while len(predicted_reds) < 6:
            available_balls = [(ball, freq) for ball, freq in red_freq.items() if ball not in used_balls]

            if available_balls:
                if use_max_prob:
                    available_balls.sort(key=lambda x: x[1], reverse=True)
                    next_ball = available_balls[0][0]
                    freq = available_balls[0][1]
                else:
                    balls, freqs = zip(*available_balls)
                    next_ball = np.random.choice(balls, p=np.array(freqs) / sum(freqs))
                    freq = red_freq[next_ball]

                predicted_reds.append(next_ball)
                used_balls.add(next_ball)

                detail = f"频率补充: {next_ball:02d} (频率: {freq:.4f})"
                prediction_details.append(detail)

                if explain:
                    print(f"  {detail}")
            else:
                remaining_balls = [i for i in range(1, 34) if i not in used_balls]
                if remaining_balls:
                    next_ball = random.choice(remaining_balls)
                    predicted_reds.append(next_ball)
                    used_balls.add(next_ball)

                    if explain:
                        print(f"  随机补充: {next_ball:02d}")

        predicted_reds.sort()

        # 预测蓝球
        if explain:
            print(f"\n蓝球预测过程:")

        if latest_blue in blue_probs and blue_probs[latest_blue]:
            candidates = list(blue_probs[latest_blue].keys())
            probabilities = list(blue_probs[latest_blue].values())

            if use_max_prob:
                max_prob_index = probabilities.index(max(probabilities))
                predicted_blue = candidates[max_prob_index]
                prob = max(probabilities)

                if explain:
                    print(f"  蓝球转移: {latest_blue:02d} -> {predicted_blue:02d} (概率: {prob:.4f}, 候选数: {len(candidates)})")
            else:
                predicted_blue = np.random.choice(candidates, p=probabilities)
                prob = blue_probs[latest_blue][predicted_blue]

                if explain:
                    print(f"  蓝球转移: {latest_blue:02d} -> {predicted_blue:02d} (概率: {prob:.4f}, 随机选择)")
        else:
            # 使用全局蓝球频率分布
            blue_freq = {}
            for i in range(1, 17):
                blue_freq[i] = 0

            for _, row in self.data.iterrows():
                blue_freq[row['blue_ball']] += 1

            total_blue_count = sum(blue_freq.values())
            blue_freq = {ball: count / total_blue_count for ball, count in blue_freq.items()}

            if use_max_prob:
                predicted_blue = max(blue_freq.items(), key=lambda x: x[1])[0]
            else:
                balls = list(blue_freq.keys())
                freqs = list(blue_freq.values())
                predicted_blue = np.random.choice(balls, p=freqs)

            if explain:
                print(f"  蓝球频率: {predicted_blue:02d} (频率: {blue_freq[predicted_blue]:.4f})")

        # 组合特征验证
        if explain:
            current_odd_count = sum(1 for x in latest_reds if x % 2 == 1)
            predicted_odd_count = sum(1 for x in predicted_reds if x % 2 == 1)

            current_big_count = sum(1 for x in latest_reds if x >= 17)
            predicted_big_count = sum(1 for x in predicted_reds if x >= 17)

            current_sum = sum(latest_reds)
            predicted_sum = sum(predicted_reds)

            print(f"\n组合特征对比:")
            print(f"  奇偶比: {current_odd_count}:{6-current_odd_count} -> {predicted_odd_count}:{6-predicted_odd_count}")
            print(f"  大小比: {current_big_count}:{6-current_big_count} -> {predicted_big_count}:{6-predicted_big_count}")
            print(f"  和值: {current_sum} -> {predicted_sum}")
            print(f"  跨度: {max(latest_reds) - min(latest_reds)} -> {max(predicted_reds) - min(predicted_reds)}")

        return predicted_reds, predicted_blue

    def _predict_with_stability_analysis(self, stability_results, latest_reds, latest_blue, periods, prediction_num, explain):
        """基于稳定性分析的预测方法"""
        red_stability_probs = stability_results['红球稳定性转移概率']
        red_position_probs = stability_results['红球位置稳定性概率']
        blue_stability_probs = stability_results['蓝球稳定性转移概率']
        stability_stats = stability_results['稳定性统计']

        if explain:
            print(f"基于{periods}期数据的稳定性概率预测分析:")
            print(f"红球稳定性: {stability_stats['红球稳定性比例']:.1%}")
            print(f"蓝球稳定性: {stability_stats['蓝球稳定性比例']:.1%}")

        # 预测红球 - 基于稳定性概率
        predicted_reds = []
        used_balls = set()

        if explain:
            print(f"\n红球稳定性概率预测过程:")

        # 策略1: 基于位置稳定性概率预测前3个红球
        for pos in range(1, 4):
            current_ball = latest_reds[pos - 1]

            if pos in red_position_probs and current_ball in red_position_probs[pos]:
                candidates_info = red_position_probs[pos][current_ball]

                # 过滤已使用的球并按稳定性概率排序
                available_candidates = []
                for next_ball, info in candidates_info.items():
                    if next_ball not in used_balls:
                        available_candidates.append((next_ball, info))

                if available_candidates:
                    # 选择稳定性概率最高的球（第一注）或次高的球（后续注数）
                    available_candidates.sort(key=lambda x: x[1]['概率'], reverse=True)
                    choice_index = min(prediction_num - 1, len(available_candidates) - 1)
                    next_ball, info = available_candidates[choice_index]

                    predicted_reds.append(next_ball)
                    used_balls.add(next_ball)

                    if explain:
                        print(f"  位置{pos}: {current_ball:02d} -> {next_ball:02d}")
                        print(f"    稳定性概率: {info['概率']:.4f}")
                        print(f"    原始概率: {info['原始概率']:.4f}")
                        print(f"    出现次数: {info['出现次数']}/{info['总转移次数']}")
                        print(f"    稳定性权重: {info['稳定性权重']:.3f}")
                        print(f"    候选数量: {len(available_candidates)}")

        # 策略2: 基于全局稳定性概率预测剩余红球
        if explain and len(predicted_reds) < 6:
            print(f"  继续使用全局稳定性概率预测剩余{6-len(predicted_reds)}个红球...")

        for current_ball in latest_reds:
            if len(predicted_reds) >= 6:
                break

            if current_ball in red_stability_probs:
                candidates_info = red_stability_probs[current_ball]

                # 过滤已使用的球并按稳定性概率排序
                available_candidates = []
                for next_ball, info in candidates_info.items():
                    if next_ball not in used_balls:
                        available_candidates.append((next_ball, info))

                if available_candidates:
                    # 选择稳定性概率最高的球（第一注）或次高的球（后续注数）
                    available_candidates.sort(key=lambda x: x[1]['概率'], reverse=True)
                    choice_index = min(prediction_num - 1, len(available_candidates) - 1)
                    next_ball, info = available_candidates[choice_index]

                    if next_ball not in used_balls:
                        predicted_reds.append(next_ball)
                        used_balls.add(next_ball)

                        if explain:
                            print(f"  全局转移: {current_ball:02d} -> {next_ball:02d}")
                            print(f"    稳定性概率: {info['概率']:.4f}")
                            print(f"    原始概率: {info['原始概率']:.4f}")
                            print(f"    出现次数: {info['出现次数']}/{info['总转移次数']}")
                            print(f"    稳定性权重: {info['稳定性权重']:.3f}")
                            print(f"    候选数量: {len(available_candidates)}")

        # 策略3: 如果还不够6个，使用频率分布补充
        while len(predicted_reds) < 6:
            # 计算基于当前数据的频率分布
            red_freq = {}
            for i in range(1, 34):
                red_freq[i] = 0

            for _, row in self.data.iterrows():
                for j in range(1, 7):
                    red_freq[row[f'red_{j}']] += 1

            total_red_count = sum(red_freq.values())
            red_freq = {ball: count / total_red_count for ball, count in red_freq.items()}

            # 选择频率最高且未使用的球
            available_balls = [(ball, freq) for ball, freq in red_freq.items() if ball not in used_balls]

            if available_balls:
                available_balls.sort(key=lambda x: x[1], reverse=True)
                next_ball = available_balls[0][0]
                freq = available_balls[0][1]

                predicted_reds.append(next_ball)
                used_balls.add(next_ball)

                if explain:
                    print(f"  频率补充: {next_ball:02d} (频率: {freq:.4f})")
            else:
                # 随机补充
                remaining_balls = [i for i in range(1, 34) if i not in used_balls]
                if remaining_balls:
                    next_ball = random.choice(remaining_balls)
                    predicted_reds.append(next_ball)
                    used_balls.add(next_ball)

                    if explain:
                        print(f"  随机补充: {next_ball:02d}")

        predicted_reds.sort()

        # 预测蓝球 - 基于稳定性概率
        if explain:
            print(f"\n蓝球稳定性概率预测过程:")

        if latest_blue in blue_stability_probs:
            candidates_info = blue_stability_probs[latest_blue]

            # 选择稳定性概率最高的蓝球（第一注）或次高的球（后续注数）
            sorted_candidates = sorted(candidates_info.items(), key=lambda x: x[1]['概率'], reverse=True)
            choice_index = min(prediction_num - 1, len(sorted_candidates) - 1)
            predicted_blue, info = sorted_candidates[choice_index]

            if explain:
                print(f"  蓝球转移: {latest_blue:02d} -> {predicted_blue:02d}")
                print(f"    稳定性概率: {info['概率']:.4f}")
                print(f"    原始概率: {info['原始概率']:.4f}")
                print(f"    出现次数: {info['出现次数']}/{info['总转移次数']}")
                print(f"    稳定性权重: {info['稳定性权重']:.3f}")
                print(f"    候选数量: {len(candidates_info)}")
        else:
            # 使用频率分布
            blue_freq = {}
            for i in range(1, 17):
                blue_freq[i] = 0

            for _, row in self.data.iterrows():
                blue_freq[row['blue_ball']] += 1

            total_blue_count = sum(blue_freq.values())
            blue_freq = {ball: count / total_blue_count for ball, count in blue_freq.items()}

            predicted_blue = max(blue_freq.items(), key=lambda x: x[1])[0]

            if explain:
                print(f"  蓝球频率: {predicted_blue:02d} (频率: {blue_freq[predicted_blue]:.4f})")

        # 稳定性验证和组合特征分析
        if explain:
            print(f"\n稳定性验证和组合特征分析:")

            # 计算预测结果的稳定性得分
            red_stability_score = 0
            for ball in predicted_reds:
                for current_ball in latest_reds:
                    if current_ball in red_stability_probs and ball in red_stability_probs[current_ball]:
                        red_stability_score += red_stability_probs[current_ball][ball]['稳定性权重']

            red_stability_score /= len(predicted_reds)

            blue_stability_score = 0
            if latest_blue in blue_stability_probs and predicted_blue in blue_stability_probs[latest_blue]:
                blue_stability_score = blue_stability_probs[latest_blue][predicted_blue]['稳定性权重']

            print(f"  红球平均稳定性得分: {red_stability_score:.3f}")
            print(f"  蓝球稳定性得分: {blue_stability_score:.3f}")

            # 组合特征对比
            current_odd_count = sum(1 for x in latest_reds if x % 2 == 1)
            predicted_odd_count = sum(1 for x in predicted_reds if x % 2 == 1)

            current_big_count = sum(1 for x in latest_reds if x >= 17)
            predicted_big_count = sum(1 for x in predicted_reds if x >= 17)

            current_sum = sum(latest_reds)
            predicted_sum = sum(predicted_reds)

            print(f"  奇偶比: {current_odd_count}:{6-current_odd_count} -> {predicted_odd_count}:{6-predicted_odd_count}")
            print(f"  大小比: {current_big_count}:{6-current_big_count} -> {predicted_big_count}:{6-predicted_big_count}")
            print(f"  和值: {current_sum} -> {predicted_sum}")
            print(f"  跨度: {max(latest_reds) - min(latest_reds)} -> {max(predicted_reds) - min(predicted_reds)}")

        return predicted_reds, predicted_blue

    def analyze_markov_prediction_accuracy(self, test_periods=50):
        """分析马尔可夫链预测的准确性"""
        print(f"分析马尔可夫链预测准确性（回测{test_periods}期）...")

        if len(self.data) < test_periods + 100:
            print("数据量不足，无法进行准确性分析")
            return None

        # 保存原始数据
        original_data = self.data.copy()

        results = {
            '红球命中统计': {'0个': 0, '1个': 0, '2个': 0, '3个': 0, '4个': 0, '5个': 0, '6个': 0},
            '蓝球命中统计': {'命中': 0, '未命中': 0},
            '详细结果': []
        }

        for i in range(test_periods):
            # 使用前面的数据进行训练
            train_data = original_data.iloc[i+1:].reset_index(drop=True)
            self.data = train_data

            # 重新分析马尔可夫链（简化版，不生成可视化）
            self._enhanced_markov_results = self._analyze_markov_chain_simple(train_data)

            # 预测
            predicted_reds, predicted_blue = self.predict_by_markov_chain(explain=False, use_max_prob=True)

            # 获取实际结果
            actual_data = original_data.iloc[i]
            actual_reds = [actual_data[f'red_{j}'] for j in range(1, 7)]
            actual_blue = actual_data['blue_ball']

            # 计算命中情况
            red_hits = len(set(predicted_reds) & set(actual_reds))
            blue_hit = predicted_blue == actual_blue

            # 统计
            results['红球命中统计'][f'{red_hits}个'] += 1
            results['蓝球命中统计']['命中' if blue_hit else '未命中'] += 1

            results['详细结果'].append({
                '期号': actual_data['issue'],
                '预测红球': predicted_reds,
                '实际红球': actual_reds,
                '红球命中': red_hits,
                '预测蓝球': predicted_blue,
                '实际蓝球': actual_blue,
                '蓝球命中': blue_hit
            })

        # 恢复原始数据
        self.data = original_data

        # 计算准确率
        total_tests = test_periods
        red_accuracy = {
            '至少1个': sum(results['红球命中统计'][f'{i}个'] for i in range(1, 7)) / total_tests,
            '至少2个': sum(results['红球命中统计'][f'{i}个'] for i in range(2, 7)) / total_tests,
            '至少3个': sum(results['红球命中统计'][f'{i}个'] for i in range(3, 7)) / total_tests,
            '6个全中': results['红球命中统计']['6个'] / total_tests
        }

        blue_accuracy = results['蓝球命中统计']['命中'] / total_tests

        results['准确率统计'] = {
            '红球准确率': red_accuracy,
            '蓝球准确率': blue_accuracy
        }

        # 保存结果
        output_file = os.path.join(self.data_dir, "advanced", "markov_accuracy_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4, default=str)

        # 打印结果
        print("\n=== 马尔可夫链预测准确性分析结果 ===")
        print(f"测试期数: {total_tests}")
        print("\n红球命中分布:")
        for hits, count in results['红球命中统计'].items():
            percentage = count / total_tests * 100
            print(f"  {hits}: {count}次 ({percentage:.1f}%)")

        print(f"\n蓝球命中率: {blue_accuracy:.1%}")

        print("\n红球准确率:")
        for desc, rate in red_accuracy.items():
            print(f"  {desc}: {rate:.1%}")

        print(f"准确性分析结果已保存到 {output_file}")

        return results

    def predict_by_ensemble(self, explain=False):
        """集成方法预测"""
        if self.data is None:
            if not self.load_data():
                return None, None

        print("使用集成方法预测...")

        # 收集多种预测方法的结果
        predictions = {}

        # 马尔可夫链预测
        try:
            red_balls, blue_ball = self.predict_by_markov_chain(explain=False, use_max_prob=True)
            predictions['markov'] = (red_balls, blue_ball)
        except:
            pass

        # 频率预测
        try:
            red_freq, blue_freq = self.analyze_number_frequency()

            # 选择频率最高的6个红球
            sorted_red = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)
            freq_reds = sorted([ball for ball, freq in sorted_red[:6]])

            # 选择频率最高的蓝球
            freq_blue = max(blue_freq.items(), key=lambda x: x[1])[0]

            predictions['frequency'] = (freq_reds, freq_blue)
        except:
            pass

        # 随机预测
        predictions['random'] = self.generate_random_numbers()

        if not predictions:
            return self.generate_random_numbers()

        # 红球投票
        red_votes = {}
        for method, (reds, blue) in predictions.items():
            for red in reds:
                if red not in red_votes:
                    red_votes[red] = 0
                red_votes[red] += 1

        # 选择得票最多的6个红球
        sorted_reds = sorted(red_votes.items(), key=lambda x: x[1], reverse=True)
        final_reds = sorted([red for red, votes in sorted_reds[:6]])

        # 蓝球投票
        blue_votes = {}
        for method, (reds, blue) in predictions.items():
            if blue not in blue_votes:
                blue_votes[blue] = 0
            blue_votes[blue] += 1

        final_blue = max(blue_votes.items(), key=lambda x: x[1])[0]

        if explain:
            print(f"集成预测结果: 红球 {' '.join([f'{ball:02d}' for ball in final_reds])} | 蓝球 {final_blue:02d}")
            print("参与投票的方法:")
            for method, (reds, blue) in predictions.items():
                formatted = self.format_numbers(reds, blue)
                print(f"  {method}: {formatted}")

        return final_reds, final_blue

    def predict_by_stats(self, explain=False):
        """基于统计学预测"""
        if self.data is None:
            if not self.load_data():
                return self.generate_random_numbers()

        try:
            # 计算历史统计特征
            red_sums = self.data['red_sum']
            red_variances = self.data['red_variance']
            red_spans = self.data['red_span']

            # 目标统计值（使用历史平均值）
            target_sum = int(red_sums.mean())
            target_variance = red_variances.mean()
            target_span = int(red_spans.mean())

            # 生成符合统计特征的红球组合
            attempts = 0
            max_attempts = 1000

            while attempts < max_attempts:
                red_balls = sorted(random.sample(range(1, 34), 6))

                # 检查是否符合统计特征
                current_sum = sum(red_balls)
                current_variance = np.var(red_balls)
                current_span = max(red_balls) - min(red_balls)

                # 允许一定的误差范围
                sum_diff = abs(current_sum - target_sum)
                variance_diff = abs(current_variance - target_variance)
                span_diff = abs(current_span - target_span)

                if sum_diff <= 10 and variance_diff <= 20 and span_diff <= 5:
                    break

                attempts += 1

            # 蓝球使用频率最高的
            blue_freq = {}
            for i in range(1, 17):
                blue_freq[i] = 0

            for _, row in self.data.iterrows():
                blue_freq[row['blue_ball']] += 1

            predicted_blue = max(blue_freq.items(), key=lambda x: x[1])[0]

            if explain:
                print(f"统计学预测: 目标和值={target_sum}, 实际和值={sum(red_balls)}")
                print(f"目标方差={target_variance:.2f}, 实际方差={np.var(red_balls):.2f}")
                print(f"目标跨度={target_span}, 实际跨度={max(red_balls) - min(red_balls)}")

            return red_balls, predicted_blue

        except Exception as e:
            if explain:
                print(f"统计学预测失败: {e}")
            return self.generate_random_numbers()

    def predict_by_probability(self, explain=False):
        """基于概率论预测"""
        if self.data is None:
            if not self.load_data():
                return self.generate_random_numbers()

        try:
            # 计算历史概率分布
            red_freq, blue_freq = self.analyze_number_frequency()

            # 使用加权随机选择
            red_numbers = list(red_freq.keys())
            red_weights = list(red_freq.values())

            # 选择6个不重复的红球
            selected_reds = []
            available_numbers = red_numbers.copy()
            available_weights = red_weights.copy()

            for _ in range(6):
                if not available_numbers:
                    break

                # 归一化权重
                total_weight = sum(available_weights)
                normalized_weights = [w / total_weight for w in available_weights]

                # 加权随机选择
                selected = np.random.choice(available_numbers, p=normalized_weights)
                selected_reds.append(selected)

                # 移除已选择的号码
                idx = available_numbers.index(selected)
                available_numbers.pop(idx)
                available_weights.pop(idx)

            selected_reds.sort()

            # 蓝球概率选择
            blue_numbers = list(blue_freq.keys())
            blue_weights = list(blue_freq.values())
            blue_weights = [w / sum(blue_weights) for w in blue_weights]

            selected_blue = np.random.choice(blue_numbers, p=blue_weights)

            if explain:
                print("概率论预测: 基于历史频率分布进行加权随机选择")
                for i, ball in enumerate(selected_reds):
                    print(f"红球{i+1}: {ball} (概率: {red_freq[ball]:.4f})")
                print(f"蓝球: {selected_blue} (概率: {blue_freq[selected_blue]:.4f})")

            return selected_reds, selected_blue

        except Exception as e:
            if explain:
                print(f"概率论预测失败: {e}")
            return self.generate_random_numbers()

    def predict_by_decision_tree_advanced(self, explain=False):
        """基于决策树预测（高级版本）"""
        if self.data is None:
            if not self.load_data():
                return self.generate_random_numbers()

        try:
            # 准备特征数据
            features = []
            red_targets = [[] for _ in range(6)]
            blue_targets = []

            # 使用前5期的数据作为特征
            n_prev = 5
            for i in range(n_prev, len(self.data)):
                feature_row = []

                # 提取前n期的特征
                for j in range(n_prev):
                    idx = i - n_prev + j
                    feature_row.extend([
                        self.data.iloc[idx]['red_sum'],
                        self.data.iloc[idx]['red_variance'],
                        self.data.iloc[idx]['red_span'],
                        self.data.iloc[idx]['blue_ball']
                    ])

                features.append(feature_row)

                # 目标值
                for k in range(6):
                    red_targets[k].append(self.data.iloc[i][f'red_{k+1}'])
                blue_targets.append(self.data.iloc[i]['blue_ball'])

            X = np.array(features)

            # 训练6个红球位置的模型
            red_predictions = []
            for pos in range(6):
                y = np.array(red_targets[pos])
                rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
                rf.fit(X, y)

                # 预测
                latest_features = features[-1] if features else [0] * (n_prev * 4)
                pred = rf.predict([latest_features])[0]
                red_predictions.append(pred)

            # 去重并补充
            unique_reds = list(set(red_predictions))
            while len(unique_reds) < 6:
                # 随机添加未选中的号码
                available = [i for i in range(1, 34) if i not in unique_reds]
                if available:
                    unique_reds.append(random.choice(available))
                else:
                    break

            # 如果超过6个，随机选择6个
            if len(unique_reds) > 6:
                unique_reds = random.sample(unique_reds, 6)

            unique_reds.sort()

            # 蓝球预测
            y_blue = np.array(blue_targets)
            rf_blue = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
            rf_blue.fit(X, y_blue)

            latest_features = features[-1] if features else [0] * (n_prev * 4)
            predicted_blue = rf_blue.predict([latest_features])[0]

            if explain:
                print("决策树预测: 使用随机森林模型基于历史特征预测")
                print(f"原始红球预测: {red_predictions}")
                print(f"去重后红球: {unique_reds}")
                print(f"蓝球预测: {predicted_blue}")

            return unique_reds, predicted_blue

        except Exception as e:
            if explain:
                print(f"决策树预测失败: {e}")
            return self.generate_random_numbers()

    def predict_based_on_patterns(self, explain=False):
        """基于模式识别预测"""
        if self.data is None:
            if not self.load_data():
                return self.generate_random_numbers()

        try:
            # 分析最近的模式
            recent_data = self.data.head(20)  # 最近20期

            # 分析奇偶模式
            odd_even_patterns = []
            for _, row in recent_data.iterrows():
                reds = [row[f'red_{i}'] for i in range(1, 7)]
                odd_count = sum(1 for x in reds if x % 2 == 1)
                odd_even_patterns.append(odd_count)

            # 预测下期奇偶比
            avg_odd = np.mean(odd_even_patterns)
            target_odd = int(round(avg_odd))
            target_odd = max(1, min(5, target_odd))  # 限制在1-5之间

            # 分析大小比模式
            big_small_patterns = []
            for _, row in recent_data.iterrows():
                reds = [row[f'red_{i}'] for i in range(1, 7)]
                big_count = sum(1 for x in reds if x >= 17)
                big_small_patterns.append(big_count)

            avg_big = np.mean(big_small_patterns)
            target_big = int(round(avg_big))
            target_big = max(1, min(5, target_big))

            # 生成符合模式的号码
            attempts = 0
            max_attempts = 1000

            while attempts < max_attempts:
                red_balls = sorted(random.sample(range(1, 34), 6))

                current_odd = sum(1 for x in red_balls if x % 2 == 1)
                current_big = sum(1 for x in red_balls if x >= 17)

                if abs(current_odd - target_odd) <= 1 and abs(current_big - target_big) <= 1:
                    break

                attempts += 1

            # 蓝球基于最近趋势
            recent_blues = recent_data['blue_ball'].tolist()
            blue_freq = Counter(recent_blues)

            if blue_freq:
                # 选择最近出现频率较高的蓝球
                predicted_blue = blue_freq.most_common(1)[0][0]
            else:
                predicted_blue = random.randint(1, 16)

            if explain:
                print(f"模式识别预测: 目标奇数个数={target_odd}, 实际={sum(1 for x in red_balls if x % 2 == 1)}")
                print(f"目标大数个数={target_big}, 实际={sum(1 for x in red_balls if x >= 17)}")
                print(f"最近蓝球趋势: {dict(blue_freq.most_common(3))}")

            return red_balls, predicted_blue

        except Exception as e:
            if explain:
                print(f"模式识别预测失败: {e}")
            return self.generate_random_numbers()

    # ==================== 工具函数 ====================

    def generate_random_numbers(self):
        """生成随机双色球号码"""
        # 生成6个不重复的红球号码（1-33）
        red_balls = sorted(random.sample(range(1, 34), 6))
        # 生成1个蓝球号码（1-16）
        blue_ball = random.randint(1, 16)

        return red_balls, blue_ball

    def generate_smart_numbers(self, method="frequency"):
        """根据历史数据生成智能双色球号码"""
        if self.data is None:
            if not self.load_data():
                return self.generate_random_numbers()

        try:
            if method == "frequency":
                # 基于频率生成
                red_freq, blue_freq = self.analyze_number_frequency()

                # 选择频率较高的红球
                sorted_red = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)
                high_freq_reds = [ball for ball, freq in sorted_red[:12]]  # 取前12个高频球
                selected_reds = sorted(random.sample(high_freq_reds, 6))

                # 选择频率较高的蓝球
                sorted_blue = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)
                high_freq_blues = [ball for ball, freq in sorted_blue[:8]]  # 取前8个高频球
                selected_blue = random.choice(high_freq_blues)

                return selected_reds, selected_blue

            elif method == "trend":
                # 基于走势生成
                recent_data = self.data.head(10)  # 最近10期

                # 统计最近出现的红球
                recent_reds = set()
                for _, row in recent_data.iterrows():
                    for i in range(1, 7):
                        recent_reds.add(row[f'red_{i}'])

                # 选择一些最近出现的球和一些冷门球
                hot_balls = list(recent_reds)
                cold_balls = [i for i in range(1, 34) if i not in recent_reds]

                selected_reds = []
                if len(hot_balls) >= 3:
                    selected_reds.extend(random.sample(hot_balls, 3))
                if len(cold_balls) >= 3:
                    selected_reds.extend(random.sample(cold_balls, 3))

                while len(selected_reds) < 6:
                    remaining = [i for i in range(1, 34) if i not in selected_reds]
                    selected_reds.append(random.choice(remaining))

                selected_reds.sort()

                # 蓝球随机选择
                selected_blue = random.randint(1, 16)

                return selected_reds, selected_blue

            else:  # hybrid
                # 混合策略
                if random.random() < 0.5:
                    return self.generate_smart_numbers("frequency")
                else:
                    return self.generate_smart_numbers("trend")

        except Exception as e:
            print(f"生成智能号码失败: {e}")
            return self.generate_random_numbers()

    def format_numbers(self, red_balls, blue_ball):
        """格式化双色球号码"""
        red_str = " ".join([f"{ball:02d}" for ball in red_balls])
        blue_str = f"{blue_ball:02d}"

        return f"红球: {red_str} | 蓝球: {blue_str}"

    def calculate_prize(self, predicted_reds, predicted_blue, winning_reds, winning_blue):
        """计算中奖等级"""
        # 计算红球命中数
        red_hits = len(set(predicted_reds) & set(winning_reds))

        # 计算蓝球是否命中
        blue_hit = predicted_blue == winning_blue

        # 判断中奖等级
        if red_hits == 6 and blue_hit:
            return "一"
        elif red_hits == 6:
            return "二"
        elif red_hits == 5 and blue_hit:
            return "三"
        elif red_hits == 5 or (red_hits == 4 and blue_hit):
            return "四"
        elif red_hits == 4 or (red_hits == 3 and blue_hit):
            return "五"
        elif blue_hit:
            return "六"
        else:
            return None

    def get_latest_draw(self, real_time=False):
        """获取最新开奖结果"""
        if real_time:
            try:
                # 尝试从网络获取最新开奖结果
                response = requests.get(
                    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice",
                    params={"name": "ssq", "pageNo": 1, "pageSize": 1, "systemType": "PC"},
                    headers=self.headers,
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("state") == 0 and "result" in data and data["result"]:
                        item = data["result"][0]
                        issue = item["code"]
                        date = item["date"]
                        red_balls = [int(ball) for ball in item["red"].split(",")]
                        blue_ball = int(item["blue"])

                        return issue, date, red_balls, blue_ball
            except Exception as e:
                print(f"实时获取最新开奖结果失败: {e}")

        # 从本地文件读取
        if self.data is not None and len(self.data) > 0:
            latest = self.data.iloc[0]

            issue = latest["issue"]
            date = latest["date"]
            red_balls = [latest[f'red_{i}'] for i in range(1, 7)]
            blue_ball = latest["blue_ball"]

            return issue, date, red_balls, blue_ball

        return None, None, None, None

    # ==================== 高级分析运行器 ====================

    def run_advanced_analysis(self, method="all"):
        """运行高级分析"""
        if not self.load_data():
            return False

        print("开始高级分析...")

        if method == "all":
            # 运行所有分析
            self.analyze_statistical_features()
            self.analyze_probability_distribution()
            self.analyze_frequency_patterns()
            self.analyze_decision_tree()
            self.analyze_cycle_patterns()
            self.analyze_historical_correlation()
            self.analyze_issue_number_correlation()
            self.analyze_markov_chain()

            # 如果PyMC可用，运行贝叶斯分析
            if PYMC_AVAILABLE:
                try:
                    self.analyze_bayesian()
                except Exception as e:
                    print(f"贝叶斯分析失败: {e}")

        elif method == "stats":
            self.analyze_statistical_features()
        elif method == "probability":
            self.analyze_probability_distribution()
        elif method == "frequency":
            self.analyze_frequency_patterns()
        elif method == "decision_tree":
            self.analyze_decision_tree()
        elif method == "cycle":
            self.analyze_cycle_patterns()
        elif method == "correlation":
            self.analyze_historical_correlation()
        elif method == "issue_correlation":
            self.analyze_issue_number_correlation()
        elif method == "markov":
            self.analyze_markov_chain()
        elif method == "bayes":
            if PYMC_AVAILABLE:
                try:
                    self.analyze_bayesian()
                except Exception as e:
                    print(f"贝叶斯分析失败: {e}")
            else:
                print("PyMC未安装，无法进行贝叶斯分析")

        print("高级分析完成！")
        return True

    def analyze_bayesian(self):
        """贝叶斯分析（可选功能）"""
        if not PYMC_AVAILABLE:
            print("PyMC未安装，无法进行贝叶斯分析")
            return None

        print("进行贝叶斯分析...")

        try:
            # 简化的贝叶斯分析
            blue_balls = self.data['blue_ball'].values

            with pm.Model() as model:
                # 先验分布：假设每个蓝球的概率服从Beta分布
                alpha = pm.Exponential('alpha', 1.0)
                beta = pm.Exponential('beta', 1.0)

                # 蓝球概率
                blue_probs = pm.Beta('blue_probs', alpha=alpha, beta=beta, shape=16)

                # 观测数据
                blue_obs = pm.Categorical('blue_obs', p=blue_probs, observed=blue_balls - 1)

                # MCMC采样
                trace = pm.sample(1000, tune=500, return_inferencedata=True, progressbar=False)

            # 计算后验统计
            posterior_means = trace.posterior['blue_probs'].mean(dim=['chain', 'draw'])

            bayes_results = {
                '蓝球后验概率': {i+1: float(posterior_means[i]) for i in range(16)}
            }

            # 保存结果
            output_file = os.path.join(self.data_dir, "advanced", "bayesian_analysis.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(bayes_results, f, ensure_ascii=False, indent=4)

            print(f"贝叶斯分析结果已保存到 {output_file}")
            return bayes_results

        except Exception as e:
            print(f"贝叶斯分析失败: {e}")
            return None


def main():
    """主函数 - 命令行界面"""
    parser = argparse.ArgumentParser(description='双色球数据分析与预测系统')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 爬取数据命令
    crawl_parser = subparsers.add_parser('crawl', help='爬取双色球历史数据')
    crawl_parser.add_argument('--count', type=int, default=300, help='爬取期数，默认300期')
    crawl_parser.add_argument('--all', action='store_true', help='爬取所有历史数据')
    crawl_parser.add_argument('--start', type=str, help='起始期号（如：2025001）')
    crawl_parser.add_argument('--end', type=str, help='结束期号（如：2025010）')
    crawl_parser.add_argument('--append', action='store_true', help='追加到现有CSV文件')

    # 基础分析命令
    analyze_parser = subparsers.add_parser('analyze', help='运行基础分析')

    # 高级分析命令
    advanced_parser = subparsers.add_parser('advanced', help='运行高级分析')
    advanced_parser.add_argument('--method', choices=['all', 'stats', 'probability', 'frequency', 'decision_tree', 'cycle', 'correlation', 'issue_correlation', 'markov', 'bayes'],
                                default='all', help='分析方法')

    # 马尔可夫链预测命令
    markov_parser = subparsers.add_parser('markov_predict', help='使用马尔可夫链预测')
    markov_parser.add_argument('--count', type=int, default=1, help='预测注数')
    markov_parser.add_argument('--explain', action='store_true', help='显示预测过程')
    markov_parser.add_argument('--use-all-data', action='store_true', help='使用所有历史数据')
    markov_parser.add_argument('--analyze-accuracy', action='store_true', help='分析预测准确性')
    markov_parser.add_argument('--periods', type=int, help='指定分析期数（如：100表示使用最近100期数据）')

    # 集成预测命令
    predict_parser = subparsers.add_parser('predict', help='使用各种方法预测')
    predict_parser.add_argument('--method', choices=['ensemble', 'markov', 'stats', 'probability', 'decision_tree', 'patterns'], default='ensemble', help='预测方法')
    predict_parser.add_argument('--count', type=int, default=1, help='预测注数')
    predict_parser.add_argument('--explain', action='store_true', help='显示预测过程')

    # 生成号码命令
    generate_parser = subparsers.add_parser('generate', help='生成号码')
    generate_parser.add_argument('--method', choices=['random', 'frequency', 'trend', 'hybrid'], default='hybrid', help='生成方法')
    generate_parser.add_argument('--count', type=int, default=1, help='生成注数')

    # 查看最新开奖命令
    latest_parser = subparsers.add_parser('latest', help='查看最新开奖结果')
    latest_parser.add_argument('--real-time', action='store_true', help='实时获取最新结果')

    # 数据验证命令
    validate_parser = subparsers.add_parser('validate', help='验证数据文件')

    # 获取最新开奖命令
    fetch_parser = subparsers.add_parser('fetch_latest', help='获取最新一期开奖结果并追加到文件')
    fetch_parser.add_argument('--file', type=str, default='ssq_data.csv', help='目标CSV文件名')

    # 追加数据命令
    append_parser = subparsers.add_parser('append', help='爬取指定数据并追加到文件')
    append_parser.add_argument('--count', type=int, help='追加最新N期数据')
    append_parser.add_argument('--start', type=str, help='起始期号')
    append_parser.add_argument('--end', type=str, help='结束期号')
    append_parser.add_argument('--file', type=str, default='ssq_data.csv', help='目标CSV文件名')

    args = parser.parse_args()

    # 创建分析器实例
    analyzer = SSQAnalyzer()

    if args.command == 'crawl':
        # 爬取数据
        if args.start and args.end:
            # 爬取指定期数范围
            results = analyzer.crawl_specific_periods(start_issue=args.start, end_issue=args.end)
            if results:
                if args.append:
                    filename = "ssq_data_all.csv" if args.all else "ssq_data.csv"
                    analyzer.append_to_csv(results, filename)
                else:
                    filename = "ssq_data_all.csv" if args.all else "ssq_data.csv"
                    analyzer.save_to_csv(results, filename)
                print("指定期数数据爬取完成")
            else:
                print("指定期数数据爬取失败")
        else:
            # 原有的爬取逻辑
            success = analyzer.crawl_data(count=None if args.all else args.count, use_all_data=args.all)
            if success:
                print("数据爬取完成")
            else:
                print("数据爬取失败")

    elif args.command == 'analyze':
        # 基础分析
        analyzer.run_basic_analysis()

    elif args.command == 'advanced':
        # 高级分析
        analyzer.run_advanced_analysis(method=args.method)

    elif args.command == 'markov_predict':
        # 马尔可夫链预测
        if not analyzer.load_data():
            print("加载数据失败")
            return

        # 如果指定了期数，使用指定期数预测
        if args.periods:
            predictions = analyzer.predict_by_markov_chain_with_periods(
                periods=args.periods,
                count=args.count,
                explain=args.explain
            )
        else:
            predictions = analyzer.predict_multiple_by_markov_chain(count=args.count, explain=args.explain)

            print(f"\n=== 马尔可夫链预测结果 ===")
            for i, (red_balls, blue_ball) in enumerate(predictions):
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")

        # 准确性分析
        if args.analyze_accuracy:
            print("\n开始准确性分析...")
            accuracy_results = analyzer.analyze_markov_prediction_accuracy(test_periods=30)
            if accuracy_results:
                print("准确性分析完成")

    elif args.command == 'predict':
        # 集成预测
        if not analyzer.load_data():
            print("加载数据失败")
            return

        if args.method == 'ensemble':
            for i in range(args.count):
                red_balls, blue_ball = analyzer.predict_by_ensemble(explain=args.explain and i == 0)
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")
        elif args.method == 'markov':
            predictions = analyzer.predict_multiple_by_markov_chain(count=args.count, explain=args.explain)
            for i, (red_balls, blue_ball) in enumerate(predictions):
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")
        elif args.method == 'stats':
            for i in range(args.count):
                red_balls, blue_ball = analyzer.predict_by_stats(explain=args.explain and i == 0)
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")
        elif args.method == 'probability':
            for i in range(args.count):
                red_balls, blue_ball = analyzer.predict_by_probability(explain=args.explain and i == 0)
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")
        elif args.method == 'decision_tree':
            for i in range(args.count):
                red_balls, blue_ball = analyzer.predict_by_decision_tree_advanced(explain=args.explain and i == 0)
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")
        elif args.method == 'patterns':
            for i in range(args.count):
                red_balls, blue_ball = analyzer.predict_based_on_patterns(explain=args.explain and i == 0)
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")

    elif args.command == 'generate':
        # 生成号码
        for i in range(args.count):
            if args.method == 'random':
                red_balls, blue_ball = analyzer.generate_random_numbers()
            else:
                red_balls, blue_ball = analyzer.generate_smart_numbers(args.method)

            formatted = analyzer.format_numbers(red_balls, blue_ball)
            print(f"第{i+1}注: {formatted}")

    elif args.command == 'latest':
        # 查看最新开奖
        issue, date, red_balls, blue_ball = analyzer.get_latest_draw(real_time=args.real_time)
        if issue:
            formatted = analyzer.format_numbers(red_balls, blue_ball)
            print(f"最新开奖({issue}期): {formatted}")
            print(f"开奖日期: {date}")
        else:
            print("获取最新开奖结果失败")

    elif args.command == 'validate':
        # 验证数据
        analyzer.validate_data()

    elif args.command == 'fetch_latest':
        # 获取最新开奖并追加
        result = analyzer.fetch_and_append_latest(filename=args.file)
        if result:
            print("最新开奖数据获取并追加成功")
        else:
            print("获取最新开奖数据失败")

    elif args.command == 'append':
        # 追加指定数据
        if args.start and args.end:
            results = analyzer.crawl_specific_periods(start_issue=args.start, end_issue=args.end)
        elif args.count:
            results = analyzer.crawl_specific_periods(count=args.count)
        else:
            print("请指定 --count 或 --start 和 --end 参数")
            return

        if results:
            analyzer.append_to_csv(results, args.file)
            print("数据追加完成")
        else:
            print("数据追加失败")

    else:
        # 显示帮助信息
        parser.print_help()


if __name__ == "__main__":
    main()
