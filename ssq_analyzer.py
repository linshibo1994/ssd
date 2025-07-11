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
from sklearn.decomposition import PCA
from scipy import stats

# 可选依赖
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

# 深度学习相关导入
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.preprocessing import MinMaxScaler
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

        if results:
            # 按期号排序（最新的在前）
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            if count is not None and len(results) > count:
                results = results[:count]

        return results
    
    def crawl_data_from_zhcw(self, max_pages=200):
        """从中彩网获取补充数据"""
        results = {}

        try:
            print(f"正在从中彩网获取补充数据（最多{max_pages}页）...")

            consecutive_empty_pages = 0
            max_consecutive_empty = 5

            for page in range(1, max_pages + 1):
                try:
                    url = f"http://kaijiang.zhcw.com/zhcw/html/ssq/list_{page}.html"

                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Referer": "http://kaijiang.zhcw.com/",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                    }

                    response = requests.get(url, headers=headers, timeout=15)
                    response.encoding = 'utf-8'

                    soup = BeautifulSoup(response.text, "html.parser")
                    table = soup.find('table', attrs={'class': 'wqhgt'})

                    if not table:
                        consecutive_empty_pages += 1
                        print(f"第{page}页未找到数据表格，连续空页数: {consecutive_empty_pages}")
                        if consecutive_empty_pages >= max_consecutive_empty:
                            break
                        continue

                    rows = table.find_all('tr')
                    if len(rows) <= 1:
                        consecutive_empty_pages += 1
                        print(f"第{page}页无数据行，连续空页数: {consecutive_empty_pages}")
                        if consecutive_empty_pages >= max_consecutive_empty:
                            break
                        continue

                    page_count = 0
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
                            page_count += 1
                        except Exception as e:
                            continue

                    if page_count > 0:
                        consecutive_empty_pages = 0
                        print(f"第{page}页获取到{page_count}期数据，累计{len(results)}期")
                    else:
                        consecutive_empty_pages += 1
                        print(f"第{page}页无有效数据，连续空页数: {consecutive_empty_pages}")
                        if consecutive_empty_pages >= max_consecutive_empty:
                            break

                    time.sleep(random.uniform(1, 3))

                except Exception as e:
                    print(f"第{page}页爬取失败: {e}")
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= max_consecutive_empty:
                        break
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
            print("开始爬取所有历史双色球数据...")
        else:
            filename = "ssq_data.csv"
            print(f"开始爬取最近{count}期双色球数据...")

        # 从官方网站获取数据
        print("\n=== 第一阶段：从中国福利彩票官方网站获取数据 ===")
        results = self.crawl_data_from_cwl(count)

        # 如果是获取所有数据，或者数据不足，从中彩网补充
        if use_all_data or (count is not None and len(results) < count):
            print(f"\n=== 第二阶段：从中彩网补充数据 ===")
            print(f"当前已获取{len(results)}期数据，开始补充更多历史数据...")

            # 对于获取所有数据的情况，增加中彩网的页数
            max_zhcw_pages = 500 if use_all_data else 100
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

            # 如果指定了数量限制，截取到指定数量
            if count is not None and len(results) > count:
                results = results[:count]

        # 保存数据
        if results:
            self.save_to_csv(results, filename)
            print(f"\n=== 爬取完成 ===")
            print(f"成功爬取{len(results)}期双色球历史数据")
            print(f"数据保存到: {filename}")

            # 显示数据范围
            if len(results) > 0:
                earliest_issue = min(results, key=lambda x: int(x["issue"]))
                latest_issue = max(results, key=lambda x: int(x["issue"]))
                print(f"数据范围: {earliest_issue['issue']}期 - {latest_issue['issue']}期")
                print(f"时间范围: {earliest_issue['date']} - {latest_issue['date']}")

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
    
    def load_data(self, data_file=None, force_all_data=True):
        """加载数据，默认强制使用完整历史数据"""
        if data_file is None:
            if force_all_data:
                # 强制使用全量历史数据
                data_file = os.path.join(self.data_dir, "ssq_data_all.csv")
                if not os.path.exists(data_file):
                    print("警告: 完整历史数据文件(ssq_data_all.csv)不存在，请先运行: python3 ssq_analyzer.py crawl --all")
                    print("正在尝试使用部分数据文件...")
                    data_file = os.path.join(self.data_dir, "ssq_data.csv")
            else:
                # 优先使用全量数据，但允许降级
                data_file = os.path.join(self.data_dir, "ssq_data_all.csv")
                if not os.path.exists(data_file):
                    data_file = os.path.join(self.data_dir, "ssq_data.csv")

        try:
            if not os.path.exists(data_file):
                print(f"数据文件不存在: {data_file}")
                print("请先运行以下命令获取数据:")
                print("  python3 ssq_analyzer.py crawl --all  # 获取完整历史数据")
                print("  python3 ssq_analyzer.py crawl        # 获取最近300期数据")
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
            print("正在加载完整历史数据进行组合特征分析...")
            if not self.load_data(force_all_data=True):
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
            print("正在加载完整历史数据进行走势分析...")
            if not self.load_data(force_all_data=True):
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
        print("正在加载完整历史数据进行基础分析...")
        if not self.load_data(force_all_data=True):
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
            print("正在加载完整历史数据进行统计特性分析...")
            if not self.load_data(force_all_data=True):
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
            print("正在加载完整历史数据进行马尔可夫链分析...")
            if not self.load_data(force_all_data=True):
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
        # 确保使用完整历史数据
        if self.data is None:
            print("正在加载完整历史数据进行马尔可夫链预测...")
            if not self.load_data(force_all_data=True):
                return self.generate_random_numbers()

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

        # 确保使用完整历史数据
        if self.data is None:
            print("正在加载完整历史数据进行马尔可夫链预测...")
            if not self.load_data(force_all_data=True):
                return []

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
            print("正在加载完整历史数据进行马尔可夫链预测...")
            if not self.load_data(force_all_data=True):
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
            print("正在加载完整历史数据进行集成预测...")
            if not self.load_data(force_all_data=True):
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
            print("正在加载完整历史数据进行统计学预测...")
            if not self.load_data(force_all_data=True):
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
            print("正在加载完整历史数据进行概率论预测...")
            if not self.load_data(force_all_data=True):
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
            print("正在加载完整历史数据进行决策树预测...")
            if not self.load_data(force_all_data=True):
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
            print("正在加载完整历史数据进行模式识别预测...")
            if not self.load_data(force_all_data=True):
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
            print("正在加载完整历史数据进行智能号码生成...")
            if not self.load_data(force_all_data=True):
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
        print("正在加载完整历史数据进行高级分析...")
        if not self.load_data(force_all_data=True):
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

        # 确保使用完整历史数据
        if self.data is None:
            print("正在加载完整历史数据进行贝叶斯分析...")
            if not self.load_data(force_all_data=True):
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

    def predict_by_lstm(self, explain=False):
        """基于LSTM深度学习预测"""
        if not TENSORFLOW_AVAILABLE:
            if explain:
                print("TensorFlow未安装，无法使用LSTM预测，使用统计学方法替代")
            return self.predict_by_stats(explain)

        try:
            if explain:
                print("使用LSTM深度学习时间序列预测...")
                print("构建LSTM神经网络模型进行序列预测...")

            # 准备数据
            if self.data is None:
                if not self.load_data(force_all_data=True):
                    return self.generate_random_numbers()

            # 使用更多历史数据进行训练
            sequence_length = 10  # 使用10期历史作为序列长度
            train_data_size = min(500, len(self.data))  # 使用最近500期数据训练

            if explain:
                print(f"使用最近{train_data_size}期数据训练LSTM模型")
                print(f"序列长度: {sequence_length}期")

            # 准备训练数据
            train_data = self.data.head(train_data_size)

            # 构建时间序列特征
            sequences = []
            targets_red = []
            targets_blue = []

            for i in range(sequence_length, len(train_data)):
                # 输入序列：前sequence_length期的特征
                sequence = []
                for j in range(i - sequence_length, i):
                    row = train_data.iloc[j]
                    # 特征：红球号码、蓝球号码、统计特征
                    features = []
                    # 红球特征（归一化到0-1）
                    for k in range(1, 7):
                        features.append(row[f'red_{k}'] / 33.0)
                    # 蓝球特征（归一化到0-1）
                    features.append(row['blue_ball'] / 16.0)
                    # 统计特征
                    red_sum = sum([row[f'red_{k}'] for k in range(1, 7)])
                    features.append(red_sum / 198.0)  # 最大和值198
                    red_span = max([row[f'red_{k}'] for k in range(1, 7)]) - min([row[f'red_{k}'] for k in range(1, 7)])
                    features.append(red_span / 32.0)  # 最大跨度32

                    sequence.append(features)

                sequences.append(sequence)

                # 目标：当前期的号码
                current_row = train_data.iloc[i]
                targets_red.append([current_row[f'red_{k}'] / 33.0 for k in range(1, 7)])
                targets_blue.append(current_row['blue_ball'] / 16.0)

            if len(sequences) < 20:
                if explain:
                    print("训练数据不足，使用传统时间序列分析方法")
                return self._lstm_fallback_prediction(explain)

            # 转换为numpy数组
            X = np.array(sequences)
            y_red = np.array(targets_red)
            y_blue = np.array(targets_blue)

            if explain:
                print(f"训练数据形状: X={X.shape}, y_red={y_red.shape}, y_blue={y_blue.shape}")

            # 构建LSTM模型
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            # 红球预测模型
            model_red = Sequential([
                LSTM(64, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(6, activation='sigmoid')  # 6个红球
            ])

            model_red.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

            # 蓝球预测模型
            model_blue = Sequential([
                LSTM(32, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
                Dropout(0.2),
                LSTM(16, return_sequences=False),
                Dropout(0.2),
                Dense(8, activation='relu'),
                Dense(1, activation='sigmoid')  # 1个蓝球
            ])

            model_blue.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

            if explain:
                print("开始训练LSTM模型...")

            # 训练模型（快速训练）
            model_red.fit(X, y_red, epochs=20, batch_size=16, verbose=0, validation_split=0.2)
            model_blue.fit(X, y_blue, epochs=20, batch_size=16, verbose=0, validation_split=0.2)

            # 准备预测输入
            last_sequence = sequences[-1]  # 最近的序列
            pred_input = np.array([last_sequence])

            # 预测
            pred_red = model_red.predict(pred_input, verbose=0)[0]
            pred_blue = model_blue.predict(pred_input, verbose=0)[0][0]

            # 转换预测结果为实际号码
            # 红球：将概率转换为号码选择
            red_probs = {}
            for i in range(1, 34):
                # 计算每个号码的选择概率
                prob = 0
                for j, pred_val in enumerate(pred_red):
                    # 基于预测值和号码的相似度计算概率
                    similarity = 1 - abs(pred_val - i/33.0)
                    prob += similarity * (j + 1) / 6  # 位置权重
                red_probs[i] = prob

            # 选择概率最高的6个红球
            sorted_reds = sorted(red_probs.items(), key=lambda x: x[1], reverse=True)
            selected_reds = [ball for ball, _ in sorted_reds[:6]]
            selected_reds.sort()

            # 蓝球：将预测值转换为最接近的号码
            predicted_blue_num = max(1, min(16, round(pred_blue * 16)))

            if explain:
                print("LSTM模型训练完成")
                print(f"红球预测概率分布: {[f'{ball}({prob:.3f})' for ball, prob in sorted_reds[:6]]}")
                print(f"蓝球预测值: {pred_blue:.3f} -> {predicted_blue_num}")

            return selected_reds, predicted_blue_num

        except Exception as e:
            if explain:
                print(f"LSTM预测失败: {e}，使用时间序列分析方法替代")
            return self._lstm_fallback_prediction(explain)

    def _lstm_fallback_prediction(self, explain=False):
        """LSTM的时间序列分析回退方法"""
        try:
            if explain:
                print("使用时间序列分析方法进行预测...")

            # 使用最近50期数据进行时间序列分析
            recent_data = self.data.head(50)

            # 分析时间序列趋势
            red_trends = {i: [] for i in range(1, 34)}
            blue_trends = []

            for idx, row in recent_data.iterrows():
                # 记录每个号码的出现位置和时间权重
                weight = (len(recent_data) - idx) / len(recent_data)

                for i in range(1, 7):
                    red_num = row[f'red_{i}']
                    red_trends[red_num].append(weight)

                blue_trends.append((row['blue_ball'], weight))

            # 计算趋势得分
            red_scores = {}
            for num in range(1, 34):
                if red_trends[num]:
                    # 趋势得分 = 平均权重 * 出现次数 * 最近出现权重
                    avg_weight = sum(red_trends[num]) / len(red_trends[num])
                    frequency = len(red_trends[num])
                    recent_weight = max(red_trends[num]) if red_trends[num] else 0
                    red_scores[num] = avg_weight * frequency * (1 + recent_weight)
                else:
                    red_scores[num] = 0.1  # 给未出现的号码一个小的基础分

            # 选择得分最高的6个红球
            sorted_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
            selected_reds = [ball for ball, _ in sorted_reds[:6]]
            selected_reds.sort()

            # 蓝球：基于加权频率选择
            blue_weights = {}
            for blue_num, weight in blue_trends:
                blue_weights[blue_num] = blue_weights.get(blue_num, 0) + weight

            selected_blue = max(blue_weights.items(), key=lambda x: x[1])[0]

            if explain:
                print("时间序列分析完成")
                print(f"红球趋势得分前6名: {[(ball, f'{score:.3f}') for ball, score in sorted_reds[:6]]}")
                print(f"蓝球权重最高: {selected_blue}({blue_weights[selected_blue]:.3f})")

            return selected_reds, selected_blue

        except Exception as e:
            if explain:
                print(f"时间序列分析失败: {e}，使用统计学方法")
            return self.predict_by_stats(explain)

    def predict_by_monte_carlo(self, explain=False):
        """基于蒙特卡洛方法预测"""
        try:
            if explain:
                print("使用蒙特卡洛模拟预测...")
                print("构建多维概率分布模型进行随机采样...")

            # 准备数据
            if self.data is None:
                if not self.load_data(force_all_data=True):
                    return self.generate_random_numbers()

            # 构建多维概率分布
            # 1. 单个号码出现概率
            red_freq = {i: 0 for i in range(1, 34)}
            blue_freq = {i: 0 for i in range(1, 17)}

            # 2. 位置概率分布
            position_probs = {i: {j: 0 for j in range(1, 34)} for i in range(1, 7)}

            # 3. 组合特征概率分布
            sum_dist = {}
            span_dist = {}
            odd_count_dist = {}

            # 4. 相邻期关联概率
            transition_probs = {}

            if explain:
                print(f"分析{len(self.data)}期历史数据构建概率分布...")

            for idx, row in self.data.iterrows():
                # 单个号码频率
                reds = [row[f'red_{i}'] for i in range(1, 7)]
                for red in reds:
                    red_freq[red] += 1

                blue_freq[row['blue_ball']] += 1

                # 位置概率
                for i, red in enumerate(reds):
                    position_probs[i+1][red] += 1

                # 组合特征
                red_sum = sum(reds)
                red_span = max(reds) - min(reds)
                odd_count = sum(1 for x in reds if x % 2 == 1)

                sum_dist[red_sum] = sum_dist.get(red_sum, 0) + 1
                span_dist[red_span] = span_dist.get(red_span, 0) + 1
                odd_count_dist[odd_count] = odd_count_dist.get(odd_count, 0) + 1

                # 转移概率（当前期与前一期的关系）
                if idx < len(self.data) - 1:
                    prev_row = self.data.iloc[idx + 1]
                    prev_reds = set([prev_row[f'red_{i}'] for i in range(1, 7)])
                    curr_reds = set(reds)

                    # 计算重复号码数量
                    overlap = len(prev_reds & curr_reds)
                    transition_probs[overlap] = transition_probs.get(overlap, 0) + 1

            # 归一化概率
            total_periods = len(self.data)

            # 单号码概率
            red_probs = {k: v/(total_periods*6) for k, v in red_freq.items()}
            blue_probs = {k: v/total_periods for k, v in blue_freq.items()}

            # 位置概率
            for pos in position_probs:
                total = sum(position_probs[pos].values())
                if total > 0:
                    position_probs[pos] = {k: v/total for k, v in position_probs[pos].items()}

            # 组合特征概率
            sum_probs = {k: v/total_periods for k, v in sum_dist.items()}
            span_probs = {k: v/total_periods for k, v in span_dist.items()}
            odd_probs = {k: v/total_periods for k, v in odd_count_dist.items()}

            if explain:
                print("概率分布构建完成，开始蒙特卡洛模拟...")

            # 蒙特卡洛模拟
            num_simulations = 50000  # 增加模拟次数
            valid_combinations = []

            import random

            for sim in range(num_simulations):
                # 方法1：基于位置概率生成（30%）
                if random.random() < 0.3:
                    sim_reds = []
                    for pos in range(1, 7):
                        candidates = list(position_probs[pos].keys())
                        weights = list(position_probs[pos].values())
                        if sum(weights) > 0:
                            selected = random.choices(candidates, weights=weights)[0]
                            if selected not in sim_reds:
                                sim_reds.append(selected)

                    # 如果不足6个，随机补充
                    while len(sim_reds) < 6:
                        candidates = [i for i in range(1, 34) if i not in sim_reds]
                        if candidates:
                            sim_reds.append(random.choice(candidates))

                # 方法2：基于组合特征约束生成（40%）
                elif random.random() < 0.7:
                    # 先选择目标组合特征
                    target_sum = random.choices(list(sum_probs.keys()), weights=list(sum_probs.values()))[0]
                    target_span = random.choices(list(span_probs.keys()), weights=list(span_probs.values()))[0]
                    target_odd = random.choices(list(odd_probs.keys()), weights=list(odd_probs.values()))[0]

                    # 基于约束生成号码
                    sim_reds = self._generate_constrained_combination(target_sum, target_span, target_odd, red_probs)

                # 方法3：纯概率采样（30%）
                else:
                    sim_reds = []
                    candidates = list(red_probs.keys())
                    weights = list(red_probs.values())

                    while len(sim_reds) < 6:
                        selected = random.choices(candidates, weights=weights)[0]
                        if selected not in sim_reds:
                            sim_reds.append(selected)

                sim_reds.sort()

                # 生成蓝球
                sim_blue = random.choices(list(blue_probs.keys()), weights=list(blue_probs.values()))[0]

                # 验证组合的合理性
                if self._validate_combination(sim_reds, sim_blue):
                    valid_combinations.append((sim_reds, sim_blue))

            if not valid_combinations:
                if explain:
                    print("未生成有效组合，使用概率论方法替代")
                return self.predict_by_probability(explain)

            # 统计最优组合
            combo_scores = {}
            for reds, blue in valid_combinations:
                combo = tuple(reds + [blue])

                # 计算组合得分（考虑多个因素）
                score = 0

                # 1. 基础频率得分
                for red in reds:
                    score += red_probs[red]
                score += blue_probs[blue]

                # 2. 组合特征得分
                combo_sum = sum(reds)
                combo_span = max(reds) - min(reds)
                combo_odd = sum(1 for x in reds if x % 2 == 1)

                score += sum_probs.get(combo_sum, 0) * 2
                score += span_probs.get(combo_span, 0) * 2
                score += odd_probs.get(combo_odd, 0) * 2

                combo_scores[combo] = combo_scores.get(combo, 0) + score

            # 选择得分最高的组合
            best_combo = max(combo_scores.keys(), key=lambda x: combo_scores[x])
            best_score = combo_scores[best_combo]

            if explain:
                print(f"蒙特卡洛模拟完成：{num_simulations}次模拟，{len(valid_combinations)}个有效组合")
                print(f"最优组合得分: {best_score:.6f}")
                print(f"组合特征: 和值={sum(best_combo[:6])}, 跨度={max(best_combo[:6])-min(best_combo[:6])}, 奇数={sum(1 for x in best_combo[:6] if x%2==1)}个")

            return list(best_combo[:6]), best_combo[6]

        except Exception as e:
            if explain:
                print(f"蒙特卡洛预测失败: {e}，使用概率论方法替代")
            return self.predict_by_probability(explain)

    def _generate_constrained_combination(self, target_sum, target_span, target_odd, red_probs):
        """基于约束条件生成号码组合"""
        import random

        max_attempts = 1000
        for _ in range(max_attempts):
            reds = []

            # 先选择一个最小值和最大值来控制跨度
            min_val = random.randint(1, 33 - target_span)
            max_val = min_val + target_span

            # 确保包含最小值和最大值
            reds.extend([min_val, max_val])

            # 根据奇偶数要求选择剩余号码
            current_odd = sum(1 for x in reds if x % 2 == 1)
            need_odd = target_odd - current_odd
            need_even = 4 - need_odd

            # 在范围内选择剩余4个号码
            candidates = list(range(min_val + 1, max_val))
            odd_candidates = [x for x in candidates if x % 2 == 1 and x not in reds]
            even_candidates = [x for x in candidates if x % 2 == 0 and x not in reds]

            # 选择奇数
            selected_odds = random.sample(odd_candidates, min(need_odd, len(odd_candidates))) if need_odd > 0 else []
            # 选择偶数
            selected_evens = random.sample(even_candidates, min(need_even, len(even_candidates))) if need_even > 0 else []

            reds.extend(selected_odds)
            reds.extend(selected_evens)

            # 如果不足6个，随机补充
            while len(reds) < 6:
                remaining = [x for x in range(min_val, max_val + 1) if x not in reds]
                if remaining:
                    reds.append(random.choice(remaining))
                else:
                    break

            if len(reds) == 6:
                current_sum = sum(reds)
                # 检查和值是否接近目标（允许一定误差）
                if abs(current_sum - target_sum) <= 10:
                    return sorted(reds)

        # 如果无法生成满足约束的组合，使用概率采样
        reds = []
        candidates = list(red_probs.keys())
        weights = list(red_probs.values())

        while len(reds) < 6:
            selected = random.choices(candidates, weights=weights)[0]
            if selected not in reds:
                reds.append(selected)

        return sorted(reds)

    def _validate_combination(self, reds, blue):
        """验证号码组合的合理性"""
        if len(reds) != 6 or len(set(reds)) != 6:
            return False

        if not all(1 <= red <= 33 for red in reds):
            return False

        if not (1 <= blue <= 16):
            return False

        # 检查基本统计特征是否合理
        red_sum = sum(reds)
        red_span = max(reds) - min(reds)

        # 和值应在合理范围内
        if not (21 <= red_sum <= 183):  # 理论范围
            return False

        # 跨度应在合理范围内
        if not (5 <= red_span <= 32):
            return False

        return True

    def predict_by_clustering(self, explain=False):
        """基于聚类分析预测"""
        try:
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            from sklearn.decomposition import PCA

            if explain:
                print("使用多算法聚类分析预测...")
                print("构建高维特征空间进行模式识别...")

            # 准备数据
            if self.data is None:
                if not self.load_data(force_all_data=True):
                    return self.generate_random_numbers()

            # 使用更多数据进行聚类分析
            analysis_size = min(1000, len(self.data))
            analysis_data = self.data.head(analysis_size)

            if explain:
                print(f"使用最近{analysis_size}期数据进行聚类分析")

            # 构建多维特征矩阵
            features = []
            for _, row in analysis_data.iterrows():
                feature = []

                # 1. 基础号码特征
                reds = [row[f'red_{i}'] for i in range(1, 7)]
                feature.extend(reds)
                feature.append(row['blue_ball'])

                # 2. 统计特征
                feature.append(sum(reds))  # 和值
                feature.append(max(reds) - min(reds))  # 跨度
                feature.append(np.var(reds))  # 方差
                feature.append(np.std(reds))  # 标准差

                # 3. 分布特征
                feature.append(sum(1 for x in reds if x % 2 == 1))  # 奇数个数
                feature.append(sum(1 for x in reds if x >= 17))  # 大数个数
                feature.append(sum(1 for x in reds if x <= 11))  # 小数个数
                feature.append(sum(1 for x in reds if 12 <= x <= 22))  # 中数个数

                # 4. 区间分布特征
                zones = [0, 0, 0]  # 三个区间：1-11, 12-22, 23-33
                for red in reds:
                    if red <= 11:
                        zones[0] += 1
                    elif red <= 22:
                        zones[1] += 1
                    else:
                        zones[2] += 1
                feature.extend(zones)

                # 5. 连号特征
                consecutive_count = 0
                sorted_reds = sorted(reds)
                for i in range(len(sorted_reds) - 1):
                    if sorted_reds[i+1] - sorted_reds[i] == 1:
                        consecutive_count += 1
                feature.append(consecutive_count)

                # 6. AC值（算术复杂性）
                ac_value = 0
                for i in range(len(sorted_reds)):
                    for j in range(i+1, len(sorted_reds)):
                        diff = abs(sorted_reds[i] - sorted_reds[j])
                        if diff not in [abs(sorted_reds[k] - sorted_reds[l])
                                       for k in range(len(sorted_reds))
                                       for l in range(k+1, len(sorted_reds))
                                       if (k, l) != (i, j)]:
                            ac_value += 1
                feature.append(ac_value)

                features.append(feature)

            features = np.array(features)

            if explain:
                print(f"构建特征矩阵: {features.shape[0]}样本 × {features.shape[1]}特征")

            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # 降维（可选）
            if features.shape[1] > 20:
                pca = PCA(n_components=min(20, features.shape[1]))
                features_scaled = pca.fit_transform(features_scaled)
                if explain:
                    print(f"PCA降维到{features_scaled.shape[1]}维，保留方差比例: {pca.explained_variance_ratio_.sum():.3f}")

            # 多种聚类算法比较
            clustering_results = {}

            # 1. K-means聚类
            best_kmeans_score = -1
            best_kmeans_k = 0
            best_kmeans_labels = None

            for k in range(3, min(15, len(features) // 20)):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features_scaled)
                    score = silhouette_score(features_scaled, labels)

                    if score > best_kmeans_score:
                        best_kmeans_score = score
                        best_kmeans_k = k
                        best_kmeans_labels = labels
                except:
                    continue

            if best_kmeans_labels is not None:
                clustering_results['kmeans'] = {
                    'labels': best_kmeans_labels,
                    'score': best_kmeans_score,
                    'n_clusters': best_kmeans_k
                }

            # 2. DBSCAN聚类
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(features_scaled)
                n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

                if n_clusters_dbscan > 1:
                    # 过滤噪声点
                    valid_indices = dbscan_labels != -1
                    if np.sum(valid_indices) > 10:
                        dbscan_score = silhouette_score(features_scaled[valid_indices],
                                                       dbscan_labels[valid_indices])
                        clustering_results['dbscan'] = {
                            'labels': dbscan_labels,
                            'score': dbscan_score,
                            'n_clusters': n_clusters_dbscan
                        }
            except:
                pass

            # 3. 层次聚类
            try:
                n_clusters_hier = best_kmeans_k if best_kmeans_k > 0 else 5
                hier = AgglomerativeClustering(n_clusters=n_clusters_hier)
                hier_labels = hier.fit_predict(features_scaled)
                hier_score = silhouette_score(features_scaled, hier_labels)

                clustering_results['hierarchical'] = {
                    'labels': hier_labels,
                    'score': hier_score,
                    'n_clusters': n_clusters_hier
                }
            except:
                pass

            if not clustering_results:
                if explain:
                    print("所有聚类算法失败，使用决策树方法替代")
                return self.predict_by_decision_tree_advanced(explain)

            # 选择最佳聚类结果
            best_method = max(clustering_results.keys(),
                            key=lambda x: clustering_results[x]['score'])
            best_result = clustering_results[best_method]

            if explain:
                print(f"最佳聚类方法: {best_method}")
                print(f"聚类数量: {best_result['n_clusters']}")
                print(f"轮廓系数: {best_result['score']:.3f}")

            # 基于聚类结果进行预测
            cluster_labels = best_result['labels']
            latest_cluster = cluster_labels[0]  # 最新一期的聚类

            # 分析同一聚类中的模式
            same_cluster_indices = np.where(cluster_labels == latest_cluster)[0]

            if len(same_cluster_indices) < 3:
                if explain:
                    print("当前聚类样本不足，扩展到相似聚类")
                # 找到相似的聚类
                cluster_centers = {}
                for cluster_id in set(cluster_labels):
                    if cluster_id == -1:  # 跳过噪声点
                        continue
                    cluster_indices = np.where(cluster_labels == cluster_id)[0]
                    cluster_features = features_scaled[cluster_indices]
                    cluster_centers[cluster_id] = np.mean(cluster_features, axis=0)

                # 找到最相似的聚类
                if latest_cluster in cluster_centers:
                    latest_center = cluster_centers[latest_cluster]
                    similarities = {}
                    for cluster_id, center in cluster_centers.items():
                        if cluster_id != latest_cluster:
                            similarity = np.dot(latest_center, center) / (
                                np.linalg.norm(latest_center) * np.linalg.norm(center))
                            similarities[cluster_id] = similarity

                    if similarities:
                        most_similar = max(similarities.keys(), key=lambda x: similarities[x])
                        similar_indices = np.where(cluster_labels == most_similar)[0]
                        same_cluster_indices = np.concatenate([same_cluster_indices, similar_indices])

            # 基于聚类模式生成预测
            cluster_data = analysis_data.iloc[same_cluster_indices]

            # 计算聚类中心的特征
            red_patterns = {i: [] for i in range(1, 34)}
            blue_patterns = []

            for _, row in cluster_data.iterrows():
                for i in range(1, 7):
                    red_patterns[row[f'red_{i}']].append(1)
                blue_patterns.append(row['blue_ball'])

            # 基于模式频率选择号码
            red_scores = {}
            for num in range(1, 34):
                frequency = len(red_patterns[num])
                # 考虑在聚类中的出现频率和位置分布
                red_scores[num] = frequency / len(cluster_data) if len(cluster_data) > 0 else 0

            # 选择得分最高的6个红球
            sorted_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)

            # 确保选择的号码组合合理
            selected_reds = []
            for red, score in sorted_reds:
                if len(selected_reds) < 6 and red not in selected_reds:
                    selected_reds.append(red)

            # 如果不足6个，补充高频号码
            if len(selected_reds) < 6:
                all_reds = [row[f'red_{i}'] for _, row in self.data.head(100).iterrows() for i in range(1, 7)]
                red_freq = {i: all_reds.count(i) for i in range(1, 34)}
                sorted_by_freq = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)

                for red, _ in sorted_by_freq:
                    if red not in selected_reds and len(selected_reds) < 6:
                        selected_reds.append(red)

            selected_reds.sort()

            # 蓝球：基于聚类中的分布选择
            if blue_patterns:
                blue_freq = {i: blue_patterns.count(i) for i in set(blue_patterns)}
                selected_blue = max(blue_freq.items(), key=lambda x: x[1])[0]
            else:
                # 回退到全局频率
                all_blues = [row['blue_ball'] for _, row in self.data.head(100).iterrows()]
                blue_freq = {i: all_blues.count(i) for i in range(1, 17)}
                selected_blue = max(blue_freq.items(), key=lambda x: x[1])[0]

            if explain:
                print(f"聚类预测完成")
                print(f"目标聚类包含{len(same_cluster_indices)}个样本")
                print(f"红球模式得分前6名: {[(ball, f'{score:.3f}') for ball, score in sorted_reds[:6]]}")
                print(f"蓝球聚类频率: {selected_blue}")

            return selected_reds, selected_blue

        except Exception as e:
            if explain:
                print(f"聚类预测失败: {e}，使用决策树方法替代")
            return self.predict_by_decision_tree_advanced(explain)

    def predict_by_super(self, explain=False):
        """超级预测器 - 集成多种高级算法的智能融合系统"""
        try:
            if explain:
                print("启动超级预测器...")
                print("构建多层次智能融合预测系统...")
                print("集成LSTM、蒙特卡洛、聚类、马尔可夫链、混合分析等算法")

            # 第一层：基础预测器集合
            base_predictions = {}
            prediction_confidence = {}

            # 1. 高级混合分析预测（最高权重）
            try:
                start_time = time.time()
                hybrid_results = self.predict_by_advanced_hybrid_analysis(periods=50, count=1, explain=False)
                if hybrid_results:
                    base_predictions['hybrid'] = hybrid_results[0]
                    prediction_confidence['hybrid'] = 0.95  # 最高置信度
                    if explain:
                        print(f"✓ 高级混合分析预测完成 ({time.time()-start_time:.2f}s)")
            except Exception as e:
                if explain:
                    print(f"✗ 高级混合分析预测失败: {e}")

            # 2. LSTM深度学习预测
            try:
                start_time = time.time()
                lstm_result = self.predict_by_lstm(explain=False)
                if lstm_result:
                    base_predictions['lstm'] = lstm_result
                    prediction_confidence['lstm'] = 0.85
                    if explain:
                        print(f"✓ LSTM深度学习预测完成 ({time.time()-start_time:.2f}s)")
            except Exception as e:
                if explain:
                    print(f"✗ LSTM预测失败: {e}")

            # 3. 蒙特卡洛模拟预测
            try:
                start_time = time.time()
                mc_result = self.predict_by_monte_carlo(explain=False)
                if mc_result:
                    base_predictions['monte_carlo'] = mc_result
                    prediction_confidence['monte_carlo'] = 0.80
                    if explain:
                        print(f"✓ 蒙特卡洛模拟预测完成 ({time.time()-start_time:.2f}s)")
            except Exception as e:
                if explain:
                    print(f"✗ 蒙特卡洛预测失败: {e}")

            # 4. 聚类分析预测
            try:
                start_time = time.time()
                cluster_result = self.predict_by_clustering(explain=False)
                if cluster_result:
                    base_predictions['clustering'] = cluster_result
                    prediction_confidence['clustering'] = 0.75
                    if explain:
                        print(f"✓ 聚类分析预测完成 ({time.time()-start_time:.2f}s)")
            except Exception as e:
                if explain:
                    print(f"✗ 聚类预测失败: {e}")

            # 5. 马尔可夫链预测
            try:
                start_time = time.time()
                markov_results = self.predict_by_markov_chain_with_periods(periods=100, count=1, explain=False)
                if markov_results:
                    base_predictions['markov'] = markov_results[0]
                    prediction_confidence['markov'] = 0.90
                    if explain:
                        print(f"✓ 马尔可夫链预测完成 ({time.time()-start_time:.2f}s)")
            except Exception as e:
                if explain:
                    print(f"✗ 马尔可夫链预测失败: {e}")

            # 6. 集成学习预测
            try:
                start_time = time.time()
                ensemble_result = self.predict_by_ensemble(explain=False)
                if ensemble_result:
                    base_predictions['ensemble'] = ensemble_result
                    prediction_confidence['ensemble'] = 0.85
                    if explain:
                        print(f"✓ 集成学习预测完成 ({time.time()-start_time:.2f}s)")
            except Exception as e:
                if explain:
                    print(f"✗ 集成学习预测失败: {e}")

            if not base_predictions:
                if explain:
                    print("所有基础预测器失败，使用统计学方法")
                return self.predict_by_stats(explain)

            if explain:
                print(f"\n第一层预测完成，成功运行{len(base_predictions)}个预测器")

            # 第二层：智能融合算法
            # 计算预测器之间的一致性
            consistency_matrix = {}
            for method1 in base_predictions:
                for method2 in base_predictions:
                    if method1 != method2:
                        reds1, blue1 = base_predictions[method1]
                        reds2, blue2 = base_predictions[method2]

                        # 红球一致性
                        red_overlap = len(set(reds1) & set(reds2))
                        red_consistency = red_overlap / 6.0

                        # 蓝球一致性
                        blue_consistency = 1.0 if blue1 == blue2 else 0.0

                        # 综合一致性
                        overall_consistency = (red_consistency * 0.8 + blue_consistency * 0.2)
                        consistency_matrix[(method1, method2)] = overall_consistency

            # 计算每个预测器的平均一致性得分
            consistency_scores = {}
            for method in base_predictions:
                scores = []
                for (m1, m2), score in consistency_matrix.items():
                    if m1 == method:
                        scores.append(score)
                consistency_scores[method] = np.mean(scores) if scores else 0.5

            if explain:
                print("预测器一致性分析:")
                for method, score in consistency_scores.items():
                    print(f"  {method}: {score:.3f}")

            # 第三层：动态权重计算
            final_weights = {}
            for method in base_predictions:
                # 基础权重（基于预测器类型）
                base_weight = {
                    'hybrid': 0.30,
                    'markov': 0.25,
                    'lstm': 0.20,
                    'ensemble': 0.15,
                    'monte_carlo': 0.10,
                    'clustering': 0.05
                }.get(method, 0.05)

                # 置信度权重
                confidence_weight = prediction_confidence.get(method, 0.5)

                # 一致性权重
                consistency_weight = consistency_scores.get(method, 0.5)

                # 综合权重计算
                final_weight = (base_weight * 0.5 +
                              confidence_weight * 0.3 +
                              consistency_weight * 0.2)

                final_weights[method] = final_weight

            # 权重归一化
            total_weight = sum(final_weights.values())
            if total_weight > 0:
                final_weights = {k: v/total_weight for k, v in final_weights.items()}

            if explain:
                print("动态权重分配:")
                for method, weight in final_weights.items():
                    print(f"  {method}: {weight:.3f}")

            # 第四层：智能投票融合
            red_scores = {}
            blue_scores = {}

            for method, (reds, blue) in base_predictions.items():
                weight = final_weights.get(method, 0.1)

                # 红球评分（考虑位置权重）
                for i, red in enumerate(reds):
                    position_weight = (6 - i) / 6.0  # 位置越靠前权重越高
                    score = weight * position_weight
                    red_scores[red] = red_scores.get(red, 0) + score

                # 蓝球评分
                blue_scores[blue] = blue_scores.get(blue, 0) + weight

            # 第五层：结果优化
            # 选择红球（确保组合合理性）
            sorted_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)

            final_reds = []
            for red, score in sorted_reds:
                if len(final_reds) < 6:
                    final_reds.append(red)

            # 验证和优化红球组合
            if len(final_reds) == 6:
                # 检查组合的统计特征
                red_sum = sum(final_reds)
                red_span = max(final_reds) - min(final_reds)
                odd_count = sum(1 for x in final_reds if x % 2 == 1)

                # 如果组合不合理，进行微调
                if not (60 <= red_sum <= 140) or red_span < 10 or red_span > 30:
                    if explain:
                        print("检测到不合理组合，进行智能优化...")

                    # 使用历史统计进行优化
                    historical_stats = self._get_historical_stats()
                    final_reds = self._optimize_combination(final_reds, historical_stats)

            # 补充红球（如果不足6个）
            if len(final_reds) < 6:
                # 基于历史频率补充
                all_reds_freq = {}
                for _, row in self.data.head(200).iterrows():
                    for i in range(1, 7):
                        red = row[f'red_{i}']
                        all_reds_freq[red] = all_reds_freq.get(red, 0) + 1

                sorted_by_freq = sorted(all_reds_freq.items(), key=lambda x: x[1], reverse=True)
                for red, _ in sorted_by_freq:
                    if red not in final_reds and len(final_reds) < 6:
                        final_reds.append(red)

            final_reds.sort()

            # 选择蓝球
            final_blue = max(blue_scores.items(), key=lambda x: x[1])[0]

            if explain:
                print(f"\n超级预测器融合完成:")
                print(f"集成了{len(base_predictions)}种预测算法")
                print(f"红球融合得分前6名: {[(r, f'{red_scores[r]:.3f}') for r in final_reds]}")
                print(f"蓝球融合得分: {final_blue}({blue_scores[final_blue]:.3f})")
                print(f"最终组合: 红球 {final_reds} | 蓝球 {final_blue}")

                # 显示组合特征
                red_sum = sum(final_reds)
                red_span = max(final_reds) - min(final_reds)
                odd_count = sum(1 for x in final_reds if x % 2 == 1)
                print(f"组合特征: 和值={red_sum}, 跨度={red_span}, 奇数={odd_count}个")

            return final_reds, final_blue

        except Exception as e:
            if explain:
                print(f"超级预测器失败: {e}，使用集成方法替代")
            return self.predict_by_ensemble(explain)

    def _get_historical_stats(self):
        """获取历史统计特征"""
        stats = {
            'sum_range': (60, 140),
            'span_range': (10, 30),
            'odd_count_range': (2, 4),
            'common_sums': [],
            'common_spans': [],
            'common_odds': []
        }

        try:
            recent_data = self.data.head(200)
            sums = []
            spans = []
            odds = []

            for _, row in recent_data.iterrows():
                reds = [row[f'red_{i}'] for i in range(1, 7)]
                sums.append(sum(reds))
                spans.append(max(reds) - min(reds))
                odds.append(sum(1 for x in reds if x % 2 == 1))

            # 计算常见值
            from collections import Counter
            sum_counter = Counter(sums)
            span_counter = Counter(spans)
            odd_counter = Counter(odds)

            stats['common_sums'] = [s for s, _ in sum_counter.most_common(10)]
            stats['common_spans'] = [s for s, _ in span_counter.most_common(5)]
            stats['common_odds'] = [s for s, _ in odd_counter.most_common(3)]

        except:
            pass

        return stats

    def _optimize_combination(self, reds, stats):
        """优化号码组合"""
        try:
            import random

            # 如果组合已经合理，直接返回
            red_sum = sum(reds)
            red_span = max(reds) - min(reds)
            odd_count = sum(1 for x in reds if x % 2 == 1)

            target_sum = random.choice(stats['common_sums']) if stats['common_sums'] else red_sum
            target_span = random.choice(stats['common_spans']) if stats['common_spans'] else red_span
            target_odd = random.choice(stats['common_odds']) if stats['common_odds'] else odd_count

            # 尝试微调
            optimized_reds = reds.copy()

            # 调整和值
            current_sum = sum(optimized_reds)
            if abs(current_sum - target_sum) > 10:
                diff = target_sum - current_sum
                # 随机选择一个号码进行调整
                idx = random.randint(0, 5)
                new_val = optimized_reds[idx] + diff // 6
                if 1 <= new_val <= 33 and new_val not in optimized_reds:
                    optimized_reds[idx] = new_val

            return sorted(optimized_reds)

        except:
            return sorted(reds)

    # ==================== 高级混合分析预测 ====================

    def predict_by_advanced_hybrid_analysis(self, periods=None, count=1, explain=True):
        """
        基于多种数学模型的高级混合分析预测
        整合统计学、概率论、马尔可夫链、贝叶斯分析、冷热号分布等方法

        Args:
            periods: 指定分析期数
            count: 预测注数
            explain: 是否显示详细分析过程

        Returns:
            预测结果列表
        """
        if self.data is None:
            print("正在加载完整历史数据进行高级混合分析预测...")
            if not self.load_data(force_all_data=True):
                return []

        print(f"\n{'='*70}")
        print(f"高级混合分析预测系统")
        print(f"整合统计学、概率论、马尔可夫链、贝叶斯分析、冷热号分布等数学模型")
        print(f"{'='*70}")

        # 确定分析数据范围
        if periods:
            if len(self.data) < periods:
                print(f"警告: 可用数据({len(self.data)}期)少于指定期数({periods}期)，将使用全部数据")
                analysis_data = self.data.copy()
                actual_periods = len(analysis_data)
            else:
                analysis_data = self.data.head(periods).copy()
                actual_periods = periods
                print(f"使用最近{periods}期数据进行高级混合分析")
        else:
            analysis_data = self.data.copy()
            actual_periods = len(analysis_data)
            print(f"使用全部{len(analysis_data)}期数据进行高级混合分析")

        print(f"分析数据期数: {actual_periods}期")
        print(f"数据范围: {analysis_data.iloc[-1]['issue']}期 - {analysis_data.iloc[0]['issue']}期")

        # 获取最近一期作为预测基础
        latest_data = analysis_data.iloc[0]
        latest_reds = [latest_data[f'red_{i}'] for i in range(1, 7)]
        latest_blue = latest_data['blue_ball']

        print(f"\n预测基础数据:")
        print(f"最近一期: {latest_data['issue']}期 ({latest_data['date']})")
        print(f"开奖号码: 红球 {' '.join([f'{ball:02d}' for ball in latest_reds])} | 蓝球 {latest_blue:02d}")

        # 执行综合分析
        if explain:
            print(f"\n开始执行多维度数学模型分析...")

        hybrid_analysis = self._execute_hybrid_analysis(analysis_data, explain)

        # 进行多注预测
        predictions = []

        for i in range(count):
            print(f"\n{'='*50}")
            print(f"第{i+1}注高级混合分析预测")
            print(f"{'='*50}")

            predicted_reds, predicted_blue = self._predict_with_hybrid_models(
                hybrid_analysis, latest_reds, latest_blue, actual_periods, i+1, explain
            )

            predictions.append((predicted_reds, predicted_blue))

            formatted = self.format_numbers(predicted_reds, predicted_blue)
            print(f"\n第{i+1}注混合模型预测结果: {formatted}")

        # 显示预测汇总
        print(f"\n{'='*70}")
        print(f"高级混合分析预测汇总")
        print(f"{'='*70}")
        print(f"分析期数: {actual_periods}期")
        print(f"预测注数: {count}注")
        print(f"预测基础: {latest_data['issue']}期")
        print(f"分析模型: 统计学+概率论+马尔可夫链+贝叶斯+冷热号分布")

        for i, (red_balls, blue_ball) in enumerate(predictions):
            formatted = self.format_numbers(red_balls, blue_ball)
            print(f"第{i+1}注: {formatted}")

        return predictions

    def _execute_hybrid_analysis(self, data, explain=True):
        """执行多维度数学模型分析"""
        analysis_results = {}

        if explain:
            print(f"\n【第一阶段：统计学分析】")

        # 1. 统计学分析
        stats_analysis = self._statistical_analysis(data, explain)
        analysis_results['统计学分析'] = stats_analysis

        if explain:
            print(f"\n【第二阶段：概率论分析】")

        # 2. 概率论分析
        prob_analysis = self._probability_analysis(data, explain)
        analysis_results['概率论分析'] = prob_analysis

        if explain:
            print(f"\n【第三阶段：马尔可夫链分析】")

        # 3. 马尔可夫链分析
        markov_analysis = self._analyze_markov_chain_stability(data)
        analysis_results['马尔可夫链分析'] = markov_analysis

        if explain:
            print(f"\n【第四阶段：贝叶斯分析】")

        # 4. 贝叶斯分析
        bayes_analysis = self._bayesian_analysis(data, explain)
        analysis_results['贝叶斯分析'] = bayes_analysis

        if explain:
            print(f"\n【第五阶段：冷热号分布分析】")

        # 5. 冷热号分布分析
        hot_cold_analysis = self._hot_cold_analysis(data, explain)
        analysis_results['冷热号分析'] = hot_cold_analysis

        if explain:
            print(f"\n【第六阶段：周期性分析】")

        # 6. 周期性分析
        cycle_analysis = self._cycle_analysis(data, explain)
        analysis_results['周期性分析'] = cycle_analysis

        if explain:
            print(f"\n【第七阶段：相关性分析】")

        # 7. 相关性分析
        correlation_analysis = self._correlation_analysis(data, explain)
        analysis_results['相关性分析'] = correlation_analysis

        return analysis_results

    def _statistical_analysis(self, data, explain=True):
        """统计学分析"""
        results = {}

        # 计算基本统计量
        red_sums = []
        red_variances = []
        red_spans = []

        for _, row in data.iterrows():
            reds = [row[f'red_{i}'] for i in range(1, 7)]
            red_sums.append(sum(reds))
            red_variances.append(np.var(reds))
            red_spans.append(max(reds) - min(reds))

        # 统计特征
        results['和值统计'] = {
            '均值': np.mean(red_sums),
            '标准差': np.std(red_sums),
            '中位数': np.median(red_sums),
            '偏度': stats.skew(red_sums),
            '峰度': stats.kurtosis(red_sums)
        }

        results['方差统计'] = {
            '均值': np.mean(red_variances),
            '标准差': np.std(red_variances),
            '中位数': np.median(red_variances)
        }

        results['跨度统计'] = {
            '均值': np.mean(red_spans),
            '标准差': np.std(red_spans),
            '中位数': np.median(red_spans)
        }

        # 正态性检验
        _, p_value = stats.normaltest(red_sums)
        results['正态性检验'] = {
            'p值': p_value,
            '是否正态分布': p_value > 0.05
        }

        if explain:
            print(f"  和值统计: 均值={results['和值统计']['均值']:.2f}, 标准差={results['和值统计']['标准差']:.2f}")
            print(f"  偏度={results['和值统计']['偏度']:.3f}, 峰度={results['和值统计']['峰度']:.3f}")
            print(f"  正态性检验: p值={results['正态性检验']['p值']:.4f}, {'符合' if results['正态性检验']['是否正态分布'] else '不符合'}正态分布")

        return results

    def _probability_analysis(self, data, explain=True):
        """概率论分析"""
        results = {}

        # 计算各号码出现概率
        red_counts = {}
        blue_counts = {}

        for i in range(1, 34):
            red_counts[i] = 0
        for i in range(1, 17):
            blue_counts[i] = 0

        for _, row in data.iterrows():
            for i in range(1, 7):
                red_counts[row[f'red_{i}']] += 1
            blue_counts[row['blue_ball']] += 1

        total_red_draws = len(data) * 6
        total_blue_draws = len(data)

        # 计算概率和期望频次
        red_probs = {ball: count / total_red_draws for ball, count in red_counts.items()}
        blue_probs = {ball: count / total_blue_draws for ball, count in blue_counts.items()}

        # 卡方检验（检验是否符合均匀分布）
        chi2_red, p_red = stats.chisquare(list(red_counts.values()))
        chi2_blue, p_blue = stats.chisquare(list(blue_counts.values()))

        results['红球概率分布'] = red_probs
        results['蓝球概率分布'] = blue_probs
        results['红球卡方检验'] = {'卡方值': chi2_red, 'p值': p_red, '均匀分布': p_red > 0.05}
        results['蓝球卡方检验'] = {'卡方值': chi2_blue, 'p值': p_blue, '均匀分布': p_blue > 0.05}

        # 计算信息熵
        red_entropy = -sum(p * np.log2(p) for p in red_probs.values() if p > 0)
        blue_entropy = -sum(p * np.log2(p) for p in blue_probs.values() if p > 0)

        results['信息熵'] = {'红球': red_entropy, '蓝球': blue_entropy}

        if explain:
            print(f"  红球概率分布: 最高概率={max(red_probs.values()):.4f}, 最低概率={min(red_probs.values()):.4f}")
            print(f"  红球卡方检验: p值={p_red:.4f}, {'符合' if p_red > 0.05 else '不符合'}均匀分布")
            print(f"  信息熵: 红球={red_entropy:.3f}, 蓝球={blue_entropy:.3f}")

        return results

    def _bayesian_analysis(self, data, explain=True):
        """贝叶斯分析"""
        results = {}

        # 贝叶斯更新概率
        # 先验概率：假设均匀分布
        red_prior = 1/33
        blue_prior = 1/16

        # 计算后验概率
        red_counts = {}
        blue_counts = {}

        for i in range(1, 34):
            red_counts[i] = 1  # 加1平滑
        for i in range(1, 17):
            blue_counts[i] = 1  # 加1平滑

        for _, row in data.iterrows():
            for i in range(1, 7):
                red_counts[row[f'red_{i}']] += 1
            blue_counts[row['blue_ball']] += 1

        # 贝叶斯后验概率
        total_red = sum(red_counts.values())
        total_blue = sum(blue_counts.values())

        red_posterior = {ball: count / total_red for ball, count in red_counts.items()}
        blue_posterior = {ball: count / total_blue for ball, count in blue_counts.items()}

        # 计算贝叶斯因子
        red_bayes_factors = {}
        blue_bayes_factors = {}

        for ball in range(1, 34):
            likelihood = red_counts[ball] / len(data)
            red_bayes_factors[ball] = likelihood / red_prior

        for ball in range(1, 17):
            likelihood = blue_counts[ball] / len(data)
            blue_bayes_factors[ball] = likelihood / blue_prior

        results['红球后验概率'] = red_posterior
        results['蓝球后验概率'] = blue_posterior
        results['红球贝叶斯因子'] = red_bayes_factors
        results['蓝球贝叶斯因子'] = blue_bayes_factors

        if explain:
            max_red_posterior = max(red_posterior.items(), key=lambda x: x[1])
            max_blue_posterior = max(blue_posterior.items(), key=lambda x: x[1])
            print(f"  红球最高后验概率: {max_red_posterior[0]}号({max_red_posterior[1]:.4f})")
            print(f"  蓝球最高后验概率: {max_blue_posterior[0]}号({max_blue_posterior[1]:.4f})")

            max_red_bf = max(red_bayes_factors.items(), key=lambda x: x[1])
            max_blue_bf = max(blue_bayes_factors.items(), key=lambda x: x[1])
            print(f"  红球最高贝叶斯因子: {max_red_bf[0]}号({max_red_bf[1]:.2f})")
            print(f"  蓝球最高贝叶斯因子: {max_blue_bf[0]}号({max_blue_bf[1]:.2f})")

        return results

    def _hot_cold_analysis(self, data, explain=True):
        """冷热号分布分析"""
        results = {}

        # 计算最近不同周期的出现频率
        periods = [10, 20, 30, 50]

        for period in periods:
            if len(data) >= period:
                recent_data = data.head(period)

                red_counts = {}
                blue_counts = {}

                for i in range(1, 34):
                    red_counts[i] = 0
                for i in range(1, 17):
                    blue_counts[i] = 0

                for _, row in recent_data.iterrows():
                    for i in range(1, 7):
                        red_counts[row[f'red_{i}']] += 1
                    blue_counts[row['blue_ball']] += 1

                # 计算热度指数
                avg_red_freq = sum(red_counts.values()) / 33
                avg_blue_freq = sum(blue_counts.values()) / 16

                red_heat_index = {ball: count / avg_red_freq if avg_red_freq > 0 else 0 for ball, count in red_counts.items()}
                blue_heat_index = {ball: count / avg_blue_freq if avg_blue_freq > 0 else 0 for ball, count in blue_counts.items()}

                # 分类冷热号
                hot_red = [ball for ball, heat in red_heat_index.items() if heat > 1.5]
                warm_red = [ball for ball, heat in red_heat_index.items() if 0.5 <= heat <= 1.5]
                cold_red = [ball for ball, heat in red_heat_index.items() if heat < 0.5]

                hot_blue = [ball for ball, heat in blue_heat_index.items() if heat > 1.5]
                warm_blue = [ball for ball, heat in blue_heat_index.items() if 0.5 <= heat <= 1.5]
                cold_blue = [ball for ball, heat in blue_heat_index.items() if heat < 0.5]

                results[f'{period}期分析'] = {
                    '红球热号': hot_red,
                    '红球温号': warm_red,
                    '红球冷号': cold_red,
                    '蓝球热号': hot_blue,
                    '蓝球温号': warm_blue,
                    '蓝球冷号': cold_blue,
                    '红球热度指数': red_heat_index,
                    '蓝球热度指数': blue_heat_index
                }

        if explain:
            for period in periods:
                if f'{period}期分析' in results:
                    analysis = results[f'{period}期分析']
                    print(f"  {period}期分析: 红球热号{len(analysis['红球热号'])}个, 冷号{len(analysis['红球冷号'])}个")
                    if analysis['红球热号']:
                        print(f"    红球热号: {analysis['红球热号'][:5]}")
                    if analysis['红球冷号']:
                        print(f"    红球冷号: {analysis['红球冷号'][:5]}")

        return results

    def _cycle_analysis(self, data, explain=True):
        """周期性分析"""
        results = {}

        # 分析和值的周期性
        red_sums = []
        for _, row in data.iterrows():
            reds = [row[f'red_{i}'] for i in range(1, 7)]
            red_sums.append(sum(reds))

        # 自相关分析
        max_lag = min(20, len(red_sums) // 3)
        autocorr = []

        for lag in range(1, max_lag + 1):
            if len(red_sums) > lag:
                corr = np.corrcoef(red_sums[:-lag], red_sums[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorr.append((lag, corr))

        # 寻找显著周期
        significant_cycles = [(lag, corr) for lag, corr in autocorr if abs(corr) > 0.1]

        # 傅里叶变换分析周期性
        if len(red_sums) >= 32:
            try:
                fft_result = np.fft.fft(red_sums)
                frequencies = np.fft.fftfreq(len(red_sums))
                power_spectrum = np.abs(fft_result) ** 2

                # 找到主要频率
                main_freq_idx = np.argsort(power_spectrum)[-5:]  # 前5个主要频率
                main_periods = [1/abs(frequencies[i]) if frequencies[i] != 0 else float('inf') for i in main_freq_idx]
                main_periods = [p for p in main_periods if 2 <= p <= len(red_sums)//2]
            except:
                main_periods = []
        else:
            main_periods = []

        results['自相关分析'] = autocorr
        results['显著周期'] = significant_cycles
        results['主要周期'] = main_periods

        if explain:
            print(f"  自相关分析: 发现{len(significant_cycles)}个显著周期")
            if significant_cycles:
                print(f"    最强周期: {significant_cycles[0][0]}期(相关系数{significant_cycles[0][1]:.3f})")
            if main_periods:
                print(f"    傅里叶分析主要周期: {main_periods[:3]}")

        return results

    def _correlation_analysis(self, data, explain=True):
        """相关性分析"""
        results = {}

        # 构建特征矩阵
        features = []
        for _, row in data.iterrows():
            reds = [row[f'red_{i}'] for i in range(1, 7)]
            feature_vector = [
                sum(reds),  # 和值
                max(reds) - min(reds),  # 跨度
                np.var(reds),  # 方差
                sum(1 for x in reds if x % 2 == 1),  # 奇数个数
                sum(1 for x in reds if x >= 17),  # 大数个数
                row['blue_ball']  # 蓝球
            ]
            features.append(feature_vector)

        features = np.array(features)
        feature_names = ['和值', '跨度', '方差', '奇数个数', '大数个数', '蓝球']

        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(features.T)

        # 主成分分析
        try:
            pca = PCA(n_components=min(6, features.shape[1]))
            pca_result = pca.fit_transform(features)
            explained_variance = pca.explained_variance_ratio_
        except:
            explained_variance = []

        # 寻找强相关特征对
        strong_correlations = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.3:  # 相关系数阈值
                    strong_correlations.append((feature_names[i], feature_names[j], corr))

        results['相关系数矩阵'] = corr_matrix.tolist()
        results['特征名称'] = feature_names
        results['强相关特征'] = strong_correlations
        results['主成分方差解释比'] = explained_variance.tolist() if len(explained_variance) > 0 else []

        if explain:
            print(f"  相关性分析: 发现{len(strong_correlations)}对强相关特征")
            for feat1, feat2, corr in strong_correlations[:3]:  # 显示前3个
                print(f"    {feat1} vs {feat2}: 相关系数={corr:.3f}")
            if len(explained_variance) > 0:
                print(f"    主成分分析: 前3个成分解释方差比={explained_variance[:3]}")

        return results

    def _predict_with_hybrid_models(self, hybrid_analysis, latest_reds, latest_blue, periods, prediction_num, explain):
        """基于混合模型的预测方法"""

        if explain:
            print(f"基于{periods}期数据的混合模型预测分析:")

        # 获取各模型分析结果
        stats_analysis = hybrid_analysis['统计学分析']
        prob_analysis = hybrid_analysis['概率论分析']
        markov_analysis = hybrid_analysis['马尔可夫链分析']
        bayes_analysis = hybrid_analysis['贝叶斯分析']
        hot_cold_analysis = hybrid_analysis['冷热号分析']
        cycle_analysis = hybrid_analysis['周期性分析']
        corr_analysis = hybrid_analysis['相关性分析']

        # 初始化候选号码评分系统
        red_scores = {i: 0.0 for i in range(1, 34)}
        blue_scores = {i: 0.0 for i in range(1, 17)}

        if explain:
            print(f"\n混合模型评分计算:")

        # 1. 统计学模型评分 (权重: 15%)
        target_sum = stats_analysis['和值统计']['均值']
        target_variance = stats_analysis['方差统计']['均值']

        for ball in range(1, 34):
            # 基于统计特征的适应性评分
            score = 1.0
            # 如果号码有助于达到目标和值，给予加分
            if abs(ball - target_sum/6) < 5:
                score += 0.2
            red_scores[ball] += score * 0.15

        if explain:
            print(f"  统计学模型: 目标和值={target_sum:.1f}, 目标方差={target_variance:.1f}")

        # 2. 概率论模型评分 (权重: 20%)
        red_probs = prob_analysis['红球概率分布']
        blue_probs = prob_analysis['蓝球概率分布']

        for ball, prob in red_probs.items():
            red_scores[ball] += prob * 20 * 0.20  # 放大概率差异

        for ball, prob in blue_probs.items():
            blue_scores[ball] += prob * 16 * 0.20

        if explain:
            max_red_prob = max(red_probs.items(), key=lambda x: x[1])
            print(f"  概率论模型: 红球最高概率={max_red_prob[0]}号({max_red_prob[1]:.4f})")

        # 3. 马尔可夫链模型评分 (权重: 25%)
        if '红球稳定性转移概率' in markov_analysis:
            red_stability_probs = markov_analysis['红球稳定性转移概率']
            blue_stability_probs = markov_analysis['蓝球稳定性转移概率']

            # 基于当前状态的转移概率
            for current_ball in latest_reds:
                if current_ball in red_stability_probs:
                    for next_ball, info in red_stability_probs[current_ball].items():
                        if isinstance(info, dict) and '概率' in info:
                            red_scores[next_ball] += info['概率'] * 0.25
                        else:
                            red_scores[next_ball] += info * 0.25

            if latest_blue in blue_stability_probs:
                for next_ball, info in blue_stability_probs[latest_blue].items():
                    if isinstance(info, dict) and '概率' in info:
                        blue_scores[next_ball] += info['概率'] * 0.25
                    else:
                        blue_scores[next_ball] += info * 0.25

        if explain:
            print(f"  马尔可夫链模型: 基于当前状态{latest_reds}的转移概率")

        # 4. 贝叶斯模型评分 (权重: 15%)
        red_posterior = bayes_analysis['红球后验概率']
        blue_posterior = bayes_analysis['蓝球后验概率']
        red_bayes_factors = bayes_analysis['红球贝叶斯因子']
        blue_bayes_factors = bayes_analysis['蓝球贝叶斯因子']

        for ball in range(1, 34):
            # 结合后验概率和贝叶斯因子
            posterior_score = red_posterior.get(ball, 0) * 10
            bayes_factor_score = min(red_bayes_factors.get(ball, 1), 3) / 3  # 限制贝叶斯因子影响
            red_scores[ball] += (posterior_score + bayes_factor_score) * 0.15

        for ball in range(1, 17):
            posterior_score = blue_posterior.get(ball, 0) * 10
            bayes_factor_score = min(blue_bayes_factors.get(ball, 1), 3) / 3
            blue_scores[ball] += (posterior_score + bayes_factor_score) * 0.15

        if explain:
            max_red_bf = max(red_bayes_factors.items(), key=lambda x: x[1])
            print(f"  贝叶斯模型: 红球最高贝叶斯因子={max_red_bf[0]}号({max_red_bf[1]:.2f})")

        # 5. 冷热号模型评分 (权重: 15%)
        # 使用最近30期的分析结果
        if '30期分析' in hot_cold_analysis:
            hot_cold_30 = hot_cold_analysis['30期分析']
            red_heat_index = hot_cold_30['红球热度指数']
            blue_heat_index = hot_cold_30['蓝球热度指数']

            for ball, heat in red_heat_index.items():
                # 热号给予正分，冷号给予负分，但保持平衡
                heat_score = (heat - 1.0) * 0.5  # 中心化处理
                red_scores[ball] += heat_score * 0.15

            for ball, heat in blue_heat_index.items():
                heat_score = (heat - 1.0) * 0.5
                blue_scores[ball] += heat_score * 0.15

        if explain:
            hot_red = hot_cold_analysis.get('30期分析', {}).get('红球热号', [])
            print(f"  冷热号模型: 当前热号{len(hot_red)}个, 热号示例={hot_red[:3]}")

        # 6. 周期性模型评分 (权重: 10%)
        significant_cycles = cycle_analysis.get('显著周期', [])
        if significant_cycles:
            # 基于周期性调整评分
            strongest_cycle = significant_cycles[0][0] if significant_cycles else 7

            # 根据周期性模式调整评分
            for ball in range(1, 34):
                cycle_adjustment = 0.1 * np.sin(2 * np.pi * ball / strongest_cycle)
                red_scores[ball] += cycle_adjustment * 0.10

        if explain:
            print(f"  周期性模型: 发现{len(significant_cycles)}个显著周期")

        # 根据预测注数调整选择策略
        choice_offset = (prediction_num - 1) * 0.1  # 后续注数选择次优选项

        # 选择红球 - 基于综合评分
        sorted_red_scores = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)

        predicted_reds = []
        used_balls = set()

        # 选择评分最高的6个红球，考虑预测注数偏移
        for i, (ball, score) in enumerate(sorted_red_scores):
            if len(predicted_reds) >= 6:
                break

            # 为不同注数引入随机性
            if prediction_num > 1 and random.random() < choice_offset:
                continue

            if ball not in used_balls:
                predicted_reds.append(ball)
                used_balls.add(ball)

        predicted_reds.sort()

        # 选择蓝球 - 基于综合评分
        sorted_blue_scores = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)

        # 为不同注数选择不同排名的蓝球
        blue_choice_index = min(prediction_num - 1, len(sorted_blue_scores) - 1)
        predicted_blue = sorted_blue_scores[blue_choice_index][0]

        if explain:
            print(f"\n综合评分结果:")
            print(f"  红球前10评分: {[(ball, f'{score:.3f}') for ball, score in sorted_red_scores[:10]]}")
            print(f"  蓝球前5评分: {[(ball, f'{score:.3f}') for ball, score in sorted_blue_scores[:5]]}")
            print(f"  选中红球: {predicted_reds}")
            print(f"  选中蓝球: {predicted_blue}")

            # 组合特征验证
            current_odd_count = sum(1 for x in latest_reds if x % 2 == 1)
            predicted_odd_count = sum(1 for x in predicted_reds if x % 2 == 1)

            current_big_count = sum(1 for x in latest_reds if x >= 17)
            predicted_big_count = sum(1 for x in predicted_reds if x >= 17)

            current_sum = sum(latest_reds)
            predicted_sum = sum(predicted_reds)

            print(f"\n组合特征验证:")
            print(f"  奇偶比: {current_odd_count}:{6-current_odd_count} -> {predicted_odd_count}:{6-predicted_odd_count}")
            print(f"  大小比: {current_big_count}:{6-current_big_count} -> {predicted_big_count}:{6-predicted_big_count}")
            print(f"  和值: {current_sum} -> {predicted_sum} (目标:{stats_analysis['和值统计']['均值']:.1f})")
            print(f"  跨度: {max(latest_reds) - min(latest_reds)} -> {max(predicted_reds) - min(predicted_reds)}")

        return predicted_reds, predicted_blue


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
    predict_parser.add_argument('--method', choices=['ensemble', 'markov', 'stats', 'probability', 'decision_tree', 'patterns', 'hybrid', 'lstm', 'monte_carlo', 'clustering', 'super'], default='ensemble', help='预测方法')
    predict_parser.add_argument('--count', type=int, default=1, help='预测注数')
    predict_parser.add_argument('--explain', action='store_true', help='显示预测过程')
    predict_parser.add_argument('--periods', type=int, help='指定分析期数（仅适用于hybrid方法）')

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

    # 高级混合分析命令
    hybrid_parser = subparsers.add_parser('hybrid_predict', help='高级混合分析预测')
    hybrid_parser.add_argument('--periods', type=int, help='指定分析期数')
    hybrid_parser.add_argument('--count', type=int, default=1, help='预测注数')
    hybrid_parser.add_argument('--explain', action='store_true', help='显示详细分析过程')

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
        print("正在加载完整历史数据进行马尔可夫链预测...")
        if not analyzer.load_data(force_all_data=True):
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
        print("正在加载完整历史数据进行预测...")
        if not analyzer.load_data(force_all_data=True):
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
        elif args.method == 'hybrid':
            predictions = analyzer.predict_by_advanced_hybrid_analysis(
                periods=args.periods,
                count=args.count,
                explain=args.explain
            )
        elif args.method == 'lstm':
            for i in range(args.count):
                red_balls, blue_ball = analyzer.predict_by_lstm(explain=args.explain and i == 0)
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")
        elif args.method == 'monte_carlo':
            for i in range(args.count):
                red_balls, blue_ball = analyzer.predict_by_monte_carlo(explain=args.explain and i == 0)
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")
        elif args.method == 'clustering':
            for i in range(args.count):
                red_balls, blue_ball = analyzer.predict_by_clustering(explain=args.explain and i == 0)
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")
        elif args.method == 'super':
            for i in range(args.count):
                red_balls, blue_ball = analyzer.predict_by_super(explain=args.explain and i == 0)
                formatted = analyzer.format_numbers(red_balls, blue_ball)
                print(f"第{i+1}注: {formatted}")

    elif args.command == 'generate':
        # 生成号码
        print("正在加载完整历史数据进行号码生成...")
        if not analyzer.load_data(force_all_data=True):
            print("加载数据失败")
            return

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

    elif args.command == 'hybrid_predict':
        # 高级混合分析预测
        print("正在加载完整历史数据进行高级混合分析预测...")
        if not analyzer.load_data(force_all_data=True):
            print("加载数据失败")
            return

        predictions = analyzer.predict_by_advanced_hybrid_analysis(
            periods=args.periods,
            count=args.count,
            explain=args.explain
        )

    else:
        # 显示帮助信息
        parser.print_help()


    # ==================== 高级混合分析预测 ====================

    def predict_by_advanced_hybrid_analysis(self, periods=None, count=1, explain=True):
        """
        基于多种数学模型的高级混合分析预测
        整合统计学、概率论、马尔可夫链、贝叶斯分析、冷热号分布等方法

        Args:
            periods: 指定分析期数
            count: 预测注数
            explain: 是否显示详细分析过程

        Returns:
            预测结果列表
        """
        if self.data is None:
            if not self.load_data():
                return []

        print(f"\n{'='*70}")
        print(f"高级混合分析预测系统")
        print(f"整合统计学、概率论、马尔可夫链、贝叶斯分析、冷热号分布等数学模型")
        print(f"{'='*70}")

        # 确定分析数据范围
        if periods:
            if len(self.data) < periods:
                print(f"警告: 可用数据({len(self.data)}期)少于指定期数({periods}期)，将使用全部数据")
                analysis_data = self.data.copy()
                actual_periods = len(analysis_data)
            else:
                analysis_data = self.data.head(periods).copy()
                actual_periods = periods
                print(f"使用最近{periods}期数据进行高级混合分析")
        else:
            analysis_data = self.data.copy()
            actual_periods = len(analysis_data)
            print(f"使用全部{len(analysis_data)}期数据进行高级混合分析")

        print(f"分析数据期数: {actual_periods}期")
        print(f"数据范围: {analysis_data.iloc[-1]['issue']}期 - {analysis_data.iloc[0]['issue']}期")

        # 获取最近一期作为预测基础
        latest_data = analysis_data.iloc[0]
        latest_reds = [latest_data[f'red_{i}'] for i in range(1, 7)]
        latest_blue = latest_data['blue_ball']

        print(f"\n预测基础数据:")
        print(f"最近一期: {latest_data['issue']}期 ({latest_data['date']})")
        print(f"开奖号码: 红球 {' '.join([f'{ball:02d}' for ball in latest_reds])} | 蓝球 {latest_blue:02d}")

        # 执行综合分析
        if explain:
            print(f"\n开始执行多维度数学模型分析...")

        hybrid_analysis = self._execute_hybrid_analysis(analysis_data, explain)

        # 进行多注预测
        predictions = []

        for i in range(count):
            print(f"\n{'='*50}")
            print(f"第{i+1}注高级混合分析预测")
            print(f"{'='*50}")

            predicted_reds, predicted_blue = self._predict_with_hybrid_models(
                hybrid_analysis, latest_reds, latest_blue, actual_periods, i+1, explain
            )

            predictions.append((predicted_reds, predicted_blue))

            formatted = self.format_numbers(predicted_reds, predicted_blue)
            print(f"\n第{i+1}注混合模型预测结果: {formatted}")

        # 显示预测汇总
        print(f"\n{'='*70}")
        print(f"高级混合分析预测汇总")
        print(f"{'='*70}")
        print(f"分析期数: {actual_periods}期")
        print(f"预测注数: {count}注")
        print(f"预测基础: {latest_data['issue']}期")
        print(f"分析模型: 统计学+概率论+马尔可夫链+贝叶斯+冷热号分布")

        for i, (red_balls, blue_ball) in enumerate(predictions):
            formatted = self.format_numbers(red_balls, blue_ball)
            print(f"第{i+1}注: {formatted}")

        return predictions

    def _execute_hybrid_analysis(self, data, explain=True):
        """执行多维度数学模型分析"""
        analysis_results = {}

        if explain:
            print(f"\n【第一阶段：统计学分析】")

        # 1. 统计学分析
        stats_analysis = self._statistical_analysis(data, explain)
        analysis_results['统计学分析'] = stats_analysis

        if explain:
            print(f"\n【第二阶段：概率论分析】")

        # 2. 概率论分析
        prob_analysis = self._probability_analysis(data, explain)
        analysis_results['概率论分析'] = prob_analysis

        if explain:
            print(f"\n【第三阶段：马尔可夫链分析】")

        # 3. 马尔可夫链分析
        markov_analysis = self._analyze_markov_chain_stability(data)
        analysis_results['马尔可夫链分析'] = markov_analysis

        if explain:
            print(f"\n【第四阶段：贝叶斯分析】")

        # 4. 贝叶斯分析
        bayes_analysis = self._bayesian_analysis(data, explain)
        analysis_results['贝叶斯分析'] = bayes_analysis

        if explain:
            print(f"\n【第五阶段：冷热号分布分析】")

        # 5. 冷热号分布分析
        hot_cold_analysis = self._hot_cold_analysis(data, explain)
        analysis_results['冷热号分析'] = hot_cold_analysis

        if explain:
            print(f"\n【第六阶段：周期性分析】")

        # 6. 周期性分析
        cycle_analysis = self._cycle_analysis(data, explain)
        analysis_results['周期性分析'] = cycle_analysis

        if explain:
            print(f"\n【第七阶段：相关性分析】")

        # 7. 相关性分析
        correlation_analysis = self._correlation_analysis(data, explain)
        analysis_results['相关性分析'] = correlation_analysis

        return analysis_results

    def _statistical_analysis(self, data, explain=True):
        """统计学分析"""
        results = {}

        # 计算基本统计量
        red_sums = []
        red_variances = []
        red_spans = []

        for _, row in data.iterrows():
            reds = [row[f'red_{i}'] for i in range(1, 7)]
            red_sums.append(sum(reds))
            red_variances.append(np.var(reds))
            red_spans.append(max(reds) - min(reds))

        # 统计特征
        results['和值统计'] = {
            '均值': np.mean(red_sums),
            '标准差': np.std(red_sums),
            '中位数': np.median(red_sums),
            '众数': stats.mode(red_sums)[0] if len(red_sums) > 0 else 0,
            '偏度': stats.skew(red_sums),
            '峰度': stats.kurtosis(red_sums)
        }

        results['方差统计'] = {
            '均值': np.mean(red_variances),
            '标准差': np.std(red_variances),
            '中位数': np.median(red_variances)
        }

        results['跨度统计'] = {
            '均值': np.mean(red_spans),
            '标准差': np.std(red_spans),
            '中位数': np.median(red_spans)
        }

        # 正态性检验
        _, p_value = stats.normaltest(red_sums)
        results['正态性检验'] = {
            'p值': p_value,
            '是否正态分布': p_value > 0.05
        }

        if explain:
            print(f"  和值统计: 均值={results['和值统计']['均值']:.2f}, 标准差={results['和值统计']['标准差']:.2f}")
            print(f"  偏度={results['和值统计']['偏度']:.3f}, 峰度={results['和值统计']['峰度']:.3f}")
            print(f"  正态性检验: p值={results['正态性检验']['p值']:.4f}, {'符合' if results['正态性检验']['是否正态分布'] else '不符合'}正态分布")

        return results

    def _probability_analysis(self, data, explain=True):
        """概率论分析"""
        results = {}

        # 计算各号码出现概率
        red_counts = {}
        blue_counts = {}

        for i in range(1, 34):
            red_counts[i] = 0
        for i in range(1, 17):
            blue_counts[i] = 0

        for _, row in data.iterrows():
            for i in range(1, 7):
                red_counts[row[f'red_{i}']] += 1
            blue_counts[row['blue_ball']] += 1

        total_red_draws = len(data) * 6
        total_blue_draws = len(data)

        # 计算概率和期望频次
        red_probs = {ball: count / total_red_draws for ball, count in red_counts.items()}
        blue_probs = {ball: count / total_blue_draws for ball, count in blue_counts.items()}

        # 卡方检验（检验是否符合均匀分布）
        expected_red = total_red_draws / 33
        expected_blue = total_blue_draws / 16

        chi2_red, p_red = stats.chisquare(list(red_counts.values()))
        chi2_blue, p_blue = stats.chisquare(list(blue_counts.values()))

        results['红球概率分布'] = red_probs
        results['蓝球概率分布'] = blue_probs
        results['红球卡方检验'] = {'卡方值': chi2_red, 'p值': p_red, '均匀分布': p_red > 0.05}
        results['蓝球卡方检验'] = {'卡方值': chi2_blue, 'p值': p_blue, '均匀分布': p_blue > 0.05}

        # 计算信息熵
        red_entropy = -sum(p * np.log2(p) for p in red_probs.values() if p > 0)
        blue_entropy = -sum(p * np.log2(p) for p in blue_probs.values() if p > 0)

        results['信息熵'] = {'红球': red_entropy, '蓝球': blue_entropy}

        if explain:
            print(f"  红球概率分布: 最高概率={max(red_probs.values()):.4f}, 最低概率={min(red_probs.values()):.4f}")
            print(f"  红球卡方检验: p值={p_red:.4f}, {'符合' if p_red > 0.05 else '不符合'}均匀分布")
            print(f"  信息熵: 红球={red_entropy:.3f}, 蓝球={blue_entropy:.3f}")

        return results

    def _bayesian_analysis(self, data, explain=True):
        """贝叶斯分析"""
        results = {}

        # 贝叶斯更新概率
        # 先验概率：假设均匀分布
        red_prior = 1/33
        blue_prior = 1/16

        # 计算后验概率
        red_counts = {}
        blue_counts = {}

        for i in range(1, 34):
            red_counts[i] = 1  # 加1平滑
        for i in range(1, 17):
            blue_counts[i] = 1  # 加1平滑

        for _, row in data.iterrows():
            for i in range(1, 7):
                red_counts[row[f'red_{i}']] += 1
            blue_counts[row['blue_ball']] += 1

        # 贝叶斯后验概率
        total_red = sum(red_counts.values())
        total_blue = sum(blue_counts.values())

        red_posterior = {ball: count / total_red for ball, count in red_counts.items()}
        blue_posterior = {ball: count / total_blue for ball, count in blue_counts.items()}

        # 计算贝叶斯因子
        red_bayes_factors = {}
        blue_bayes_factors = {}

        for ball in range(1, 34):
            likelihood = red_counts[ball] / len(data)
            red_bayes_factors[ball] = likelihood / red_prior

        for ball in range(1, 17):
            likelihood = blue_counts[ball] / len(data)
            blue_bayes_factors[ball] = likelihood / blue_prior

        results['红球后验概率'] = red_posterior
        results['蓝球后验概率'] = blue_posterior
        results['红球贝叶斯因子'] = red_bayes_factors
        results['蓝球贝叶斯因子'] = blue_bayes_factors

        if explain:
            max_red_posterior = max(red_posterior.items(), key=lambda x: x[1])
            max_blue_posterior = max(blue_posterior.items(), key=lambda x: x[1])
            print(f"  红球最高后验概率: {max_red_posterior[0]}号({max_red_posterior[1]:.4f})")
            print(f"  蓝球最高后验概率: {max_blue_posterior[0]}号({max_blue_posterior[1]:.4f})")

            max_red_bf = max(red_bayes_factors.items(), key=lambda x: x[1])
            max_blue_bf = max(blue_bayes_factors.items(), key=lambda x: x[1])
            print(f"  红球最高贝叶斯因子: {max_red_bf[0]}号({max_red_bf[1]:.2f})")
            print(f"  蓝球最高贝叶斯因子: {max_blue_bf[0]}号({max_blue_bf[1]:.2f})")

        return results

    def _hot_cold_analysis(self, data, explain=True):
        """冷热号分布分析"""
        results = {}

        # 计算最近不同周期的出现频率
        periods = [10, 20, 30, 50]

        for period in periods:
            if len(data) >= period:
                recent_data = data.head(period)

                red_counts = {}
                blue_counts = {}

                for i in range(1, 34):
                    red_counts[i] = 0
                for i in range(1, 17):
                    blue_counts[i] = 0

                for _, row in recent_data.iterrows():
                    for i in range(1, 7):
                        red_counts[row[f'red_{i}']] += 1
                    blue_counts[row['blue_ball']] += 1

                # 计算热度指数
                avg_red_freq = sum(red_counts.values()) / 33
                avg_blue_freq = sum(blue_counts.values()) / 16

                red_heat_index = {ball: count / avg_red_freq for ball, count in red_counts.items()}
                blue_heat_index = {ball: count / avg_blue_freq for ball, count in blue_counts.items()}

                # 分类冷热号
                hot_red = [ball for ball, heat in red_heat_index.items() if heat > 1.5]
                warm_red = [ball for ball, heat in red_heat_index.items() if 0.5 <= heat <= 1.5]
                cold_red = [ball for ball, heat in red_heat_index.items() if heat < 0.5]

                hot_blue = [ball for ball, heat in blue_heat_index.items() if heat > 1.5]
                warm_blue = [ball for ball, heat in blue_heat_index.items() if 0.5 <= heat <= 1.5]
                cold_blue = [ball for ball, heat in blue_heat_index.items() if heat < 0.5]

                results[f'{period}期分析'] = {
                    '红球热号': hot_red,
                    '红球温号': warm_red,
                    '红球冷号': cold_red,
                    '蓝球热号': hot_blue,
                    '蓝球温号': warm_blue,
                    '蓝球冷号': cold_blue,
                    '红球热度指数': red_heat_index,
                    '蓝球热度指数': blue_heat_index
                }

        if explain:
            for period in periods:
                if f'{period}期分析' in results:
                    analysis = results[f'{period}期分析']
                    print(f"  {period}期分析: 红球热号{len(analysis['红球热号'])}个, 冷号{len(analysis['红球冷号'])}个")
                    print(f"    红球热号: {analysis['红球热号'][:5]}...")  # 显示前5个
                    print(f"    红球冷号: {analysis['红球冷号'][:5]}...")

        return results

    def _cycle_analysis(self, data, explain=True):
        """周期性分析"""
        results = {}

        # 分析和值的周期性
        red_sums = []
        for _, row in data.iterrows():
            reds = [row[f'red_{i}'] for i in range(1, 7)]
            red_sums.append(sum(reds))

        # 自相关分析
        max_lag = min(20, len(red_sums) // 3)
        autocorr = []

        for lag in range(1, max_lag + 1):
            if len(red_sums) > lag:
                corr = np.corrcoef(red_sums[:-lag], red_sums[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorr.append((lag, corr))

        # 寻找显著周期
        significant_cycles = [(lag, corr) for lag, corr in autocorr if abs(corr) > 0.1]

        # 傅里叶变换分析周期性
        if len(red_sums) >= 32:
            fft_result = np.fft.fft(red_sums)
            frequencies = np.fft.fftfreq(len(red_sums))
            power_spectrum = np.abs(fft_result) ** 2

            # 找到主要频率
            main_freq_idx = np.argsort(power_spectrum)[-5:]  # 前5个主要频率
            main_periods = [1/abs(frequencies[i]) if frequencies[i] != 0 else float('inf') for i in main_freq_idx]
            main_periods = [p for p in main_periods if 2 <= p <= len(red_sums)//2]
        else:
            main_periods = []

        results['自相关分析'] = autocorr
        results['显著周期'] = significant_cycles
        results['主要周期'] = main_periods

        if explain:
            print(f"  自相关分析: 发现{len(significant_cycles)}个显著周期")
            if significant_cycles:
                print(f"    最强周期: {significant_cycles[0][0]}期(相关系数{significant_cycles[0][1]:.3f})")
            if main_periods:
                print(f"    傅里叶分析主要周期: {main_periods[:3]}")

        return results

    def _correlation_analysis(self, data, explain=True):
        """相关性分析"""
        results = {}

        # 构建特征矩阵
        features = []
        for _, row in data.iterrows():
            reds = [row[f'red_{i}'] for i in range(1, 7)]
            feature_vector = [
                sum(reds),  # 和值
                max(reds) - min(reds),  # 跨度
                np.var(reds),  # 方差
                sum(1 for x in reds if x % 2 == 1),  # 奇数个数
                sum(1 for x in reds if x >= 17),  # 大数个数
                row['blue_ball']  # 蓝球
            ]
            features.append(feature_vector)

        features = np.array(features)
        feature_names = ['和值', '跨度', '方差', '奇数个数', '大数个数', '蓝球']

        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(features.T)

        # 主成分分析
        try:
            pca = PCA(n_components=min(6, features.shape[1]))
            pca_result = pca.fit_transform(features)
            explained_variance = pca.explained_variance_ratio_
        except:
            explained_variance = []

        # 寻找强相关特征对
        strong_correlations = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.3:  # 相关系数阈值
                    strong_correlations.append((feature_names[i], feature_names[j], corr))

        results['相关系数矩阵'] = corr_matrix.tolist()
        results['特征名称'] = feature_names
        results['强相关特征'] = strong_correlations
        results['主成分方差解释比'] = explained_variance.tolist() if len(explained_variance) > 0 else []

        if explain:
            print(f"  相关性分析: 发现{len(strong_correlations)}对强相关特征")
            for feat1, feat2, corr in strong_correlations[:3]:  # 显示前3个
                print(f"    {feat1} vs {feat2}: 相关系数={corr:.3f}")
            if len(explained_variance) > 0:
                print(f"    主成分分析: 前3个成分解释方差比={explained_variance[:3]}")

        return results

    def _predict_with_hybrid_models(self, hybrid_analysis, latest_reds, latest_blue, periods, prediction_num, explain):
        """基于混合模型的预测方法"""

        if explain:
            print(f"基于{periods}期数据的混合模型预测分析:")

        # 获取各模型分析结果
        stats_analysis = hybrid_analysis['统计学分析']
        prob_analysis = hybrid_analysis['概率论分析']
        markov_analysis = hybrid_analysis['马尔可夫链分析']
        bayes_analysis = hybrid_analysis['贝叶斯分析']
        hot_cold_analysis = hybrid_analysis['冷热号分析']
        cycle_analysis = hybrid_analysis['周期性分析']
        corr_analysis = hybrid_analysis['相关性分析']

        # 初始化候选号码评分系统
        red_scores = {i: 0.0 for i in range(1, 34)}
        blue_scores = {i: 0.0 for i in range(1, 17)}

        if explain:
            print(f"\n混合模型评分计算:")

        # 1. 统计学模型评分 (权重: 15%)
        target_sum = stats_analysis['和值统计']['均值']
        target_variance = stats_analysis['方差统计']['均值']

        for ball in range(1, 34):
            # 基于统计特征的适应性评分
            score = 1.0
            # 如果号码有助于达到目标和值，给予加分
            if abs(ball - target_sum/6) < 5:
                score += 0.2
            red_scores[ball] += score * 0.15

        if explain:
            print(f"  统计学模型: 目标和值={target_sum:.1f}, 目标方差={target_variance:.1f}")

        # 2. 概率论模型评分 (权重: 20%)
        red_probs = prob_analysis['红球概率分布']
        blue_probs = prob_analysis['蓝球概率分布']

        for ball, prob in red_probs.items():
            red_scores[ball] += prob * 20 * 0.20  # 放大概率差异

        for ball, prob in blue_probs.items():
            blue_scores[ball] += prob * 16 * 0.20

        if explain:
            max_red_prob = max(red_probs.items(), key=lambda x: x[1])
            print(f"  概率论模型: 红球最高概率={max_red_prob[0]}号({max_red_prob[1]:.4f})")

        # 3. 马尔可夫链模型评分 (权重: 25%)
        if '红球稳定性转移概率' in markov_analysis:
            red_stability_probs = markov_analysis['红球稳定性转移概率']
            blue_stability_probs = markov_analysis['蓝球稳定性转移概率']

            # 基于当前状态的转移概率
            for current_ball in latest_reds:
                if current_ball in red_stability_probs:
                    for next_ball, info in red_stability_probs[current_ball].items():
                        if isinstance(info, dict) and '概率' in info:
                            red_scores[next_ball] += info['概率'] * 0.25
                        else:
                            red_scores[next_ball] += info * 0.25

            if latest_blue in blue_stability_probs:
                for next_ball, info in blue_stability_probs[latest_blue].items():
                    if isinstance(info, dict) and '概率' in info:
                        blue_scores[next_ball] += info['概率'] * 0.25
                    else:
                        blue_scores[next_ball] += info * 0.25

        if explain:
            print(f"  马尔可夫链模型: 基于当前状态{latest_reds}的转移概率")

        # 4. 贝叶斯模型评分 (权重: 15%)
        red_posterior = bayes_analysis['红球后验概率']
        blue_posterior = bayes_analysis['蓝球后验概率']
        red_bayes_factors = bayes_analysis['红球贝叶斯因子']
        blue_bayes_factors = bayes_analysis['蓝球贝叶斯因子']

        for ball in range(1, 34):
            # 结合后验概率和贝叶斯因子
            posterior_score = red_posterior.get(ball, 0) * 10
            bayes_factor_score = min(red_bayes_factors.get(ball, 1), 3) / 3  # 限制贝叶斯因子影响
            red_scores[ball] += (posterior_score + bayes_factor_score) * 0.15

        for ball in range(1, 17):
            posterior_score = blue_posterior.get(ball, 0) * 10
            bayes_factor_score = min(blue_bayes_factors.get(ball, 1), 3) / 3
            blue_scores[ball] += (posterior_score + bayes_factor_score) * 0.15

        if explain:
            max_red_bf = max(red_bayes_factors.items(), key=lambda x: x[1])
            print(f"  贝叶斯模型: 红球最高贝叶斯因子={max_red_bf[0]}号({max_red_bf[1]:.2f})")

        # 5. 冷热号模型评分 (权重: 15%)
        # 使用最近30期的分析结果
        if '30期分析' in hot_cold_analysis:
            hot_cold_30 = hot_cold_analysis['30期分析']
            red_heat_index = hot_cold_30['红球热度指数']
            blue_heat_index = hot_cold_30['蓝球热度指数']

            for ball, heat in red_heat_index.items():
                # 热号给予正分，冷号给予负分，但保持平衡
                heat_score = (heat - 1.0) * 0.5  # 中心化处理
                red_scores[ball] += heat_score * 0.15

            for ball, heat in blue_heat_index.items():
                heat_score = (heat - 1.0) * 0.5
                blue_scores[ball] += heat_score * 0.15

        if explain:
            hot_red = hot_cold_analysis.get('30期分析', {}).get('红球热号', [])
            print(f"  冷热号模型: 当前热号{len(hot_red)}个, 热号示例={hot_red[:3]}")

        # 6. 周期性模型评分 (权重: 10%)
        significant_cycles = cycle_analysis.get('显著周期', [])
        if significant_cycles:
            # 基于周期性调整评分
            strongest_cycle = significant_cycles[0][0] if significant_cycles else 7

            # 根据周期性模式调整评分
            for ball in range(1, 34):
                cycle_adjustment = 0.1 * np.sin(2 * np.pi * ball / strongest_cycle)
                red_scores[ball] += cycle_adjustment * 0.10

        if explain:
            print(f"  周期性模型: 发现{len(significant_cycles)}个显著周期")

        # 根据预测注数调整选择策略
        choice_offset = (prediction_num - 1) * 0.1  # 后续注数选择次优选项

        # 选择红球 - 基于综合评分
        sorted_red_scores = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)

        predicted_reds = []
        used_balls = set()

        # 选择评分最高的6个红球，考虑预测注数偏移
        for i, (ball, score) in enumerate(sorted_red_scores):
            if len(predicted_reds) >= 6:
                break

            # 为不同注数引入随机性
            if prediction_num > 1 and random.random() < choice_offset:
                continue

            if ball not in used_balls:
                predicted_reds.append(ball)
                used_balls.add(ball)

        predicted_reds.sort()

        # 选择蓝球 - 基于综合评分
        sorted_blue_scores = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)

        # 为不同注数选择不同排名的蓝球
        blue_choice_index = min(prediction_num - 1, len(sorted_blue_scores) - 1)
        predicted_blue = sorted_blue_scores[blue_choice_index][0]

        if explain:
            print(f"\n综合评分结果:")
            print(f"  红球前10评分: {[(ball, f'{score:.3f}') for ball, score in sorted_red_scores[:10]]}")
            print(f"  蓝球前5评分: {[(ball, f'{score:.3f}') for ball, score in sorted_blue_scores[:5]]}")
            print(f"  选中红球: {predicted_reds}")
            print(f"  选中蓝球: {predicted_blue}")

            # 组合特征验证
            current_odd_count = sum(1 for x in latest_reds if x % 2 == 1)
            predicted_odd_count = sum(1 for x in predicted_reds if x % 2 == 1)

            current_big_count = sum(1 for x in latest_reds if x >= 17)
            predicted_big_count = sum(1 for x in predicted_reds if x >= 17)

            current_sum = sum(latest_reds)
            predicted_sum = sum(predicted_reds)

            print(f"\n组合特征验证:")
            print(f"  奇偶比: {current_odd_count}:{6-current_odd_count} -> {predicted_odd_count}:{6-predicted_odd_count}")
            print(f"  大小比: {current_big_count}:{6-current_big_count} -> {predicted_big_count}:{6-predicted_big_count}")
            print(f"  和值: {current_sum} -> {predicted_sum} (目标:{stats_analysis['和值统计']['均值']:.1f})")
            print(f"  跨度: {max(latest_reds) - min(latest_reds)} -> {max(predicted_reds) - min(predicted_reds)}")

        return predicted_reds, predicted_blue


if __name__ == "__main__":
    main()
