#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 中国福利彩票官方网站版本
从中国福利彩票官方网站获取最近300期双色球开奖结果
如果官方网站数据不足，则从500彩票网获取补充数据
"""

import os
import csv
import time
import json
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime


class SSQCWLCrawler:
    """双色球数据获取类 - 中国福利彩票官方网站版本"""

    def __init__(self, data_dir="../data"):
        """
        初始化数据获取器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 中国福利彩票官方网站API
        self.api_url = "https://www.cwl.gov.cn"
        
        # 500彩票网双色球历史数据URL
        self.cp500_url = "https://datachart.500.com/ssq/history/history.shtml"
        
        # 请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://www.cwl.gov.cn"
        }
        
        # 500彩票网请求头
        self.cp500_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://datachart.500.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
        }
    
    def get_history_data_from_cwl(self, count=None):
        """
        从中国福利彩票官方网站获取历史开奖数据
        
        Args:
            count: 获取的记录数量，默认None表示获取所有期数

        Returns:
            开奖结果列表
        """
        results = []
        
        try:
            print(f"正在从中国福利彩票官方网站获取双色球开奖结果...")
            
            # 计算需要请求的页数 (每页30条数据)
            page_size = 30
            total_pages = (count + page_size - 1) // page_size if count else 100  # 向上取整，如果没有指定count，默认尝试获取100页
            
            # 使用分页方式获取数据
            for page in range(1, total_pages + 1):
                print(f"正在获取第{page}页数据 (每页{page_size}条)...")
                
                # 设置请求参数 - 使用分页方式
                params = {
                    "name": "ssq",  # 双色球
                    "pageNo": page,  # 页码
                    "pageSize": page_size,  # 每页数量
                    "systemType": "PC"  # 系统类型
                }
                
                # 添加重试机制
                max_retries = 3
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        # 发送请求
                    response = requests.get(self.api_url, headers=self.headers, params=params, timeout=15)
                        response.raise_for_status()
                        
                        # 解析JSON数据
                        data = response.json()
                        success = True
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"请求失败，正在进行第{retry_count}次重试: {e}")
                            time.sleep(2 * retry_count)  # 指数退避
                        else:
                            print(f"请求失败，已达到最大重试次数: {e}")
                            raise
                
                # 如果所有重试都失败，跳过当前页
                if not success:
                    continue
                
                # 检查是否有结果数据
                if "result" in data and isinstance(data["result"], list) and len(data["result"]) > 0:
                    # 提取开奖结果
                    for item in data["result"]:
                        try:
                            issue = item["code"]  # 期号
                            date = item["date"]  # 开奖日期
                            
                            # 获取红球号码（格式为 "01,02,03,04,05,06"）
                            red_str = item["red"]
                            red_balls = red_str.split(",")
                            
                            # 获取蓝球号码
                            blue_ball = item["blue"]
                            
                            # 确保所有号码都是两位数格式
                            red_balls = [ball.zfill(2) for ball in red_balls]
                            blue_ball = blue_ball.zfill(2)
                            
                            results.append({
                                "issue": issue,
                                "date": date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_ball
                            })
                        except Exception as e:
                            print(f"解析数据项失败: {e}")
                            continue
                    
                    print(f"成功获取第{page}页数据，当前共{len(results)}期")
                    
                    # 如果指定了期数限制且已经获取足够的数据，则退出循环
                    if count is not None and len(results) >= count:
                        break
                    
                    # 添加随机延迟，避免请求过于频繁
                    time.sleep(random.uniform(1, 3))
                else:
                    error_msg = data.get("message", "未知错误")
                    print(f"获取第{page}页数据失败: {error_msg}")
                    break
                
                # 如果当前页的数据量小于page_size，说明已经没有更多数据了
                if len(data.get("result", [])) < page_size:
                    break
            
            print(f"从中国福利彩票官方网站成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中国福利彩票官方网站获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 按期号排序（降序）
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            # 只保留需要的期数（如果指定了count）
            if count is not None and len(results) > count:
                results = results[:count]
        
        return results
    
    def get_history_data_from_500cp(self):
        """
        从500彩票网获取双色球历史数据
        
        Returns:
            开奖结果列表
        """
        results = []
        
        try:
            print(f"正在从500彩票网获取双色球历史数据...")
            
            # 使用带参数的URL - 500彩票网的历史数据页面
            # 使用start和end参数控制期数范围，从03001(2003年第1期)到最新期
            # 最新期可以使用一个较大的数字，如25001(2025年第1期)
            history_url = "https://datachart.500.com/ssq/history/newinc/history.php?start=03001&end=25001"
            
            # 添加重试机制
            max_retries = 3
            retry_count = 0
            success = False
            html_content = ""
            
            # 尝试带参数的URL
            while retry_count < max_retries and not success:
                try:
                    # 发送请求，禁用代理
                    session = requests.Session()
                    session.trust_env = False  # 禁用环境变量中的代理设置
                    response = session.get(history_url, headers=self.cp500_headers, timeout=30)
                    # 设置正确的编码
                    response.encoding = 'gb2312'
                    html_content = response.text
                    success = True
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"请求500彩票网历史数据URL失败，正在进行第{retry_count}次重试: {e}")
                        time.sleep(2 * retry_count)  # 指数退避
                    else:
                        print(f"请求500彩票网历史数据URL失败，将尝试备用方法: {e}")
            
            # 如果带参数的URL失败，尝试原始URL
            if not success:
                retry_count = 0
                while retry_count < max_retries and not success:
                    try:
                        # 发送请求，禁用代理
                        session = requests.Session()
                        session.trust_env = False  # 禁用环境变量中的代理设置
                        response = session.get(self.cp500_url, headers=self.cp500_headers, timeout=15)
                        # 设置正确的编码
                        response.encoding = 'gb2312'
                        html_content = response.text
                        success = True
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"请求500彩票网主URL失败，正在进行第{retry_count}次重试: {e}")
                            time.sleep(2 * retry_count)  # 指数退避
                        else:
                            print(f"请求500彩票网主URL失败，已达到最大重试次数: {e}")
                            return results
            
            # 如果所有URL都失败，返回空结果
            if not success or not html_content:
                return results
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(html_content, "html.parser")
            
            # 尝试查找不同格式的表格
            tables = []
            tables.append(soup.find('table', attrs={'width': '100%'}))
            tables.append(soup.find('table', attrs={'class': 'history_table'}))
            tables.append(soup.find('table', attrs={'id': 'tdata'}))
            
            # 过滤掉None值
            tables = [t for t in tables if t]
            
            if not tables:
                print("未找到数据表格")
                return results
            
            # 遍历所有找到的表格
            for table in tables:
                # 获取表格中的所有行
                rows = table.find_all('tr')
                if len(rows) <= 1:
                    continue
                
                # 跳过表头行，解析数据行
                for row in rows[1:]:
                    try:
                        cells = row.find_all('td')
                        if len(cells) < 9:
                            continue
                        
                        # 获取期号
                        issue = cells[0].text.strip()
                        # 如果期号不是纯数字，尝试提取数字部分
                        if not issue.isdigit():
                            import re
                            issue_match = re.search(r'\d+', issue)
                            if issue_match:
                                issue = issue_match.group(0)
                            else:
                                continue
                        
                        # 获取开奖日期
                        date = cells[1].text.strip()
                        
                        # 获取红球
                        red_balls = []
                        for i in range(2, 8):
                            if i < len(cells):
                                ball_text = cells[i].text.strip()
                                # 如果球号包含非数字字符，尝试提取数字部分
                                if not ball_text.isdigit():
                                    import re
                                    ball_match = re.search(r'\d+', ball_text)
                                    if ball_match:
                                        ball_text = ball_match.group(0)
                                red_balls.append(ball_text.zfill(2))  # 补0，保持两位数格式
                        
                        # 确保有6个红球
                        if len(red_balls) != 6:
                            continue
                        
                        # 获取蓝球
                        if 8 < len(cells):
                            blue_ball = cells[8].text.strip()
                            # 如果球号包含非数字字符，尝试提取数字部分
                            if not blue_ball.isdigit():
                                import re
                                ball_match = re.search(r'\d+', blue_ball)
                                if ball_match:
                                    blue_ball = ball_match.group(0)
                            blue_ball = blue_ball.zfill(2)  # 补0，保持两位数格式
                        else:
                            continue
                        
                        results.append({
                            "issue": issue,
                            "date": date,
                            "red_balls": ",".join(red_balls),
                            "blue_ball": blue_ball
                        })
                    except Exception as e:
                        print(f"解析行数据失败: {e}")
                        continue
            
            # 去重并验证数据
            unique_results = []
            seen_issues = set()
            for item in results:
                # 验证期号是否为有效数字
                if not item["issue"].isdigit():
                    continue
                
                # 验证红球和蓝球格式
                red_balls = item["red_balls"].split(",")
                if len(red_balls) != 6:
                    continue
                
                # 检查是否有重复
                if item["issue"] not in seen_issues:
                    unique_results.append(item)
                    seen_issues.add(item["issue"])
            
            results = unique_results
            print(f"从500彩票网成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从500彩票网获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        return results

    def get_history_data(self, count=None):
        """
        获取双色球历史数据，优先从官方网站获取，不足则从500彩票网补充

        Args:
            count: 获取的记录数量，默认None表示获取所有期数

        Returns:
            开奖结果列表
        """
        results = []
        
        # 尝试从官方网站获取数据
        try:
            results = self.get_history_data_from_cwl(count)
            print(f"从官方网站成功获取了{len(results)}期数据")
        except Exception as e:
            print(f"从官方网站获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 尝试从500彩票网获取补充数据
        try:
            print(f"将从500彩票网获取补充数据...")
            results_500cp = self.get_history_data_from_500cp()
            print(f"从500彩票网成功获取了{len(results_500cp)}期数据")
            
            # 合并数据，去重
            existing_issues = set(item["issue"] for item in results)
            for item in results_500cp:
                # 验证期号是否为有效数字
                if not item["issue"].isdigit():
                    print(f"跳过无效期号: {item['issue']}")
                    continue
                
                # 验证期号长度（双色球期号通常为5-7位数字）
                if len(item["issue"]) < 5 or int(item["issue"]) < 3001:
                    print(f"跳过异常期号: {item['issue']}")
                    continue
                
                # 验证红球和蓝球格式
                try:
                    red_balls = item["red_balls"].split(",")
                    if len(red_balls) != 6:
                        print(f"跳过红球数量异常的数据: {item['issue']} - {item['red_balls']}")
                        continue
                    
                    # 验证红球数字是否有效
                    for ball in red_balls:
                        if not ball.isdigit() and not (ball.startswith('0') and ball[1:].isdigit()):
                            raise ValueError(f"无效的红球: {ball}")
                    
                    # 验证蓝球是否有效
                    blue_ball = item["blue_ball"]
                    if not blue_ball.isdigit() and not (blue_ball.startswith('0') and blue_ball[1:].isdigit()):
                        raise ValueError(f"无效的蓝球: {blue_ball}")
                except Exception as e:
                    print(f"跳过数据验证失败的记录: {item['issue']} - {e}")
                    continue
                
                # 检查是否有重复
                if item["issue"] not in existing_issues:
                    results.append(item)
                    existing_issues.add(item["issue"])
        except Exception as e:
            print(f"从500彩票网获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 如果两个数据源都失败且没有获取到任何数据，尝试使用现有的数据文件
        if not results:
            print("两个数据源都获取失败，尝试使用现有的数据文件...")
            try:
                existing_file = os.path.join(self.data_dir, "ssq_data.csv")
                if os.path.exists(existing_file):
                    with open(existing_file, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        next(reader)  # 跳过表头
                        for row in reader:
                            if len(row) >= 4:
                                results.append({
                                    "issue": row[0],
                                    "date": row[1],
                                    "red_balls": row[2],
                                    "blue_ball": row[3]
                                })
                    print(f"从现有数据文件中读取了{len(results)}期数据")
            except Exception as e:
                print(f"读取现有数据文件失败: {e}")
        
        # 按期号排序（降序）
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            # 只保留需要的期数（如果指定了count）
            if count is not None and len(results) > count:
                results = results[:count]
        
        print(f"共获取{len(results)}期双色球开奖结果")
        return results

    def save_to_csv(self, results, filename="ssq_data.csv"):
        """
        保存开奖结果到CSV文件

        Args:
            results: 开奖结果列表
            filename: 文件名
        """
        if not results:
            print("没有数据可保存")
            return
        
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            # 再次去重和验证数据
            valid_results = []
            seen_issues = set()
            for item in results:
                # 验证期号是否为有效数字
                if not item["issue"].isdigit():
                    print(f"保存前跳过无效期号: {item['issue']}")
                    continue
                
                # 验证期号长度（双色球期号通常为5-7位数字）
                if len(item["issue"]) < 5 or int(item["issue"]) < 3001:
                    print(f"保存前跳过异常期号: {item['issue']}")
                    continue
                
                # 验证红球和蓝球格式
                try:
                    red_balls = item["red_balls"].split(",")
                    if len(red_balls) != 6:
                        print(f"保存前跳过红球数量异常的数据: {item['issue']} - {item['red_balls']}")
                        continue
                    
                    # 验证红球数字是否有效
                    for ball in red_balls:
                        if not ball.isdigit() and not (ball.startswith('0') and ball[1:].isdigit()):
                            raise ValueError(f"无效的红球: {ball}")
                    
                    # 验证蓝球是否有效
                    blue_ball = item["blue_ball"]
                    if not blue_ball.isdigit() and not (blue_ball.startswith('0') and blue_ball[1:].isdigit()):
                        raise ValueError(f"无效的蓝球: {blue_ball}")
                except Exception as e:
                    print(f"保存前跳过数据验证失败的记录: {item['issue']} - {e}")
                    continue
                
                # 检查是否有重复
                if item["issue"] not in seen_issues:
                    valid_results.append(item)
                    seen_issues.add(item["issue"])
            
            # 按期号排序
            valid_results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(["issue", "date", "red_balls", "blue_ball"])
                # 写入数据
                for item in valid_results:
                    writer.writerow([item["issue"], item["date"], item["red_balls"], item["blue_ball"]])
            
            print(f"数据已保存到文件: {file_path}")
            print(f"成功保存了{len(valid_results)}期有效双色球数据")
        except Exception as e:
            print(f"保存数据失败: {e}")
            
        return file_path


def main():
    """主函数"""
    # 创建爬虫实例
    crawler = SSQCWLCrawler()
    
    # 获取所有历史数据
    results = crawler.get_history_data()
    
    # 保存数据
    if results:
        crawler.save_to_csv(results, filename="ssq_data_all.csv")


if __name__ == "__main__":
    main()