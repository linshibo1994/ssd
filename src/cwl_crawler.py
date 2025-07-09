#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 中国福利彩票官方网站版本
从中国福利彩票官方网站获取最近300期双色球开奖结果
如果官方网站数据不足，则从中彩网和500彩票网获取补充数据
"""

import os
import csv
import time
import json
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd


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
        self.api_url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        
        # 中彩网双色球历史数据URL
        self.zhcw_url = "https://www.zhcw.com/kjxx/ssq/"
        
        # 500彩票网双色球历史数据URL
        self.cp500_url = "https://datachart.500.com/ssq/history/history.shtml"
        
        # 请求头
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
        
        # 中彩网请求头
        self.zhcw_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.zhcw.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
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
    
    def get_history_data_from_cwl(self, count=300):
        """
        从中国福利彩票官方网站获取历史开奖数据
        
        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        results = []
        
        try:
            print(f"正在从中国福利彩票官方网站获取最近{count}期双色球开奖结果...")
            
            # 计算需要请求的页数 (每页30条数据)
            page_size = 30
            total_pages = (count + page_size - 1) // page_size  # 向上取整
            
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
                
                # 发送请求
                response = requests.get(self.api_url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                
                # 解析JSON数据
                data = response.json()
                
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
                    
                    # 如果已经获取足够的数据，则退出循环
                    if len(results) >= count:
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
            
            # 只保留需要的期数
            if len(results) > count:
                results = results[:count]
        
        return results
    
    def get_history_data_from_zhcw(self, count=300):
        """
        从中彩网获取双色球历史数据
        
        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        results = []
        
        try:
            print(f"正在从中彩网获取双色球历史数据...")
            
            # 计算需要请求的页数（每页30条数据）
            page_size = 30
            total_pages = (count + page_size - 1) // page_size  # 向上取整
            
            # 中彩网API接口
            api_url = "https://jc.zhcw.com/port/client_json.php"
            
            for page in range(1, total_pages + 1):
                try:
                    print(f"正在获取中彩网第{page}页数据 (每页{page_size}条)...")
                    
                    # 设置请求参数
                    params = {
                        'callback': f'jQuery{int(time.time() * 1000)}_{int(random.random() * 10000000000000000)}',
                        'transactionType': '10001001',
                        'lotteryId': '1',  # 1是双色球的ID
                        'issueCount': count,
                        'startIssue': '',
                        'endIssue': '',
                        'startDate': '',
                        'endDate': '',
                        'type': '0',
                        'pageNum': page,
                        'pageSize': page_size,
                        'tt': random.random(),
                        '_': int(time.time() * 1000)
                    }
                    
                    # 发送请求
                    response = requests.get(
                        api_url, 
                        headers=self.zhcw_headers, 
                        params=params, 
                        timeout=10
                    )
                    
                    # 检查响应状态
                    response.raise_for_status()
                    
                    # 解析JSON数据
                    # API返回的是JSONP格式，需要提取JSON部分
                    text = response.text
                    json_str = text[text.find('(') + 1:text.rfind(')')]
                    data = json.loads(json_str)
                    
                    # 检查是否有结果数据
                    if "data" in data and "records" in data["data"] and isinstance(data["data"]["records"], list):
                        records = data["data"]["records"]
                        
                        # 提取开奖结果
                        for item in records:
                            try:
                                issue = item["issue"]  # 期号
                                date = item["openTime"]  # 开奖日期
                                
                                # 获取红球号码
                                red_balls = []
                                for i in range(1, 7):
                                    ball_key = f"red{i}"
                                    if ball_key in item:
                                        red_balls.append(str(item[ball_key]).zfill(2))
                                
                                # 获取蓝球号码
                                blue_ball = str(item["blue"]).zfill(2) if "blue" in item else ""
                                
                                if issue and date and len(red_balls) == 6 and blue_ball:
                                    results.append({
                                        "issue": issue,
                                        "date": date,
                                        "red_balls": ",".join(red_balls),
                                        "blue_ball": blue_ball
                                    })
                            except Exception as e:
                                print(f"解析中彩网数据项失败: {e}")
                                continue
                        
                        print(f"成功获取中彩网第{page}页数据，当前共{len(results)}期")
                        
                        # 如果已经获取足够的数据，则退出循环
                        if len(results) >= count:
                            break
                        
                        # 添加随机延迟，避免请求过于频繁
                        time.sleep(random.uniform(1, 3))
                    else:
                        print(f"获取中彩网第{page}页数据失败: 无效的数据格式")
                        # 尝试备用解析方法
                        if "data" in data:
                            print(f"API响应内容: {data}")
                except Exception as e:
                    print(f"获取中彩网第{page}页数据失败: {e}")
                    # 如果API请求失败，尝试使用备用方法
                    break
            
            # 如果API方法没有获取到数据，尝试使用HTML解析方法
            if not results:
                print("API方法未获取到数据，尝试使用HTML解析方法...")
                # 访问中彩网双色球历史数据页面
                response = requests.get(self.zhcw_url, headers=self.zhcw_headers, timeout=10)
                response.encoding = 'utf-8'
                
                # 尝试使用pandas读取HTML表格
                try:
                    dfs = pd.read_html(response.text, header=0)
                    if dfs and len(dfs) > 0:
                        df = dfs[0]
                        
                        # 处理DataFrame数据
                        for _, row in df.iterrows():
                            try:
                                # 检查必要的列是否存在
                                if '期号' in df.columns and '开奖日期' in df.columns:
                                    issue = str(row['期号'])
                                    date = str(row['开奖日期'])
                                    
                                    # 处理红球和蓝球
                                    red_balls = []
                                    blue_ball = ""
                                    
                                    # 尝试不同的列名格式
                                    if '红球号码' in df.columns and '蓝球号码' in df.columns:
                                        red_str = str(row['红球号码'])
                                        blue_ball = str(row['蓝球号码']).zfill(2)
                                        red_balls = [num.strip().zfill(2) for num in red_str.split()]
                                    elif '开奖号码' in df.columns:
                                        ball_data = str(row['开奖号码'])
                                        if '+' in ball_data:
                                            red_part, blue_part = ball_data.split('+')
                                            red_balls = [num.strip().zfill(2) for num in red_part.split()]
                                            blue_ball = blue_part.strip().zfill(2)
                                    
                                    if issue and date and len(red_balls) == 6 and blue_ball:
                                        results.append({
                                            "issue": issue,
                                            "date": date,
                                            "red_balls": ",".join(red_balls),
                                            "blue_ball": blue_ball
                                        })
                            except Exception as e:
                                print(f"处理中彩网pandas数据项失败: {e}")
                                continue
                        
                        print(f"通过pandas成功获取中彩网数据，共{len(results)}期")
                except Exception as e:
                    print(f"使用pandas解析中彩网HTML表格失败: {e}")
            
            print(f"从中彩网成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中彩网获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 按期号排序（降序）
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            # 只保留需要的期数
            if len(results) > count:
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
            
            # 发送请求
            response = requests.get(self.cp500_url, headers=self.cp500_headers, timeout=10)
            # 设置正确的编码
            response.encoding = 'gb2312'
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 查找数据表格
            table = soup.find('table', attrs={'width': '100%'})
            if not table:
                print("未找到数据表格")
                return results
            
            # 获取表格中的所有行
            rows = table.find_all('tr')
            if len(rows) <= 1:
                print("表格中没有数据行")
                return results
            
            # 跳过表头行，解析数据行
            for row in rows[1:]:
                try:
                    cells = row.find_all('td')
                    if len(cells) < 9:
                        continue
                    
                    # 获取期号
                    issue = cells[0].text.strip()
                    
                    # 获取开奖日期
                    date = cells[1].text.strip()
                    
                    # 获取红球
                    red_balls = []
                    for i in range(2, 8):
                        red_balls.append(cells[i].text.strip().zfill(2))  # 补0，保持两位数格式
                    
                    # 获取蓝球
                    blue_ball = cells[8].text.strip().zfill(2)  # 补0，保持两位数格式
                    
                    results.append({
                        "issue": issue,
                        "date": date,
                        "red_balls": ",".join(red_balls),
                        "blue_ball": blue_ball
                    })
                except Exception as e:
                    print(f"解析行数据失败: {e}")
                    continue
            
            print(f"从500彩票网成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从500彩票网获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        return results

    def get_history_data(self, count=300):
        """
        获取双色球历史数据，优先从官方网站获取，不足则从中彩网和500彩票网补充

        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        # 从官方网站获取数据
        results = self.get_history_data_from_cwl(count)
        
        # 如果官方网站数据不足，先从中彩网获取补充数据
        if len(results) < count:
            print(f"从官方网站获取的数据不足{count}期，将从中彩网获取补充数据...")
            zhcw_results = self.get_history_data_from_zhcw(count)
            
            # 合并数据，去重
            existing_issues = set(item["issue"] for item in results)
            for item in zhcw_results:
                if item["issue"] not in existing_issues:
                    results.append(item)
                    existing_issues.add(item["issue"])
            
            # 按期号排序（降序）
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        # 如果中彩网数据仍不足，从500彩票网获取补充数据
        if len(results) < count:
            print(f"从官方网站和中彩网获取的数据不足{count}期，将从500彩票网获取补充数据...")
            results_500cp = self.get_history_data_from_500cp()
            
            # 合并数据，去重
            existing_issues = set(item["issue"] for item in results)
            for item in results_500cp:
                if item["issue"] not in existing_issues:
                    results.append(item)
                    existing_issues.add(item["issue"])
            
            # 按期号排序（降序）
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            # 只保留需要的期数
            if len(results) > count:
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
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(["issue", "date", "red_balls", "blue_ball"])
                # 写入数据
                for item in results:
                    writer.writerow([item["issue"], item["date"], item["red_balls"], item["blue_ball"]])
            
            print(f"数据已保存到文件: {file_path}")
        except Exception as e:
            print(f"保存数据失败: {e}")


def main():
    """主函数"""
    # 创建爬虫实例
    crawler = SSQCWLCrawler()
    
    # 获取历史数据
    results = crawler.get_history_data(count=300)
    
    # 保存数据
    if results:
        crawler.save_to_csv(results)


if __name__ == "__main__":
    main() 