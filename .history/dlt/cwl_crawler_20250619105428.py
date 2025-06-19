#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大乐透数据爬虫模块
从中彩网获取大乐透历史开奖数据
"""

import os
import csv
import time
import random
import requests
from bs4 import BeautifulSoup


class DLTCWLCrawler:
    """大乐透中彩网数据爬虫
    从中彩网获取大乐透历史开奖数据
    """

    def __init__(self, data_dir="data"):
        """初始化爬虫

        Args:
            data_dir: 数据保存目录，默认为data
        """
        # 数据保存目录
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)
        
        # 中彩网请求头
        self.cwl_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.cwl.gov.cn/kjxx/dlt/kjgg/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
        }

    def get_history_data_from_cwl(self, count=None):
        """从中彩网获取大乐透历史开奖数据

        Args:
            count: 获取的记录数量，默认为None表示获取所有期数

        Returns:
            开奖结果列表
        """
        results = []
        try:
            print("从中彩网获取大乐透历史数据...")
            
            # 中彩网API
            api_url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
            
            # 请求头
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": "https://www.cwl.gov.cn/kjxx/dlt/kjgg/",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Connection": "keep-alive",
                "X-Requested-With": "XMLHttpRequest",
                "Origin": "https://www.cwl.gov.cn"
            }
            
            # 计算需要请求的页数
            # 每页显示30条数据
            page_size = 30
            page_count = 1
            
            if count is not None:
                page_count = (count + page_size - 1) // page_size
            else:
                # 如果未指定数量，默认获取所有数据（最多50页，约1500期）
                page_count = 50
            
            # 已获取的期号集合，用于去重
            seen_issues = set()
            
            # 逐页获取数据
            for page in range(1, page_count + 1):
                try:
                    print(f"正在获取第{page}页数据...")
                    
                    # 设置请求参数
                    params = {
                        "name": "dlt",  # 大乐透
                        "pageNo": page,
                        "pageSize": page_size,
                        "systemType": "PC"
                    }
                    
                    # 添加重试机制
                    max_retries = 3
                    retry_count = 0
                    retry_delay = 2
                    
                    while retry_count < max_retries:
                        try:
                            # 发送请求
                            response = requests.get(api_url, headers=headers, params=params, timeout=15)
                            response.raise_for_status()
                            # 请求成功，跳出重试循环
                            break
                        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as err:
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise Exception(f"请求失败，已重试{max_retries}次: {err}")
                            print(f"请求失败，正在进行第{retry_count}次重试: {err}")
                            # 增加随机延迟时间
                            time.sleep(retry_delay * retry_count + random.uniform(1, 3))
                    
                    # 解析JSON数据
                    data = response.json()
                    
                    # 检查是否有结果数据
                    if "result" in data and isinstance(data["result"], list):
                        items = data["result"]
                        
                        if not items:
                            print(f"第{page}页没有数据，可能已到达最后一页")
                            break
                        
                        # 处理每一期数据
                        for item in items:
                            issue = item["code"]  # 期号
                            
                            # 检查期号是否已存在，避免重复添加
                            if issue in seen_issues:
                                continue
                            seen_issues.add(issue)
                            
                            date = item["date"]  # 开奖日期
                            
                            # 获取前区号码（格式为 "01,02,03,04,05"）
                            front_str = item["front"]
                            front_balls = front_str
                            
                            # 获取后区号码（格式为 "01,02"）
                            back_str = item["back"]
                            back_balls = back_str
                            
                            results.append({
                                "issue": issue,
                                "date": date,
                                "front_balls": front_balls,
                                "back_balls": back_balls
                            })
                        
                        # 如果已获取足够数量的数据，结束循环
                        if count is not None and len(results) >= count:
                            results = results[:count]  # 确保只返回指定数量的结果
                            break
                    else:
                        print(f"第{page}页数据格式异常")
                        break
                    
                    # 添加随机延迟，避免请求过于频繁
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    print(f"获取第{page}页数据失败: {e}")
                    # 继续获取下一页
                    continue
            
            print(f"从中彩网成功获取{len(results)}期大乐透开奖结果")
            
            # 按期号排序（降序）
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
        except Exception as e:
            print(f"从中彩网获取数据失败: {e}")
        
        return results

    def get_history_data(self, count=300):
        """获取大乐透历史数据

        Args:
            count: 获取的记录数量，默认300期，如果为None则获取所有期数

        Returns:
            开奖结果列表
        """
        # 从中彩网获取数据
        results = self.get_history_data_from_cwl(count)
        
        # 按期号排序（降序）
        results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        # 只保留需要的期数（如果指定了期数）
        if count is not None and len(results) > count:
            results = results[:count]
        
        print(f"共获取{len(results)}期大乐透开奖结果")
        return results

    def save_to_csv(self, results, filename="dlt_data.csv"):
        """将开奖结果保存到CSV文件
        
        Args:
            results: 开奖结果列表
            filename: 保存的文件名
        """
        if not results:
            print("没有数据可保存")
            return
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 构建完整的文件路径
        file_path = os.path.join(self.data_dir, filename)
        
        # 检查文件是否已存在
        file_exists = os.path.exists(file_path)
        
        # 如果文件已存在，读取已有数据，确保不重复添加
        existing_data = []
        existing_issues = set()
        if file_exists:
            print(f"发现已有数据文件: {file_path}，将检查期号确保不重复")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data.append(row)
                        existing_issues.add(row['issue'])
            except Exception as e:
                print(f"读取已有数据文件失败: {e}")
                existing_data = []
                existing_issues = set()
        
        # 过滤掉已存在的期号
        new_results = []
        for result in results:
            if result['issue'] not in existing_issues:
                new_results.append(result)
                existing_issues.add(result['issue'])
        
        if file_exists and new_results:
            print(f"将添加{len(new_results)}期新数据到已有文件")
            # 追加新数据到已有文件
            with open(file_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['issue', 'date', 'front_balls', 'back_balls'])
                for result in new_results:
                    writer.writerow(result)
            print(f"数据已追加到文件: {file_path}")
        elif not file_exists:
            print(f"创建新数据文件: {file_path}")
            # 创建新文件并写入所有数据
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['issue', 'date', 'front_balls', 'back_balls'])
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
            print(f"数据已保存到新文件: {file_path}")
        else:
            print("没有新数据需要添加")
        
        # 打印保存的数据总数
        total_count = len(existing_data) + len(new_results) if file_exists else len(results)
        print(f"文件中共有{total_count}期大乐透开奖结果")
        
        return file_path


def main():
    """主函数"""
    # 创建爬虫实例
    crawler = DLTCWLCrawler()
    
    # 获取历史数据
    results = crawler.get_history_data(count=300)
    
    # 保存数据
    if results:
        crawler.save_to_csv(results)


if __name__ == "__main__":
    main()