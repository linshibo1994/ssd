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
                    # 添加JSON解析的错误处理
                    try:
                        data = response.json()
                    except ValueError as e:
                        print(f"解析JSON数据失败: {e}")
                        print(f"响应内容: {response.text[:200]}...")
                        # 如果JSON解析失败，尝试下一页
                        continue
                    
                    # 检查响应状态
                    if "status" in data and data["status"] != "200":
                        print(f"API返回错误状态: {data.get('status')} - {data.get('message', '未知错误')}")
                        # 如果API返回错误状态，尝试下一页
                        continue
                    
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
                    # 记录更详细的错误信息
                    if isinstance(e, requests.exceptions.HTTPError):
                        print(f"HTTP错误: {e.response.status_code} - {e.response.reason}")
                    elif isinstance(e, requests.exceptions.ConnectionError):
                        print("连接错误: 请检查网络连接")
                    elif isinstance(e, requests.exceptions.Timeout):
                        print("请求超时: 服务器响应时间过长")
                    elif isinstance(e, requests.exceptions.RequestException):
                        print(f"请求异常: {e}")
                    
                    # 添加延迟后继续获取下一页
                    time.sleep(random.uniform(3, 5))
                    continue
            
            print(f"从中彩网成功获取{len(results)}期大乐透开奖结果")
            
            # 按期号排序（降序）
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
        except Exception as e:
            print(f"从中彩网获取数据失败: {e}")
        
        return results

    def get_history_data_from_html(self, count=None):
        """从中彩网HTML页面获取大乐透历史开奖数据
        
        Args:
            count: 获取的记录数量，默认为None表示获取所有期数
            
        Returns:
            开奖结果列表
        """
        results = []
        try:
            print("从中彩网HTML页面获取大乐透历史数据...")
            
            # 中彩网大乐透开奖公告页面
            base_url = "https://www.cwl.gov.cn/kjxx/dlt/kjgg/"
            
            # 请求头
            headers = self.cwl_headers.copy()
            
            # 已获取的期号集合，用于去重
            seen_issues = set()
            
            # 获取首页数据
            try:
                print("正在获取开奖公告页面...")
                response = requests.get(base_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                # 解析HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找开奖信息表格
                table = soup.find('table', class_='kj_tablelist02')
                if table:
                    rows = table.find_all('tr')
                    # 跳过表头
                    for row in rows[1:]:
                        cells = row.find_all('td')
                        if len(cells) >= 5:
                            # 期号
                            issue = cells[0].text.strip()
                            if issue in seen_issues:
                                continue
                            seen_issues.add(issue)
                            
                            # 开奖日期
                            date = cells[1].text.strip()
                            
                            # 前区号码
                            front_balls_div = cells[2].find('div', class_='red_ball')
                            front_balls = ""
                            if front_balls_div:
                                front_balls_spans = front_balls_div.find_all('span')
                                front_balls = ",".join([span.text.strip() for span in front_balls_spans])
                            
                            # 后区号码
                            back_balls_div = cells[2].find('div', class_='blue_ball')
                            back_balls = ""
                            if back_balls_div:
                                back_balls_spans = back_balls_div.find_all('span')
                                back_balls = ",".join([span.text.strip() for span in back_balls_spans])
                            
                            results.append({
                                "issue": issue,
                                "date": date,
                                "front_balls": front_balls,
                                "back_balls": back_balls
                            })
                            
                            # 如果已获取足够数量的数据，结束循环
                            if count is not None and len(results) >= count:
                                break
                else:
                    print("未找到开奖信息表格")
            except Exception as e:
                print(f"获取开奖公告页面失败: {e}")
            
            print(f"从中彩网HTML页面成功获取{len(results)}期大乐透开奖结果")
            
        except Exception as e:
            print(f"从中彩网HTML页面获取数据失败: {e}")
        
        return results

    def generate_mock_data(self, count=300):
        """生成模拟的大乐透历史数据
        当所有数据获取方式都失败时，生成模拟数据用于测试
        
        Args:
            count: 生成的记录数量
            
        Returns:
            模拟的开奖结果列表
        """
        print(f"生成{count}期模拟大乐透数据用于测试...")
        results = []
        
        # 从最新一期开始生成
        latest_issue = 23001  # 假设的最新一期期号
        latest_date = "2023-01-01"  # 假设的最新一期日期
        
        for i in range(count):
            issue = str(latest_issue - i)
            
            # 生成随机的前区号码（5个，1-35之间）
            front_balls = []
            while len(front_balls) < 5:
                num = random.randint(1, 35)
                if num not in front_balls:
                    front_balls.append(num)
            front_balls.sort()
            front_balls_str = ",".join([f"{num:02d}" for num in front_balls])
            
            # 生成随机的后区号码（2个，1-12之间）
            back_balls = []
            while len(back_balls) < 2:
                num = random.randint(1, 12)
                if num not in back_balls:
                    back_balls.append(num)
            back_balls.sort()
            back_balls_str = ",".join([f"{num:02d}" for num in back_balls])
            
            results.append({
                "issue": issue,
                "date": latest_date,  # 简化处理，所有期数使用相同日期
                "front_balls": front_balls_str,
                "back_balls": back_balls_str
            })
        
        print(f"成功生成{len(results)}期模拟大乐透数据")
        return results

    def get_history_data(self, count=300):
        """获取大乐透历史数据

        Args:
            count: 获取的记录数量，默认300期，如果为None则获取所有期数

        Returns:
            开奖结果列表
        """
        # 从中彩网API获取数据
        results = self.get_history_data_from_cwl(count)
        
        # 如果API获取失败，尝试从HTML页面获取
        if not results:
            print("从中彩网API获取数据失败，尝试从HTML页面获取...")
            results = self.get_history_data_from_html(count)
        
        # 如果从中彩网获取数据失败或数据不足，可以尝试从本地缓存读取
        if not results and count is not None:
            print("从中彩网获取数据失败，尝试从本地缓存读取...")
            # 检查是否有本地缓存数据
            cache_file = os.path.join(self.data_dir, "dlt_data.csv")
            if os.path.exists(cache_file):
                try:
                    cached_results = []
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            cached_results.append(row)
                    
                    if cached_results:
                        print(f"从本地缓存读取到{len(cached_results)}期数据")
                        results = cached_results
                except Exception as e:
                    print(f"读取本地缓存失败: {e}")
        
        # 如果所有方法都失败，生成模拟数据
        if not results and count is not None:
            print("所有数据获取方式都失败，生成模拟数据...")
            results = self.generate_mock_data(count)
        
        # 按期号排序（降序）
        if results:
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