#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块
从API接口获取最近300期双色球开奖结果，如果API调用失败则生成模拟数据
"""

import os
import csv
import time
import random
import requests
from datetime import datetime, timedelta


class SSQCrawler:
    """双色球数据获取类"""

    def __init__(self, data_dir="../data"):
        """
        初始化数据获取器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_data_from_api(self, count=300):
        """
        从API获取双色球历史数据

        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        # 使用探数数据平台的API
        api_url = "https://api.tanshuapi.com/api/caipiao/v1/history"
        
        # 这里需要替换为您申请的API密钥
        # 注意：实际使用时需要申请自己的API密钥
        # 这里使用的是示例密钥，可能无法正常工作
        api_key = "yourkey"  # 需要替换为实际申请的密钥
        
        # 双色球的彩票ID为11
        caipiao_id = 11
        
        params = {
            "key": api_key,
            "caipiaoid": caipiao_id,
            "pagesize": count,
            "page": 1
        }
        
        try:
            print(f"正在从API获取双色球历史数据...")
            response = requests.get(api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") == 1 and "data" in data and "list" in data["data"]:
                results = []
                for item in data["data"]["list"]:
                    # 解析API返回的数据
                    issue = item.get("issueno", "")
                    date = item.get("opendate", "")
                    
                    # 解析开奖号码
                    number = item.get("number", "")
                    ref_number = item.get("refernumber", "")
                    
                    # 分离红球和蓝球
                    if number and ref_number:
                        red_balls = number.strip()
                        blue_ball = ref_number.strip()
                        
                        results.append({
                            "issue": issue,
                            "date": date,
                            "red_balls": red_balls,
                            "blue_ball": blue_ball
                        })
                
                print(f"成功从API获取{len(results)}期双色球开奖结果")
                return results
            else:
                print(f"API返回错误: {data.get('msg', '未知错误')}")
                return self.fetch_data_from_alternative_api(count)
        except Exception as e:
            print(f"从API获取数据失败: {e}")
            # 尝试备用API
            return self.fetch_data_from_alternative_api(count)
    
    def fetch_data_from_alternative_api(self, count=300):
        """
        从备用API获取双色球历史数据

        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        # 使用聚合数据API作为备用
        api_url = "http://apis.juhe.cn/lottery/history"
        
        # 这里需要替换为您申请的API密钥
        # 注意：实际使用时需要申请自己的API密钥
        api_key = "yourkey"  # 需要替换为实际申请的密钥
        
        params = {
            "key": api_key,
            "lottery_id": "ssq",  # 双色球的ID
            "page_size": min(count, 50),  # 聚合数据API每页最多返回50条
            "page": 1
        }
        
        all_results = []
        pages_needed = (count + 49) // 50  # 计算需要请求的页数
        
        try:
            print(f"正在从备用API获取双色球历史数据...")
            
            for page in range(1, pages_needed + 1):
                params["page"] = page
                response = requests.get(api_url, params=params, headers=self.headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get("error_code") == 0 and "result" in data and "lotteryResList" in data["result"]:
                    for item in data["result"]["lotteryResList"]:
                        # 解析API返回的数据
                        issue = item.get("lottery_no", "")
                        date = item.get("lottery_date", "")
                        
                        # 解析开奖号码
                        lottery_res = item.get("lottery_res", "")
                        if lottery_res:
                            # 假设格式为 "01,02,03,04,05,06+07"
                            parts = lottery_res.split("+")
                            if len(parts) == 2:
                                red_balls = parts[0].strip()
                                blue_ball = parts[1].strip()
                                
                                all_results.append({
                                    "issue": issue,
                                    "date": date,
                                    "red_balls": red_balls,
                                    "blue_ball": blue_ball
                                })
                
                # 避免频繁请求
                if page < pages_needed:
                    time.sleep(random.uniform(1, 2))
                    
                # 如果已经获取足够的数据，就停止请求
                if len(all_results) >= count:
                    break
            
            print(f"成功从备用API获取{len(all_results)}期双色球开奖结果")
            return all_results[:count]  # 确保不超过请求的数量
        except Exception as e:
            print(f"从备用API获取数据失败: {e}")
            # 如果备用API也失败，尝试使用免费API
            return self.fetch_data_from_free_api(count)
    
    def fetch_data_from_free_api(self, count=300):
        """
        从免费API获取双色球历史数据

        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        # 使用RollToolsApi作为免费API
        api_url = "https://www.mxnzp.com/api/lottery/common/history"
        
        params = {
            "app_id": "your_app_id",  # 需要替换为实际申请的app_id
            "app_secret": "your_app_secret",  # 需要替换为实际申请的app_secret
            "code": "ssq",  # 双色球的代码
            "size": min(count, 100)  # 每次最多返回100条
        }
        
        try:
            print(f"正在从免费API获取双色球历史数据...")
            response = requests.get(api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") == 1 and "data" in data:
                results = []
                for item in data["data"]:
                    # 解析API返回的数据
                    issue = item.get("expect", "")
                    date = item.get("time", "")[:10]  # 只取日期部分
                    
                    # 解析开奖号码
                    open_code = item.get("openCode", "")
                    if open_code:
                        # 假设格式为 "01,02,03,04,05,06+07"
                        parts = open_code.split("+")
                        if len(parts) == 2:
                            red_balls = parts[0].strip()
                            blue_ball = parts[1].strip()
                            
                            results.append({
                                "issue": issue,
                                "date": date,
                                "red_balls": red_balls,
                                "blue_ball": blue_ball
                            })
                
                print(f"成功从免费API获取{len(results)}期双色球开奖结果")
                return results
            else:
                print(f"免费API返回错误: {data.get('msg', '未知错误')}")
                # 如果所有API都失败，生成模拟数据
                return self.generate_mock_data(count)
        except Exception as e:
            print(f"从免费API获取数据失败: {e}")
            # 如果所有API都失败，生成模拟数据
            return self.generate_mock_data(count)
    
    def generate_mock_data(self, count=300):
        """
        生成模拟的双色球历史数据

        Args:
            count: 生成的记录数量，默认300期

        Returns:
            模拟的开奖结果列表
        """
        print(f"所有API获取数据失败，生成{count}期模拟数据用于测试")
        
        results = []
        # 从当前日期开始，向前推算
        current_date = datetime.now()
        # 双色球每周二、四、日开奖
        draw_days = [1, 3, 6]  # 0=周一, 1=周二, ..., 6=周日
        
        # 生成最近一期的期号，格式为年份后两位+期号(三位数)
        # 例如：2023年第1期为23001
        current_year = current_date.year
        year_prefix = str(current_year)[-2:]
        issue_number = int(year_prefix + "001")
        
        # 向前推算，找到最近一次开奖日
        while current_date.weekday() not in draw_days:
            current_date -= timedelta(days=1)
        
        # 生成模拟数据
        for i in range(count):
            # 生成期号
            issue = str(issue_number + i)
            
            # 生成开奖日期
            draw_date = current_date - timedelta(days=i * 2)  # 假设平均每2天开奖一次
            date_str = draw_date.strftime("%Y-%m-%d")
            
            # 生成红球号码（6个不重复的1-33之间的数字）
            red_balls = sorted(random.sample(range(1, 34), 6))
            # 格式化为两位数字
            red_balls_str = ",".join([f"{num:02d}" for num in red_balls])
            
            # 生成蓝球号码（1个1-16之间的数字）
            blue_ball = random.randint(1, 16)
            blue_ball_str = f"{blue_ball:02d}"
            
            results.append({
                "issue": issue,
                "date": date_str,
                "red_balls": red_balls_str,
                "blue_ball": blue_ball_str
            })
        
        print(f"成功生成{len(results)}期模拟双色球开奖结果")
        return results

    def crawl_history_data(self, count=300):
        """
        获取历史开奖数据

        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        # 尝试从API获取数据
        results = self.fetch_data_from_api(count)
        
        # 如果API获取失败，生成模拟数据
        if not results:
            print("所有API获取数据失败，将生成模拟数据")
            results = self.generate_mock_data(count)
        
        return results

    def save_to_csv(self, results, filename="ssq_data.csv"):
        """
        保存结果到CSV文件

        Args:
            results: 开奖结果列表
            filename: 保存的文件名

        Returns:
            保存的文件路径
        """
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["issue", "date", "red_balls", "blue_ball"])
                writer.writeheader()
                writer.writerows(results)
            print(f"数据已保存到 {filepath}")
            return filepath
        except Exception as e:
            print(f"保存数据失败: {e}")
            return None


def main():
    """主函数"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据目录为上一级的data目录
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    # 创建爬虫实例
    crawler = SSQCrawler(data_dir=data_dir)
    
    # 获取历史数据
    print(f"开始获取双色球历史数据，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = crawler.crawl_history_data(count=300)  # 获取300期
    
    # 保存数据
    if results:
        print(f"共获取{len(results)}期双色球开奖结果")
        crawler.save_to_csv(results)
    else:
        print("未获取到数据")


if __name__ == "__main__":
    main()