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
        self.api_url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        
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
            
            # 获取最新一期数据，以确定当前期号
            params = {
                "name": "ssq",
                "pageNo": 1,
                "pageSize": 1,
                "systemType": "PC"
            }
            
            response = requests.get(self.api_url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            latest_issue = None
            if "result" in data and isinstance(data["result"], list) and len(data["result"]) > 0:
                latest_issue = data["result"][0]["code"]
                print(f"最新一期为: {latest_issue}")
            else:
                print("无法获取最新一期期号")
                return results
            
            # 计算开始期号
            start_issue = str(int(latest_issue) - count + 1)
            print(f"开始获取从第{start_issue}期到第{latest_issue}期的数据...")
            
            # 由于API可能有分页限制，我们需要分批获取数据
            batch_size = 100  # 每次请求获取的期数
            num_batches = (count + batch_size - 1) // batch_size  # 向上取整
            
            for batch in range(num_batches):
                # 计算当前批次的开始和结束期号
                current_batch_start = str(int(latest_issue) - (batch + 1) * batch_size + 1)
                current_batch_end = str(int(latest_issue) - batch * batch_size)
                
                # 确保开始期号不小于1
                if int(current_batch_start) < 1:
                    current_batch_start = "1"
                
                print(f"获取第{batch+1}批数据，期号范围: {current_batch_start} - {current_batch_end}")
                
                # 设置请求参数
                params = {
                    "name": "ssq",  # 双色球
                    "issueStart": current_batch_start,  # 开始期号
                    "issueEnd": current_batch_end,  # 结束期号
                    "pageNo": 1,  # 页码
                    "pageSize": batch_size,  # 每页数量
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
                    
                    print(f"成功获取第{batch+1}批数据，当前共{len(results)}期")
                    
                    # 如果已经获取足够的数据，或者已经获取到第1期，则退出循环
                    if len(results) >= count or int(current_batch_start) <= 1:
                        break
                    
                    # 添加随机延迟，避免请求过于频繁
                    time.sleep(random.uniform(1, 3))
                else:
                    error_msg = data.get("message", "未知错误")
                    print(f"获取第{batch+1}批数据失败: {error_msg}")
                    break
            
            print(f"从中国福利彩票官方网站成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中国福利彩票官方网站获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 按期号排序（降序）
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
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
        获取历史开奖数据，优先从中国福利彩票官方网站获取，
        如果数据不足，则从500彩票网获取补充数据

        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        # 首先从中国福利彩票官方网站获取数据
        results = self.get_history_data_from_cwl(count)
        
        # 如果从官方网站获取的数据不足，则从500彩票网获取补充数据
        if len(results) < count:
            print(f"从中国福利彩票官方网站只获取到{len(results)}期数据，尝试从500彩票网获取补充数据...")
            cp500_results = self.get_history_data_from_500cp()
            
            if cp500_results:
                # 合并结果并去重
                existing_issues = set(r["issue"] for r in results)
                for result in cp500_results:
                    if result["issue"] not in existing_issues:
                        results.append(result)
                        existing_issues.add(result["issue"])
                
                # 重新排序
                results.sort(key=lambda x: int(x["issue"]), reverse=True)
                
                print(f"合并后共有{len(results)}期数据")
        
        # 只返回前count条数据
        return results[:count]
    
    def save_to_csv(self, results, filename="ssq_data.csv"):
        """
        保存结果到CSV文件

        Args:
            results: 开奖结果列表
            filename: 文件名
        """
        if not results:
            print("没有数据需要保存")
            return
        
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["issue", "date", "red_balls", "blue_ball"])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"成功保存{len(results)}条数据到 {file_path}")
        except Exception as e:
            print(f"保存数据失败: {e}")


def main():
    """主函数"""
    # 创建爬虫实例
    crawler = SSQCWLCrawler()
    
    # 获取历史数据
    results = crawler.get_history_data(count=300)
    
    # 保存数据
    crawler.save_to_csv(results)


if __name__ == "__main__":
    main() 