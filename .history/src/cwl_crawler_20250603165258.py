#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 中国福利彩票官方网站版本
从中国福利彩票官方网站获取最近300期双色球开奖结果
"""

import os
import csv
import time
import json
import random
import requests
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
    
    def get_history_data(self, count=300):
        """
        获取历史开奖数据

        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        results = []
        
        try:
            print(f"正在从中国福利彩票官方网站获取最近{count}期双色球开奖结果...")
            
            # 由于API可能有请求数量限制，分批获取数据
            batch_size = 100  # 每次请求获取的期数
            num_batches = (count + batch_size - 1) // batch_size  # 向上取整
            
            for batch in range(num_batches):
                # 设置请求参数
                params = {
                    "name": "ssq",  # 双色球
                    "issueCount": batch_size,  # 每次获取的期数
                    "issueStart": "",  # 开始期号，留空表示从最新一期开始
                    "issueEnd": "",  # 结束期号，留空表示不限制
                    "dayStart": "",  # 开始日期，留空表示不限制
                    "dayEnd": "",  # 结束日期，留空表示不限制
                    "pageNo": batch + 1,  # 页码
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
                    
                    # 如果已经获取足够的数据，则退出循环
                    if len(results) >= count:
                        break
                    
                    # 添加随机延迟，避免请求过于频繁
                    time.sleep(random.uniform(1, 3))
                else:
                    error_msg = data.get("message", "未知错误")
                    print(f"获取第{batch+1}批数据失败: {error_msg}")
                    break
            
            print(f"成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"获取历史数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 按期号排序（降序）
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
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