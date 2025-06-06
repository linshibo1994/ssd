#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 新浪彩票版本
使用新浪彩票网的API获取最近300期双色球开奖结果
"""

import os
import csv
import time
import json
import random
import requests
from datetime import datetime


class SSQSinaCrawler:
    """双色球数据获取类 - 新浪彩票版本"""

    def __init__(self, data_dir="../data"):
        """
        初始化数据获取器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 新浪彩票双色球API
        self.api_url = "https://match.lottery.sina.com.cn/lotto/pc_zst/data/getData.php"
        
        # 请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://match.lottery.sina.com.cn/",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "X-Requested-With": "XMLHttpRequest"
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
            print(f"正在从新浪彩票获取最近{count}期双色球开奖结果...")
            
            # 设置请求参数
            params = {
                "lottoType": "ssq",  # 双色球
                "pageSize": count,   # 获取的期数
                "pageNum": 1,        # 页码
                "sortType": "desc",  # 降序排序
                "_": int(time.time() * 1000)  # 时间戳，避免缓存
            }
            
            # 发送请求
            response = requests.get(self.api_url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            # 解析JSON数据
            data = response.json()
            
            if data["result"] == "success" and "data" in data:
                # 提取开奖结果
                for item in data["data"]:
                    try:
                        issue = item["issue"]
                        date = item["date"]
                        
                        # 获取红球
                        red_balls = []
                        for i in range(1, 7):
                            red_balls.append(item[f"red{i}"])
                        
                        # 获取蓝球
                        blue_ball = item["blue"]
                        
                        results.append({
                            "issue": issue,
                            "date": date,
                            "red_balls": ",".join(red_balls),
                            "blue_ball": blue_ball
                        })
                    except Exception as e:
                        print(f"解析数据项失败: {e}")
                        continue
                
                print(f"成功获取{len(results)}期双色球开奖结果")
            else:
                print(f"获取数据失败，返回结果: {data.get('msg', '未知错误')}")
        except Exception as e:
            print(f"获取历史数据失败: {e}")
        
        return results
    
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
    crawler = SSQSinaCrawler()
    
    # 获取历史数据
    results = crawler.get_history_data(count=300)
    
    # 保存数据
    crawler.save_to_csv(results)


if __name__ == "__main__":
    main() 