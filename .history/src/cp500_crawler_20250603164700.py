#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 500彩票网版本
从500彩票网获取最近300期双色球开奖结果
"""

import os
import csv
import time
import random
import requests
from bs4 import BeautifulSoup


class SSQ500Crawler:
    """双色球数据获取类 - 500彩票网版本"""

    def __init__(self, data_dir="../data"):
        """
        初始化数据获取器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 500彩票网双色球历史数据URL
        self.history_url = "https://datachart.500.com/ssq/history/history.shtml"
        
        # 请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://datachart.500.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
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
            print(f"正在从500彩票网获取最近{count}期双色球开奖结果...")
            
            # 发送请求
            response = requests.get(self.history_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 查找数据表格
            table = soup.select_one("table[width='100%']")
            if not table:
                print("未找到数据表格")
                return results
            
            # 获取表格中的所有行
            rows = table.select("tr")
            if len(rows) <= 1:
                print("表格中没有数据行")
                return results
            
            # 跳过表头行，解析数据行
            for row in rows[1:]:
                try:
                    cells = row.select("td")
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
                    
                    # 如果已经获取足够的数据，则退出循环
                    if len(results) >= count:
                        break
                except Exception as e:
                    print(f"解析行数据失败: {e}")
                    continue
            
            print(f"成功获取{len(results)}期双色球开奖结果")
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
    crawler = SSQ500Crawler()
    
    # 获取历史数据
    results = crawler.get_history_data(count=300)
    
    # 保存数据
    crawler.save_to_csv(results)


if __name__ == "__main__":
    main() 