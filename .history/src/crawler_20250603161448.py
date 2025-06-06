#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球爬虫模块
从中彩网获取最近300期双色球开奖结果
"""

import os
import re
import csv
import time
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime


class SSQCrawler:
    """双色球爬虫类"""

    def __init__(self, base_url="https://www.zhcw.com/kjxx/ssq/", data_dir="../data"):
        """初始化爬虫

        Args:
            base_url: 中彩网双色球开奖结果页面URL
            data_dir: 数据存储目录
        """
        self.base_url = base_url
        self.data_dir = data_dir
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        }
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)

    def get_page(self, url):
        """获取页面内容

        Args:
            url: 页面URL

        Returns:
            BeautifulSoup对象
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()  # 如果响应状态码不是200，就主动抛出异常
            response.encoding = "utf-8"
            return BeautifulSoup(response.text, "lxml")
        except requests.RequestException as e:
            print(f"获取页面失败: {e}")
            return None

    def parse_draw_results(self, soup):
        """解析开奖结果

        Args:
            soup: BeautifulSoup对象

        Returns:
            开奖结果列表，每项包含期号、开奖日期、红球号码、蓝球号码
        """
        results = []
        try:
            # 查找开奖结果表格
            table = soup.find("table", class_="zxkj_table")
            if not table:
                print("未找到开奖结果表格")
                return results

            # 解析表格行
            rows = table.find_all("tr")
            for row in rows[1:]:  # 跳过表头
                cells = row.find_all("td")
                if len(cells) >= 4:
                    # 期号
                    issue_number = cells[0].text.strip()
                    # 开奖日期
                    draw_date = cells[1].text.strip()
                    # 红球号码
                    red_balls_cell = cells[2]
                    red_balls = [ball.text.strip() for ball in red_balls_cell.find_all("em", class_="rr")]
                    # 蓝球号码
                    blue_ball_cell = cells[2]
                    blue_ball = blue_ball_cell.find("em", class_="bb").text.strip()

                    results.append({
                        "issue": issue_number,
                        "date": draw_date,
                        "red_balls": ",".join(red_balls),
                        "blue_ball": blue_ball
                    })
        except Exception as e:
            print(f"解析开奖结果出错: {e}")

        return results

    def crawl_history_data(self, page_count=15):
        """爬取历史开奖数据

        Args:
            page_count: 爬取的页面数量，每页20条，默认15页约300期

        Returns:
            开奖结果列表
        """
        all_results = []

        for page in range(1, page_count + 1):
            url = f"{self.base_url}?pageNum={page}"
            print(f"正在爬取第{page}页: {url}")
            soup = self.get_page(url)
            
            if soup:
                results = self.parse_draw_results(soup)
                all_results.extend(results)
                print(f"成功获取{len(results)}条记录")
            
            # 随机延迟，避免被反爬
            time.sleep(random.uniform(1, 3))

        return all_results

    def save_to_csv(self, results, filename="ssq_data.csv"):
        """保存结果到CSV文件

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
    
    # 爬取历史数据
    print(f"开始爬取双色球历史数据，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = crawler.crawl_history_data(page_count=15)  # 15页约300期
    
    # 保存数据
    if results:
        print(f"共获取{len(results)}期双色球开奖结果")
        crawler.save_to_csv(results)
    else:
        print("未获取到数据")


if __name__ == "__main__":
    main()