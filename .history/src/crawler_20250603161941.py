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

    def __init__(self, base_url="https://www.zhcw.com/kjxx/ssq/kjxq/", data_dir="../data"):
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
            return BeautifulSoup(response.text, "html.parser")  # 使用html.parser替代lxml
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
            # 尝试不同的选择器来找到包含开奖结果的元素
            table = soup.find("table", class_="zxkj_table")
            if not table:
                # 尝试其他可能的选择器
                table = soup.find("table", class_="kjxq_table")
            
            if not table:
                # 尝试查找包含开奖信息的div
                content_div = soup.find("div", class_="kjxq_box")
                if content_div:
                    # 打印找到的内容结构，帮助调试
                    print("找到开奖信息容器，尝试解析内容")
                    # 尝试查找期号信息
                    issue_info = content_div.find("div", class_="kjxq_qh")
                    if issue_info:
                        issue_number = issue_info.text.strip()
                        print(f"找到期号信息: {issue_number}")
                    
                    # 尝试查找开奖号码
                    ball_div = content_div.find("div", class_="kjxq_hm")
                    if ball_div:
                        print("找到开奖号码区域")
                        # 查找红球
                        red_balls = []
                        red_elements = ball_div.find_all("span", class_="kjxq_hmhq")
                        for elem in red_elements:
                            red_balls.append(elem.text.strip())
                        
                        # 查找蓝球
                        blue_ball = ""
                        blue_element = ball_div.find("span", class_="kjxq_hmlq")
                        if blue_element:
                            blue_ball = blue_element.text.strip()
                        
                        # 查找开奖日期
                        date_info = content_div.find("div", class_="kjxq_rq")
                        draw_date = ""
                        if date_info:
                            draw_date = date_info.text.strip()
                        
                        if red_balls and blue_ball:
                            results.append({
                                "issue": issue_number,
                                "date": draw_date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_ball
                            })
                            print(f"成功解析第{issue_number}期开奖结果")
                else:
                    print("未找到开奖结果容器")
                    # 打印页面结构，帮助调试
                    print("页面结构:")
                    print(soup.prettify()[:500])  # 只打印前500个字符
                return results
            
            # 如果找到表格，按原来的方式解析
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

        # 首先尝试获取最新一期的详细信息
        print("尝试获取最新一期开奖详情")
        soup = self.get_page(self.base_url)
        if soup:
            results = self.parse_draw_results(soup)
            all_results.extend(results)
            print(f"从详情页获取了{len(results)}条记录")
        
        # 然后尝试获取历史列表页
        history_url = "https://www.zhcw.com/kjxx/ssq/"
        for page in range(1, page_count + 1):
            url = f"{history_url}?pageNum={page}"
            print(f"正在爬取第{page}页: {url}")
            soup = self.get_page(url)
            
            if soup:
                # 尝试查找历史开奖列表
                history_table = soup.find("table", class_="kjhis_table")
                if history_table:
                    print("找到历史开奖列表表格")
                    rows = history_table.find_all("tr")
                    for row in rows[1:]:  # 跳过表头
                        try:
                            cells = row.find_all("td")
                            if len(cells) >= 3:
                                # 期号
                                issue_number = cells[0].text.strip()
                                # 开奖日期
                                draw_date = cells[1].text.strip()
                                # 开奖号码
                                ball_cell = cells[2]
                                
                                # 查找红球和蓝球
                                red_balls = []
                                blue_ball = ""
                                
                                # 尝试不同的类名查找红球和蓝球
                                red_elements = ball_cell.find_all("span", class_="kjhis_red")
                                if not red_elements:
                                    red_elements = ball_cell.find_all("em", class_="rr")
                                
                                for elem in red_elements:
                                    red_balls.append(elem.text.strip())
                                
                                blue_element = ball_cell.find("span", class_="kjhis_blue")
                                if not blue_element:
                                    blue_element = ball_cell.find("em", class_="bb")
                                
                                if blue_element:
                                    blue_ball = blue_element.text.strip()
                                
                                if red_balls and blue_ball:
                                    results = {
                                        "issue": issue_number,
                                        "date": draw_date,
                                        "red_balls": ",".join(red_balls),
                                        "blue_ball": blue_ball
                                    }
                                    all_results.append(results)
                                    print(f"成功解析第{issue_number}期开奖结果")
                        except Exception as e:
                            print(f"解析行数据出错: {e}")
                else:
                    print("未找到历史开奖列表表格")
                    # 尝试使用原来的解析方法
                    results = self.parse_draw_results(soup)
                    all_results.extend(results)
                    print(f"使用原方法获取了{len(results)}条记录")
            
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