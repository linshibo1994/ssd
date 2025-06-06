#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块
从中彩网爬取最近300期双色球开奖结果
"""

import os
import csv
import time
import random
import requests
from datetime import datetime
from bs4 import BeautifulSoup


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
        
        # 中彩网双色球开奖结果页面URL
        self.base_url = "https://www.zhcw.com/kjxx/ssq/"
        self.history_url = "https://www.zhcw.com/kjxx/ssq/lskj_list.jsp"

    def get_page_content(self, url, params=None):
        """
        获取页面内容

        Args:
            url: 页面URL
            params: 请求参数

        Returns:
            BeautifulSoup对象
        """
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            # 使用html.parser解析器，避免对lxml的依赖
            return BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            print(f"获取页面内容失败: {e}")
            return None

    def parse_draw_results(self, soup):
        """
        解析开奖结果

        Args:
            soup: BeautifulSoup对象

        Returns:
            开奖结果字典
        """
        try:
            # 尝试获取期号和开奖日期
            issue_element = soup.select_one(".kjxq_firstBox .qh")
            date_element = soup.select_one(".kjxq_firstBox .kjrq")
            
            if issue_element and date_element:
                issue = issue_element.text.strip().replace("期号：", "")
                date = date_element.text.strip().replace("开奖日期：", "")
                
                # 获取红球号码
                red_ball_elements = soup.select(".kjxq_box01 .kjxq_ball01")
                red_balls = [ball.text.strip() for ball in red_ball_elements]
                
                # 获取蓝球号码
                blue_ball_element = soup.select_one(".kjxq_box01 .kjxq_ball02")
                blue_ball = blue_ball_element.text.strip() if blue_ball_element else ""
                
                if red_balls and blue_ball:
                    return {
                        "issue": issue,
                        "date": date,
                        "red_balls": ",".join(red_balls),
                        "blue_ball": blue_ball
                    }
            
            # 如果上面的方法失败，尝试其他选择器
            # 尝试获取表格中的数据
            table = soup.select_one(".kjxq_table")
            if table:
                rows = table.select("tr")
                if len(rows) > 1:  # 确保有数据行
                    cells = rows[1].select("td")
                    if len(cells) >= 3:
                        issue = cells[0].text.strip()
                        date = cells[1].text.strip()
                        
                        # 获取红球和蓝球
                        ball_elements = cells[2].select(".kjxq_ball01, .kjxq_ball02")
                        if len(ball_elements) >= 7:  # 6个红球+1个蓝球
                            red_balls = [ball.text.strip() for ball in ball_elements[:6]]
                            blue_ball = ball_elements[6].text.strip()
                            
                            return {
                                "issue": issue,
                                "date": date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_ball
                            }
            
            # 尝试其他可能的HTML结构
            ball_box = soup.select_one(".ball_box01")
            if ball_box:
                # 获取红球
                red_balls = [li.text.strip() for li in ball_box.select("li.ball_red")]
                # 获取蓝球
                blue_balls = [li.text.strip() for li in ball_box.select("li.ball_blue")]
                
                if red_balls and blue_balls:
                    # 尝试从其他地方获取期号和日期
                    issue_date_element = soup.select_one(".iSelectList a.cur")
                    if issue_date_element:
                        issue_text = issue_date_element.text.strip()
                        # 假设格式为 "2023001期(2023-01-01)"
                        parts = issue_text.split("期")
                        if len(parts) == 2:
                            issue = parts[0]
                            date = parts[1].strip("()").split(" ")[0]  # 提取日期部分
                            
                            return {
                                "issue": issue,
                                "date": date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_balls[0] if blue_balls else ""
                            }
            
            print("无法解析开奖结果，尝试其他方法")
            return None
        except Exception as e:
            print(f"解析开奖结果失败: {e}")
            return None

    def parse_history_list(self, soup):
        """
        解析历史列表页面

        Args:
            soup: BeautifulSoup对象

        Returns:
            开奖结果列表
        """
        results = []
        try:
            # 尝试查找历史列表表格
            tables = soup.select(".kjhis_table")
            for table in tables:
                rows = table.select("tr")
                for row in rows[1:]:  # 跳过表头
                    cells = row.select("td")
                    if len(cells) >= 4:
                        issue = cells[0].text.strip()
                        date = cells[1].text.strip()
                        
                        # 获取红球
                        red_ball_elements = cells[2].select(".kjhis_ball01")
                        red_balls = [ball.text.strip() for ball in red_ball_elements]
                        
                        # 获取蓝球
                        blue_ball_element = cells[2].select_one(".kjhis_ball02")
                        blue_ball = blue_ball_element.text.strip() if blue_ball_element else ""
                        
                        if red_balls and blue_ball:
                            results.append({
                                "issue": issue,
                                "date": date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_ball
                            })
            
            # 如果上面的方法失败，尝试其他选择器
            if not results:
                # 尝试其他可能的表格结构
                table = soup.select_one(".wqhgt")
                if table:
                    rows = table.select("tr")
                    for row in rows:
                        # 获取期号和日期
                        issue_cell = row.select_one("td:nth-child(2)")
                        date_cell = row.select_one("td:nth-child(1)")
                        
                        if issue_cell and date_cell:
                            issue = issue_cell.text.strip()
                            date = date_cell.text.strip()
                            
                            # 获取红球和蓝球
                            ball_elements = row.select("em")
                            if len(ball_elements) >= 7:  # 6个红球+1个蓝球
                                red_balls = [ball.text.strip() for ball in ball_elements[:6]]
                                blue_ball = ball_elements[6].text.strip()
                                
                                results.append({
                                    "issue": issue,
                                    "date": date,
                                    "red_balls": ",".join(red_balls),
                                    "blue_ball": blue_ball
                                })
        except Exception as e:
            print(f"解析历史列表失败: {e}")
        
        return results

    def crawl_history_data(self, count=300):
        """
        获取历史开奖数据

        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        results = []
        
        # 首先尝试获取最新一期的详情
        print(f"正在获取最新一期双色球开奖结果...")
        soup = self.get_page_content(self.base_url)
        if soup:
            latest_result = self.parse_draw_results(soup)
            if latest_result:
                results.append(latest_result)
                print(f"成功获取第{latest_result['issue']}期双色球开奖结果")
        
        # 然后获取历史列表
        print(f"正在获取历史双色球开奖结果...")
        page = 1
        while len(results) < count and page <= 30:  # 最多爬取30页，避免无限循环
            print(f"正在获取第{page}页历史数据...")
            params = {"page": page}
            soup = self.get_page_content(self.history_url, params)
            
            if not soup:
                print(f"获取第{page}页历史数据失败")
                break
            
            page_results = self.parse_history_list(soup)
            if not page_results:
                print(f"第{page}页没有找到开奖结果")
                break
            
            results.extend(page_results)
            print(f"第{page}页成功获取{len(page_results)}期双色球开奖结果")
            
            # 随机延迟，避免被反爬
            time.sleep(random.uniform(1, 3))
            page += 1
        
        # 确保不超过请求的数量
        return results[:count]

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