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
            "Referer": "https://www.zhcw.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
        }
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 中彩网双色球开奖结果页面URL
        self.base_url = "https://www.zhcw.com/kjxx/ssq/"
        self.history_url = "https://www.zhcw.com/kjxx/ssq/lskj_list.jsp"
        self.api_url = "https://www.zhcw.com/kj_ssq_hq/ssq_hqkj_all.json"  # 接口URL，可能会用到

    def get_page_content(self, url, params=None, retry=3):
        """
        获取页面内容

        Args:
            url: 页面URL
            params: 请求参数
            retry: 重试次数

        Returns:
            BeautifulSoup对象
        """
        for attempt in range(retry):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                # 使用html.parser解析器，避免对lxml的依赖
                return BeautifulSoup(response.text, "html.parser")
            except Exception as e:
                print(f"获取页面内容失败 (尝试 {attempt+1}/{retry}): {e}")
                if attempt < retry - 1:
                    # 随机延迟1-3秒后重试
                    time.sleep(random.uniform(1, 3))
                else:
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
                    for row in rows[1:]:  # 跳过表头
                        # 获取期号和日期
                        cells = row.select("td")
                        if len(cells) >= 3:
                            issue = cells[0].text.strip()
                            date = cells[1].text.strip()
                            
                            # 获取红球和蓝球
                            ball_cell = cells[2]
                            red_ball_elements = ball_cell.select(".red")
                            blue_ball_element = ball_cell.select_one(".blue")
                            
                            if red_ball_elements and blue_ball_element:
                                red_balls = [ball.text.strip() for ball in red_ball_elements]
                                blue_ball = blue_ball_element.text.strip()
                                
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
                print(f"获取第{page}页失败，尝试继续...")
                page += 1
                continue
            
            page_results = self.parse_history_list(soup)
            if not page_results:
                print(f"第{page}页未解析到数据，尝试其他方法...")
                # 如果常规方法失败，尝试使用500彩票网作为备用数据源
                if page == 1 and len(results) < 10:
                    backup_results = self.get_data_from_500cp()
                    if backup_results:
                        results.extend(backup_results)
                        print(f"从备用数据源获取了{len(backup_results)}期数据")
                        break
                page += 1
                continue
            
            # 去重添加
            for result in page_results:
                if not any(r["issue"] == result["issue"] for r in results):
                    results.append(result)
            
            print(f"当前已获取{len(results)}期数据")
            
            # 如果已经获取足够的数据，或者没有更多数据，则退出循环
            if len(results) >= count or len(page_results) < 10:  # 假设每页至少有10条数据
                break
            
            page += 1
            # 添加随机延迟，避免请求过于频繁
            time.sleep(random.uniform(1, 3))
        
        # 按期号排序（降序）
        results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        # 如果数据不足，尝试使用备用数据源
        if len(results) < count:
            print(f"从中彩网只获取到{len(results)}期数据，尝试从备用数据源获取...")
            backup_results = self.get_data_from_500cp()
            if backup_results:
                # 合并结果并去重
                for result in backup_results:
                    if not any(r["issue"] == result["issue"] for r in results):
                        results.append(result)
                
                # 重新排序
                results.sort(key=lambda x: int(x["issue"]), reverse=True)
                print(f"合并后共有{len(results)}期数据")
        
        # 只返回前count条数据
        return results[:count]

    def get_data_from_500cp(self):
        """
        从500彩票网获取双色球历史数据（备用数据源）
        
        Returns:
            开奖结果列表
        """
        try:
            print("尝试从500彩票网获取数据...")
            url = "https://datachart.500.com/ssq/history/history.shtml"
            soup = self.get_page_content(url)
            if not soup:
                return []
            
            results = []
            # 查找表格
            table = soup.select_one("table[width='100%']")
            if not table:
                return []
            
            rows = table.select("tr")
            for row in rows[1:]:  # 跳过表头
                cells = row.select("td")
                if len(cells) >= 8:
                    try:
                        issue = cells[0].text.strip()
                        date = cells[1].text.strip()
                        
                        # 获取红球
                        red_balls = []
                        for i in range(2, 8):
                            red_balls.append(cells[i].text.strip())
                        
                        # 获取蓝球
                        blue_ball = cells[8].text.strip()
                        
                        results.append({
                            "issue": issue,
                            "date": date,
                            "red_balls": ",".join(red_balls),
                            "blue_ball": blue_ball
                        })
                    except Exception as e:
                        print(f"解析500彩票网数据行失败: {e}")
                        continue
            
            print(f"从500彩票网获取了{len(results)}期数据")
            return results
        except Exception as e:
            print(f"从500彩票网获取数据失败: {e}")
            return []

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
    crawler = SSQCrawler()
    
    # 获取历史数据
    results = crawler.crawl_history_data(count=300)
    
    # 保存数据
    crawler.save_to_csv(results)


if __name__ == "__main__":
    main()