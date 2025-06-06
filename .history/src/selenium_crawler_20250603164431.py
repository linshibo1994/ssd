#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - Selenium版本
使用Selenium从中彩网爬取最近300期双色球开奖结果
"""

import os
import csv
import time
import random
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


class SSQSeleniumCrawler:
    """双色球数据获取类 - Selenium版本"""

    def __init__(self, data_dir="../data"):
        """
        初始化数据获取器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 中彩网双色球开奖结果页面URL
        self.base_url = "https://www.zhcw.com/kjxx/ssq/"
        self.history_url = "https://www.zhcw.com/kjxx/ssq/lskj_list.jsp"
        
        # 500彩票网双色球历史数据URL
        self.cp500_url = "https://datachart.500.com/ssq/history/history.shtml"
        
        # 初始化WebDriver
        self.driver = None
    
    def init_driver(self):
        """初始化WebDriver"""
        try:
            chrome_options = Options()
            # 无头模式，不显示浏览器窗口
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            # 设置User-Agent
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # 自动下载并配置ChromeDriver
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            self.driver.implicitly_wait(10)  # 设置隐式等待时间
            return True
        except Exception as e:
            print(f"初始化WebDriver失败: {e}")
            return False
    
    def close_driver(self):
        """关闭WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def get_data_from_zhcw(self):
        """
        从中彩网获取双色球历史数据
        
        Returns:
            开奖结果列表
        """
        results = []
        
        try:
            if not self.driver:
                if not self.init_driver():
                    return results
            
            # 访问中彩网双色球历史开奖页面
            print("正在访问中彩网双色球历史开奖页面...")
            self.driver.get(self.history_url)
            
            # 等待页面加载完成
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "wqhgt"))
            )
            
            # 点击查询按钮，显示更多历史数据
            try:
                # 点击自定义查询
                custom_query_btn = self.driver.find_element(By.XPATH, '//div[@class="wq-xlk01"]')
                custom_query_btn.click()
                time.sleep(1)
                
                # 输入查询期数
                query_input = self.driver.find_element(By.XPATH, '//div[@class="xc-q"]/input[@class="qscount"]')
                query_input.clear()
                query_input.send_keys("1000")  # 查询最近1000期，确保能获取到300期
                time.sleep(1)
                
                # 点击查询按钮
                search_btn = self.driver.find_element(By.XPATH, '//div[@class="JG-an03"]')
                search_btn.click()
                time.sleep(2)
            except Exception as e:
                print(f"设置查询参数失败: {e}")
            
            # 解析数据
            page = 1
            while page <= 30 and len(results) < 300:  # 最多爬取30页
                print(f"正在解析第{page}页数据...")
                
                try:
                    # 等待表格加载完成
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "wqhgt"))
                    )
                    
                    # 获取表格中的所有行
                    rows = self.driver.find_elements(By.XPATH, '//table[@class="wqhgt"]/tbody/tr')
                    
                    if not rows:
                        print(f"第{page}页未找到数据行")
                        break
                    
                    for row in rows:
                        try:
                            # 获取期号和日期
                            cells = row.find_elements(By.TAG_NAME, "td")
                            if len(cells) < 3:
                                continue
                            
                            issue = cells[0].text.strip()
                            date = cells[1].text.strip()
                            
                            # 获取红球和蓝球
                            ball_cell = cells[2]
                            red_balls = [ball.text.strip() for ball in ball_cell.find_elements(By.CLASS_NAME, "red")]
                            blue_ball_elements = ball_cell.find_elements(By.CLASS_NAME, "blue")
                            
                            if not red_balls or not blue_ball_elements:
                                continue
                            
                            blue_ball = blue_ball_elements[0].text.strip()
                            
                            results.append({
                                "issue": issue,
                                "date": date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_ball
                            })
                        except Exception as e:
                            print(f"解析行数据失败: {e}")
                            continue
                    
                    print(f"当前已获取{len(results)}期数据")
                    
                    # 如果已经获取足够的数据，或者没有更多页面，则退出循环
                    if len(results) >= 300:
                        break
                    
                    # 点击下一页
                    try:
                        next_page = self.driver.find_element(By.XPATH, f'//a[@title="{page + 1}"]')
                        next_page.click()
                        time.sleep(2)  # 等待页面加载
                        page += 1
                    except Exception as e:
                        print(f"点击下一页失败: {e}")
                        break
                except Exception as e:
                    print(f"解析第{page}页数据失败: {e}")
                    break
        except Exception as e:
            print(f"从中彩网获取数据失败: {e}")
        
        return results
    
    def get_data_from_500cp(self):
        """
        从500彩票网获取双色球历史数据
        
        Returns:
            开奖结果列表
        """
        results = []
        
        try:
            if not self.driver:
                if not self.init_driver():
                    return results
            
            # 访问500彩票网双色球历史数据页面
            print("正在访问500彩票网双色球历史数据页面...")
            self.driver.get(self.cp500_url)
            
            # 等待页面加载完成
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//table[@width="100%"]'))
            )
            
            # 解析数据表格
            table = self.driver.find_element(By.XPATH, '//table[@width="100%"]')
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            # 跳过表头行
            for row in rows[1:]:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 9:
                        continue
                    
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
                    
                    # 如果已经获取足够的数据，则退出循环
                    if len(results) >= 300:
                        break
                except Exception as e:
                    print(f"解析行数据失败: {e}")
                    continue
            
            print(f"从500彩票网获取了{len(results)}期数据")
        except Exception as e:
            print(f"从500彩票网获取数据失败: {e}")
        
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
        
        try:
            # 初始化WebDriver
            if not self.init_driver():
                print("初始化WebDriver失败，无法获取数据")
                return results
            
            # 首先尝试从中彩网获取数据
            print("尝试从中彩网获取数据...")
            zhcw_results = self.get_data_from_zhcw()
            if zhcw_results:
                results.extend(zhcw_results)
                print(f"从中彩网获取了{len(zhcw_results)}期数据")
            
            # 如果从中彩网获取的数据不足，则尝试从500彩票网获取补充数据
            if len(results) < count:
                print(f"从中彩网获取的数据不足{count}期，尝试从500彩票网获取补充数据...")
                cp500_results = self.get_data_from_500cp()
                
                # 合并结果并去重
                for result in cp500_results:
                    if not any(r["issue"] == result["issue"] for r in results):
                        results.append(result)
            
            # 按期号排序（降序）
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            print(f"共获取{len(results)}期数据")
        finally:
            # 关闭WebDriver
            self.close_driver()
        
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
    crawler = SSQSeleniumCrawler()
    
    # 获取历史数据
    results = crawler.crawl_history_data(count=300)
    
    # 保存数据
    crawler.save_to_csv(results)


if __name__ == "__main__":
    main() 