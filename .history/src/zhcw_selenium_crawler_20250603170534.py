#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 中彩网Selenium版本
使用Selenium从中彩网获取最近300期双色球开奖结果
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


class SSQZhCWSeleniumCrawler:
    """双色球数据获取类 - 中彩网Selenium版本"""

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
            if not self.driver:
                if not self.init_driver():
                    print("初始化WebDriver失败，无法获取数据")
                    return results
            
            print(f"正在从中彩网获取最近{count}期双色球开奖结果...")
            
            # 访问中彩网双色球开奖结果页面
            self.driver.get(self.base_url)
            
            # 等待页面加载完成
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "cxqk"))
            )
            
            # 点击"自定义查询"按钮
            try:
                custom_query_btn = self.driver.find_element(By.CLASS_NAME, "wq-xlk01")
                custom_query_btn.click()
                time.sleep(2)  # 等待下拉菜单显示
                
                # 确保在"按期数"选项卡
                tabs = self.driver.find_elements(By.CLASS_NAME, "tj0")
                if tabs and len(tabs) > 0:
                    # 如果第一个选项卡没有"on"类，则点击它
                    if "on" not in tabs[0].get_attribute("class"):
                        tabs[0].click()
                        time.sleep(1)
                
                # 输入查询期数
                query_input = self.driver.find_element(By.CLASS_NAME, "qscount")
                query_input.clear()
                query_input.send_keys(str(count))  # 查询最近count期
                time.sleep(1)
                
                # 点击"开始查询"按钮
                search_btn = self.driver.find_element(By.XPATH, "//div[contains(@class, 'JG-an03')]")
                search_btn.click()
                time.sleep(5)  # 等待查询结果加载
                
                # 等待表格加载完成
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "wqhgt"))
                )
            except Exception as e:
                print(f"设置自定义查询失败: {e}")
                # 如果自定义查询失败，尝试使用默认的近100期查询
                try:
                    near_100_btn = self.driver.find_element(By.XPATH, "//span[@class='annq' and text()='近100期']")
                    near_100_btn.click()
                    time.sleep(5)  # 等待查询结果加载
                except Exception as e2:
                    print(f"设置近100期查询也失败了: {e2}")
            
            # 解析表格数据
            try:
                # 查找表格
                table = self.driver.find_element(By.CLASS_NAME, "wqhgt")
                
                # 获取所有行
                rows = table.find_elements(By.TAG_NAME, "tr")
                
                # 跳过表头行，解析数据行
                for row in rows[1:]:
                    try:
                        # 获取所有单元格
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) < 3:
                            continue
                        
                        # 获取期号
                        issue = cells[0].text.strip()
                        
                        # 获取开奖日期
                        date = cells[1].text.strip()
                        
                        # 获取开奖号码
                        ball_cell = cells[2]
                        
                        # 获取红球
                        red_balls = []
                        red_elements = ball_cell.find_elements(By.CLASS_NAME, "rr")
                        for elem in red_elements:
                            red_balls.append(elem.text.strip().zfill(2))
                        
                        # 获取蓝球
                        blue_elements = ball_cell.find_elements(By.CLASS_NAME, "br")
                        if not blue_elements:
                            continue
                        blue_ball = blue_elements[0].text.strip().zfill(2)
                        
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
                
                print(f"成功获取{len(results)}期数据")
                
                # 如果获取的数据不足，尝试点击"下一页"按钮获取更多数据
                page = 1
                max_page = 30  # 最大翻页次数，避免无限循环
                
                while len(results) < count and page < max_page:
                    try:
                        # 查找下一页按钮
                        next_page_btn = self.driver.find_element(By.XPATH, "//a[text()='»']")
                        
                        # 如果下一页按钮不可点击，则退出循环
                        if "disabled" in next_page_btn.get_attribute("class"):
                            break
                        
                        # 点击下一页按钮
                        next_page_btn.click()
                        time.sleep(3)  # 等待页面加载
                        
                        page += 1
                        print(f"正在获取第{page}页数据...")
                        
                        # 解析新页面的表格数据
                        table = self.driver.find_element(By.CLASS_NAME, "wqhgt")
                        rows = table.find_elements(By.TAG_NAME, "tr")
                        
                        for row in rows[1:]:  # 跳过表头
                            try:
                                # 获取所有单元格
                                cells = row.find_elements(By.TAG_NAME, "td")
                                if len(cells) < 3:
                                    continue
                                
                                # 获取期号
                                issue = cells[0].text.strip()
                                
                                # 检查是否已经获取过这个期号
                                if any(r["issue"] == issue for r in results):
                                    continue
                                
                                # 获取开奖日期
                                date = cells[1].text.strip()
                                
                                # 获取开奖号码
                                ball_cell = cells[2]
                                
                                # 获取红球
                                red_balls = []
                                red_elements = ball_cell.find_elements(By.CLASS_NAME, "rr")
                                for elem in red_elements:
                                    red_balls.append(elem.text.strip().zfill(2))
                                
                                # 获取蓝球
                                blue_elements = ball_cell.find_elements(By.CLASS_NAME, "br")
                                if not blue_elements:
                                    continue
                                blue_ball = blue_elements[0].text.strip().zfill(2)
                                
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
                        
                        print(f"当前已获取{len(results)}期数据")
                    except Exception as e:
                        print(f"翻页失败: {e}")
                        break
            except Exception as e:
                print(f"解析表格数据失败: {e}")
            
            print(f"从中彩网成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"获取历史数据失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 关闭WebDriver
            self.close_driver()
        
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
    crawler = SSQZhCWSeleniumCrawler()
    
    # 获取历史数据
    results = crawler.get_history_data(count=300)
    
    # 保存数据
    crawler.save_to_csv(results)


if __name__ == "__main__":
    main() 