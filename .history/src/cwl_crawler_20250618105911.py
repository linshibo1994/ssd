#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 中国福利彩票官方网站版本
从中国福利彩票官方网站获取最近300期双色球开奖结果
如果官方网站数据不足，则从中彩网和500彩票网获取补充数据
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
        self.cp500_url = "https://datachart.500star.com/ssq/history/history.shtml"
        
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
            count: 获取的记录数量，默认300期，如果为None则获取所有期数

        Returns:
            开奖结果列表
        """
        results = []
        
        try:
            if count is None:
                print("正在从中国福利彩票官方网站获取所有期数的双色球开奖结果...")
            else:
                print(f"正在从中国福利彩票官方网站获取最近{count}期双色球开奖结果...")
            
            # 设置页面大小和初始页码
            page_size = 30
            page = 1
            
            # 如果指定了期数，计算需要请求的页数
            if count is not None:
                total_pages = (count + page_size - 1) // page_size  # 向上取整
            else:
                # 如果没有指定期数，设置一个较大的值，实际会在没有更多数据时退出
                total_pages = 1000  # 假设最多1000页，实际会在没有更多数据时退出
            
            # 使用分页方式获取数据
            while page <= total_pages:
                print(f"正在获取第{page}页数据 (每页{page_size}条)...")
                
                # 设置请求参数 - 使用分页方式
                params = {
                    "name": "ssq",  # 双色球
                    "pageNo": page,  # 页码
                    "pageSize": page_size,  # 每页数量
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
                    
                    print(f"成功获取第{page}页数据，当前共{len(results)}期")
                    
                    # 如果已经获取足够的数据，则退出循环（仅当count不为None时）
                    if count is not None and len(results) >= count:
                        break
                    
                    # 添加随机延迟，避免请求过于频繁
                    time.sleep(random.uniform(1, 3))
                else:
                    error_msg = data.get("message", "未知错误")
                    print(f"获取第{page}页数据失败: {error_msg}")
                    break
                
                # 如果当前页的数据量小于page_size，说明已经没有更多数据了
                if len(data.get("result", [])) < page_size:
                    break
                
                # 增加页码，继续获取下一页
                page += 1
                
                # 如果已经获取到足够的数据且不是获取所有期数，则退出循环
                if count is not None and len(results) >= count:
                    break
            
            print(f"从中国福利彩票官方网站成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中国福利彩票官方网站获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 按期号排序（降序）
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            # 只保留需要的期数（仅当count不为None时）
            if count is not None and len(results) > count:
                results = results[:count]
        
        return results
    
    def get_history_data_from_500cp(self):
        """
        从中彩网获取双色球历史数据
        
        Returns:
            开奖结果列表
        """
        results = []
        
        try:
            print(f"正在从中彩网获取双色球历史数据...")
            
            # 中彩网双色球历史数据页面数量（根据实际情况调整）
            total_pages = 125  # 预估页数，实际会在没有数据时退出
            
            for page in range(1, total_pages + 1):
                try:
                    print(f"正在获取第{page}页数据...")
                    
                    # 中彩网双色球历史数据URL
                    url = f"http://kaijiang.zhcw.com/zhcw/html/ssq/list_{page}.html"
                    
                    # 发送请求
                    response = requests.get(url, headers=self.cp500_headers, timeout=10)
                    # 设置正确的编码
                    response.encoding = 'utf-8'
                    
                    # 使用BeautifulSoup解析HTML
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # 查找数据表格
                    table = soup.find('table', attrs={'class': 'wqhgt'})
                    if not table:
                        print(f"第{page}页未找到数据表格，可能已到达最后一页")
                        break
                    
                    # 获取表格中的所有行
                    rows = table.find_all('tr')
                    if len(rows) <= 1:
                        print(f"第{page}页表格中没有数据行，可能已到达最后一页")
                        break
                    
                    # 跳过表头行，解析数据行
                    has_data = False
                    for row in rows[1:]:
                        try:
                            # 获取所有单元格
                            cells = row.find_all('td')
                            if len(cells) < 3:
                                continue
                            
                            # 获取开奖日期
                            date = cells[0].text.strip()
                            
                            # 获取期号
                            issue = cells[1].text.strip()
                            
                            # 获取红球和蓝球
                            ball_cell = cells[2]
                            all_balls = ball_cell.find_all('em')
                            
                            if len(all_balls) != 7:  # 6个红球 + 1个蓝球
                                continue
                            
                            # 获取红球
                            red_balls = []
                            for i in range(6):
                                red_balls.append(all_balls[i].text.strip().zfill(2))  # 补0，保持两位数格式
                            
                            # 获取蓝球
                            blue_ball = all_balls[6].text.strip().zfill(2)  # 补0，保持两位数格式
                            
                            results.append({
                                "issue": issue,
                                "date": date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_ball
                            })
                            has_data = True
                        except Exception as e:
                            print(f"解析行数据失败: {e}")
                            continue
                    
                    # 如果当前页没有有效数据，可能已到达最后一页
                    if not has_data:
                        print(f"第{page}页没有有效数据，可能已到达最后一页")
                        break
                    
                    # 添加随机延迟，避免请求过于频繁
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    print(f"获取第{page}页数据失败: {e}")
                    continue
            
            print(f"从中彩网成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中彩网获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        return results

    def get_history_data(self, count=300):
        """
        获取双色球历史数据，优先从官方网站获取，不足则从500彩票网补充

        Args:
            count: 获取的记录数量，默认300期，如果为None则获取所有期数

        Returns:
            开奖结果列表
        """
        # 从官方网站获取数据
        results = self.get_history_data_from_cwl(count)
        
        # 如果官方网站数据不足或需要获取所有期数，从500彩票网获取补充数据
        if count is None or (count is not None and len(results) < count):
            print(f"从官方网站获取的数据不足，将从500彩票网获取补充数据...")
            results_500cp = self.get_history_data_from_500cp()
            
            # 合并数据，去重
            existing_issues = set(item["issue"] for item in results)
            for item in results_500cp:
                if item["issue"] not in existing_issues:
                    results.append(item)
                    existing_issues.add(item["issue"])
            
            # 按期号排序（降序）
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            # 只保留需要的期数（如果指定了期数）
            if count is not None and len(results) > count:
                results = results[:count]
        
        print(f"共获取{len(results)}期双色球开奖结果")
        return results

    def save_to_csv(self, results, filename="ssq_data.csv"):
        """
        保存开奖结果到CSV文件

        Args:
            results: 开奖结果列表
            filename: 文件名
        """
        if not results:
            print("没有数据可保存")
            return
        
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(["issue", "date", "red_balls", "blue_ball"])
                # 写入数据
                for item in results:
                    writer.writerow([item["issue"], item["date"], item["red_balls"], item["blue_ball"]])
            
            print(f"数据已保存到文件: {file_path}")
        except Exception as e:
            print(f"保存数据失败: {e}")


def main():
    """主函数"""
    # 创建爬虫实例
    crawler = SSQCWLCrawler()
    
    # 获取历史数据
    results = crawler.get_history_data(count=300)
    
    # 保存数据
    if results:
        crawler.save_to_csv(results)


if __name__ == "__main__":
    main()