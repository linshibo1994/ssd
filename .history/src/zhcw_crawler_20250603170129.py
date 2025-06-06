#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 中彩网版本
从中彩网获取最近300期双色球开奖结果
模拟真实的翻页操作获取历史数据
"""

import os
import csv
import time
import json
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime


class SSQZhCWCrawler:
    """双色球数据获取类 - 中彩网版本"""

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
        self.history_url = "https://www.zhcw.com/kjxx/ssq/kjhmlist_1.shtml"
        self.query_url = "https://www.zhcw.com/kjxx/ssq/kjhmlist.shtml"
        
        # 请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.zhcw.com/kjxx/ssq/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
        }
        
        # 存储已获取的期号，避免重复
        self.fetched_issues = set()
    
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
            # 查找开奖结果表格
            table = soup.select_one("table.wqhgt")
            if not table:
                print("未找到开奖结果表格")
                return results
            
            # 获取所有行
            rows = table.select("tr")
            if len(rows) <= 1:  # 跳过表头
                print("表格中没有数据行")
                return results
            
            # 解析每一行数据
            for row in rows[1:]:  # 跳过表头
                try:
                    # 获取所有单元格
                    cells = row.select("td")
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
                    red_elements = ball_cell.select("em.rr")
                    for elem in red_elements:
                        red_balls.append(elem.text.strip().zfill(2))
                    
                    # 获取蓝球
                    blue_elements = ball_cell.select("em.br")
                    if not blue_elements:
                        continue
                    blue_ball = blue_elements[0].text.strip().zfill(2)
                    
                    # 如果已经获取过这个期号，则跳过
                    if issue in self.fetched_issues:
                        continue
                    
                    self.fetched_issues.add(issue)
                    
                    results.append({
                        "issue": issue,
                        "date": date,
                        "red_balls": ",".join(red_balls),
                        "blue_ball": blue_ball
                    })
                except Exception as e:
                    print(f"解析行数据失败: {e}")
                    continue
        except Exception as e:
            print(f"解析历史列表失败: {e}")
        
        return results
    
    def get_history_data_by_page(self, count=300):
        """
        通过翻页获取历史开奖数据

        Args:
            count: 获取的记录数量，默认300期

        Returns:
            开奖结果列表
        """
        results = []
        page = 1
        max_page = 50  # 最大翻页次数，避免无限循环
        
        try:
            print(f"正在从中彩网获取最近{count}期双色球开奖结果...")
            
            while len(results) < count and page <= max_page:
                # 构造翻页URL
                page_url = f"https://www.zhcw.com/kjxx/ssq/kjhmlist_{page}.shtml"
                if page == 1:
                    page_url = self.history_url
                
                print(f"正在获取第{page}页数据...")
                
                # 获取页面内容
                soup = self.get_page_content(page_url)
                if not soup:
                    print(f"获取第{page}页失败，尝试继续...")
                    page += 1
                    continue
                
                # 解析页面内容
                page_results = self.parse_history_list(soup)
                if not page_results:
                    print(f"第{page}页未解析到数据，尝试继续...")
                    page += 1
                    continue
                
                # 添加到结果列表
                results.extend(page_results)
                print(f"当前已获取{len(results)}期数据")
                
                # 如果已经获取足够的数据，则退出循环
                if len(results) >= count:
                    break
                
                # 翻页
                page += 1
                
                # 添加随机延迟，避免请求过于频繁
                time.sleep(random.uniform(1, 3))
            
            print(f"从中彩网成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"获取历史数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 按期号排序（降序）
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        # 只返回前count条数据
        return results[:count]
    
    def get_history_data_by_custom_query(self, count=300):
        """
        通过自定义查询获取历史开奖数据
        
        Args:
            count: 获取的记录数量，默认300期
            
        Returns:
            开奖结果列表
        """
        results = []
        
        try:
            print(f"正在从中彩网通过自定义查询获取最近{count}期双色球开奖结果...")
            
            # 获取最新一期数据，确定当前期号
            soup = self.get_page_content(self.history_url)
            if not soup:
                print("获取最新一期数据失败")
                return results
            
            latest_issue = None
            table = soup.select_one("table.wqhgt")
            if table:
                rows = table.select("tr")
                if len(rows) > 1:
                    cells = rows[1].select("td")
                    if len(cells) > 0:
                        latest_issue = cells[0].text.strip()
            
            if not latest_issue:
                print("无法获取最新一期期号")
                return results
            
            print(f"最新一期为: {latest_issue}")
            
            # 计算开始期号
            start_issue = str(int(latest_issue) - count + 1)
            if int(start_issue) < 1:
                start_issue = "1"
            
            print(f"开始获取从第{start_issue}期到第{latest_issue}期的数据...")
            
            # 构造自定义查询参数
            # 中彩网的自定义查询表单需要提交POST请求
            form_data = {
                "issueCount": count,
                "issueStart": start_issue,
                "issueEnd": latest_issue,
                "queryType": "1"  # 按期号查询
            }
            
            # 发送POST请求
            for attempt in range(3):
                try:
                    response = requests.post(
                        self.query_url, 
                        headers=self.headers, 
                        data=form_data, 
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    # 解析响应内容
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # 解析开奖结果
                    results = self.parse_history_list(soup)
                    
                    if results:
                        print(f"自定义查询成功获取{len(results)}期数据")
                        break
                    else:
                        print(f"自定义查询未返回数据 (尝试 {attempt+1}/3)")
                        time.sleep(random.uniform(1, 3))
                except Exception as e:
                    print(f"自定义查询失败 (尝试 {attempt+1}/3): {e}")
                    time.sleep(random.uniform(1, 3))
            
            # 如果自定义查询失败，则尝试分批次查询
            if not results:
                print("自定义查询失败，尝试分批次查询...")
                
                # 每批次查询的期数
                batch_size = 50
                
                # 计算需要查询的批次数
                num_batches = (count + batch_size - 1) // batch_size
                
                for batch in range(num_batches):
                    # 计算当前批次的开始和结束期号
                    current_batch_end = str(int(latest_issue) - batch * batch_size)
                    current_batch_start = str(int(latest_issue) - (batch + 1) * batch_size + 1)
                    
                    # 确保开始期号不小于1
                    if int(current_batch_start) < 1:
                        current_batch_start = "1"
                    
                    print(f"获取第{batch+1}批数据，期号范围: {current_batch_start} - {current_batch_end}")
                    
                    # 构造查询参数
                    form_data = {
                        "issueStart": current_batch_start,
                        "issueEnd": current_batch_end,
                        "queryType": "1"  # 按期号查询
                    }
                    
                    # 发送POST请求
                    try:
                        response = requests.post(
                            self.query_url, 
                            headers=self.headers, 
                            data=form_data, 
                            timeout=10
                        )
                        response.raise_for_status()
                        
                        # 解析响应内容
                        soup = BeautifulSoup(response.text, "html.parser")
                        
                        # 解析开奖结果
                        batch_results = self.parse_history_list(soup)
                        
                        if batch_results:
                            results.extend(batch_results)
                            print(f"成功获取第{batch+1}批数据，当前共{len(results)}期")
                        else:
                            print(f"第{batch+1}批未返回数据")
                        
                        # 如果已经获取足够的数据，则退出循环
                        if len(results) >= count:
                            break
                        
                        # 添加随机延迟，避免请求过于频繁
                        time.sleep(random.uniform(1, 3))
                    except Exception as e:
                        print(f"获取第{batch+1}批数据失败: {e}")
                        time.sleep(random.uniform(1, 3))
            
            print(f"从中彩网成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"获取历史数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 按期号排序（降序）
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        # 只返回前count条数据
        return results[:count]
    
    def get_history_data(self, count=300):
        """
        获取历史开奖数据，尝试多种方法
        
        Args:
            count: 获取的记录数量，默认300期
            
        Returns:
            开奖结果列表
        """
        # 首先尝试通过自定义查询获取数据
        results = self.get_history_data_by_custom_query(count)
        
        # 如果自定义查询获取的数据不足，则尝试通过翻页获取
        if len(results) < count:
            print(f"通过自定义查询只获取到{len(results)}期数据，尝试通过翻页获取更多数据...")
            
            # 记录已获取的期号
            self.fetched_issues = set(r["issue"] for r in results)
            
            # 通过翻页获取更多数据
            page_results = self.get_history_data_by_page(count - len(results))
            
            # 合并结果
            results.extend(page_results)
            
            # 重新排序
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
            
            print(f"合并后共有{len(results)}期数据")
        
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
    crawler = SSQZhCWCrawler()
    
    # 获取历史数据
    results = crawler.get_history_data(count=300)
    
    # 保存数据
    crawler.save_to_csv(results)


if __name__ == "__main__":
    main() 