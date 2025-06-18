#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 全期数版本
从中国福利彩票官方网站和中彩网获取所有期双色球开奖结果
确保爬取所有期数，不会因去重而遗漏任何期号
"""

import os
import csv
import time
import json
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime


class SSQAllCrawler:
    """双色球数据获取类 - 全部期数版本"""

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
        
        # 中彩网双色球历史数据URL
        self.zhcw_url = "http://kaijiang.zhcw.com/zhcw/html/ssq/list_{}.html"
        
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
        
        # 中彩网请求头
        self.zhcw_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://kaijiang.zhcw.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
        }
    
    def get_history_data_from_cwl(self):
        """
        从中国福利彩票官方网站获取所有历史开奖数据
        
        Returns:
            开奖结果列表, 已爬取的期号集合
        """
        results = []
        issues = set()  # 记录已爬取的期号
        
        try:
            print("正在从中国福利彩票官方网站获取所有期数的双色球开奖结果...")
            
            # 设置页面大小和初始页码
            page_size = 50  # 增大每页数量，减少请求次数
            page = 1
            
            # 使用分页方式获取数据
            while True:
                print(f"正在获取第{page}页数据 (每页{page_size}条)...")
                
                # 设置请求参数 - 使用分页方式
                params = {
                    "name": "ssq",  # 双色球
                    "pageNo": page,  # 页码
                    "pageSize": page_size,  # 每页数量
                    "systemType": "PC"  # 系统类型
                }
                
                try:
                    # 发送请求
                    response = requests.get(self.api_url, headers=self.headers, params=params, timeout=15)
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
                                
                                if issue not in issues:  # 避免重复添加
                                    results.append({
                                        "issue": issue,
                                        "date": date,
                                        "red_balls": ",".join(red_balls),
                                        "blue_ball": blue_ball
                                    })
                                    issues.add(issue)
                            except Exception as e:
                                print(f"解析数据项失败: {e}")
                                continue
                        
                        print(f"已从官网获取{len(results)}期数据")
                        
                        # 如果当前页的数据量小于page_size，说明已经没有更多数据了
                        if len(data.get("result", [])) < page_size:
                            break
                        
                        # 添加随机延迟，避免请求过于频繁
                        time.sleep(random.uniform(1.5, 3))
                    else:
                        error_msg = data.get("message", "未知错误")
                        print(f"获取第{page}页数据失败: {error_msg}")
                        break
                except Exception as e:
                    print(f"请求第{page}页数据失败: {e}")
                    if "403" in str(e):
                        print("检测到网站反爬，将切换到中彩网获取数据")
                        break
                
                # 增加页码，继续获取下一页
                page += 1
            
            print(f"从中国福利彩票官方网站成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中国福利彩票官方网站获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 按期号排序（降序）
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        return results, issues
    
    def get_history_data_from_zhcw(self, existing_issues=None):
        """
        从中彩网获取双色球历史数据
        
        Args:
            existing_issues: 已存在的期号集合，用于去重
        
        Returns:
            开奖结果列表, 已爬取的期号集合
        """
        results = []
        issues = set() if existing_issues is None else existing_issues.copy()
        
        try:
            print(f"正在从中彩网获取双色球历史数据...")
            
            # 中彩网双色球历史数据页面数量（根据实际情况调整）
            total_pages = 150  # 增大预估页数，确保爬取所有历史数据
            
            for page in range(1, total_pages + 1):
                try:
                    print(f"正在获取第{page}页数据...")
                    
                    # 中彩网双色球历史数据URL
                    url = self.zhcw_url.format(page)
                    
                    # 发送请求
                    response = requests.get(url, headers=self.zhcw_headers, timeout=15)
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
                            
                            # 检查期号是否已存在，避免重复添加
                            if issue in issues:
                                continue
                            issues.add(issue)
                            
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
                    time.sleep(random.uniform(1.5, 3))
                    
                except Exception as e:
                    print(f"获取第{page}页数据失败: {e}")
                    continue
            
            print(f"从中彩网成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中彩网获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 按期号排序（降序）
        if results:
            results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        return results, issues

    def get_all_history_data(self):
        """
        获取所有双色球历史数据，优先从官方网站获取，再从中彩网获取补充

        Returns:
            开奖结果列表
        """
        print("开始获取所有双色球历史数据...")
        
        # 从官方网站获取数据
        cwl_results, existing_issues = self.get_history_data_from_cwl()
        print(f"从官方网站获取了{len(cwl_results)}期数据")
        
        # 从中彩网获取数据，包括官方网站没有的早期数据
        zhcw_results, all_issues = self.get_history_data_from_zhcw(existing_issues)
        print(f"从中彩网补充获取了{len(zhcw_results)}期数据")
        
        # 合并两个来源的数据
        all_results = cwl_results + zhcw_results
        
        # 按期号排序（降序）
        all_results.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        print(f"共获取{len(all_results)}期双色球开奖结果")
        return all_results

    def save_to_csv(self, results, filename="ssq_data_all.csv", mode="merge"):
        """
        将开奖结果保存到CSV文件
        
        Args:
            results: 开奖结果列表
            filename: 保存的文件名
            mode: 保存模式，可选 'merge'(合并已有数据) 或 'overwrite'(覆盖已有数据)
        
        Returns:
            保存的文件路径
        """
        if not results:
            print("没有数据可保存")
            return None
        
        # 构建完整的文件路径
        file_path = os.path.join(self.data_dir, filename)
        
        # 检查文件是否已存在
        file_exists = os.path.exists(file_path)
        
        # 如果文件已存在且为合并模式，读取已有数据并合并
        if file_exists and mode == "merge":
            print(f"发现已有数据文件: {file_path}")
            try:
                # 读取现有数据
                existing_data = []
                existing_issues = set()
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data.append(row)
                        existing_issues.add(row['issue'])
                
                print(f"已有文件包含{len(existing_data)}期数据")
                
                # 过滤新数据中的重复期号
                new_data = []
                for result in results:
                    if result['issue'] not in existing_issues:
                        new_data.append(result)
                        existing_issues.add(result['issue'])
                
                print(f"发现{len(new_data)}期新数据需要添加")
                
                if new_data:
                    # 合并数据并按期号排序
                    merged_data = existing_data + new_data
                    merged_data.sort(key=lambda x: int(x['issue']), reverse=True)
                    
                    # 重新写入文件
                    with open(file_path, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['issue', 'date', 'red_balls', 'blue_ball'])
                        writer.writeheader()
                        writer.writerows(merged_data)
                    
                    print(f"合并后文件共包含{len(merged_data)}期数据")
                else:
                    print("没有新数据需要添加")
                    
            except Exception as e:
                print(f"读取或合并已有数据失败: {e}，将覆盖现有文件")
                mode = "overwrite"
        
        # 如果文件不存在或为覆盖模式，则直接写入所有数据
        if not file_exists or mode == "overwrite":
            print(f"{'创建新文件' if not file_exists else '覆盖现有文件'}: {file_path}")
            
            # 写入数据
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['issue', 'date', 'red_balls', 'blue_ball'])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"文件保存成功，共{len(results)}期数据")
        
        # 验证数据完整性
        self.verify_data_integrity(file_path)
        
        return file_path
    
    def verify_data_integrity(self, file_path):
        """
        验证数据完整性，检查期号序列是否有缺失
        
        Args:
            file_path: 数据文件路径
        """
        print("开始验证数据完整性...")
        
        try:
            # 读取数据文件
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # 获取所有期号并转为整数
            issues = [int(row['issue']) for row in rows]
            issues.sort(reverse=True)  # 降序排序
            
            # 检查期号是否连续
            if not issues:
                print("警告: 数据文件为空")
                return
            
            # 获取最早和最晚的期号
            latest_issue = issues[0]
            earliest_issue = issues[-1]
            
            # 预期的总期数（如果完全连续）
            expected_count = latest_issue - earliest_issue + 1
            
            # 检查是否有期号缺失
            if len(issues) == expected_count:
                print(f"数据完整性验证通过，从{earliest_issue}期到{latest_issue}期，共{len(issues)}期，无缺失")
            else:
                print(f"警告: 数据不完整，从{earliest_issue}期到{latest_issue}期应有{expected_count}期，但实际只有{len(issues)}期")
                
                # 查找缺失的期号
                expected_issues = set(range(earliest_issue, latest_issue + 1))
                missing_issues = expected_issues - set(issues)
                
                if missing_issues:
                    print(f"缺失的期号: {', '.join(map(str, sorted(missing_issues)))}")
        
        except Exception as e:
            print(f"验证数据完整性失败: {e}")


def main():
    """主函数"""
    # 创建爬虫实例
    crawler = SSQAllCrawler()
    
    # 获取所有期数的历史数据
    results = crawler.get_all_history_data()
    
    # 保存数据（合并模式）
    if results:
        crawler.save_to_csv(results, filename="ssq_data_all_complete.csv", mode="merge")


if __name__ == "__main__":
    main() 