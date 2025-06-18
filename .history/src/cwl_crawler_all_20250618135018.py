#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 全期数版本
从中国福利彩票官方网站和中彩网获取所有期双色球开奖结果
确保爬取所有历史期数，不会因去重而遗漏任何期号
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

    def __init__(self, data_dir="data"):
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
        
        # 中彩网双色球历史数据URL模板
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
            "Referer": "http://kaijiang.zhcw.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive"
        }
    
    def get_all_data_from_zhcw(self):
        """
        从中彩网获取所有期双色球历史数据
        
        Returns:
            开奖结果字典，键为期号
        """
        results = {}  # 使用字典存储结果，以期号为键，避免重复
        
        try:
            print(f"正在从中彩网获取所有期双色球历史数据...")
            
            # 中彩网双色球历史数据页面数量（设置足够大的数字，实际会在没有数据时退出）
            max_pages = 150  # 预估页数，实际会在没有数据时退出
            
            for page in range(1, max_pages + 1):
                try:
                    print(f"正在获取第{page}页数据...")
                    
                    # 中彩网双色球历史数据URL
                    url = self.zhcw_url.format(page)
                    
                    # 发送请求
                    response = requests.get(url, headers=self.zhcw_headers, timeout=10)
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
                            
                            # 使用期号作为键保存结果，自动去重
                            results[issue] = {
                                "issue": issue,
                                "date": date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_ball
                            }
                            has_data = True
                        except Exception as e:
                            print(f"解析行数据失败: {e}")
                            continue
                    
                    # 如果当前页没有有效数据，可能已到达最后一页
                    if not has_data:
                        print(f"第{page}页没有有效数据，可能已到达最后一页")
                        break
                    
                    print(f"已获取 {len(results)} 期双色球数据")
                    
                    # 添加随机延迟，避免请求过于频繁
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    print(f"获取第{page}页数据失败: {e}")
                    continue
            
            print(f"从中彩网成功获取 {len(results)} 期双色球开奖结果")
        except Exception as e:
            print(f"从中彩网获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def get_data_from_cwl(self):
        """
        从中国福利彩票官方网站获取历史开奖数据
        
        Returns:
            开奖结果字典，键为期号
        """
        results = {}  # 使用字典存储结果，以期号为键，避免重复
        
        try:
            print("正在从中国福利彩票官方网站获取双色球开奖结果...")
            
            # 设置页面大小和初始页码
            page_size = 30
            page = 1
            
            # 设置一个较大的值，实际会在没有更多数据时退出
            max_pages = 100  # 假设最多100页
            
            # 使用分页方式获取数据
            while page <= max_pages:
                print(f"正在获取第{page}页数据 (每页{page_size}条)...")
                
                # 设置请求参数 - 使用分页方式
                params = {
                    "name": "ssq",  # 双色球
                    "pageNo": page,  # 页码
                    "pageSize": page_size,  # 每页数量
                    "systemType": "PC"  # 系统类型
                }
                
                # 发送请求
                try:
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
                                
                                # 使用期号作为键保存结果，自动去重
                                results[issue] = {
                                    "issue": issue,
                                    "date": date,
                                    "red_balls": ",".join(red_balls),
                                    "blue_ball": blue_ball
                                }
                            except Exception as e:
                                print(f"解析数据项失败: {e}")
                                continue
                        
                        print(f"已获取 {len(results)} 期双色球数据")
                        
                        # 如果当前页的数据量小于page_size，说明已经没有更多数据了
                        if len(data.get("result", [])) < page_size:
                            break
                        
                        # 添加随机延迟，避免请求过于频繁
                        time.sleep(random.uniform(1, 3))
                    else:
                        error_msg = data.get("message", "未知错误")
                        print(f"获取第{page}页数据失败: {error_msg}")
                        break
                except Exception as e:
                    print(f"请求第{page}页数据失败: {e}")
                    break
                
                # 增加页码，继续获取下一页
                page += 1
            
            print(f"从中国福利彩票官方网站成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"从中国福利彩票官方网站获取数据失败: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def get_all_history_data(self):
        """
        获取所有期双色球历史数据，综合官方网站和中彩网的数据
        
        Returns:
            所有期双色球开奖结果列表（已排序）
        """
        # 从官方网站获取数据
        cwl_results = self.get_data_from_cwl()
        print(f"从官方网站获取了 {len(cwl_results)} 期数据")
        
        # 从中彩网获取所有历史数据
        zhcw_results = self.get_all_data_from_zhcw()
        print(f"从中彩网获取了 {len(zhcw_results)} 期数据")
        
        # 合并数据（以官方网站数据为准，中彩网数据作为补充）
        all_results = {}
        
        # 先添加中彩网数据
        for issue, data in zhcw_results.items():
            all_results[issue] = data
        
        # 再添加官方网站数据（会覆盖同期号的中彩网数据，因为官方更权威）
        for issue, data in cwl_results.items():
            all_results[issue] = data
        
        print(f"合并后共有 {len(all_results)} 期双色球数据")
        
        # 将字典转换为列表，并按期号排序（降序）
        results_list = list(all_results.values())
        results_list.sort(key=lambda x: int(x["issue"]), reverse=True)
        
        return results_list
    
    def save_to_csv(self, results, filename="ssq_data_all.csv"):
        """
        将开奖结果保存到CSV文件
        
        Args:
            results: 开奖结果列表
            filename: 保存的文件名
        """
        if not results:
            print("没有数据可保存")
            return
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 构建完整的文件路径
        file_path = os.path.join(self.data_dir, filename)
        
        # 检查文件是否已存在
        file_exists = os.path.exists(file_path)
        
        # 如果文件已存在，读取已有数据，确保不重复添加
        existing_data = {}
        if file_exists:
            print(f"发现已有数据文件: {file_path}，将检查期号确保不重复")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data[row['issue']] = row
            except Exception as e:
                print(f"读取已有数据文件失败: {e}")
                existing_data = {}
        
        # 将新数据与已有数据合并（以新数据为准）
        all_data = {}
        
        # 先添加已有数据
        for issue, row in existing_data.items():
            all_data[issue] = row
        
        # 再添加新数据（会覆盖同期号的已有数据，确保使用最新爬取的信息）
        for row in results:
            all_data[row['issue']] = row
        
        # 转换为列表并排序
        sorted_data = list(all_data.values())
        sorted_data.sort(key=lambda x: int(x['issue']), reverse=True)
        
        # 保存所有数据到文件
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['issue', 'date', 'red_balls', 'blue_ball'])
            writer.writeheader()
            for row in sorted_data:
                writer.writerow(row)
        
        print(f"成功保存 {len(sorted_data)} 期双色球数据到文件: {file_path}")
        
        # 检查期号范围
        issues = [int(row['issue']) for row in sorted_data]
        min_issue = min(issues) if issues else 0
        max_issue = max(issues) if issues else 0
        
        # 验证期号连续性
        issue_set = set(issues)
        missing_issues = []
        for i in range(min_issue, max_issue + 1):
            if i not in issue_set:
                missing_issues.append(str(i))
        
        if missing_issues:
            print(f"警告：期号不连续，缺失的期号有 {len(missing_issues)} 个: {', '.join(missing_issues[:10])}{'...' if len(missing_issues) > 10 else ''}")
        else:
            print(f"期号连续性验证通过！所有期号从 {min_issue} 到 {max_issue} 都已获取")
        
        return file_path


def main():
    """主函数"""
    # 创建爬虫实例
    crawler = SSQAllCrawler()
    
    # 获取所有期数历史数据
    results = crawler.get_all_history_data()
    
    # 保存数据
    if results:
        crawler.save_to_csv(results, "ssq_data_all.csv")


if __name__ == "__main__":
    main() 