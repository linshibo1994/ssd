#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双色球数据获取模块 - 中彩网API版本
通过中彩网的API获取最近300期双色球开奖结果
"""

import os
import csv
import time
import json
import random
import requests
from datetime import datetime


class SSQZhCWApiCrawler:
    """双色球数据获取类 - 中彩网API版本"""

    def __init__(self, data_dir="../data"):
        """
        初始化数据获取器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 中彩网双色球API URL
        self.api_url = "https://www.zhcw.com/kjxx/ssq/findDrawNotice"
        
        # 请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.zhcw.com/kjxx/ssq/",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://www.zhcw.com"
        }
    
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
            print(f"正在从中彩网API获取最近{count}期双色球开奖结果...")
            
            # 中彩网API可能有分页限制，分批获取数据
            batch_size = 30  # 每次请求获取的期数
            num_batches = (count + batch_size - 1) // batch_size  # 向上取整
            
            for batch in range(num_batches):
                # 计算当前批次的页码
                page = batch + 1
                
                print(f"获取第{page}批数据...")
                
                # 设置请求参数
                params = {
                    "pageNo": page,
                    "pageSize": batch_size,
                    "czId": "15982",  # 双色球彩种ID
                    "issueCount": "",
                    "issueStart": "",
                    "issueEnd": "",
                    "dayStart": "",
                    "dayEnd": "",
                    "kjDate": "",
                    "kjDateStart": "",
                    "kjDateEnd": "",
                    "_": int(time.time() * 1000)  # 时间戳，避免缓存
                }
                
                # 发送请求
                response = requests.get(self.api_url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                
                # 解析JSON数据
                data = response.json()
                
                # 打印响应内容，帮助调试
                print(f"API响应状态码: {response.status_code}")
                print(f"API响应内容: {response.text[:200]}...")  # 只打印前200个字符，避免输出过长
                
                # 检查是否有结果数据
                if data.get("result") == "success" and "data" in data:
                    # 提取开奖结果
                    for item in data["data"]:
                        try:
                            issue = item["qh"]  # 期号
                            date = item["kjsj"]  # 开奖日期
                            
                            # 获取红球号码
                            red_balls = []
                            for i in range(1, 7):
                                red_balls.append(item[f"hq{i}"].zfill(2))
                            
                            # 获取蓝球号码
                            blue_ball = item["lq"].zfill(2)
                            
                            results.append({
                                "issue": issue,
                                "date": date,
                                "red_balls": ",".join(red_balls),
                                "blue_ball": blue_ball
                            })
                        except Exception as e:
                            print(f"解析数据项失败: {e}")
                            continue
                    
                    print(f"成功获取第{page}批数据，当前共{len(results)}期")
                    
                    # 如果已经获取足够的数据，则退出循环
                    if len(results) >= count:
                        break
                    
                    # 添加随机延迟，避免请求过于频繁
                    time.sleep(random.uniform(1, 3))
                else:
                    error_msg = data.get("msg", "未知错误")
                    print(f"获取第{page}批数据失败: {error_msg}")
                    break
            
            print(f"从中彩网API成功获取{len(results)}期双色球开奖结果")
        except Exception as e:
            print(f"获取历史数据失败: {e}")
            import traceback
            traceback.print_exc()
        
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
    crawler = SSQZhCWApiCrawler()
    
    # 获取历史数据
    results = crawler.get_history_data(count=300)
    
    # 保存数据
    crawler.save_to_csv(results)


if __name__ == "__main__":
    main() 