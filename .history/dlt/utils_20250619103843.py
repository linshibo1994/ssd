#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块
提供大乐透数据处理和分析的通用功能
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime


def validate_dlt_data(data_file):
    """验证大乐透数据文件的完整性

    Args:
        data_file: 数据文件路径

    Returns:
        验证结果，成功返回True，失败返回False
    """
    try:
        if not os.path.exists(data_file):
            print(f"数据文件不存在: {data_file}")
            return False
        
        # 读取数据文件
        df = pd.read_csv(data_file)
        
        # 检查必要的列是否存在
        required_columns = ["issue", "date", "front_balls", "back_balls"]
        for col in required_columns:
            if col not in df.columns:
                print(f"数据文件缺少必要的列: {col}")
                return False
        
        # 检查数据行数
        if len(df) == 0:
            print("数据文件为空")
            return False
        
        # 检查前区号码格式
        for _, row in df.iterrows():
            front_balls = row["front_balls"].split(",")
            if len(front_balls) != 5:
                print(f"前区号码数量不正确: {row['issue']}期 {row['front_balls']}")
                return False
            
            # 检查后区号码格式
            back_balls = row["back_balls"].split(",")
            if len(back_balls) != 2:
                print(f"后区号码数量不正确: {row['issue']}期 {row['back_balls']}")
                return False
        
        print(f"数据验证成功，共{len(df)}条记录")
        return True
    except Exception as e:
        print(f"数据验证失败: {e}")
        return False


def generate_random_numbers():
    """生成随机大乐透号码

    Returns:
        (前区号码列表, 后区号码列表)
    """
    # 生成5个不重复的前区号码（1-35）
    front_balls = sorted(random.sample(range(1, 36), 5))
    # 生成2个不重复的后区号码（1-12）
    back_balls = sorted(random.sample(range(1, 13), 2))
    
    return front_balls, back_balls


def generate_smart_numbers(data_file, method="frequency"):
    """根据历史数据生成智能大乐透号码

    Args:
        data_file: 数据文件路径
        method: 生成方法，可选值：
                - frequency: 基于频率
                - trend: 基于走势
                - hybrid: 混合策略

    Returns:
        (前区号码列表, 后区号码列表)
    """
    try:
        # 读取历史数据
        df = pd.read_csv(data_file)
        
        # 拆分前区号码
        front_balls_lists = []
        for _, row in df.iterrows():
            front_balls = [int(ball) for ball in row["front_balls"].split(",")]
            front_balls_lists.append(front_balls)
        
        # 拆分后区号码
        back_balls_lists = []
        for _, row in df.iterrows():
            back_balls = [int(ball) for ball in row["back_balls"].split(",")]
            back_balls_lists.append(back_balls)
        
        if method == "frequency":
            # 基于频率生成号码
            # 统计前区号码频率
            front_counts = {}
            for i in range(1, 36):
                front_counts[i] = 0
            
            for front_list in front_balls_lists:
                for ball in front_list:
                    front_counts[ball] += 1
            
            # 按频率排序
            sorted_front = sorted(front_counts.items(), key=lambda x: x[1], reverse=True)
            
            # 从高频前区号码中随机选择3个，从低频前区号码中随机选择2个
            high_freq_fronts = [ball for ball, _ in sorted_front[:15]]
            low_freq_fronts = [ball for ball, _ in sorted_front[15:]]
            
            selected_fronts = random.sample(high_freq_fronts, 3) + random.sample(low_freq_fronts, 2)
            selected_fronts.sort()
            
            # 统计后区号码频率
            back_counts = {}
            for i in range(1, 13):
                back_counts[i] = 0
            
            for back_list in back_balls_lists:
                for ball in back_list:
                    back_counts[ball] += 1
            
            # 按频率排序
            sorted_back = sorted(back_counts.items(), key=lambda x: x[1], reverse=True)
            
            # 从前4个高频后区号码中随机选择1个，从其余号码中随机选择1个
            high_freq_backs = [ball for ball, _ in sorted_back[:4]]
            low_freq_backs = [ball for ball, _ in sorted_back[4:]]
            
            selected_backs = [random.choice(high_freq_backs), random.choice(low_freq_backs)]
            selected_backs.sort()
            
        elif method == "trend":
            # 基于走势生成号码
            # 获取最近30期数据
            recent_fronts = front_balls_lists[:30]
            recent_backs = back_balls_lists[:30]
            
            # 统计最近未出现的前区号码
            recent_front_flat = [ball for sublist in recent_fronts for ball in sublist]
            missing_fronts = [i for i in range(1, 36) if i not in recent_front_flat[-30:]]
            
            # 如果缺失的前区号码不足5个，从最近出现频率较低的前区号码中补充
            if len(missing_fronts) < 5:
                front_freq = {}
                for i in range(1, 36):
                    front_freq[i] = recent_front_flat.count(i)
                
                sorted_front_freq = sorted(front_freq.items(), key=lambda x: x[1])
                low_freq_fronts = [ball for ball, _ in sorted_front_freq if ball not in missing_fronts]
                missing_fronts.extend(low_freq_fronts[:5-len(missing_fronts)])
            
            # 从缺失的前区号码中随机选择5个
            selected_fronts = sorted(random.sample(missing_fronts, 5))
            
            # 统计最近未出现的后区号码
            recent_back_flat = [ball for sublist in recent_backs for ball in sublist]
            missing_backs = [i for i in range(1, 13) if i not in recent_back_flat[-15:]]
            
            # 如果缺失的后区号码不足2个，从所有后区号码中随机选择
            if len(missing_backs) < 2:
                selected_backs = sorted(random.sample(range(1, 13), 2))
            else:
                selected_backs = sorted(random.sample(missing_backs, 2))
            
        else:  # hybrid或其他方法
            # 混合策略
            # 统计前区号码频率
            front_counts = {}
            for i in range(1, 36):
                front_counts[i] = 0
            
            # 对最近50期的数据给予更高权重
            for i, front_list in enumerate(front_balls_lists):
                weight = 2 if i < 50 else 1
                for ball in front_list:
                    front_counts[ball] += weight
            
            # 计算前区号码冷热指数
            max_count = max(front_counts.values())
            front_heat = {ball: count / max_count for ball, count in front_counts.items()}
            
            # 结合随机因素选择前区号码
            selected_fronts = []
            while len(selected_fronts) < 5:
                for ball in range(1, 36):
                    if ball not in selected_fronts and random.random() < front_heat[ball] * 0.3:
                        selected_fronts.append(ball)
                        if len(selected_fronts) >= 5:
                            break
            
            # 如果选择的前区号码不足5个，随机补充
            if len(selected_fronts) < 5:
                remaining = [ball for ball in range(1, 36) if ball not in selected_fronts]
                selected_fronts.extend(random.sample(remaining, 5 - len(selected_fronts)))
            
            selected_fronts.sort()
            
            # 后区号码选择策略类似
            back_counts = {}
            for i in range(1, 13):
                back_counts[i] = 0
            
            for i, back_list in enumerate(back_balls_lists):
                weight = 2 if i < 50 else 1
                for ball in back_list:
                    back_counts[ball] += weight
            
            max_back_count = max(back_counts.values())
            back_heat = {ball: count / max_back_count for ball, count in back_counts.items()}
            
            # 结合随机因素选择后区号码
            selected_backs = []
            while len(selected_backs) < 2:
                for ball in range(1, 13):
                    if ball not in selected_backs and random.random() < back_heat[ball] * 0.4:
                        selected_backs.append(ball)
                        if len(selected_backs) >= 2:
                            break
            
            # 如果选择的后区号码不足2个，随机补充
            if len(selected_backs) < 2:
                remaining = [ball for ball in range(1, 13) if ball not in selected_backs]
                selected_backs.extend(random.sample(remaining, 2 - len(selected_backs)))
            
            selected_backs.sort()
        
        return selected_fronts, selected_backs
    except Exception as e:
        print(f"生成智能号码失败: {e}")
        # 出错时返回随机号码
        return generate_random_numbers()


def format_dlt_numbers(front_balls, back_balls):
    """格式化大乐透号码

    Args:
        front_balls: 前区号码列表
        back_balls: 后区号码列表

    Returns:
        格式化后的字符串
    """
    front_str = " ".join([f"{ball:02d}" for ball in front_balls])
    back_str = " ".join([f"{ball:02d}" for ball in back_balls])
    
    return f"前区: {front_str} | 后区: {back_str}"


def calculate_prize(my_fronts, my_backs, winning_fronts, winning_backs):
    """计算中奖等级

    Args:
        my_fronts: 我的前区号码列表
        my_backs: 我的后区号码列表
        winning_fronts: 中奖前区号码列表
        winning_backs: 中奖后区号码列表

    Returns:
        中奖等级（0表示未中奖）
    """
    # 计算前区匹配数
    front_matches = len(set(my_fronts) & set(winning_fronts))
    # 计算后区匹配数
    back_matches = len(set(my_backs) & set(winning_backs))
    
    # 判断中奖等级
    if front_matches == 5 and back_matches == 2:
        return 1  # 一等奖
    elif front_matches == 5 and back_matches == 1:
        return 2  # 二等奖
    elif front_matches == 5 and back_matches == 0:
        return 3  # 三等奖
    elif front_matches == 4 and back_matches == 2:
        return 4  # 四等奖
    elif (front_matches == 4 and back_matches == 1) or (front_matches == 3 and back_matches == 2):
        return 5  # 五等奖
    elif (front_matches == 4 and back_matches == 0) or (front_matches == 3 and back_matches == 1) or (front_matches == 2 and back_matches == 2):
        return 6  # 六等奖
    elif (front_matches == 3 and back_matches == 0) or (front_matches == 2 and back_matches == 1) or (front_matches == 1 and back_matches == 2) or (front_matches == 0 and back_matches == 2):
        return 7  # 七等奖
    elif (front_matches == 2 and back_matches == 0) or (front_matches == 1 and back_matches == 1) or (front_matches == 0 and back_matches == 1):
        return 8  # 八等奖
    else:
        return 0  # 未中奖


def get_latest_draw(data_file, real_time=False):
    """获取最新一期开奖结果

    Args:
        data_file: 数据文件路径
        real_time: 是否实时从网络获取最新数据，默认为False

    Returns:
        (期号, 开奖日期, 前区号码列表, 后区号码列表)
    """
    if real_time:
        try:
            import requests
            print("正在实时获取最新一期大乐透开奖结果...")
            
            # 中彩网API
            api_url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
            
            # 请求头
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": "https://www.cwl.gov.cn/kjxx/dlt/kjgg/",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Connection": "keep-alive",
                "X-Requested-With": "XMLHttpRequest",
                "Origin": "https://www.cwl.gov.cn"
            }
            
            # 设置请求参数 - 只获取最新一期
            params = {
                "name": "dlt",  # 大乐透
                "pageNo": 1,     # 第一页
                "pageSize": 1,   # 只获取一条
                "systemType": "PC"  # 系统类型
            }
            
            # 发送请求
            response = requests.get(api_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            # 解析JSON数据
            data = response.json()
            
            # 检查是否有结果数据
            if "result" in data and isinstance(data["result"], list) and len(data["result"]) > 0:
                # 提取开奖结果
                item = data["result"][0]  # 获取最新一期
                
                issue = item["code"]  # 期号
                date = item["date"]  # 开奖日期
                
                # 获取前区号码（格式为 "01,02,03,04,05"）
                front_str = item["front"]
                front_balls = [int(ball) for ball in front_str.split(",")]
                
                # 获取后区号码（格式为 "01,02"）
                back_str = item["back"]
                back_balls = [int(ball) for ball in back_str.split(",")]
                
                print(f"成功获取最新一期({issue})开奖结果")
                return issue, date, front_balls, back_balls
            else:
                print("未获取到最新开奖结果，将使用本地数据")
        except Exception as e:
            print(f"实时获取最新开奖结果失败: {e}")
            print("将使用本地数据作为备选")
    
    # 如果实时获取失败或不需要实时获取，则从本地文件读取
    try:
        # 读取数据文件
        df = pd.read_csv(data_file)
        
        # 获取最新一期数据
        latest = df.iloc[0]
        
        issue = latest["issue"]
        date = latest["date"]
        front_balls = [int(ball) for ball in latest["front_balls"].split(",")]
        back_balls = [int(ball) for ball in latest["back_balls"].split(",")]
        
        return issue, date, front_balls, back_balls
    except Exception as e:
        print(f"获取最新开奖结果失败: {e}")
        return None, None, None, None


if __name__ == "__main__":
    # 测试生成随机号码
    front_balls, back_balls = generate_random_numbers()
    print("随机生成号码:")
    print(format_dlt_numbers(front_balls, back_balls))