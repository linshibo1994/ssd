#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块
提供双色球数据处理和分析的通用功能
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime


def validate_ssq_data(data_file):
    """验证双色球数据文件的完整性

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
        required_columns = ["issue", "date", "red_balls", "blue_ball"]
        for col in required_columns:
            if col not in df.columns:
                print(f"数据文件缺少必要的列: {col}")
                return False
        
        # 检查数据行数
        if len(df) == 0:
            print("数据文件为空")
            return False
        
        # 检查红球格式
        for _, row in df.iterrows():
            red_balls = row["red_balls"].split(",")
            if len(red_balls) != 6:
                print(f"红球数量不正确: {row['issue']}期 {row['red_balls']}")
                return False
        
        print(f"数据验证成功，共{len(df)}条记录")
        return True
    except Exception as e:
        print(f"数据验证失败: {e}")
        return False


def generate_random_numbers():
    """生成随机双色球号码

    Returns:
        (红球列表, 蓝球)
    """
    # 生成6个不重复的红球号码（1-33）
    red_balls = sorted(random.sample(range(1, 34), 6))
    # 生成1个蓝球号码（1-16）
    blue_ball = random.randint(1, 16)
    
    return red_balls, blue_ball


def generate_smart_numbers(data_file, method="frequency"):
    """根据历史数据生成智能双色球号码

    Args:
        data_file: 数据文件路径
        method: 生成方法，可选值：
                - frequency: 基于频率
                - trend: 基于走势
                - hybrid: 混合策略

    Returns:
        (红球列表, 蓝球)
    """
    try:
        # 读取历史数据
        df = pd.read_csv(data_file)
        
        # 拆分红球
        red_balls_lists = []
        for _, row in df.iterrows():
            red_balls = [int(ball) for ball in row["red_balls"].split(",")]
            red_balls_lists.append(red_balls)
        
        # 蓝球列表
        blue_balls = df["blue_ball"].astype(int).tolist()
        
        if method == "frequency":
            # 基于频率生成号码
            # 统计红球频率
            red_counts = {}
            for i in range(1, 34):
                red_counts[i] = 0
            
            for red_list in red_balls_lists:
                for ball in red_list:
                    red_counts[ball] += 1
            
            # 按频率排序
            sorted_red = sorted(red_counts.items(), key=lambda x: x[1], reverse=True)
            
            # 从高频红球中随机选择4个，从低频红球中随机选择2个
            high_freq_reds = [ball for ball, _ in sorted_red[:15]]
            low_freq_reds = [ball for ball, _ in sorted_red[15:]]
            
            selected_reds = random.sample(high_freq_reds, 4) + random.sample(low_freq_reds, 2)
            selected_reds.sort()
            
            # 统计蓝球频率
            blue_counts = {}
            for i in range(1, 17):
                blue_counts[i] = 0
            
            for ball in blue_balls:
                blue_counts[ball] += 1
            
            # 按频率排序
            sorted_blue = sorted(blue_counts.items(), key=lambda x: x[1], reverse=True)
            
            # 从前5个高频蓝球中随机选择1个
            high_freq_blues = [ball for ball, _ in sorted_blue[:5]]
            selected_blue = random.choice(high_freq_blues)
            
        elif method == "trend":
            # 基于走势生成号码
            # 获取最近30期数据
            recent_reds = red_balls_lists[-30:]
            recent_blues = blue_balls[-30:]
            
            # 统计最近未出现的红球
            recent_red_flat = [ball for sublist in recent_reds for ball in sublist]
            missing_reds = [i for i in range(1, 34) if i not in recent_red_flat[-30:]]
            
            # 如果缺失的红球不足6个，从最近出现频率较低的红球中补充
            if len(missing_reds) < 6:
                red_freq = {}
                for i in range(1, 34):
                    red_freq[i] = recent_red_flat.count(i)
                
                sorted_red_freq = sorted(red_freq.items(), key=lambda x: x[1])
                low_freq_reds = [ball for ball, _ in sorted_red_freq if ball not in missing_reds]
                missing_reds.extend(low_freq_reds[:6-len(missing_reds)])
            
            # 从缺失的红球中随机选择6个
            selected_reds = sorted(random.sample(missing_reds, 6))
            
            # 统计最近未出现的蓝球
            missing_blues = [i for i in range(1, 17) if i not in recent_blues[-10:]]
            
            # 如果没有缺失的蓝球，从所有蓝球中随机选择
            if not missing_blues:
                selected_blue = random.randint(1, 16)
            else:
                selected_blue = random.choice(missing_blues)
            
        else:  # hybrid或其他方法
            # 混合策略
            # 统计红球频率
            red_counts = {}
            for i in range(1, 34):
                red_counts[i] = 0
            
            # 对最近50期的数据给予更高权重
            for i, red_list in enumerate(red_balls_lists):
                weight = 2 if i >= len(red_balls_lists) - 50 else 1
                for ball in red_list:
                    red_counts[ball] += weight
            
            # 计算红球冷热指数
            max_count = max(red_counts.values())
            red_heat = {ball: count / max_count for ball, count in red_counts.items()}
            
            # 结合随机因素选择红球
            selected_reds = []
            while len(selected_reds) < 6:
                for ball in range(1, 34):
                    if ball not in selected_reds and random.random() < red_heat[ball] * 0.3:
                        selected_reds.append(ball)
                        if len(selected_reds) >= 6:
                            break
            
            # 如果选择的红球不足6个，随机补充
            if len(selected_reds) < 6:
                remaining = [ball for ball in range(1, 34) if ball not in selected_reds]
                selected_reds.extend(random.sample(remaining, 6 - len(selected_reds)))
            
            selected_reds.sort()
            
            # 蓝球选择策略类似
            blue_counts = {}
            for i in range(1, 17):
                blue_counts[i] = 0
            
            for i, ball in enumerate(blue_balls):
                weight = 2 if i >= len(blue_balls) - 50 else 1
                blue_counts[ball] += weight
            
            max_blue_count = max(blue_counts.values())
            blue_heat = {ball: count / max_blue_count for ball, count in blue_counts.items()}
            
            # 结合随机因素选择蓝球
            selected_blue = 0
            for ball in range(1, 17):
                if random.random() < blue_heat[ball] * 0.5:
                    selected_blue = ball
                    break
            
            # 如果没有选中蓝球，随机选择一个
            if selected_blue == 0:
                selected_blue = random.randint(1, 16)
        
        return selected_reds, selected_blue
    except Exception as e:
        print(f"生成智能号码失败: {e}")
        # 出错时返回随机号码
        return generate_random_numbers()


def format_ssq_numbers(red_balls, blue_ball):
    """格式化双色球号码

    Args:
        red_balls: 红球列表
        blue_ball: 蓝球

    Returns:
        格式化后的字符串
    """
    red_str = " ".join([f"{ball:02d}" for ball in red_balls])
    blue_str = f"{blue_ball:02d}"
    
    return f"红球: {red_str} | 蓝球: {blue_str}"


def calculate_prize(my_reds, my_blue, winning_reds, winning_blue):
    """计算中奖等级

    Args:
        my_reds: 我的红球列表
        my_blue: 我的蓝球
        winning_reds: 中奖红球列表
        winning_blue: 中奖蓝球

    Returns:
        中奖等级（0表示未中奖）
    """
    # 计算红球匹配数
    red_matches = len(set(my_reds) & set(winning_reds))
    # 计算蓝球是否匹配
    blue_match = my_blue == winning_blue
    
    # 判断中奖等级
    if red_matches == 6 and blue_match:
        return 1  # 一等奖
    elif red_matches == 6:
        return 2  # 二等奖
    elif red_matches == 5 and blue_match:
        return 3  # 三等奖
    elif red_matches == 5 or (red_matches == 4 and blue_match):
        return 4  # 四等奖
    elif red_matches == 4 or (red_matches == 3 and blue_match):
        return 5  # 五等奖
    elif blue_match:
        return 6  # 六等奖
    else:
        return 0  # 未中奖


def get_latest_draw(data_file):
    """获取最新一期开奖结果

    Args:
        data_file: 数据文件路径

    Returns:
        (期号, 开奖日期, 红球列表, 蓝球)
    """
    try:
        # 读取数据文件
        df = pd.read_csv(data_file)
        
        # 获取最新一期数据
        latest = df.iloc[0]
        
        issue = latest["issue"]
        date = latest["date"]
        red_balls = [int(ball) for ball in latest["red_balls"].split(",")]
        blue_ball = int(latest["blue_ball"])
        
        return issue, date, red_balls, blue_ball
    except Exception as e:
        print(f"获取最新开奖结果失败: {e}")
        return None, None, None, None


if __name__ == "__main__":
    # 测试生成随机号码
    red_balls, blue_ball = generate_random_numbers()
    print("随机生成号码:")
    print(format_ssq_numbers(red_balls, blue_ball))
