#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查CSV文件内容
"""

import csv
import sys

def check_file(file_path):
    """
    检查CSV文件内容
    
    Args:
        file_path: CSV文件路径
    """
    print(f"检查文件: {file_path}")
    
    # 读取CSV文件
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    print(f"文件总行数: {len(rows)}")
    
    # 打印前5行内容
    print("前5行内容:")
    for i, row in enumerate(rows[:5]):
        print(f"{i}: {row}")
    
    # 检查是否有重复行
    unique_rows = set(tuple(row) for row in rows)
    print(f"唯一行数: {len(unique_rows)}")
    
    # 检查是否有重复的表头
    if len(rows) >= 2 and rows[0] == rows[1]:
        print("检测到重复的表头行")

def main():
    if len(sys.argv) < 2:
        print("用法: python check_file.py <文件路径>")
        return
    
    file_path = sys.argv[1]
    check_file(file_path)

if __name__ == "__main__":
    main()