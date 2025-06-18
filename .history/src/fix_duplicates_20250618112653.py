#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复CSV文件中的重复行问题
"""

import os
import csv
from collections import OrderedDict

def fix_duplicates(file_path):
    """
    修复CSV文件中的重复行问题
    
    Args:
        file_path: CSV文件路径
    """
    print(f"正在处理文件: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    # 读取CSV文件
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    print(f"原始文件行数: {len(rows)}")
    
    # 去除重复行
    unique_rows = []
    seen = set()
    
    for i, row in enumerate(rows):
        # 将行转换为字符串用于去重
        row_str = ','.join(row)
        
        # 如果是表头或者未见过的行，则保留
        if i == 0 or row_str not in seen:
            unique_rows.append(row)
            seen.add(row_str)
    
    print(f"去重后行数: {len(unique_rows)}")
    
    # 保存去重后的数据
    try:
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(unique_rows)
        print(f"文件已成功更新: {file_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")

def main():
    # 数据目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # 处理所有数据文件
    files_to_fix = [
        os.path.join(data_dir, "ssq_data.csv"),
        os.path.join(data_dir, "ssq_data_all.csv")
    ]
    
    for file_path in files_to_fix:
        fix_duplicates(file_path)

if __name__ == "__main__":
    main()