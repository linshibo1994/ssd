#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复CSV文件中的重复行问题
"""

import os
import csv

def analyze_file(file_path):
    """
    分析CSV文件内容
    
    Args:
        file_path: CSV文件路径
    """
    print(f"正在分析文件: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
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
    duplicate_count = 0
    for i in range(len(rows) - 1):
        if rows[i] == rows[i + 1]:
            duplicate_count += 1
            print(f"发现重复行 {i} 和 {i+1}: {rows[i]}")
            if duplicate_count >= 5:  # 只显示前5个重复行
                print("...更多重复行...")
                break
    
    print(f"检测到 {duplicate_count} 对相邻重复行")
    
    # 修复重复行问题
    fixed_rows = []
    i = 0
    while i < len(rows):
        fixed_rows.append(rows[i])
        if i + 1 < len(rows) and rows[i] == rows[i + 1]:
            i += 2  # 跳过下一行（重复行）
        else:
            i += 1
    
    print(f"修复后行数: {len(fixed_rows)}")
    
    # 保存修复后的数据
    fixed_file_path = file_path.replace('.csv', '_fixed.csv')
    try:
        with open(fixed_file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(fixed_rows)
        print(f"修复后的数据已保存到: {fixed_file_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")
    
    return fixed_file_path

def replace_original_with_fixed(original_path, fixed_path):
    """
    用修复后的文件替换原始文件
    
    Args:
        original_path: 原始文件路径
        fixed_path: 修复后的文件路径
    """
    try:
        import shutil
        shutil.copy2(fixed_path, original_path)
        print(f"已用修复后的文件替换原始文件: {original_path}")
        os.remove(fixed_path)  # 删除临时文件
        print(f"已删除临时文件: {fixed_path}")
    except Exception as e:
        print(f"替换文件失败: {e}")

def main():
    # 数据目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # 处理所有数据文件
    files_to_fix = [
        os.path.join(data_dir, "ssq_data.csv"),
        os.path.join(data_dir, "ssq_data_all.csv")
    ]
    
    for file_path in files_to_fix:
        fixed_path = analyze_file(file_path)
        if fixed_path:
            replace_original_with_fixed(file_path, fixed_path)

if __name__ == "__main__":
    main()