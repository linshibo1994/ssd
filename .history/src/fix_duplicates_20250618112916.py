#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复CSV文件中的重复行问题
"""

import os
import csv

def analyze_file(file_path):
    """
    分析CSV文件内容并修复重复行问题
    
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
    
    # 打印前10行内容
    print("前10行内容:")
    for i, row in enumerate(rows[:10]):
        print(f"{i}: {row}")
    
    # 直接采用每隔一行取一行的方式修复文件
    # 这种方法适用于每行都重复一次的情况
    fixed_rows = []
    for i in range(0, len(rows), 2):
        if i < len(rows):
            fixed_rows.append(rows[i])
    
    print(f"修复后行数: {len(fixed_rows)}")
    
    # 保存修复后的数据
    fixed_file_path = file_path.replace('.csv', '_fixed.csv')
    try:
        with open(fixed_file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(fixed_rows)
        print(f"修复后的数据已保存到: {fixed_file_path}")
        
        # 打印修复后文件的前5行
        print("修复后文件的前5行:")
        with open(fixed_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                print(f"{i}: {row}")
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