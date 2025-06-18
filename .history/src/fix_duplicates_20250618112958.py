#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复CSV文件中的重复行问题
"""

import os
import csv
import subprocess

def run_command(cmd):
    """
    运行shell命令并返回输出
    
    Args:
        cmd: 要运行的命令
    
    Returns:
        命令的输出
    """
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        return None

def verify_file(file_path):
    """
    验证文件内容
    
    Args:
        file_path: 文件路径
    """
    print(f"\n验证文件: {file_path}")
    
    # 检查文件行数
    line_count = run_command(f"wc -l {file_path} | awk '{{print $1}}'")
    print(f"文件行数: {line_count}")
    
    # 查看文件前5行
    head_output = run_command(f"head -n 5 {file_path}")
    print(f"文件前5行:\n{head_output}")

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
    
    # 验证原始文件
    verify_file(file_path)
    
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
    
    # 检查是否有重复的表头
    has_duplicate_header = False
    if len(rows) >= 2 and rows[0] == rows[1]:
        has_duplicate_header = True
        print("检测到重复的表头行")
    
    # 检查是否有重复行模式
    has_duplicate_pattern = True
    for i in range(0, min(10, len(rows) - 1), 2):
        if i + 1 < len(rows) and rows[i] != rows[i + 1]:
            has_duplicate_pattern = False
            break
    
    if has_duplicate_pattern:
        print("检测到重复行模式：每行都重复一次")
        # 修复重复行模式
        fixed_rows = []
        for i in range(0, len(rows), 2):
            if i < len(rows):
                fixed_rows.append(rows[i])
    else:
        print("未检测到特定的重复行模式，使用常规去重方法")
        # 使用集合去重
        seen = set()
        fixed_rows = []
        for row in rows:
            row_tuple = tuple(row)  # 转换为元组，使其可哈希
            if row_tuple not in seen:
                seen.add(row_tuple)
                fixed_rows.append(row)
    
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
    
    # 验证修复后的文件
    verify_file(fixed_file_path)
    
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
        
        # 验证替换后的文件
        verify_file(original_path)
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