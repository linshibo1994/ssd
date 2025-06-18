#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 读取原始CSV文件
with open('/Users/linshibo/GithubProject/ssd/data/ssq_data_all.csv', 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# 打印原始行数
print(f'原始文件行数: {len(lines)}')

# 检查是否有重复行
for i in range(0, len(lines), 2):
    if i+1 < len(lines) and lines[i] == lines[i+1]:
        print(f'发现重复行: {lines[i].strip()}')

# 去除重复行（每隔一行保留一行）
unique_lines = []
for i in range(0, len(lines), 2):
    if i < len(lines):
        unique_lines.append(lines[i])

# 写入去重后的CSV文件
with open('/Users/linshibo/GithubProject/ssd/data/ssq_data_all_unique.csv', 'w', encoding='utf-8') as outfile:
    outfile.writelines(unique_lines)

# 打印去重后行数
print(f'去重后行数: {len(unique_lines)}')