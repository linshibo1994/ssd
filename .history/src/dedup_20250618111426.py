#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv

# 读取原始CSV文件
with open('/Users/linshibo/GithubProject/ssd/data/ssq_data_all.csv', 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# 打印原始行数
print(f'原始文件行数: {len(lines)}')

# 使用字典去重，保持顺序
seen = {}
for line in lines:
    if line not in seen:
        seen[line] = True

unique_lines = list(seen.keys())

# 写入去重后的CSV文件
with open('/Users/linshibo/GithubProject/ssd/data/ssq_data_all_unique.csv', 'w', encoding='utf-8') as outfile:
    outfile.writelines(unique_lines)

# 打印去重后行数
print(f'去重后行数: {len(unique_lines)}')