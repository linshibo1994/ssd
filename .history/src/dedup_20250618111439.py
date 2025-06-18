#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv

# 读取原始CSV文件并解析
with open('/Users/linshibo/GithubProject/ssd/data/ssq_data_all.csv', 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    rows = list(reader)

# 打印原始行数
print(f'原始文件行数: {len(rows)}')

# 检查是否有重复的标题行
if rows[0] == rows[1]:
    print(f'发现重复的标题行: {rows[0]}')
    # 移除第二个标题行
    rows.pop(1)

# 创建一个新的行列表，每两行只保留一行
new_rows = []
for i in range(0, len(rows), 2):
    if i < len(rows):
        new_rows.append(rows[i])

# 写入去重后的CSV文件
with open('/Users/linshibo/GithubProject/ssd/data/ssq_data_all_unique.csv', 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(new_rows)

# 打印去重后行数
print(f'去重后行数: {len(new_rows)}')