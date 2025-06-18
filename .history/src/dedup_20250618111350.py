#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv

# 读取原始CSV文件
seen = set()
unique_rows = []

with open('/Users/linshibo/GithubProject/ssd/data/ssq_data_all.csv', 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    for row in reader:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)

# 写入去重后的CSV文件
with open('/Users/linshibo/GithubProject/ssd/data/ssq_data_all_unique.csv', 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(unique_rows)

# 打印统计信息
print(f'原始文件行数: {sum(1 for _ in open("/Users/linshibo/GithubProject/ssd/data/ssq_data_all.csv", encoding="utf-8"))}')
print(f'去重后行数: {len(unique_rows)}')