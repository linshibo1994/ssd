import csv
import os

def check_duplicates(file_path):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    issues = set()
    duplicates = []
    rows = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            issue = row['issue']
            if issue in issues:
                duplicates.append(issue)
            else:
                issues.add(issue)
    
    print(f"文件: {file_path}")
    print(f"总行数: {len(rows)}")
    print(f"唯一期号数: {len(issues)}")
    print(f"重复期号数: {len(duplicates)}")
    
    if duplicates:
        print(f"重复的期号: {', '.join(duplicates[:10])}{'...' if len(duplicates) > 10 else ''}")

if __name__ == "__main__":
    # 检查项目中的数据文件
    check_duplicates("data/ssq_data_all.csv")
    check_duplicates("data/ssq_data.csv") 