# 双色球数据分析与预测系统

## 项目简介

这是一个基于Python开发的双色球数据分析与预测系统，集成了数据爬取、统计分析、机器学习预测和可视化功能。系统采用多种算法和分析方法，为双色球彩票提供智能分析和预测服务。

## 核心特性

- **多源数据爬取**：从中国福利彩票官方网站和中彩网获取历史开奖数据
- **基础统计分析**：号码频率、组合特征、走势分析
- **高级分析算法**：统计学、概率论、机器学习、贝叶斯分析、马尔可夫链
- **智能预测系统**：多种预测方法和集成算法
- **数据可视化**：丰富的图表展示分析结果
- **命令行界面**：完整的CLI工具，支持各种操作

## 技术架构

### 技术栈
- **编程语言**：Python 3.8+
- **数据处理**：pandas, numpy, scipy
- **机器学习**：scikit-learn
- **贝叶斯分析**：PyMC, arviz（可选）
- **数据可视化**：matplotlib, seaborn
- **网络分析**：networkx
- **网络爬虫**：requests, beautifulsoup4, lxml

### 项目结构
```
ssd/
├── README.md                     # 项目说明文档
├── requirements.txt              # 依赖包列表
├── src/                          # 源代码目录
│   ├── main.py                   # 主程序入口
│   ├── cwl_crawler.py            # 基础爬虫模块（300期数据）
│   ├── cwl_crawler_all.py        # 全量爬虫模块（所有历史数据）
│   ├── analyzer.py               # 基础分析模块
│   ├── advanced_analyzer.py      # 高级分析模块
│   └── utils.py                  # 工具函数模块
└── data/                         # 数据存储目录
    ├── ssq_data.csv             # 最近300期数据
    ├── ssq_data_all.csv         # 所有历史数据
    └── advanced/                # 高级分析结果
```

## 环境要求

- Python 3.8+
- 操作系统：Windows/macOS/Linux

## 安装步骤

1. **克隆项目代码**
```bash
git clone <项目地址>
cd ssd
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **验证安装**
```bash
python src/main.py
```

## 快速开始

### 1. 获取数据
```bash
# 爬取最近300期数据
python src/main.py crawl

# 爬取所有历史数据
python src/main.py crawl --all
```

### 2. 基础分析
```bash
python src/main.py analyze
```

### 3. 高级分析
```bash
# 运行所有高级分析
python src/main.py advanced --method all

# 运行马尔可夫链分析
python src/main.py advanced --method markov
```

### 4. 智能预测
```bash
# 使用改进的马尔可夫链预测
python src/main.py markov_predict --count 5

# 使用集成方法预测
python src/main.py predict --method ensemble --count 3
```

## 主要功能

### 数据爬取
- 从中国福利彩票官方网站获取数据
- 支持获取指定期数或全部历史数据
- 自动数据验证和去重

### 基础分析
- 号码频率分析
- 号码组合特征分析（奇偶比、大小比、和值）
- 号码走势分析

### 高级分析
- **统计特性分析**：计算各种统计指标
- **概率分布分析**：分析号码概率分布
- **频率模式分析**：识别冷热号码
- **决策树分析**：使用机器学习预测
- **周期分析**：分析号码出现周期
- **贝叶斯分析**：贝叶斯推断（可选）
- **历史关联性分析**：分析不同期数的关联
- **期号关联性分析**：分析期号与号码的关联
- **马尔可夫链分析**：状态转移概率分析

### 智能预测
- **改进马尔可夫链预测**：基于全量历史数据的转移概率
- **统计学预测**：基于统计特征
- **概率论预测**：基于概率分布
- **决策树预测**：机器学习模型
- **贝叶斯预测**：贝叶斯后验概率
- **集成方法预测**：多种方法综合

### 数据可视化
- 号码频率图
- 组合特征图
- 走势图
- 马尔可夫链热力图
- 网络图

## 命令行使用指南

### 数据爬取命令
```bash
# 爬取最近300期数据
python src/main.py crawl

# 爬取所有历史数据
python src/main.py crawl --all

# 爬取指定期数
python src/main.py crawl --count 100
```

### 分析命令
```bash
# 基础分析
python src/main.py analyze

# 高级分析 - 所有方法
python src/main.py advanced --method all

# 高级分析 - 特定方法
python src/main.py advanced --method stats
python src/main.py advanced --method probability
python src/main.py advanced --method decision_tree
python src/main.py advanced --method bayes
python src/main.py advanced --method markov
```

### 预测命令
```bash
# 改进马尔可夫链预测（推荐）
python src/main.py markov_predict --count 5

# 使用全量历史数据预测
python src/main.py markov_predict --use-all-data --count 3

# 详细解释预测过程
python src/main.py markov_predict --explain --count 2

# 与最新开奖结果比对
python src/main.py markov_predict --check-latest

# 准确性分析
python src/main.py markov_predict --analyze-accuracy

# 传统预测方法
python src/main.py predict --method ensemble --count 3
python src/main.py predict --method stats --explain
```

### 其他命令
```bash
# 查看最新开奖结果
python src/main.py latest

# 获取并保存最新开奖结果
python src/main.py fetch-latest

# 生成号码
python src/main.py generate --method hybrid
```

## 改进马尔可夫链预测

### 核心改进
1. **全量历史数据分析**：使用所有历史数据计算转移概率
2. **多维度转移分析**：全局转移、位置转移、组合模式转移
3. **智能预测策略**：多策略融合，自动去重
4. **任意注数支持**：一次预测任意注数的号码

### 使用示例
```bash
# 基础预测
python src/main.py markov_predict

# 预测多注
python src/main.py markov_predict --count 10

# 使用全量数据
python src/main.py markov_predict --use-all-data --count 5

# 详细解释
python src/main.py markov_predict --explain --count 3

# 准确性分析
python src/main.py markov_predict --analyze-accuracy
```

### 分析结果
- 红球全局转移概率
- 红球位置转移概率
- 蓝球转移概率
- 组合模式转移概率
- 转移概率演化分析

## 编程接口

```python
from src.advanced_analyzer import SSQAdvancedAnalyzer

# 创建分析器
analyzer = SSQAdvancedAnalyzer("data/ssq_data_all.csv")

# 加载数据
analyzer.load_data()

# 马尔可夫链分析
results = analyzer.analyze_markov_chain()

# 预测多注号码
predictions = analyzer.predict_multiple_by_markov_chain(count=5)

# 准确性分析
accuracy = analyzer.analyze_markov_prediction_accuracy(test_periods=50)
```

## 输出文件

### 数据文件
- `data/ssq_data.csv`：最近300期数据
- `data/ssq_data_all.csv`：所有历史数据

### 分析结果
- `data/advanced/enhanced_markov_chain_analysis.json`：完整马尔可夫链分析
- `data/advanced/markov_chain_analysis.json`：兼容格式分析结果

### 可视化图表
- `data/number_frequency.png`：号码频率图
- `data/number_combinations.png`：组合特征图
- `data/red_ball_trend.png`：红球走势图
- `data/blue_ball_trend.png`：蓝球走势图
- `data/advanced/red_ball_global_transition_heatmap.png`：红球全局转移热力图
- `data/advanced/blue_ball_enhanced_transition_heatmap.png`：蓝球转移热力图
- `data/advanced/combo_pattern_transitions.png`：组合模式转移图

## 注意事项

1. **数据要求**：建议使用至少300期以上的历史数据
2. **网络访问**：爬取数据时注意访问频率限制
3. **计算时间**：全量数据分析可能需要较长时间
4. **预测性质**：彩票具有随机性，预测结果仅供参考
5. **依赖管理**：确保所有依赖包版本兼容

## 许可证

MIT License

---

**免责声明**：本系统仅用于数据分析和算法研究，预测结果仅供参考，不构成任何投注建议。彩票具有随机性，请理性对待。
