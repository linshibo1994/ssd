# 双色球数据分析与预测系统

## 项目简介

本项目是一个全面的双色球数据分析与预测系统，能够自动从中国福利彩票官方网站和500彩票网获取双色球历史开奖数据，并提供多种数据分析、可视化和预测功能。系统支持基础统计分析、高级数据分析、智能预测和号码生成等功能，帮助用户深入了解双色球开奖规律和趋势。

## 功能特点

- **数据获取**：从中国福利彩票官方网站和500彩票网获取双色球历史开奖数据，支持获取所有历史数据或指定期数
- **数据验证**：严格的数据验证和清洗机制，确保数据的准确性和完整性
- **基础分析**：分析号码频率、组合特征、走势等基本统计特性
- **高级分析**：使用统计学、概率论、机器学习等方法进行深度分析
  - 统计特性分析
  - 概率分布分析
  - 频率模式分析
  - 决策树分析
  - 周期分析
  - 贝叶斯分析
  - 历史关联性分析
  - 期号关联性分析
- **智能预测**：基于多种算法的号码预测功能
- **🆕 新增预测方法**：整合全网最先进的5种预测算法
  - LSTM深度学习预测器 🧠
  - 集成学习预测器 🤖
  - 蒙特卡洛模拟预测器 🎲
  - K-Means聚类分析预测器 🔍
  - 超级预测器 🌟（智能融合所有方法）
- **号码生成**：提供随机、频率、走势和混合策略的号码生成方法
- **可视化**：丰富的数据可视化图表，直观展示分析结果

## 环境要求

- Python 3.8+
- 相关依赖包（详见requirements.txt）

## 安装步骤

1. 克隆或下载项目代码

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

项目提供了丰富的命令行接口，可以通过以下命令查看帮助信息：

```bash
python src/main.py
```

### 主要命令

#### 1. 爬取数据

```bash
# 爬取最近300期数据
python src/main.py crawl

# 爬取指定期数的数据
python src/main.py crawl --count 100

# 爬取所有历史数据
python src/main.py crawl --all
```

#### 2. 基础分析

```bash
python src/main.py analyze
```

#### 3. 高级分析

```bash
# 运行所有高级分析
python src/main.py advanced --method all

# 运行特定分析方法
python src/main.py advanced --method stats
python src/main.py advanced --method probability
python src/main.py advanced --method frequency
python src/main.py advanced --method decision_tree
python src/main.py advanced --method cycle
python src/main.py advanced --method bayes
python src/main.py advanced --method correlation
python src/main.py advanced --method issue_correlation
```

#### 4. 智能预测

```bash
# 使用集成方法预测
python src/main.py predict

# 使用特定方法预测
python src/main.py predict --method stats
python src/main.py predict --method probability
python src/main.py predict --method decision_tree
python src/main.py predict --method bayes
python src/main.py predict --method pattern

# 生成多注号码
python src/main.py predict --count 5

# 解释预测结果
python src/main.py predict --explain

# 与历史数据对比
python src/main.py predict --compare
```

#### 5. 🆕 新增预测方法

```bash
# 超级预测器（推荐）- 集成所有方法
python src/main.py super -m ensemble -n 5

# 快速预测
python src/main.py super -m quick -n 3

# 全方法对比分析
python src/main.py super -m compare -n 1

# LSTM深度学习预测
python src/main.py lstm -n 3

# 集成学习预测
python src/main.py ensemble -n 3

# 蒙特卡洛模拟预测
python src/main.py monte_carlo -n 3 -s 20000

# 聚类分析预测
python src/main.py clustering -n 3

# 训练所有模型
python src/main.py super --train -p 500
```

#### 6. 号码生成

```bash
# 使用混合策略生成号码
python src/main.py generate

# 使用特定方法生成号码
python src/main.py generate --method random
python src/main.py generate --method frequency
python src/main.py generate --method trend
```

#### 6. 查看最新开奖结果

```bash
python src/main.py latest
```

## 项目结构

```
ssd/
├── README.md                      # 项目说明文档
├── requirements.txt               # 项目依赖
├── test_predictors.py            # 🆕 预测器测试脚本
├── 使用文档.md                    # 详细使用文档
├── 大乐透预测开源项目分析报告.md    # 🆕 技术分析报告
├── 新增预测方法使用文档.md         # 🆕 新功能使用文档
├── data/                         # 数据存储目录
│   ├── ssq_data.csv             # 双色球数据文件
│   ├── models/                  # 🆕 训练模型存储
│   ├── advanced/                # 高级分析结果目录
│   ├── monte_carlo/             # 🆕 蒙特卡洛分析结果
│   ├── clustering/              # 🆕 聚类分析结果
│   └── super/                   # 🆕 超级预测器结果
└── src/                         # 源代码目录
    ├── main.py                  # 主程序
    ├── cwl_crawler.py           # 爬虫模块
    ├── analyzer.py              # 基础分析模块
    ├── advanced_analyzer.py     # 高级分析模块
    ├── lstm_predictor.py        # 🆕 LSTM深度学习预测器
    ├── ensemble_predictor.py    # 🆕 集成学习预测器
    ├── monte_carlo_predictor.py # 🆕 蒙特卡洛模拟预测器
    ├── clustering_predictor.py  # 🆕 聚类分析预测器
    ├── super_predictor.py       # 🆕 超级预测器
    └── utils.py                 # 工具函数
```

## 🧪 测试预测器

运行测试脚本验证所有预测器是否正常工作：

```bash
python test_predictors.py
```

## 🎯 推荐使用方式

### 日常预测（最推荐）
```bash
python src/main.py super -m ensemble -n 5
```

### 快速预测
```bash
python src/main.py super -m quick -n 3
```

### 深度分析
```bash
python src/main.py super -m compare -n 1
```

## 📖 详细文档

- [使用文档.md](使用文档.md) - 完整的使用说明和功能介绍
- [新增预测方法使用文档.md](新增预测方法使用文档.md) - 新增预测方法详细说明
- [大乐透预测开源项目分析报告.md](大乐透预测开源项目分析报告.md) - 技术分析和算法原理

## 📈 算法权重配置

在超级预测器中，各算法的默认权重为：

| 算法 | 权重 | 说明 |
|------|------|------|
| 混合分析 | 30% | 原有的7种数学模型综合 |
| LSTM深度学习 | 25% | 时间序列深度学习 |
| 集成学习 | 20% | XGBoost等多模型融合 |
| 蒙特卡洛模拟 | 15% | 概率分布随机采样 |
| 聚类分析 | 10% | 模式识别和聚类 |

## 注意事项

- 🆕 新增预测方法需要安装额外依赖（TensorFlow、XGBoost、LightGBM等）
- 贝叶斯分析功能需要安装PyMC和arviz包
- 爬取数据时可能会受到网站访问限制，建议适当控制爬取频率
- 高级分析功能计算量较大，可能需要较长时间
- 首次使用建议先运行测试脚本验证环境配置
- 分析结果仅供参考，彩票具有随机性，请理性购彩

## 免责声明

本项目仅用于技术研究和学习目的，不构成任何投资建议。彩票具有随机性，任何预测结果都不保证准确性。请理性购彩，适度娱乐。

## 许可证

MIT License