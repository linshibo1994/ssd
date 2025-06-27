# 双色球数据分析与预测系统 - 整合版

## 项目简介

这是一个基于Python开发的双色球数据分析与预测系统整合版，将所有功能集成到单个文件中，包含数据爬取、统计分析、机器学习预测和可视化功能。系统采用多种算法和分析方法，为双色球彩票提供智能分析和预测服务。

## 🎯 项目特点

- **单文件集成**：所有功能整合到 `ssq_analyzer.py` 一个文件中
- **功能完整**：保留原有所有分析和预测功能
- **代码优化**：去除重复代码，提高运行效率
- **易于部署**：只需一个Python文件即可运行

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
├── ssq_analyzer.py               # 主程序（整合所有功能）
├── .gitignore                    # Git忽略文件
└── data/                         # 数据存储目录
    ├── ssq_data.csv             # 最近300期数据
    ├── ssq_data_all.csv         # 所有历史数据
    └── advanced/                # 高级分析结果
        ├── enhanced_markov_chain_analysis.json
        ├── statistical_features.json
        ├── probability_distribution.json
        ├── frequency_patterns.json
        ├── decision_tree_analysis.json
        ├── red_ball_global_transition_heatmap.png
        └── blue_ball_transition_heatmap.png
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
python3 ssq_analyzer.py --help
```

## 快速开始

### 1. 获取数据
```bash
# 爬取最近300期数据
python3 ssq_analyzer.py crawl

# 爬取所有历史数据
python3 ssq_analyzer.py crawl --all
```

### 2. 基础分析
```bash
python3 ssq_analyzer.py analyze
```

### 3. 高级分析
```bash
# 运行所有高级分析
python3 ssq_analyzer.py advanced --method all

# 运行马尔可夫链分析
python3 ssq_analyzer.py advanced --method markov
```

### 4. 智能预测
```bash
# 使用改进的马尔可夫链预测
python3 ssq_analyzer.py markov_predict --count 5

# 使用集成方法预测
python3 ssq_analyzer.py predict --method ensemble --count 3
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
- **统计特性分析**：计算各种统计指标（和值、方差、跨度等）
- **概率分布分析**：分析号码概率分布和统计特征
- **频率模式分析**：识别冷热号码和频率规律
- **决策树分析**：使用随机森林机器学习模型预测
- **周期分析**：分析号码出现的周期性和自相关性
- **历史关联性分析**：分析不同期数间隔的号码重复规律
- **期号关联性分析**：分析期号与开奖号码的数学关联
- **马尔可夫链分析**：状态转移概率分析和可视化
- **贝叶斯分析**：贝叶斯推断和后验概率计算（可选）

### 智能预测
- **马尔可夫链预测**：基于全量历史数据的状态转移概率预测
- **统计学预测**：基于历史统计特征（和值、方差、跨度）的预测
- **概率论预测**：基于历史概率分布的加权随机预测
- **决策树预测**：使用随机森林机器学习模型的预测
- **模式识别预测**：基于奇偶比、大小比等模式的预测
- **集成方法预测**：多种预测方法投票综合的预测
- **贝叶斯预测**：基于贝叶斯后验概率的预测（可选）

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
python ssq_analyzer.py crawl

# 爬取所有历史数据
python ssq_analyzer.py crawl --all

# 爬取指定期数
python ssq_analyzer.py crawl --count 100

# 爬取指定期数范围并追加到文件
python3 ssq_analyzer.py crawl --start 2025070 --end 2025072 --append

# 获取最新一期开奖并追加到文件
python3 ssq_analyzer.py fetch_latest

# 追加最新N期数据到文件
python3 ssq_analyzer.py append --count 5

# 追加指定期数范围到文件
python3 ssq_analyzer.py append --start 2025070 --end 2025072
```

### 分析命令
```bash
# 基础分析
python ssq_analyzer.py analyze

# 高级分析 - 所有方法
python ssq_analyzer.py advanced --method all

# 高级分析 - 特定方法
python3 ssq_analyzer.py advanced --method stats
python3 ssq_analyzer.py advanced --method probability
python3 ssq_analyzer.py advanced --method frequency
python3 ssq_analyzer.py advanced --method decision_tree
python3 ssq_analyzer.py advanced --method cycle
python3 ssq_analyzer.py advanced --method correlation
python3 ssq_analyzer.py advanced --method issue_correlation
python3 ssq_analyzer.py advanced --method markov
python3 ssq_analyzer.py advanced --method bayes
```

### 预测命令
```bash
# 改进马尔可夫链预测（推荐）
python3 ssq_analyzer.py markov_predict --count 5

# 使用全量历史数据预测
python3 ssq_analyzer.py markov_predict --use-all-data --count 3

# 详细解释预测过程
python3 ssq_analyzer.py markov_predict --explain --count 2

# 指定期数分析预测（新功能）
python3 ssq_analyzer.py markov_predict --periods 100 --count 3 --explain

# 指定期数分析预测（使用最近50期数据）
python3 ssq_analyzer.py markov_predict --periods 50 --count 2 --explain

# 各种预测方法
python3 ssq_analyzer.py predict --method ensemble --count 3
python3 ssq_analyzer.py predict --method markov --count 5
python3 ssq_analyzer.py predict --method stats --count 3
python3 ssq_analyzer.py predict --method probability --count 3
python3 ssq_analyzer.py predict --method decision_tree --count 3
python3 ssq_analyzer.py predict --method patterns --count 3
```

### 其他命令
```bash
# 查看最新开奖结果
python ssq_analyzer.py latest

# 实时获取最新开奖结果
python ssq_analyzer.py latest --real-time

# 获取最新一期开奖并追加到文件（新功能）
python3 ssq_analyzer.py fetch_latest

# 生成号码
python ssq_analyzer.py generate --method hybrid --count 5

# 验证数据文件
python ssq_analyzer.py validate
```

## 改进马尔可夫链预测

### 核心改进
1. **全量历史数据分析**：使用所有历史数据计算转移概率
2. **多维度转移分析**：全局转移、位置转移、组合模式转移
3. **智能预测策略**：多策略融合，自动去重
4. **任意注数支持**：一次预测任意注数的号码
5. **指定期数分析**：可指定使用最近N期数据进行分析预测
6. **详细预测过程**：显示完整的预测推理过程

### 使用示例
```bash
# 基础预测
python ssq_analyzer.py markov_predict

# 预测多注
python ssq_analyzer.py markov_predict --count 10

# 使用全量数据
python ssq_analyzer.py markov_predict --use-all-data --count 5

# 详细解释
python ssq_analyzer.py markov_predict --explain --count 3

# 指定期数分析预测（新功能）
python3 ssq_analyzer.py markov_predict --periods 100 --count 3 --explain

# 指定期数分析预测（使用最近30期数据）
python3 ssq_analyzer.py markov_predict --periods 30 --count 1 --explain
```

### 分析结果
- 红球全局转移概率（所有红球间的转移关系）
- 红球位置转移概率（每个位置的转移概率）
- 蓝球转移概率（蓝球状态转移）
- 组合模式转移概率（奇偶比、大小比转移）
- 周期性分析（自相关性和显著周期）
- 历史关联性（不同期数间隔的重复规律）
- 期号关联性（期号与号码的数学关联）

## 编程接口

```python
from ssq_analyzer import SSQAnalyzer

# 创建分析器
analyzer = SSQAnalyzer("data")

# 加载数据
analyzer.load_data()

# 马尔可夫链分析
results = analyzer.analyze_markov_chain()

# 预测多注号码
predictions = analyzer.predict_multiple_by_markov_chain(count=5)

# 集成预测
red_balls, blue_ball = analyzer.predict_by_ensemble(explain=True)

# 生成智能号码
smart_numbers = analyzer.generate_smart_numbers("frequency")
```

## 输出文件

### 数据文件
- `data/ssq_data.csv`：最近300期数据
- `data/ssq_data_all.csv`：所有历史数据

### 分析结果文件
- `data/advanced/enhanced_markov_chain_analysis.json`：完整马尔可夫链分析
- `data/advanced/statistical_features.json`：统计特性分析结果
- `data/advanced/probability_distribution.json`：概率分布分析结果
- `data/advanced/frequency_patterns.json`：频率模式分析结果
- `data/advanced/decision_tree_analysis.json`：决策树分析结果
- `data/advanced/cycle_patterns.json`：周期分析结果
- `data/advanced/historical_correlation.json`：历史关联性分析结果
- `data/advanced/issue_number_correlation.json`：期号关联性分析结果

### 可视化图表
- `data/number_frequency.png`：号码频率图
- `data/number_combinations.png`：组合特征图
- `data/trend_analysis.png`：号码走势图
- `data/advanced/red_ball_global_transition_heatmap.png`：红球全局转移热力图
- `data/advanced/blue_ball_transition_heatmap.png`：蓝球转移热力图

## 注意事项

1. **数据要求**：建议使用至少300期以上的历史数据
2. **网络访问**：爬取数据时注意访问频率限制
3. **计算时间**：全量数据分析可能需要较长时间
4. **预测性质**：彩票具有随机性，预测结果仅供参考
5. **依赖管理**：确保所有依赖包版本兼容

## 许可证

MIT License

---

## 🆕 新增功能

### 数据管理功能
1. **指定期数爬取**：支持爬取指定期数范围的数据
2. **数据追加功能**：新数据自动按期号倒序追加到CSV文件
3. **最新开奖获取**：一键获取最新一期开奖结果并追加到文件
4. **智能数据更新**：避免重复数据，自动维护数据完整性

### 马尔可夫链预测增强
1. **指定期数分析**：可指定使用最近N期数据进行马尔可夫链分析
2. **详细预测过程**：显示完整的预测推理过程和概率计算
3. **多策略预测**：位置转移、全局转移、频率补充的组合策略
4. **组合特征验证**：预测结果的奇偶比、大小比、和值分析

### 使用示例
```bash
# 数据管理
python3 ssq_analyzer.py fetch_latest                    # 获取最新开奖
python3 ssq_analyzer.py append --count 10               # 追加最新10期
python3 ssq_analyzer.py crawl --start 2025070 --end 2025072 --append  # 追加指定期数

# 指定期数预测
python3 ssq_analyzer.py markov_predict --periods 50 --count 3 --explain  # 使用最近50期预测
python3 ssq_analyzer.py markov_predict --periods 100 --count 2 --explain # 使用最近100期预测
```

## 🚀 项目优化成果

### 整合前后对比
- **文件数量**：从 6 个 Python 文件整合为 1 个文件
- **代码行数**：优化重复代码，提高代码复用率
- **功能完整性**：保留所有原有功能，新增实用功能
- **运行效率**：去除重复导入和初始化，提升运行速度

### 主要改进
1. **代码整合**：将爬虫、分析、预测功能整合到单文件
2. **重复代码优化**：合并相似功能，减少代码冗余
3. **接口统一**：统一的类接口和方法调用
4. **错误处理**：改进异常处理和错误提示
5. **文档更新**：更新使用文档和示例

### 测试验证
✅ 数据爬取功能正常（官方网站+中彩网）
✅ 基础分析功能正常（频率、组合、走势）
✅ 高级分析功能正常（9种分析方法）
✅ 马尔可夫链预测正常（多维度转移概率）
✅ 多种预测方法正常（6种预测算法）
✅ 号码生成功能正常（4种生成策略）
✅ 命令行界面正常（完整CLI支持）
✅ 可视化输出正常（图表和热力图）
✅ 准确性分析正常（回测验证）
✅ 数据追加功能正常（增量更新）
✅ 最新开奖获取正常（实时更新）
✅ 指定期数分析正常（自定义数据范围）
✅ 详细预测过程正常（完整推理展示）

---

**免责声明**：本系统仅用于数据分析和算法研究，预测结果仅供参考，不构成任何投注建议。彩票具有随机性，请理性对待。
