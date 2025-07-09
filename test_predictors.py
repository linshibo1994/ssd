#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测器测试脚本
测试所有新增的预测器是否正常工作
"""

import os
import sys
import traceback

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_lstm_predictor():
    """测试LSTM预测器"""
    print("=" * 50)
    print("测试LSTM深度学习预测器")
    print("=" * 50)
    
    try:
        from lstm_predictor import SSQLSTMPredictor
        
        data_file = "data/ssq_data.csv"
        model_dir = "data/models"
        
        if not os.path.exists(data_file):
            print("❌ 数据文件不存在，跳过LSTM测试")
            return False
        
        predictor = SSQLSTMPredictor(data_file=data_file, model_dir=model_dir)
        
        # 测试数据加载
        if predictor.load_data():
            print("✅ 数据加载成功")
        else:
            print("❌ 数据加载失败")
            return False
        
        # 测试预测（不训练模型，使用随机预测）
        predictions = predictor.predict(num_predictions=1)
        if predictions:
            red_balls, blue_ball = predictions[0]
            print(f"✅ 预测成功: 红球 {red_balls}, 蓝球 {blue_ball}")
            return True
        else:
            print("❌ 预测失败")
            return False
            
    except ImportError as e:
        print(f"❌ LSTM预测器导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ LSTM预测器测试失败: {e}")
        traceback.print_exc()
        return False

def test_ensemble_predictor():
    """测试集成学习预测器"""
    print("=" * 50)
    print("测试集成学习预测器")
    print("=" * 50)
    
    try:
        from ensemble_predictor import SSQEnsemblePredictor
        
        data_file = "data/ssq_data.csv"
        model_dir = "data/models"
        
        if not os.path.exists(data_file):
            print("❌ 数据文件不存在，跳过集成学习测试")
            return False
        
        predictor = SSQEnsemblePredictor(data_file=data_file, model_dir=model_dir)
        
        # 测试数据加载
        if predictor.load_data():
            print("✅ 数据加载成功")
        else:
            print("❌ 数据加载失败")
            return False
        
        # 测试特征提取
        features_df = predictor.extract_features(periods=50)
        if features_df is not None and len(features_df) > 0:
            print(f"✅ 特征提取成功，共{len(features_df.columns)}个特征")
            return True
        else:
            print("❌ 特征提取失败")
            return False
            
    except ImportError as e:
        print(f"❌ 集成学习预测器导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 集成学习预测器测试失败: {e}")
        traceback.print_exc()
        return False

def test_monte_carlo_predictor():
    """测试蒙特卡洛预测器"""
    print("=" * 50)
    print("测试蒙特卡洛模拟预测器")
    print("=" * 50)
    
    try:
        from monte_carlo_predictor import SSQMonteCarloPredictor
        
        data_file = "data/ssq_data.csv"
        output_dir = "data/monte_carlo"
        
        if not os.path.exists(data_file):
            print("❌ 数据文件不存在，跳过蒙特卡洛测试")
            return False
        
        predictor = SSQMonteCarloPredictor(data_file=data_file, output_dir=output_dir)
        
        # 测试数据加载
        if predictor.load_data():
            print("✅ 数据加载成功")
        else:
            print("❌ 数据加载失败")
            return False
        
        # 测试概率分布计算
        predictor.calculate_probability_distribution(periods=50)
        if predictor.red_probs and predictor.blue_probs:
            print("✅ 概率分布计算成功")
        else:
            print("❌ 概率分布计算失败")
            return False
        
        # 测试预测（使用较少的模拟次数）
        predictions = predictor.predict(num_predictions=1, num_simulations=100)
        if predictions:
            red_balls, blue_ball, confidence = predictions[0]
            print(f"✅ 预测成功: 红球 {red_balls}, 蓝球 {blue_ball}, 置信度 {confidence:.2%}")
            return True
        else:
            print("❌ 预测失败")
            return False
            
    except ImportError as e:
        print(f"❌ 蒙特卡洛预测器导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 蒙特卡洛预测器测试失败: {e}")
        traceback.print_exc()
        return False

def test_clustering_predictor():
    """测试聚类分析预测器"""
    print("=" * 50)
    print("测试聚类分析预测器")
    print("=" * 50)
    
    try:
        from clustering_predictor import SSQClusteringPredictor
        
        data_file = "data/ssq_data.csv"
        output_dir = "data/clustering"
        
        if not os.path.exists(data_file):
            print("❌ 数据文件不存在，跳过聚类分析测试")
            return False
        
        predictor = SSQClusteringPredictor(data_file=data_file, output_dir=output_dir)
        
        # 测试数据加载
        if predictor.load_data():
            print("✅ 数据加载成功")
        else:
            print("❌ 数据加载失败")
            return False
        
        # 测试特征提取
        features_df = predictor.extract_clustering_features(periods=50)
        if features_df is not None and len(features_df) > 0:
            print(f"✅ 特征提取成功，共{len(features_df.columns)}个特征")
        else:
            print("❌ 特征提取失败")
            return False
        
        # 测试聚类（使用固定的聚类数）
        clustering_results = predictor.perform_clustering(features_df, k=3)
        if clustering_results is not None:
            print("✅ 聚类分析成功")
            return True
        else:
            print("❌ 聚类分析失败")
            return False
            
    except ImportError as e:
        print(f"❌ 聚类分析预测器导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 聚类分析预测器测试失败: {e}")
        traceback.print_exc()
        return False

def test_super_predictor():
    """测试超级预测器"""
    print("=" * 50)
    print("测试超级预测器")
    print("=" * 50)
    
    try:
        from super_predictor import SSQSuperPredictor
        
        data_file = "data/ssq_data.csv"
        model_dir = "data/models"
        output_dir = "data/super"
        
        if not os.path.exists(data_file):
            print("❌ 数据文件不存在，跳过超级预测器测试")
            return False
        
        predictor = SSQSuperPredictor(
            data_file=data_file, 
            model_dir=model_dir, 
            output_dir=output_dir
        )
        
        # 测试数据加载
        if predictor.load_data():
            print("✅ 数据加载成功")
        else:
            print("❌ 数据加载失败")
            return False
        
        print(f"✅ 初始化了{len(predictor.predictors)}个预测器")
        
        # 测试快速预测
        predictions = predictor.quick_predict(num_predictions=1)
        if predictions:
            red_balls, blue_ball = predictions[0]
            print(f"✅ 快速预测成功: 红球 {red_balls}, 蓝球 {blue_ball}")
            return True
        else:
            print("❌ 快速预测失败")
            return False
            
    except ImportError as e:
        print(f"❌ 超级预测器导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 超级预测器测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 开始测试所有预测器")
    print("=" * 80)
    
    # 检查数据文件
    data_file = "data/ssq_data.csv"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行爬虫程序获取数据:")
        print("python src/main.py crawl --count 100")
        return
    
    test_results = []
    
    # 测试各个预测器
    test_results.append(("LSTM预测器", test_lstm_predictor()))
    test_results.append(("集成学习预测器", test_ensemble_predictor()))
    test_results.append(("蒙特卡洛预测器", test_monte_carlo_predictor()))
    test_results.append(("聚类分析预测器", test_clustering_predictor()))
    test_results.append(("超级预测器", test_super_predictor()))
    
    # 输出测试结果
    print("\n" + "=" * 80)
    print("🧪 测试结果汇总")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print("=" * 80)
    print(f"测试完成: {passed}/{total} 个预测器通过测试")
    
    if passed == total:
        print("🎉 所有预测器测试通过！")
    else:
        print("⚠️  部分预测器测试失败，请检查依赖安装")
        print("\n安装所有依赖:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
