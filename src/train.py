"""
主训练脚本
基于树模型（XGBoost + LightGBM + Ridge）的集成学习方案
针对小样本线路预测优化
"""

import os
import sys
import yaml
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data.timeseries_dataset import load_timeseries_data
from src.models.tree_models import TreeModelTrainer
from src.models.ensemble import create_ensemble


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def main():
    """主训练流程"""
    print("="*70)
    print("小样本线路时序预测 - 树模型集成方案")
    print("="*70)
    
    # 1. 加载配置
    config = load_config('configs/model_config.yaml')
    print("\n[1/6] 配置加载完成")
    
    # 2. 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 3. 加载并切分数据
    print("\n[2/6] 开始加载数据...")
    data_path = config['data']['data_path']
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = load_timeseries_data(
        data_path,
        route_col=config['data']['route_col'],
        date_col=config['data']['date_col'],
        target_col=config['data']['target_col'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio']
    )
    
    print(f"\n数据加载完成:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  特征数: {len(feature_names)}")
    
    # 4. 初始化训练器
    print("\n[3/6] 初始化模型训练器...")
    trainer = TreeModelTrainer(
        task='regression',
        use_log_transform=config['training']['use_log_transform']
    )
    
    # 5. 训练所有模型
    print("\n[4/6] 开始训练模型...")
    models = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # 6. 创建集成模型
    print("\n[5/6] 创建集成模型...")
    ensemble = create_ensemble(
        models,
        X_val, y_val,
        inverse_transform_func=trainer.inverse_transform_target,
        optimize=True
    )
    
    # 7. 在测试集上评估
    print("\n[6/6] 在测试集上评估...")
    print("\n" + "="*70)
    print("测试集评估结果")
    print("="*70)
    
    ensemble_metrics, individual_metrics = ensemble.evaluate(
        X_test, y_test,
        inverse_transform_func=trainer.inverse_transform_target
    )
    
    print("\n集成模型:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n各个模型:")
    for model_name, metrics in individual_metrics.items():
        print(f"\n  {model_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    # 8. 保存模型和结果
    print("\n" + "="*70)
    print("保存模型和结果")
    print("="*70)
    
    # 保存集成模型
    ensemble.save('outputs')
    
    # 保存训练器（包含逆变换函数）
    with open('outputs/trainer.pkl', 'wb') as f:
        pickle.dump(trainer, f)
    print("训练器已保存至: outputs/trainer.pkl")
    
    # 保存特征名称
    with open('outputs/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("特征名称已保存至: outputs/feature_names.pkl")
    
    # 保存评估结果
    results = {
        'ensemble_metrics': ensemble_metrics,
        'individual_metrics': individual_metrics,
        'ensemble_weights': ensemble.weights
    }
    with open('outputs/evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("评估结果已保存至: outputs/evaluation_results.pkl")
    
    # 9. 绘制特征重要性
    print("\n生成特征重要性图...")
    for model_name in ['xgboost', 'lightgbm']:
        trainer.plot_feature_importance(model_name, feature_names, top_n=20)
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"\n模型和结果已保存至: outputs/")
    print(f"集成权重: {ensemble.weights}")
    print(f"\n最终测试集 RMSE: {ensemble_metrics['RMSE']:.4f}")
    print(f"最终测试集 MAE: {ensemble_metrics['MAE']:.4f}")
    print(f"最终测试集 R²: {ensemble_metrics['R2']:.4f}")


if __name__ == '__main__':
    main()