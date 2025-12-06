#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速演示脚本
支持GPU训练和按线路+日期查询预测
"""

import os
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))


def train_with_gpu(use_gpu=True):
    """
    GPU训练演示
    """
    print("\n" + "="*70)
    print("开始GPU训练模型...")
    print("="*70)
    
    from src.train import main as train_main
    
    # 执行训练（支持GPU）
    train_main(use_gpu=use_gpu)
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print("\n检查 outputs/ 目录查看训练结果:")
    print("  - 模型文件: xgboost_model.pkl, lightgbm_model.pkl, ridge_model.pkl")
    print("  - 集成权重: ensemble_weights.pkl")
    print("  - 评估结果: evaluation_results.pkl")
    print("  - 特征重要性图: *_feature_importance.png")


def predict_route(route_id, date_str):
    """
    查询线路+日期的预测
    """
    print("\n" + "="*70)
    print(f"查询线路 '{route_id}' 在 '{date_str}' 的预测...")
    print("="*70)
    
    from src.query_predict import query_single
    
    result = query_single(route_id, date_str)
    
    if result and 'error' not in result:
        print(f"\n✓ 预测结果:")
        print(f"  线路: {result['route_id']}")
        print(f"  日期: {result['date']}")
        print(f"  预测人数: {result['prediction']:.0f}")
        
        if 'actual' in result:
            print(f"  实际人数: {result['actual']:.0f}")
            print(f"  误差: {result['error']:.0f} ({result['error_pct']:.1f}%)")
        
        print(f"\n各模型预测:")
        for model, pred in result['individual_predictions'].items():
            print(f"  {model}: {pred:.0f}")
        
        print(f"\n集成权重:")
        for model, weight in result['ensemble_weights'].items():
            print(f"  {model}: {weight:.1%}")


def interactive_query():
    """
    交互式查询界面
    """
    from src.query_predict import interactive_query as query_main
    query_main()


def main():
    parser = argparse.ArgumentParser(
        description='小样本线路时序预测 - GPU训练和查询系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # GPU训练
  python quick_demo.py --mode train --use_gpu

  # CPU训练
  python quick_demo.py --mode train --no_gpu

  # 单次查询
  python quick_demo.py --mode query --route kyoto_nara-A --date 2024-06-15

  # 交互式查询
  python quick_demo.py --mode query --interactive
        """
    )
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'query'],
                       help='运行模式: train(训练) 或 query(查询)')
    
    # 训练相关参数
    parser.add_argument('--use_gpu', action='store_true', dest='use_gpu',
                       help='使用GPU训练（默认）')
    parser.add_argument('--no_gpu', action='store_false', dest='use_gpu',
                       help='使用CPU训练')
    parser.set_defaults(use_gpu=True)
    
    # 查询相关参数
    parser.add_argument('--interactive', action='store_true',
                       help='交互式查询模式')
    parser.add_argument('--route', type=str,
                       help='线路ID（例: kyoto_nara-A）')
    parser.add_argument('--date', type=str,
                       help='日期（例: 2024-06-15）')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_with_gpu(use_gpu=args.use_gpu)
    
    elif args.mode == 'query':
        if args.interactive:
            interactive_query()
        elif args.route and args.date:
            predict_route(args.route, args.date)
        else:
            print("错误: 查询模式需要指定 --route 和 --date，或使用 --interactive")
            print("示例:")
            print("  python quick_demo.py --mode query --route kyoto_nara-A --date 2024-06-15")
            print("  python quick_demo.py --mode query --interactive")


if __name__ == "__main__":
    main()
