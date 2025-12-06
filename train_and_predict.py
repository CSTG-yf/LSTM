#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整演示脚本
一键执行：GPU 训练 + 线路预测
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))


def train_model(use_gpu=True):
    """
    训练模型
    """
    print("\n" + "="*80)
    print("开始训练模型".center(80))
    print("="*80)
    
    from src.train import main as train_main
    train_main(use_gpu=use_gpu)
    
    print("\n" + "="*80)
    print("✓ 训练完成！".center(80))
    print("="*80)


def predict_example():
    """
    预测示例
    """
    print("\n" + "="*80)
    print("预测示例".center(80))
    print("="*80)
    
    from src.query_predict import RoutePredictorQuery
    
    try:
        # 初始化
        print("\n初始化预测器...")
        predictor = RoutePredictorQuery()
        
        # 获取可用线路
        routes = predictor.list_routes()
        print(f"\n找到 {len(routes)} 条线路:")
        for i, route in enumerate(routes[:5], 1):
            print(f"  {i}. {route}")
        if len(routes) > 5:
            print(f"  ... 还有 {len(routes)-5} 条")
        
        # 查询第一条线路的最后几条数据
        if len(routes) > 0:
            sample_route = routes[0]
            print(f"\n查询线路 '{sample_route}' 的预测...")
            
            # 获取该线路的样本数据
            import pandas as pd
            df = predictor.df[predictor.df['route_id'] == sample_route].copy()
            
            if len(df) > 0:
                # 按日期排序
                df = df.sort_values('date')
                
                # 预测最后一条数据的下一天（如果存在）
                last_date = df['date'].max()
                next_date = last_date + pd.Timedelta(days=1)
                
                # 如果下一天有数据，进行查询
                sample_data = df[df['date'] == last_date]
                if len(sample_data) > 0:
                    print(f"\n线路: {sample_route}")
                    print(f"日期: {last_date.date()}")
                    
                    result = predictor.predict(sample_route, str(last_date.date()), return_details=True)
                    
                    if result and 'error' not in result:
                        print(f"预测人数: {result['prediction']:.0f}")
                        if 'actual' in result:
                            print(f"实际人数: {result['actual']:.0f}")
                            print(f"误差: {result['error']:.0f} ({result['error_pct']:.1f}%)")
                        
                        print(f"\n各模型预测:")
                        for model, pred in result['individual_predictions'].items():
                            print(f"  {model}: {pred:.0f}")
    
    except Exception as e:
        print(f"\n✗ 预测出错: {e}")
        print("\n提示: 如果是因为找不到模型文件，请先运行训练")


def interactive_mode():
    """
    交互式模式
    """
    from src.query_predict import interactive_query
    interactive_query()


def main():
    parser = argparse.ArgumentParser(
        description='完整演示：GPU 训练 + 线路预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（训练 + 预测）
  python train_and_predict.py

  # 只训练
  python train_and_predict.py --train-only

  # 只预测
  python train_and_predict.py --predict-only

  # 交互式查询
  python train_and_predict.py --interactive

  # 使用 CPU 训练
  python train_and_predict.py --no-gpu
        """
    )
    
    parser.add_argument('--train-only', action='store_true',
                       help='只执行训练')
    parser.add_argument('--predict-only', action='store_true',
                       help='只执行预测')
    parser.add_argument('--interactive', action='store_true',
                       help='进入交互式查询模式')
    parser.add_argument('--no-gpu', action='store_false', dest='use_gpu',
                       help='使用 CPU 训练')
    parser.set_defaults(use_gpu=True)
    
    args = parser.parse_args()
    
    # 显示欢迎信息
    print("\n" + "█"*80)
    print("█" + "小样本线路时序预测 - 完整演示".center(78) + "█")
    print("█"*80)
    
    # 执行相应操作
    if args.interactive:
        interactive_mode()
    elif args.predict_only:
        predict_example()
    else:
        # 训练 + 预测
        train_model(use_gpu=args.use_gpu)
        
        if not args.train_only:
            predict_example()
    
    print("\n" + "█"*80)
    print("█" + "演示完成！".center(78) + "█")
    print("█"*80 + "\n")


if __name__ == "__main__":
    main()
