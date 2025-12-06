"""
自定义查询预测模块
支持按线路名称和日期进行当天人数预测
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.models.ensemble import EnsembleModel


class RoutePredictorQuery:
    """
    线路预测查询器
    支持按线路名称和日期查询当天的访客数预测
    """
    
    def __init__(self, model_dir='outputs', data_path='f:/Pytorch/Dataset/visitordata.csv'):
        """
        初始化预测器
        
        参数:
        model_dir: 模型保存目录
        data_path: 原始数据路径（用于获取特征）
        """
        print(f"初始化线路预测查询器...")
        self.model_dir = model_dir
        self.data_path = data_path
        
        # 加载模型和训练器
        self.ensemble = EnsembleModel.load(model_dir)
        
        with open(os.path.join(model_dir, 'trainer.pkl'), 'rb') as f:
            self.trainer = pickle.load(f)
        
        with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # 加载原始数据
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 获取可用的线路列表
        self.available_routes = self.df['route_id'].unique().tolist()
        
        print(f"✓ 预测器初始化完成！")
        print(f"  可用线路: {len(self.available_routes)} 条")
        print(f"  特征数: {len(self.feature_names)}")
    
    def list_routes(self):
        """
        列出所有可用的线路
        
        返回:
        routes: 线路列表
        """
        return self.available_routes
    
    def get_route_date_data(self, route_id, date_str):
        """
        获取指定线路和日期的特征数据
        
        参数:
        route_id: 线路ID（例如: 'kyoto_nara-A'）
        date_str: 日期字符串（例如: '2024-01-01' 或 '2024/1/1'）
        
        返回:
        X: 特征矩阵
        found: 是否找到该数据
        """
        # 转换日期格式
        try:
            date = pd.to_datetime(date_str)
        except:
            print(f"错误: 无法识别日期格式 '{date_str}'")
            print(f"请使用格式: 'YYYY-MM-DD' 或 'YYYY/M/D'")
            return None, False
        
        # 检查线路是否存在
        if route_id not in self.available_routes:
            print(f"错误: 线路 '{route_id}' 不存在")
            print(f"可用线路: {self.available_routes}")
            return None, False
        
        # 查找指定的线路和日期
        mask = (self.df['route_id'] == route_id) & (self.df['date'] == date)
        data = self.df[mask]
        
        if len(data) == 0:
            print(f"错误: 未找到线路 '{route_id}' 在日期 '{date_str}' 的数据")
            
            # 显示该线路有数据的日期范围
            route_data = self.df[self.df['route_id'] == route_id]
            if len(route_data) > 0:
                min_date = route_data['date'].min()
                max_date = route_data['date'].max()
                print(f"  该线路数据范围: {min_date.date()} 至 {max_date.date()}")
            
            return None, False
        
        # 准备特征
        X, _ = self._prepare_features(data)
        
        return X, True
    
    def _prepare_features(self, data):
        """
        准备特征（内部方法）
        """
        from src.data.timeseries_dataset import TimeSeriesDataset
        
        dataset = TimeSeriesDataset(
            route_col='route_id',
            date_col='date',
            target_col='visitor_count'
        )
        
        X, y = dataset.prepare_features(data, fit=False)
        
        # 确保特征顺序和名称一致
        if hasattr(self, 'feature_names') and len(self.feature_names) > 0:
            available_features = [f for f in self.feature_names if f in X.columns]
            X = X[available_features]
        
        return X, y
    
    def predict(self, route_id, date_str, return_details=False):
        """
        预测指定线路和日期的访客数
        
        参数:
        route_id: 线路ID
        date_str: 日期字符串
        return_details: 是否返回详细信息
        
        返回:
        prediction: 预测的访客数（如果return_details=True，则返回字典）
        """
        # 获取特征数据
        X, found = self.get_route_date_data(route_id, date_str)
        
        if not found:
            return None if not return_details else {'error': '未找到数据'}
        
        # 进行预测
        y_pred, individual_pred = self.ensemble.predict(
            X,
            inverse_transform_func=self.trainer.inverse_transform_target
        )
        
        # 获取真实值（如果存在）
        date = pd.to_datetime(date_str)
        mask = (self.df['route_id'] == route_id) & (self.df['date'] == date)
        actual_data = self.df[mask]
        
        if return_details:
            result = {
                'route_id': route_id,
                'date': str(date.date()),
                'prediction': float(y_pred[0]),
                'individual_predictions': {
                    name: float(pred[0]) for name, pred in individual_pred.items()
                },
                'ensemble_weights': self.ensemble.weights
            }
            
            # 添加真实值（如果有）
            if len(actual_data) > 0:
                actual_value = actual_data['visitor_count'].values[0]
                result['actual'] = float(actual_value)
                result['error'] = float(actual_value - y_pred[0])
                result['error_pct'] = float((result['error'] / (actual_value + 1e-8)) * 100)
            
            return result
        else:
            return float(y_pred[0])
    
    def predict_range(self, route_id, start_date, end_date):
        """
        预测指定线路在日期范围内的访客数
        
        参数:
        route_id: 线路ID
        start_date: 开始日期字符串
        end_date: 结束日期字符串
        
        返回:
        results_df: 结果DataFrame
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        except:
            print(f"错误: 无法识别日期格式")
            return None
        
        # 检查线路
        if route_id not in self.available_routes:
            print(f"错误: 线路 '{route_id}' 不存在")
            return None
        
        # 获取日期范围内的所有数据
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        results = []
        for date in date_range:
            try:
                result = self.predict(route_id, str(date.date()), return_details=True)
                if result and 'error' not in result:
                    results.append(result)
            except:
                continue
        
        if len(results) == 0:
            print(f"在指定日期范围内未找到数据")
            return None
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def batch_predict(self, routes, date_str):
        """
        批量预测多条线路在同一日期的访客数
        
        参数:
        routes: 线路ID列表或 'all' 表示所有线路
        date_str: 日期字符串
        
        返回:
        results_df: 结果DataFrame
        """
        if routes == 'all':
            routes = self.available_routes
        
        results = []
        for route in routes:
            try:
                pred = self.predict(route, date_str, return_details=True)
                if pred and 'error' not in pred:
                    results.append(pred)
            except:
                continue
        
        if len(results) == 0:
            print(f"未能预测任何线路")
            return None
        
        results_df = pd.DataFrame(results)
        
        return results_df


def interactive_query():
    """
    交互式查询界面
    """
    print("\n" + "="*70)
    print("线路访客数预测查询系统")
    print("="*70)
    
    # 初始化预测器
    predictor = RoutePredictorQuery()
    
    print("\n可用命令:")
    print("  1. query <线路ID> <日期>  - 查询单条线路的预测（例: query kyoto_nara-A 2024-06-15）")
    print("  2. list - 列出所有可用线路")
    print("  3. range <线路ID> <开始日期> <结束日期> - 查询日期范围的预测")
    print("  4. batch <日期> [线路1] [线路2] ... - 批量查询多条线路（不指定则查询所有）")
    print("  5. help - 显示帮助")
    print("  6. exit - 退出")
    
    while True:
        print("\n" + "-"*70)
        cmd = input("\n请输入命令 > ").strip()
        
        if not cmd:
            continue
        
        parts = cmd.split()
        command = parts[0].lower()
        
        if command == 'exit' or command == 'quit':
            print("再见！")
            break
        
        elif command == 'help':
            print("\n可用命令:")
            print("  1. query <线路ID> <日期>")
            print("     示例: query kyoto_nara-A 2024-06-15")
            print("           query mt_fuji-D 2024/6/15")
            print()
            print("  2. list - 列出所有可用线路")
            print()
            print("  3. range <线路ID> <开始日期> <结束日期>")
            print("     示例: range kyoto_nara-A 2024-06-01 2024-06-30")
            print()
            print("  4. batch <日期> [线路1] [线路2] ...")
            print("     示例: batch 2024-06-15")
            print("           batch 2024-06-15 kyoto_nara-A mt_fuji-D")
            print()
            print("  5. exit - 退出程序")
        
        elif command == 'list':
            routes = predictor.list_routes()
            print(f"\n共有 {len(routes)} 条线路:")
            for i, route in enumerate(routes, 1):
                print(f"  {i}. {route}")
        
        elif command == 'query' and len(parts) >= 3:
            route_id = parts[1]
            date_str = parts[2]
            
            result = predictor.predict(route_id, date_str, return_details=True)
            
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
        
        elif command == 'range' and len(parts) >= 4:
            route_id = parts[1]
            start_date = parts[2]
            end_date = parts[3]
            
            results = predictor.predict_range(route_id, start_date, end_date)
            
            if results is not None:
                print(f"\n✓ 预测结果 ({len(results)} 天):")
                print(results[['date', 'prediction', 'actual', 'error']] if 'actual' in results.columns else results[['date', 'prediction']])
        
        elif command == 'batch' and len(parts) >= 2:
            date_str = parts[1]
            routes = parts[2:] if len(parts) > 2 else ['all']
            
            results = predictor.batch_predict(routes if routes != ['all'] else 'all', date_str)
            
            if results is not None:
                print(f"\n✓ 批量预测结果 ({len(results)} 条线路):")
                print(results[['route_id', 'prediction', 'actual', 'error']] if 'actual' in results.columns else results[['route_id', 'prediction']])
        
        else:
            print("❌ 无效命令，请输入 'help' 查看帮助")


def query_single(route_id, date_str):
    """
    快速单次查询函数
    
    参数:
    route_id: 线路ID
    date_str: 日期字符串
    
    返回:
    prediction: 预测的访客数
    """
    predictor = RoutePredictorQuery()
    return predictor.predict(route_id, date_str, return_details=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='线路访客数预测查询系统')
    parser.add_argument('--mode', type=str, default='interactive', 
                       choices=['interactive', 'query'],
                       help='运行模式: interactive(交互) 或 query(单次查询)')
    parser.add_argument('--route', type=str, help='线路ID（query模式必需）')
    parser.add_argument('--date', type=str, help='日期（query模式必需）')
    parser.add_argument('--model_dir', type=str, default='outputs',
                       help='模型保存目录')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_query()
    elif args.mode == 'query':
        if not args.route or not args.date:
            print("错误: query模式需要指定 --route 和 --date")
            print("示例: python src/query_predict.py --mode query --route kyoto_nara-A --date 2024-06-15")
        else:
            result = query_single(args.route, args.date)
            if result:
                print(f"\n预测结果: {result['prediction']:.0f} 人")
