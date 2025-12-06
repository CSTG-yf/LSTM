"""
预测模块
使用训练好的树模型集成进行预测
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.models.ensemble import EnsembleModel


def load_trained_models(model_dir='outputs'):
    """
    加载训练好的模型
    
    参数:
    model_dir: 模型保存目录
    
    返回:
    ensemble: 集成模型
    trainer: 训练器（包含逆变换函数）
    feature_names: 特征名称列表
    """
    print(f"从 {model_dir} 加载模型...")
    
    # 加载集成模型
    ensemble = EnsembleModel.load(model_dir)
    
    # 加载训练器
    with open(os.path.join(model_dir, 'trainer.pkl'), 'rb') as f:
        trainer = pickle.load(f)
    
    # 加载特征名称
    with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"模型加载完成！")
    print(f"集成权重: {ensemble.weights}")
    print(f"特征数量: {len(feature_names)}")
    
    return ensemble, trainer, feature_names


def predict_single_file(data_path, model_dir='outputs', output_path='outputs/predictions.csv'):
    """
    对单个文件进行预测
    
    参数:
    data_path: 数据文件路径
    model_dir: 模型保存目录
    output_path: 预测结果保存路径
    
    返回:
    predictions_df: 预测结果DataFrame
    """
    print("="*70)
    print("开始预测")
    print("="*70)
    
    # 加载模型
    ensemble, trainer, feature_names = load_trained_models(model_dir)
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 保存原始数据的关键列
    original_cols = ['date', 'route_id']
    if 'visitor_count' in df.columns:
        original_cols.append('visitor_count')
        has_ground_truth = True
    else:
        has_ground_truth = False
    
    original_data = df[original_cols].copy()
    
    # 准备特征
    print("\n准备特征...")
    from src.data.timeseries_dataset import TimeSeriesDataset
    
    dataset = TimeSeriesDataset(
        route_col='route_id',
        date_col='date',
        target_col='visitor_count'
    )
    
    X, y = dataset.prepare_features(df, fit=False)
    
    # 确保特征顺序一致
    X = X[feature_names]
    
    print(f"特征形状: {X.shape}")
    
    # 预测
    print("\n开始预测...")
    y_pred, individual_predictions = ensemble.predict(
        X,
        inverse_transform_func=trainer.inverse_transform_target
    )
    
    # 整理预测结果
    predictions_df = original_data.copy()
    predictions_df['prediction'] = y_pred
    
    # 添加各个模型的预测
    for model_name, pred in individual_predictions.items():
        predictions_df[f'pred_{model_name}'] = pred
    
    # 如果有真实值，计算误差
    if has_ground_truth:
        predictions_df['error'] = predictions_df['visitor_count'] - predictions_df['prediction']
        predictions_df['abs_error'] = np.abs(predictions_df['error'])
        predictions_df['pct_error'] = (predictions_df['error'] / (predictions_df['visitor_count'] + 1e-8)) * 100
        
        print(f"\n预测性能:")
        print(f"  MAE: {predictions_df['abs_error'].mean():.4f}")
        print(f"  RMSE: {np.sqrt((predictions_df['error']**2).mean()):.4f}")
        print(f"  MAPE: {np.abs(predictions_df['pct_error']).mean():.2f}%")
    
    # 保存结果
    predictions_df.to_csv(output_path, index=False)
    print(f"\n预测结果已保存至: {output_path}")
    
    print("\n预测样例:")
    print(predictions_df.head(10))
    
    return predictions_df


def predict_new_data(X_new, feature_names=None, model_dir='outputs'):
    """
    对新数据进行预测（已处理好的特征矩阵）
    
    参数:
    X_new: 新数据特征矩阵（DataFrame或numpy array）
    feature_names: 特征名称列表
    model_dir: 模型保存目录
    
    返回:
    predictions: 预测结果
    """
    # 加载模型
    ensemble, trainer, saved_feature_names = load_trained_models(model_dir)
    
    # 如果没有提供特征名称，使用保存的
    if feature_names is None:
        feature_names = saved_feature_names
    
    # 确保特征顺序一致
    if isinstance(X_new, pd.DataFrame):
        X_new = X_new[feature_names]
    
    # 预测
    y_pred, individual_predictions = ensemble.predict(
        X_new,
        inverse_transform_func=trainer.inverse_transform_target
    )
    
    return y_pred, individual_predictions


def main():
    """主预测流程（示例）"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--data', type=str, default='f:/Pytorch/Dataset/visitordata.csv',
                       help='输入数据文件路径')
    parser.add_argument('--model_dir', type=str, default='outputs',
                       help='模型保存目录')
    parser.add_argument('--output', type=str, default='outputs/predictions.csv',
                       help='预测结果保存路径')
    
    args = parser.parse_args()
    
    # 执行预测
    predictions_df = predict_single_file(
        data_path=args.data,
        model_dir=args.model_dir,
        output_path=args.output
    )
    
<<<<<<< HEAD
    print("\n预测完成！")

=======
    # Get the last sequence from the data for prediction
    seq_length = config['training']['sequence_length']
    last_sequence = torch.FloatTensor(data[-seq_length:])
    
    # Predict next 7 days
    print('Predicting next 7 days...')
    predictions = predict_next_days(model, last_sequence, scaler, device, num_days=7)
    
    # Print predictions
    last_date = pd.to_datetime(dates[-1])
    print("\nPredictions for the next 7 days:")
    for i, pred in enumerate(predictions):
        date = last_date + timedelta(days=i+1)
        # Round to nearest integer for visitor count
        print(f"{date.strftime('%Y-%m-%d')}: {round(pred)} visitors")
    
    # Save predictions to file
    pred_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(predictions))]
    pred_df = pd.DataFrame({
        'date': pred_dates,
        'predicted_visitors': [round(pred) for pred in predictions]  # Round to nearest integer
    })
    pred_df.to_csv('outputs/predictions.csv', index=False)
    print('\nPredictions saved to outputs/predictions.csv')
>>>>>>> 5605b211501a696d6b392e87a0862d98db18b2d5

if __name__ == '__main__':
    main()
