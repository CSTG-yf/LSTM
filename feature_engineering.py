import pandas as pd
import numpy as np

def create_lag_features(df, target_col, lags):
    """
    创建滞后特征
    
    参数:
    df: DataFrame, 包含时间序列数据
    target_col: str, 目标变量列名
    lags: list, 滞后天数列表
    
    返回:
    DataFrame: 包含滞后特征的新DataFrame
    """
    df = df.copy()
    
    # 按照日期和线路排序
    df = df.sort_values(['route_id', 'date']).reset_index(drop=True)
    
    # 为每条线路分别创建滞后特征
    for lag in lags:
        lag_col_name = f"{target_col}_lag_{lag}"
        df[lag_col_name] = df.groupby('route_id')[target_col].shift(lag)
        
    return df

def create_rolling_features(df, target_col, windows, stats):
    """
    创建滑动窗口统计特征
    
    参数:
    df: DataFrame, 包含时间序列数据
    target_col: str, 目标变量列名
    windows: list, 窗口大小列表
    stats: list, 统计方法列表 ('mean', 'median', 'std')
    
    返回:
    DataFrame: 包含滑动窗口统计特征的新DataFrame
    """
    df = df.copy()
    
    # 按照日期和线路排序
    df = df.sort_values(['route_id', 'date']).reset_index(drop=True)
    
    # 为每条线路分别创建滑动窗口特征
    for window in windows:
        for stat in stats:
            rolling_col_name = f"{target_col}_rolling_{stat}_{window}"
            
            if stat == 'mean':
                df[rolling_col_name] = df.groupby('route_id')[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                ).round()
            elif stat == 'median':
                df[rolling_col_name] = df.groupby('route_id')[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).median()
                ).round()
            elif stat == 'std':
                df[rolling_col_name] = df.groupby('route_id')[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                ).round()
                
    return df

def engineer_time_series_features(input_file, output_file):
    """
    对时序数据进行特征工程处理
    
    参数:
    input_file: str, 输入文件路径
    output_file: str, 输出文件路径
    """
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 转换日期列为datetime类型
    df['date'] = pd.to_datetime(df['date'])
    
    # 按照日期和线路排序
    df = df.sort_values(['route_id', 'date']).reset_index(drop=True)
    
    print(f"原始数据形状: {df.shape}")
    print(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"线路数量: {df['route_id'].nunique()}")
    print(f"线路列表: {df['route_id'].unique()}")
    
    # 定义要使用的参数
    target_col = 'visitor_count'
    lags = [1, 7, 30]
    windows = [3, 7, 14]
    stats = ['mean', 'median', 'std']
    
    # 创建滞后特征
    print("\n正在创建滞后特征...")
    df = create_lag_features(df, target_col, lags)
    
    # 创建滑动窗口统计特征
    print("正在创建滑动窗口统计特征...")
    df = create_rolling_features(df, target_col, windows, stats)
    
    # 处理缺失值（用0填充滞后特征的缺失值）
    lag_columns = [col for col in df.columns if col.startswith(f"{target_col}_lag_")]
    for col in lag_columns:
        df[col] = df[col].fillna(0)
    
    # 处理滑动窗口统计特征的缺失值（用原始值填充）
    rolling_columns = [col for col in df.columns if col.startswith(f"{target_col}_rolling_")]
    for col in rolling_columns:
        df[col] = df[col].fillna(df[target_col])
    
    print(f"\n特征工程完成后数据形状: {df.shape}")
    print(f"新增特征列: {[col for col in df.columns if col not in ['date', 'route_id', 'visitor_count', 'weather_type', 'IsWorkingDay', 'order_count']]}")
    
    # 保存结果
    df.to_csv(output_file, index=False)
    print(f"\n特征工程完成，结果已保存至: {output_file}")
    
    return df

if __name__ == "__main__":
    # 执行特征工程
    engineered_df = engineer_time_series_features(
        input_file="final_wide_updated.csv",
        output_file="final_wide_engineered_rounded.csv"
    )
    
    # 显示部分结果
    print("\n特征工程结果示例:")
    print(engineered_df[['date', 'route_id', 'visitor_count', 'visitor_count_lag_1', 
                        'visitor_count_rolling_mean_3', 'visitor_count_rolling_std_7']].head(10))