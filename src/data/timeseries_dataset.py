"""
时间序列数据集处理模块
实现前向验证（Walk-Forward Validation）策略
严禁随机切分，防止数据泄露
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset:
    """
    时间序列数据集处理类
    专为小样本线路预测设计，按时间顺序切分数据
    """
    
    def __init__(self, route_col='route_id', date_col='date', target_col='visitor_count'):
        """
        初始化
        
        参数:
        route_col: 线路ID列名
        date_col: 日期列名
        target_col: 目标变量列名
        """
        self.route_col = route_col
        self.date_col = date_col
        self.target_col = target_col
        self.label_encoder = LabelEncoder()
        
    def time_series_split(self, df, train_ratio=0.67, val_ratio=0.17, test_ratio=0.16):
        """
        时间序列切分策略（前向验证）
        严禁随机切分，按时间顺序切分
        
        参数:
        df: DataFrame，必须已按日期排序
        train_ratio: 训练集比例（默认0.67，约400/600）
        val_ratio: 验证集比例（默认0.17，约100/600）
        test_ratio: 测试集比例（默认0.16，约100/600）
        
        返回:
        train_df, val_df, test_df: 三个数据集
        """
        # 确保数据按线路和日期排序
        df = df.sort_values([self.route_col, self.date_col]).reset_index(drop=True)
        
        # 对每条线路分别切分
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for route_id in df[self.route_col].unique():
            route_df = df[df[self.route_col] == route_id].copy()
            n = len(route_df)
            
            # 计算切分点
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            # 按时间顺序切分
            train_dfs.append(route_df.iloc[:train_end])
            val_dfs.append(route_df.iloc[train_end:val_end])
            test_dfs.append(route_df.iloc[val_end:])
            
            print(f"线路 {route_id}: 训练集={train_end}, 验证集={val_end-train_end}, 测试集={n-val_end}")
        
        # 合并所有线路的数据
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        print(f"\n总计: 训练集={len(train_df)}, 验证集={len(val_df)}, 测试集={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, df, fit=False):
        """
        准备特征
        
        参数:
        df: DataFrame
        fit: 是否拟合编码器（仅训练集时为True）
        
        返回:
        X, y: 特征矩阵和目标变量
        """
        df = df.copy()
        
        # 分离特征和目标
        if self.target_col in df.columns:
            y = df[self.target_col].values
        else:
            y = None
            
        # 移除不用于训练的列
        feature_cols = [col for col in df.columns 
                       if col not in [self.target_col, self.date_col]]
        
        X = df[feature_cols].copy()
        
        # 对分类特征进行编码
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col == self.route_col:
                if fit:
                    X[col] = self.label_encoder.fit_transform(X[col])
                else:
                    X[col] = self.label_encoder.transform(X[col])
            elif col == 'weather_type':
                # One-Hot编码天气类型
                weather_dummies = pd.get_dummies(X[col], prefix='weather')
                X = pd.concat([X.drop(col, axis=1), weather_dummies], axis=1)
        
        # 确保所有特征都是数值型
        X = X.select_dtypes(include=[np.number])
        
        # 填充缺失值
        X = X.fillna(0)
        
        return X, y
    
    def load_and_split_data(self, file_path, train_ratio=0.67, val_ratio=0.17, test_ratio=0.16):
        """
        加载数据并进行时序切分
        
        参数:
        file_path: 数据文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
        返回:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names
        """
        print(f"加载数据: {file_path}")
        df = pd.read_csv(file_path)
        
        # 转换日期列
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        print(f"数据形状: {df.shape}")
        print(f"时间范围: {df[self.date_col].min()} 到 {df[self.date_col].max()}")
        print(f"线路数量: {df[self.route_col].nunique()}")
        print(f"目标变量统计:\n{df[self.target_col].describe()}")
        
        # 时序切分
        print("\n=== 时序切分（前向验证） ===")
        train_df, val_df, test_df = self.time_series_split(df, train_ratio, val_ratio, test_ratio)
        
        # 准备特征
        print("\n=== 准备特征 ===")
        X_train, y_train = self.prepare_features(train_df, fit=True)
        X_val, y_val = self.prepare_features(val_df, fit=False)
        X_test, y_test = self.prepare_features(test_df, fit=False)
        
        print(f"训练集特征形状: {X_train.shape}")
        print(f"验证集特征形状: {X_val.shape}")
        print(f"测试集特征形状: {X_test.shape}")
        print(f"特征列表: {X_train.columns.tolist()}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), X_train.columns.tolist()


def load_timeseries_data(file_path, route_col='route_id', date_col='date', 
                        target_col='visitor_count', train_ratio=0.67, 
                        val_ratio=0.17, test_ratio=0.16):
    """
    便捷函数：加载时序数据
    
    参数:
    file_path: 数据文件路径
    route_col: 线路ID列名
    date_col: 日期列名
    target_col: 目标变量列名
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    test_ratio: 测试集比例
    
    返回:
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names
    """
    dataset = TimeSeriesDataset(route_col=route_col, date_col=date_col, target_col=target_col)
    return dataset.load_and_split_data(file_path, train_ratio, val_ratio, test_ratio)


if __name__ == "__main__":
    # 测试
    data_path = "f:/Pytorch/Dataset/visitordata.csv"
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = load_timeseries_data(
        data_path,
        route_col='route_id',
        date_col='date',
        target_col='visitor_count',
        train_ratio=0.67,
        val_ratio=0.17,
        test_ratio=0.16
    )
    
    print("\n=== 数据加载完成 ===")
    print(f"训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"验证集: X={X_val.shape}, y={y_val.shape}")
    print(f"测试集: X={X_test.shape}, y={y_test.shape}")
