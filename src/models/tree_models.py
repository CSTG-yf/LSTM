"""
树模型训练模块
实现 XGBoost、LightGBM 和 Ridge Regression
针对小样本数据优化，防止过拟合
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class TreeModelTrainer:
    """
    树模型训练器
    集成 XGBoost、LightGBM 和 Ridge Regression
    """
    
    def __init__(self, task='regression', use_log_transform=True):
        """
        初始化
        
        参数:
        task: 任务类型（'regression' 或 'classification'）
        use_log_transform: 是否对目标变量进行对数变换
        """
        self.task = task
        self.use_log_transform = use_log_transform
        self.models = {}
        self.best_params = {}
        
    def transform_target(self, y):
        """对目标变量进行对数变换"""
        if self.use_log_transform:
            return np.log1p(y)
        return y
    
    def inverse_transform_target(self, y_transformed):
        """对数变换的逆变换"""
        if self.use_log_transform:
            return np.expm1(y_transformed)
        return y_transformed
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """
        训练 XGBoost 模型
        
        参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据（可选）
        params: 模型参数（可选）
        
        返回:
        model: 训练好的模型
        """
        print("\n=== 训练 XGBoost ===")
        
        # 对数变换
        y_train_t = self.transform_target(y_train)
        y_val_t = self.transform_target(y_val) if y_val is not None else None
        
        # 默认参数（针对小样本优化）
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 4,  # 限制树深，防止过拟合
            'learning_rate': 0.05,  # 较小的学习率
            'n_estimators': 500,
            'subsample': 0.8,  # 行采样，增加随机性
            'colsample_bytree': 0.8,  # 列采样，增加随机性
            'reg_alpha': 1.0,  # L1正则化
            'reg_lambda': 2.0,  # L2正则化
            'min_child_weight': 3,  # 最小叶子节点权重
            'gamma': 0.1,  # 分裂最小损失
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if params:
            default_params.update(params)
        
        # 训练模型
        eval_set = [(X_train, y_train_t)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val_t))
        
        model = xgb.XGBRegressor(**default_params)
        model.fit(
            X_train, y_train_t,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        self.models['xgboost'] = model
        self.best_params['xgboost'] = default_params
        
        # 评估
        train_pred = self.inverse_transform_target(model.predict(X_train))
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        print(f"训练集 - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        
        if X_val is not None:
            val_pred = self.inverse_transform_target(model.predict(X_val))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            print(f"验证集 - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        
        print(f"最佳迭代轮数: {model.best_iteration}")
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """
        训练 LightGBM 模型
        
        参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据（可选）
        params: 模型参数（可选）
        
        返回:
        model: 训练好的模型
        """
        print("\n=== 训练 LightGBM ===")
        
        # 对数变换
        y_train_t = self.transform_target(y_train)
        y_val_t = self.transform_target(y_val) if y_val is not None else None
        
        # 默认参数（针对小样本优化）
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': 4,  # 限制树深
            'learning_rate': 0.05,
            'n_estimators': 500,
            'num_leaves': 15,  # 叶子节点数（2^max_depth - 1）
            'subsample': 0.8,  # 行采样
            'colsample_bytree': 0.8,  # 列采样
            'reg_alpha': 1.0,  # L1正则化
            'reg_lambda': 2.0,  # L2正则化
            'min_child_samples': 10,  # 最小叶子节点样本数
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        if params:
            default_params.update(params)
        
        # 训练模型
        eval_set = [(X_train, y_train_t)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val_t))
        
        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            X_train, y_train_t,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        self.models['lightgbm'] = model
        self.best_params['lightgbm'] = default_params
        
        # 评估
        train_pred = self.inverse_transform_target(model.predict(X_train))
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        print(f"训练集 - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        
        if X_val is not None:
            val_pred = self.inverse_transform_target(model.predict(X_val))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            print(f"验证集 - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        
        print(f"最佳迭代轮数: {model.best_iteration_}")
        
        return model
    
    def train_ridge(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """
        训练 Ridge Regression 模型
        
        参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据（可选）
        params: 模型参数（可选）
        
        返回:
        model: 训练好的模型
        """
        print("\n=== 训练 Ridge Regression ===")
        
        # 对数变换
        y_train_t = self.transform_target(y_train)
        
        # 默认参数
        default_params = {
            'alpha': 10.0,  # 正则化强度
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        # 训练模型
        model = Ridge(**default_params)
        model.fit(X_train, y_train_t)
        
        self.models['ridge'] = model
        self.best_params['ridge'] = default_params
        
        # 评估
        train_pred = self.inverse_transform_target(model.predict(X_train))
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        print(f"训练集 - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        
        if X_val is not None:
            val_pred = self.inverse_transform_target(model.predict(X_val))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            print(f"验证集 - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        
        return model
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练所有模型
        
        参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据（可选）
        
        返回:
        models: 所有训练好的模型字典
        """
        print("=" * 50)
        print("开始训练所有模型")
        print("=" * 50)
        
        # 训练 XGBoost
        self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # 训练 LightGBM
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # 训练 Ridge
        self.train_ridge(X_train, y_train, X_val, y_val)
        
        print("\n" + "=" * 50)
        print("所有模型训练完成")
        print("=" * 50)
        
        return self.models
    
    def predict(self, X, model_name='xgboost'):
        """
        使用指定模型进行预测
        
        参数:
        X: 特征矩阵
        model_name: 模型名称
        
        返回:
        predictions: 预测结果
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 尚未训练")
        
        model = self.models[model_name]
        y_pred_t = model.predict(X)
        y_pred = self.inverse_transform_target(y_pred_t)
        
        return y_pred
    
    def evaluate(self, X, y_true, model_name='xgboost'):
        """
        评估模型性能
        
        参数:
        X: 特征矩阵
        y_true: 真实值
        model_name: 模型名称
        
        返回:
        metrics: 评估指标字典
        """
        y_pred = self.predict(X, model_name)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }
        
        return metrics
    
    def plot_feature_importance(self, model_name='xgboost', feature_names=None, top_n=20):
        """
        绘制特征重要性
        
        参数:
        model_name: 模型名称
        feature_names: 特征名称列表
        top_n: 显示前N个重要特征
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 尚未训练")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # 创建DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            # 绘图
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'{model_name} - Top {top_n} 特征重要性')
            plt.xlabel('重要性')
            plt.ylabel('特征')
            plt.tight_layout()
            plt.savefig(f'outputs/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n特征重要性图已保存至: outputs/{model_name}_feature_importance.png")
            
            return importance_df
        else:
            print(f"模型 {model_name} 不支持特征重要性分析")
            return None


if __name__ == "__main__":
    # 简单测试
    print("树模型训练模块测试")
