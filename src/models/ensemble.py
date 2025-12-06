"""
模型集成模块
实现加权平均集成策略
通过组合多个模型降低方差，提高泛化能力
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os


class EnsembleModel:
    """
    集成模型类
    支持多个模型的加权平均
    """
    
    def __init__(self, models_dict, weights=None):
        """
        初始化集成模型
        
        参数:
        models_dict: 模型字典 {model_name: model_object}
        weights: 权重字典 {model_name: weight}，默认为均等权重
        """
        self.models = models_dict
        self.model_names = list(models_dict.keys())
        
        if weights is None:
            # 默认均等权重
            n_models = len(self.models)
            self.weights = {name: 1.0/n_models for name in self.model_names}
        else:
            # 归一化权重
            total_weight = sum(weights.values())
            self.weights = {name: w/total_weight for name, w in weights.items()}
        
        print(f"集成模型初始化完成")
        print(f"包含模型: {self.model_names}")
        print(f"模型权重: {self.weights}")
    
    def predict(self, X, inverse_transform_func=None):
        """
        集成预测
        
        参数:
        X: 特征矩阵
        inverse_transform_func: 逆变换函数（如果使用了对数变换）
        
        返回:
        y_pred: 加权平均预测结果
        """
        predictions = {}
        
        # 获取每个模型的预测
        for name, model in self.models.items():
            pred = model.predict(X)
            
            # 如果需要逆变换
            if inverse_transform_func is not None:
                pred = inverse_transform_func(pred)
            
            predictions[name] = pred
        
        # 加权平均
        y_pred = np.zeros(len(X))
        for name in self.model_names:
            y_pred += self.weights[name] * predictions[name]
        
        return y_pred, predictions
    
    def optimize_weights(self, X_val, y_val, inverse_transform_func=None):
        """
        在验证集上优化权重
        
        参数:
        X_val: 验证集特征
        y_val: 验证集标签
        inverse_transform_func: 逆变换函数
        
        返回:
        best_weights: 最优权重
        """
        print("\n=== 优化集成权重 ===")
        
        # 获取每个模型在验证集上的预测
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_val)
            if inverse_transform_func is not None:
                pred = inverse_transform_func(pred)
            predictions[name] = pred
        
        # 计算每个模型的RMSE
        model_scores = {}
        for name, pred in predictions.items():
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            model_scores[name] = rmse
            print(f"{name} RMSE: {rmse:.4f}")
        
        # 使用RMSE的倒数作为权重（性能越好，权重越大）
        inverse_scores = {name: 1.0/score for name, score in model_scores.items()}
        total_inverse = sum(inverse_scores.values())
        
        optimized_weights = {name: inv/total_inverse for name, inv in inverse_scores.items()}
        
        print(f"\n优化后的权重: {optimized_weights}")
        
        # 更新权重
        self.weights = optimized_weights
        
        # 使用新权重进行预测并评估
        y_pred = np.zeros(len(X_val))
        for name in self.model_names:
            y_pred += self.weights[name] * predictions[name]
        
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        ensemble_mae = mean_absolute_error(y_val, y_pred)
        ensemble_r2 = r2_score(y_val, y_pred)
        
        print(f"\n集成模型性能:")
        print(f"RMSE: {ensemble_rmse:.4f}")
        print(f"MAE: {ensemble_mae:.4f}")
        print(f"R²: {ensemble_r2:.4f}")
        
        return optimized_weights
    
    def evaluate(self, X, y_true, inverse_transform_func=None):
        """
        评估集成模型
        
        参数:
        X: 特征矩阵
        y_true: 真实标签
        inverse_transform_func: 逆变换函数
        
        返回:
        metrics: 评估指标字典
        individual_metrics: 各个模型的评估指标
        """
        # 集成预测
        y_pred, predictions = self.predict(X, inverse_transform_func)
        
        # 集成模型指标
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }
        
        # 各个模型的指标
        individual_metrics = {}
        for name, pred in predictions.items():
            individual_metrics[name] = {
                'RMSE': np.sqrt(mean_squared_error(y_true, pred)),
                'MAE': mean_absolute_error(y_true, pred),
                'R2': r2_score(y_true, pred),
                'MAPE': np.mean(np.abs((y_true - pred) / (y_true + 1e-8))) * 100
            }
        
        return metrics, individual_metrics
    
    def save(self, save_dir='outputs'):
        """
        保存集成模型
        
        参数:
        save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存每个模型
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f'{name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"{name} 模型已保存至: {model_path}")
        
        # 保存权重
        weights_path = os.path.join(save_dir, 'ensemble_weights.pkl')
        with open(weights_path, 'wb') as f:
            pickle.dump(self.weights, f)
        print(f"集成权重已保存至: {weights_path}")
    
    @classmethod
    def load(cls, save_dir='outputs'):
        """
        加载集成模型
        
        参数:
        save_dir: 模型保存目录
        
        返回:
        ensemble_model: 加载的集成模型
        """
        # 加载权重
        weights_path = os.path.join(save_dir, 'ensemble_weights.pkl')
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
        
        # 加载模型
        models_dict = {}
        for name in weights.keys():
            model_path = os.path.join(save_dir, f'{name}_model.pkl')
            with open(model_path, 'rb') as f:
                models_dict[name] = pickle.load(f)
        
        print(f"集成模型加载完成")
        print(f"包含模型: {list(models_dict.keys())}")
        
        return cls(models_dict, weights)


def create_ensemble(models_dict, X_val=None, y_val=None, 
                   inverse_transform_func=None, optimize=True):
    """
    创建集成模型（便捷函数）
    
    参数:
    models_dict: 模型字典
    X_val: 验证集特征（用于优化权重）
    y_val: 验证集标签
    inverse_transform_func: 逆变换函数
    optimize: 是否优化权重
    
    返回:
    ensemble_model: 集成模型
    """
    ensemble = EnsembleModel(models_dict)
    
    if optimize and X_val is not None and y_val is not None:
        ensemble.optimize_weights(X_val, y_val, inverse_transform_func)
    
    return ensemble


if __name__ == "__main__":
    # 简单测试
    print("模型集成模块测试")
