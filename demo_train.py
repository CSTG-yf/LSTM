"""
快速演示脚本
测试整个训练流程
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """运行完整的训练流程"""
    print("=" * 80)
    print("小样本线路时序预测 - 树模型集成方案")
    print("快速演示")
    print("=" * 80)
    
    # 导入训练模块
    from src.train import main as train_main
    
    # 执行训练
    train_main()
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    print("\n检查 outputs/ 目录查看训练结果:")
    print("  - 模型文件: xgboost_model.pkl, lightgbm_model.pkl, ridge_model.pkl")
    print("  - 集成权重: ensemble_weights.pkl")
    print("  - 评估结果: evaluation_results.pkl")
    print("  - 特征重要性图: *_feature_importance.png")
    

if __name__ == "__main__":
    main()
