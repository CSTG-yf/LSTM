# GPU 训练和线路预测查询系统

## 📌 快速开始

### 1. GPU 训练

**使用 GPU 进行训练（推荐）：**
```bash
python quick_demo.py --mode train --use_gpu
```

**使用 CPU 进行训练：**
```bash
python quick_demo.py --mode train --no_gpu
```

### 2. 查询预测

**单次查询（指定线路和日期）：**
```bash
python quick_demo.py --mode query --route kyoto_nara-A --date 2024-06-15
```

**交互式查询（推荐）：**
```bash
python quick_demo.py --mode query --interactive
```

---

## 🎯 详细使用指南

### GPU 训练模式

#### 什么是 GPU 训练？
GPU 训练利用图形处理器的并行计算能力，加速树模型的训练过程。对于中等数据量的任务，GPU 训练可以比 CPU 快 5-10 倍。

#### 系统要求
- **NVIDIA GPU**：支持 CUDA Compute Capability 3.5 或更高
- **CUDA 11.0+** 和 **cuDNN 8.0+**
- **PyTorch** 和 **XGBoost/LightGBM** 的 GPU 支持

#### 检查 GPU 是否可用
```python
import torch
print(torch.cuda.is_available())  # True 表示 GPU 可用
print(torch.cuda.get_device_name(0))  # 显示 GPU 名称
```

#### 自动 GPU 检测
脚本会自动检测 GPU：
```
✓ 检测到GPU: NVIDIA GeForce RTX 3090
  CUDA版本: 11.8
```

如果没有检测到 GPU，会自动降级到 CPU：
```
⚠ 未检测到GPU，将使用CPU训练
```

---

### 交互式查询模式

启动交互式查询界面后，你可以使用以下命令：

#### 1. 列出所有可用线路
```
> list
```

输出示例：
```
共有 5 条线路:
  1. kyoto_nara-A
  2. kyoto_nara-B
  3. mt_fuji-A
  4. mt_fuji-C
  5. sea_of_kyoto-A
```

#### 2. 查询单条线路的预测
```
> query <线路ID> <日期>

示例：
> query kyoto_nara-A 2024-06-15
> query mt_fuji-A 2024/6/15
```

输出示例：
```
✓ 预测结果:
  线路: kyoto_nara-A
  日期: 2024-06-15
  预测人数: 1523
  实际人数: 1500
  误差: 23 (1.5%)

各模型预测:
  xgboost: 1520
  lightgbm: 1525
  ridge: 1520

集成权重:
  xgboost: 40.0%
  lightgbm: 40.0%
  ridge: 20.0%
```

#### 3. 查询日期范围的预测
```
> range <线路ID> <开始日期> <结束日期>

示例：
> range kyoto_nara-A 2024-06-01 2024-06-30
```

输出示例：
```
✓ 预测结果 (30 天):
        date  prediction  actual  error
0 2024-06-01      1500    1512     -12
1 2024-06-02      1520    1505      15
2 2024-06-03      1480    1492     -12
...
```

#### 4. 批量查询多条线路
```
> batch <日期> [线路1] [线路2] ...

示例：
> batch 2024-06-15                    # 查询所有线路
> batch 2024-06-15 kyoto_nara-A mt_fuji-A  # 查询特定线路
```

输出示例：
```
✓ 批量预测结果 (5 条线路):
       route_id  prediction  actual  error
0  kyoto_nara-A      1523   1500     23
1  kyoto_nara-B      1450   1430     20
2     mt_fuji-A      2100   2080     20
3     mt_fuji-C      1850   1870    -20
4 sea_of_kyoto-A      800    810    -10
```

#### 5. 显示帮助
```
> help
```

#### 6. 退出程序
```
> exit
或
> quit
```

---

## 🔧 API 接口

### 使用 Python 代码进行预测

#### 方法 1: 简单查询
```python
from src.query_predict import query_single

# 单次查询
result = query_single('kyoto_nara-A', '2024-06-15')
print(f"预测人数: {result['prediction']:.0f}")
```

#### 方法 2: 创建查询器对象
```python
from src.query_predict import RoutePredictorQuery

# 初始化
predictor = RoutePredictorQuery()

# 查询可用线路
routes = predictor.list_routes()
print(f"可用线路: {routes}")

# 单次预测
prediction = predictor.predict('kyoto_nara-A', '2024-06-15')
print(f"预测人数: {prediction:.0f}")

# 详细预测（包含各模型预测和误差）
result = predictor.predict('kyoto_nara-A', '2024-06-15', return_details=True)
print(f"预测: {result['prediction']:.0f}")
print(f"实际: {result['actual']:.0f}")
print(f"误差: {result['error']:.0f}")
```

#### 方法 3: 日期范围查询
```python
# 查询一个月的数据
results_df = predictor.predict_range('kyoto_nara-A', '2024-06-01', '2024-06-30')
print(results_df)
```

#### 方法 4: 批量查询
```python
# 查询所有线路
results = predictor.batch_predict('all', '2024-06-15')
print(results)

# 查询特定线路
results = predictor.batch_predict(['kyoto_nara-A', 'mt_fuji-A'], '2024-06-15')
print(results)
```

---

## 📊 输出文件说明

训练完成后，在 `outputs/` 目录下会生成以下文件：

### 模型文件
- `xgboost_model.pkl` - XGBoost 模型（占用约 5-10MB）
- `lightgbm_model.pkl` - LightGBM 模型（占用约 3-8MB）
- `ridge_model.pkl` - Ridge 回归模型（占用约 1MB）

### 集成配置文件
- `ensemble_weights.pkl` - 集成模型的权重
- `trainer.pkl` - 训练器对象（包含变换函数）
- `feature_names.pkl` - 特征名称列表

### 评估和分析
- `evaluation_results.pkl` - 测试集上的评估结果
- `xgboost_feature_importance.png` - XGBoost 特征重要性图
- `lightgbm_feature_importance.png` - LightGBM 特征重要性图

---

## ⚙️ 性能优化

### GPU 训练的性能对比

| 参数 | CPU 训练 | GPU 训练 |
|------|---------|---------|
| XGBoost 训练时间 | ~30-60 秒 | ~5-10 秒 |
| LightGBM 训练时间 | ~20-40 秒 | ~3-8 秒 |
| 总训练时间 | ~60-120 秒 | ~15-30 秒 |

### GPU 选择建议
- **NVIDIA RTX 30 系列或以上**：完整支持，推荐
- **NVIDIA RTX 20 系列**：基本支持，可用
- **NVIDIA GTX 16 系列**：基本支持，可用
- **其他型号**：需自行验证驱动和 CUDA 支持

---

## 🐛 常见问题

### Q: 出现 "CUDA 错误" 怎么办？
**A:** 
1. 检查 NVIDIA 驱动版本：`nvidia-smi`
2. 检查 PyTorch CUDA 支持：`python -c "import torch; print(torch.cuda.is_available())"`
3. 如果有问题，使用 CPU 训练：`python quick_demo.py --mode train --no_gpu`

### Q: 查询时找不到数据怎么办？
**A:**
1. 检查线路 ID 是否正确：`python quick_demo.py --mode query --interactive`，然后输入 `list`
2. 检查日期是否在数据范围内，系统会显示该线路的日期范围
3. 确保数据文件 `f:/Pytorch/Dataset/visitordata.csv` 存在

### Q: 预测结果的准确性如何？
**A:**
- RMSE（均方根误差）：通常在 100-500 之间
- R² 分数：通常在 0.7-0.9 之间
- 实际准确性取决于数据质量和线路特征

### Q: 可以修改模型参数吗？
**A:** 可以，修改 `configs/model_config.yaml` 文件：
```yaml
training:
  xgboost:
    max_depth: 4          # 树的最大深度
    learning_rate: 0.05   # 学习率
    n_estimators: 500     # 树的数量
    reg_alpha: 1.0        # L1 正则化
    reg_lambda: 2.0       # L2 正则化
```

---

## 📚 相关文件

- `quick_demo.py` - 主演示脚本（本文件）
- `src/train.py` - 训练脚本（支持 GPU）
- `src/query_predict.py` - 查询预测模块
- `src/models/tree_models.py` - 树模型实现
- `src/models/ensemble.py` - 集成模型实现
- `configs/model_config.yaml` - 模型配置文件

---

## 🚀 高级用法

### 自定义训练
```python
from src.train import load_config, main
import torch

# 检查 GPU
print(f"GPU Available: {torch.cuda.is_available()}")

# 使用 GPU 训练
main(use_gpu=True)
```

### 结果导出
```python
from src.query_predict import RoutePredictorQuery
import pandas as pd

predictor = RoutePredictorQuery()

# 获取整个月的预测
results = predictor.predict_range('kyoto_nara-A', '2024-06-01', '2024-06-30')

# 保存为 CSV
results.to_csv('predictions_2024_06.csv', index=False)
```

---

**祝你使用愉快！** 🎉
