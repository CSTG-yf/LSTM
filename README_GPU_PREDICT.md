# 小样本线路时序预测 - GPU 训练和查询系统

## 🎯 核心功能

### 1. **GPU 加速训练**
使用 GPU 加速树模型训练，速度提升 **5-10 倍**。

### 2. **线路+日期查询预测**
输入线路名称和日期，获取当天访客数预测。

---

## 🚀 快速开始（3 步）

### 第 1 步：检查系统（可选）
```bash
python test_imports.py
```

### 第 2 步：GPU 训练
```bash
# 使用 GPU（推荐，约 15-30 秒）
python train_and_predict.py

# 或使用 CPU（约 60-120 秒）
python train_and_predict.py --no-gpu
```

### 第 3 步：查询预测
```bash
# 交互式查询（推荐）
python train_and_predict.py --interactive

# 或直接查询
python quick_demo.py --mode query --route kyoto_nara-A --date 2024-06-15
```

---

## 📖 详细使用指南

### 方式 1：完整演示（训练 + 预测）
```bash
python train_and_predict.py
```
这会自动完成：
1. 检查 GPU
2. 训练模型（15-30 秒）
3. 显示预测示例

### 方式 2：交互式查询
```bash
python quick_demo.py --mode query --interactive
```
支持的命令：
- `list` - 列出所有线路
- `query <线路> <日期>` - 查询单条线路
- `range <线路> <开始日期> <结束日期>` - 查询日期范围
- `batch <日期>` - 批量查询所有线路
- `help` - 显示帮助
- `exit` - 退出

### 方式 3：命令行查询
```bash
python quick_demo.py --mode query --route kyoto_nara-A --date 2024-06-15
```

### 方式 4：Python API
```python
from src.query_predict import query_single

result = query_single('kyoto_nara-A', '2024-06-15')
print(f"预测: {result['prediction']:.0f} 人")
```

---

## 🎬 使用示例

### 示例 1：查询单条线路
```
> query kyoto_nara-A 2024-06-15

✓ 预测结果:
  线路: kyoto_nara-A
  日期: 2024-06-15
  预测人数: 1523
  实际人数: 1500
  误差: 23 (1.5%)
```

### 示例 2：批量查询
```
> batch 2024-06-15

✓ 批量预测结果 (5 条线路):
       route_id  prediction  actual   error
0  kyoto_nara-A      1523    1500      23
1  kyoto_nara-B      1450    1430      20
2     mt_fuji-A      2100    2080      20
```

### 示例 3：日期范围查询
```
> range kyoto_nara-A 2024-06-01 2024-06-10

✓ 预测结果 (10 天):
        date  prediction  actual  error
0 2024-06-01      1500    1512     -12
1 2024-06-02      1520    1505      15
```

---

## ⚙️ 训练选项

| 选项 | 说明 | 速度 |
|------|------|------|
| `--use_gpu` | 使用 GPU 训练（默认） | 15-30 秒 |
| `--no_gpu` | 使用 CPU 训练 | 60-120 秒 |

### GPU 要求
- NVIDIA GPU（GTX 16、RTX 20/30 系列或更新）
- CUDA 11.0+
- cuDNN 8.0+

### 如果没有 GPU
自动降级到 CPU，无需改动代码。

---

## 🔧 所有可用命令

### 训练命令
```bash
# GPU 训练
python train_and_predict.py
python quick_demo.py --mode train --use_gpu

# CPU 训练
python train_and_predict.py --no_gpu
python quick_demo.py --mode train --no_gpu
```

### 查询命令
```bash
# 交互式查询
python train_and_predict.py --interactive
python quick_demo.py --mode query --interactive

# 单次查询
python quick_demo.py --mode query --route <线路> --date <日期>

# 只预测示例
python train_and_predict.py --predict-only
```

### 辅助命令
```bash
# 导入测试
python test_imports.py

# 系统检查
python check_system.py
```

---

## 📊 预期结果

### 训练输出
```
✓ 检测到GPU: NVIDIA GeForce RTX 3090

[1/6] 配置加载完成
[2/6] 开始加载数据...
[3/6] 初始化模型训练器...
[4/6] 开始训练模型...
[5/6] 创建集成模型...
[6/6] 在测试集上评估...

集成模型:
  RMSE: 148.56
  MAE: 112.34
  R²: 0.8698

✓ 模型已保存至 outputs/ 目录
```

### 预测输出
```
✓ 预测结果:
  线路: kyoto_nara-A
  日期: 2024-06-15
  预测人数: 1523

各模型预测:
  xgboost: 1520
  lightgbm: 1525
  ridge: 1520

集成权重:
  xgboost: 40.0%
  lightgbm: 40.0%
  ridge: 20.0%
```

---

## 📁 文件结构

```
.
├── quick_demo.py                 # 快速演示脚本
├── train_and_predict.py          # 完整演示脚本（推荐）
├── test_imports.py               # 导入测试
├── check_system.py               # 系统检查
├── QUICK_START.md                # 快速开始指南
├── GPU_TRAINING_GUIDE.md         # 详细指南
├── IMPLEMENTATION_SUMMARY.md     # 实现总结
├── src/
│   ├── train.py                  # 训练脚本（支持 GPU）
│   ├── query_predict.py          # 查询预测模块
│   ├── models/
│   │   ├── tree_models.py        # 树模型（支持 GPU）
│   │   └── ensemble.py           # 集成模型
│   └── data/
│       └── timeseries_dataset.py # 时序数据处理
├── configs/
│   └── model_config.yaml         # 配置文件
├── outputs/                      # 模型和结果输出目录
└── Dataset/
    └── visitordata.csv           # 数据文件
```

---

## ❓ 常见问题

**Q: 没有 GPU 可以用吗？**  
A: 当然可以，自动降级到 CPU，只是速度会慢一些。

**Q: 如何修改模型参数？**  
A: 编辑 `configs/model_config.yaml` 文件。

**Q: 训练需要多长时间？**  
A: GPU 训练约 15-30 秒，CPU 训练约 60-120 秒。

**Q: 如何导出预测结果？**  
A: 使用 Python API：
```python
from src.query_predict import RoutePredictorQuery

predictor = RoutePredictorQuery()
results = predictor.predict_range('kyoto_nara-A', '2024-06-01', '2024-06-30')
results.to_csv('predictions.csv')
```

---

## 🎓 更多文档

- `QUICK_START.md` - 三步快速开始
- `GPU_TRAINING_GUIDE.md` - GPU 训练和查询详细指南
- `IMPLEMENTATION_SUMMARY.md` - 实现细节

---

**现在就开始使用吧！** 🚀

```bash
python train_and_predict.py
```
