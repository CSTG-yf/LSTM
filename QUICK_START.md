# 快速开始指南

## 🚀 三步快速开始

### 第 1 步：系统检查（可选但推荐）
```bash
python check_system.py
```

这会检查：
- ✓ GPU 是否可用
- ✓ 依赖包是否完整
- ✓ 数据文件是否存在
- ✓ 项目结构是否完整

### 第 2 步：GPU 训练

**使用 GPU 训练（推荐，快 5-10 倍）：**
```bash
python quick_demo.py --mode train --use_gpu
```

**或使用 CPU 训练：**
```bash
python quick_demo.py --mode train --no_gpu
```

训练完成后会显示：
```
✓ 所有模型训练完成
✓ 测试集评估结果
  集成模型 RMSE: xxx.xx
```

### 第 3 步：线路+日期查询预测

**最简单的方式 - 交互式查询：**
```bash
python quick_demo.py --mode query --interactive
```

然后你可以输入命令，例如：
```
> list                                    # 列出所有线路
> query kyoto_nara-A 2024-06-15          # 查询单条线路
> range kyoto_nara-A 2024-06-01 2024-06-30  # 查询日期范围
> batch 2024-06-15                        # 批量查询所有线路
> exit                                    # 退出
```

**或者直接单次查询：**
```bash
python quick_demo.py --mode query --route kyoto_nara-A --date 2024-06-15
```

---

## 📋 完整命令列表

### 训练模式

| 命令 | 说明 | 示例 |
|------|------|------|
| `--mode train --use_gpu` | 使用 GPU 训练（默认） | `python quick_demo.py --mode train --use_gpu` |
| `--mode train --no_gpu` | 使用 CPU 训练 | `python quick_demo.py --mode train --no_gpu` |

### 查询模式

| 命令 | 说明 | 示例 |
|------|------|------|
| `--mode query --interactive` | 交互式查询 | `python quick_demo.py --mode query --interactive` |
| `--mode query --route <线路> --date <日期>` | 单次查询 | `python quick_demo.py --mode query --route kyoto_nara-A --date 2024-06-15` |

### 交互式查询命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `list` | 列出所有可用线路 | `> list` |
| `query <线路> <日期>` | 查询单条线路的预测 | `> query kyoto_nara-A 2024-06-15` |
| `range <线路> <开始日期> <结束日期>` | 查询日期范围的预测 | `> range kyoto_nara-A 2024-06-01 2024-06-30` |
| `batch <日期> [线路...]` | 批量查询多条线路 | `> batch 2024-06-15` 或 `> batch 2024-06-15 kyoto_nara-A mt_fuji-A` |
| `help` | 显示帮助 | `> help` |
| `exit` 或 `quit` | 退出程序 | `> exit` |

---

## 🎯 场景示例

### 场景 1：我想快速训练一个模型

```bash
# 1. 检查系统
python check_system.py

# 2. GPU 训练（推荐，约 15-30 秒）
python quick_demo.py --mode train --use_gpu

# 完成！
```

### 场景 2：我想查询某个线路在某个日期的访客数预测

```bash
# 方式 A：直接查询
python quick_demo.py --mode query --route kyoto_nara-A --date 2024-06-15

# 方式 B：交互式查询
python quick_demo.py --mode query --interactive
> query kyoto_nara-A 2024-06-15
```

### 场景 3：我想看看某条线路整个月的预测

```bash
python quick_demo.py --mode query --interactive
> range kyoto_nara-A 2024-06-01 2024-06-30
```

### 场景 4：我想对比多条线路在同一天的预测

```bash
python quick_demo.py --mode query --interactive
> batch 2024-06-15
```

### 场景 5：我想在 Python 代码中使用预测

```python
from src.query_predict import query_single

# 单次查询
result = query_single('kyoto_nara-A', '2024-06-15')
print(f"预测人数: {result['prediction']:.0f}")
print(f"实际人数: {result['actual']:.0f}")
print(f"误差: {result['error']:.0f}")
```

---

## 📊 预期输出

### 训练输出示例
```
======================================================================
小样本线路时序预测 - 树模型集成方案
======================================================================

✓ 检测到GPU: NVIDIA GeForce RTX 3090
  CUDA版本: 11.8

[1/6] 配置加载完成

[2/6] 开始加载数据...

数据加载完成:
  训练集: (400, 25)
  验证集: (100, 25)
  测试集: (100, 25)
  特征数: 25

[3/6] 初始化模型训练器...

[4/6] 开始训练模型...
==================================================
开始训练所有模型
==================================================

=== 训练 XGBoost ===
✓ 检测到GPU: NVIDIA GeForce RTX 3090
训练集 - RMSE: 125.34, MAE: 95.23, R²: 0.8945
验证集 - RMSE: 145.67, MAE: 110.45, R²: 0.8723
最佳迭代轮数: 425

...（更多训练输出）

[5/6] 创建集成模型...

[6/6] 在测试集上评估...
======================================================================
测试集评估结果
======================================================================

集成模型:
  RMSE: 148.56
  MAE: 112.34
  R²: 0.8698

✓ 模型已保存至 outputs/ 目录
```

### 预测输出示例
```
======================================================================
查询线路 'kyoto_nara-A' 在 '2024-06-15' 的预测...
======================================================================

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

---

## 🔧 常见问题

### Q: GPU 训练失败，怎么办？
**A:** 使用 CPU 训练：
```bash
python quick_demo.py --mode train --no_gpu
```

### Q: 找不到线路 'kyoto_nara-A'，怎么办？
**A:** 先列出所有可用线路：
```bash
python quick_demo.py --mode query --interactive
> list
```

### Q: 日期应该是什么格式？
**A:** 支持多种格式：
- `2024-06-15`
- `2024/6/15`
- `2024-6-15`

### Q: 训练需要多长时间？
**A:** 
- GPU 训练：约 15-30 秒
- CPU 训练：约 60-120 秒

### Q: 我想修改模型参数，怎么办？
**A:** 编辑 `configs/model_config.yaml` 文件，然后重新训练。

---

## 📚 相关文档

- `GPU_TRAINING_GUIDE.md` - GPU 训练和查询详细指南
- `configs/model_config.yaml` - 模型配置文件
- `src/query_predict.py` - 预测查询模块源代码

---

## 💡 提示

1. **首次使用**：先运行 `python check_system.py` 检查系统
2. **GPU 加速**：如果有 GPU，务必使用 `--use_gpu` 标志
3. **交互式查询**：推荐使用交互式查询模式，更直观
4. **批量查询**：需要对比多条线路时，使用 `batch` 命令

---

**开始体验吧！** 🎉
