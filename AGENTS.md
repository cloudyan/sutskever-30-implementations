# AGENTS.md

本项目是 Sutskever 30 论文的纯 NumPy 实现，旨在教育和理解深度学习核心机制。

## 项目概览

- **技术栈**: 纯 NumPy (无 PyTorch/TensorFlow)
- **主要格式**: Jupyter Notebook (30 个实现) + Python 模块 (训练工具、推理任务)
- **运行方式**: Jupyter Notebook 交互式运行或直接执行 Python 脚本
- **测试方式**: 自定义测试框架，直接运行 Python 文件

## 构建和运行命令

### Jupyter Notebook (主要工作方式)
```bash
# 启动 Jupyter
jupyter notebook

# 或直接打开特定 notebook
jupyter notebook 02_char_rnn_karpathy.ipynb
```

### 运行测试
```bash
# 运行快速测试
python test_training_utils_quick.py

# 运行推理任务测试
python test_reasoning_tasks.py

# 运行集成测试
python test_relational_memory_integration.py

# 运行单个模块内嵌测试
python training_utils.py  # 运行 test_loss_functions, test_optimization_utilities
python lstm_baseline.py   # 运行 test_lstm
python relational_memory.py  # 运行多个测试函数
```

### 运行训练脚本
```bash
# 训练 LSTM 基线
python train_lstm_baseline.py

# 训练 Relational RNN
python train_relational_rnn.py

# 运行演示
python training_demo.py
python relational_memory_demo.py
python lstm_baseline_demo.py
```

### 单个测试运行
```python
# 通过导入运行特定测试
python -c "from training_utils import test_loss_functions; test_loss_functions()"
python -c "from relational_memory import test_layer_norm; test_layer_norm()"
```

## 代码风格指南

### 导入顺序
```python
# 1. 标准库
import numpy as np
import matplotlib.pyplot as plt

# 2. 项目内部模块
from lstm_baseline import LSTMCell, xavier_initializer, orthogonal_initializer
from attention_mechanism import multi_head_attention, init_attention_params
```

### 命名约定
- **类名**: `LSTMCell`, `RelationalMemory`, `RelationalRNN` (大驼峰)
- **函数名**: `forward`, `init_attention_params`, `create_causal_mask` (蛇形)
- **变量名**: `input_size`, `hidden_size`, `num_heads` (蛇形)
- **常量**: 大写加下划线 (如维度描述)
- **内部函数**: `_sigmoid` (下划线前缀)

### 格式化规则
- **缩进**: 4 个空格
- **行长度**: ~80 字符，长行自动换行
- **空行**: 函数间 2 行，类方法间 1 行，逻辑块间 1 行
- **运算符**: 周围添加空格
- **参数对齐**: 多参数时垂直对齐

### 文档注释 (三引号)
```python
def function_name(param1, param2):
    """
    功能描述

    Args:
        param1: 参数说明，形状 (batch, seq_len, dim)
        param2: 参数说明

    Returns:
        返回值说明，形状 (batch, output_dim)

    数学公式或算法步骤
    """
```

### 类型注解
```python
# 使用 typing 模块
from typing import Dict, Tuple, List, Optional, Any

def train_model(
    model,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Dict[str, List[float]]:
    """训练模型并返回历史记录"""
```

## 错误处理和验证

### 输入验证 (assert)
```python
# 形状检查
assert Q.ndim == 3, f"Q must be 3D, got shape {Q.shape}"

# 参数有效性
assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

# 数值稳定性
assert not np.any(np.isnan(attn_weights)), "NaN detected in attention weights"
assert not np.any(np.isinf(attn_weights)), "Inf detected in attention weights"
```

### 自定义异常
```python
raise ValueError("Invalid parameter value")
raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
```

## 编码约定

### 参数初始化
- **输入权重**: `xavier_initializer(shape)`
- **循环权重**: `orthogonal_initializer(shape, gain=1.0)`
- **Forget gate 偏置**: 初始化为 1.0 (帮助学习长期依赖)
- **其他偏置**: 初始化为 0.0

### 张量形状注释
在注释中明确说明张量形状变化：
```python
# x: (batch, seq_len, input_size)
# W: (hidden_size, input_size)
# h: (hidden_size, 1)
# output: (batch, output_size)
```

### 数学公式注释
```python
# Mathematical formulation:
#   f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)
#   i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)
```

## 项目特色

1. **纯 NumPy 实现**: 所有算法从零构建，无框架依赖
2. **教育导向**: 详细注释、数学公式、形状追踪
3. **合成数据**: 每个实现自带可运行的测试数据
4. **测试集成**: 模块内嵌测试函数，易于验证
5. **交互式学习**: Jupyter Notebook 支持可视化探索

## 常见任务

### 添加新模块
1. 创建 `.py` 文件，包含三引号模块级注释
2. 实现类和函数，遵循命名和文档约定
3. 添加 `if __name__ == "__main__":` 测试代码
4. 在测试文件中添加测试函数

### 修改现有代码
1. 保持一致的代码风格
2. 更新相关注释和文档
3. 添加测试验证修改
4. 确保 NumPy 兼容性

### 调试技巧
1. 使用 assert 进行快速验证
2. 打印张量形状检查维度
3. 检查 NaN/Inf 使用 `np.isnan()`, `np.isinf()`
4. 使用小批次和少量样本快速迭代
