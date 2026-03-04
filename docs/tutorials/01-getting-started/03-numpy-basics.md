# 1.3 NumPy 基础回顾：深度学习中最常用的操作都在这了！

问下大家，你在写深度学习代码时，用得最多的操作是什么？

晓寒刚开始学的时候，发现 80% 的代码都是在做数组操作：创建数组、改变形状、矩阵乘法、广播机制...

今天我们就来把这些高频操作一网打尽！看完这节，后面的实现你就得心应手了！

## NumPy 数组基础

首先，我们来回顾一下 NumPy 数组的基本概念。

### 创建数组

```python
import numpy as np

# 从列表创建
a = np.array([1, 2, 3, 4, 5])
print(f"一维数组: {a}")

# 创建二维数组
b = np.array([[1, 2, 3], [4, 5, 6]])
print(f"二维数组:\n{b}")

# 创建全 0 数组
zeros = np.zeros((2, 3))
print(f"全 0 数组:\n{zeros}")

# 创建全 1 数组
ones = np.ones((3, 2))
print(f"全 1 数组:\n{ones}")

# 创建随机数组
np.random.seed(42)  # 设置随机种子，保证结果可复现
random_arr = np.random.randn(2, 3)  # 标准正态分布
print(f"随机数组:\n{random_arr}")

# 创建范围数组
range_arr = np.arange(0, 10, 2)  # 从 0 到 10，步长 2
print(f"范围数组: {range_arr}")
```

### 数组属性

每个 NumPy 数组都有一些重要属性：

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f"形状: {arr.shape}")      # (2, 3) - 2 行 3 列
print(f"维度: {arr.ndim}")       # 2 - 二维数组
print(f"元素总数: {arr.size}")   # 6 - 总共 6 个元素
print(f"数据类型: {arr.dtype}")  # int64 - 整数类型
```

**形状（shape）是深度学习中最重要的概念之一！** 我们会在注释中反复强调形状变化。

## 索引和切片

如何从数组中取出我们想要的部分？

### 基本索引

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# 取单个元素
print(f"第 0 行第 1 列: {arr[0, 1]}")  # 2

# 取整行
print(f"第 1 行: {arr[1, :]}")  # [5, 6, 7, 8]

# 取整列
print(f"第 2 列: {arr[:, 2]}")  # [3, 7, 11]

# 取子数组
print(f"前 2 行，后 3 列:\n{arr[:2, 1:]}")
# [[2, 3, 4]
#  [6, 7, 8]]
```

### 布尔索引

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# 选出大于 3 的元素
mask = arr > 3
print(f"布尔掩码: {mask}")  # [False, False, False, True, True, True]
print(f"大于 3 的元素: {arr[mask]}")  # [4, 5, 6]

# 一步到位
print(f"大于 3 的元素: {arr[arr > 3]}")
```

## 形状变换

深度学习中最常做的操作就是改变数组形状！

### reshape - 改变形状

```python
arr = np.arange(12)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 变成 3 行 4 列
arr_3x4 = arr.reshape(3, 4)
print(f"3x4 数组:\n{arr_3x4}")

# 变成 2 行 6 列
arr_2x6 = arr.reshape(2, 6)
print(f"2x6 数组:\n{arr_2x6}")

# 用 -1 自动计算
arr_2x_ = arr.reshape(2, -1)  # -1 表示自动计算
print(f"2x自动数组:\n{arr_2x_}")
```

**注意**：reshape 的总元素数必须一致！

### expand_dims - 增加维度

```python
arr = np.array([1, 2, 3])  # 形状 (3,)

# 在第 0 维增加
arr_expand_0 = np.expand_dims(arr, axis=0)
print(f"在第 0 维增加: {arr_expand_0.shape}")  # (1, 3)

# 在第 1 维增加
arr_expand_1 = np.expand_dims(arr, axis=1)
print(f"在第 1 维增加: {arr_expand_1.shape}")  # (3, 1)
```

### squeeze - 去除单维度

```python
arr = np.array([[[1, 2, 3]]])  # 形状 (1, 1, 3)

# 去除所有单维度
arr_squeezed = np.squeeze(arr)
print(f"去除单维度后: {arr_squeezed.shape}")  # (3,)
```

### transpose - 转置

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])  # 形状 (2, 3)

# 转置
arr_transposed = arr.T
print(f"转置后形状: {arr_transposed.shape}")  # (3, 2)
print(f"转置数组:\n{arr_transposed}")

# 高维数组转置
arr_3d = np.random.randn(2, 3, 4)
arr_3d_transposed = arr_3d.transpose(1, 0, 2)  # 交换第 0 和第 1 维
print(f"原形状: {arr_3d.shape}")          # (2, 3, 4)
print(f"转置后形状: {arr_3d_transposed.shape}")  # (3, 2, 4)
```

## 数组拼接和分割

### concatenate - 拼接

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 按行拼接（axis=0）
concat_row = np.concatenate([a, b], axis=0)
print(f"按行拼接:\n{concat_row}")
# [[1, 2]
#  [3, 4]
#  [5, 6]
#  [7, 8]]

# 按列拼接（axis=1）
concat_col = np.concatenate([a, b], axis=1)
print(f"按列拼接:\n{concat_col}")
# [[1, 2, 5, 6]
#  [3, 4, 7, 8]]
```

### stack - 堆叠（增加新维度）

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 在新维度堆叠
stacked = np.stack([a, b], axis=0)
print(f"堆叠后:\n{stacked}")
# [[1, 2, 3]
#  [4, 5, 6]]
print(f"堆叠后形状: {stacked.shape}")  # (2, 3)
```

## 数学运算

### 逐元素运算

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"加法: {a + b}")      # [5, 7, 9]
print(f"减法: {a - b}")      # [-3, -3, -3]
print(f"乘法: {a * b}")      # [4, 10, 18] - 逐元素乘法！
print(f"除法: {a / b}")      # [0.25, 0.4, 0.5]
print(f"幂运算: {a ** 2}")   # [1, 4, 9]
```

**注意**：`*` 是逐元素乘法，不是矩阵乘法！

### 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])  # 形状 (2, 2)
B = np.array([[5, 6], [7, 8]])  # 形状 (2, 2)

# 矩阵乘法 - 三种方式
matmul1 = np.matmul(A, B)
matmul2 = A @ B  # Python 3.5+ 推荐
matmul3 = np.dot(A, B)  # 也可以用 dot

print(f"矩阵乘法结果:\n{matmul1}")
# [[1*5+2*7, 1*6+2*8],
#  [3*5+4*7, 3*6+4*8]]
# = [[19, 22],
#    [43, 50]]
```

### 向量点积

```python
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

dot_product = np.dot(v, w)
print(f"向量点积: {dot_product}")  # 1*4 + 2*5 + 3*6 = 32
```

## 广播机制（Broadcasting）

广播是 NumPy 最强大的特性之一！它让不同形状的数组也能进行运算。

### 广播规则

1. 如果数组维度不同，在维度较小的数组前面补 1
2. 如果某个维度上大小为 1，可以扩展到和另一个数组相同

### 广播示例

```python
# 示例 1: 标量和数组
a = np.array([1, 2, 3])
b = 2
print(f"标量广播: {a + b}")  # [3, 4, 5]

# 示例 2: 一维和二维
a = np.array([[1, 2, 3], [4, 5, 6]])  # 形状 (2, 3)
b = np.array([10, 20, 30])              # 形状 (3,)
# b 自动扩展为 [[10, 20, 30], [10, 20, 30]]
print(f"一维广播到二维:\n{a + b}")
# [[11, 22, 33]
#  [14, 25, 36]]

# 示例 3: 两个维度都为 1
a = np.array([[1], [2], [3]])  # 形状 (3, 1)
b = np.array([[10, 20]])        # 形状 (1, 2)
print(f"双向广播:\n{a + b}")
# [[11, 21]
#  [12, 22]
#  [13, 23]]
```

## 聚合函数

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f"总和: {np.sum(arr)}")              # 21
print(f"按行求和: {np.sum(arr, axis=0)}")  # [5, 7, 9]
print(f"按列求和: {np.sum(arr, axis=1)}")  # [6, 15]

print(f"平均值: {np.mean(arr)}")            # 3.5
print(f"最大值: {np.max(arr)}")              # 6
print(f"最小值: {np.min(arr)}")              # 1
print(f"标准差: {np.std(arr)}")              # 约 1.7078

# 最大值索引
print(f"最大值索引（展平）: {np.argmax(arr)}")  # 5
print(f"按列最大值索引: {np.argmax(arr, axis=0)}")  # [1, 1, 1]
```

## 常用激活函数

深度学习中常用的激活函数，我们用 NumPy 实现一下：

```python
def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def tanh(x):
    """Tanh 激活函数"""
    return np.tanh(x)

def softmax(x, axis=-1):
    """Softmax 激活函数"""
    # 减去最大值防止溢出
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 测试
x = np.array([-2, -1, 0, 1, 2])
print(f"Sigmoid: {sigmoid(x)}")
print(f"ReLU: {relu(x)}")
print(f"Tanh: {tanh(x)}")
print(f"Softmax: {softmax(x)}")
```

## 形状注释规范

在我们的代码中，会用注释明确说明张量形状：

```python
# 输入形状: (batch_size, seq_len, input_size)
# 例如: (32, 10, 128) 表示 32 个样本，每个序列长度 10，输入维度 128

# 权重形状: (hidden_size, input_size)
# 偏置形状: (hidden_size,)

# 输出形状: (batch_size, seq_len, hidden_size)
```

## 实战小练习

来做几个小练习巩固一下！

### 练习 1: 线性层前向传播

实现一个简单的线性层前向传播：

```python
def linear_forward(x, W, b):
    """
    线性层前向传播
    
    Args:
        x: 输入，形状 (batch_size, input_size)
        W: 权重，形状 (output_size, input_size)
        b: 偏置，形状 (output_size,)
    
    Returns:
        输出，形状 (batch_size, output_size)
    """
    # 你的代码: y = x @ W.T + b
    return x @ W.T + b

# 测试
np.random.seed(42)
x = np.random.randn(2, 3)    # (batch=2, input=3)
W = np.random.randn(4, 3)     # (output=4, input=3)
b = np.random.randn(4)        # (output=4,)

y = linear_forward(x, W, b)
print(f"线性层输出形状: {y.shape}")  # 应该是 (2, 4)
```

### 练习 2: Batch Normalization

实现简单的 Batch Normalization：

```python
def batch_norm(x, gamma, beta, eps=1e-8):
    """
    Batch Normalization
    
    Args:
        x: 输入，形状 (batch_size, feature_dim)
        gamma: 缩放参数，形状 (feature_dim,)
        beta: 偏移参数，形状 (feature_dim,)
        eps: 小常数防止除零
    
    Returns:
        归一化后的输出
    """
    # 计算均值和方差
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    
    # 归一化
    x_normalized = (x - mean) / np.sqrt(var + eps)
    
    # 缩放和偏移
    out = gamma * x_normalized + beta
    
    return out

# 测试
x = np.random.randn(10, 5)  # 10 个样本，5 个特征
gamma = np.ones(5)
beta = np.zeros(5)

out = batch_norm(x, gamma, beta)
print(f"BN 输出均值: {np.mean(out, axis=0)}")  # 应该接近 0
print(f"BN 输出方差: {np.var(out, axis=0)}")    # 应该接近 1
```

## 总结

综上所述，NumPy 的核心操作包括：

1. **数组创建**：array、zeros、ones、random、arange
2. **形状变换**：reshape、expand_dims、squeeze、transpose
3. **索引切片**：基本索引、布尔索引
4. **数学运算**：逐元素运算、矩阵乘法、广播机制
5. **聚合函数**：sum、mean、max、min、argmax
6. **形状意识**：始终关注数组的 shape！

掌握这些操作，你就能看懂并实现后面的所有算法了！

## 恭喜！

恭喜你完成了第一章的学习！现在你已经：

✅ 了解了 Sutskever 30 论文列表  
✅ 搭建好了开发环境  
✅ 掌握了 NumPy 核心操作  

准备好开始深度学习之旅了吗？让我们进入第二章，从字符级 RNN 开始！
