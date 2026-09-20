# Day 2: 矩阵与矩阵运算

> 神经引擎的核心 —— 理解矩阵如何驱动神经网络
> 
> _难度：⭐⭐ | 预计时间：1-2 小时 | AI 应用：神经网络前向传播_

---

## 📌 一句话核心

**矩阵是向量的数组，矩阵乘法是神经网络信息传递的核心运算。**

---

## 🎯 核心问题

学完今天的内容，你应该能够回答：
1. 什么是矩阵？它和向量有什么关系？
2. 矩阵乘法的规则是什么？为什么这样定义？
3. 矩阵乘法的几何意义是什么？
4. 神经网络如何用矩阵运算实现前向传播？

---

## 📚 核心定义

### 什么是矩阵？

```
数学定义：
矩阵是一个 m×n 的矩形数组，包含 m 行 n 列

表示方法：
        [a₁₁  a₁₂  ...  a₁ₙ]
    A = [a₂₁  a₂₂  ...  a₂ₙ]
        [...  ...  ...  ...]
        [aₘ₁  aₘ₂  ...  aₘₙ]

元素 aᵢⱼ：第 i 行第 j 列的值
```

### 矩阵的维度

```
2×3 矩阵（2 行 3 列）：
    [1  2  3]
    [4  5  6]

3×3 方阵（3 行 3 列）：
    [1  2  3]
    [4  5  6]
    [7  8  9]

n×1 矩阵 = 列向量：
    [x₁]
    [x₂]
    [...]
    [xₙ]
```

### 矩阵 vs 向量

```
关系：向量是特殊的矩阵

向量 [1, 2, 3] 可以看作：
- 行向量：1×3 矩阵 [1  2  3]
- 列向量：3×1 矩阵 [1]
                    [2]
                    [3]

在 AI 中，默认使用行向量表示
```

---

## 📐 几何直观

### 矩阵作为变换

```
核心思想：矩阵乘法 = 空间变换

2×2 矩阵可以表示 2D 空间的变换：

原始向量：[1, 0]（x 轴单位向量）
变换矩阵：[2  0]
          [0  2]
结果向量：[2, 0]（拉伸 2 倍）

这个矩阵表示"均匀放大 2 倍"的变换
```

### 常见变换矩阵

```
1. 缩放变换（放大 2 倍）：
   [2  0]
   [0  2]

2. 旋转变换（逆时针 90°）：
   [0  -1]
   [1   0]

3. 剪切变换：
   [1  1]
   [0  1]

4. 镜像变换（关于 y 轴）：
   [-1  0]
   [0   1]
```

### 变换可视化

```
原始正方形：
  (0,1) ┌─────┐ (1,1)
        │     │
        │     │
  (0,0) └─────┘ (1,0)

经过矩阵 [2  0] 变换后：
         [0  2]

  (0,2) ┌───────┐ (2,2)
        │       │
        │       │
        │       │
        │       │
  (0,0) └───────┘ (2,0)

正方形被放大 2 倍
```

---

## 🔢 核心运算

### 1. 矩阵加法

```python
# 规则：对应位置相加
A = [1  2]    B = [5  6]
    [3  4]        [7  8]

A + B = [1+5  2+6] = [6   8]
        [3+7  4+8]   [10 12]

# Python 实现
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A + B  # [[6, 8], [10, 12]]
```

### 2. 矩阵数乘

```python
# 规则：每个元素乘以标量
k = 2
A = [1  2]
    [3  4]

k × A = [2×1  2×2] = [2  4]
        [2×3  2×4]   [6  8]

# Python 实现
k = 2
C = k * A  # [[2, 4], [6, 8]]
```

### 3. 矩阵乘法 ⭐

```python
# 规则：行×列
A = [a  b]    B = [e  f]
    [c  d]        [g  h]

A × B = [a×e+b×g  a×f+b×h]
        [c×e+d×g  c×f+d×h]

# 记忆方法：第 i 行 × 第 j 列 = 结果的第 (i,j) 元素
```

### 矩阵乘法的条件

```
A(m×n) × B(n×p) = C(m×p)

关键：A 的列数 = B 的行数

示例：
A(2×3) × B(3×4) = C(2×4)  ✓

A(2×3) × B(4×3)           ✗（无法相乘）
```

### 矩阵乘法代码实现

```python
import numpy as np

# 方法 1：使用 np.dot
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)

# 方法 2：使用 @ 运算符（推荐）
C = A @ B

# 方法 3：手动实现（理解原理）
def matrix_multiply(A, B):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "矩阵维度不匹配"
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

C = matrix_multiply(A, B)
print(C)  # [[19, 22], [43, 50]]
```

### 手动计算示例

```
A = [1  2]    B = [5  6]
    [3  4]        [7  8]

C[0,0] = 1×5 + 2×7 = 5 + 14 = 19
C[0,1] = 1×6 + 2×8 = 6 + 16 = 22
C[1,0] = 3×5 + 4×7 = 15 + 28 = 43
C[1,1] = 3×6 + 4×8 = 18 + 32 = 50

C = [19  22]
    [43  50]
```

---

## 🤖 AI 中的应用

### 应用 1：神经网络前向传播

```
单层神经网络：

输入：x = [x₁, x₂, x₃]（1×3 向量）
权重：W = [w₁₁  w₁₂  w₁₃]（3×2 矩阵）
            [w₂₁  w₂₂  w₂₃]
            [w₃₁  w₃₂  w₃₃]
偏置：b = [b₁, b₂]（1×2 向量）

输出：y = x × W + b

计算过程：
y₁ = x₁×w₁₁ + x₂×w₂₁ + x₃×w₃₁ + b₁
y₂ = x₁×w₁₂ + x₂×w₂₂ + x₃×w₃₂ + b₂
```

### 神经网络代码实现

```python
import numpy as np

class NeuralLayer:
    """神经网络层"""
    
    def __init__(self, input_size, output_size):
        # 初始化权重（随机）
        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros((1, output_size))
    
    def forward(self, x):
        """前向传播：y = xW + b"""
        return x @ self.W + self.b

# 创建一层：3 个输入 → 2 个输出
layer = NeuralLayer(3, 2)

# 输入数据（batch_size=4）
x = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# 前向传播
output = layer.forward(x)
print(f"输出形状：{output.shape}")  # (4, 2)
print(output)
```

### 应用 2：批量数据处理

```
传统方式（循环，慢）：
for i in range(batch_size):
    y[i] = x[i] @ W + b

矩阵方式（向量化，快 100 倍+）：
Y = X @ W + b

X: (batch_size, input_size)
W: (input_size, output_size)
Y: (batch_size, output_size)
```

### 应用 3：卷积神经网络的卷积层

```
卷积操作本质也是矩阵乘法：

输入图片：28×28
卷积核：3×3
输出：26×26

通过 im2col 技巧，卷积可以转换为矩阵乘法：
- 将输入图片展开为矩阵
- 将卷积核展开为矩阵
- 矩阵乘法得到结果

这就是为什么 GPU 能加速卷积（擅长矩阵运算）
```

---

## 💻 代码实践

### 练习 1：矩阵基本运算

```python
import numpy as np

# 创建矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

print(f"A 的形状：{A.shape}")  # (2, 3)
print(f"B 的形状：{B.shape}")  # (3, 2)

# 矩阵加法（需要相同形状）
C = np.array([[1, 2, 3],
              [4, 5, 6]])
D = np.array([[7, 8, 9],
              [10, 11, 12]])
E = C + D
print(f"C + D = \n{E}")

# 矩阵乘法
F = A @ B
print(f"A × B = \n{F}")  # 形状 (2, 2)
```

### 练习 2：实现全连接层

```python
import numpy as np

class FullyConnectedLayer:
    """全连接层（线性层）"""
    
    def __init__(self, input_size, output_size):
        # Xavier 初始化
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
    
    def forward(self, x):
        """前向传播"""
        return x @ self.W + self.b
    
    def __call__(self, x):
        return self.forward(x)

# 测试
layer = FullyConnectedLayer(784, 128)
x = np.random.randn(32, 784)  # batch_size=32, 输入维度 784
output = layer(x)
print(f"输出形状：{output.shape}")  # (32, 128)
```

### 练习 3：实现简单神经网络

```python
import numpy as np

class SimpleNN:
    """两层神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = FullyConnectedLayer(input_size, hidden_size)
        self.layer2 = FullyConnectedLayer(hidden_size, output_size)
    
    def relu(self, x):
        """ReLU 激活函数"""
        return np.maximum(0, x)
    
    def forward(self, x):
        """前向传播"""
        h = self.relu(self.layer1(x))  # 隐藏层 + ReLU
        out = self.layer2(h)           # 输出层
        return out

# 测试（MNIST 分类）
model = SimpleNN(784, 128, 10)
x = np.random.randn(64, 784)  # 64 张 28×28 图片
output = model.forward(x)
print(f"输出形状：{output.shape}")  # (64, 10) - 10 个类别的分数
```

### 练习 4：矩阵变换可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_transform(matrix, title):
    """可视化矩阵变换"""
    # 原始单位正方形
    square = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0]  # 闭合
    ])
    
    # 变换后
    transformed = square @ matrix.T
    
    plt.figure(figsize=(6, 6))
    plt.plot(square[:, 0], square[:, 1], 'b--', label='原始', alpha=0.5)
    plt.plot(transformed[:, 0], transformed[:, 1], 'r-', label='变换后', linewidth=2)
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.show()

# 测试不同变换
plot_transform(np.array([[2, 0], [0, 2]]), '放大 2 倍')
plot_transform(np.array([[0, -1], [1, 0]]), '旋转 90°')
plot_transform(np.array([[1, 1], [0, 1]]), '剪切变换')
```

---

## 🧠 思考题

1. **基础题**：计算以下矩阵乘法：
   ```
   [1  2  3]   [10]
   [4  5  6] × [20]
               [30]
   ```

2. **理解题**：为什么矩阵乘法不满足交换律？即 A×B ≠ B×A？

3. **应用题**：一个神经网络有 1000 个输入，128 个隐藏单元，10 个输出。计算权重矩阵 W1 和 W2 的形状，以及总参数数量。

4. **挑战题**：证明矩阵乘法的结合律：(A×B)×C = A×(B×C)。

---

## 📝 关键公式总结

```
1. 矩阵加法：(A + B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ

2. 矩阵数乘：(kA)ᵢⱼ = k × Aᵢⱼ

3. 矩阵乘法：(A × B)ᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ

4. 神经网络前向传播：y = xW + b

5. 转置：(Aᵀ)ᵢⱼ = Aⱼᵢ
```

---

## ✅ 今日检查清单

- [ ] 理解矩阵的定义和表示
- [ ] 掌握矩阵加法和数乘
- [ ] 能够手动计算矩阵乘法
- [ ] 理解矩阵乘法的几何意义
- [ ] 实现神经网络前向传播
- [ ] 完成所有代码练习
- [ ] 回答所有思考题

---

> _矩阵是神经网络的引擎，每一次矩阵乘法都是信息的流动。_
> 
> _—— 悟空_
