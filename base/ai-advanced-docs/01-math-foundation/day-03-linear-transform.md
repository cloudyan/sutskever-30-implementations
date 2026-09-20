# Day 3: 线性变换

> 矩阵是变换的语言 —— 理解矩阵如何改变空间
> 
> _难度：⭐⭐⭐ | 预计时间：1-2 小时 | AI 应用：数据变换、降维、神经网络_

---

## 📌 一句话核心

**矩阵乘法本质上是空间变换，它将向量从一个位置移动到另一个位置。**

---

## 🎯 核心问题

学完今天的内容，你应该能够回答：
1. 什么是线性变换？它满足什么性质？
2. 矩阵如何表示变换？
3. 常见变换有哪些（旋转、缩放、剪切）？
4. 线性变换在 AI 中有什么应用？

---

## 📚 核心定义

### 什么是线性变换？

```
数学定义：
变换 T 是线性的，当且仅当满足：

1. 可加性：T(u + v) = T(u) + T(v)
2. 齐次性：T(cu) = c·T(u)

直观理解：
- 直线变换后还是直线
- 原点变换后还在原点
- 平行线变换后还是平行
```

### 矩阵表示变换

```
核心定理：
每个线性变换都可以用矩阵表示

2D 空间：
T(x) = A·x

其中：
- x: 输入向量 (2×1)
- A: 变换矩阵 (2×2)
- T(x): 输出向量 (2×1)

示例：
[3]     [2  1]   [1]
[4]  =  [1  2] × [2]

向量 [1, 2] 经过矩阵 [2,1;1,2] 变换后变成 [3, 4]
```

---

## 📐 几何直观

### 常见 2D 变换矩阵

```
1. 恒等变换（不变）：
   [1  0]
   [0  1]

2. 缩放变换（x 轴 2 倍，y 轴 3 倍）：
   [2  0]
   [0  3]

3. 旋转（逆时针θ角）：
   [cosθ  -sinθ]
   [sinθ   cosθ]
   
   旋转 90°:
   [0  -1]
   [1   0]

4. 剪切变换（沿 x 轴）：
   [1  k]
   [0  1]

5. 镜像（关于 y 轴）：
   [-1  0]
   [0   1]

6. 投影（到 x 轴）：
   [1  0]
   [0  0]
```

### 变换可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_transform(matrix, title, xlim=(-2, 2), ylim=(-2, 2)):
    """可视化矩阵变换"""
    # 原始单位正方形
    square = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]
    ])
    
    # 变换后
    transformed = square @ matrix.T
    
    # 基向量
    basis = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
    basis_transformed = basis @ matrix.T
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 绘制原始正方形
    ax.plot(square[:, 0], square[:, 1], 'b--', alpha=0.3, label='原始')
    
    # 绘制变换后
    ax.plot(transformed[:, 0], transformed[:, 1], 'r-', linewidth=2, label='变换后')
    
    # 绘制基向量
    ax.arrow(0, 0, matrix[0, 0], matrix[1, 0], 
             head_width=0.1, fc='r', ec='r', alpha=0.5)
    ax.arrow(0, 0, matrix[0, 1], matrix[1, 1], 
             head_width=0.1, fc='g', ec='g', alpha=0.5)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.show()

# 测试各种变换
plot_transform(np.array([[1, 0], [0, 1]]), '恒等变换')
plot_transform(np.array([[2, 0], [0, 2]]), '放大 2 倍')
plot_transform(np.array([[0, -1], [1, 0]]), '旋转 90°')
plot_transform(np.array([[1, 1], [0, 1]]), '剪切变换')
plot_transform(np.array([[-1, 0], [0, 1]]), '镜像（y 轴）')
```

---

## 🔢 核心运算

### 1. 变换的合成

```python
"""
变换合成 = 矩阵乘法

先旋转 90°，再放大 2 倍：
A = 旋转矩阵 × 缩放矩阵

[0  -1]   [2  0]   [0  -2]
[1   0] × [0  2] = [2   0]

注意：矩阵乘法不满足交换律
先缩放再旋转 ≠ 先旋转再缩放
"""

import numpy as np

# 旋转 90°
R = np.array([[0, -1], [1, 0]])

# 放大 2 倍
S = np.array([[2, 0], [0, 2]])

# 先旋转后缩放
A = S @ R
print("先旋转后缩放：")
print(A)

# 先缩放后旋转
B = R @ S
print("\n先缩放后旋转：")
print(B)

# 验证：对于这个例子，结果相同（因为缩放是均匀的）
print(f"\n是否相等：{np.allclose(A, B)}")
```

### 2. 逆变换

```python
"""
逆变换 = 逆矩阵

如果 A 是变换矩阵，那么 A⁻¹ 是逆变换

性质：
A × A⁻¹ = I（单位矩阵）

几何意义：
如果 A 将 x 变到 y，那么 A⁻¹ 将 y 变回 x
"""

# 旋转 90°
R = np.array([[0, -1], [1, 0]])

# 逆矩阵（旋转 -90°）
R_inv = np.linalg.inv(R)
print("旋转 90°的逆变换：")
print(R_inv)
# 应该等于 [0, 1; -1, 0]（旋转 -90°）

# 验证
I = R @ R_inv
print(f"\nR × R⁻¹ = \n{I}")  # 应该是单位矩阵
```

### 3. 特征向量（不变方向）

```python
"""
特征向量：变换后方向不变的向量

A·v = λ·v

其中：
- v: 特征向量（方向不变）
- λ: 特征值（缩放倍数）

几何意义：
- 特征向量方向在变换下保持不变
- 特征值表示该方向被拉伸/压缩的倍数
"""

# 示例矩阵
A = np.array([[2, 0], [0, 3]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"特征值：{eigenvalues}")  # [2, 3]
print(f"特征向量：\n{eigenvectors}")

# 验证：A·v = λ·v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    λ = eigenvalues[i]
    Av = A @ v
    λv = λ * v
    print(f"\n验证 v{i+1}: A·v = {Av}, λ·v = {λv}")
    print(f"是否相等：{np.allclose(Av, λv)}")
```

---

## 🤖 AI 中的应用

### 应用 1：数据预处理中的变换

```python
"""
标准化（Standardization）是一种线性变换：

x_normalized = (x - μ) / σ

可以写成矩阵形式：
x_normalized = (1/σ)·x - μ/σ

这是缩放 + 平移的组合
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

# 原始数据
X = np.array([
    [25, 50000],
    [35, 80000],
    [45, 120000],
    [22, 30000],
    [50, 150000]
])

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("原始数据：")
print(X)
print("\n标准化后：")
print(X_scaled)
print(f"\n均值：{X_scaled.mean(axis=0)}")  # 接近 0
print(f"标准差：{X_scaled.std(axis=0)}")  # 接近 1
```

### 应用 2：PCA 降维

```python
"""
PCA 本质：
1. 计算协方差矩阵的特征向量
2. 选择最大的 k 个特征向量
3. 投影到这些特征向量张成的空间

这是线性变换的典型应用
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 生成 2D 数据
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 0] = X[:, 0] * 2 + X[:, 1] * 0.5  # 制造相关性

# PCA 降维到 1D
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

print(f"主成分方向：{pca.components_}")
print(f"解释方差：{pca.explained_variance_ratio_}")

# 可视化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('原始 2D 数据')
plt.xlabel('特征 1')
plt.ylabel('特征 2')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], np.zeros_like(X_pca), alpha=0.5)
plt.title(f'PCA 后 1D 数据\n(保留{pca.explained_variance_ratio_[0]*100:.1f}%方差)')
plt.xlabel('主成分 1')

plt.tight_layout()
plt.show()
```

### 应用 3：神经网络层

```python
"""
神经网络的全连接层是线性变换 + 非线性激活：

y = f(W·x + b)

其中：
- W·x: 线性变换（旋转 + 缩放）
- b: 平移（仿射变换）
- f: 非线性激活（如 ReLU）

多层神经网络 = 多个线性变换的复合 + 非线性
"""

import torch
import torch.nn as nn

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 每个 Linear 层都是一个线性变换
        self.layer1 = nn.Linear(10, 5)   # 10D → 5D
        self.layer2 = nn.Linear(5, 3)    # 5D → 3D
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)  # 第一次线性变换
        x = self.relu(x)     # 非线性
        x = self.layer2(x)   # 第二次线性变换
        return x

# 测试
model = SimpleNet()
x = torch.randn(4, 10)  # batch=4, 输入 10 维
output = model(x)

print(f"输入形状：{x.shape}")
print(f"输出形状：{output.shape}")

# 查看变换矩阵（权重）
print(f"\n第一层权重形状：{model.layer1.weight.shape}")  # (5, 10)
print(f"第二层权重形状：{model.layer2.weight.shape}")  # (3, 5)
```

---

## 💻 代码实践

### 练习 1：实现 2D 变换

```python
import numpy as np
import matplotlib.pyplot as plt

def create_transform_matrix(type='rotation', **kwargs):
    """创建变换矩阵"""
    if type == 'rotation':
        angle = kwargs.get('angle', 0)  # 角度
        theta = np.radians(angle)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    
    elif type == 'scaling':
        sx = kwargs.get('sx', 1)
        sy = kwargs.get('sy', 1)
        return np.array([[sx, 0], [0, sy]])
    
    elif type == 'shear':
        k = kwargs.get('k', 0)
        return np.array([[1, k], [0, 1]])
    
    elif type == 'reflection':
        axis = kwargs.get('axis', 'y')
        if axis == 'y':
            return np.array([[-1, 0], [0, 1]])
        else:
            return np.array([[1, 0], [0, -1]])

# 测试
R = create_transform_matrix('rotation', angle=45)
S = create_transform_matrix('scaling', sx=2, sy=0.5)
H = create_transform_matrix('shear', k=0.5)

print("旋转 45°:")
print(R)
print("\n缩放 (2, 0.5):")
print(S)
print("\n剪切 (k=0.5):")
print(H)
```

### 练习 2：变换合成动画

```python
def animate_transform(matrix_sequence, points, interval=500):
    """动画展示变换序列"""
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 初始点
    scatter = ax.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.5)
    
    def update(frame):
        matrix = matrix_sequence[frame]
        transformed = points @ matrix.T
        scatter.set_offsets(transformed)
        ax.set_title(f'Frame {frame}')
        return scatter,
    
    anim = FuncAnimation(fig, update, frames=len(matrix_sequence), 
                         interval=interval, blit=True)
    plt.show()

# 创建旋转变换序列
angles = np.linspace(0, 360, 20)
matrices = [create_transform_matrix('rotation', angle=a) for a in angles]

# 测试点
points = np.array([
    [0, 0], [1, 0], [1, 1], [0, 1],
    [0.5, 0.5], [1.5, 0.5], [1.5, 1.5]
])

# 运行动画（Jupyter 中）
# animate_transform(matrices, points)
```

### 练习 3：特征向量可视化

```python
def visualize_eigenvectors(matrix):
    """可视化矩阵的特征向量"""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 绘制单位圆
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, 'gray', alpha=0.3)
    
    # 变换后的圆（椭圆）
    transformed = np.column_stack([circle_x, circle_y]) @ matrix.T
    ax.plot(transformed[:, 0], transformed[:, 1], 'r-', alpha=0.5)
    
    # 绘制特征向量
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        λ = eigenvalues[i]
        ax.arrow(0, 0, v[0]*2, v[1]*2, 
                 head_width=0.1, fc='blue', ec='blue', 
                 label=f'v{i+1} (λ={λ:.2f})' if i==0 else f'v{i+1} (λ={λ:.2f})')
        # 变换后的特征向量
        ax.arrow(0, 0, λ*v[0]*2, λ*v[1]*2, 
                 head_width=0.1, fc='red', ec='red', alpha=0.5)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('特征向量：变换后方向不变')
    ax.set_aspect('equal')
    plt.show()

# 测试
A = np.array([[2, 1], [1, 2]])
visualize_eigenvectors(A)
```

---

## 🧠 思考题

1. **基础题**：写出旋转 180°、缩放 (3, 0.5)、剪切 (k=2) 的变换矩阵。

2. **理解题**：为什么线性变换必须保持原点不变？平移是线性变换吗？

3. **应用题**：如果一个 2×2 矩阵的行列式为 0，它表示什么几何变换？

4. **挑战题**：证明两个线性变换的合成仍然是线性变换。

---

## 📝 关键公式总结

```
1. 线性变换定义：
   T(u + v) = T(u) + T(v)
   T(cu) = c·T(u)

2. 矩阵表示：
   y = A·x

3. 旋转变换（2D）：
   [cosθ  -sinθ]
   [sinθ   cosθ]

4. 特征值方程：
   A·v = λ·v

5. 逆变换：
   A × A⁻¹ = I
```

---

## ✅ 今日检查清单

- [ ] 理解线性变换的定义和性质
- [ ] 掌握常见变换矩阵（旋转、缩放、剪切）
- [ ] 能够可视化变换效果
- [ ] 理解特征向量的几何意义
- [ ] 完成所有代码练习
- [ ] 回答所有思考题

---

> _矩阵是变换的语言，特征向量是变换的不变本质。_
> 
> _—— 悟空_
