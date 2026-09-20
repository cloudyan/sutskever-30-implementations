# Day 4: 行列式

> 变换的缩放因子 —— 理解行列式的几何意义
> 
> _难度：⭐⭐⭐ | 预计时间：1-2 小时 | AI 应用：可逆性判断、体积变化、雅可比矩阵_

---

## 📌 一句话核心

**行列式表示线性变换对空间的缩放比例，为 0 时空间被压缩（不可逆）。**

---

## 🎯 核心问题

学完今天的内容，你应该能够回答：
1. 行列式的几何意义是什么？
2. 如何计算 2×2 和 3×3 矩阵的行列式？
3. 行列式为 0 意味着什么？
4. 行列式在 AI 中有什么应用？

---

## 📚 核心定义

### 什么是行列式？

```
数学定义：
行列式是一个从方阵到标量的函数，记作 det(A) 或 |A|

几何意义：
- 2D：变换后平行四边形的面积
- 3D：变换后平行六面体的体积
- nD：变换后 n 维平行体的"超体积"

符号含义：
- det(A) > 0: 保持定向（右手系→右手系）
- det(A) < 0: 反转定向（右手系→左手系）
- det(A) = 0: 空间被压缩（降维）
```

### 2×2 矩阵的行列式

```
公式：
        [a  b]
    A = [c  d]
    
    det(A) = a×d - b×c

记忆方法：
  主对角线乘积 - 副对角线乘积
     ↘           ↙
     [a  b]
     [c  d]
     ↙           ↘

示例：
    [3  1]
A = [2  4]

det(A) = 3×4 - 1×2 = 12 - 2 = 10

几何意义：
单位正方形（面积 1）经过 A 变换后，
变成面积为 10 的平行四边形
```

### 3×3 矩阵的行列式

```
公式（按第一行展开）：
        [a  b  c]
    A = [d  e  f]
        [g  h  i]
    
    det(A) = a×(e×i - f×h) - b×(d×i - f×g) + c×(d×h - e×g)

记忆方法：萨鲁斯法则（仅适用于 3×3）
  a  b  c  a  b
  d  e  f  d  e
  g  h  i  g  h
  
  主对角线：aei + bfg + cdh
  副对角线：ceg + bdi + afh
  
  det(A) = (aei + bfg + cdh) - (ceg + bdi + afh)
```

---

## 📐 几何直观

### 行列式的几何意义

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_determinant(matrix):
    """可视化行列式的几何意义"""
    # 单位正方形
    square = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]
    ])
    
    # 变换后
    transformed = square @ matrix.T
    
    # 计算行列式
    det = np.linalg.det(matrix)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 原始正方形
    ax.fill(square[:, 0], square[:, 1], 'blue', alpha=0.2, label='原始 (面积=1)')
    ax.plot(square[:, 0], square[:, 1], 'b--')
    
    # 变换后
    ax.fill(transformed[:, 0], transformed[:, 1], 'red', alpha=0.3, 
            label=f'变换后 (面积={abs(det):.2f})')
    ax.plot(transformed[:, 0], transformed[:, 1], 'r-')
    
    # 基向量
    ax.arrow(0, 0, matrix[0, 0], matrix[1, 0], 
             head_width=0.1, fc='r', ec='r', label='基向量 i')
    ax.arrow(0, 0, matrix[0, 1], matrix[1, 1], 
             head_width=0.1, fc='g', ec='g', label='基向量 j')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'行列式 = {det:.2f}\n面积缩放倍数 = {abs(det):.2f}')
    ax.set_aspect('equal')
    plt.show()

# 测试不同矩阵
print("示例 1：放大 2 倍")
A1 = np.array([[2, 0], [0, 2]])
print(f"det(A1) = {np.linalg.det(A1):.2f}")
plot_determinant(A1)

print("\n示例 2：剪切变换")
A2 = np.array([[1, 1], [0, 1]])
print(f"det(A2) = {np.linalg.det(A2):.2f}")
plot_determinant(A2)

print("\n示例 3：行列式为 0（降维）")
A3 = np.array([[2, 0], [0, 0]])
print(f"det(A3) = {np.linalg.det(A3):.2f}")
plot_determinant(A3)
```

### 行列式与可逆性

```
关键定理：
矩阵 A 可逆 ⟺ det(A) ≠ 0

几何解释：
- det(A) ≠ 0: 变换是一一映射，空间没有被压缩，可以逆变换
- det(A) = 0: 空间被压缩到低维，信息丢失，无法完全恢复

示例：
    [1  2]
A = [2  4]

det(A) = 1×4 - 2×2 = 0

这个矩阵将 2D 空间压缩到 1D 直线，无法恢复
```

---

## 🔢 核心运算

### 1. 行列式的性质

```python
import numpy as np

"""
行列式的重要性质：

1. det(I) = 1（单位矩阵）
2. det(A×B) = det(A) × det(B)
3. det(Aᵀ) = det(A)
4. det(A⁻¹) = 1/det(A)（如果 A 可逆）
5. 交换两行/列，行列式变号
6. 某行/列乘以 k，行列式乘以 k
7. 某行/列加上另一行/列的倍数，行列式不变
"""

# 验证性质
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("性质验证：")
print(f"det(A) = {np.linalg.det(A):.4f}")
print(f"det(B) = {np.linalg.det(B):.4f}")
print(f"det(A×B) = {np.linalg.det(A @ B):.4f}")
print(f"det(A)×det(B) = {np.linalg.det(A) * np.linalg.det(B):.4f}")
print(f"相等：{np.allclose(np.linalg.det(A @ B), np.linalg.det(A) * np.linalg.det(B))}")

print(f"\ndet(Aᵀ) = {np.linalg.det(A.T):.4f}")
print(f"det(A) = {np.linalg.det(A):.4f}")
print(f"相等：{np.allclose(np.linalg.det(A.T), np.linalg.det(A))}")

# 逆矩阵的行列式
A_inv = np.linalg.inv(A)
print(f"\ndet(A⁻¹) = {np.linalg.det(A_inv):.4f}")
print(f"1/det(A) = {1/np.linalg.det(A):.4f}")
print(f"相等：{np.allclose(np.linalg.det(A_inv), 1/np.linalg.det(A))}")
```

### 2. 行列式计算

```python
def det_2x2(matrix):
    """计算 2×2 矩阵的行列式"""
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    return a * d - b * c

def det_3x3(matrix):
    """计算 3×3 矩阵的行列式（按第一行展开）"""
    a, b, c = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    
    # 余子式
    M_a = matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]
    M_b = matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0]
    M_c = matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]
    
    return a * M_a - b * M_b + c * M_c

# 测试
A2 = np.array([[3, 1], [2, 4]])
print(f"2×2 行列式：{det_2x2(A2):.4f}")
print(f"NumPy 验证：{np.linalg.det(A2):.4f}")

A3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n3×3 行列式：{det_3x3(A3):.4f}")
print(f"NumPy 验证：{np.linalg.det(A3):.4f}")
```

### 3. 雅可比行列式

```python
"""
雅可比行列式（Jacobian Determinant）：

在多变量微积分和变量变换中，雅可比行列式表示
坐标变换对体积元的缩放比例

对于变换：
u = f(x, y)
v = g(x, y)

雅可比矩阵：
    [∂u/∂x  ∂u/∂y]
J = [∂v/∂x  ∂v/∂y]

雅可比行列式：det(J)

应用：
- 变量替换积分
- 概率密度变换
- 归一化流（Normalizing Flow）
"""

# 极坐标变换示例
# x = r·cos(θ), y = r·sin(θ)

def jacobian_polar(r, theta):
    """极坐标变换的雅可比矩阵"""
    # J = [∂x/∂r  ∂x/∂θ]
    #     [∂y/∂r  ∂y/∂θ]
    
    # ∂x/∂r = cos(θ), ∂x/∂θ = -r·sin(θ)
    # ∂y/∂r = sin(θ), ∂y/∂θ = r·cos(θ)
    
    J = np.array([
        [np.cos(theta), -r * np.sin(theta)],
        [np.sin(theta), r * np.cos(theta)]
    ])
    return J

# 计算雅可比行列式
r, theta = 2, np.pi/4
J = jacobian_polar(r, theta)
det_J = np.linalg.det(J)

print(f"极坐标变换 (r={r}, θ={theta:.2f})")
print(f"雅可比矩阵:\n{J}")
print(f"雅可比行列式：{det_J:.4f}")
print(f"理论值 r = {r:.4f}")
print(f"相等：{np.allclose(det_J, r)}")

# 这就是为什么极坐标积分要乘以 r：
# ∫∫f(x,y)dxdy = ∫∫f(r,θ)·r·drdθ
```

---

## 🤖 AI 中的应用

### 应用 1：判断矩阵可逆性

```python
"""
在神经网络和机器学习中，经常需要判断矩阵是否可逆

例如：
- 求解线性方程组
- 计算协方差矩阵的逆
- 归一化流中的可逆变换
"""

def is_invertible(matrix, tol=1e-10):
    """判断矩阵是否可逆"""
    det = np.linalg.det(matrix)
    return abs(det) > tol

# 示例
A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 2], [2, 4]])  # 第二行是第一行的 2 倍

print(f"A 可逆：{is_invertible(A)}")  # True
print(f"B 可逆：{is_invertible(B)}")  # False

# 如果不可逆，尝试求逆会出错或数值不稳定
try:
    A_inv = np.linalg.inv(A)
    print(f"A 的逆存在")
except:
    print(f"A 的逆不存在")

try:
    B_inv = np.linalg.inv(B)
    print(f"B 的逆存在")
except np.linalg.LinAlgError:
    print(f"B 的逆不存在（奇异矩阵）")
```

### 应用 2：归一化流（Normalizing Flow）

```python
"""
归一化流是一种生成模型，核心思想：

简单分布 → 可逆变换 → 复杂分布
  (如高斯)              (如数据分布)

关键：变量变换公式
p_Y(y) = p_X(x) · |det(J⁻¹)|

其中 J 是变换的雅可比矩阵

应用：
- 密度估计
- 生成样本
- 变分推断
"""

import torch
import torch.nn as nn
import torch.distributions as dist

class SimpleNormalizingFlow:
    """简单归一化流示例"""
    
    def __init__(self):
        # 基础分布（标准高斯）
        self.base_dist = dist.Normal(0, 1)
        
        # 可逆变换：y = x + tanh(x)
        # 这不是线性的，但可逆
    
    def transform(self, x):
        """前向变换"""
        return x + torch.tanh(x)
    
    def inverse(self, y):
        """逆变换（数值求解）"""
        # 对于简单情况可以解析求解
        # 这里用牛顿法数值求解
        x = y.clone()  # 初始猜测
        for _ in range(10):
            f = x + torch.tanh(x) - y
            df = 1 + 1/torch.cosh(x)**2
            x = x - f / df
        return x
    
    def log_det_jacobian(self, x):
        """计算雅可比行列式的对数"""
        # J = dy/dx = 1 + sech²(x)
        jacobian = 1 + 1/torch.cosh(x)**2
        return torch.log(jacobian)
    
    def log_prob(self, y):
        """计算变换后分布的对数概率"""
        x = self.inverse(y)
        log_p_x = self.base_dist.log_prob(x).sum(dim=-1)
        log_det_J = self.log_det_jacobian(x).sum(dim=-1)
        return log_p_x - log_det_J  # 变量变换公式

# 测试
flow = SimpleNormalizingFlow()
y = torch.randn(10, 2)
log_prob = flow.log_prob(y)
print(f"对数概率：{log_prob.mean():.4f}")
```

### 应用 3：协方差矩阵分析

```python
"""
协方差矩阵的行列式表示数据的"散布体积"

应用：
- 异常检测：行列式小表示数据集中
- 特征选择：行列式大表示特征冗余少
- 马氏距离：需要协方差矩阵的逆
"""

from sklearn.covariance import EmpiricalCovariance

# 生成数据
np.random.seed(42)
X1 = np.random.randn(100, 2) @ np.array([[2, 0], [0, 1]])  # 分散
X2 = np.random.randn(100, 2) @ np.array([[0.5, 0], [0, 0.5]])  # 集中

# 计算协方差矩阵
cov1 = np.cov(X1.T)
cov2 = np.cov(X2.T)

det1 = np.linalg.det(cov1)
det2 = np.linalg.det(cov2)

print(f"分散数据的协方差行列式：{det1:.4f}")
print(f"集中数据的协方差行列式：{det2:.4f}")
print(f"比值：{det1/det2:.2f}倍")

# 可视化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X1[:, 0], X1[:, 1], alpha=0.5)
plt.title(f'分散数据\ndet(Σ)={det1:.4f}')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(X2[:, 0], X2[:, 1], alpha=0.5)
plt.title(f'集中数据\ndet(Σ)={det2:.4f}')
plt.axis('equal')

plt.tight_layout()
plt.show()
```

---

## 💻 代码实践

### 练习 1：行列式计算器

```python
def det_recursive(matrix):
    """递归计算 n×n 矩阵的行列式（拉普拉斯展开）"""
    n = matrix.shape[0]
    
    # 基础情况
    if n == 1:
        return matrix[0, 0]
    if n == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    
    det = 0
    for j in range(n):
        # 余子式（去掉第 0 行和第 j 列）
        minor = np.delete(np.delete(matrix, 0, axis=0), j, axis=1)
        det += ((-1) ** j) * matrix[0, j] * det_recursive(minor)
    
    return det

# 测试
for n in range(2, 6):
    A = np.random.randn(n, n)
    det1 = det_recursive(A)
    det2 = np.linalg.det(A)
    print(f"{n}×{n}: 递归={det1:.6f}, NumPy={det2:.6f}, 相等={np.allclose(det1, det2)}")
```

### 练习 2：行列式性质验证器

```python
def verify_determinant_properties():
    """验证行列式的各种性质"""
    np.random.seed(42)
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    
    print("=== 行列式性质验证 ===\n")
    
    # 性质 1: det(I) = 1
    I = np.eye(3)
    print(f"1. det(I) = {np.linalg.det(I):.6f} (应该 = 1)")
    
    # 性质 2: det(A×B) = det(A)×det(B)
    print(f"\n2. det(A×B) = {np.linalg.det(A @ B):.6f}")
    print(f"   det(A)×det(B) = {np.linalg.det(A) * np.linalg.det(B):.6f}")
    
    # 性质 3: det(Aᵀ) = det(A)
    print(f"\n3. det(Aᵀ) = {np.linalg.det(A.T):.6f}")
    print(f"   det(A) = {np.linalg.det(A):.6f}")
    
    # 性质 4: det(A⁻¹) = 1/det(A)
    if abs(np.linalg.det(A)) > 1e-10:
        A_inv = np.linalg.inv(A)
        print(f"\n4. det(A⁻¹) = {np.linalg.det(A_inv):.6f}")
        print(f"   1/det(A) = {1/np.linalg.det(A):.6f}")
    
    # 性质 5: 交换两行，行列式变号
    A_swapped = A.copy()
    A_swapped[[0, 1]] = A_swapped[[1, 0]]
    print(f"\n5. det(A) = {np.linalg.det(A):.6f}")
    print(f"   det(A 交换行) = {np.linalg.det(A_swapped):.6f}")
    print(f"   变号：{np.allclose(np.linalg.det(A_swapped), -np.linalg.det(A))}")

verify_determinant_properties()
```

### 练习 3：线性方程组求解器

```python
def solve_linear_system(A, b):
    """
    使用行列式求解线性方程组 Ax = b（克莱姆法则）
    
    仅用于教学，实际应使用 np.linalg.solve
    """
    n = A.shape[0]
    det_A = np.linalg.det(A)
    
    if abs(det_A) < 1e-10:
        raise ValueError("矩阵不可逆，方程组无唯一解")
    
    x = np.zeros(n)
    for i in range(n):
        # 用 b 替换第 i 列
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A
    
    return x

# 测试
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 6])

x = solve_linear_system(A, b)
print(f"解：x = {x}")

# 验证
print(f"验证：Ax = {A @ x}")
print(f"应该等于 b = {b}")
print(f"相等：{np.allclose(A @ x, b)}")
```

---

## 🧠 思考题

1. **基础题**：计算以下矩阵的行列式：
   ```
   [3  1]
   [2  4]
   ```

2. **理解题**：为什么行列式为 0 的矩阵不可逆？从几何角度解释。

3. **应用题**：如果一个 3×3 矩阵的行列式为 -5，它表示什么几何变换？

4. **挑战题**：证明 det(A×B) = det(A)×det(B)。

---

## 📝 关键公式总结

```
1. 2×2 行列式：
   det([a b; c d]) = ad - bc

2. 3×3 行列式（展开）：
   det = a(ei-fh) - b(di-fg) + c(dh-eg)

3. 重要性质：
   det(A×B) = det(A)×det(B)
   det(Aᵀ) = det(A)
   det(A⁻¹) = 1/det(A)

4. 可逆性：
   A 可逆 ⟺ det(A) ≠ 0

5. 变量变换：
   p_Y(y) = p_X(x) · |det(J⁻¹)|
```

---

## ✅ 今日检查清单

- [ ] 理解行列式的几何意义
- [ ] 能够计算 2×2 和 3×3 行列式
- [ ] 掌握行列式的重要性质
- [ ] 理解行列式与可逆性的关系
- [ ] 完成所有代码练习
- [ ] 回答所有思考题

---

> _行列式是变换的体积计，为零时空间塌陷，非零时天地可逆。_
> 
> _—— 悟空_
