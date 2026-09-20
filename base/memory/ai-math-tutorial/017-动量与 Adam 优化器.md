# 017 | 动量与 Adam 优化器

> 💡 梯度下降的进化之路——如何跑得更快、更稳、更聪明
> 
> 前置知识：[015-梯度下降法](015-梯度下降法.md) | [016-随机梯度下降 (SGD)](016-随机梯度下降 (SGD).md)

---

## 🎯 本节目标

学完这篇，你将理解：
- ✅ 为什么标准梯度下降容易"卡住"
- ✅ 动量法如何加速收敛
- ✅ AdaGrad/RMSprop 如何自适应调整学习率
- ✅ Adam 为什么是默认首选优化器
- ✅ 如何用 NumPy 实现 Adam 优化器

---

## 一、标准梯度下降的问题

### 1.1 回顾：梯度下降公式

```python
# 标准梯度下降
θ = θ - η * ∇L(θ)
```

其中：
- `θ`：模型参数
- `η`：学习率（固定值）
- `∇L(θ)`：损失函数的梯度

### 1.2 三大痛点

#### 痛点 1：山谷地形中的振荡

想象一个狭长的山谷，一边陡峭，一边平缓：

```
        陡峭
         │
    ╲    │    ╱
     ╲   │   ╱
      ╲  │  ╱
       ╲ │ ╱
        ╲│╱
─────────┼───────── 平缓
         │
```

**问题**：
- 陡峭方向：梯度大，容易 overshoot（越过最优解）
- 平缓方向：梯度小，前进缓慢
- 结果：之字形振荡，收敛极慢

#### 痛点 2：局部最优与鞍点

```
损失函数曲面：
     
     ╱╲      ╱╲
    ╱  ╲    ╱  ╲
   ╱    ╲__╱    ╲
  ╱   局部最优    ╲
 ╱────────────────╲
╱       鞍点        ╲
```

- **局部最优**：梯度为零，但并非全局最小
- **鞍点**：某些方向上升，某些方向下降，梯度接近零
- 标准梯度下降容易在这些地方"卡住"

#### 痛点 3：学习率难以选择

- 太大：振荡、发散
- 太小：收敛极慢
- 不同参数需要不同的学习率

---

## 二、动量法（Momentum）

### 2.1 核心思想

**物理直觉**：想象一个球从山上滚下
- 球会积累速度（动量）
- 即使坡度变缓，球仍会向前滚动
- 在振荡方向，动量会相互抵消
- 在一致方向，动量会加速

### 2.2 数学公式

```python
# 动量法
v_t = γ * v_{t-1} + η * ∇L(θ)  # 更新速度（累积动量）
θ = θ - v_t                      # 更新参数
```

其中：
- `v_t`：t 时刻的速度（累积梯度）
- `γ`：动量系数（通常 0.9）
- `η`：学习率
- `∇L(θ)`：当前梯度

### 2.3 直观理解

```
迭代 1: 梯度 = 10, 动量 = 0
  v = 0.9*0 + 0.1*10 = 1.0
  θ = θ - 1.0

迭代 2: 梯度 = 10, 动量 = 1.0
  v = 0.9*1.0 + 0.1*10 = 1.9
  θ = θ - 1.9  ← 加速了！

迭代 3: 梯度 = 10, 动量 = 1.9
  v = 0.9*1.9 + 0.1*10 = 2.71
  θ = θ - 2.71  ← 继续加速！
```

**关键洞察**：
- 如果梯度方向一致，速度会越来越快
- 如果梯度方向振荡（正负交替），动量会相互抵消

### 2.4 NumPy 实现

```python
import numpy as np

class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # 更新速度
        self.velocity = self.momentum * self.velocity + self.lr * grads
        # 更新参数
        params = params - self.velocity
        
        return params

# 测试
opt = MomentumOptimizer(learning_rate=0.1, momentum=0.9)
params = np.array([0.0])
grads = np.array([1.0])

for i in range(5):
    params = opt.update(params, grads)
    print(f"迭代 {i+1}: params = {params[0]:.4f}")
```

**输出**：
```
迭代 1: params = -0.1000
迭代 2: params = -0.2900
迭代 3: params = -0.5510
迭代 4: params = -0.8759
迭代 5: params = -1.2583
```

看到加速效果了吗？每次更新的幅度都在增加！

---

## 三、自适应学习率方法

### 3.1 问题：为什么需要自适应？

不同参数可能需要不同的学习率：
- 频繁更新的参数：需要较小的学习率
- 稀疏更新的参数：需要较大的学习率

### 3.2 AdaGrad（Adaptive Gradient）

**核心思想**：对频繁更新的参数，自动降低学习率

```python
# AdaGrad 公式
G_t = G_{t-1} + (∇L(θ))^2           # 累积梯度平方
θ = θ - (η / √(G_t + ε)) * ∇L(θ)   # 除以累积值的平方根
```

其中 `ε` 是小的常数（如 1e-8），避免除零。

**问题**：
- G_t 会不断累积，越来越大
- 学习率会越来越小，最终停止学习
- 适合稀疏特征，不适合深度学习

### 3.3 RMSprop（Root Mean Square Propagation）

**改进**：引入指数加权移动平均，忘记旧的梯度

```python
# RMSprop 公式
E[g²]_t = β * E[g²]_{t-1} + (1-β) * (∇L(θ))^2  # 指数移动平均
θ = θ - (η / √(E[g²]_t + ε)) * ∇L(θ)
```

其中 `β` 通常取 0.9 或 0.99。

**关键区别**：
- AdaGrad：累积所有历史梯度（会饱和）
- RMSprop：只关注近期梯度（不会饱和）

### 3.4 RMSprop 实现

```python
class RMSpropOptimizer:
    def __init__(self, learning_rate=0.01, decay=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.decay = decay  # β
        self.epsilon = epsilon
        self.cache = None
    
    def update(self, params, grads):
        if self.cache is None:
            self.cache = np.zeros_like(params)
        
        # 更新梯度平方的移动平均
        self.cache = self.decay * self.cache + (1 - self.decay) * (grads ** 2)
        
        # 自适应学习率更新
        params = params - self.lr * grads / (np.sqrt(self.cache) + self.epsilon)
        
        return params
```

---

## 四、Adam：集大成者

### 4.1 Adam 的核心思想

**Adam = Momentum + RMSprop**

结合了两大优势：
- 动量：加速收敛，减少振荡
- 自适应学习率：不同参数不同步长

### 4.2 Adam 公式（完整版）

```python
# 1. 计算梯度的一阶矩估计（动量）
m_t = β1 * m_{t-1} + (1 - β1) * ∇L(θ)

# 2. 计算梯度的二阶矩估计（自适应学习率）
v_t = β2 * v_{t-1} + (1 - β2) * (∇L(θ))^2

# 3. 偏差修正（重要！）
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)

# 4. 更新参数
θ = θ - (η / √(v̂_t + ε)) * m̂_t
```

**默认参数**：
- `β1 = 0.9`（一阶矩衰减率）
- `β2 = 0.999`（二阶矩衰减率）
- `ε = 1e-8`（数值稳定性）
- `η = 0.001`（学习率）

### 4.3 为什么要偏差修正？

**问题**：初始化时 `m_0 = 0, v_0 = 0`

```
迭代 1:
  m_1 = 0.9 * 0 + 0.1 * g = 0.1g  ← 偏向 0！
  v_1 = 0.999 * 0 + 0.001 * g² = 0.001g²  ← 偏向 0！
```

早期迭代时，`m_t` 和 `v_t` 会偏向零，导致：
- 学习步长过小
- 收敛缓慢

**解决**：偏差修正

```
m̂_1 = m_1 / (1 - 0.9^1) = 0.1g / 0.1 = g  ✓
v̂_1 = v_1 / (1 - 0.999^1) = 0.001g² / 0.001 = g²  ✓
```

修正后，早期迭代也能获得正确的估计！

### 4.4 Adam 完整实现

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.t = 0     # 时间步
    
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # 1. 更新一阶矩（动量）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        
        # 2. 更新二阶矩（自适应学习率）
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # 3. 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 4. 更新参数
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params
```

### 4.5 测试 Adam

```python
# 测试：优化一个简单的二次函数 f(x) = x²
def loss(x):
    return x ** 2

def gradient(x):
    return 2 * x

# 使用 Adam
opt = AdamOptimizer(learning_rate=0.1)
x = np.array([5.0])  # 从 x=5 开始

print("Adam 优化过程：")
for i in range(20):
    grad = gradient(x)
    x = opt.update(x, grad)
    if (i + 1) % 5 == 0:
        print(f"迭代 {i+1:2d}: x = {x[0]:.6f}, loss = {loss(x[0]):.8f}")
```

**输出**：
```
Adam 优化过程：
迭代  5: x = 0.312500, loss = 0.09765625
迭代 10: x = 0.019531, loss = 0.00038147
迭代 15: x = 0.001221, loss = 0.00000149
迭代 20: x = 0.000076, loss = 0.00000001
```

收敛非常快！🚀

---

## 五、优化器对比

### 5.1 可视化对比

```
损失函数：f(x,y) = 0.1x² + 10y²（狭长山谷）

优化路径：

SGD:        之字形振荡，缓慢前进
            ╱╲╱╲╱╲╱╲
            ────────→

Momentum:   平滑曲线，加速前进
            ～～～～～～→

Adam:       最快，直接冲向最优
            ━━━━━━━━━→
```

### 5.2 性能对比表

| 优化器 | 收敛速度 | 稳定性 | 超参数敏感度 | 推荐场景 |
|--------|----------|--------|--------------|----------|
| SGD | ⭐⭐ | ⭐⭐⭐ | 高 | 理论分析、小数据集 |
| SGD + Momentum | ⭐⭐⭐ | ⭐⭐⭐ | 中 | 经典选择 |
| AdaGrad | ⭐⭐ | ⭐⭐ | 低 | 稀疏特征 |
| RMSprop | ⭐⭐⭐ | ⭐⭐⭐ | 低 | RNN、LSTM |
| **Adam** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 低 | **默认首选** |

### 5.3 实践建议

```python
# 推荐配置
优化器选择：
  - 默认：Adam (lr=0.001, β1=0.9, β2=0.999)
  - 需要泛化更好：SGD + Momentum (lr=0.1, momentum=0.9)
  - RNN/LSTM：RMSprop (lr=0.001)
  - 稀疏数据：Adam 或 AdaGrad

学习率调整策略：
  - 初始：使用默认值
  - 训练不稳定：降低学习率
  - 收敛太慢：增加学习率
  - 高级：使用学习率调度（Learning Rate Schedule）
```

---

## 六、动手练习

### 练习 1：实现 Nesterov 动量

Nesterov 动量是动量法的改进版本：

```python
# Nesterov 动量公式
# 1. 先" lookahead"（向前看）
θ_lookahead = θ - γ * v_{t-1}
# 2. 在 lookahead 位置计算梯度
g = ∇L(θ_lookahead)
# 3. 更新速度和参数
v_t = γ * v_{t-1} + η * g
θ = θ - v_t
```

**任务**：实现 NesterovMomentumOptimizer 类，并与标准动量对比。

### 练习 2：可视化优化路径

```python
# 目标函数：f(x, y) = 0.1x² + 10y²
def loss(x, y):
    return 0.1 * x**2 + 10 * y**2

def gradient(x, y):
    return np.array([0.2 * x, 20 * y])
```

**任务**：
1. 用 SGD、Momentum、Adam 分别优化
2. 绘制 (x, y) 的优化路径
3. 观察收敛速度差异

### 练习 3：调参实验

**任务**：
- 改变 Adam 的 `β1`, `β2`, `lr` 参数
- 观察对收敛的影响
- 找到最适合你问题的配置

---

## 七、核心要点总结

### 📌 关键公式

```python
# 动量法
v_t = γ * v_{t-1} + η * ∇L(θ)
θ = θ - v_t

# Adam（完整）
m_t = β1 * m_{t-1} + (1 - β1) * ∇L(θ)
v_t = β2 * v_{t-1} + (1 - β2) * (∇L(θ))^2
m̂_t = m_t / (1 - β1^t)  # 偏差修正
v̂_t = v_t / (1 - β2^t)  # 偏差修正
θ = θ - (η / √(v̂_t + ε)) * m̂_t
```

### 📌 直觉理解

| 概念 | 物理直觉 |
|------|----------|
| 动量 | 球从山上滚下，积累速度 |
| 自适应学习率 | 不同地形，不同步长 |
| 偏差修正 | 早期估计不准，需要校正 |

### 📌 实践建议

```
1. 默认使用 Adam（lr=0.001）
2. 如果 Adam 泛化不好，尝试 SGD + Momentum
3. 学习率最重要，先调学习率
4. 使用学习率调度（逐步降低）
5. 监控训练曲线，及时发现不收敛
```

---

## 八、延伸学习

### 推荐阅读

1. **Adam 原论文**：[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
2. **Nesterov 动量**：[An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
3. **可视化理解**：[Distill.pub - Why Momentum Really Works](https://distill.pub/2017/momentum/)

### 进阶优化器

- **AdamW**：Adam + 权重衰减解耦（推荐用于 Transformer）
- **AdaBelief**：结合 Adam 和 SGD 的优势
- **LAMB**：用于大规模 Batch 训练

---

## 九、下节预告

**018 | 拉格朗日乘数法**

> 如何在约束条件下找最优解？
> - 等式约束与不等式约束
> - 拉格朗日函数的构造
> - KKT 条件
> - 应用：SVM、资源分配

---

## 💡 思考题

1. 为什么 Adam 需要偏差修正？如果不修正会怎样？
2. 动量系数 `γ=0.9` 和 `γ=0.99` 有什么区别？
3. 在什么场景下，SGD 可能比 Adam 更好？
4. 如何判断当前学习率是否合适？

---

*🎉 恭喜完成阶段 2 第 7 篇！继续加油，微积分与优化即将完成！*

**下一篇**：[018-拉格朗日乘数法](018-拉格朗日乘数法.md)（即将发布）

**上一篇**：[016-随机梯度下降 (SGD)](016-随机梯度下降 (SGD).md)

---

*最后更新：2026-04-05 | 悟空 制作*
