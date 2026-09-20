# Softmax 详解：从 Logits 到概率

_为什么需要归一化？如何统一转为概率？完整数学推导与直观解释_

---

## 📖 目录

- [1. 什么是 Logits](#1-什么是-logits)
- [2. 为什么需要归一化](#2-为什么需要归一化)
- [3. Softmax 公式详解](#3-softmax-公式详解)
- [4. 逐步计算示例](#4-逐步计算示例)
- [5. 为什么用指数函数](#5-为什么用指数函数)
- [6. 数学性质证明](#6-数学性质证明)
- [7. 几何直观](#7-几何直观)
- [8. 在 LLM 中的应用](#8-在-llm-中的应用)
- [9. 代码实现](#9-代码实现)
- [10. 常见问题](#10-常见问题)

---

## 1. 什么是 Logits

### 1.1 定义

**Logits = 模型的原始输出（未归一化的分数）**

在神经网络中，logits 是最后一层线性层的输出，在应用激活函数之前的值。

### 1.2 数学表示

<div class="formula-box">

```
对于分类问题：
logits = W · x + b

其中：
- W：权重矩阵 (类别数 × 特征数)
- x：输入特征向量
- b：偏置向量
- logits：原始分数向量
```

</div>

### 1.3 问题示例

<div class="formula-box">

```
三分类问题（猫、狗、鸟）：

模型输出 logits: [2.0, 0.5, -1.0]
                  ↑    ↑     ↑
                 猫    狗    鸟

问题：
❌ 和不为 1: 2.0 + 0.5 + (-1.0) = 1.5 ≠ 1
❌ 有负数：-1.0 无法解释为概率
❌ 范围任意：可以是任何实数
```

</div>

### 1.4 为什么叫 Logits？

"Logit" = "Log-odds"（对数几率）

<div class="formula-box">

```
logit(p) = log(p / (1-p))

反过来：
p = 1 / (1 + exp(-logit)) = sigmoid(logit)
```

</div>

---

## 2. 为什么需要归一化

### 2.1 概率的公理化定义

根据柯尔莫哥洛夫概率公理：

<div class="formula-box">

```
概率必须满足：
1. 非负性：P(A) ≥ 0
2. 规范性：P(Ω) = 1（所有可能事件的概率和为 1）
3. 可加性：互斥事件的概率可加
```

</div>

### 2.2 Logits 不满足概率定义

<div class="formula-box">

```
例子：logits = [2.0, 0.5, -1.0]

检查：
1. 非负性 ❌：-1.0 < 0
2. 规范性 ❌：2.0 + 0.5 + (-1.0) = 1.5 ≠ 1
3. 可解释性 ❌：无法说"鸟的概率是 -1.0"
```

</div>

### 2.3 为什么需要概率解释？

#### 原因 1：不确定性量化

<div class="formula-box">

```
模型预测：
- 猫：78.4%  ← 知道模型有多确定
- 狗：17.5%
- 鸟：3.9%

而不是：
- 猫：2.0  ← 这个数字什么意思？
- 狗：0.5
- 鸟：-1.0
```

</div>

#### 原因 2：损失函数需要

<div class="formula-box">

```
交叉熵损失：
L = -Σ y_true × log(y_pred)

要求 y_pred 必须是概率分布！
```

</div>

#### 原因 3：采样需要

<div class="formula-box">

```
从分布中采样：
- 概率分布：可以按概率采样
- 原始 logits：无法直接采样
```

</div>

---

## 3. Softmax 公式详解

### 3.1 公式定义

<div class="formula-box">

```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)

其中：
- zᵢ：第 i 个 logit
- exp(zᵢ)：e 的 zᵢ 次方
- Σⱼ exp(zⱼ)：所有 logits 的指数和
- i：当前类别索引
- j：遍历所有类别
```

</div>

### 3.2 公式拆解

<div class="formula-box">

```
步骤 1：指数化
将每个 logit 转为正数：exp(zᵢ)

步骤 2：求和
计算所有指数的和：S = Σⱼ exp(zⱼ)

步骤 3：归一化
每个指数除以总和：exp(zᵢ) / S
```

</div>

### 3.3 为什么这样设计？

#### 设计目标

| 目标 | 解决方案 |
|------|---------|
| 转为正数 | exp() 函数（永远>0） |
| 和为 1 | 除以总和（归一化） |
| 保持排序 | exp() 单调递增 |
| 可微 | exp() 和除法都可微 |

---

## 4. 逐步计算示例

### 4.1 示例数据

<div class="formula-box">

```
输入 logits: z = [2.0, 0.5, -1.0]
类别：[猫，狗，鸟]
```

</div>

### 4.2 步骤 1：计算指数

<div class="formula-box">

```
exp(2.0)   = e^2.0   ≈ 7.389
exp(0.5)   = e^0.5   ≈ 1.649
exp(-1.0)  = e^-1.0  ≈ 0.368

结果：[7.389, 1.649, 0.368]
```

</div>

### 4.3 步骤 2：计算总和

<div class="formula-box">

```
S = 7.389 + 1.649 + 0.368 = 9.406
```

</div>

### 4.4 步骤 3：归一化

<div class="formula-box">

```
P(猫) = 7.389 / 9.406 ≈ 0.7856 (78.56%)
P(狗) = 1.649 / 9.406 ≈ 0.1753 (17.53%)
P(鸟) = 0.368 / 9.406 ≈ 0.0391 (3.91%)

验证：0.7856 + 0.1753 + 0.0391 = 1.0000 ✓
```

</div>

### 4.5 可视化对比

<div class="formula-box">

```
原始 logits:
猫 ████████████████████ 2.0
狗 █████ 0.5
鸟 █ -1.0

Softmax 后:
猫 ████████████████████████████████ 0.786 (78.6%)
狗 ███████ 0.175 (17.5%)
鸟 ██ 0.039 (3.9%)
```

</div>

---

## 5. 为什么用指数函数

### 5.1 替代方案对比

#### 方案 1：直接归一化（❌ 失败）

<div class="formula-box">

```
logits = [2.0, 0.5, -1.0]
总和 = 1.5

归一化：[2.0/1.5, 0.5/1.5, -1.0/1.5]
      = [1.33, 0.33, -0.67]
      
问题：
❌ 有负数
❌ 有大于 1 的数
❌ 不满足概率定义
```

</div>

#### 方案 2：绝对值归一化（❌ 失败）

<div class="formula-box">

```
logits = [2.0, 0.5, -1.0]
绝对值：[2.0, 0.5, 1.0]
总和 = 3.5

归一化：[0.57, 0.14, 0.29]

问题：
❌ 改变了相对大小（鸟>狗）
❌ 丢失了负号信息
```

</div>

#### 方案 3：平方归一化（❌ 失败）

<div class="formula-box">

```
logits² = [4.0, 0.25, 1.0]
总和 = 5.25

归一化：[0.76, 0.05, 0.19]

问题：
❌ 负数变正数后无法区分
❌ 放大程度不够灵活
```

</div>

### 5.2 指数函数的优势

#### 优势 1：永远为正

<div class="formula-box">

```
exp(x) > 0 对所有实数 x

无论 logit 是正是负，exp 后都是正数 ✓
```

</div>

#### 优势 2：单调递增

<div class="formula-box">

```
如果 a > b，则 exp(a) > exp(b)

保持原始排序：
logits: 2.0 > 0.5 > -1.0
exp 后：7.39 > 1.65 > 0.37 ✓
```

</div>

#### 优势 3：放大差异

<div class="formula-box">

```
原始差距：
2.0 - (-1.0) = 3.0

exp 后差距：
7.39 / 0.37 ≈ 20 倍！

→ 让模型对预测更"自信"
```

</div>

#### 优势 4：可微

<div class="formula-box">

```
d/dx exp(x) = exp(x)

处处可导，支持反向传播 ✓
```

</div>

### 5.3 指数放大的效果

<div class="formula-box">

```
不同 x 值的 exp(x)：

x = -2: exp(-2) = 0.135
x = -1: exp(-1) = 0.368
x =  0: exp(0)  = 1.000
x =  1: exp(1)  = 2.718
x =  2: exp(2)  = 7.389
x =  3: exp(3)  = 20.086

→ 线性增长 → 指数增长！
```

</div>

---

## 6. 数学性质证明

### 6.1 性质 1：输出和为 1

<div class="formula-box">

```
证明：
Σᵢ softmax(z)ᵢ = Σᵢ [exp(zᵢ) / Σⱼ exp(zⱼ)]
              = [Σᵢ exp(zᵢ)] / [Σⱼ exp(zⱼ)]
              = 1 ✓
```

</div>

### 6.2 性质 2：输出非负

<div class="formula-box">

```
证明：
exp(zᵢ) > 0 对所有实数 zᵢ
Σⱼ exp(zⱼ) > 0

因此：softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ) > 0 ✓
```

</div>

### 6.3 性质 3：保持排序

<div class="formula-box">

```
证明：
如果 zᵢ > zⱼ，则 exp(zᵢ) > exp(zⱼ)（exp 单调递增）

因此：exp(zᵢ)/S > exp(zⱼ)/S

即：softmax(z)ᵢ > softmax(z)ⱼ ✓
```

</div>

### 6.4 性质 4：可微性

<div class="formula-box">

```
Softmax 的导数：
∂softmax(z)ᵢ/∂zⱼ = softmax(z)ᵢ × (δᵢⱼ - softmax(z)ⱼ)

其中 δᵢⱼ 是 Kronecker delta：
δᵢⱼ = 1 如果 i=j，否则 0

写成矩阵形式：
∂s/∂z = diag(s) - s·sᵀ

其中 s = softmax(z)
```

</div>

### 6.5 梯度计算示例

<div class="formula-box">

```
对于损失 L，梯度计算：

∂L/∂zᵢ = softmax(z)ᵢ - yᵢ

其中 yᵢ 是真实标签（one-hot 编码）

例子：
预测：[0.786, 0.175, 0.039]
真实：[1.0, 0.0, 0.0]（猫）

梯度：[0.786-1.0, 0.175-0.0, 0.039-0.0]
     = [-0.214, 0.175, 0.039]
```

</div>

---

## 7. 几何直观

### 7.1 单纯形（Simplex）可视化

<div class="formula-box">

```
概率分布空间是一个单纯形：

二维（3 类别）：
        猫 (1,0,0)
         /\
        /  \
       /    \
      /______\
   鸟 (0,0,1)  狗 (0,1,0)

任何概率分布都是这个三角形内的点
```

</div>

### 7.2 Softmax 映射

<div class="formula-box">

```
Logits 空间（整个 R³）
        ↓ softmax
概率单纯形（三角形表面）

每个 logits 向量映射到单纯形上的一个点
```

</div>

### 7.3 温度的影响

<div class="formula-box">

```
带温度的 softmax：
softmax(z/T)ᵢ = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)

T > 1：分布更平坦（更不确定）
T < 1：分布更尖锐（更确定）
T = 1：标准 softmax

例子 logits = [2.0, 0.5, -1.0]：

T=0.5: [0.93, 0.06, 0.01]  ← 更尖锐
T=1.0: [0.79, 0.18, 0.04]  ← 标准
T=2.0: [0.62, 0.29, 0.09]  ← 更平坦
```

</div>

---

## 8. 在 LLM 中的应用

### 8.1 语言模型预测流程

<div class="formula-box">

```
1. 输入文本："今天天气真"
        ↓
2. Embedding：转为向量
        ↓
3. Transformer 层：处理上下文
        ↓
4. 线性层：输出 logits（词汇表大小）
        ↓
5. Softmax：转为概率分布
        ↓
6. 采样/贪心：选择下一个词
```

</div>

### 8.2 实际示例

<div class="formula-box">

```
词汇表：["好"，"不错"，"糟糕"，"冷"，...]

logits: [3.5, 2.0, -1.0, 0.5, ...]
        ↑
      "好"

Softmax 后：
P("好") = 0.80
P("不错") = 0.15
P("糟糕") = 0.01
P("冷") = 0.04

预测："好"（概率最高）
```

</div>

### 8.3 为什么需要概率？

#### 原因 1：多样性采样

<div class="formula-box">

```
如果总是选概率最高的：
→ 输出 deterministic，缺乏多样性

按概率采样：
→ 有时选"不错"，有时选"好"
→ 更自然、更多样
```

</div>

#### 原因 2：不确定性估计

<div class="formula-box">

```
高置信度：
[0.95, 0.03, 0.02] → 模型很确定

低置信度：
[0.34, 0.33, 0.33] → 模型不确定
```

</div>

#### 原因 3：束搜索（Beam Search）

<div class="formula-box">

```
保留多个候选序列：
序列 1：P = 0.8 × 0.7 × 0.9 = 0.504
序列 2：P = 0.6 × 0.8 × 0.8 = 0.384
序列 3：P = 0.5 × 0.6 × 0.7 = 0.210

选择概率最高的序列
```

</div>

---

## 9. 代码实现

### 9.1 基础版本

```python
import numpy as np

def softmax(logits):
    """基础 softmax 实现"""
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

# 测试
logits = np.array([2.0, 0.5, -1.0])
probs = softmax(logits)
print(f"概率：{probs}")
print(f"和：{np.sum(probs)}")
```

### 9.2 数值稳定版本（推荐）

```python
def softmax_stable(logits):
    """数值稳定的 softmax 实现"""
    # 减去最大值防止 exp 溢出
    z = logits - np.max(logits)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

# 测试大数值
logits_large = np.array([1000, 500, -100])
probs = softmax_stable(logits_large)
print(f"概率：{probs}")  # 不会溢出
```

### 9.3 PyTorch 版本

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 0.5, -1.0])
probs = F.softmax(logits, dim=0)
print(probs)
```

### 9.4 带温度参数版本

```python
def softmax_with_temperature(logits, temperature=1.0):
    """带温度的 softmax"""
    z = logits / temperature
    z = z - np.max(z)  # 数值稳定
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

# 测试不同温度
logits = np.array([2.0, 0.5, -1.0])
print(softmax_with_temperature(logits, T=0.5))  # 更尖锐
print(softmax_with_temperature(logits, T=1.0))  # 标准
print(softmax_with_temperature(logits, T=2.0))  # 更平坦
```

---

## 10. 常见问题

### Q1: 为什么 softmax 输出和一定是 1？

<div class="formula-box">

```
因为每个输出都除以了相同的总和：

softmax(z)ᵢ = exp(zᵢ) / S，其中 S = Σⱼ exp(zⱼ)

所以：Σᵢ softmax(z)ᵢ = Σᵢ [exp(zᵢ)/S] = S/S = 1 ✓
```

</div>

### Q2: 如果两个 logits 相同怎么办？

<div class="formula-box">

```
logits = [2.0, 2.0, 2.0]

softmax = [1/3, 1/3, 1/3] = [0.333, 0.333, 0.333]

→ 均匀分布，表示模型无法区分
```

</div>

### Q3: Softmax 会输出 0 或 1 吗？

<div class="formula-box">

```
理论上：不会（exp(x) > 0 对所有 x）

实际上：
- 非常大的正数 → 接近 1
- 非常小的负数 → 接近 0

例子：
logits = [100, 0, 0]
softmax ≈ [1.0, 0.0, 0.0]（数值上）
```

</div>

### Q4: 为什么不用 sigmoid 做多分类？

<div class="formula-box">

```
Sigmoid：每个类别独立，和不一定为 1

Softmax：所有类别竞争，和一定为 1

多分类需要互斥 → 用 Softmax
多标签可以共存 → 用 Sigmoid
```

</div>

### Q5: Logits 的范围是多少？

<div class="formula-box">

```
Logits 可以是任何实数：(-∞, +∞)

典型范围：-10 到 +10

过大可能导致：
- 数值溢出（用稳定版本解决）
- 梯度消失（用温度参数调节）
```

</div>

### Q6: Softmax 的梯度什么时候为 0？

<div class="formula-box">

```
当某个类别的概率接近 1 时：

softmax = [0.999, 0.001, 0.000]

梯度 ≈ 0

→ 模型已经非常确定
→ 学习会很慢
```

</div>

---

## 📚 总结

### 核心要点

| 问题 | 答案 |
|------|------|
| **什么是 logits** | 模型原始输出，未归一化的分数 |
| **为什么需要归一化** | 转为概率分布（和为 1，非负） |
| **如何归一化** | exp() 转正面 → 除以总和 |
| **为什么用 exp** | 永远为正、单调递增、放大差异、可微 |
| **在 LLM 中作用** | 将输出转为概率，支持采样和损失计算 |

### 一句话总结

> **Softmax = exp() 归一化函数**
> 
> **输入：任意实数向量（logits）**
> 
> **输出：概率分布（和为 1）**

---

*最后更新：2026-04-22*
