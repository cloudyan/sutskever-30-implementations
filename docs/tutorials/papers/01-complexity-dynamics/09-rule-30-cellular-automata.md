# 09 Rule 30 元胞自动机：复杂性的可视化

问下大家，你们见过最神奇的"简单规则产生复杂行为"的例子是什么？

我第一次看到 **Rule 30** 的时候，简直被震撼到了——就这么简单的规则，居然能产生如此复杂的图案！而且，这个复杂度还不是随机的，而是有结构的！

今天，我们就来深入理解 Rule 30，并用它来可视化"复杂动力学"。

## 什么是元胞自动机？

**元胞自动机**（Cellular Automaton, CA）是由 Stanislaw Ulam 和 John von Neumann 在 1940 年代提出的离散模型。

### 基本组成

1. **网格**：一维、二维或更高维的格子
2. **状态**：每个格子有一个状态（如 0 或 1）
3. **邻居**：每个格子有一组邻居格子
4. **规则**：根据邻居的状态决定下一时刻的状态

### 一维元胞自动机

最简单的一维 CA：
- 格子排成一条线
- 每个格子看自己和左右两个邻居（共 3 个格子）
- 根据这 3 个格子的状态决定下一时刻的状态

3 个格子，每个有 2 种状态，共有 $2^3 = 8$ 种可能的邻居配置。

对于每种配置，下一状态可以是 0 或 1，所以共有 $2^8 = 256$ 种不同的规则！

## Rule 30：简单规则的复杂行为

**Rule 30** 是 Stephen Wolfram 在 1983 年发现的一个一维元胞自动机规则。

### 规则定义

```
当前模式:  111  110  101  100  011  010  001  000
下一状态:   0    0    0    1    1    1    1    0
```

二进制 `00011110` = 十进制 **30**，所以叫 Rule 30。

### Python 实现

```python
import numpy as np
import matplotlib.pyplot as plt

def rule_30(left, center, right):
    """Rule 30 转移函数"""
    pattern = (left << 2) | (center << 1) | right
    # 00011110 是 Rule 30 的二进制表示
    return (30 >> pattern) & 1

def simulate_rule_30(width, steps):
    """
    模拟 Rule 30
    width: 网格宽度
    steps: 模拟步数
    """
    # 初始化：中间一个格子为 1，其余为 0
    grid = np.zeros((steps, width), dtype=int)
    grid[0, width // 2] = 1
    
    for t in range(steps - 1):
        for i in range(1, width - 1):
            left = grid[t, i - 1]
            center = grid[t, i]
            right = grid[t, i + 1]
            grid[t + 1, i] = rule_30(left, center, right)
    
    return grid

# 模拟
width, steps = 200, 100
grid = simulate_rule_30(width, steps)

# 可视化
plt.figure(figsize=(12, 6))
plt.imshow(grid, cmap='binary', interpolation='nearest')
plt.title('Rule 30 Cellular Automaton')
plt.xlabel('Space')
plt.ylabel('Time')
plt.show()
```

### 运行结果

你会看到一个三角形图案，从顶部的单个黑点扩展到底部的复杂图案。

**关键观察**：
1. **左侧**：看起来是随机的、不可预测的
2. **右侧**：有规则的三角形结构
3. **整体**：既不是完全有序，也不是完全随机

## 为什么 Rule 30 如此特别？

### Wolfram 的分类

Stephen Wolfram 将元胞自动机分为 4 类：

1. **Class 1**：演化到均匀状态（如 Rule 0, Rule 255）
2. **Class 2**：演化到周期性结构（如 Rule 90, Rule 250）
3. **Class 3**：产生混沌、随机图案（如 **Rule 30**）
4. **Class 4**：产生复杂、持久的结构（如 Rule 110）

**Rule 30 属于 Class 3**，但它产生的"随机性"有一些特别之处：

### 伪随机数生成器

Rule 30 的中心列被用作 **Mathematica** 软件的伪随机数生成器！

为什么？因为中心列通过了大量的随机性统计测试，尽管它是由确定性规则生成的。

```python
def extract_center_column(grid):
    """提取中心列作为伪随机序列"""
    width = grid.shape[1]
    center = width // 2
    return grid[:, center]

# 测试随机性
random_sequence = extract_center_column(grid)
print(f"0 的比例: {np.mean(random_sequence == 0):.4f}")
print(f"1 的比例: {np.mean(random_sequence == 1):.4f}")

# 计算游程（run）统计
runs = []
current_run = 1
for i in range(1, len(random_sequence)):
    if random_sequence[i] == random_sequence[i-1]:
        current_run += 1
    else:
        runs.append(current_run)
        current_run = 1
runs.append(current_run)

print(f"平均游程长度: {np.mean(runs):.2f}")
print(f"期望游程长度（随机）: 2.00")
```

## 熵与复杂度的演化

现在让我们用 Rule 30 来观察"复杂动力学"！

### 测量熵

我们可以测量每一行的香农熵：

```python
def compute_entropy(row):
    """计算一行的香农熵"""
    p1 = np.mean(row)
    p0 = 1 - p1
    
    if p0 == 0 or p1 == 0:
        return 0
    
    return -(p0 * np.log2(p0) + p1 * np.log2(p1))

# 计算每行的熵
entropies = [compute_entropy(grid[t, :]) for t in range(steps)]

plt.figure(figsize=(10, 4))
plt.plot(entropies)
plt.xlabel('Time Step')
plt.ylabel('Entropy')
plt.title('Entropy Evolution in Rule 30')
plt.grid(True)
plt.show()
```

**观察**：熵会快速增加，然后趋于稳定在接近 1（最大熵）的水平。

### 近似 Kolmogorov 复杂度

用压缩比来近似 KC：

```python
import zlib

def approximate_kc(row):
    """用压缩长度近似 KC"""
    row_bytes = ''.join(map(str, row)).encode()
    compressed = zlib.compress(row_bytes)
    return len(compressed)

# 计算每行的近似 KC
kcs = [approximate_kc(grid[t, :]) for t in range(steps)]

# 归一化
kcs = np.array(kcs) / max(kcs)

plt.figure(figsize=(10, 4))
plt.plot(kcs, label='Approximate KC')
plt.plot(entropies, label='Entropy')
plt.xlabel('Time Step')
plt.ylabel('Normalized Value')
plt.title('Complexity vs Entropy in Rule 30')
plt.legend()
plt.grid(True)
plt.show()
```

**观察**：
- 早期：KC 和熵都低
- 中期：KC 快速增长
- 晚期：KC 趋于稳定，但熵继续增加

### 与咖啡混合的对比

等等，Rule 30 的 KC 似乎单调递增，而不是先增后减？

**关键区别**：
- **Rule 30**：是**开放系统**，不断扩展边界，信息持续流入
- **咖啡混合**：是**封闭系统**，总信息量固定，只是重新分布

对于封闭系统（固定网格大小），我们需要不同的度量...

## 封闭系统的复杂度演化

让我们修改 Rule 30，使用**周期性边界条件**（首尾相连）：

```python
def simulate_rule_30_periodic(width, steps):
    """周期性边界的 Rule 30"""
    grid = np.zeros((steps, width), dtype=int)
    grid[0, width // 2] = 1
    
    for t in range(steps - 1):
        for i in range(width):
            left = grid[t, (i - 1) % width]
            center = grid[t, i]
            right = grid[t, (i + 1) % width]
            grid[t + 1, i] = rule_30(left, center, right)
    
    return grid

# 模拟封闭系统
width, steps = 100, 200
grid_closed = simulate_rule_30_periodic(width, steps)

# 可视化
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(grid_closed[:50, :], cmap='binary')
plt.title('Rule 30 with Periodic Boundary (First 50 steps)')

plt.subplot(2, 1, 2)
plt.imshow(grid_closed[50:, :], cmap='binary')
plt.title('Rule 30 with Periodic Boundary (Steps 50-200)')
plt.show()
```

**观察**：在封闭系统中，图案最终会进入周期或准周期状态！

## 统计复杂度（Statistical Complexity）

为了更精确地度量"结构复杂度"，我们可以使用 **统计复杂度**（Crutchfield & Young, 1989）：

```python
from collections import Counter

def compute_statistical_complexity(row, history_length=3):
    """
    计算统计复杂度（简化版）
    基于因果态重构
    """
    # 构建历史-未来对
    pairs = []
    for i in range(history_length, len(row) - 1):
        history = tuple(row[i-history_length:i])
        future = row[i]
        pairs.append((history, future))
    
    # 计算因果态分布
    history_counts = Counter([p[0] for p in pairs])
    total = len(pairs)
    
    # 统计复杂度 = 因果态分布的熵
    complexity = 0
    for count in history_counts.values():
        p = count / total
        complexity -= p * np.log2(p)
    
    return complexity

# 计算每行的统计复杂度
scs = [compute_statistical_complexity(grid_closed[t, :]) for t in range(steps)]

plt.figure(figsize=(10, 4))
plt.plot(scs)
plt.xlabel('Time Step')
plt.ylabel('Statistical Complexity')
plt.title('Statistical Complexity Evolution (Closed System)')
plt.grid(True)
plt.show()
```

## 小结

问下大家，现在理解 Rule 30 和复杂动力学的关系了吗？

**核心要点**：

1. **Rule 30**：简单规则产生复杂行为，属于 Wolfram Class 3
2. **熵的演化**：单调递增，趋于最大熵
3. **复杂度度量**：
   - Kolmogorov 复杂度：不可计算，可用压缩比近似
   - 统计复杂度：基于因果态，度量结构复杂度
4. **开放 vs 封闭系统**：
   - 开放系统：复杂度可能持续增长
   - 封闭系统：复杂度可能先增后减

**关键洞察**：
- Rule 30 展示了"确定性混沌"——简单规则产生看似随机的行为
- 但真正的"复杂度"需要更精细的度量，如统计复杂度或 sophistication
- 封闭系统中的复杂度演化更接近 Aaronson 的"第一定律"

下一篇，我们将实现**咖啡混合模拟**，这是 Aaronson 文章中的核心例子！

---

**思考题**：
1. 你能找到其他 Wolfram Class 4 的规则吗？它们有什么特点？
2. 如何用 Rule 30 生成真正的随机数？有什么局限性？
3. 在机器学习中，元胞自动机有什么潜在应用？

*关注「云言 AI」，回复"Rule30"获取完整的元胞自动机探索代码！*
