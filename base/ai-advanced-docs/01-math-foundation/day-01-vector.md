# Day 1: 向量与向量空间

> 一切 AI 的起点 —— 理解向量如何表示世界
> 
> _难度：⭐⭐ | 预计时间：1-2 小时 | AI 应用：词向量、特征表示_

---

## 📌 一句话核心

**向量是有方向的箭头，AI 用它表示一切事物（文字、图片、声音）。**

---

## 🎯 核心问题

学完今天的内容，你应该能够回答：
1. 什么是向量？它和标量有什么区别？
2. 为什么 AI 要用向量表示一切？
3. 向量的点积有什么几何意义？
4. 如何用向量表示词语？

---

## 📚 核心定义

### 什么是向量？

```
数学定义：
向量是一个有序的数列，既有大小（magnitude）又有方向（direction）

表示方法：
- 几何：带箭头的线段 →
- 代数：a = [a₁, a₂, ..., aₙ]
- 物理：速度、力、加速度
```

### 向量 vs 标量

| 类型 | 定义 | 示例 |
|------|------|------|
| **标量** | 只有大小，没有方向 | 温度 25°C、质量 5kg、时间 10s |
| **向量** | 既有大小又有方向 | 速度 5m/s 向东、力 10N 向下 |

### 向量的维度

```
2 维向量：[x, y]        → 平面上的点
3 维向量：[x, y, z]     → 空间中的点
n 维向量：[x₁, x₂, ..., xₙ]  → 高维空间的点

AI 中的向量维度：
- 词向量：50-1000 维
- 图片向量：几千到几万维
- 用户特征向量：几百维
```

---

## 📐 几何直观

### 向量的可视化

```
2 维平面：
        y
        ↑
        │     • B(3, 4)
        │    ↗
        │   / 向量 AB = [3, 4]
        │  /
        │ /
───────┼/────────→ x
       O(0, 0)

向量长度（模）：|AB| = √(3² + 4²) = 5
```

### 向量加法（三角形法则）

```
    b
    ↑
    │
    │    a + b
    │   ↗
    │  /
    │ /
    →/
    a

a = [3, 0], b = [0, 4]
a + b = [3, 4]
```

### 向量数乘（缩放）

```
原始向量 a = [2, 2]

2a = [4, 4]  ← 长度翻倍，方向不变
0.5a = [1, 1] ← 长度减半，方向不变
-a = [-2, -2] ← 方向相反，长度不变
```

---

## 🔢 核心运算

### 1. 向量加法

```python
# 代数定义
a = [a₁, a₂, a₃]
b = [b₁, b₂, b₃]
a + b = [a₁+b₁, a₂+b₂, a₃+b₃]

# Python 实现
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b  # [5, 7, 9]
```

### 2. 向量数乘

```python
# 代数定义
k × a = [k×a₁, k×a₂, k×a₃]

# Python 实现
k = 2
c = k * a  # [2, 4, 6]
```

### 3. 向量点积（内积）⭐

```python
# 代数定义
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

# 几何定义
a · b = |a| × |b| × cos(θ)
        ↑    ↑    ↑
       长度  长度  夹角余弦

# Python 实现
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # 1×4 + 2×5 + 3×6 = 32
```

### 点积的几何意义

```
点积结果告诉我们两个向量的"相似度"：

a · b > 0  → 夹角 < 90°  → 方向相近
a · b = 0  → 夹角 = 90°  → 正交（垂直）
a · b < 0  → 夹角 > 90°  → 方向相反

点积越大，两个向量越"相似"
```

---

## 🤖 AI 中的应用

### 应用 1：词向量（Word Embedding）

```
核心思想：用向量表示词语，语义相似的词向量也相似

示例：
"国王" = [0.8, 0.6, -0.2, ...]
"王后" = [0.7, 0.7, -0.1, ...]
"男人" = [0.6, -0.5, 0.3, ...]
"女人" = [0.5, -0.4, 0.4, ...]

神奇发现：
国王 - 男人 + 女人 ≈ 王后

验证：
[0.8, 0.6, -0.2] - [0.6, -0.5, 0.3] + [0.5, -0.4, 0.4]
= [0.7, 0.7, -0.1] ≈ "王后"
```

### 应用 2：文本相似度

```python
# 用点积计算两个句子的相似度
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentences = ["我爱机器学习", "我喜欢深度学习"]

# 转换为向量
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(sentences)

# 计算余弦相似度
similarity = cosine_similarity(vectors)
print(f"相似度：{similarity[0][1]:.4f}")
```

### 应用 3：特征表示

```
机器学习中的特征向量：

用户画像 = [年龄，收入，消费次数，平均金额，...]
         = [25, 15000, 12, 500, ...]

图片特征 = [R 均值，G 均值，B 均值，纹理，边缘，...]
         = [128, 100, 95, 0.8, 0.6, ...]
```

---

## 💻 代码实践

### 练习 1：向量的基本运算

```python
import numpy as np

# 创建向量
a = np.array([3, 4])
b = np.array([1, 2])

# 向量加法
c = a + b
print(f"a + b = {c}")  # [4, 6]

# 向量数乘
d = 2 * a
print(f"2 * a = {d}")  # [6, 8]

# 向量长度（模）
length_a = np.linalg.norm(a)
print(f"|a| = {length_a}")  # 5.0

# 单位向量（方向不变，长度为 1）
unit_a = a / length_a
print(f"单位向量 = {unit_a}")  # [0.6, 0.8]
```

### 练习 2：点积与相似度

```python
import numpy as np

def cosine_similarity(a, b):
    """计算余弦相似度"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# 测试
a = np.array([1, 2, 3])
b = np.array([2, 4, 6])  # 与 a 同方向
c = np.array([1, 0, 0])  # 与 a 垂直

print(f"a 与 b 的相似度：{cosine_similarity(a, b):.4f}")  # 1.0（完全相似）
print(f"a 与 c 的相似度：{cosine_similarity(a, c):.4f}")  # 0.267
```

### 练习 3：词向量可视化

```python
import numpy as np
import matplotlib.pyplot as plt

# 简化的词向量（2 维用于可视化）
words = {
    "国王": np.array([0.8, 0.6]),
    "王后": np.array([0.7, 0.7]),
    "男人": np.array([0.6, -0.5]),
    "女人": np.array([0.5, -0.4]),
    "苹果": np.array([-0.5, 0.3]),
    "香蕉": np.array([-0.6, 0.2])
}

# 可视化
plt.figure(figsize=(8, 8))
for word, vec in words.items():
    plt.arrow(0, 0, vec[0], vec[1], head_width=0.05, length_includes_head=True)
    plt.text(vec[0]*1.1, vec[1]*1.1, word, fontsize=12)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.title('词向量可视化')
plt.show()
```

---

## 🧠 思考题

1. **基础题**：计算向量 [3, 4] 和 [1, 2] 的点积，并解释结果的几何意义。

2. **理解题**：为什么余弦相似度的取值范围是 [-1, 1]？当相似度为 0 时意味着什么？

3. **应用题**：如果有两个用户的特征向量分别是 [25, 15000, 12] 和 [30, 20000, 8]，如何计算他们的相似度？

4. **挑战题**：证明向量点积的几何定义 a·b = |a||b|cos(θ) 与代数定义等价。

---

## 📝 学习笔记模板

```markdown
# Day 1 学习笔记

## 今天我学到了
- 

## 最让我惊讶的是
- 

## 我还不太理解的是
- 

## 代码实践记录
```python
# 在这里记录你的代码
```

## 明日计划
- 复习向量点积
- 学习矩阵基础
```

---

## 🔗 延伸资源

### 推荐阅读
- 3Blue1Brown《线性代数的本质》第 1 集：向量是什么？
- 《线性代数及其应用》第 1 章

### 视频课程
- [MIT 18.06 第 1 讲](https://www.youtube.com/watch?v=QkF3oxziUI4)
- [B 站 3Blue1Brown](https://www.bilibili.com/video/BV1ys411472E)

### 在线工具
- [GeoGebra 向量可视化](https://www.geogebra.org/3d)
- [Desmos 向量计算器](https://www.desmos.com/calculator)

---

## ✅ 今日检查清单

- [ ] 理解向量的定义和表示
- [ ] 掌握向量加法和数乘
- [ ] 理解点积的代数和几何定义
- [ ] 能够计算向量长度和夹角
- [ ] 完成所有代码练习
- [ ] 回答所有思考题
- [ ] 记录学习笔记

---

> _向量是 AI 的字母表，掌握它，你才能读懂 AI 的语言。_
> 
> _—— 悟空_
