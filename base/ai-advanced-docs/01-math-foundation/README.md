# 阶段 1：数学基础

_AI 的数学基石_

---

## 📖 学习指南

**前置知识**：高中数学基础

**学习目标**：
- ✅ 掌握线性代数核心概念
- ✅ 掌握微积分与优化基础
- ✅ 掌握概率论与统计基础
- ✅ 掌握信息论基础
- ✅ 理解数学在 AI 中的应用

**预计时间**：30 天

**重要提示**：
本阶段是 AI 进阶的基础，建议系统学习。已有基础可快速复习。

---

## 1.1 线性代数核心

### 向量与矩阵

<div class="formula-box">

```
向量：
- 点积：a · b = Σ(aᵢ × bᵢ) = |a||b|cos(θ)
- 模长：||a|| = √(a · a)
- 余弦相似度：cos(θ) = (a · b) / (||a|| × ||b||)

矩阵：
- 矩阵乘法：(AB)ᵢⱼ = Σₖ(Aᵢₖ × Bₖⱼ)
- 转置：(AB)ᵀ = BᵀAᵀ
- 逆矩阵：A⁻¹A = AA⁻¹ = I
```

</div>

### 特征值与特征向量

<div class="formula-box">

```
定义：
Av = λv

其中：
- v：特征向量（变换后方向不变）
- λ：特征值（缩放倍数）

应用：
- PCA 降维
- 谱聚类
- PageRank 算法
```

</div>

### SVD（奇异值分解）

<div class="formula-box">

```
公式：
A = UΣVᵀ

其中：
- U：左奇异向量（AAᵀ的特征向量）
- Σ：奇异值对角矩阵
- V：右奇异向量（AᵀA 的特征向量）

应用：
- 降维
- 推荐系统
- 图像压缩
- LSA（潜在语义分析）
```

</div>

### 范数与距离

<div class="formula-box">

```
L1 范数：||x||₁ = Σ|xᵢ|
L2 范数：||x||₂ = √(Σxᵢ²)
L∞ 范数：||x||∞ = max|xᵢ|

应用：
- L1 正则化（Lasso）→ 稀疏解
- L2 正则化（Ridge）→ 防止过拟合
```

</div>

---

## 1.2 微积分与优化

### 导数与梯度

<div class="formula-box">

```
导数：
f'(x) = lim(h→0) [f(x+h) - f(x)] / h

梯度：
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

几何意义：
- 梯度指向函数增长最快的方向
- 负梯度方向是下降最快的方向
```

</div>

### 链式法则

<div class="formula-box">

```
单变量：
dy/dx = dy/du × du/dx

多变量：
∂y/∂x = Σ(∂y/∂uᵢ × ∂uᵢ/∂x)

应用：
- 反向传播的核心
- 神经网络梯度计算
```

</div>

### 梯度下降

<div class="formula-box">

```
公式：
wₜ₊₁ = wₜ - η∇L(wₜ)

变体：
- BGD：批量梯度下降（全部数据）
- SGD：随机梯度下降（一个样本）
- Mini-batch：小批量（折中方案）

优化器：
- Momentum：vₜ = γvₜ₋₁ + η∇L(wₜ)
- Adam：动量 + 自适应学习率
```

</div>

---

## 1.3 概率论与统计

### 贝叶斯定理

<div class="formula-box">

```
公式：
P(A|B) = P(B|A)P(A) / P(B)

其中：
- P(A)：先验概率
- P(B|A)：似然
- P(A|B)：后验概率

应用：
- 朴素贝叶斯分类
- 贝叶斯推断
- 贝叶斯优化
```

</div>

### 常见分布

<div class="formula-box">

```
伯努利分布：
P(X=1) = p, P(X=0) = 1-p

高斯分布：
p(x) = (1/√(2πσ²))exp(-(x-μ)²/(2σ²))

多项式分布：
P(X₁=k₁,...,Xₖ=kₖ) = n!/(k₁!...kₖ!) × p₁ᵏ¹...pₖᵏᵏ
```

</div>

### 期望与方差

<div class="formula-box">

```
期望：
E[X] = ΣxᵢP(X=xᵢ) (离散)
E[X] = ∫x·p(x)dx (连续)

方差：
Var(X) = E[X²] - (E[X])²

性质：
E[aX+b] = aE[X] + b
Var(aX+b) = a²Var(X)
```

</div>

### 最大似然估计（MLE）

<div class="formula-box">

```
定义：
θ̂ = argmax L(θ|D) = argmax log L(θ|D)

其中：
L(θ|D) = P(D|θ) = ΠᵢP(xᵢ|θ)

应用：
- 线性回归（等价于最小二乘）
- 逻辑回归
- 语言模型
```

</div>

---

## 1.4 信息论基础

### 熵（Entropy）

<div class="formula-box">

```
定义：
H(X) = -Σ P(x)log₂ P(x)

含义：
- 度量不确定性
- 均匀分布时熵最大
- 确定性分布时熵为 0

应用：
- 决策树分裂标准
- 交叉熵损失函数
```

</div>

### 交叉熵

<div class="formula-box">

```
定义：
H(p,q) = -Σ p(x)log q(x)

与熵的关系：
H(p,q) = H(p) + D_KL(p||q)

应用：
- 分类问题损失函数
- softmax + 交叉熵
```

</div>

### KL 散度

<div class="formula-box">

```
定义：
D_KL(p||q) = Σ p(x)log(p(x)/q(x))

性质：
- D_KL(p||q) ≥ 0
- 非对称：D_KL(p||q) ≠ D_KL(q||p)

应用：
- 变分推断
- VAE
- 知识蒸馏
```

</div>

---

## 1.5 数学在 AI 中的应用

### 线性代数应用

<div class="formula-box">

```
1. 神经网络
   - 全连接层：y = Wx + b
   - 卷积层：卷积核矩阵
   - 注意力：QKV 矩阵运算

2. PCA 降维
   - 协方差矩阵特征分解
   - 投影到主成分方向

3. 推荐系统
   - 矩阵分解
   - SVD 降维
```

</div>

### 微积分应用

<div class="formula-box">

```
1. 反向传播
   - 链式法则计算梯度
   - ∂L/∂W = δ·aᵀ

2. 优化算法
   - 梯度下降
   - 牛顿法（二阶导数）
```

</div>

### 概率论应用

<div class="formula-box">

```
1. 机器学习
   - 朴素贝叶斯分类
   - 高斯混合模型
   - 隐马尔可夫模型

2. 深度学习
   - Dropout（伯努利分布）
   - 权重初始化（高斯分布）
```

</div>

### 信息论应用

<div class="formula-box">

```
1. 损失函数
   - 交叉熵损失
   - KL 散度正则化

2. 注意力机制
   - softmax 注意力权重
   - 信息瓶颈理论
```

</div>

---

## 📚 学习资源

### 在线课程

- [AI 数学基础 70 讲](https://cloudyan.github.io/ai-math-docs/) - 完整视频教程
- [3Blue1Brown 线性代数](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_abr) - 可视化讲解
- [3Blue1Brown 微积分](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) - 直观理解

### 书籍

- 《线性代数应该这样学》Axler
- 《概率论与数理统计》陈希孺
- 《Elements of Information Theory》Cover

---

## ✅ 学习检查清单

- [ ] 掌握向量与矩阵运算
- [ ] 理解特征值与特征向量
- [ ] 理解 SVD 分解
- [ ] 掌握导数与梯度
- [ ] 掌握链式法则
- [ ] 掌握梯度下降
- [ ] 理解贝叶斯定理
- [ ] 掌握常见概率分布
- [ ] 理解熵与交叉熵
- [ ] 理解 KL 散度
- [ ] 理解数学在 AI 中的应用

---

*最后更新：2026-04-22*
