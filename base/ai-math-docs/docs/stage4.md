# 阶段 4-6 详细文档

_统计推断、信息论与综合应用_

---

## 阶段 4：统计与推断（Day 31-40）

---

### Day 31: 最大似然估计 MLE

**前置知识**：Day 22（贝叶斯）、Day 24（分布）、Day 25（期望方差）  
**后续应用**：Day 32（MAP）、Day 47（最小二乘）、Day 50（EM）

**核心定义**：
- **似然函数**：L(θ|D) = P(D|θ)
- **MLE**：θ̂ = argmax L(θ|D) = argmax log L(θ|D)

**关键公式**：
<div class="formula-box">

```
似然：L(θ|D) = ΠᵢP(xᵢ|θ)
对数似然：ℓ(θ) = Σᵢlog P(xᵢ|θ)
高斯 MLE：μ̂ = (1/n)Σxᵢ, σ̂² = (1/n)Σ(xᵢ-μ̂)²
```

</div>

---

### Day 32: 最大后验估计 MAP

**核心定义**：
- **后验**：P(θ|D) ∝ P(D|θ)P(θ)
- **MAP**：θ̂ = argmax [log P(D|θ) + log P(θ)]

**关键公式**：
<div class="formula-box">

```
高斯先验：P(θ) = N(0, σ₀²I) → L2 正则化
拉普拉斯先验：P(θ) ∝ exp(-λ||θ||₁) → L1 正则化
```

</div>

---

### Day 33: 贝叶斯推断

**核心公式**：
<div class="formula-box">

```
后验：p(θ|D) = p(D|θ)p(θ) / p(D)
后验预测：p(x_new|D) = ∫p(x_new|θ)p(θ|D)dθ
```

</div>

---

### Day 34-40 详见完整卡片

完整内容请参考原始知识卡片文件。

---

## 阶段 5：信息论基础（Day 41-45）

---

### Day 41: 信息量与自信息

**核心公式**：
<div class="formula-box">

```
I(x) = -log₂ P(x)  (单位：bit)
I(x,y) = I(x) + I(y)  (独立事件)
```

</div>

---

### Day 42: 熵 Entropy

**核心公式**：
<div class="formula-box">

```
H(X) = -Σ P(x)log₂ P(x)
H(p) = -p log₂ p - (1-p)log₂(1-p)  (二元熵)
```

</div>

---

### Day 43: 交叉熵

**核心公式**：
<div class="formula-box">

```
H(p,q) = -Σ p(x)log q(x)
H(p,q) = H(p) + D_KL(p||q)
```

</div>

---

### Day 44: KL 散度

**核心公式**：
<div class="formula-box">

```
D_KL(p||q) = Σ p(x)log(p(x)/q(x))
D_KL(p||q) = H(p,q) - H(p)
```

</div>

---

### Day 45: 互信息

**核心公式**：
<div class="formula-box">

```
I(X;Y) = H(X) - H(X|Y)
I(X;Y) = D_KL(P(X,Y) || P(X)P(Y))
```

</div>

---

## 阶段 6：综合与应用（Day 46-60）

---

### Day 46: softmax 与交叉熵

**前置知识**：Day 42-43（熵）、Day 11-13（导数）

**核心公式**：
<div class="formula-box">

```
softmax: σ(z)ᵢ = exp(zᵢ)/Σⱼexp(zⱼ)
交叉熵：L = -Σₖ yₖ log ŷₖ
梯度：∂L/∂zᵢ = ŷᵢ - yᵢ
```

</div>

---

### Day 56: 注意力机制的数学

**核心公式**：
<div class="formula-box">

```
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

</div>

---

### Day 57: 位置编码的数学

**核心公式**：
<div class="formula-box">

```
PE(pos,2i) = sin(pos/10000^(2i/d))
PE(pos,2i+1) = cos(pos/10000^(2i/d))
```

</div>

---

### Day 58: 归一化技术

**核心公式**：
<div class="formula-box">

```
BatchNorm: x̂ = (x-μ)/σ, y = γx̂ + β
LayerNorm: 对单个样本的所有特征归一化
```

</div>

---

### Day 59: 初始化方法

**核心公式**：
<div class="formula-box">

```
Xavier: W ~ N(0, 2/(fan_in+fan_out))
He: W ~ N(0, 2/fan_in) (适用于 ReLU)
```

</div>

---

### Day 60: 学习率调度

**核心方法**：
- Warmup：η = η_max × t/T_warmup
- 余弦退火：ηₜ = η_min + (η_max - η_min) × (1+cos(πt/T))/2
- 逆平方根：η = d_model^(-0.5) × min(t^(-0.5), t×T_warmup^(-1.5))

---

## 📖 完整内容

详细的 60 张知识卡片请参考：
- 📁 [原始知识卡片文件](https://github.com/你的仓库/memory/ai-math-cards-*.md)
- 🗺️ [学习路径与依赖图](#/docs/learning-path)

---

*阶段 4-6 完成！🎓*
