# 注意力机制详解

_从数学原理到代码实现的完整解析_

---

## 📖 学习指南

**前置知识**：
- ✅ 线性代数（向量、矩阵、点积）
- ✅ 微积分（导数、梯度）
- ✅ Python 与 NumPy

**学习目标**：
- ✅ 理解注意力的数学原理
- ✅ 掌握 Self-Attention 机制
- ✅ 理解 Multi-Head Attention
- ✅ 能从零实现注意力机制

**预计时间**：7 天

---

## 1. 为什么需要注意力？

### 1.1 传统序列模型的问题

<div class="formula-box">

```
RNN/LSTM 处理序列：

输入："我 喜欢 机器 学习"
        ↓
RNN 依次处理每个词
        ↓
问题：
1. 长距离依赖困难（"学习"很难关注到"我"）
2. 无法并行计算（必须按顺序）
3. 信息瓶颈（所有信息压缩到一个向量）
```

</div>

### 1.2 注意力的核心思想

<div class="formula-box">

```
人类阅读时的注意力：

句子："The animal didn't cross the street because it was too tired"

理解"it"指代什么：
- 人类会关注"animal"和"tired"
- 而不是平均关注所有词

注意力机制：
让模型学会"关注"相关信息，"忽略"无关信息
```

</div>

---

## 2. 注意力的数学原理

### 2.1 定义

<div class="formula-box">

```
注意力函数：
Attention(Query, Key, Value) → Output

其中：
- Query (Q): 查询向量，表示"我要找什么"
- Key (K): 键向量，表示"我有什么信息"
- Value (V): 值向量，表示"信息的具体内容"
- Output: 加权聚合的 Value
```

</div>

### 2.2 Scaled Dot-Product Attention

<div class="formula-box">

```
公式：
Attention(Q, K, V) = softmax(QKᵀ/√d_k) V

计算步骤：
1. 计算相似度分数：S = QKᵀ
2. 缩放：S_scaled = S / √d_k
3. 归一化：A = softmax(S_scaled)
4. 加权求和：Output = A × V

其中：
- d_k: Key 的维度
- √d_k: 缩放因子，防止点积过大
```

</div>

### 2.3 为什么除以√d_k？

<div class="formula-box">

```
问题：当 d_k 很大时，QKᵀ的点积会很大

例子：
Q = [1, 1, ..., 1] (d_k=1000)
K = [1, 1, ..., 1]
QKᵀ = 1000

softmax([1000, 500, 100]) ≈ [1.0, 0.0, 0.0]  ← 梯度消失！

解决：除以√d_k
QKᵀ/√d_k = 1000/31.6 ≈ 31.6
softmax([31.6, 15.8, 3.16]) ≈ [0.99, 0.01, 0.0]  ← 梯度正常 ✓
```

</div>

---

## 3. Self-Attention 详解

### 3.1 什么是 Self-Attention？

<div class="formula-box">

```
Self-Attention = 序列内部的注意力

输入序列：X = [x₁, x₂, ..., xₙ]

对于每个位置 i：
- Query: qᵢ = W_Q · xᵢ
- Key: kⱼ = W_K · xⱼ (对所有 j)
- Value: vⱼ = W_V · xⱼ (对所有 j)

输出：
zᵢ = Σⱼ softmax(qᵢ·kⱼ/√d) × vⱼ
```

</div>

### 3.2 矩阵形式（并行计算）

<div class="formula-box">

```
输入：X ∈ R^(n×d_model)

Q = XW_Q    (n×d_k)
K = XW_K    (n×d_k)
V = XW_V    (n×d_v)

Attention = softmax(QKᵀ/√d_k) V

输出：Z ∈ R^(n×d_v)

优势：可以并行计算所有位置！
```

</div>

### 3.3 可视化示例

<div class="formula-box">

```
句子："The animal didn't cross the street because it was too tired"

对于"it"这个词的注意力分布：

animal:  ████████████████████  0.50
tired:   ████████████          0.30
because: ████                  0.10
street:  ██                    0.05
其他：   ██                    0.05

→ "it"主要关注"animal"和"tired"
→ 正确理解指代关系
```

</div>

---

## 4. Multi-Head Attention

### 4.1 为什么需要多头？

<div class="formula-box">

```
单头注意力的局限：
- 只能学习一种注意力模式
- 可能遗漏不同类型的信息

多头注意力：
- 多个头学习不同的注意力模式
- 类似 CNN 的多个卷积核
- 捕获不同类型的依赖关系
```

</div>

### 4.2 多头注意力机制

<div class="formula-box">

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ)Wᴼ

其中每个头：
headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)

参数：
- h: 头数（通常 8 或 16）
- WᵢQ ∈ R^(d_model×d_k): 第 i 个头的 Query 权重
- WᵢK ∈ R^(d_model×d_k): 第 i 个头的 Key 权重
- WᵢV ∈ R^(d_model×d_v): 第 i 个头的 Value 权重
- Wᴼ ∈ R^(h·d_v×d_model): 输出投影权重
```

</div>

### 4.3 示例（d_model=512, h=8）

<div class="formula-box">

```
每个头的维度：d_k = d_v = 512/8 = 64

头 1: 关注语法关系（主谓、动宾）
头 2: 关注指代关系（it→animal）
头 3: 关注修饰关系（形容词→名词）
头 4: 关注长距离依赖
...
头 8: 关注其他模式

最后拼接：[head₁, head₂, ..., head₈] ∈ R^(512)
```

</div>

---

## 5. 从零实现注意力机制

### 5.1 NumPy 实现

<div class="formula-box">

```python
import numpy as np

class SelfAttention:
    def __init__(self, d_model, d_k=None, d_v=None):
        self.d_model = d_model
        self.d_k = d_k if d_k else d_model
        self.d_v = d_v if d_v else d_model
        
        # 初始化权重
        self.W_Q = np.random.randn(d_model, self.d_k) / np.sqrt(d_model)
        self.W_K = np.random.randn(d_model, self.d_k) / np.sqrt(d_model)
        self.W_V = np.random.randn(d_model, self.d_v) / np.sqrt(d_model)
        self.W_O = np.random.randn(self.d_v, d_model) / np.sqrt(d_model)
    
    def softmax(self, x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(self, X, mask=None):
        """
        X: 输入序列 (batch_size, seq_len, d_model)
        mask: 掩码矩阵 (可选)
        """
        batch_size, seq_len, _ = X.shape
        
        # 计算 Q, K, V
        Q = X @ self.W_Q      # (batch, seq_len, d_k)
        K = X @ self.W_K      # (batch, seq_len, d_k)
        V = X @ self.W_V      # (batch, seq_len, d_v)
        
        # 计算注意力分数
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)  # (batch, seq_len, seq_len)
        
        # 应用掩码（用于 Decoder）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 归一化
        attention_weights = self.softmax(scores)  # (batch, seq_len, seq_len)
        
        # 加权求和
        output = attention_weights @ V  # (batch, seq_len, d_v)
        
        # 输出投影
        output = output @ self.W_O  # (batch, seq_len, d_model)
        
        return output, attention_weights
```

</div>

### 5.2 PyTorch 实现

<div class="formula-box">

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.scale = np.sqrt(self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # 线性变换并分头
        # Q: (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)  # (batch, heads, seq_len, seq_len)
        
        # 加权求和
        attention_output = torch.matmul(attention_weights, V)  # (batch, heads, seq_len, d_k)
        
        # 拼接并投影
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_O(attention_output)
        
        return output, attention_weights
```

</div>

### 5.3 测试代码

<div class="formula-box">

```python
# 测试 Self-Attention
batch_size = 2
seq_len = 10
d_model = 512

X = torch.randn(batch_size, seq_len, d_model)

attention = MultiHeadAttention(d_model=512, num_heads=8)
output, attention_weights = attention(X, X, X)

print(f"输入形状：{X.shape}")           # (2, 10, 512)
print(f"输出形状：{output.shape}")       # (2, 10, 512)
print(f"注意力权重形状：{attention_weights.shape}")  # (2, 8, 10, 10)

# 可视化注意力权重
import matplotlib.pyplot as plt

plt.imshow(attention_weights[0, 0].detach().numpy())
plt.colorbar()
plt.title('Head 1 Attention Weights')
plt.show()
```

</div>

---

## 6. 注意力的变体

### 6.1 掩码注意力（Masked Attention）

<div class="formula-box">

```
用于 Decoder，防止看到未来信息

掩码矩阵：
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]

应用掩码后：
scores = scores.masked_fill(mask == 0, -1e9)
softmax 后，未来位置的权重为 0
```

</div>

### 6.2 稀疏注意力

<div class="formula-box">

```
标准注意力复杂度：O(n²)

稀疏注意力：
- Local Attention: 只关注附近窗口
- Sparse Attention: 只关注部分位置
- Linear Attention: 近似为 O(n)

应用：长文本处理（Longformer、BigBird）
```

</div>

### 6.3 交叉注意力（Cross Attention）

<div class="formula-box">

```
用于 Encoder-Decoder 架构

Q 来自 Decoder
K, V 来自 Encoder

例如机器翻译：
- Decoder: "I love"
- Encoder: "我 喜欢 机器 学习"
- Cross Attention: 让"I love"关注中文句子
```

</div>

---

## 7. 论文解读

### 7.1 Attention Is All You Need (2017)

**核心贡献**：
- ✅ 提出 Transformer 架构
- ✅ 证明注意力机制足以替代 RNN/CNN
- ✅ 实现并行化，大幅提升训练速度

**关键设计**：
- Multi-Head Attention
- Positional Encoding
- Layer Normalization
- Residual Connection

### 7.2 后续发展

| 模型 | 年份 | 注意力改进 |
|------|------|-----------|
| BERT | 2018 | Encoder-only |
| GPT | 2018 | Decoder-only + Masked |
| T5 | 2019 | Encoder-Decoder |
| Longformer | 2020 | 稀疏注意力 |
| Linear Transformer | 2020 | 线性注意力 |

---

## 8. 实战项目

### 项目 1：实现完整 Transformer

<div class="formula-box">

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        src = self.embedding(src) + self.pos_encoder(src)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        # Decoder
        tgt = self.embedding(tgt) + self.pos_encoder(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask)
        
        # 输出
        output = self.fc_out(tgt)
        return output
```

</div>

### 项目 2：可视化注意力

<div class="formula-box">

```python
def visualize_attention(sentence, attention_weights, layer_idx=0, head_idx=0):
    """可视化注意力权重"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    words = sentence.split()
    weights = attention_weights[layer_idx][head_idx]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(weights, xticklabels=words, yticklabels=words, cmap='Blues')
    plt.title(f'Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.show()

# 使用
sentence = "The animal didn't cross the street because it was too tired"
visualize_attention(sentence, attention_weights)
```

</div>

---

## 📚 学习资源

### 论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化讲解

### 代码

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 逐行注释
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 工业级实现

### 视频

- [3Blue1Brown - Attention Mechanism](https://www.youtube.com/) - 可视化讲解

---

## ✅ 学习检查清单

- [ ] 理解注意力的数学原理
- [ ] 掌握 Q、K、V 的含义
- [ ] 理解为什么除以√d_k
- [ ] 掌握 Self-Attention 机制
- [ ] 理解 Multi-Head Attention
- [ ] 能从零实现注意力机制
- [ ] 理解掩码注意力的作用
- [ ] 完成至少 1 个实战项目

---

*最后更新：2026-04-22*
