# Transformer 为什么只需要注意力?

问下大家,有没有想过为什么现在的 GPT、BERT 等大模型都基于 Transformer,而不是以前的 RNN 或 CNN?

晓寒刚开始学 NLP 的时候,用的都是 LSTM、GRU,觉得序列模型就该这么建。直到 2017 年 Transformer 横空出世,才发现卧槽,原来纯注意力机制就能搞定一切!

今天我们就来揭开 Transformer 的神秘面纱,看看它为什么能成为现代 AI 的基石!

## RNN/LSTM 的困境

### 序列建模的老大难

```
RNN/LSTM 的问题:

1. 串行计算:
   h_t = f(h_{t-1}, x_t)
   
   必须等 h_{t-1} 算完才能算 h_t
   无法并行,训练慢!

2. 长程依赖:
   即使有 LSTM,长距离依赖还是难
   
   "我出生在...[很长很长的文字]...那里的美食..."
   LSTM 早就忘了"出生在"后面是哪里了

3. 信息瓶颈:
   所有信息都要压缩到固定大小的隐状态
   信息丢失严重
```

### Transformer 的革命性想法

```
Transformer 的核心洞见:

"不需要循环,只需要注意力!"

注意力机制:
- 每个位置可以直接看到所有其他位置
- 不需要一步一步传递信息
- 完全并行化

这就像:
- RNN: 大家排成队,每个人只能和前面的人说话
- Transformer: 大家围成圈,每个人都可以和所有人说话
```

## 自注意力机制

### 核心公式

**缩放点积注意力**(Scaled Dot-Product Attention):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中:
- $Q$ (Query): 查询矩阵
- $K$ (Key): 键矩阵  
- $V$ (Value): 值矩阵
- $d_k$: 键的维度

### 直观理解

```
自注意力就像是开会:

每个人(每个位置)都在问:
- "我和谁最相关?" (Q 问 K)
- 得到相关性分数 (注意力权重)
- 综合大家的信息 (加权求和 V)

例子:
句子: "我 爱 北京 天安门"

位置"爱"的查询:
- 和"我"的相关性: 0.3
- 和"北京"的相关性: 0.5
- 和"天安门"的相关性: 0.1

"爱"的表示 = 0.3*"我" + 0.5*"北京" + 0.1*"天安门"
```

### 用 NumPy 实现

```python
import numpy as np

def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力
    
    参数:
        Q: 查询 (seq_len_q, d_k)
        K: 键 (seq_len_k, d_k)
        V: 值 (seq_len_v, d_v)
        mask: 可选的掩码 (seq_len_q, seq_len_k)
    
    返回:
        output: 注意力输出
        attention_weights: 注意力权重
    """
    d_k = Q.shape[-1]
    
    # 1. 计算注意力分数
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # 2. 应用掩码(如果有)
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    # 3. Softmax 得到注意力权重
    attention_weights = softmax(scores, axis=-1)
    
    # 4. 加权求和
    output = np.dot(attention_weights, V)
    
    return output, attention_weights

# 测试
seq_len = 5
d_model = 8

Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print(f"注意力输出形状: {output.shape}")
print(f"注意力权重形状: {attn_weights.shape}")
print(f"\n注意力权重(每行和为1):")
print(attn_weights.sum(axis=1))
```

### 为什么除以 √d_k?

```
问题: 当 d_k 很大时,点积结果会很大
- Q·K 可能很大 → softmax 输入很大
- softmax 输出会接近 one-hot(梯度很小)
- 训练困难

解决: 除以 √d_k 进行缩放
- 保持点积结果的方差稳定
- 避免 softmax 饱和
- 梯度流动更顺畅
```

## 多头注意力

### 为什么需要多头?

```
单头注意力的局限:
- 只能学习一种"相关性"
- 就像只有一只眼睛,看东西角度单一

多头注意力的优势:
- 多个头 = 多个视角
- 每个头学习不同的"相关性"
- 有的关注语法,有的关注语义,有的关注位置...
```

### 实现

```python
class MultiHeadAttention:
    """
    多头注意力
    
    将输入分割成多个头,每个头独立计算注意力,最后合并
    """
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q, K, V 的线性变换
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        
        # 输出的线性变换
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def split_heads(self, x):
        """
        分割多头: (seq_len, d_model) → (num_heads, seq_len, d_k)
        """
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)
    
    def combine_heads(self, x):
        """
        合并多头: (num_heads, seq_len, d_k) → (seq_len, d_model)
        """
        seq_len = x.shape[1]
        x = x.transpose(1, 0, 2)
        return x.reshape(seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        
        参数:
            Q, K, V: (seq_len, d_model)
        
        返回:
            output: (seq_len, d_model)
        """
        # 1. 线性变换
        Q = np.dot(Q, self.W_q.T)
        K = np.dot(K, self.W_k.T)
        V = np.dot(V, self.W_v.T)
        
        # 2. 分割多头
        Q = self.split_heads(Q)  # (num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. 每个头计算注意力
        head_outputs = []
        attention_weights = []
        
        for i in range(self.num_heads):
            head_out, head_attn = scaled_dot_product_attention(
                Q[i], K[i], V[i], mask
            )
            head_outputs.append(head_out)
            attention_weights.append(head_attn)
        
        # 4. 合并多头
        heads = np.stack(head_outputs, axis=0)
        combined = self.combine_heads(heads)
        
        # 5. 最终线性变换
        output = np.dot(combined, self.W_o.T)
        
        return output

# 测试
d_model = 64
num_heads = 8
seq_len = 10

mha = MultiHeadAttention(d_model, num_heads)

X = np.random.randn(seq_len, d_model)
output = mha.forward(X, X, X)  # 自注意力

print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")
print(f"头数: {num_heads}")
print(f"每头维度: {mha.d_k}")
```

## 位置编码

### 为什么需要位置编码?

```
问题: 自注意力是排列不变的!

句子: "我 爱 你"
打乱: "你 爱 我"

自注意力给出的表示是一样的!
因为注意力只看内容,不看顺序

但顺序很重要:
- "我爱你" vs "你爱我"
- "猫吃鱼" vs "鱼吃猫"
```

### 正弦位置编码

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

### 实现

```python
def positional_encoding(seq_len, d_model):
    """
    正弦位置编码
    
    参数:
        seq_len: 序列长度
        d_model: 模型维度
    
    返回:
        pe: (seq_len, d_model) 位置编码
    """
    pe = np.zeros((seq_len, d_model))
    
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * 
                      -(np.log(10000.0) / d_model))
    
    # 偶数维度用 sin
    pe[:, 0::2] = np.sin(position * div_term)
    
    # 奇数维度用 cos
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

# 测试
seq_len = 50
d_model = 64

pe = positional_encoding(seq_len, d_model)

print(f"位置编码形状: {pe.shape}")
print("\n位置编码特点:")
print("1. 每个位置有唯一的编码")
print("2. 不同位置可以通过线性变换关联")
print("3. 不同频率编码不同尺度的位置信息")
```

### 位置编码可视化

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.imshow(pe.T, cmap='RdBu', aspect='auto')
plt.colorbar(label='编码值')
plt.xlabel('位置')
plt.ylabel('维度')
plt.title('位置编码热力图')

plt.subplot(2, 1, 2)
# 绘制前几个维度
for i in [0, 1, 2, 3, 10, 20]:
    plt.plot(pe[:, i], label=f'维度 {i}')
plt.xlabel('位置')
plt.ylabel('编码值')
plt.title('位置编码曲线')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Transformer 编码器

### 完整架构

```
Transformer 编码器层:

输入 x
  ↓
  ├─────────────────────┐
  ↓                     │
多头注意力              │
  ↓                     │
Add & Norm ←───────────┘
  ↓
  ├─────────────────────┐
  ↓                     │
前馈网络 (FFN)          │
  ↓                     │
Add & Norm ←───────────┘
  ↓
输出
```

### 实现

```python
class FeedForward:
    """
    前馈网络
    
    两个线性变换 + ReLU
    """
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        # 第一层 + ReLU
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)
        
        # 第二层
        output = np.dot(hidden, self.W2) + self.b2
        
        return output

class LayerNorm:
    """层归一化"""
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        return self.gamma * x_norm + self.beta

class TransformerEncoderLayer:
    """
    Transformer 编码器层
    
    包含:
    1. 多头自注意力
    2. 前馈网络
    3. 残差连接
    4. 层归一化
    """
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # 1. 多头自注意力 + 残差连接 + 层归一化
        attn_out = self.mha.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_out)
        
        # 2. 前馈网络 + 残差连接 + 层归一化
        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)
        
        return x

class TransformerEncoder:
    """完整的 Transformer 编码器"""
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        self.layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x

# 测试
num_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
seq_len = 32

encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)

# 输入嵌入 + 位置编码
x = np.random.randn(seq_len, d_model)
pe = positional_encoding(seq_len, d_model)
x = x + pe

# 前向传播
output = encoder.forward(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"层数: {num_layers}")
```

## Transformer 的优势

### 1. 并行计算

```
RNN:
h_1 → h_2 → h_3 → ... → h_T
串行,无法并行

Transformer:
所有位置同时计算自注意力
完全并行,GPU 加速

速度提升:
- 训练速度: 快 10-100 倍
- 推理速度: 也更快
```

### 2. 长程依赖

```
RNN:
信息要一步步传递,容易丢失

Transformer:
每个位置可以直接看到所有位置
距离为 1,不存在长程依赖问题
```

### 3. 可解释性

```
注意力权重可视化:
- 可以看到模型关注哪些位置
- 理解模型的推理过程
- 调试和分析更容易
```

## 实际应用

### 1. GPT (生成式预训练)

```
GPT = Transformer 解码器

训练: 自回归语言模型
应用: 文本生成、对话、代码生成

GPT-3 参数: 1750 亿
训练数据: 互联网文本
能力: 写文章、编程、翻译...
```

### 2. BERT (双向编码)

```
BERT = Transformer 编码器

训练: 掩码语言模型 + 下一句预测
应用: 文本理解、问答、分类

特点:
- 双向理解(不像 GPT 只看左边)
- 预训练 + 微调范式
```

### 3. 机器翻译

```
Transformer 编码器-解码器

编码器: 理解源语言
解码器: 生成目标语言
注意力: 对齐源语言和目标语言

效果:
- 显著超越 RNN 模型
- 成为机器翻译的标准架构
```

## 小结

今天我们深入理解了 Transformer 的核心机制:

### 核心创新

1. **自注意力**: 每个位置可以直接看到所有位置
2. **多头注意力**: 多个视角学习不同的相关性
3. **位置编码**: 为无序的注意力提供顺序信息
4. **并行计算**: 完全并行化,训练速度快

### 为什么有效

1. **解决长程依赖**: 距离为 1,信息直接传递
2. **并行化**: GPU 加速,大规模训练成为可能
3. **可扩展**: 容易堆叠更多层,扩大模型容量

### 影响深远

Transformer 成为现代 AI 的基石:
- **NLP**: GPT、BERT、T5、LLaMA
- **CV**: Vision Transformer (ViT)
- **多模态**: CLIP、DALL-E、Stable Diffusion
- **科学**: AlphaFold、AlphaCode

## 练习题

### 1. 概念理解

**问题 1**: 为什么自注意力需要除以 √d_k?不除会怎样?

**问题 2**: 多头注意力和单头注意力有什么区别?为什么多头效果更好?

**问题 3**: Transformer 的位置编码为什么用 sin/cos?有什么优势?

### 2. 编程实践

**练习 1**: 实现 Transformer 解码器:

```python
class TransformerDecoderLayer:
    """Transformer 解码器层"""
    # 包含:
    # 1. 带掩码的自注意力
    # 2. 编码器-解码器注意力
    # 3. 前馈网络
    # TODO: 实现
    pass
```

**练习 2**: 实现因果掩码:

```python
def create_causal_mask(seq_len):
    """
    创建因果掩码
    
    用于自回归生成,防止看到未来信息
    """
    # TODO: 实现
    pass
```

**练习 3**: 训练一个小型 Transformer:

```python
# 任务: 英文翻译
# 数据: IWSLT 数据集
# TODO: 实现数据加载、训练、评估
```

### 3. 深度思考

**思考 1**: Transformer 的计算复杂度是 O(n²),如何处理超长序列?

**思考 2**: 为什么 Transformer 需要大量数据训练?如何改进?

**思考 3**: Transformer 和 CNN、RNN 的关系?能否结合?

## 延伸阅读

### 经典论文

1. **Transformer**: Vaswani et al. (2017). "Attention Is All You Need"
2. **BERT**: Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
3. **GPT-3**: Brown et al. (2020). "Language Models are Few-Shot Learners"

### 教程资源

- "The Annotated Transformer" - 逐行注释的 PyTorch 实现
- "Illustrated Transformer" - Jay Alammar 的可视化教程
- "Transformers from Scratch" - 从零实现

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 13 篇。上一篇我们学习了图神经网络如何传递消息,下一篇我们将深入理解 Bahdanau 注意力机制如何改进机器翻译。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!** 📚