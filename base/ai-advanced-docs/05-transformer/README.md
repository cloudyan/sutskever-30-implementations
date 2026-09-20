# 第 5 章：Transformer 与大模型基础（8-10 周）

> 深入理解现代 AI 的核心架构 —— 从注意力机制到 BERT/GPT
> 
> _学习周期：8-10 周 | 难度：⭐⭐⭐⭐⭐ | 重要性：⭐⭐⭐⭐⭐_

---

## 📖 本章概述

### Transformer 为什么如此重要？

```
┌─────────────────────────────────────────────────────────────────┐
│  Transformer 的革命性贡献                                        │
├─────────────────────────────────────────────────────────────────┤
│  ✅ 并行计算：告别 RNN 的序列依赖，训练速度提升 10 倍 +              │
│  ✅ 长程依赖：任意位置直接连接，解决梯度消失                      │
│  ✅ 通用架构：NLP、CV、多模态，一统江湖                          │
│  ✅ 可扩展性：从百万到万亿参数，Scaling Laws 成立                 │
│  ✅ 大模型基石：BERT、GPT、LLaMA 都基于 Transformer              │
└─────────────────────────────────────────────────────────────────┘
```

### 本章学习目标

学完本章后，你将能够：
- ✅ 深入理解自注意力机制的数学原理
- ✅ 从零实现完整的 Transformer 模型
- ✅ 理解 BERT 和 GPT 的架构差异
- ✅ 微调预训练模型解决下游任务
- ✅ 理解大模型的训练原理和技术挑战

---

## 📚 学习大纲

### 5.1 注意力机制深度理解（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 核心概念：什么是注意力？

```
人类注意力：
  看一张照片 → 聚焦关键区域 → 忽略背景
  
机器注意力：
  处理一句话 → 聚焦关键词 → 加权聚合信息
```

#### 自注意力机制（Self-Attention）

**数学公式**：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

其中：
- Q (Query): 查询向量，表示"我要找什么"
- K (Key):   键向量，表示"我有什么"
- V (Value): 值向量，表示"具体内容"
- d_k:       Q 和 K 的维度，用于缩放
```

**直观理解**：
```
输入句子："我 爱 中国"

对于"爱"这个词：
1. 计算 Q_爱 与所有 K 的相似度
   - Q_爱 · K_我 = 0.3
   - Q_爱 · K_爱 = 0.8  ← 最高
   - Q_爱 · K_中国 = 0.6

2. Softmax 归一化
   - [0.3, 0.8, 0.6] → [0.18, 0.52, 0.30]

3. 加权聚合 V
   - 输出 = 0.18×V_我 + 0.52×V_爱 + 0.30×V_中国

结果："爱"的表示聚合了上下文信息
```

**代码实现**：
```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "embed_size 必须能被 heads 整除"
        
        # 学习 Q, K, V 的投影矩阵
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # 1. 线性投影并分头
        # Q: (batch, seq_len, embed_size) → (batch, heads, seq_len, head_dim)
        queries = self.query(Q).reshape(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        keys = self.key(K).reshape(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        values = self.value(V).reshape(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # 2. 计算注意力分数 Q·K^T
        # energy: (batch, heads, seq_len, seq_len)
        energy = torch.matmul(queries, keys.transpose(-2, -1))
        
        # 3. 缩放
        energy = energy / math.sqrt(self.head_dim)
        
        # 4. Mask（用于 Decoder）
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        
        # 5. Softmax 归一化
        attention = torch.softmax(energy, dim=-1)
        
        # 6. 加权聚合 V
        # out: (batch, heads, seq_len, head_dim)
        out = torch.matmul(attention, values)
        
        # 7. 合并多头
        out = out.transpose(1, 2).reshape(batch_size, -1, self.embed_size)
        
        # 8. 输出投影
        out = self.fc_out(out)
        
        return out, attention

# 测试
attention = SelfAttention(embed_size=512, heads=8)
Q = K = V = torch.randn(32, 100, 512)  # batch=32, seq_len=100
output, attn_weights = attention(Q, K, V)
print(output.shape)  # torch.Size([32, 100, 512])
print(attn_weights.shape)  # torch.Size([32, 8, 100, 100])
```

---

#### 多头注意力（Multi-Head Attention）

**为什么需要多头？**
```
单头注意力：
  只能学习一种注意力模式
  
多头注意力：
  Head 1: 关注语法关系（主谓宾）
  Head 2: 关注语义关系（同义词）
  Head 3: 关注指代关系（代词 - 名词）
  Head 4: 关注位置关系（相邻词）
  ...
  
多个头并行学习不同的表示子空间
```

**架构图**：
```
         Input
           │
    ┌──────┴──────┐
    │   线性投影   │
    └──────┬──────┘
           │
    ┌──────┴──────┬──────┬──────┐
    │             │      │      │
  Head1        Head2  Head3  Head4  ← 并行计算
    │             │      │      │
    └──────┬──────┴──────┴──────┘
           │
    ┌──────┴──────┐
    │   拼接 + 投影 │
    └──────┬──────┘
           │
        Output
```

**代码实现**：
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # 为所有头一次性投影（效率更高）
        self.qkv_projection = nn.Linear(embed_size, embed_size * 3)
        self.output_projection = nn.Linear(embed_size, embed_size)
        
    def split_heads(self, x, batch_size):
        """将最后的维度拆分为 (num_heads, head_dim)"""
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 1. QKV 投影
        qkv = self.qkv_projection(query)  # (batch, seq_len, 3*embed_size)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # 2. 计算注意力
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e20'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 3. 加权求和
        context = torch.matmul(attention_weights, value)  # (batch, heads, seq_len, head_dim)
        
        # 4. 合并多头
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        
        # 5. 输出投影
        output = self.output_projection(context)
        
        return output, attention_weights
```

---

#### 注意力可视化与分析

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens, head_idx=0):
    """可视化注意力权重"""
    # attention_weights: (batch, heads, seq_len, seq_len)
    attn = attention_weights[0, head_idx].cpu().detach().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap='Blues')
    plt.title(f'Attention Head {head_idx}')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.tight_layout()
    plt.show()

# 示例
tokens = ['我', '爱', '中国']
visualize_attention(attn_weights, tokens, head_idx=0)
```

---

#### ✅ 实践项目：注意力机制分析

```python
"""
项目：分析自注意力机制的行为
要求：
1. 实现自注意力
2. 输入不同句子
3. 可视化注意力权重
4. 分析不同头的关注模式
"""

# 学员完成
```

</details>

---

### 5.2 Transformer 架构（3 周）

<details>
<summary>📋 查看详细知识点</summary>

#### Transformer 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Transformer                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Encoder（编码器）                    Decoder（解码器）          │
│  ┌─────────────────┐                ┌─────────────────┐        │
│  │  Multi-Head     │                │  Masked Multi-  │        │
│  │  Attention      │                │  Head Attention │        │
│  │       +         │                │       +         │        │
│  │  Add & Norm     │                │  Add & Norm     │        │
│  │       +         │                │       +         │        │
│  │  Multi-Head     │                │  Multi-Head     │        │
│  │  Attention      │◄──────────────►│  Attention      │        │
│  │       +         │   (Cross Attn) │  Add & Norm     │        │
│  │  Add & Norm     │                │       +         │        │
│  │       +         │                │  Feed Forward   │        │
│  │  Feed Forward   │                │       +         │        │
│  │       +         │                │  Add & Norm     │        │
│  │  Add & Norm     │                └─────────────────┘        │
│  └─────────────────┘                         │                 │
│        │                                     ▼                 │
│        ▼                                Output + Softmax        │
│   Context Vector                           │                    │
│                                            ▼                    │
│                                      下一个词概率                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

#### 位置编码（Positional Encoding）

**为什么需要位置编码？**
```
自注意力机制本身是位置无关的：
  "我 爱 中国" 和 "中国 爱 我"
  会得到相同的注意力权重（只是顺序不同）

解决方案：加入位置信息
  输入 = 词嵌入 + 位置编码
```

**正弦位置编码**：
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, embed_size)  # (max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # 计算不同频率
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        # 填充正弦和余弦
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
        
        # 调整形状为 (1, max_len, embed_size)
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不参与梯度更新）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, embed_size)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 可视化位置编码
pe = PositionalEncoding(embed_size=512, max_len=100)
pe_matrix = pe.pe[0, :, :].numpy()

plt.figure(figsize=(12, 6))
plt.imshow(pe_matrix, aspect='auto', cmap='coolwarm')
plt.colorbar(label='Value')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding')
plt.show()
```

---

#### Layer Normalization

**为什么需要归一化？**
```
问题：深层网络训练不稳定
  - 梯度消失/爆炸
  - 不同层的激活值分布差异大

解决方案：Layer Normalization
  - 将每层的激活值归一化为均值为 0、方差为 1
  - 学习缩放和平移参数
```

**代码实现**：
```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))   # 可学习的缩放
        self.bias = nn.Parameter(torch.zeros(features))    # 可学习的平移
        self.eps = eps
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        mean = x.mean(-1, keepdim=True)    # 沿最后一个维度求均值
        std = x.std(-1, keepdim=True)      # 沿最后一个维度求标准差
        
        # 归一化
        x_norm = (x - mean) / (std + self.eps)
        
        # 缩放和平移
        return self.weight * x_norm + self.bias

# PyTorch 内置版本
layer_norm = nn.LayerNorm(embed_size)
```

---

#### 残差连接（Residual Connection）

**为什么需要残差连接？**
```
问题：深层网络难以训练
  - 梯度需要传递很多层
  - 信息容易丢失

解决方案：残差连接
  Output = F(x) + x
  
好处：
  - 梯度可以直接流到前面层
  - 至少不会比恒等映射差
```

**Transformer 中的残差**：
```python
class ResidualConnection(nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        x: 输入
        sublayer: 要应用的子层（如 Attention、FFN）
        
        输出：LayerNorm(x + Dropout(sublayer(x)))
        """
        return x + self.dropout(sublayer(self.norm(x)))

# 使用示例
attention = MultiHeadAttention(embed_size=512)
residual = ResidualConnection(features=512)

# Pre-LN 模式（更稳定）
output = residual(x, attention)
```

---

#### Feed-Forward Network（前馈网络）

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # 升维
        self.linear2 = nn.Linear(d_ff, d_model)    # 降维
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 为什么需要升维？
# 更大的中间维度可以学习更复杂的特征
# 类似"瓶颈"结构，先扩展再压缩
```

---

#### 完整 Transformer 实现

```python
class TransformerBlock(nn.Module):
    """Transformer Encoder 层"""
    def __init__(self, embed_size=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = PositionwiseFeedForward(embed_size, ff_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-Attention + Residual
        attention_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed-Forward + Residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class TransformerEncoder(nn.Module):
    """多层 Transformer Encoder"""
    def __init__(self, vocab_size, embed_size=512, num_layers=6, 
                 num_heads=8, ff_dim=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.position_encoding(x)
        x = self.dropout(x)
        
        # 通过所有 Transformer 层
        attention_weights_list = []
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            attention_weights_list.append(attention_weights)
        
        return x, attention_weights_list


# 测试
vocab_size = 30000
encoder = TransformerEncoder(vocab_size, embed_size=512, num_layers=6)
input_ids = torch.randint(0, vocab_size, (32, 100))  # batch=32, seq_len=100
output, attn_weights = encoder(input_ids)
print(output.shape)  # torch.Size([32, 100, 512])
```

---

#### ✅ 实践项目：从零实现 Transformer

```python
"""
项目：完整 Transformer 实现
要求：
1. 实现 Encoder 和 Decoder
2. 在小型数据集上训练（如 IWSLT 翻译）
3. 达到可工作的翻译效果
4. 可视化注意力权重
"""

# 学员完成
```

</details>

---

### 5.3 BERT 系列（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### BERT 架构解析

```
┌─────────────────────────────────────────────────────────────────┐
│                        BERT 架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入：[CLS] 我 爱 中 国 [SEP]                                  │
│         │    │  │  │  │     │                                   │
│         ▼    ▼  ▼  ▼  ▼     ▼                                   │
│  ┌─────────────────────────────────────────┐                   │
│  │     Token Embeddings + Position + Segment│                   │
│  └─────────────────────────────────────────┘                   │
│                      │                                          │
│                      ▼                                          │
│  ┌─────────────────────────────────────────┐                   │
│  │          Transformer Encoder × 12       │  ← BERT-Base     │
│  │          (或 × 24 for BERT-Large)       │                   │
│  └─────────────────────────────────────────┘                   │
│                      │                                          │
│         ┌────────────┴────────────┐                            │
│         ▼                         ▼                            │
│    [CLS] 向量              每个 Token 输出                       │
│         │                         │                            │
│         ▼                         ▼                            │
│   句子级任务              Token 级任务                           │
│   (分类、蕴含)            (NER、QA)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### BERT 预训练任务

**1. Masked Language Model (MLM)**
```
原始句子：我 爱 中 国
Mask 后：  我 爱 [MASK] 国

任务：预测 [MASK] 位置应该是"中"

训练目标：最大化被 mask 词的对数似然
```

**2. Next Sentence Prediction (NSP)**
```
句子 A: 我 爱 中 国
句子 B: 中 国 是 我 的 家 乡  (50% 概率是下一句)
句子 C: 今 天 天 气 很 好  (50% 概率是随机句)

任务：判断 B/C 是否是 A 的下一句
```

---

#### 使用 Hugging Face 微调 BERT

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# 1. 加载预训练模型和分词器
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2  # 二分类
)

# 2. 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# 3. 训练配置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# 4. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 5. 开始训练
trainer.train()

# 6. 评估
results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.4f}")
```

---

#### ✅ 实践项目：BERT 文本分类

```python
"""
项目：使用 BERT 进行文本情感分析
数据集：ChnSentiCorp（中文情感分析）
目标：准确率 > 90%
"""

# 学员完成
```

</details>

---

### 5.4 GPT 系列（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### GPT vs BERT

| 特性 | BERT | GPT |
|------|------|-----|
| 架构 | Encoder-only | Decoder-only |
| 注意力 | 双向 | 单向（Causal） |
| 预训练任务 | MLM + NSP | Causal LM |
| 擅长任务 | 理解类（分类、NER） | 生成类（写作、对话） |
| 输入长度 | 512 | 更长（GPT-4 达 128K） |

#### Causal Language Model

```
GPT 生成过程：

输入：今 天 天 气
      │
      ▼
   预测：很
      │
      ▼
输入：今 天 天 气 很
      │
      ▼
   预测：好
```

**Causal Mask**：
```python
def generate_causal_mask(seq_len):
    """生成因果掩码，防止看到未来位置"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # mask:
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    return mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
```

---

#### GPT 推理实现

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_length=100, temperature=1.0):
    """GPT 文本生成"""
    model.eval()
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    for _ in range(max_length):
        # 前向传播
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :] / temperature
        
        # 采样
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        # 拼接
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        
        # 检查是否生成结束符
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    
    # 解码
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# 使用示例
result = generate(model, tokenizer, "从前有座山", max_length=50)
print(result)
```

</details>

---

## 📊 进度追踪

### 打卡表

| 章节 | 周数 | 已完成 | 进度 | 状态 |
|------|------|--------|------|------|
| 5.1 注意力机制 | 2 周 | - | 0% | ⏳ |
| 5.2 Transformer 架构 | 3 周 | - | 0% | ⏳ |
| 5.3 BERT 系列 | 2 周 | - | 0% | ⏳ |
| 5.4 GPT 系列 | 2 周 | - | 0% | ⏳ |

### 项目清单

- [ ] 从零实现自注意力机制
- [ ] 从零实现完整 Transformer
- [ ] 微调 BERT 做文本分类（准确率>90%）
- [ ] 实现 GPT 文本生成

---

> _Transformer 是 AI 领域的"通用计算机"，掌握它，你就掌握了现代 AI 的核心。_
> 
> _—— 悟空_
