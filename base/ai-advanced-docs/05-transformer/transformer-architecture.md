# Transformer 完整架构

_从 Attention 到 Encoder-Decoder 的完整实现_

---

## 📖 学习指南

**前置知识**：
- ✅ 注意力机制详解
- ✅ 深度学习基础
- ✅ PyTorch 基础

**学习目标**：
- ✅ 理解 Transformer 完整架构
- ✅ 掌握位置编码原理
- ✅ 掌握 Encoder-Decoder 结构
- ✅ 能从零实现 Transformer
- ✅ 理解 BERT、GPT 的架构差异

**预计时间**：14 天

---

## 1. Transformer 架构概览

### 1.1 整体结构

<div class="formula-box">

```
Transformer = Encoder + Decoder

Encoder（编码器）：
输入 → Embedding + Positional Encoding
        ↓
    [Encoder Layer] × N
        ↓
    编码表示

Decoder（解码器）：
目标输入 → Embedding + Positional Encoding
        ↓
    [Decoder Layer] × N
        ↓
    输出 → Linear → Softmax → 概率

应用：
- 机器翻译：Encoder-Decoder
- BERT：Encoder-only
- GPT：Decoder-only
```

</div>

### 1.2 Encoder 结构

<div class="formula-box">

```
Encoder Layer：
输入
  ↓
Multi-Head Attention（Self-Attention）
  ↓
Add & Norm（残差 + 层归一化）
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
输出

× N 层（通常 6 层）
```

</div>

### 1.3 Decoder 结构

<div class="formula-box">

```
Decoder Layer：
输入
  ↓
Masked Multi-Head Attention（防止看到未来）
  ↓
Add & Norm
  ↓
Cross Attention（关注 Encoder 输出）
  ↓
Add & Norm
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
输出

× N 层（通常 6 层）
```

</div>

---

## 2. 位置编码（Positional Encoding）

### 2.1 为什么需要位置编码？

<div class="formula-box">

```
问题：
Self-Attention 是排列不变的

例子：
"我 爱 你" vs "你 爱 我"
Attention 计算结果相同！

解决：
注入位置信息
```

</div>

### 2.2 正弦位置编码

<div class="formula-box">

```
公式：
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
- pos：位置（0, 1, 2, ...）
- i：维度索引（0, 1, 2, ..., d_model/2-1）
- d_model：嵌入维度（通常 512）

性质：
- 每个位置有唯一编码
- 相对位置可以学习（PE(pos+k) 可由 PE(pos) 线性表示）
- 可以外推到更长序列
```

</div>

<div class="formula-box">

```python
import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        
        # 位置
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算 sin 和 cos 的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 应用 sin 和 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 调整形状 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不参与梯度更新）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

</div>

### 2.3 可学习位置编码

<div class="formula-box">

```
BERT 使用可学习位置编码：

PE = nn.Embedding(max_len, d_model)
位置编码 = PE(position_ids)

优势：
- 可以学习最优的位置表示
- 更灵活

劣势：
- 不能外推到更长的序列
- 需要更多参数
```

</div>

---

## 3. Encoder 实现

### 3.1 Encoder Layer

<div class="formula-box">

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-Head Attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        
        # Self-Attention
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        attn_output = self.dropout(attn_output)
        
        # Add & Norm
        x = self.norm1(x + attn_output)
        
        # Feed-Forward
        ffn_output = self.ffn(x)
        
        # Add & Norm
        x = self.norm2(x + ffn_output)
        
        return x
```

</div>

### 3.2 Encoder Stack

<div class="formula-box">

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder Layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        
        # Embedding + Positional Encoding
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Encoder Layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Layer Norm
        x = self.norm(x)
        
        return x
```

</div>

---

## 4. Decoder 实现

### 4.1 Decoder Layer

<div class="formula-box">

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Masked Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        # x: (batch, tgt_seq_len, d_model)
        # encoder_output: (batch, src_seq_len, d_model)
        
        # Masked Self-Attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        attn_output = self.dropout(attn_output)
        
        # Add & Norm
        x = self.norm1(x + attn_output)
        
        # Cross Attention
        cross_output, _ = self.cross_attn(x, encoder_output, encoder_output, attn_mask=memory_mask)
        cross_output = self.dropout(cross_output)
        
        # Add & Norm
        x = self.norm2(x + cross_output)
        
        # Feed-Forward
        ffn_output = self.ffn(x)
        
        # Add & Norm
        x = self.norm3(x + ffn_output)
        
        return x
```

</div>

### 4.2 Decoder Stack

<div class="formula-box">

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        # x: (batch, tgt_seq_len)
        
        # Embedding + Positional Encoding
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Decoder Layers
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)
        
        # Layer Norm
        x = self.norm(x)
        
        return x
```

</div>

---

## 5. 掩码（Mask）机制

### 5.1 Padding Mask

<div class="formula-box">

```
用途：
忽略填充的 token（<pad>）

生成：
src_mask = (src != pad_id)  # True 表示有效位置

应用：
在 Attention 中，将无效位置的分数设为 -1e9
```

</div>

<div class="formula-box">

```python
def generate_padding_mask(seq, pad_id=0):
    # seq: (batch, seq_len)
    # 返回：(batch, 1, 1, seq_len)
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)
```

</div>

### 5.2 Look-ahead Mask（Decoder）

<div class="formula-box">

```
用途：
防止 Decoder 看到未来位置

生成：
size = tgt_seq_len
mask = torch.triu(torch.ones(size, size), diagonal=1).bool()

示例（size=4）：
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]
0=可见，1=遮蔽
```

</div>

<div class="formula-box">

```python
def generate_look_ahead_mask(size):
    # 生成上三角掩码
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    # (seq_len, seq_len)
    return mask
```

</div>

---

## 6. 完整 Transformer

### 6.1 Transformer 类

<div class="formula-box">

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_layers=6, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: (batch, src_seq_len)
        # tgt: (batch, tgt_seq_len)
        
        # Encoder
        encoder_output = self.encoder(src, src_mask)
        
        # Decoder
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask=tgt_mask)
        
        # Output
        output = self.fc_out(decoder_output)
        
        return output
```

</div>

### 6.2 训练循环

<div class="formula-box">

```python
import torch.optim as optim

# 模型
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.1
)

# 优化器（Adam + Learning Rate Scheduler）
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Learning Rate Scheduler（Transformer 专用）
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

scheduler = TransformerLRScheduler(optimizer, d_model=512)

# 训练
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for src_batch, tgt_batch in dataloader:
        optimizer.zero_grad()
        
        # 生成掩码
        src_mask = generate_padding_mask(src_batch)
        tgt_mask = generate_look_ahead_mask(tgt_batch.size(1))
        
        # 前向传播
        output = model(src_batch, tgt_batch[:-1], src_mask, tgt_mask)
        
        # 计算损失
        loss = criterion(output.view(-1, output.size(-1)), tgt_batch[1:].reshape(-1))
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        scheduler.step()
```

</div>

---

## 7. 架构变体

### 7.1 BERT（Encoder-only）

<div class="formula-box">

```
架构：
只使用 Encoder 部分

预训练任务：
1. Masked LM：随机 mask 15% 的词，预测被 mask 的词
2. Next Sentence Prediction：预测两句是否连续

特点：
- 双向上下文
- 适合理解类任务（分类、问答）
```

</div>

### 7.2 GPT（Decoder-only）

<div class="formula-box">

```
架构：
只使用 Decoder 部分（去掉 Cross Attention）

预训练任务：
自回归语言建模：预测下一个词

特点：
- 单向上下文（从左到右）
- 适合生成类任务（文本生成、对话）
```

</div>

### 7.3 T5（Encoder-Decoder）

<div class="formula-box">

```
架构：
标准 Encoder-Decoder

预训练任务：
Text-to-Text：所有任务都转化为文本生成

特点：
- 统一框架
- 适合 seq2seq 任务（翻译、摘要）
```

</div>

---

## 8. 实战项目

### 项目 1：机器翻译（英→法）

<div class="formula-box">

```python
# 完整训练流程
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 1. 数据准备
train_data = Multi30k(split='train', language_pair=('en', 'de'))

# 2. 构建词表
def yield_tokens(data_iter, lang):
    tokenizer = get_tokenizer('basic')
    for src, tgt in data_iter:
        yield tokenizer(src) if lang == 'src' else tokenizer(tgt)

src_vocab = build_vocab_from_iterator(yield_tokens(train_data, 'src'), specials=['<pad>', '<bos>', '<eos>', '<unk>'])
tgt_vocab = build_vocab_from_iterator(yield_tokens(train_data, 'tgt'), specials=['<pad>', '<bos>', '<eos>', '<unk>'])

# 3. 创建模型
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6
)

# 4. 训练（见上方训练循环）
```

</div>

---

## 📚 学习资源

### 论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [BERT](https://arxiv.org/abs/1810.04805) - BERT 论文
- [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - GPT 论文

### 代码

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 逐行注释
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 工业级实现

### 可视化

- [Transformer 可视化工具](https://www.bert-viz.org/)
- [Attention 可视化](https://github.com/jessevig/bert-viz)

---

## ✅ 学习检查清单

- [ ] 理解 Transformer 整体架构
- [ ] 理解位置编码原理
- [ ] 掌握 Encoder 实现
- [ ] 掌握 Decoder 实现
- [ ] 理解掩码机制
- [ ] 能从零实现 Transformer
- [ ] 理解 BERT 与 GPT 的差异
- [ ] 完成至少 1 个实战项目

---

*最后更新：2026-04-22*
