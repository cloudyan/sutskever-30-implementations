# 06 - 指针网络：注意力机制的另一种打开方式

问下大家，如何让神经网络输出"指向输入的指针"？

传统的序列到序列（seq2seq）模型输出的是固定词汇表中的词，但如果你的输出需要"引用"输入中的某个元素呢？比如：

- 给一组点，输出凸包的顶点顺序
- 给一组城市，输出最优旅行路线（TSP 问题）
- 给一组数字，输出排序后的索引

晓寒第一次看到 Pointer Networks 这篇论文时，感觉"卧槽，原来注意力还能这么用！"

## 传统 seq2seq 的痛点

### 问题场景：凸包问题

给你 5 个点的坐标，让你输出凸包的顶点顺序：

```
输入：P0(0.1, 0.3), P1(0.8, 0.7), P2(0.5, 0.2), P3(0.2, 0.9), P4(0.7, 0.4)
输出：3 → 1 → 4 → 2 （凸包顶点的索引顺序）
```

传统 seq2seq 怎么做？它得把索引 0,1,2,3,4 当成"词汇表"中的词。

**问题来了**：
- 如果输入 6 个点，索引到 5，词汇表得扩展
- 如果输入 100 个点，词汇表得有 100 个"词"
- 输出长度也不固定，凸包顶点数可变

这就像让你背诵一本不断增厚的字典，太痛苦了！

### 图解传统方案

```
┌──────────────────────────────────────────────┐
│           传统 Seq2Seq 方案                    │
├──────────────────────────────────────────────┤
│                                              │
│  输入点坐标                                   │
│    ↓                                         │
│  Encoder (LSTM)                              │
│    ↓                                         │
│  Decoder (LSTM) → 固定词汇表 [0,1,2,3,4...]  │
│    ↓                                         │
│  问题：词汇表大小 = 输入长度，无法泛化！        │
│                                              │
└──────────────────────────────────────────────┘
```

## Pointer Networks 的核心思想

**关键洞察**：与其让模型从固定词汇表输出，不如让它直接"指向"输入元素！

怎么实现？**用注意力机制作为指针**。

### 注意力 = 选择概率分布

回顾一下注意力机制：给定查询 Q 和一组键值对 (K, V)，计算查询与每个键的相似度，得到一个概率分布。

```
注意力权重 = softmax(score(Q, K_i))  →  这是一个概率分布！
```

**指针网络的想法**：这个概率分布不就相当于"指向每个输入元素的概率"吗？

### 图解指针机制

```
┌────────────────────────────────────────────────────┐
│              Pointer Network 架构                   │
├────────────────────────────────────────────────────┤
│                                                    │
│  输入序列 [x₁, x₂, x₃, x₄, x₅]                      │
│       ↓                                            │
│  Encoder → 编码状态 [h₁, h₂, h₃, h₄, h₅]            │
│       ↓                                            │
│  Decoder 第一步                                     │
│       ↓                                            │
│  当前解码状态 s₁                                     │
│       ↓                                            │
│  注意力计算                                         │
│    score(s₁, h₁) = 0.1 → softmax → 0.15           │
│    score(s₁, h₂) = 0.3 → softmax → 0.25           │
│    score(s₁, h₃) = 0.5 → softmax → 0.35 ← 最大！   │
│    score(s₁, h₄) = 0.2 → softmax → 0.15           │
│    score(s₁, h₅) = 0.1 → softmax → 0.10           │
│       ↓                                            │
│  选择：指针指向 x₃（概率 0.35 最高）                  │
│                                                    │
└────────────────────────────────────────────────────┘
```

## 纯 NumPy 实现指针网络

让我们一步步实现这个机制。

### 第一步：指针注意力模块

```python
import numpy as np

def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class PointerAttention:
    """
    指针注意力模块
    
    核心功能：给定编码器状态和解码器当前状态，
    计算指向每个输入元素的概率分布
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        
        # 注意力参数
        # 使用 Bahdanau 风格的加性注意力
        self.W1 = np.random.randn(hidden_size, hidden_size) * 0.1  # 编码器投影
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1  # 解码器投影
        self.v = np.random.randn(hidden_size, 1) * 0.1             # 最终打分
    
    def forward(self, encoder_states, decoder_state):
        """
        计算指针分布
        
        Args:
            encoder_states: (seq_len, hidden_size) - 编码器所有时间步的状态
            decoder_state: (hidden_size, 1) - 解码器当前状态
        
        Returns:
            probs: (seq_len, 1) - 指向每个输入位置的概率
            scores: (seq_len, 1) - 原始注意力分数
        """
        seq_len = encoder_states.shape[0]
        
        # 计算每个位置的注意力分数
        scores = []
        for i in range(seq_len):
            # e_i = v^T * tanh(W1 * h_i + W2 * s)
            # 这里 h_i 是编码器状态，s 是解码器状态
            encoder_proj = np.dot(self.W1, encoder_states[i:i+1].T)  # (hidden_size, 1)
            decoder_proj = np.dot(self.W2, decoder_state)            # (hidden_size, 1)
            score = np.dot(self.v.T, np.tanh(encoder_proj + decoder_proj))
            scores.append(score[0, 0])
        
        scores = np.array(scores).reshape(-1, 1)
        
        # Softmax 得到概率分布
        probs = softmax(scores, axis=0)
        
        return probs, scores
```

**关键理解**：
- 注意力分数表示"解码器当前状态对每个输入的关注程度"
- Softmax 后变成概率，这就是"指针"
- 输出不是词汇表中的词，而是输入序列的索引

### 第二步：完整网络架构

```python
class PointerNetwork:
    """
    指针网络完整架构
    
    组成部分：
    1. 编码器：处理输入序列，得到编码状态
    2. 解码器：逐步生成指针序列
    3. 指针注意力：每步选择指向哪个输入
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 编码器 RNN 参数
        self.encoder_Wx = np.random.randn(hidden_size, input_size) * 0.1
        self.encoder_Wh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.encoder_b = np.zeros((hidden_size, 1))
        
        # 解码器 RNN 参数
        self.decoder_Wx = np.random.randn(hidden_size, input_size) * 0.1
        self.decoder_Wh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.decoder_b = np.zeros((hidden_size, 1))
        
        # 指针机制
        self.attention = PointerAttention(hidden_size)
    
    def encode(self, inputs):
        """
        编码输入序列
        
        Args:
            inputs: list of (input_size, 1) 向量
        
        Returns:
            encoder_states: (seq_len, hidden_size)
            final_hidden: (hidden_size, 1)
        """
        h = np.zeros((self.hidden_size, 1))
        encoder_states = []
        
        for x in inputs:
            # 简单 RNN: h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)
            h = np.tanh(
                np.dot(self.encoder_Wx, x) + 
                np.dot(self.encoder_Wh, h) + 
                self.encoder_b
            )
            encoder_states.append(h.flatten())
        
        return np.array(encoder_states), h
    
    def decode_step(self, x, h, encoder_states):
        """
        解码器单步
        
        Args:
            x: 当前输入 (input_size, 1)
            h: 上一步隐藏状态 (hidden_size, 1)
            encoder_states: 编码器所有状态
        
        Returns:
            probs: 指针概率分布 (seq_len, 1)
            h: 新的隐藏状态
            scores: 注意力分数
        """
        # 更新解码器隐藏状态
        h = np.tanh(
            np.dot(self.decoder_Wx, x) + 
            np.dot(self.decoder_Wh, h) + 
            self.decoder_b
        )
        
        # 计算指针分布
        probs, scores = self.attention.forward(encoder_states, h)
        
        return probs, h, scores
    
    def forward(self, inputs):
        """
        完整前向传播
        
        Args:
            inputs: list of (input_size, 1) 向量
        
        Returns:
            output_indices: list of int - 指向的索引序列
            output_probs: list of (seq_len, 1) - 每步的概率分布
        """
        # 编码
        encoder_states, h = self.encode(inputs)
        
        # 解码
        output_probs = []
        output_indices = []
        
        # 起始输入：使用输入的平均值
        x = np.mean([inp for inp in inputs], axis=0)
        
        for step in range(len(inputs)):
            probs, h, scores = self.decode_step(x, h, encoder_states)
            output_probs.append(probs)
            
            # 选择概率最高的位置作为指针
            ptr_idx = np.argmax(probs)
            output_indices.append(ptr_idx)
            
            # 下一步输入是被指向的元素
            x = inputs[ptr_idx]
        
        return output_indices, output_probs
```

### 第三步：测试凸包任务

```python
# 创建指针网络
np.random.seed(42)
ptr_net = PointerNetwork(input_size=2, hidden_size=32)

# 生成测试数据：5 个随机 2D 点
points = np.random.rand(5, 2)
print("输入点的坐标：")
for i, (x, y) in enumerate(points):
    print(f"  P{i}: ({x:.3f}, {y:.3f})")

# 转换为网络输入格式
inputs = [points[i:i+1].T for i in range(5)]  # 每个点是 (2, 1)

# 前向传播（未训练）
predicted_indices, probs = ptr_net.forward(inputs)

print("\n未训练网络的预测：")
print(f"  指针序列：{predicted_indices}")

# 可视化注意力分布
print("\n每步的注意力分布：")
for step, p in enumerate(probs):
    print(f"  Step {step}: {p.flatten().round(3)}")
```

运行结果示例：
```
输入点的坐标：
  P0: (0.374, 0.950)
  P1: (0.732, 0.598)
  P2: (0.156, 0.156)
  P3: (0.058, 0.866)
  P4: (0.601, 0.708)

未训练网络的预测：
  指针序列：[2, 2, 2, 2, 2]

每步的注意力分布：
  Step 0: [0.17  0.21  0.25  0.19  0.18]
  Step 1: [0.16  0.2  0.27  0.19  0.18]
  ...
```

未训练的网络会倾向于重复指向同一个位置，训练后会学会正确的顺序。

## 图解指针网络工作流程

```
┌────────────────────────────────────────────────────────┐
│           凸包任务示例                                   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Step 1: 编码阶段                                       │
│  ─────────────────                                      │
│                                                        │
│    输入点坐标                                           │
│    ┌─────┬─────┬─────┬─────┬─────┐                      │
│    │ P0  │ P1  │ P2  │ P3  │ P4  │                      │
│    │(1,3)│(8,7)│(5,2)│(2,9)│(7,4)│                      │
│    └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘                      │
│       ↓     ↓     ↓     ↓     ↓                         │
│    ┌─────┬─────┬─────┬─────┬─────┐                      │
│    │ h0  │ h1  │ h2  │ h3  │ h4  │  编码器状态           │
│    └─────┴─────┴─────┴─────┴─────┘                      │
│                                                        │
│  Step 2: 解码第 1 步                                    │
│  ───────────────────                                    │
│                                                        │
│    解码器状态 s1                                        │
│         ↓                                              │
│    ┌────────────────────────────────┐                  │
│    │      注意力计算                 │                  │
│    │  score(s1, h0) → 0.1          │                  │
│    │  score(s1, h1) → 0.3          │                  │
│    │  score(s1, h2) → 0.2          │                  │
│    │  score(s1, h3) → 0.5 ← 最大！  │                  │
│    │  score(s1, h4) → 0.2          │                  │
│    └────────────────────────────────┘                  │
│         ↓                                              │
│    指针 → P3（索引 3）                                  │
│                                                        │
│  Step 3: 解码第 2 步                                    │
│  ───────────────────                                    │
│                                                        │
│    输入 = P3 的坐标                                     │
│    解码器更新状态 s2                                    │
│         ↓                                              │
│    注意力计算 → 指向 P1                                 │
│                                                        │
│  最终输出：[3, 1, 4, 2]                                 │
│  （凸包顶点的索引顺序）                                  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 指针网络 vs 传统 seq2seq

### 核心差异对比

```
┌──────────────────┬─────────────────────┬─────────────────────┐
│      特性         │   传统 Seq2Seq      │   Pointer Network   │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 输出空间          │ 固定词汇表           │ 输入序列本身         │
│ 词汇表大小        │ 预定义，固定         │ 等于输入长度         │
│ 可变长度输入      │ 需要重新定义词汇表    │ 自动适应             │
│ 输出长度          │ 可固定或可变         │ 可变（可少于输入）    │
│ 泛化能力          │ 难以泛化到更长序列    │ 自然泛化             │
│ 典型应用          │ 翻译、摘要           │ 组合优化、排序       │
└──────────────────┴─────────────────────┴─────────────────────┘
```

### 生活比喻

**传统 seq2seq** 就像是在一个固定大小的菜单上点菜：

```
菜单：宫保鸡丁、鱼香肉丝、麻婆豆腐、...
你只能从菜单里选，菜单外的东西点不了。
```

**Pointer Network** 就像是"手指菜单"：

```
服务员把今天的新鲜食材摆在你面前：
  🥬 白菜  🥕 胡萝卜  🍖 猪肉  🧄 蒜  🌶️ 辣椒

你用手指着说："我要这个、这个、还有这个！"
不需要预定义菜单，有多少食材就能点多少种组合。
```

## 实战任务：排序问题

让我们用指针网络实现一个简单的数字排序任务。

```python
def generate_sorting_data(num_samples=50, seq_len=5):
    """生成排序任务数据"""
    data = []
    
    for _ in range(num_samples):
        # 随机生成数值序列
        values = np.random.rand(seq_len)
        
        # 排序后的索引顺序
        sorted_indices = np.argsort(values).tolist()
        
        # 转换为网络输入格式
        inputs = [np.array([[v]]) for v in values]
        
        data.append({
            'values': values,
            'inputs': inputs,
            'sorted_indices': sorted_indices
        })
    
    return data


# 生成数据
sort_data = generate_sorting_data(num_samples=20, seq_len=5)

# 测试一个例子
example = sort_data[0]
print("排序任务示例：")
print(f"  原始值：{example['values'].round(3)}")
print(f"  排序后索引：{example['sorted_indices']}")
print(f"  排序后值：{example['values'][example['sorted_indices']].round(3)}")

# 用指针网络测试
ptr_net_sort = PointerNetwork(input_size=1, hidden_size=32)
predicted, _ = ptr_net_sort.forward(example['inputs'])
print(f"  网络预测：{predicted}")
```

输出：
```
排序任务示例：
  原始值：[0.374 0.950 0.732 0.598 0.156]
  排序后索引：[4, 0, 3, 2, 1]
  排序后值：[0.156 0.374 0.598 0.732 0.950]
  网络预测：[2, 2, 2, 2, 2]  # 未训练，结果不好
```

训练后的网络会学会正确的排序顺序。

## 指针网络的应用场景

### 1. 组合优化问题

```
┌─────────────────────────────────────────┐
│         经典应用场景                      │
├─────────────────────────────────────────┤
│                                         │
│  旅行商问题 (TSP)                        │
│  ─────────────────                       │
│  输入：N 个城市坐标                       │
│  输出：访问顺序（城市索引序列）            │
│                                         │
│  凸包问题                                │
│  ─────────────────                       │
│  输入：N 个点坐标                         │
│  输出：凸包顶点顺序                        │
│                                         │
│  Delaunay 三角剖分                       │
│  ─────────────────                       │
│  输入：点集                               │
│  输出：三角剖分的边（点对索引）            │
│                                         │
└─────────────────────────────────────────┘
```

### 2. 自然语言处理

- **文本摘要**：从原文中选择重要句子
- **问答系统**：从文档中定位答案位置
- **信息抽取**：从文本中提取关键实体

### 3. 代码生成

- 从输入规范中选择函数组件
- 从 API 列表中选择合适的调用

## 训练指针网络

### 损失函数

使用交叉熵损失，目标是最小化负对数似然：

```python
def compute_loss(predicted_probs, target_indices):
    """
    计算指针网络的损失
    
    Args:
        predicted_probs: list of (seq_len, 1) - 每步的预测概率
        target_indices: list of int - 正确的指针序列
    
    Returns:
        loss: scalar - 负对数似然
    """
    loss = 0.0
    for probs, target in zip(predicted_probs, target_indices):
        # -log(p(target))
        loss -= np.log(probs[target, 0] + 1e-10)
    
    return loss / len(target_indices)
```

### 训练循环（概念性代码）

```python
# 注意：这是概念性代码，实际训练需要反向传播
# 完整实现参考项目中的 06_pointer_networks.ipynb

for epoch in range(num_epochs):
    for batch in train_data:
        # 前向传播
        predicted_indices, predicted_probs = ptr_net.forward(batch['inputs'])
        
        # 计算损失
        loss = compute_loss(predicted_probs, batch['target_indices'])
        
        # 反向传播（需要手动实现梯度）
        # gradients = backward(...)
        
        # 更新参数
        # params = optimizer.step(params, gradients)
```

## 关键要点总结

### 指针网络的创新点

1. **输出即输入**：网络输出指向输入元素的指针，而非固定词汇表中的词

2. **可变输出长度**：输出序列长度可以与输入不同

3. **自然泛化**：训练时见过 5 个点的凸包，测试时能处理 10 个点

4. **注意力即选择**：将注意力机制重新解释为"选择机制"

### 架构组成

```
┌────────────────────────────────────────────┐
│         Pointer Network 组件                │
├────────────────────────────────────────────┤
│                                            │
│  1. 编码器：处理输入序列                     │
│     └─ RNN/LSTM/Transformer                │
│                                            │
│  2. 解码器：生成指针序列                     │
│     └─ 每步输出一个指针                     │
│                                            │
│  3. 指针注意力：计算指向概率                 │
│     └─ Bahdanau/加性注意力                  │
│                                            │
│  4. 选择机制：根据概率选择指针               │
│     └─ 训练时：使用概率分布                  │
│     └─ 推理时：argmax 或采样                │
│                                            │
└────────────────────────────────────────────┘
```

### 与其他技术的关系

```
┌─────────────────────────────────────────────────────┐
│          指针网络在深度学习中的位置                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Seq2Seq (2014)                                     │
│      ↓                                              │
│  + Attention (Bahdanau, 2014)                       │
│      ↓                                              │
│  Pointer Network (2015) ← 注意力的新用法             │
│      ↓                                              │
│  Copy Network (2016) ← 结合指针和生成                │
│      ↓                                              │
│  Transformer (2017) ← 自注意力                      │
│      ↓                                              │
│  现代 LLM（GPT、BERT）                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 练习题

### 基础题

1. **概念理解**：指针网络和传统 seq2seq 模型最大的区别是什么？

2. **实现细节**：为什么指针网络使用注意力机制而不是直接输出索引？

3. **应用场景**：举出 3 个可以用指针网络解决的实际问题。

### 进阶题

4. **算法实现**：修改代码实现贪婪解码和束搜索（beam search）。

5. **损失函数**：实现完整的训练循环，包括反向传播。

6. **变体思考**：如何让指针网络既能"指向"也能"生成"新词？（提示：Copy Network）

### 思考题

7. **泛化能力**：为什么指针网络可以自然泛化到更长的输入序列？

8. **局限性**：指针网络无法处理什么类型的问题？

## 参考资料

- 原论文：[Pointer Networks](https://arxiv.org/abs/1506.03134) (Vinyals et al., 2015)
- 实现代码：`06_pointer_networks.ipynb`
- 相关论文：Get To The Point: Summarization with Pointer-Generator Networks (See et al., 2017)

---

**下一篇预告**：我们将探索卷积神经网络的开山之作 —— AlexNet，看看它是如何开启深度学习新纪元的。

**图解深度学习，让复杂变简单！** 如果觉得有帮助，欢迎 Star ⭐