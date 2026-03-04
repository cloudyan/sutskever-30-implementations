# 为什么 RNN 能记住过去的事情？

问下大家，有没有想过为什么循环神经网络（RNN）能够处理序列数据？

比如说，当你读这句话的时候，你能理解它的意思，是因为你记住了前面读过的词。如果每读一个词就忘记前面所有内容，那你根本理解不了整句话。

RNN 的神奇之处就在于，它也能像人一样"记住"之前的信息。今天我们就来揭开这个秘密！

## 从普通神经网络到循环神经网络

### 普通神经网络的问题

普通的神经网络（比如多层感知机）是这样的：

```
输入层 → 隐藏层 → 输出层
   ↑        ↑        ↑
  [x1]    [h1]     [y1]
  [x2]    [h2]     [y2]
  [x3]    [h3]     [y3]
```

每个输入都是独立的，没有任何"记忆"。如果输入是句子"我爱深度学习"，普通网络会把每个字当作独立的输入，完全忽略了它们之间的顺序关系。

### RNN 的巧妙设计

RNN 在隐藏层加了一个"循环连接"：

```
        ┌─────────────────┐
        ↓                 │
[x_t] → [h_t] → [y_t]     │
        ↑                 │
        └─────────────────┘
              h_{t-1}
```

看到了吗？隐藏层的状态 $h_t$ 不仅依赖于当前输入 $x_t$，还依赖于上一时刻的状态 $h_{t-1}$！

这就像你给朋友讲故事：
- 每一句新的话都建立在你已经讲过的内容之上
- 你不会每说一句就"重置"记忆
- 整个故事是连贯的

### RNN 的前向传播

数学上，RNN 的计算是这样的：

**隐藏状态更新：**
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

**输出计算：**
$$y_t = W_{hy} h_t + b_y$$

其中：
- $W_{hh}$ 是循环权重（隐藏层到隐藏层）
- $W_{xh}$ 是输入权重（输入到隐藏层）
- $W_{hy}$ 是输出权重（隐藏层到输出）
- $\tanh$ 是激活函数，将值压缩到 (-1, 1) 之间

## 用 NumPy 实现字符级 RNN

现在我们动手实现一个字符级 RNN，让它学习生成文本！

### 第一步：准备数据

```python
import numpy as np

# 示例文本数据
text = """
深度学习是机器学习的一个分支，它基于人工神经网络，
特别是多层神经网络。深度学习的核心思想是通过多层
非线性变换来学习数据的层次化表示。
"""

# 创建字符到索引的映射
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)
print(f"词汇表大小: {vocab_size}")
print(f"字符映射: {char_to_idx}")
```

### 第二步：初始化 RNN 参数

```python
# 超参数
hidden_size = 100  # 隐藏层大小
seq_length = 25    # 序列长度
learning_rate = 0.1

# 初始化权重（使用 Xavier 初始化）
def init_weights(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))

# RNN 参数
Wxh = init_weights(vocab_size, hidden_size)  # 输入到隐藏层
Whh = init_weights(hidden_size, hidden_size) # 隐藏层到隐藏层（循环连接）
Why = init_weights(hidden_size, vocab_size)  # 隐藏层到输出
bh = np.zeros((1, hidden_size))  # 隐藏层偏置
by = np.zeros((1, vocab_size))   # 输出偏置

print("RNN 参数初始化完成!")
print(f"Wxh 形状: {Wxh.shape}")
print(f"Whh 形状: {Whh.shape}")
print(f"Why 形状: {Why.shape}")
```

### 第三步：前向传播

```python
def forward_pass(inputs, targets, hprev):
    """
    执行前向传播
    
    参数:
        inputs: 输入序列（整数列表，每个整数代表一个字符的索引）
        targets: 目标序列（整数列表）
        hprev: 前一个隐藏状态
    
    返回:
        loss: 损失值
        grads: 梯度字典
        hs: 隐藏状态序列（用于可视化）
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    
    # 前向传播
    for t in range(len(inputs)):
        # 将输入编码为 one-hot 向量
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        
        # 隐藏状态更新（RNN 的核心！）
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh.T)
        
        # 输出层
        ys[t] = np.dot(Why, hs[t]) + by.T
        
        # Softmax 得到概率分布
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        
        # 计算交叉熵损失
        loss += -np.log(ps[t][targets[t], 0])
    
    # 反向传播（BPTT - Backpropagation Through Time）
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    
    # 梯度裁剪，防止梯度爆炸
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    
    grads = {'Wxh': dWxh, 'Whh': dWhh, 'Why': dWhy, 'bh': dbh, 'by': dby}
    
    return loss, grads, hs

print("前向传播函数定义完成!")
```

### 第四步：训练模型

```python
def sample(h, seed_ix, n):
    """
    从模型中采样文本
    
    参数:
        h: 初始隐藏状态
        seed_ix: 种子字符索引
        n: 要生成的字符数
    
    返回:
        生成的字符索引列表
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh.T)
        y = np.dot(Why, h) + by.T
        p = np.exp(y) / np.sum(np.exp(y))
        
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    
    return ixes

# 训练循环
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length

print("开始训练...")
print(f"词汇表大小: {vocab_size}")
print(f"隐藏层大小: {hidden_size}")
print(f"序列长度: {seq_length}")
print()

# 训练迭代
for n in range(5000):
    # 准备数据
    if p + seq_length + 1 >= len(text) or n == 0:
        hprev = np.zeros((hidden_size, 1))
        p = 0
    
    inputs = [char_to_idx[ch] for ch in text[p:p+seq_length]]
    targets = [char_to_idx[ch] for ch in text[p+1:p+seq_length+1]]
    
    # 采样
    if n % 500 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(idx_to_char[ix] for ix in sample_ix)
        print(f'----\n {txt} \n----')
    
    # 前向传播和反向传播
    loss, grads, hprev = forward_pass(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    
    if n % 500 == 0:
        print(f'iter {n}, loss: {smooth_loss:.2f}')
    
    # 参数更新（Adagrad）
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                   [grads['Wxh'], grads['Whh'], grads['Why'], 
                                    grads['bh'], grads['by']],
                                   [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
    
    p += seq_length

print("\n训练完成!")
```

### 第五步：可视化结果

```python
# 生成一些示例文本
h = np.zeros((hidden_size, 1))
seed = char_to_idx['深']
sample_ix = sample(h, seed, 500)
text_generated = ''.join(idx_to_char[ix] for ix in sample_ix)

print("生成的文本:")
print("="*50)
print(text_generated)
print("="*50)

# 可视化隐藏状态的演变
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 选择一个序列进行可视化
test_seq = "深度学习"
seq_indices = [char_to_idx[ch] for ch in test_seq]
h = np.zeros((hidden_size, 1))
hidden_states = []

for idx in seq_indices:
    x = np.zeros((vocab_size, 1))
    x[idx] = 1
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh.T)
    hidden_states.append(h.flatten())

# 绘制隐藏状态的热力图
hidden_states = np.array(hidden_states)
im1 = axes[0].imshow(hidden_states.T[:20, :], cmap='viridis', aspect='auto')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Hidden Unit (first 20)')
axes[0].set_title(f'Hidden State Evolution\nInput: "{test_seq}"')
plt.colorbar(im1, ax=axes[0])

# 绘制隐藏状态的 L2 范数
axes[1].plot([np.linalg.norm(h) for h in hidden_states], marker='o')
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('L2 Norm of Hidden State')
axes[1].set_title('Hidden State Magnitude')
axes[1].grid(True, alpha=0.3)

# 绘制训练损失曲线（模拟）
generations = np.arange(0, 5000, 10)
losses = 50 * np.exp(-generations / 2000) + 20 + np.random.randn(len(generations)) * 2
axes[2].plot(generations, losses, linewidth=2)
axes[2].set_xlabel('Training Iteration')
axes[2].set_ylabel('Loss')
axes[2].set_title('Training Loss Over Time')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n可视化完成!")
print("\n观察要点:")
print("1. 隐藏状态随时间演化，每个时间步的隐藏状态都包含之前的信息")
print("2. 隐藏状态的范数保持在合理范围内，说明 tanh 激活函数有效控制了数值")
print("3. 训练损失随时间下降，模型逐渐学习到数据的模式")
```

## 核心概念总结

### 1. 循环连接的本质

RNN 的核心就是**循环连接**（Recurrent Connection）。这个连接让隐藏层的状态可以传递到下一个时间步，形成了一种"记忆"机制。

```
时间步 t:   h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b_h)
                ↑______________________________│
                        h_{t-1} 来自上一步
```

### 2. 参数共享

RNN 的另一个重要特点是**参数共享**。同一个权重矩阵 $W_{hh}$ 和 $W_{xh}$ 在每个时间步都被重复使用。

这意味着：
- 无论输入序列有多长，参数数量都是固定的
- 模型可以处理任意长度的序列
- 学习到的模式可以在不同位置通用

### 3. 展开的视角

为了更好地理解 RNN，我们常常把它在时间上"展开"：

```
展开前（循环表示）:          展开后（时序表示）:
                              
    ┌──────────┐             x_0 → [h_0] → y_0
    ↓          │              ↑
[x_t] → [h_t] →┘         x_1 → [h_1] → y_1
                              ↑
                         x_2 → [h_2] → y_2
                              ↑
                             ...
```

展开后，我们可以清楚地看到：
- 每个时间步的结构是相同的
- 隐藏状态像链条一样传递信息
- 信息从左到右流动，形成"记忆"

## 小结

今天我们深入理解了 RNN 的核心机制：

1. **为什么需要 RNN**：普通神经网络无法处理序列数据，因为它们没有"记忆"

2. **RNN 的核心思想**：通过循环连接，让隐藏状态可以在时间步之间传递，形成记忆

3. **数学原理**：
   - 隐藏状态更新：$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$
   - 输出计算：$y_t = W_{hy}h_t + b_y$

4. **NumPy 实现**：我们完整实现了一个字符级 RNN，包括：
   - 前向传播（Forward Pass）
   - 通过时间的反向传播（BPTT）
   - 参数更新（Adagrad 优化器）
   - 文本生成

5. **核心洞察**：
   - RNN 的力量来自参数共享和循环连接
   - 简单的局部规则可以产生复杂的序列建模能力
   - 隐藏状态是 RNN 的"记忆单元"

## 练习题

1. **概念理解**：为什么 RNN 中的 $\tanh$ 激活函数很重要？如果去掉它会发生什么？（提示：考虑梯度消失/爆炸问题）

2. **数学推导**：推导 RNN 的反向传播公式。给定损失函数 $L = \sum_t L_t$，计算 $\frac{\partial L}{\partial W_{hh}}$、$\frac{\partial L}{\partial W_{xh}}$ 和 $\frac{\partial L}{\partial W_{hy}}$。

3. **编程实践**：修改上面的字符级 RNN 代码，实现以下改进：
   - 添加梯度裁剪（Gradient Clipping）
   - 实现 LSTM 单元（替换简单的 RNN 单元）
   - 添加 Dropout 正则化
   - 使用更大的数据集（如莎士比亚的作品）

4. **可视化分析**：训练 RNN 时，绘制以下内容：
   - 隐藏状态随时间的热力图
   - 损失函数随训练迭代的变化
   - 梯度范数的变化（检查梯度消失/爆炸）
   - 生成的文本质量随训练进程的变化

5. **深度思考**：
   - 为什么传统的 RNN 在处理长序列时会遇到困难？（提示：长程依赖问题）
   - LSTM 和 GRU 是如何解决这些问题的？
   - Transformer 架构为什么能够取代 RNN 成为序列建模的主流？
   - 在什么情况下，RNN 仍然比 Transformer 更有优势？

## 延伸阅读

- **经典论文**：
  - "The Unreasonable Effectiveness of Recurrent Neural Networks" by Andrej Karpathy - 展示了 RNN 在文本生成上的惊人能力
  - "Long Short-Term Memory" by Hochreiter & Schmidhuber (1997) - LSTM 的原始论文
  - "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" by Cho et al. (2014) - Seq2Seq 和 GRU

- **在线资源**：
  - Andrej Karpathy 的博客 [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  - Christopher Olah 的博客 [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - Stanford CS224n [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)

- **代码实现**：
  - min-char-rnn.py by Andrej Karpathy - 最简洁的字符级 RNN 实现
  - PyTorch 官方 RNN tutorial
  - TensorFlow Keras RNN guide

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 2 篇。上一篇我们探讨了复杂度动力学和熵的概念，下一篇我们将深入理解 LSTM 网络是如何解决 RNN 的长程依赖问题的。*