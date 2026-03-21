# 普通神经网络为什么不能处理序列？

问下大家，你有没有想过，为什么普通的神经网络处理不了序列数据？

云言刚开始学深度学习的时候，也觉得奇怪：神经网络不是万能的吗？怎么连句子都处理不了？

直到后来理解了序列数据的本质，才发现卧槽，原来普通神经网络的设计从一开始就没考虑过"变长"这件事！

## 先回顾一下普通神经网络

在深入问题之前，咱们先快速回顾一下普通神经网络（全连接网络）是怎么工作的。

### 标准的全连接网络

```
输入层          隐藏层          输出层
  ○              ○              
  ○    ────→    ○    ────→    ○
  ○              ○              
  ○              ○              
固定维度        固定维度        固定维度
```

全连接网络的核心假设：

| 特点 | 说明 |
|------|------|
| **固定输入维度** | 输入向量的大小必须预先确定 |
| **固定输出维度** | 输出向量的大小也是固定的 |
| **独立同分布** | 每个样本独立处理，互不影响 |
| **一步计算** | 输入→输出，中间计算步数固定 |

### 一个简单的例子

假设我们要做图片分类，输入是一张 28×28 的手写数字图片：

```python
import numpy as np

class SimpleNN:
    """一个简单的全连接神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入向量，形状 (input_size, 1)
               必须是固定大小！
        """
        # 隐藏层
        h = np.tanh(self.W1 @ x + self.b1)
        # 输出层
        y = self.W2 @ h + self.b2
        return y

# 使用示例
nn = SimpleNN(input_size=784, hidden_size=128, output_size=10)

# 输入必须正好是 784 维
image = np.random.randn(784, 1)  # 28×28 = 784
output = nn.forward(image)

print(f"输出形状: {output.shape}")  # (10, 1) - 固定的！
```

这个网络工作得很好，因为图片大小是固定的（28×28）。但是...

## 序列数据的挑战来了！

当我们尝试用这种网络处理序列数据时，问题就来了。

### 挑战一：变长输入

句子长度是不固定的：

```
句子1: "你好"                  → 2 个词
句子2: "今天天气真好"           → 5 个词  
句子3: "深度学习改变了人工智能领域" → 9 个词
```

全连接网络要求输入维度固定，那怎么办？

**方案 A：填充到固定长度？**

```python
# 假设最大长度是 10
max_len = 10

# 句子1 填充
sentence1 = "你好"
padded1 = sentence1 + "□" * 8  # "你好□□□□□□□□"

# 句子2 填充
sentence2 = "今天天气真好"
padded2 = sentence2 + "□" * 5  # "今天天气真好□□□□□"
```

这样做有两个问题：
1. 浪费计算资源（大量无意义的填充）
2. 最大长度设多少？设小了截断信息，设大了浪费更多

**方案 B：每个长度一个网络？**

```
长度2 → 网络1
长度3 → 网络2
长度4 → 网络3
...
长度100 → 网络99
```

这显然不现实！参数爆炸，而且无法泛化到没见过的长度。

### 挑战二：参数爆炸

假设我们想用独立的方式处理序列中的每个位置：

```python
# 错误示范：每个位置用独立的权重
class BadSequenceNet:
    """一个糟糕的序列处理方案"""
    
    def __init__(self, max_seq_len, input_size, hidden_size):
        # 为每个位置创建独立的权重！
        self.weights_per_position = []
        for i in range(max_seq_len):
            W = np.random.randn(hidden_size, input_size)
            self.weights_per_position.append(W)
        
        print(f"总参数量: {max_seq_len * hidden_size * input_size}")
        # 如果 max_seq_len=100, input_size=100, hidden_size=128
        # 参数量 = 100 * 128 * 100 = 1,280,000
        # 序列长度翻倍，参数量翻倍！
    
    def forward(self, sequence):
        """处理序列"""
        outputs = []
        for i, x in enumerate(sequence):
            # 每个位置用不同的权重
            h = np.tanh(self.weights_per_position[i] @ x)
            outputs.append(h)
        return outputs
```

这太蠢了！参数量随序列长度线性增长，而且不同位置之间无法共享知识。

### 挑战三：缺乏记忆

这是最致命的问题。

普通神经网络处理每个输入时都是"失忆"的：

```
时刻 t=1: 输入 "我" → 网络 → 输出 (网络不知道之前发生了什么)
时刻 t=2: 输入 "爱" → 网络 → 输出 (网络也不知道之前发生了什么)
时刻 t=3: 输入 "你" → 网络 → 输出 (网络还是不知道之前发生了什么)
```

举个例子：

```python
# 情感分析：判断句子是积极还是消极
def analyze_sentiment(sentence):
    """用普通网络分析情感 - 错误示范"""
    
    # 假设我们有一个预训练的网络
    nn = SimpleNN(input_size=100, hidden_size=64, output_size=2)
    
    # 逐词处理
    for word in sentence.split():
        word_vector = get_word_vector(word)  # 获取词向量
        output = nn.forward(word_vector)
        # 问题：网络完全不记得之前见过什么词！
    
    return output

# 测试
result1 = analyze_sentiment("这部电影 很 好看")    # 积极？
result2 = analyze_sentiment("这部电影 不 好看")   # 消极？
# 网络无法区分这两个句子，因为它看不到"不"和"好看"的关系！
```

关键问题：**"不"这个词的含义，取决于后面的词是什么。** 普通网络做不到这一点。

## 用生活比喻来理解

### 比喻1：看照片 vs 看电影

| 普通神经网络 | 序列处理任务 |
|-------------|-------------|
| 像看一张张**照片** | 像看一部**电影** |
| 每张照片独立判断 | 需要记住前面的剧情 |
| 不知道上一张是什么 | 才能理解当前这一帧 |
| 看到一张脸 → 判断表情 | 看到一个人笑 → 要知道他之前在哭，才能理解情感变化 |

想象一下，如果你看电影时，每一秒都"失忆"了，你能理解剧情吗？

```
第1秒: 看到一个人拿着枪 → "可能是坏人"
第2秒: (失忆) 看到枪响了 → "谁开的枪？"
第3秒: (失忆) 看到有人倒下 → "谁中枪了？"
```

完全懵逼，对吧？

### 比喻2：逐字阅读

更具体一点，想象你在读这句话：

```
"小明不喜欢吃苹果"
```

普通人阅读时的心智活动：

```
读 "小明" → 记住主角是小明
读 "不"   → 记住有个否定词
读 "喜欢" → 心里想"不"+"喜欢"=不喜欢
读 "吃"   → 继续，不喜欢吃...
读 "苹果" → 最终理解：小明不喜欢吃苹果
```

但如果像普通神经网络那样"失忆"：

```
读 "小明" → 嗯，小明
读 "不"   → (忘掉小明) 不？
读 "喜欢" → (忘掉不) 喜欢？
读 "吃"   → (忘掉喜欢) 吃？
读 "苹果" → (忘掉吃) 苹果？
```

最后你可能得出："苹果...？" 完全不知所云。

## 我们需要什么？

通过上面的分析，处理序列数据需要三个关键能力：

### 1. 记忆机制

网络需要能够"记住"之前看到的内容：

```python
# 我们需要这样的能力
h = initial_state  # 初始记忆

for x in sequence:
    h = update_memory(h, x)  # 用当前输入更新记忆
    y = predict(h)           # 基于记忆做预测
```

### 2. 参数共享

无论序列多长，使用同一套参数：

```python
# 参数共享：同一套权重处理所有位置
W = init_weights()  # 只有一套权重

for x in sequence:
    h = W @ x + ...  # 所有位置用同一套 W
```

### 3. 循环连接

把上一时刻的输出，作为下一时刻的输入：

```
     ┌─────────────────────────┐
     │                         │
     ▼                         │
┌────────┐    ┌────────┐    ┌────────┐
│  x_t   │───→│   RNN  │───→│  y_t   │
└────────┘    └────────┘    └────────┘
                  │
                  ▼
              ┌────────┐
              │  h_t   │ ──→ 传递给下一步
              └────────┘
```

这三点，正是 **RNN（循环神经网络）** 的核心设计！

## 代码对比：普通网络 vs 我们需要的

让我们用代码来感受这个差距：

```python
import numpy as np

# ============ 普通神经网络 ============
class RegularNN:
    """普通神经网络：无记忆，固定输入"""
    
    def __init__(self, input_size, hidden_size):
        self.W = np.random.randn(hidden_size, input_size) * 0.01
        self.b = np.zeros((hidden_size, 1))
    
    def forward(self, x):
        """
        输入 x 必须是固定维度
        处理每个样本时完全"失忆"
        """
        return np.tanh(self.W @ x + self.b)


# ============ 我们期望的序列处理器 ============
class WhatWeNeed:
    """我们需要的序列处理器：有记忆，可变长"""
    
    def __init__(self, input_size, hidden_size):
        # 输入到隐藏层的权重
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01
        # 隐藏层到隐藏层的权重（记忆连接！）
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b = np.zeros((hidden_size, 1))
    
    def forward(self, sequence):
        """
        可以处理任意长度的序列！
        有记忆机制！
        参数共享！
        """
        h = np.zeros((self.Wh.shape[0], 1))  # 初始记忆
        
        outputs = []
        for x in sequence:
            # 关键：h 既依赖当前输入 x，也依赖上一时刻的 h
            h = np.tanh(self.Wx @ x + self.Wh @ h + self.b)
            outputs.append(h)
        
        return outputs, h  # 返回所有输出和最终记忆

# 对比使用
print("=== 普通神经网络 ===")
nn = RegularNN(input_size=10, hidden_size=8)

# 只能处理固定维度的单个输入
x = np.random.randn(10, 1)
out = nn.forward(x)
print(f"输入维度: {x.shape}, 输出维度: {out.shape}")

print("\n=== 我们需要的序列处理器 ===")
seq_net = WhatWeNeed(input_size=10, hidden_size=8)

# 可以处理任意长度的序列！
seq1 = [np.random.randn(10, 1) for _ in range(3)]
seq2 = [np.random.randn(10, 1) for _ in range(5)]

outputs1, final_h1 = seq_net.forward(seq1)
outputs2, final_h2 = seq_net.forward(seq2)

print(f"序列1 长度: {len(seq1)}, 输出数量: {len(outputs1)}")
print(f"序列2 长度: {len(seq2)}, 输出数量: {len(outputs2)}")
print(f"每个输出的维度: {outputs1[0].shape}")
print(f"最终记忆维度: {final_h1.shape}")
```

输出：

```
=== 普通神经网络 ===
输入维度: (10, 1), 输出维度: (8, 1)

=== 我们需要的序列处理器 ===
序列1 长度: 3, 输出数量: 3
序列2 长度: 5, 输出数量: 5
每个输出的维度: (8, 1)
最终记忆维度: (8, 1)
```

注意到关键区别了吗？

| 特性 | 普通网络 | 序列处理器（RNN雏形） |
|------|---------|---------------------|
| 输入长度 | 必须固定 | 任意长度 |
| 记忆能力 | 无 | 有（通过 h 传递） |
| 参数共享 | N/A | 是（同一套 Wx, Wh） |
| 参数量 | 固定 | 固定（不随序列长度增长） |

## 总结

问下大家，现在理解为什么普通神经网络处理不了序列了吗？

核心原因有三个：

1. **变长问题**：普通网络要求固定维度，但序列长度千变万化
2. **参数问题**：如果为每个位置独立建模，参数量爆炸
3. **记忆问题**：最关键的一点——普通网络没有记忆机制

解决方案就是 RNN 的三大设计：

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   记忆机制 ────→ 隐藏状态 h 保存历史信息           │
│                                                     │
│   参数共享 ────→ 同一套权重处理所有位置           │
│                                                     │
│   循环连接 ────→ h_{t-1} → h_t 实现记忆传递       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

下一篇，我们将深入探讨 **RNN 的核心机制**，看看循环连接是如何实现"记忆"的，敬请期待！

---

> 本文是《图解 Char-RNN》系列的第 2 篇。关注「云言 AI」公众号，获取更多深度学习图解教程！