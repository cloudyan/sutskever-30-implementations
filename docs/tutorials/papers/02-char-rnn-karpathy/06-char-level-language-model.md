# 字符级语言模型：让 AI 学会写文章

问下大家，有没有想过怎么让 AI 学会写文章？

2015 年，Karpathy 做了一个有趣的实验：他把莎士比亚的剧本喂给一个简单的 RNN，结果 AI 生成了这样的文本：

```
KING LEAR:
O, if you were present, do not be so;
But you shall not be strange.

DUKE VINCENTIO:
The noble and the complexion of the state
Shall be a man of virtue, and the world
```

卧槽！这居然是 AI 写的！虽然有点语法错误，但那个"莎士比亚味儿"是真的像。

今天我们就来揭开这个魔法：**字符级语言模型**是怎么让 AI 学会"写作"的。

## 什么是字符级语言模型？

### 语言模型的本质

先回答一个核心问题：什么是语言模型？

**语言模型就是预测下一个字符（或词）的概率分布。**

比如输入"今天天气"，语言模型会告诉你：

| 下一个字符 | 概率 |
|-----------|------|
| 真 | 0.35 |
| 很 | 0.25 |
| 不 | 0.15 |
| 好 | 0.10 |
| ... | ... |

就这么简单！但就是这么简单的东西，支撑起了 ChatGPT、Siri、机器翻译...

### 字符级 vs 词级 vs 子词级

处理文本有三种粒度：

| 粒度 | 切分示例 | 词汇表大小 | 优缺点 |
|------|---------|-----------|--------|
| **字符级** | "你好" → "你","好" | 很小（几十到几千） | 无需分词，泛化强，但序列很长 |
| **词级** | "你好世界" → "你好","世界" | 很大（几万到几十万） | 序列短，但需要分词，OOV问题 |
| **子词级** | "unhappiness" → "un","happiness" | 中等（几千到几万） | 兼顾两者，现代主流 |

为什么 Karpathy 选择字符级？

```
┌────────────────────────────────────────────────────────────┐
│                    字符级的优势                            │
├────────────────────────────────────────────────────────────┤
│ ✅ 无需分词：不依赖任何语言学的分词工具                    │
│ ✅ 词汇表小：英文 26 个字母 + 符号，中文几千个汉字         │
│ ✅ 泛化能力强：能生成任何可能的字符串                      │
│ ✅ 学习拼写：模型必须学会如何拼写每个单词                  │
│ ✅ 语言无关：同样的架构可以处理任何语言                    │
└────────────────────────────────────────────────────────────┘
```

这就是字符级语言模型的魅力！

![](https://cdn.xiaolincoding.com/gh/xiaolincoder/ImageHost4@main/深度学习/char-rnn/字符级vs词级.png)

## 数据准备：文本变成数字

神经网络只能处理数字，所以第一步是把文本变成数字。

### 字符到索引的映射

```python
import numpy as np

# 示例文本
text = """深度学习是机器学习的一个分支，
它基于人工神经网络，特别是多层神经网络。
深度学习的核心思想是通过多层非线性变换来学习数据的层次化表示。"""

# 第一步：找出所有不重复的字符
chars = sorted(list(set(text)))
print(f"字符列表: {chars}")
print(f"词汇表大小: {len(chars)}")

# 第二步：创建映射字典
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 测试映射
print("\n字符映射示例:")
for ch in "深度学习":
    print(f"  '{ch}' → {char_to_idx[ch]}")
```

输出：

```
字符列表: ['\n', ',', '。', '一', '个', '习', '人', '...']
词汇表大小: 42

字符映射示例:
  '深' → 15
  '度' → 16
  '学' → 17
  '习' → 18
```

### 序列采样策略

训练时，我们要把文本切成一个个训练样本：

```
文本: "深度学习是机器学习的一个分支"

序列长度 = 10 时：
┌─────────────────────────────────────────┐
│ 样本 1:                                  │
│   输入:  "深度学习是机器学习的一"        │
│   目标:  "度学习是机器学习的一个"        │
│   (目标 = 输入向后偏移一位)              │
├─────────────────────────────────────────┤
│ 样本 2:                                  │
│   输入:  "的一个分支"                    │
│   目标:  "一个分支" (可能填充)           │
└─────────────────────────────────────────┘
```

**为什么要偏移一位？**

因为语言模型的任务是"预测下一个字符"：

- 输入"深"，模型应该预测"度"
- 输入"深度"，模型应该预测"学"
- 以此类推...

```python
def prepare_sequences(text, seq_length):
    """
    准备训练序列
    
    Args:
        text: 原始文本
        seq_length: 序列长度
    
    Returns:
        输入序列列表和目标序列列表
    """
    # 转换为索引
    data = [char_to_idx[ch] for ch in text]
    
    inputs = []
    targets = []
    
    # 滑动窗口采样
    for i in range(0, len(data) - seq_length):
        inputs.append(data[i:i+seq_length])
        targets.append(data[i+1:i+seq_length+1])  # 偏移一位
    
    return inputs, targets

# 测试
seq_length = 10
inputs, targets = prepare_sequences(text, seq_length)

print(f"总样本数: {len(inputs)}")
print(f"\n样本示例:")
print(f"输入: {''.join([idx_to_char[i] for i in inputs[0]])}")
print(f"目标: {''.join([idx_to_char[i] for i in targets[0]])}")
```

## 模型架构：从字符到概率

字符级语言模型的架构其实很简单：

```
输入字符序列 (one-hot)
      ↓
   嵌入层 (可选)
      ↓
   RNN/LSTM 层
      ↓
   全连接层
      ↓
   Softmax → 下一个字符的概率分布
```

### 输入层：One-Hot 编码

每个字符用一个独热向量表示：

```python
vocab_size = len(chars)

def one_hot_encode(idx, vocab_size):
    """
    将索引转换为 one-hot 向量
    
    Args:
        idx: 字符索引
        vocab_size: 词汇表大小
    
    Returns:
        one-hot 向量，形状 (vocab_size, 1)
    """
    vec = np.zeros((vocab_size, 1))
    vec[idx] = 1.0
    return vec

# 示例
idx = char_to_idx['深']
one_hot = one_hot_encode(idx, vocab_size)
print(f"字符 '深' 的 one-hot 向量形状: {one_hot.shape}")
print(f"非零位置: {np.where(one_hot == 1)[0][0]}")
```

### RNN 层：记忆历史信息

RNN 的核心公式：

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

```python
class CharRNN:
    """字符级 RNN 语言模型"""
    
    def __init__(self, vocab_size, hidden_size):
        """
        初始化 RNN
        
        Args:
            vocab_size: 词汇表大小
            hidden_size: 隐藏层维度
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 初始化权重
        # Xavier 初始化：保持输入输出的方差一致
        self.Wxh = np.random.randn(hidden_size, vocab_size) * np.sqrt(2.0 / (hidden_size + vocab_size))
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.Why = np.random.randn(vocab_size, hidden_size) * np.sqrt(2.0 / (vocab_size + hidden_size))
        
        # 偏置
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        
        print(f"CharRNN 初始化完成:")
        print(f"  词汇表大小: {vocab_size}")
        print(f"  隐藏层维度: {hidden_size}")
        print(f"  参数总量: {self._count_params()}")
    
    def _count_params(self):
        """计算参数总数"""
        return (self.Wxh.size + self.Whh.size + 
                self.Why.size + self.bh.size + self.by.size)
    
    def forward(self, inputs, h_prev=None):
        """
        前向传播
        
        Args:
            inputs: 输入序列（索引列表）
            h_prev: 初始隐藏状态
        
        Returns:
            loss: 交叉熵损失
            cache: 缓存数据（用于反向传播）
            h: 最终隐藏状态
        """
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
        
        # 存储中间结果
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        
        loss = 0
        
        # 逐时间步前向传播
        for t in range(len(inputs)):
            # One-hot 编码
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            
            # 隐藏状态更新
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh)
            
            # 输出层
            ys[t] = self.Why @ hs[t] + self.by
            
            # Softmax
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        
        cache = {'xs': xs, 'hs': hs, 'ys': ys, 'ps': ps}
        
        return cache, hs[len(inputs)-1]
    
    def compute_loss(self, cache, targets):
        """计算损失"""
        loss = 0
        ps = cache['ps']
        for t in range(len(targets)):
            loss += -np.log(ps[t][targets[t], 0] + 1e-10)
        return loss
```

### 输出层：Softmax 预测

输出层把隐藏状态转成概率分布：

$$P(\text{下一个字符}) = \text{Softmax}(W_{hy} h_t + b_y)$$

```python
def softmax(x):
    """
    数值稳定的 Softmax
    
    Args:
        x: 输入向量
    
    Returns:
        概率分布
    """
    e_x = np.exp(x - np.max(x))  # 减去最大值防止溢出
    return e_x / np.sum(e_x)

# 测试
logits = np.array([[1.0], [2.0], [3.0]])
probs = softmax(logits)
print(f"Softmax 输出: {probs.T}")
print(f"概率和: {np.sum(probs):.4f}")
```

## 训练过程：让模型学会预测

### 损失函数：交叉熵

语言模型使用交叉熵损失：

$$L = -\sum_t \log P(\text{目标字符}_t)$$

```python
def cross_entropy_loss(probs, target_idx):
    """
    计算单个时间步的交叉熵损失
    
    Args:
        probs: Softmax 概率分布
        target_idx: 目标字符索引
    
    Returns:
        损失值
    """
    # 加小常数防止 log(0)
    return -np.log(probs[target_idx, 0] + 1e-10)
```

### 反向传播（BPTT）

RNN 的反向传播叫 **BPTT（Backpropagation Through Time）**，本质就是把时间展开后做普通反向传播：

```python
def backward(self, cache, targets):
    """
    通过时间的反向传播 (BPTT)
    
    Args:
        cache: 前向传播的缓存
        targets: 目标序列
    
    Returns:
        grads: 梯度字典
    """
    xs, hs, ys, ps = cache['xs'], cache['hs'], cache['ys'], cache['ps']
    
    # 初始化梯度
    dWxh = np.zeros_like(self.Wxh)
    dWhh = np.zeros_like(self.Whh)
    dWhy = np.zeros_like(self.Why)
    dbh = np.zeros_like(self.bh)
    dby = np.zeros_like(self.by)
    
    # 反向传播
    dhnext = np.zeros_like(hs[0])
    
    for t in reversed(range(len(targets))):
        # 输出层梯度
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # Softmax + Cross-entropy 的梯度
        
        dWhy += dy @ hs[t].T
        dby += dy
        
        # 隐藏层梯度
        dh = self.Why.T @ dy + dhnext
        
        # tanh 的梯度
        dhraw = (1 - hs[t] * hs[t]) * dh
        
        dbh += dhraw
        dWxh += dhraw @ xs[t].T
        dWhh += dhraw @ hs[t-1].T
        
        # 传递给上一时间步
        dhnext = self.Whh.T @ dhraw
    
    # 梯度裁剪（防止梯度爆炸）
    for grad in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(grad, -5, 5, out=grad)
    
    grads = {
        'Wxh': dWxh, 'Whh': dWhh, 'Why': dWhy,
        'bh': dbh, 'by': dby
    }
    
    return grads
```

### 完整训练循环

```python
class CharRNNTrainer:
    """Char-RNN 训练器"""
    
    def __init__(self, model, learning_rate=0.01):
        """
        初始化训练器
        
        Args:
            model: CharRNN 模型
            learning_rate: 学习率
        """
        self.model = model
        self.learning_rate = learning_rate
        
        # Adagrad 优化器的累积梯度平方
        self.m = {
            'Wxh': np.zeros_like(model.Wxh),
            'Whh': np.zeros_like(model.Whh),
            'Why': np.zeros_like(model.Why),
            'bh': np.zeros_like(model.bh),
            'by': np.zeros_like(model.by)
        }
    
    def update(self, grads):
        """
        Adagrad 参数更新
        
        Args:
            grads: 梯度字典
        """
        for param_name in ['Wxh', 'Whh', 'Why', 'bh', 'by']:
            param = getattr(self.model, param_name)
            grad = grads[param_name]
            
            # 累积梯度平方
            self.m[param_name] += grad * grad
            
            # 更新参数
            param -= self.learning_rate * grad / (np.sqrt(self.m[param_name]) + 1e-8)
    
    def train_step(self, inputs, targets, h_prev=None):
        """
        单步训练
        
        Args:
            inputs: 输入序列
            targets: 目标序列
            h_prev: 初始隐藏状态
        
        Returns:
            loss: 损失值
            h: 最终隐藏状态
        """
        # 前向传播
        cache, h = self.model.forward(inputs, h_prev)
        
        # 计算损失
        loss = self.model.compute_loss(cache, targets)
        
        # 反向传播
        grads = self.model.backward(cache, targets)
        
        # 更新参数
        self.update(grads)
        
        return loss, h
```

## 文本生成：让模型开始写作

训练好模型后，怎么让它生成文本？

### 生成策略

#### 策略一：贪婪搜索

每次选择概率最高的字符：

```python
def greedy_sample(probs):
    """
    贪婪采样：选择概率最高的字符
    
    Args:
        probs: 概率分布
    
    Returns:
        选中的字符索引
    """
    return np.argmax(probs)
```

**问题**：输出会变得重复、无聊。

```
生成示例：
"深度学习是机器学习的一个分支。深度学习是机器学习的一个分支。深度学习是..."
```

#### 策略二：随机采样

按概率随机采样：

```python
def random_sample(probs):
    """
    随机采样：按概率分布采样
    
    Args:
        probs: 概率分布
    
    Returns:
        选中的字符索引
    """
    return np.random.choice(len(probs), p=probs.ravel())
```

**问题**：可能生成不连贯的内容。

#### 策略三：Temperature 采样

通过 Temperature 参数控制随机性：

```python
def temperature_sample(probs, temperature=1.0):
    """
    Temperature 采样
    
    Args:
        probs: 原始概率分布
        temperature: 温度参数
            - temperature → 0: 更确定（趋向贪婪）
            - temperature = 1: 标准随机采样
            - temperature → ∞: 更随机（均匀分布）
    
    Returns:
        选中的字符索引
    """
    # 调整概率分布
    probs = np.power(probs, 1.0 / temperature)
    probs = probs / np.sum(probs)
    
    return np.random.choice(len(probs), p=probs.ravel())
```

**Temperature 效果对比：**

| Temperature | 效果 | 适用场景 |
|-------------|------|---------|
| 0.1 - 0.3 | 保守、连贯、可能重复 | 代码生成、正式文本 |
| 0.5 - 0.7 | 平衡 | 通用场景 |
| 0.8 - 1.0 | 创意、多样、可能离谱 | 创意写作、头脑风暴 |
| 1.0+ | 随机、混乱 | 实验用途 |

### 完整生成代码

```python
def generate_text(model, seed_char, length, temperature=0.8, char_to_idx=None, idx_to_char=None):
    """
    生成文本
    
    Args:
        model: 训练好的模型
        seed_char: 种子字符（生成起点）
        length: 生成长度
        temperature: 采样温度
        char_to_idx: 字符到索引映射
        idx_to_char: 索引到字符映射
    
    Returns:
        生成的文本
    """
    # 初始化
    h = np.zeros((model.hidden_size, 1))
    x = np.zeros((model.vocab_size, 1))
    x[char_to_idx[seed_char]] = 1
    
    generated = [seed_char]
    
    for _ in range(length):
        # 前向传播
        h = np.tanh(model.Wxh @ x + model.Whh @ h + model.bh)
        y = model.Why @ h + model.by
        
        # Softmax + Temperature
        probs = np.exp(y / temperature) / np.sum(np.exp(y / temperature))
        
        # 采样
        idx = np.random.choice(model.vocab_size, p=probs.ravel())
        char = idx_to_char[idx]
        
        generated.append(char)
        
        # 准备下一步输入
        x = np.zeros((model.vocab_size, 1))
        x[idx] = 1
    
    return ''.join(generated)
```

## 完整 Python 实现

把上面所有代码整合起来：

```python
import numpy as np

# ============ 数据准备 ============
def load_text(filepath):
    """加载文本数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def build_vocab(text):
    """构建词汇表"""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char

# ============ 模型定义 ============
class CharRNN:
    """字符级 RNN 语言模型"""
    
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Xavier 初始化
        self.Wxh = np.random.randn(hidden_size, vocab_size) * np.sqrt(2.0 / (hidden_size + vocab_size))
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.Why = np.random.randn(vocab_size, hidden_size) * np.sqrt(2.0 / (vocab_size + hidden_size))
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
    
    def forward(self, inputs, h_prev=None):
        """前向传播"""
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
        
        xs, hs, ps = {}, {}, {}
        hs[-1] = np.copy(h_prev)
        
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh)
            y = self.Why @ hs[t] + self.by
            ps[t] = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
        
        return {'xs': xs, 'hs': hs, 'ps': ps}, hs[len(inputs)-1]
    
    def loss(self, cache, targets):
        """计算损失"""
        ps = cache['ps']
        loss = 0
        for t in range(len(targets)):
            loss += -np.log(ps[t][targets[t], 0] + 1e-10)
        return loss
    
    def backward(self, cache, targets):
        """反向传播"""
        xs, hs, ps = cache['xs'], cache['hs'], cache['ps']
        
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        
        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            
            dWhy += dy @ hs[t].T
            dby += dy
            
            dh = self.Why.T @ dy + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            
            dbh += dhraw
            dWxh += dhraw @ xs[t].T
            dWhh += dhraw @ hs[t-1].T
            dhnext = self.Whh.T @ dhraw
        
        # 梯度裁剪
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        return {'Wxh': dWxh, 'Whh': dWhh, 'Why': dWhy, 'bh': dbh, 'by': dby}

# ============ 训练器 ============
class Trainer:
    """训练器"""
    
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in model.__dict__.items() if v.ndim >= 1}
    
    def step(self, inputs, targets, h):
        """单步训练"""
        cache, h_new = self.model.forward(inputs, h)
        loss = self.model.loss(cache, targets)
        grads = self.model.backward(cache, targets)
        
        # Adagrad 更新
        for name, grad in grads.items():
            param = getattr(self.model, name)
            self.m[name] += grad * grad
            param -= self.lr * grad / (np.sqrt(self.m[name]) + 1e-8)
        
        return loss, h_new

# ============ 文本生成 ============
def generate(model, seed, length, temperature, char_to_idx, idx_to_char):
    """生成文本"""
    h = np.zeros((model.hidden_size, 1))
    x = np.zeros((model.vocab_size, 1))
    x[char_to_idx[seed]] = 1
    
    result = [seed]
    
    for _ in range(length):
        h = np.tanh(model.Wxh @ x + model.Whh @ h + model.bh)
        y = model.Why @ h + model.by
        p = np.exp(y / temperature) / np.sum(np.exp(y / temperature))
        
        idx = np.random.choice(model.vocab_size, p=p.ravel())
        result.append(idx_to_char[idx])
        
        x = np.zeros((model.vocab_size, 1))
        x[idx] = 1
    
    return ''.join(result)

# ============ 主程序 ============
if __name__ == "__main__":
    # 示例文本（实际使用时替换为更大的语料）
    text = """
    深度学习是机器学习的一个分支，它基于人工神经网络，特别是多层神经网络。
    深度学习的核心思想是通过多层非线性变换来学习数据的层次化表示。
    深度学习在计算机视觉、自然语言处理、语音识别等领域取得了巨大成功。
    """
    
    # 构建词汇表
    chars, char_to_idx, idx_to_char = build_vocab(text)
    vocab_size = len(chars)
    print(f"词汇表大小: {vocab_size}")
    
    # 创建模型
    hidden_size = 128
    model = CharRNN(vocab_size, hidden_size)
    trainer = Trainer(model, lr=0.05)
    
    # 准备数据
    seq_length = 25
    data = [char_to_idx[ch] for ch in text]
    
    # 训练
    print("\n开始训练...")
    h = np.zeros((hidden_size, 1))
    smooth_loss = -np.log(1.0/vocab_size) * seq_length
    
    for epoch in range(1000):
        # 随机采样
        p = np.random.randint(0, len(data) - seq_length - 1)
        inputs = data[p:p+seq_length]
        targets = data[p+1:p+seq_length+1]
        
        # 训练一步
        loss, h = trainer.step(inputs, targets, h)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        
        # 打印进度
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {smooth_loss:.2f}")
            sample = generate(model, '深', 100, 0.5, char_to_idx, idx_to_char)
            print(f"生成示例: {sample[:50]}...")
            print()
    
    # 最终生成
    print("="*50)
    print("训练完成！最终生成示例：")
    print("="*50)
    generated = generate(model, '深', 200, 0.7, char_to_idx, idx_to_char)
    print(generated)
```

## 实验与调优

### 不同数据集的效果

| 数据集 | 训练数据 | 生成效果 |
|--------|---------|---------|
| 莎士比亚 | 剧本文本 | 古风英语，戏剧感强 |
| Linux 源码 | C 代码 | 语法基本正确，像代码 |
| 数学论文 | LaTeX | 公式格式对，内容随机 |
| 中文小说 | 网络小说 | 语句通顺，有点意思 |

### 超参数影响

| 超参数 | 增大效果 | 建议值 |
|--------|---------|--------|
| 隐藏层维度 | 模型更强，训练更慢 | 128-512 |
| 序列长度 | 捕获更长依赖 | 50-200 |
| 学习率 | 收敛更快，可能不稳定 | 0.01-0.1 |
| Temperature | 更随机/创意 | 0.5-0.8 |

### 常见问题

**Q：生成的文本越来越重复怎么办？**

A：调高 Temperature，或者使用 Top-K 采样限制候选字符。

**Q：训练 loss 不下降？**

A：检查学习率是否太小，或者数据是否有问题。

**Q：生成的内容毫无意义？**

A：训练时间不够，或者数据量太小。字符级模型需要大量数据才能学到有意义的内容。

![](https://cdn.xiaolincoding.com/gh/xiaolincoder/ImageHost4@main/深度学习/char-rnn/调参指南.png)

## 总结

今天我们从零实现了一个字符级语言模型：

| 步骤 | 核心内容 |
|------|---------|
| **数据准备** | 字符→索引映射，序列采样 |
| **模型架构** | One-hot → RNN → Softmax |
| **训练过程** | BPTT 反向传播，Adagrad 优化 |
| **文本生成** | Temperature 采样控制随机性 |

**字符级语言模型的核心洞察：**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   简单的规则 + 大量数据 = 惊人的能力                        │
│                                                             │
│   预测下一个字符 → 学会语言的结构                          │
│   无需语言学知识 → 纯数据驱动学习                          │
│   同一套架构 → 处理任何语言的文本                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

这就是 Karpathy 所说的 "The Unreasonable Effectiveness of Recurrent Neural Networks"——简单的模型，在足够多的数据上训练，就能学到语言的深层结构。

当然，字符级 RNN 有它的局限：训练慢、长程依赖问题。但这些思想（预测下一个 token、自回归生成）延续到了今天的 GPT 系列。

下一篇，我们将聊聊 **训练技巧与调参心得**，分享实战中的经验教训，敬请期待！

---

> 本文是《图解 Char-RNN》系列的第 6 篇。
>
> 如果觉得有帮助，欢迎关注「云言 AI」公众号，一起学习深度学习！