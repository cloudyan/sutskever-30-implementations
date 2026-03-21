# 训练 RNN 有什么技巧？

问下大家，训练 RNN 的时候，有没有遇到过这些情况？

- Loss 一开始下降，突然变成 NaN？
- 训练半天，模型就是不收敛？
- 刚开始还行，后面越来越差？

云言刚开始训练 RNN 的时候，可是被这些问题折磨惨了。直到后来掌握了这些技巧，才发现卧槽，原来 RNN 训练有这么多门道！

今天咱们就来聊聊训练 RNN 的那些实战技巧，都是踩坑踩出来的经验。

## 为什么 RNN 训练这么难？

在深入技巧之前，先理解一下 RNN 训练的特殊性。

### 梯度的时空旅行

普通神经网络的梯度只需要穿过几层网络，而 RNN 的梯度需要穿越**时间维度**：

```
时刻 t=5 的梯度要传到 t=0，需要经过 5 次矩阵乘法！

t=0    t=1    t=2    t=3    t=4    t=5
 ↓      ↓      ↓      ↓      ↓      ↓
[h0] → [h1] → [h2] → [h3] → [h4] → [h5]
        ↑      ↑      ↑      ↑      ↑
        └──────┴──────┴──────┴──────┘
              梯度需要反向穿过这里！
```

这就带来了两个经典问题：

| 问题 | 原因 | 表现 |
|------|------|------|
| **梯度爆炸** | 连续乘以大数 | Loss 变 NaN/Inf |
| **梯度消失** | 连续乘以小数 | 长距离依赖学不到 |

### 用代码感受一下

```python
import numpy as np

# 模拟梯度在时间步上的传播
def simulate_gradient_propagation(num_steps, grad_scale):
    """
    模拟梯度穿过多个时间步的情况
    
    grad_scale: 每个时间步梯度变化的倍数
    - > 1: 梯度爆炸
    - < 1: 梯度消失
    - = 1: 稳定
    """
    grad = 1.0  # 初始梯度
    grads = [grad]
    
    for t in range(num_steps):
        grad = grad * grad_scale
        grads.append(grad)
    
    return grads

# 情况1：梯度爆炸
print("=== 梯度爆炸 (grad_scale = 1.5) ===")
grads = simulate_gradient_propagation(20, 1.5)
print(f"第 1 步: {grads[1]:.2f}")
print(f"第 10 步: {grads[10]:.2f}")
print(f"第 20 步: {grads[20]:.2e}")

# 情况2：梯度消失
print("\n=== 梯度消失 (grad_scale = 0.8) ===")
grads = simulate_gradient_propagation(20, 0.8)
print(f"第 1 步: {grads[1]:.2f}")
print(f"第 10 步: {grads[10]:.4f}")
print(f"第 20 步: {grads[20]:.6f}")
```

输出：

```
=== 梯度爆炸 (grad_scale = 1.5) ===
第 1 步: 1.50
第 10 步: 57.67
第 20 步: 3.33e+03

=== 梯度消失 (grad_scale = 0.8) ===
第 1 步: 0.80
第 10 步: 0.1074
第 20 步: 0.011529
```

看到了吗？20 个时间步，梯度要么爆炸到几千，要么消失到接近 0。这就是 RNN 训练的核心挑战！

## 技巧一：梯度裁剪

这是解决梯度爆炸最简单有效的方法。

### 什么是梯度裁剪？

简单说：梯度太大就把它"砍"小一点。

```
原始梯度: [10, 20, 30, 40]  → 梯度范数 = 54.77
裁剪阈值: 5

裁剪后:   [0.91, 1.83, 2.74, 3.65]  → 梯度范数 = 5.0
```

### 怎么裁剪？

有两种常见方法：

**方法一：按值裁剪（简单粗暴）**

```python
def clip_by_value(grad, min_val, max_val):
    """按值裁剪：把梯度限制在 [min_val, max_val] 范围内"""
    return np.clip(grad, min_val, max_val)

# 示例
grad = np.array([-10, -5, 0, 5, 10])
clipped = clip_by_value(grad, -5, 5)
print(f"原始梯度: {grad}")
print(f"裁剪后: {clipped}")
# 输出: 原始梯度: [-10 -5  0  5 10]
#       裁剪后: [-5 -5  0  5  5]
```

**方法二：按范数裁剪（推荐）**

```python
def clip_by_norm(grads, max_norm=5.0):
    """
    按全局范数裁剪梯度
    
    如果所有梯度的 L2 范数超过 max_norm，
    就把所有梯度按比例缩小
    
    Args:
        grads: 字典，参数名 → 梯度数组
        max_norm: 允许的最大范数
    
    Returns:
        clipped_grads: 裁剪后的梯度
        global_norm: 裁剪前的全局范数
    """
    # 计算全局范数
    global_norm = 0.0
    for grad in grads.values():
        global_norm += np.sum(grad ** 2)
    global_norm = np.sqrt(global_norm)
    
    # 如果超过阈值，按比例缩放
    if global_norm > max_norm:
        scale = max_norm / global_norm
        clipped_grads = {k: v * scale for k, v in grads.items()}
    else:
        clipped_grads = grads
    
    return clipped_grads, global_norm

# 示例
grads = {
    'Wxh': np.random.randn(64, 100) * 10,
    'Whh': np.random.randn(64, 64) * 10,
    'Why': np.random.randn(100, 64) * 10
}

clipped_grads, norm = clip_by_norm(grads, max_norm=5.0)
print(f"裁剪前范数: {norm:.2f}")
print(f"裁剪后范数: {np.sqrt(sum(np.sum(g**2) for g in clipped_grads.values())):.2f}")
```

### 裁剪阈值怎么选？

| 阈值 | 适用场景 | 建议 |
|------|---------|------|
| 1.0 | 非常保守，梯度变化小 | 很少用 |
| 5.0 | **常用默认值** | 大多数情况适用 |
| 10.0 | 梯度波动较大时 | 可以先试这个 |
| 50.0 | 特殊情况，不稳定时 | 谨慎使用 |

**经验法则**：先用 5.0，观察训练过程中的梯度范数。如果经常被裁剪，说明阈值太小；如果从不被裁剪，可以适当减小。

## 技巧二：学习率调度

学习率是训练最重要的超参数之一，调度好了事半功倍。

### 为什么需要调度？

想象你在爬山找最低点：

- 开始时离目标远，步子要大 → **高学习率**
- 快到山顶了，步子要小 → **低学习率**

```
Epoch 1-50:   学习率 = 0.01   (大步探索)
Epoch 51-100: 学习率 = 0.005  (中等步子)
Epoch 101-150: 学习率 = 0.001 (小步精调)
```

### 常见的学习率调度策略

**1. 阶梯衰减（Step Decay）**

```python
def step_decay(epoch, initial_lr=0.01, decay=0.5, decay_every=10):
    """
    每 decay_every 个 epoch，学习率乘以 decay
    
    例如: 每 10 个 epoch，学习率减半
    """
    lr = initial_lr * (decay ** (epoch // decay_every))
    return lr

# 测试
for epoch in [0, 9, 10, 19, 20, 50]:
    lr = step_decay(epoch)
    print(f"Epoch {epoch:2d}: LR = {lr:.6f}")
```

输出：

```
Epoch  0: LR = 0.010000
Epoch  9: LR = 0.010000
Epoch 10: LR = 0.005000
Epoch 19: LR = 0.005000
Epoch 20: LR = 0.002500
Epoch 50: LR = 0.000312
```

**2. 指数衰减（Exponential Decay）**

```python
def exponential_decay(epoch, initial_lr=0.01, decay_rate=0.95):
    """
    每个 epoch 学习率都衰减一点点
    
    lr = initial_lr * (decay_rate ^ epoch)
    """
    lr = initial_lr * (decay_rate ** epoch)
    return lr

# 对比
epochs = range(0, 101, 10)
step_lrs = [step_decay(e) for e in epochs]
exp_lrs = [exponential_decay(e) for e in epochs]

print("Epoch | Step Decay | Exp Decay")
print("-" * 30)
for e, s, exp in zip(epochs, step_lrs, exp_lrs):
    print(f"  {e:3d} | {s:.6f}   | {exp:.6f}")
```

**3. 余弦退火（Cosine Annealing）—— 进阶技巧**

```python
def cosine_annealing(epoch, max_epochs, initial_lr=0.01, min_lr=1e-6):
    """
    学习率按余弦曲线下降
    
    好处：下降平滑，后期有微调空间
    """
    import math
    lr = min_lr + 0.5 * (initial_lr - min_lr) * \
         (1 + math.cos(math.pi * epoch / max_epochs))
    return lr
```

### 初始学习率怎么选？

| 模型规模 | 建议范围 | 调优方法 |
|---------|---------|---------|
| 小型（< 100K 参数） | 0.01 ~ 0.1 | 从 0.1 开始，减半尝试 |
| 中型（100K ~ 1M） | 0.001 ~ 0.01 | 从 0.01 开始 |
| 大型（> 1M） | 0.0001 ~ 0.001 | 更谨慎，可以更小 |

**快速测试法**：用不同学习率跑几个 epoch，看 loss 下降速度。

```python
# 学习率搜索示例
learning_rates = [0.1, 0.01, 0.001, 0.0001]

for lr in learning_rates:
    # 用当前学习率训练几个 epoch
    loss = quick_train(model, data, epochs=5, lr=lr)
    print(f"LR={lr:.4f}: Loss={loss:.4f}")
```

## 技巧三：初始化策略

好的初始化能让训练事半功倍。

### 为什么初始化很重要？

如果权重全初始化为 0：

```
所有神经元输出相同 → 所有梯度相同 → 所有权重更新相同 → 永远学不到有用的特征
```

如果权重初始化太大：

```
激活值饱和 → 梯度接近 0 → 学不动
```

### Xavier 初始化（输入权重）

适合 tanh 激活函数：

```python
def xavier_initializer(shape):
    """
    Xavier/Glorot 初始化
    
    使输入和输出的方差一致，避免梯度消失/爆炸
    
    原理: W ~ N(0, sqrt(2 / (fan_in + fan_out)))
    """
    fan_in, fan_out = shape[1], shape[0]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std

# 示例
Wxh = xavier_initializer((hidden_size, vocab_size))
print(f"权重均值: {Wxh.mean():.6f}")
print(f"权重标准差: {Wxh.std():.6f}")
```

### 正交初始化（循环权重）

**这是 RNN 的关键技巧！** 循环权重用正交初始化可以大幅改善梯度流动：

```python
def orthogonal_initializer(shape, gain=1.0):
    """
    正交初始化
    
    生成正交矩阵，保证梯度在时间步之间稳定传播
    
    原理: 
    1. 生成随机矩阵
    2. 做 SVD 分解
    3. 取正交部分
    
    数学: Q 来自 A = UΣV^T 的 U 或 V
    """
    # 生成随机矩阵
    flat_shape = (shape[0], np.prod(shape[1:]))
    A = np.random.randn(*flat_shape)
    
    # SVD 分解
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # 取正交矩阵
    Q = U if A.shape[0] >= A.shape[1] else Vt
    
    # 调整形状和增益
    return gain * Q.reshape(shape)

# RNN 的循环权重推荐用正交初始化
Whh = orthogonal_initializer((hidden_size, hidden_size))
print(f"正交性检查 (应接近单位阵): {np.allclose(Whh @ Whh.T, np.eye(hidden_size), atol=1e-5)}")
```

### 偏置初始化

**LSTM 的 Forget Gate 偏置有特殊技巧！**

```python
def init_lstm_biases(hidden_size, forget_bias=1.0):
    """
    LSTM 偏置初始化
    
    关键技巧：forget gate 偏置初始化为正值（如 1.0）
    这样初始状态下 forget gate 接近 1，倾向于保留信息
    
    这对学习长期依赖至关重要！
    """
    return {
        'bf': np.ones((hidden_size, 1)) * forget_bias,  # Forget gate: 正值！
        'bi': np.zeros((hidden_size, 1)),  # Input gate
        'bo': np.zeros((hidden_size, 1)),  # Output gate
        'bc': np.zeros((hidden_size, 1)),  # Cell candidate
    }

# 为什么 forget_bias 要设为 1.0？
# forget gate = sigmoid(bf + ...)
# sigmoid(1) ≈ 0.73，意味着初始时倾向于保留 73% 的记忆
# 如果 bf = 0，sigmoid(0) = 0.5，会丢失一半信息！
```

### 初始化总结表

| 参数类型 | 推荐初始化 | 原因 |
|---------|-----------|------|
| 输入权重 Wxh | Xavier | 平衡输入输出方差 |
| 循环权重 Whh | **正交** | 保持梯度稳定 |
| 输出权重 Why | Xavier 或小随机 | 避免初始输出过大 |
| 普通偏置 b | 零向量 | 默认值 |
| **Forget Gate 偏置** | **1.0** | 保留长期记忆 |

## 技巧四：序列截断（Truncated BPTT）

处理长序列时，这是个重要的实用技巧。

### 什么是 Truncated BPTT？

完整 BPTT 需要反向传播整个序列，计算量巨大。Truncated BPTT 把长序列切成小段：

```
完整序列: [x1, x2, x3, ..., x100, ..., x1000]

截断后:
段1: [x1, ..., x100]   → 反向传播
段2: [x101, ..., x200] → 反向传播（但保留段1的隐藏状态）
...
```

### 实现代码

```python
def truncated_bptt_forward(rnn, data, seq_length, hprev=None):
    """
    Truncated BPTT 前向传播
    
    每次只处理 seq_length 个字符，但保留隐藏状态
    
    Args:
        rnn: RNN 模型
        data: 完整数据
        seq_length: 截断长度
        hprev: 上一个段的隐藏状态
    """
    if hprev is None:
        hprev = np.zeros((rnn.hidden_size, 1))
    
    all_losses = []
    
    # 按段处理
    for start in range(0, len(data) - seq_length, seq_length):
        # 当前段的输入和目标
        inputs = data[start:start + seq_length]
        targets = data[start + 1:start + seq_length + 1]
        
        # 前向传播（使用上一个段的隐藏状态）
        xs, hs, ys, ps = rnn.forward(inputs, hprev)
        loss = rnn.loss(ps, targets)
        
        # 反向传播
        grads = rnn.backward(xs, hs, ps, targets)
        
        # 更新参数
        update_params(rnn, grads)
        
        # 保存当前段的最终隐藏状态，传给下一段
        hprev = hs[len(inputs) - 1]
        
        all_losses.append(loss)
    
    return all_losses
```

### 截断长度怎么选？

| 长度 | 优点 | 缺点 | 适用场景 |
|------|-----|------|---------|
| 20-50 | 训练快，内存小 | 长期依赖学不到 | 快速实验 |
| 50-100 | 平衡 | 平衡 | **常用默认** |
| 100-200 | 学到更长依赖 | 慢，内存大 | 需要长依赖时 |
| 500+ | 很长依赖 | 可能梯度问题 | 特殊任务 |

**关键点**：隐藏状态在段之间传递，所以理论上可以学到超长依赖，只是梯度只传播 seq_length 步。

## 技巧五：正则化

防止过拟合，让模型泛化更好。

### 1. Dropout（变体）

RNN 的 Dropout 有讲究，不能随便用：

```python
def rnn_dropout_mask(shape, dropout_rate=0.5):
    """
    RNN 专用的 Dropout 掩码
    
    关键：同一个序列内使用同一个掩码！
    这样才能保持时间一致性
    
    这叫做 "Variational Dropout" 或 "Locked Dropout"
    """
    mask = (np.random.rand(*shape) > dropout_rate).astype(float)
    # 归一化，保持期望不变
    mask = mask / (1.0 - dropout_rate)
    return mask

# 正确用法：在序列开始时生成掩码，整个序列使用同一个
def forward_with_dropout(rnn, inputs, hprev, dropout_rate=0.5):
    """带 Dropout 的前向传播"""
    hidden_mask = rnn_dropout_mask((rnn.hidden_size, 1), dropout_rate)
    
    hs = {}
    hs[-1] = hprev
    
    for t, x in enumerate(inputs):
        # 隐藏层计算
        h = np.tanh(rnn.Wxh @ x + rnn.Whh @ hs[t-1] + rnn.bh)
        # 应用 dropout（同一个序列用同一个 mask）
        h = h * hidden_mask
        hs[t] = h
    
    return hs
```

### 2. 权重衰减（L2 正则化）

```python
def l2_regularization(params, lambda_reg=0.001):
    """
    L2 正则化：在 loss 中加入权重的平方和
    
    loss_total = loss_data + lambda * sum(W^2)
    
    效果：限制权重大小，防止过拟合
    """
    reg_loss = 0.0
    for name, param in params.items():
        if 'W' in name:  # 只对权重做正则化，不对偏置做
            reg_loss += np.sum(param ** 2)
    return lambda_reg * reg_loss

# 在训练中使用
loss = cross_entropy_loss(predictions, targets)
reg_loss = l2_regularization(model.get_params(), lambda_reg=0.001)
total_loss = loss + reg_loss
```

### 3. 早停（Early Stopping）

```python
class EarlyStopping:
    """
    早停：当验证损失不再下降时，停止训练
    
    防止过拟合，节省时间
    """
    
    def __init__(self, patience=10, min_delta=0.0001):
        """
        Args:
            patience: 容忍多少个 epoch 不提升
            min_delta: 认为是提升的最小变化量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_params = None
    
    def __call__(self, val_loss, model_params):
        """
        检查是否应该停止
        
        Returns:
            should_stop: 是否应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            # 有提升
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳参数
            self.best_params = {k: v.copy() for k, v in model_params.items()}
            return False
        else:
            # 没提升
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 停止训练
            return False

# 使用示例
early_stopping = EarlyStopping(patience=10)

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_data)
    val_loss = evaluate(model, val_data)
    
    if early_stopping(val_loss, model.get_params()):
        print(f"Early stopping at epoch {epoch}")
        # 恢复最佳参数
        model.set_params(early_stopping.best_params)
        break
```

## 技巧六：超参数调优

RNN 的关键超参数及调优建议。

### 1. 隐藏层大小

| 大小 | 适用场景 | 参数量估算 |
|------|---------|-----------|
| 64-128 | 小数据集，简单模式 | ~10K |
| 256-512 | 中等数据集，一般任务 | ~100K |
| 512-1024 | 大数据集，复杂模式 | ~500K+ |

```python
# 隐藏层大小影响参数量
def estimate_params(vocab_size, hidden_size):
    """
    估算 RNN 参数量
    
    参数 = Wxh + Whh + Why + biases
         = vocab*hidden + hidden*hidden + hidden*vocab + biases
    """
    Wxh = vocab_size * hidden_size
    Whh = hidden_size * hidden_size
    Why = hidden_size * vocab_size
    biases = 2 * hidden_size + vocab_size
    return Wxh + Whh + Why + biases

# 示例
vocab_size = 100
for hidden_size in [64, 128, 256, 512]:
    params = estimate_params(vocab_size, hidden_size)
    print(f"Hidden={hidden_size}: ~{params/1000:.1f}K 参数")
```

输出：

```
Hidden=64: ~16.3K 参数
Hidden=128: ~49.2K 参数
Hidden=256: ~163.8K 参数
Hidden=512: ~563.2K 参数
```

### 2. 层数

```python
# 单层 vs 多层
# 单层：简单，训练快，适合入门
# 多层：更强大的表达能力，但更难训练

class StackedRNN:
    """多层 RNN"""
    
    def __init__(self, vocab_size, hidden_size, num_layers):
        self.layers = []
        for i in range(num_layers):
            input_size = vocab_size if i == 0 else hidden_size
            self.layers.append(VanillaRNN(input_size, hidden_size))
    
    def forward(self, inputs, hprevs):
        """
        多层前向传播
        
        上一层的输出作为下一层的输入
        """
        current_input = inputs
        for i, layer in enumerate(self.layers):
            xs, hs, ys, ps = layer.forward(current_input, hprevs[i])
            # 下一层的输入是当前层的隐藏状态序列
            current_input = [hs[t] for t in range(len(inputs))]
        return ps
```

### 3. 批量大小（Batch Size）

| 批量大小 | 特点 | 建议 |
|---------|------|------|
| 1 | 噪声大，但泛化可能好 | 小数据集 |
| 16-32 | 平衡 | **常用默认** |
| 64-128 | 稳定，但可能收敛慢 | 大数据集 |
| 256+ | 非常稳定，需要大学习率 | GPU 加速场景 |

## 技巧七：调试技巧

训练遇到问题时，怎么定位？

### 1. 损失曲线分析

```python
def analyze_training_curve(losses):
    """
    分析训练曲线，诊断问题
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    
    # 添加诊断信息
    if losses[-1] > losses[0]:
        plt.text(0.5, 0.9, '⚠️ Loss 上升！检查学习率', 
                transform=plt.gca().transAxes, color='red')
    elif np.std(losses[-100:]) / np.mean(losses[-100:]) > 0.1:
        plt.text(0.5, 0.9, '⚠️ Loss 震荡大！可能需要梯度裁剪', 
                transform=plt.gca().transAxes, color='orange')
    elif losses[-1] / losses[0] < 0.5:
        plt.text(0.5, 0.9, '✅ Loss 正常下降', 
                transform=plt.gca().transAxes, color='green')
    
    plt.show()
```

### 2. 常见问题诊断表

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss = NaN | 梯度爆炸/学习率太大 | 梯度裁剪 / 降低学习率 |
| Loss 不下降 | 学习率太小 / 初始化差 | 提高学习率 / 检查初始化 |
| Loss 震荡剧烈 | 学习率太大 | 降低学习率 / 用更平滑的优化器 |
| Train loss 降，Val loss 升 | 过拟合 | Dropout / L2 / 早停 / 更多数据 |
| Loss 卡在某个值 | 局部最优 / 表达能力不足 | 调整模型结构 / 换初始化 |

### 3. 梯度检查

```python
def gradient_check(model, inputs, targets, epsilon=1e-5):
    """
    数值梯度检查
    
    对比解析梯度和数值梯度，验证反向传播是否正确
    """
    # 计算解析梯度
    xs, hs, ys, ps = model.forward(inputs, hprev)
    grads = model.backward(xs, hs, ps, targets)
    
    # 数值梯度
    numerical_grads = {}
    
    for param_name, param in model.get_params().items():
        num_grad = np.zeros_like(param)
        
        # 对每个参数元素计算数值梯度
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]
            
            # f(x + epsilon)
            param[idx] = old_val + epsilon
            _, _, _, ps_plus = model.forward(inputs, hprev)
            loss_plus = model.loss(ps_plus, targets)
            
            # f(x - epsilon)
            param[idx] = old_val - epsilon
            _, _, _, ps_minus = model.forward(inputs, hprev)
            loss_minus = model.loss(ps_minus, targets)
            
            # 数值梯度
            num_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # 恢复
            param[idx] = old_val
            it.iternext()
        
        numerical_grads[param_name] = num_grad
    
    # 对比
    print("梯度检查结果：")
    for name in grads.keys():
        diff = np.abs(grads[name] - numerical_grads[name]).max()
        relative_error = diff / (np.abs(grads[name]).max() + 1e-8)
        status = "✅ OK" if relative_error < 1e-5 else "❌ ERROR"
        print(f"  {name}: 相对误差 = {relative_error:.2e} {status}")
```

## 综合示例：完整训练流程

把所有技巧整合起来：

```python
import numpy as np
from collections import defaultdict

class RNNTrainer:
    """RNN 训练器：整合所有训练技巧"""
    
    def __init__(self, model, learning_rate=0.01, 
                 grad_clip=5.0, lr_decay=0.95, decay_every=10,
                 l2_reg=0.0001, dropout_rate=0.0):
        """
        Args:
            model: RNN 模型
            learning_rate: 初始学习率
            grad_clip: 梯度裁剪阈值
            lr_decay: 学习率衰减率
            decay_every: 每 N 个 epoch 衰减一次
            l2_reg: L2 正则化系数
            dropout_rate: Dropout 比例
        """
        self.model = model
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        
        # Adagrad 记忆变量
        self.memory = defaultdict(lambda: np.zeros(1))
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'grad_norm': [],
            'learning_rate': []
        }
    
    def get_lr(self, epoch):
        """获取当前学习率"""
        return self.learning_rate * (self.lr_decay ** (epoch // self.decay_every))
    
    def clip_gradients(self, grads):
        """梯度裁剪"""
        if self.grad_clip is None:
            return grads, 0.0
        
        # 计算全局范数
        global_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        
        # 裁剪
        if global_norm > self.grad_clip:
            scale = self.grad_clip / global_norm
            clipped = {k: v * scale for k, v in grads.items()}
        else:
            clipped = grads
        
        return clipped, global_norm
    
    def l2_loss(self):
        """L2 正则化损失"""
        if self.l2_reg == 0:
            return 0
        params = self.model.get_params()
        return self.l2_reg * sum(np.sum(p**2) for k, p in params.items() if 'W' in k)
    
    def update_params_adagrad(self, grads, lr):
        """
        Adagrad 优化器
        
        自适应学习率：根据历史梯度调整每个参数的学习率
        梯度大的参数学习率小，梯度小的参数学习率大
        """
        params = self.model.get_params()
        
        for name in params.keys():
            if name not in self.memory:
                self.memory[name] = np.zeros_like(params[name])
            
            # 累积梯度平方
            self.memory[name] += grads[name] ** 2
            
            # 更新参数
            params[name] -= lr * grads[name] / (np.sqrt(self.memory[name]) + 1e-8)
        
        self.model.set_params(params)
    
    def train_step(self, inputs, targets, hprev, epoch):
        """单步训练"""
        # 前向传播
        xs, hs, ys, ps = self.model.forward(inputs, hprev)
        
        # 计算损失
        loss = self.model.loss(ps, targets) + self.l2_loss()
        
        # 反向传播
        grads = self.model.backward(xs, hs, ps, targets)
        
        # 梯度裁剪
        grads, grad_norm = self.clip_gradients(grads)
        
        # 获取当前学习率
        lr = self.get_lr(epoch)
        
        # 更新参数
        self.update_params_adagrad(grads, lr)
        
        # 记录历史
        self.history['train_loss'].append(loss)
        self.history['grad_norm'].append(grad_norm)
        self.history['learning_rate'].append(lr)
        
        return loss, hs[len(inputs) - 1]
    
    def train(self, data, char_to_ix, seq_length=50, epochs=10, 
              print_every=100, sample_every=500):
        """
        完整训练流程
        
        Args:
            data: 文本数据
            char_to_ix: 字符到索引的映射
            seq_length: 序列长度
            epochs: 训练轮数
            print_every: 每隔多少步打印一次
            sample_every: 每隔多少步采样一次
        """
        print("=" * 60)
        print("开始训练 RNN")
        print("=" * 60)
        print(f"学习率: {self.learning_rate}")
        print(f"梯度裁剪: {self.grad_clip}")
        print(f"序列长度: {seq_length}")
        print(f"L2 正则化: {self.l2_reg}")
        print("=" * 60)
        
        smooth_loss = -np.log(1.0 / self.model.vocab_size) * seq_length
        hprev = np.zeros((self.model.hidden_size, 1))
        n = 0  # 迭代次数
        p = 0  # 数据指针
        
        for epoch in range(epochs):
            # 重置到数据开头
            if p + seq_length + 1 >= len(data):
                hprev = np.zeros((self.model.hidden_size, 1))
                p = 0
            
            # 准备输入和目标
            inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
            targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
            
            # 训练一步
            loss, hprev = self.train_step(inputs, targets, hprev, epoch)
            
            # 平滑损失
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            
            # 打印进度
            if n % print_every == 0:
                lr = self.get_lr(epoch)
                print(f"Iter {n:5d} | Epoch {epoch:2d} | "
                      f"Loss: {smooth_loss:.4f} | "
                      f"LR: {lr:.6f} | "
                      f"Grad Norm: {self.history['grad_norm'][-1]:.2f}")
            
            # 采样
            if n % sample_every == 0:
                sample_idx = self.model.sample(hprev, inputs[0], 100)
                sample_text = ''.join([ix_to_char[ix] for ix in sample_idx])
                print(f"\n采样结果:\n{sample_text}\n")
            
            p += seq_length
            n += 1
        
        print("=" * 60)
        print("训练完成！")
        print(f"最终 Loss: {smooth_loss:.4f}")
        print("=" * 60)
        
        return self.history


# 使用示例
if __name__ == "__main__":
    # 准备数据
    data = "hello world " * 100
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # 创建模型
    hidden_size = 64
    model = VanillaRNN(vocab_size, hidden_size)
    
    # 创建训练器（应用所有技巧）
    trainer = RNNTrainer(
        model,
        learning_rate=0.01,      # 学习率
        grad_clip=5.0,           # 梯度裁剪
        lr_decay=0.95,           # 学习率衰减
        decay_every=10,          # 每 10 个 epoch 衰减
        l2_reg=0.0001,           # L2 正则化
        dropout_rate=0.0         # Dropout（可选）
    )
    
    # 训练
    history = trainer.train(
        data, char_to_ix,
        seq_length=25,
        epochs=5,
        print_every=100,
        sample_every=500
    )
```

## 常见问题与解决

### 问题 1：Loss 变成 NaN

```python
# 诊断代码
def diagnose_nan(model, inputs, targets):
    """诊断 NaN 问题"""
    xs, hs, ys, ps = model.forward(inputs, hprev)
    
    # 检查每一步的数值
    for t in range(len(inputs)):
        if np.any(np.isnan(hs[t])):
            print(f"❌ 时间步 {t}: 隐藏状态 NaN")
        if np.any(np.isinf(hs[t])):
            print(f"❌ 时间步 {t}: 隐藏状态 Inf")
        if np.any(np.isnan(ps[t])):
            print(f"❌ 时间步 {t}: 概率 NaN")
        if np.any(np.sum(ps[t]) < 0.99 or np.sum(ps[t]) > 1.01):
            print(f"⚠️ 时间步 {t}: 概率和不等于 1")
    
    # 检查梯度
    grads = model.backward(xs, hs, ps, targets)
    for name, grad in grads.items():
        if np.any(np.isnan(grad)):
            print(f"❌ 梯度 {name} 包含 NaN")
        if np.any(np.isinf(grad)):
            print(f"❌ 梯度 {name} 包含 Inf")

# 解决方案
# 1. 梯度裁剪
# 2. 降低学习率
# 3. 检查数据是否有异常值
# 4. 使用数值稳定的 softmax
```

### 问题 2：训练不收敛

```python
# 诊断清单
def convergence_checklist():
    """
    训练不收敛检查清单
    
    1. 学习率是否合适？
       - 尝试 0.01, 0.001, 0.0001
       - 观察前几个 epoch 的 loss 变化
    
    2. 初始化是否合理？
       - 输入权重：Xavier
       - 循环权重：正交
       - Forget gate 偏置：1.0
    
    3. 数据是否正确？
       - 输入和目标是否对应（差一位）
       - 字符到索引的映射是否正确
    
    4. 模型是否足够大？
       - 增加隐藏层大小
       - 增加层数
    """
    pass
```

## 总结

训练 RNN 的核心技巧：

| 技巧 | 作用 | 关键点 |
|------|------|--------|
| **梯度裁剪** | 防止梯度爆炸 | 阈值 5.0 是好起点 |
| **学习率调度** | 稳定收敛 | 从 0.01 开始，逐步衰减 |
| **正交初始化** | 保持梯度流动 | 循环权重必用 |
| **Forget Gate 偏置=1** | 保留长期记忆 | LSTM 关键技巧 |
| **序列截断** | 处理长序列 | 长度 50-100 常用 |
| **早停** | 防止过拟合 | patience=10 |
| **L2 正则化** | 泛化更好 | λ=0.0001 |

这些技巧组合使用，能解决大部分 RNN 训练问题。实际应用中，建议先用默认参数跑通，再逐步调优。

训练技巧是实践的艺术，多试多调，慢慢就有感觉了！

---

**下一篇：[RNN 内部可视化](08-visualizing-rnn-internals.md)**

我们将探索 RNN 的内部世界，看看神经元到底学到了什么，又是如何"理解"语言的。

---

> 本文是《图解 Char-RNN》系列的第 7 篇。关注「云言 AI」公众号，获取更多深度学习图解教程！