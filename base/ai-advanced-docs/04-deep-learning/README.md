# 阶段 4：深度学习

_神经网络原理与实现_

---

## 📖 学习指南

**前置知识**：
- ✅ Python 编程基础
- ✅ 机器学习基础
- ✅ 线性代数、微积分、概率论

**学习目标**：
- ✅ 理解神经网络的基本原理
- ✅ 掌握反向传播算法
- ✅ 掌握 CNN 原理与实现
- ✅ 掌握 RNN、LSTM、GRU
- ✅ 了解经典网络架构
- ✅ 能独立完成深度学习项目

**预计时间**：60 天

---

## 4.1 神经网络基础

### 从生物神经元到人工神经元

<div class="formula-box">

```
生物神经元：
树突（输入）→ 细胞体（处理）→ 轴突（输出）
        ↓
人工神经元（感知机）：
输入 x → 加权求和 → 激活函数 → 输出 y
```

</div>

### 感知机

<div class="formula-box">

```
模型：
y = f(w·x + b)

其中：
- x：输入向量
- w：权重向量
- b：偏置
- f：激活函数

局限：
只能解决线性可分问题
无法解决 XOR 问题
```

</div>

<div class="formula-box">

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=100):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = lr
        self.epochs = epochs
    
    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                prediction = self.step_function(np.dot(xi, self.weights) + self.bias)
                error = yi - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
    
    def predict(self, X):
        return np.array([self.step_function(np.dot(xi, self.weights) + self.bias) for xi in X])
```

</div>

### 多层感知机（MLP）

<div class="formula-box">

```
结构：
输入层 → 隐藏层 → 输出层

前向传播：
z₁ = W₁·x + b₁
a₁ = f(z₁)
z₂ = W₂·a₁ + b₂
ŷ = g(z₂)

其中：
- f：隐藏层激活函数（ReLU、tanh）
- g：输出层激活函数（softmax、sigmoid）
```

</div>

### 激活函数

<div class="formula-box">

```
1. Sigmoid
   f(x) = 1 / (1 + exp(-x))
   范围：(0, 1)
   问题：梯度消失

2. Tanh
   f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
   范围：(-1, 1)
   零中心化，优于 Sigmoid

3. ReLU（最常用）
   f(x) = max(0, x)
   范围：[0, +∞)
   优势：计算简单、缓解梯度消失

4. Leaky ReLU
   f(x) = max(αx, x)
   解决 ReLU 的"神经元死亡"问题

5. Softmax（多分类输出层）
   f(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

</div>

<div class="formula-box">

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

</div>

---

## 4.2 反向传播算法

### 核心思想

<div class="formula-box">

```
反向传播 = 链式法则 + 梯度下降

目标：
最小化损失函数 L(w, b)

方法：
计算 ∂L/∂w 和 ∂L/∂b
更新 w = w - η·∂L/∂w
更新 b = b - η·∂L/∂b
```

</div>

### 损失函数

<div class="formula-box">

```
1. 回归问题：MSE（均方误差）
   L = (1/n) Σ(yᵢ - ŷᵢ)²

2. 二分类：二元交叉熵
   L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

3. 多分类：交叉熵
   L = -Σ yᵢ·log(ŷᵢ)
```

</div>

### 反向传播推导（以 MLP 为例）

<div class="formula-box">

```
网络结构：
输入 x → 隐藏层 z₁=W₁x+b₁, a₁=f(z₁) → 输出 z₂=W₂a₁+b₂, ŷ=g(z₂)

损失函数：L = (1/2)||y - ŷ||²

反向传播：
1. 输出层误差：δ₂ = ∂L/∂z₂ = (ŷ - y) ⊙ g'(z₂)
2. 隐藏层误差：δ₁ = (W₂ᵀ·δ₂) ⊙ f'(z₁)
3. 梯度计算：
   ∂L/∂W₂ = δ₂·a₁ᵀ
   ∂L/∂b₂ = δ₂
   ∂L/∂W₁ = δ₁·xᵀ
   ∂L/∂b₁ = δ₁
```

</div>

<div class="formula-box">

```python
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重（Xavier 初始化）
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.output = self.z2  # 回归问题
        return self.output
    
    def backward(self, X, y, lr=0.01):
        m = X.shape[0]
        
        # 输出层误差
        delta2 = (self.output - y)  # MSE 导数
        dW2 = (self.a1.T @ delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        # 隐藏层误差
        delta1 = (delta2 @ self.W2.T) * self.relu_derivative(self.z1)
        dW1 = (X.T @ delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        # 更新权重
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
    
    def train(self, X, y, epochs=1000, lr=0.01):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, lr)
            if i % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {i}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return self.forward(X)
```

</div>

---

## 4.3 优化算法

### SGD（随机梯度下降）

<div class="formula-box">

```
w = w - η·∇L(w)

特点：
- 每次更新用一个样本
- 方差大，收敛不稳定
- 容易跳出局部最优
```

</div>

### Mini-batch SGD

<div class="formula-box">

```
w = w - η·∇L(w; B)

其中 B 是一个 batch（通常 32、64、128）

特点：
- 平衡了稳定性和速度
- 最常用的优化方式
```

</div>

### 动量法（Momentum）

<div class="formula-box">

```
vₜ = γ·vₜ₋₁ + η·∇L(w)
w = w - vₜ

其中：
- γ：动量系数（通常 0.9）
- v：累积梯度

优势：
- 加速收敛
- 减少震荡
```

</div>

### Adam（最常用）

<div class="formula-box">

```
mₜ = β₁·mₜ₋₁ + (1-β₁)·∇L(w)  # 一阶矩（动量）
vₜ = β₂·vₜ₋₁ + (1-β₂)·(∇L(w))²  # 二阶矩（自适应）
m̂ₜ = mₜ / (1-β₁ᵗ)  # 偏差校正
v̂ₜ = vₜ / (1-β₂ᵗ)
w = w - η·m̂ₜ / (√v̂ₜ + ε)

默认参数：
β₁=0.9, β₂=0.999, ε=1e-8
```

</div>

<div class="formula-box">

```python
import numpy as np

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {id(p): np.zeros_like(p) for p in params}
        self.v = {id(p): np.zeros_like(p) for p in params}
        self.t = 0
    
    def step(self, grads):
        self.t += 1
        for param, grad in zip(self.params, grads):
            key = id(param)
            # 更新一阶矩
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            # 更新二阶矩
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            # 偏差校正
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            # 更新参数
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

</div>

---

## 4.4 正则化技术

### Dropout

<div class="formula-box">

```
训练时：
随机"关闭"一部分神经元（概率 p）

测试时：
所有神经元都保留，输出乘以 (1-p)

作用：
- 减少过拟合
- 强制网络学习冗余表示
```

</div>

<div class="formula-box">

```python
def dropout(x, p=0.5, training=True):
    if training and p > 0:
        mask = (np.random.rand(*x.shape) > p) / (1 - p)
        return x * mask
    return x
```

</div>

### Batch Normalization

<div class="formula-box">

```
对每个 batch 进行归一化：

μ = (1/m) Σxᵢ
σ² = (1/m) Σ(xᵢ - μ)²
x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
yᵢ = γ·x̂ᵢ + β

其中：
- γ、β 是可学习参数
- ε 防止除零

优势：
- 加速收敛
- 减少对初始化的敏感
- 有一定的正则化效果
```

</div>

<div class="formula-box">

```python
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            # 更新运行统计
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和平移
        out = self.gamma * x_norm + self.beta
        
        return out
```

</div>

### L2 正则化（权重衰减）

<div class="formula-box">

```
损失函数：
L = L_original + (λ/2) Σw²

梯度更新：
w = w - η·(∂L/∂w + λ·w)

作用：
- 限制权重大小
- 减少过拟合
```

</div>

---

## 4.5 卷积神经网络（CNN）

### 卷积层

<div class="formula-box">

```
卷积操作：
用卷积核在图像上滑动，计算点积

参数：
- 卷积核大小（3×3, 5×5）
- 步长（stride）
- 填充（padding）
- 卷积核数量

输出尺寸：
H_out = (H_in - kernel_size + 2×padding) / stride + 1
```

</div>

<div class="formula-box">

```python
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Xavier 初始化
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros(out_channels)
    
    def forward(self, x):
        # x: (batch, in_channels, H, W)
        batch_size = x.shape[0]
        
        # 填充
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        
        # 输出尺寸
        out_h = (x.shape[2] - self.kernel_size) // self.stride + 1
        out_w = (x.shape[3] - self.kernel_size) // self.stride + 1
        
        # 卷积
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                # 提取 patch
                patch = x[:, :, h_start:h_end, w_start:w_end]
                
                # 卷积计算
                for k in range(self.out_channels):
                    out[:, k, i, j] = np.sum(patch * self.weights[k], axis=(1, 2, 3)) + self.bias[k]
        
        return out
```

</div>

### 池化层

<div class="formula-box">

```
最大池化（最常用）：
取窗口内的最大值

平均池化：
取窗口内的平均值

作用：
- 降维
- 减少计算量
- 增加平移不变性
```

</div>

<div class="formula-box">

```python
class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
    
    def forward(self, x):
        batch_size, channels, h, w = x.shape
        
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        
        out = np.zeros((batch_size, channels, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                out[:, :, i, j] = np.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
        
        return out
```

</div>

### 经典 CNN 架构

<div class="formula-box">

```
LeNet-5 (1998)：
输入 → Conv → Pool → Conv → Pool → FC → FC → 输出
应用：手写数字识别

AlexNet (2012)：
5 个卷积层 + 3 个全连接层
创新：ReLU、Dropout、数据增强
应用：ImageNet 竞赛冠军

VGG (2014)：
使用小卷积核（3×3）堆叠
VGG16、VGG19
特点：结构简单、参数多

ResNet (2015)：
残差连接：y = F(x) + x
解决深层网络退化问题
ResNet18/34/50/101/152
```

</div>

---

## 4.6 循环神经网络（RNN）

### RNN 基础

<div class="formula-box">

```
RNN 结构：
hₜ = f(W·hₜ₋₁ + U·xₜ + b)
yₜ = g(V·hₜ + c)

其中：
- hₜ：t 时刻的隐藏状态
- xₜ：t 时刻的输入
- W、U、V：权重矩阵

特点：
- 有"记忆"能力
- 可处理变长序列
- 参数共享
```

</div>

### LSTM（长短期记忆）

<div class="formula-box">

```
LSTM 单元：
遗忘门：fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)
输入门：iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)
候选值：c̃ₜ = tanh(W_c·[hₜ₋₁, xₜ] + b_c)
细胞状态：cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ
输出门：oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)
隐藏状态：hₜ = oₜ ⊙ tanh(cₜ)

优势：
- 解决长距离依赖
- 解决梯度消失
```

</div>

### GRU（门控循环单元）

<div class="formula-box">

```
GRU 单元（LSTM 的简化版）：
更新门：zₜ = σ(W_z·[hₜ₋₁, xₜ])
重置门：rₜ = σ(W_r·[hₜ₋₁, xₜ])
候选值：h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])
隐藏状态：hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ

优势：
- 参数更少
- 训练更快
- 效果与 LSTM 相当
```

</div>

---

## 4.7 实战项目

### 项目 1：MNIST 手写数字识别

<div class="formula-box">

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 加载数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

# 2. 定义网络
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# 3. 训练
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    
    # 测试
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")
```

</div>

### 项目 2：文本情感分析（LSTM）

<div class="formula-box">

```python
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        # text: (batch, seq_len)
        embedded = self.dropout(self.embedding(text))  # (batch, seq_len, embedding_dim)
        
        # LSTM
        output, (hidden, cell) = self.lstm(embedded)
        # hidden: (n_layers, batch, hidden_dim)
        
        # 使用最后一层的隐藏状态
        hidden = self.dropout(hidden[-1])
        
        # 分类
        return self.fc(hidden)

# 使用示例
model = SentimentLSTM(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    output_dim=2,  # 正面/负面
    n_layers=2,
    dropout=0.5
)
```

</div>

---

## 📚 学习资源

### 课程

- [吴恩达深度学习专项](https://www.coursera.org/specializations/deep-learning)
- [李飞飞 CS231n](http://cs231n.stanford.edu/) - CNN 视觉识别
- [李宏毅深度学习](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)

### 书籍

- 《深度学习》Ian Goodfellow（花书）
- 《动手学深度学习》李沐
- 《Deep Learning with Python》François Chollet

### 框架

- [PyTorch](https://pytorch.org/) - 研究首选
- [TensorFlow](https://www.tensorflow.org/) - 工业部署
- [Keras](https://keras.io/) - 快速原型

---

## ✅ 学习检查清单

- [ ] 理解感知机与 MLP
- [ ] 掌握常见激活函数
- [ ] 掌握反向传播推导
- [ ] 掌握常见优化算法
- [ ] 掌握 Dropout、BatchNorm
- [ ] 理解卷积操作
- [ ] 理解池化操作
- [ ] 了解经典 CNN 架构
- [ ] 理解 RNN、LSTM、GRU
- [ ] 完成至少 2 个实战项目

---

*最后更新：2026-04-22*
