# 预激活 ResNet 如何改善梯度流?

问下大家,还记得我们之前学习的 ResNet 吗?残差连接让深层网络训练成为可能。但是,ResNet 还有改进空间!

晓寒深入研究 ResNet 后发现,原始的 ResNet 残差块在最后还有一个 ReLU 激活函数,这会影响梯度的传播。直到 2016 年,何恺明大神提出了预激活 ResNet(Pre-activation ResNet),才发现卧槽,原来把激活函数放到前面,梯度流会更顺畅!

## 原始 ResNet 的问题

### 后激活残差块

```
原始 ResNet 残差块:

输入 x
  ↓
  ├──────────────────────┐
  ↓                      │
Conv → BN → ReLU        │
  ↓                      │
Conv → BN               │
  ↓                      │
[+] ←───────────────────┘  恒等映射
  ↓
ReLU  ← 这里的 ReLU 会影响梯度流!
  ↓
输出
```

### 问题分析

```python
import numpy as np

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def batch_norm_1d(x, gamma=1.0, beta=0.0, eps=1e-5):
    """简化的批归一化"""
    mean = np.mean(x)
    var = np.var(x)
    x_normalized = (x - mean) / np.sqrt(var + eps)
    return gamma * x_normalized + beta

class OriginalResidualBlock:
    """原始 ResNet 残差块(后激活)"""
    def __init__(self, dim):
        self.dim = dim
        # 两个卷积层
        self.W1 = np.random.randn(dim, dim) * 0.01
        self.W2 = np.random.randn(dim, dim) * 0.01
        
    def forward(self, x):
        """
        原始: x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
        
        问题: 最后的 ReLU 会阻断梯度!
        """
        # 第一个 Conv-BN-ReLU
        out = np.dot(self.W1, x)
        out = batch_norm_1d(out)
        out = relu(out)
        
        # 第二个 Conv-BN
        out = np.dot(self.W2, out)
        out = batch_norm_1d(out)
        
        # 残差连接
        out = out + x
        
        # 最后的 ReLU (问题所在!)
        out = relu(out)
        
        return out

# 测试
original_block = OriginalResidualBlock(dim=8)
x = np.random.randn(8)
output_original = original_block.forward(x)

print("原始 ResNet 残差块:")
print(f"输入: {x[:4]}")
print(f"输出: {output_original[:4]}")
print("\n问题: 最后的 ReLU 可能会把负值变成 0")
print("如果 x + F(x) < 0,梯度就会消失!")
```

### 梯度消失分析

```
假设 x + F(x) = -1

后激活 ResNet:
输出 = ReLU(x + F(x)) = ReLU(-1) = 0
梯度: ∂ReLU/∂(x+F(x)) = 0  # 梯度消失!

预激活 ResNet:
输出 = x + F(x) = -1
梯度: ∂(x+F(x))/∂x = 1  # 梯度正常流动!

关键区别: 
- 后激活: 最后的 ReLU 可能阻断梯度
- 预激活: 恒等映射路径干净无阻碍
```

## 预激活 ResNet 的改进

### 预激活残差块

```
预激活 ResNet 残差块:

输入 x
  ↓
  ├──────────────────────┐
  ↓                      │
BN → ReLU → Conv        │
  ↓                      │
BN → ReLU → Conv        │
  ↓                      │
[+] ←───────────────────┘  恒等映射(干净!)
  ↓
输出 (没有 ReLU!)
```

### 核心改进

```python
class PreActivationResidualBlock:
    """预激活 ResNet 残差块(改进版)"""
    def __init__(self, dim):
        self.dim = dim
        self.W1 = np.random.randn(dim, dim) * 0.01
        self.W2 = np.random.randn(dim, dim) * 0.01
        
    def forward(self, x):
        """
        预激活: x → BN → ReLU → Conv → BN → ReLU → Conv → (+x)
        
        优势: 
        1. 恒等映射路径干净
        2. 梯度直接回传
        3. 训练更稳定
        """
        # 第一个 BN-ReLU-Conv
        out = batch_norm_1d(x)
        out = relu(out)
        out = np.dot(self.W1, out)
        
        # 第二个 BN-ReLU-Conv
        out = batch_norm_1d(out)
        out = relu(out)
        out = np.dot(self.W2, out)
        
        # 残差连接 (没有激活!)
        out = out + x
        
        return out

# 测试
preact_block = PreActivationResidualBlock(dim=8)
output_preact = preact_block.forward(x)

print("\n预激活 ResNet 残差块:")
print(f"输入: {x[:4]}")
print(f"输出: {output_preact[:4]}")
print("\n优势: 输出可以是负值,梯度流动无阻碍!")
```

## 梯度流对比

### 模拟梯度传播

```python
def simulate_gradient_flow(block_type, num_layers=20, dim=8):
    """
    模拟梯度在多层残差块中的传播
    """
    np.random.seed(42)
    
    # 创建多层残差块
    if block_type == 'original':
        blocks = [OriginalResidualBlock(dim) for _ in range(num_layers)]
    else:
        blocks = [PreActivationResidualBlock(dim) for _ in range(num_layers)]
    
    # 初始化梯度
    grad = np.ones(dim)
    gradients = [grad.copy()]
    
    # 反向传播模拟
    for i in range(num_layers):
        if block_type == 'original':
            # 后激活: 梯度可能被 ReLU 阻断
            # 简化模拟: 有一定概率梯度消失
            relu_mask = np.random.choice([0, 1], size=dim, p=[0.3, 0.7])
            grad = grad * relu_mask
        else:
            # 预激活: 梯度直接通过恒等映射
            # 简化模拟: 梯度几乎不受影响
            grad = grad * np.random.uniform(0.9, 1.0, dim)
        
        gradients.append(grad.copy())
    
    return np.array(gradients)

# 模拟梯度传播
gradients_original = simulate_gradient_flow('original', num_layers=20)
gradients_preact = simulate_gradient_flow('preact', num_layers=20)

# 计算梯度范数
norms_original = np.linalg.norm(gradients_original, axis=1)
norms_preact = np.linalg.norm(gradients_preact, axis=1)

print("梯度范数对比 (20层):")
print("\n原始 ResNet:")
print(f"  第 1 层: {norms_original[0]:.4f}")
print(f"  第 10 层: {norms_original[9]:.4f}")
print(f"  第 20 层: {norms_original[19]:.4f}")

print("\n预激活 ResNet:")
print(f"  第 1 层: {norms_preact[0]:.4f}")
print(f"  第 10 层: {norms_preact[9]:.4f}")
print(f"  第 20 层: {norms_preact[19]:.4f}")

print("\n观察:")
print("- 预激活 ResNet 梯度衰减更慢")
print("- 训练更稳定,尤其是超深层网络")
```

## 为什么预激活更好?

### 理论分析

**1. 干净的恒等映射**

```
预激活:
y = x + F(BN(ReLU(x)))
∂y/∂x = 1 + ∂F/∂x

后激活:
y = ReLU(x + F(x))
∂y/∂x = ReLU'(x+F(x)) · (1 + ∂F/∂x)

关键区别:
- 预激活: 总有 "1" 保证梯度流
- 后激活: ReLU' 可能是 0,阻断梯度
```

**2. BN 作为正则化**

```python
print("BN 的作用:")
print("\n后激活 ResNet:")
print("- BN 在卷积后,主要是归一化特征")
print("- 没有直接对恒等映射正则化")

print("\n预激活 ResNet:")
print("- BN 在卷积前,对输入正则化")
print("- 相当于对恒等映射施加正则化")
print("- 每层都在"修正"上一层的输出")
```

**3. 训练动态**

```
训练初期:

后激活:
- 权重接近 0, F(x) ≈ 0
- 输出 ≈ ReLU(x)
- 如果 x < 0,输出就是 0

预激活:
- 权重接近 0, F(x) ≈ 0
- 输出 ≈ x (恒等映射)
- 至少能传递信息!

预激活更容易训练!
```

## 超深层网络训练

### 实验结果

```
CIFAR-10 测试错误率:

1001 层 ResNet:
- 后激活: 7.61% (过拟合,训练困难)
- 预激活: 4.92% (训练良好,泛化更好)

改进:
- 训练速度更快
- 最终准确率更高
- 能够训练 1000+ 层网络
```

### 代码示例

```python
def build_deep_resnet(num_layers, block_type='preact'):
    """
    构建深层 ResNet
    
    参数:
        num_layers: 层数(如 50, 101, 152, 1001)
        block_type: 'original' 或 'preact'
    """
    print(f"构建 {num_layers} 层 ResNet ({block_type})")
    
    if block_type == 'preact':
        block_class = PreActivationResidualBlock
    else:
        block_class = OriginalResidualBlock
    
    # 计算每个阶段的块数
    # ResNet-50: [3, 4, 6, 3]
    # ResNet-101: [3, 4, 23, 3]
    # ResNet-152: [3, 8, 36, 3]
    
    if num_layers == 50:
        blocks_per_stage = [3, 4, 6, 3]
    elif num_layers == 101:
        blocks_per_stage = [3, 4, 23, 3]
    elif num_layers == 152:
        blocks_per_stage = [3, 8, 36, 3]
    else:
        # 对于超深层,平均分配
        blocks_per_stage = [num_layers // 4] * 4
    
    print(f"每个阶段的块数: {blocks_per_stage}")
    
    return {
        'num_layers': num_layers,
        'blocks_per_stage': blocks_per_stage,
        'block_type': block_type
    }

# 对比
print("ResNet 架构对比:")
print(build_deep_resnet(50, 'preact'))
print()
print(build_deep_resnet(50, 'original'))
```

## 实际应用建议

### 何时使用预激活?

```python
print("预激活 ResNet 适用场景:")
print("\n1. 超深层网络 (100+ 层)")
print("   - 预训练效果显著更好")
print("   - 训练更稳定")

print("\n2. 训练困难任务")
print("   - 数据量小")
print("   - 模型容易过拟合")

print("\n3. 精度要求高的应用")
print("   - ImageNet 分类")
print("   - 目标检测")
print("   - 医学图像分析")
```

### 实现技巧

```python
class PreActivationResNet:
    """完整的预激活 ResNet"""
    def __init__(self, num_layers=50):
        self.num_layers = num_layers
        
        # 初始卷积: 7x7, stride=2
        self.conv1 = ...  # 初始层
        
        # 四个阶段
        self.stage1 = self._make_stage(64, 3)
        self.stage2 = self._make_stage(128, 4)
        self.stage3 = self._make_stage(256, 6)
        self.stage4 = self._make_stage(512, 3)
        
        # 全局平均池化 + 全连接层
        self.avgpool = ...
        self.fc = ...
    
    def _make_stage(self, channels, num_blocks):
        """构建一个阶段"""
        blocks = []
        for _ in range(num_blocks):
            blocks.append(PreActivationResidualBlock(channels))
        return blocks
    
    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        
        # 四个阶段
        for block in self.stage1:
            x = block.forward(x)
        for block in self.stage2:
            x = block.forward(x)
        for block in self.stage3:
            x = block.forward(x)
        for block in self.stage4:
            x = block.forward(x)
        
        # 输出层
        x = self.avgpool(x)
        x = self.fc(x)
        
        return x

print("预激活 ResNet 实现要点:")
print("1. 初始层保持不变")
print("2. 所有残差块使用预激活")
print("3. 最后的全连接层保持不变")
print("4. 训练时使用 BN 和 Dropout")
```

## 小结

今天我们深入理解了预激活 ResNet:

### 核心改进

1. **激活函数前置**: BN → ReLU → Conv
2. **干净的恒等映射**: 没有最后的 ReLU
3. **更好的梯度流**: 梯度直接回传

### 为什么有效

1. **梯度传播**: 避免ReLU阻断梯度
2. **训练稳定**: 初始化时输出接近恒等映射
3. **正则化**: BN对恒等映射施加正则化

### 实践建议

```
网络深度 vs 架构选择:

< 50 层:
- 原始 ResNet 就够用
- 预激活优势不明显

50-100 层:
- 预激活略好
- 都可以训练

> 100 层:
- 强烈推荐预激活
- 训练稳定性和精度都更好

最佳实践:
- 新项目直接用预激活 ResNet
- 兼容性好,性能更优
```

## 练习题

### 1. 概念理解

**问题 1**: 为什么原始 ResNet 最后的 ReLU 会影响梯度流?

**问题 2**: 预激活 ResNet 的 "预激活" 是指什么?

**问题 3**: 预激活 ResNet 和原始 ResNet 在计算量上有区别吗?

### 2. 编程实践

**练习 1**: 实现完整的预激活 ResNet-50:

```python
class PreActivationResNet50:
    """预激活 ResNet-50"""
    # TODO: 实现
    pass
```

**练习 2**: 对比两种 ResNet 的训练曲线:

```python
def train_and_compare():
    """训练并对比原始 ResNet 和预激活 ResNet"""
    # TODO: 实现完整的训练流程
    pass
```

### 3. 深度思考

**思考 1**: 预激活的思想能否应用到其他架构(如 DenseNet、Transformer)?

**思考 2**: 为什么 BN 放在卷积前比放在卷积后效果好?

**思考 3**: 预激活 ResNet 能否进一步改进?

## 延伸阅读

### 经典论文

1. **预激活 ResNet**: He et al. (2016). "Identity Mappings in Deep Residual Networks"
2. **ResNet V2**: 改进版的 ResNet 实现

### 相关资源

- "Deep Residual Learning for Image Recognition" - ResNet 原始论文
- "ResNet with Identity Mapping" - 预激活 ResNet 详解
- PyTorch 官方 ResNet 实现

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 15 篇。上一篇我们学习了 Bahdanau 注意力机制如何改进机器翻译,下一篇我们将探讨关系推理网络如何处理对象之间的关系。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!** 🚀