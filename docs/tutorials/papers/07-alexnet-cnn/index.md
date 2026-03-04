# 07 AlexNet：CNN 是怎么让计算机"看懂"图片的？

问下大家，计算机是怎么"看懂"一张图片的？

晓寒刚开始学深度学习的时候，觉得这事儿太神奇了。我们人类看一眼就知道这是猫、那是狗，但计算机看到的只是一堆数字（像素值）啊！

直到后来理解了卷积神经网络（CNN），才发现卧槽，原来计算机是这样"看"图片的——它学会了像人眼一样提取特征！

2012年，AlexNet 在 ImageNet 比赛中一鸣惊人，把错误率从 26% 降到了 15%，从此开启了深度学习的黄金时代。今天我们就来拆解一下，AlexNet 到底做了什么？

## 卷积操作：用"小窗口"扫描图片

首先，我们得理解什么是卷积。

想象一下，你拿一个小放大镜，在图片上从左到右、从上到下慢慢移动。每到一个位置，你就把放大镜覆盖的区域"总结"成一个数字。这就是卷积！

### 卷积的数学原理

卷积操作用一个小的矩阵（叫卷积核或滤波器）在输入图像上滑动，逐点计算：

```
输入图像（5x5）          卷积核（3x3）
┌─┬─┬─┬─┬─┐            ┌─┬─┬─┐
│1│1│1│0│0│            │1│0│1│
├─┼─┼─┼─┼─┤            ├─┼─┼─┤
│0│1│1│1│0│     ⊛      │0│1│0│
├─┼─┼─┼─┼─┤            ├─┼─┼─┤
│0│0│1│1│1│            │1│0│1│
├─┼─┼─┼─┼─┤            └─┴─┴─┘
│0│0│1│1│0│
├─┼─┼─┼─┼─┤
│0│1│1│0│0│
└─┴─┴─┴─┴─┘

计算过程（第一个位置）：
= 1×1 + 1×0 + 1×1 + 0×0 + 1×1 + 1×0 + 0×1 + 0×0 + 1×1
= 1 + 0 + 1 + 0 + 1 + 0 + 0 + 0 + 1
= 4

输出特征图（3x3）：
┌─┬─┬─┐
│4│3│4│
├─┼─┼─┤
│2│4│3│
├─┼─┼─┤
│2│3│4│
└─┴─┴─┘
```

### 卷积的核心参数

**步长（Stride）**：卷积核每次移动几格
- stride=1：每次移动1格，输出尺寸大
- stride=2：每次移动2格，输出尺寸小

**填充（Padding）**：在图片边缘补0
- 不填充：输出会变小
- 填充：保持输出尺寸不变

**输出尺寸计算公式**：
```
输出尺寸 = ⌊(输入尺寸 - 卷积核尺寸 + 2×填充) / 步长⌋ + 1
```

### 纯 NumPy 实现卷积

```python
import numpy as np

def conv2d(input_data, kernel, stride=1, padding=0):
    """
    二维卷积操作
    
    Args:
        input_data: 输入图像，形状 (H, W) 或 (C, H, W)
        kernel: 卷积核，形状 (kH, kW) 或 (out_C, in_C, kH, kW)
        stride: 步长
        padding: 填充
    
    Returns:
        输出特征图
    """
    # 处理单通道情况
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, ...]  # (1, H, W)
    if kernel.ndim == 2:
        kernel = kernel[np.newaxis, np.newaxis, ...]  # (1, 1, kH, kW)
    
    # 添加填充
    if padding > 0:
        input_data = np.pad(input_data, 
                           ((0, 0), (padding, padding), (padding, padding)),
                           mode='constant')
    
    # 获取尺寸
    in_C, in_H, in_W = input_data.shape
    out_C, _, kH, kW = kernel.shape
    
    # 计算输出尺寸
    out_H = (in_H - kH) // stride + 1
    out_W = (in_W - kW) // stride + 1
    
    # 初始化输出
    output = np.zeros((out_C, out_H, out_W))
    
    # 执行卷积
    for oc in range(out_C):  # 遍历输出通道
        for i in range(out_H):  # 遍历高度
            for j in range(out_W):  # 遍历宽度
                # 提取局部区域
                h_start = i * stride
                w_start = j * stride
                region = input_data[:, h_start:h_start+kH, w_start:w_start+kW]
                
                # 计算卷积（所有输入通道求和）
                output[oc, i, j] = np.sum(region * kernel[oc])
    
    return output

# 测试卷积
input_img = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
], dtype=np.float32)

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
], dtype=np.float32)

output = conv2d(input_img, kernel)
print("卷积结果：")
print(output[0])
```

## 池化操作：把特征"压缩"一下

卷积提取了特征，但特征图还是很大。怎么办？池化！

池化就像给图片"降采样"，把相邻的几个像素合并成一个。最常用的是最大池化（Max Pooling）。

### 最大池化的过程

```
输入特征图（4x4）          2x2 池化窗口
┌─┬─┬─┬─┐
│1│3│2│1│            ┌─────┐
├─┼─┼─┼─┤            │ max │ stride=2
│2│9│1│1│   ────>     └─────┘
├─┼─┼─┼─┤
│1│3│2│3│            输出（2x2）
├─┼─┼─┼─┤            ┌─┬─┐
│5│6│1│2│            │9│2│
└─┴─┴─┴─┘            ├─┼─┤
                     │6│3│
                     └─┴─┘

每个窗口取最大值：
┌─────┐
│1  3│ -> max = 9    ┌─────┐
│2  9│              │2  1│ -> max = 2
└─────┘              │1  1│
                     └─────┘
```

### 纯 NumPy 实现池化

```python
def max_pool2d(input_data, pool_size=2, stride=2):
    """
    最大池化操作
    
    Args:
        input_data: 输入特征图，形状 (C, H, W)
        pool_size: 池化窗口大小
        stride: 步长
    
    Returns:
        池化后的特征图
    """
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, ...]
    
    C, in_H, in_W = input_data.shape
    out_H = (in_H - pool_size) // stride + 1
    out_W = (in_W - pool_size) // stride + 1
    
    output = np.zeros((C, out_H, out_W))
    
    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * stride
                w_start = j * stride
                region = input_data[c, h_start:h_start+pool_size, 
                                   w_start:w_start+pool_size]
                output[c, i, j] = np.max(region)
    
    return output

# 测试池化
feature_map = np.array([
    [1, 3, 2, 1],
    [2, 9, 1, 1],
    [1, 3, 2, 3],
    [5, 6, 1, 2]
], dtype=np.float32)

pooled = max_pool2d(feature_map)
print("池化结果：")
print(pooled[0])
```

## ReLU 激活函数：让网络"活"起来

AlexNet 之前，大家用 Sigmoid 或 Tanh。但这两个函数有个问题：梯度消失。

ReLU（Rectified Linear Unit）简单粗暴：
```
ReLU(x) = max(0, x)
```

负数变0，正数保持不变。就这么简单！

### ReLU 的优势

```
Sigmoid 函数              ReLU 函数
    1 ────────┐              │
   ╱          │              │
  ╱           │              │
 ╱            │         ─────┼─────
╱   0.5 ──────┤              │
0.5            │              │
               │              │
-4 -2 0 2 4    1         -4 -2 0 2 4

问题：两端梯度接近0        优势：正区间梯度恒为1
     → 梯度消失            → 梯度不消失
```

### 纯 NumPy 实现 ReLU

```python
def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU 导数"""
    return (x > 0).astype(np.float32)

# 测试
x = np.array([-2, -1, 0, 1, 2])
print("输入：", x)
print("ReLU：", relu(x))
print("导数：", relu_derivative(x))
```

## AlexNet 架构：深度学习的里程碑

AlexNet 的架构其实不复杂，就是"卷积+池化+全连接"的堆叠。但它的成功在于：

1. **够深**：8层（5卷积+3全连接）
2. **ReLU**：替代 Sigmoid，训练更快
3. **Dropout**：防止过拟合
4. **GPU 训练**：首次大规模使用 GPU
5. **数据增强**：随机裁剪、翻转

### AlexNet 网络结构

```
输入：224×224×3 的彩色图片

第1层：Conv1
├─ 96个 11×11 卷积核，stride=4
├─ 输出：55×55×96
├─ ReLU 激活
└─ Max Pooling 3×3，stride=2 → 27×27×96

第2层：Conv2
├─ 256个 5×5 卷积核，padding=2
├─ 输出：27×27×256
├─ ReLU 激活
└─ Max Pooling 3×3，stride=2 → 13×13×256

第3层：Conv3
├─ 384个 3×3 卷积核，padding=1
├─ 输出：13×13×384
└─ ReLU 激活

第4层：Conv4
├─ 384个 3×3 卷积核，padding=1
├─ 输出：13×13×384
└─ ReLU 激活

第5层：Conv5
├─ 256个 3×3 卷积核，padding=1
├─ 输出：13×13×256
├─ ReLU 激活
└─ Max Pooling 3×3，stride=2 → 6×6×256

第6层：FC6
├─ 展平：6×6×256 = 9216 → 4096
├─ ReLU 激活
└─ Dropout (p=0.5)

第7层：FC7
├─ 4096 → 4096
├─ ReLU 激活
└─ Dropout (p=0.5)

第8层：FC8
└─ 4096 → 1000 (ImageNet 1000类)
```

### 数据预处理流程

```
原始图片处理流程

256×256 原图
┌────────────────────┐
│                    │
│    原始图片        │
│                    │
└────────────────────┘
         │
         ├─ 随机裁剪 224×224
         │  ┌──────────┐
         │  │ 裁剪区域 │
         │  └──────────┘
         │
         ├─ 水平翻转
         │  ┌──────────┐    ┌──────────┐
         │  │ 原图     │ => │  镜像    │
         │  └──────────┘    └──────────┘
         │
         ├─ 颜色抖动
         │  改变亮度、对比度、饱和度
         │
         └─ 归一化
            减去均值，除以标准差
            使数据分布更稳定

效果：数据量增加 2048 倍！
```

### 网络结构可视化

```
AlexNet 完整架构（8层）

输入层                     卷积层                      全连接层
224×224×3                5层卷积                      3层FC

┌─────────┐
│ 输入图像 │
│ 224×224 │
│  3通道  │
└────┬────┘
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────┐
│ Conv1       │     │ 特征图变化：                          │
│ 11×11, s=4  │     │ 224×224×3 → 55×55×96 → 27×27×96     │
│ 96 filters  │────▶│ ↓ ReLU + MaxPool(3×3, s=2)          │
└─────────────┘     └──────────────────────────────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────┐
│ Conv2       │     │ 27×27×96 → 27×27×256 → 13×13×256    │
│ 5×5, p=2    │────▶│ ↓ ReLU + MaxPool(3×3, s=2)          │
│ 256 filters │     └──────────────────────────────────────┘
└─────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────┐
│ Conv3       │     │ 13×13×256 → 13×13×384               │
│ 3×3, p=1    │────▶│ ↓ ReLU                              │
│ 384 filters │     └──────────────────────────────────────┘
└─────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────┐
│ Conv4       │     │ 13×13×384 → 13×13×384               │
│ 3×3, p=1    │────▶│ ↓ ReLU                              │
│ 384 filters │     └──────────────────────────────────────┘
└─────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────┐
│ Conv5       │     │ 13×13×384 → 13×13×256 → 6×6×256     │
│ 3×3, p=1    │────▶│ ↓ ReLU + MaxPool(3×3, s=2)          │
│ 256 filters │     └──────────────────────────────────────┘
└─────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────┐
│ Flatten     │     │ 6×6×256 = 9216 → 一维向量            │
└─────────────┘     └──────────────────────────────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────┐
│ FC6         │     │ 9216 → 4096                          │
│ + Dropout   │────▶│ ↓ ReLU + Dropout(0.5)                │
└─────────────┘     └──────────────────────────────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────┐
│ FC7         │     │ 4096 → 4096                          │
│ + Dropout   │────▶│ ↓ ReLU + Dropout(0.5)                │
└─────────────┘     └──────────────────────────────────────┘
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────┐
│ FC8         │     │ 4096 → 1000                          │
│ (输出层)    │────▶│ ↓ Softmax（1000类概率）              │
└─────────────┘     └──────────────────────────────────────┘
```

### 参数量统计

```
层名称      │ 输入尺寸       │ 输出尺寸       │ 参数量
────────────┼────────────────┼────────────────┼──────────
Conv1       │ 224×224×3      │ 55×55×96       │ 34,944
Conv2       │ 27×27×96       │ 27×27×256      │ 614,656
Conv3       │ 13×13×256      │ 13×13×384      │ 885,120
Conv4       │ 13×13×384      │ 13×13×384      │ 1,327,488
Conv5       │ 13×13×384      │ 13×13×256      │ 884,992
FC6         │ 9216           │ 4096           │ 37,752,832
FC7         │ 4096           │ 4096           │ 16,781,312
FC8         │ 4096           │ 1000           │ 4,097,000
────────────┴────────────────┴────────────────┴──────────
总计        │                │                │ 62,378,344
                                                  ≈ 60M 参数

关键观察：
- 全连接层参数最多（占 94%）
- 卷积层参数少但计算量大
- 现代网络多用 1×1 卷积替代部分 FC
```

### 纯 NumPy 实现 AlexNet（简化版）

```python
import numpy as np

class AlexNetSimplified:
    """
    简化版 AlexNet（纯 NumPy 实现）
    
    为了教学目的，简化了部分结构：
    - 输入尺寸：64×64×3（原版224×224×3）
    - 通道数减少
    - 全连接层简化
    """
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.params = {}
        self._init_params()
    
    def _init_params(self):
        """初始化参数（使用 He 初始化）"""
        # Conv1: 3 -> 32, 11x11
        self.params['W1'] = np.random.randn(32, 3, 11, 11) * np.sqrt(2.0 / (3 * 11 * 11))
        self.params['b1'] = np.zeros(32)
        
        # Conv2: 32 -> 64, 5x5
        self.params['W2'] = np.random.randn(64, 32, 5, 5) * np.sqrt(2.0 / (32 * 5 * 5))
        self.params['b2'] = np.zeros(64)
        
        # Conv3: 64 -> 128, 3x3
        self.params['W3'] = np.random.randn(128, 64, 3, 3) * np.sqrt(2.0 / (64 * 3 * 3))
        self.params['b3'] = np.zeros(128)
        
        # FC1: 根据实际输出尺寸计算
        # 假设经过卷积池化后是 2x2x128 = 512
        self.params['W4'] = np.random.randn(512, 256) * np.sqrt(2.0 / 512)
        self.params['b4'] = np.zeros(256)
        
        # FC2: 256 -> num_classes
        self.params['W5'] = np.random.randn(256, self.num_classes) * np.sqrt(2.0 / 256)
        self.params['b5'] = np.zeros(self.num_classes)
    
    def forward(self, x, training=True):
        """
        前向传播
        
        Args:
            x: 输入图像，形状 (N, C, H, W)
            training: 是否训练模式（影响 Dropout）
        
        Returns:
            logits: 分类得分，形状 (N, num_classes)
        """
        # Conv1 + ReLU + Pool
        out1 = conv2d(x, self.params['W1'], stride=4, padding=0) + self.params['b1'].reshape(1, -1, 1, 1)
        out1 = relu(out1)
        out1 = max_pool2d(out1, pool_size=3, stride=2)
        
        # Conv2 + ReLU + Pool
        out2 = conv2d(out1, self.params['W2'], stride=1, padding=2) + self.params['b2'].reshape(1, -1, 1, 1)
        out2 = relu(out2)
        out2 = max_pool2d(out2, pool_size=3, stride=2)
        
        # Conv3 + ReLU
        out3 = conv2d(out2, self.params['W3'], stride=1, padding=1) + self.params['b3'].reshape(1, -1, 1, 1)
        out3 = relu(out3)
        
        # 展平
        N = x.shape[0]
        flat = out3.reshape(N, -1)
        
        # FC1 + ReLU + Dropout
        out4 = flat @ self.params['W4'] + self.params['b4']
        out4 = relu(out4)
        if training:
            dropout_mask = (np.random.rand(*out4.shape) > 0.5).astype(np.float32)
            out4 = out4 * dropout_mask / 0.5
        
        # FC2
        logits = out4 @ self.params['W5'] + self.params['b5']
        
        return logits
    
    def predict(self, x):
        """预测类别"""
        logits = self.forward(x, training=False)
        return np.argmax(logits, axis=1)

# 测试网络
model = AlexNetSimplified(num_classes=10)
dummy_input = np.random.randn(2, 3, 64, 64).astype(np.float32)
output = model.forward(dummy_input)
print(f"输入形状：{dummy_input.shape}")
print(f"输出形状：{output.shape}")
print(f"预测类别：{model.predict(dummy_input)}")
```

## 特征可视化：CNN 学到了什么？

CNN 的神奇之处在于，它能自动学习特征。我们来看看每一层学到了什么：

### 第一层：边缘和颜色

```
第一层卷积核可视化（类似 Gabor 滤波器）

┌─────────┐ ┌─────────┐ ┌─────────┐
│ ▓░▓░▓░▓ │ │ ░▓░▓░▓░ │ │ ╱╱╱╱╱╱ │
│ ░▓░▓░▓░ │ │ ▓░▓░▓░▓ │ │ ╱╱╱╱╱╱ │
│ ▓░▓░▓░▓ │ │ ░▓░▓░▓░ │ │ ╱╱╱╱╱╱ │
│ ░▓░▓░▓░ │ │ ▓░▓░▓░▓ │ │ ╱╱╱╱╱╱ │
└─────────┘ └─────────┘ └─────────┘
  水平边缘     垂直边缘      斜向边缘

这些卷积核学会了检测：
- 水平线条
- 垂直线条
- 斜向线条
- 颜色对比
```

### 第二层：纹理和图案

```
第二层特征可视化

┌───────────┐ ┌───────────┐ ┌───────────┐
│ ◯ ◯ ◯ ◯  │ │ ▲ ▲ ▲ ▲  │ │ ▭ ▭ ▭ ▭  │
│ ◯ ◯ ◯ ◯  │ │ ▲ ▲ ▲ ▲  │ │ ▥ ▥ ▥ ▥  │
│ ◯ ◯ ◯ ◯  │ │ ▲ ▲ ▲ ▲  │ │ ▭ ▭ ▭ ▭  │
└───────────┘ └───────────┘ └───────────┘
   圆点纹理      三角纹理      方格纹理

学会了更复杂的图案：
- 圆点图案
- 条纹图案
- 网格图案
```

### 第三层：物体部件

```
第三层特征可视化

┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   ◯     ◯   │ │   ══════    │ │    ╲ ╱     │
│    \   /    │ │   │    │    │ │     ◯      │
│     \ /     │ │   │    │    │ │    ╱ ╲     │
│      ◯      │ │   ══════    │ │   ══════    │
└─────────────┘ └─────────────┘ └─────────────┘
    眼睛特征        身体轮廓        轮胎特征

学会了物体部件：
- 眼睛、鼻子
- 轮胎、车窗
- 树叶、花瓣
```

### 高层：完整物体

```
第五层特征可视化

┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   ╭─────╮     │ │   ┌───┐      │ │    ╱╲        │
│   │ ◯ ◯ │     │ │   │   │      │ │   ╱  ╲       │
│   │  ▽  │     │ │   │   │      │ │  ╱    ╲      │
│   ╰─────╯     │ │   └───┘      │ │ ╱______╲     │
└───────────────┘ └───────────────┘ └───────────────┘
     人脸            汽车            树木

学会了完整物体：
- 人脸、狗、猫
- 汽车、自行车
- 飞机、船
```

### 特征层次总结

```
低层特征 → 中层特征 → 高层特征 → 分类结果

边缘/颜色    纹理/图案    物体部件    完整物体

┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│ ─── │     │ ▓▓░░ │     │ ◯ ◯ │     │ 🐱  │
│ │││ │ ──> │ ░░▓▓ │ ──> │  ▽  │ ──> │ 猫  │
│ ╱╱╱ │     │ ▓▓░░ │     │ ─── │     │     │
└─────┘     └─────┘     └─────┘     └─────┘

这就是 CNN 的"魔法"：
从简单特征逐步组合成复杂特征！
```

## 感受野计算：CNN 的"视野范围"

感受野（Receptive Field）是指输出特征图上一个像素对应输入图像上的区域大小。理解感受野对于理解 CNN 非常重要！

### 感受野的计算方法

```
单层感受野 = 卷积核尺寸

多层感受野计算公式：
RF_n = RF_{n-1} + (kernel_size - 1) × ∏_{i=1}^{n-1} stride_i

例子：AlexNet 前两层的感受野

第1层 Conv1：
- kernel = 11, stride = 4
- RF_1 = 11

第2层 Pool1：
- kernel = 3, stride = 2
- RF_2 = 11 + (3 - 1) × 4 = 11 + 8 = 19

第3层 Conv2：
- kernel = 5, stride = 1
- RF_3 = 19 + (5 - 1) × 4 × 2 = 19 + 32 = 51

可视化感受野扩张：

第1层：看到 11×11 区域
┌───────────┐
│ Conv1     │  RF = 11
└───────────┘

第2层：看到 19×19 区域
┌───────────────────┐
│     Pool1         │  RF = 19
└───────────────────┘

第3层：看到 51×51 区域
┌─────────────────────────────────────────┐
│              Conv2                       │  RF = 51
└─────────────────────────────────────────┘

第5层：看到约 195×195 区域（覆盖大部分输入）
┌──────────────────────────────────────────────────────────┐
│                      Conv5                                │  RF ≈ 195
└──────────────────────────────────────────────────────────┘

感受野的意义：
- 小感受野：只能看到局部细节（边缘、纹理）
- 大感受野：能看到整体结构（物体、场景）
- 深层网络：感受野逐层扩大，理解能力增强
```

### 感受野计算代码

```python
def compute_receptive_field(layers):
    """
    计算多层卷积的感受野
    
    Args:
        layers: 列表，每个元素是 (kernel_size, stride) 元组
    
    Returns:
        每层的感受野大小
    """
    rf = 1  # 初始感受野
    stride_product = 1  # 步长累积乘积
    rf_list = []
    
    for kernel_size, stride in layers:
        rf = rf + (kernel_size - 1) * stride_product
        stride_product *= stride
        rf_list.append(rf)
        print(f"Layer (k={kernel_size}, s={stride}): RF = {rf}")
    
    return rf_list

# AlexNet 前几层的感受野
alexnet_layers = [
    (11, 4),  # Conv1
    (3, 2),   # Pool1
    (5, 1),   # Conv2
    (3, 2),   # Pool2
    (3, 1),   # Conv3
]

print("AlexNet 感受野计算：")
rfs = compute_receptive_field(alexnet_layers)
```

## AlexNet 的创新点总结

AlexNet 为什么能成功？来看看它的五大创新：

### 1. ReLU 激活函数

```
传统 Sigmoid          AlexNet ReLU
    ____                  /
   /    \                /
  /      \              /
 /        \____________/
/                        \
梯度消失！              梯度不消失！

训练速度：慢            训练速度：快 6 倍
```

### 2. Dropout 正则化

```python
def dropout(x, p=0.5, training=True):
    """
    Dropout：随机"关掉"一部分神经元
    
    Args:
        x: 输入
        p: 丢弃概率
        training: 是否训练模式
    """
    if not training or p == 0:
        return x
    
    # 生成 mask
    mask = (np.random.rand(*x.shape) > p).astype(np.float32)
    
    # 缩放（保持期望不变）
    return x * mask / (1 - p)

# 为什么有效？
# 训练时：随机丢弃，防止过拟合
# 测试时：使用全部神经元，相当于集成学习
```

### 3. 数据增强

```python
def data_augmentation(image):
    """
    数据增强：从一张图片生成多个变体
    """
    augmented = []
    
    # 1. 随机裁剪
    h, w = image.shape[:2]
    for _ in range(5):
        i = np.random.randint(0, h - 224)
        j = np.random.randint(0, w - 224)
        crop = image[i:i+224, j:j+224]
        augmented.append(crop)
    
    # 2. 水平翻转
    augmented.append(np.fliplr(image))
    
    # 3. 颜色抖动
    # ...（省略实现）
    
    return augmented

# 效果：数据量增加 2048 倍！
```

### 4. GPU 并行训练

```
单 GPU 训练            双 GPU 并行
┌─────────┐           ┌─────┬─────┐
│  GPU 1  │           │GPU 1│GPU 2│
│  所有层 │           │前半 │后半 │
└─────────┘           └─────┴─────┘
慢！                   快 2-3 倍！

AlexNet 把网络拆成两半，
分别在两个 GTX 580 上训练
```

### 5. 局部响应归一化（LRN）

```python
def local_response_norm(x, k=2, n=5, alpha=1e-4, beta=0.75):
    """
    局部响应归一化
    
    模拟生物神经系统的"侧抑制"：
    当一个神经元激活时，抑制周围神经元
    """
    N, C, H, W = x.shape
    x_squared = x ** 2
    
    # 对每个位置，在通道维度上归一化
    scale = np.zeros_like(x)
    for i in range(C):
        # 计算局部和
        start = max(0, i - n // 2)
        end = min(C, i + n // 2 + 1)
        scale[:, i, :, :] = np.sum(x_squared[:, start:end, :, :], axis=1)
    
    scale = k + alpha * scale
    return x / (scale ** beta)

# 注：后来 Batch Norm 更好用，LRN 逐渐被淘汰
```

## 完整示例：用 AlexNet 识别手写数字

让我们用简化版 AlexNet 来识别 MNIST 手写数字：

```python
import numpy as np

def load_mnist_dummy():
    """生成模拟 MNIST 数据（实际使用时替换为真实数据）"""
    # 模拟 100 张 28x28 的灰度图
    X_train = np.random.rand(100, 1, 28, 28).astype(np.float32)
    y_train = np.random.randint(0, 10, 100)
    
    # 调整到 AlexNet 输入尺寸（简化版用 64x64）
    # 实际应用中用 resize 或 padding
    X_train_resized = np.zeros((100, 1, 64, 64))
    for i in range(100):
        X_train_resized[i, 0, 18:46, 18:46] = X_train[i, 0]
    
    # 转换为 3 通道（复制灰度图）
    X_train_rgb = np.repeat(X_train_resized, 3, axis=1)
    
    return X_train_rgb, y_train

def softmax(x):
    """Softmax 激活函数"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(logits, labels):
    """交叉熵损失"""
    probs = softmax(logits)
    N = logits.shape[0]
    log_probs = np.log(probs[np.arange(N), labels] + 1e-10)
    return -np.mean(log_probs)

# 训练示例
print("=== AlexNet 训练示例 ===\n")

# 加载数据
X_train, y_train = load_mnist_dummy()
print(f"训练数据：{X_train.shape}")
print(f"标签：{y_train.shape}\n")

# 创建模型
model = AlexNetSimplified(num_classes=10)

# 前向传播
logits = model.forward(X_train[:5], training=True)
print(f"输出 logits 形状：{logits.shape}")

# 计算损失
loss = cross_entropy_loss(logits, y_train[:5])
print(f"损失值：{loss:.4f}")

# 预测
predictions = model.predict(X_train[:5])
print(f"预测类别：{predictions}")
print(f"真实标签：{y_train[:5]}\n")

print("=== 训练完成 ===")
print("注：实际训练需要实现反向传播和优化器")
```

## 训练流程详解

### 训练循环伪代码

```python
def train_alexnet(model, train_data, val_data, epochs=90, batch_size=128):
    """
    AlexNet 训练流程
    
    训练策略：
    - 学习率：初始 0.01，验证集错误率不降时 ÷10
    - 权重衰减：0.0005
    - 动量：0.9
    - 总轮数：90
    """
    # 学习率调度
    lr = 0.01
    lr_schedule = {0: 0.01, 30: 0.001, 60: 0.0001, 80: 0.00001}
    
    # 优化器状态
    velocity = {k: np.zeros_like(v) for k, v in model.params.items()}
    
    for epoch in range(epochs):
        # 学习率调整
        if epoch in lr_schedule:
            lr = lr_schedule[epoch]
        
        # 训练一个 epoch
        train_loss = 0.0
        for batch_x, batch_y in get_batches(train_data, batch_size):
            # 1. 数据增强
            batch_x = random_crop(batch_x)
            batch_x = random_flip(batch_x)
            
            # 2. 前向传播
            logits = model.forward(batch_x, training=True)
            
            # 3. 计算损失
            loss = cross_entropy_loss(logits, batch_y)
            loss += weight_decay(model.params, decay=0.0005)
            
            # 4. 反向传播
            grads = backward(model, batch_x, batch_y)
            
            # 5. 参数更新（SGD + Momentum）
            for param_name in model.params:
                velocity[param_name] = 0.9 * velocity[param_name] - lr * grads[param_name]
                model.params[param_name] += velocity[param_name]
            
            train_loss += loss
        
        # 验证
        val_acc = evaluate(model, val_data)
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val Acc={val_acc:.2%}, LR={lr}")
    
    return model

# 关键训练技巧总结
training_tips = """
AlexNet 训练技巧：

1. 学习率预热
   - 初始 LR = 0.01
   - 验证错误率不降时手动降低
   - 共降低 3 次

2. 数据增强
   - 随机裁剪 224×224
   - 水平翻转
   - PCA 颜色增强
   - 数据量增加 2048 倍

3. 正则化
   - Dropout (p=0.5) 在 FC 层
   - 权重衰减 (0.0005)
   - 数据增强本身就是正则

4. 优化技巧
   - SGD + Momentum (0.9)
   - 权重衰减
   - 批量归一化（现代版本用 BN 替代 LRN）

5. 硬件
   - 2 × GTX 580 (3GB each)
   - 训练时间：5-6 天
   - 现代硬件：几小时
"""
```

### 损失函数详解

```python
def cross_entropy_with_regularization(logits, labels, params, weight_decay=0.0005):
    """
    带权重衰减的交叉熵损失
    
    L = -1/N * ∑ log(p_correct) + λ/2 * ∑ w²
    
    为什么需要权重衰减？
    - 防止权重过大
    - L2 正则化
    - 提高泛化能力
    """
    N = logits.shape[0]
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 交叉熵
    log_probs = np.log(probs[np.arange(N), labels] + 1e-10)
    ce_loss = -np.mean(log_probs)
    
    # L2 正则化（权重衰减）
    reg_loss = 0.0
    for name, param in params.items():
        if 'W' in name:  # 只对权重衰减，不对偏置
            reg_loss += np.sum(param ** 2)
    
    reg_loss *= weight_decay / 2
    
    total_loss = ce_loss + reg_loss
    
    return total_loss, ce_loss, reg_loss

# 测试损失函数
logits = np.random.randn(10, 1000).astype(np.float32)
labels = np.random.randint(0, 1000, 10)
params = {'W1': np.random.randn(100, 100), 'b1': np.zeros(100)}

total, ce, reg = cross_entropy_with_regularization(logits, labels, params)
print(f"总损失: {total:.4f}")
print(f"交叉熵: {ce:.4f}")
print(f"正则项: {reg:.4f}")
```

### 优化器：SGD + Momentum

```python
class SGDMomentum:
    """
    SGD with Momentum 优化器
    
    v_t = μ * v_{t-1} - lr * grad
    θ_t = θ_{t-1} + v_t
    
    优势：
    - 加速收敛
    - 跳出局部最小值
    - 减少震荡
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
        
        for name, param in params.items():
            self.velocity[name] = np.zeros_like(param)
    
    def step(self, params, grads):
        """更新参数"""
        for name in params:
            # 更新速度
            self.velocity[name] = (self.momentum * self.velocity[name] 
                                   - self.lr * grads[name])
            # 更新参数
            params[name] += self.velocity[name]
    
    def zero_grad(self):
        """清空梯度（这里仅作示例）"""
        pass

# Momentum 的作用可视化
momentum_intuition = """
没有 Momentum：
  ──→──→──→──→（沿着梯度方向走，可能震荡）

有 Momentum：
  ──→──→──→→→→（惯性推动，加速前进）

就像小球从山上滚下来：
- 没有惯性：走走停停
- 有惯性：越滚越快，能冲过小坑

超参数选择：
- momentum = 0.9（常用）
- 学习率需要相应调小
"""
```

## 练习题

### 1. 卷积计算

给定输入图像和卷积核，计算输出特征图：

```
输入（4x4）：          卷积核（2x2）：
┌─┬─┬─┬─┐            ┌─┬─┐
│1│2│3│4│            │1│0│
├─┼─┼─┼─┤            ├─┼─┤
│5│6│7│8│            │0│1│
├─┼─┼─┼─┤            └─┴─┘
│9│10│11│12│
├─┼─┼─┼─┤
│13│14│15│16│
└─┴─┴─┴─┘

stride=1, padding=0
输出尺寸是多少？第一个元素是多少？
```

<details>
<summary>点击查看答案</summary>

```
输出尺寸：(4-2)/1 + 1 = 3x3

第一个元素：
位置 (0,0) 的局部区域：
┌─┬─┐
│1│2│
├─┼─┤
│5│6│
└─┴─┘

计算：1×1 + 2×0 + 5×0 + 6×1 = 1 + 0 + 0 + 6 = 7

完整输出：
┌─┬─┬─┐
│7│9│11│
├─┼─┼─┤
│15│17│19│
├─┼─┼─┤
│23│25│27│
└─┴─┴─┘
```
</details>

### 2. 池化计算

给定特征图，计算最大池化结果：

```
输入（4x4）：          pool_size=2, stride=2
┌─┬─┬─┬─┐
│2│4│1│3│
├─┼─┼─┼─┤
│6│5│7│8│
├─┼─┼─┼─┤
│9│3│2│1│
├─┼─┼─┼─┤
│4│7│5│6│
└─┴─┴─┴─┘

输出是多少？
```

<details>
<summary>点击查看答案</summary>

```
输出（2x2）：
┌─┬─┐
│6│8│
├─┼─┤
│9│6│
└─┴─┘

计算过程：
左上：max(2,4,6,5) = 6
右上：max(1,3,7,8) = 8
左下：max(9,3,4,7) = 9
右下：max(2,1,5,6) = 6
```
</details>

### 3. 参数计算

AlexNet 第一层卷积：
- 输入：224×224×3
- 卷积核：11×11，stride=4，96个
- 无 padding

问：
1. 输出特征图尺寸是多少？
2. 这一层的参数量是多少？

<details>
<summary>点击查看答案</summary>

```
1. 输出尺寸：
   H_out = (224 - 11) / 4 + 1 = 54
   W_out = (224 - 11) / 4 + 1 = 54
   输出：54×54×96

2. 参数量：
   卷积核：11×11×3×96 = 34,848
   偏置：96
   总计：34,944 个参数
```
</details>

### 4. 编程练习

实现一个完整的卷积层类，包含前向传播：

```python
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """初始化卷积层"""
        # TODO: 初始化参数
        
    def forward(self, x):
        """前向传播"""
        # TODO: 实现卷积
        pass

# 测试
conv = ConvLayer(3, 16, kernel_size=3, stride=1, padding=1)
x = np.random.randn(1, 3, 32, 32)
out = conv.forward(x)
print(f"输出形状：{out.shape}")  # 应该是 (1, 16, 32, 32)
```

<details>
<summary>点击查看答案</summary>

```python
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """初始化卷积层"""
        self.stride = stride
        self.padding = padding
        
        # He 初始化
        self.W = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        
        self.b = np.zeros(out_channels)
    
    def forward(self, x):
        """前向传播"""
        return conv2d(x, self.W, self.stride, self.padding) + self.b.reshape(1, -1, 1, 1)

# 测试
conv = ConvLayer(3, 16, kernel_size=3, stride=1, padding=1)
x = np.random.randn(1, 3, 32, 32)
out = conv.forward(x)
print(f"输出形状：{out.shape}")  # (1, 16, 32, 32)
```
</details>

## 延伸阅读

### 论文原文

- **ImageNet Classification with Deep Convolutional Neural Networks** (AlexNet, 2012)
  - 作者：Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
  - 链接：https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

### 相关论文

1. **LeNet-5** (1998) - CNN 的鼻祖
   - Yann LeCun 的经典之作
   - 手写数字识别

2. **VGGNet** (2014) - 更深的网络
   - 使用小卷积核（3×3）堆叠
   - 证明深度的重要性

3. **ResNet** (2015) - 残差连接
   - 解决深层网络训练问题
   - 跳跃连接

4. **Inception/GoogLeNet** (2014) - 多尺度特征
   - 并行使用不同尺寸卷积核
   - 提高特征提取能力

### 可视化工具

- **TensorFlow Playground**：在线可视化神经网络
- **CNN Explainer**：交互式 CNN 可视化
- **Feature Map Visualization**：特征图可视化工具

### 进阶主题

1. **Batch Normalization**：替代 LRN，效果更好
2. **转置卷积**：用于上采样和生成任务
3. **空洞卷积**：扩大感受野
4. **深度可分离卷积**：MobileNet 的核心

## 总结

综上所述，AlexNet 的成功不是偶然，而是多个创新点的组合：

1. **ReLU 激活**：解决梯度消失，训练更快
2. **Dropout**：防止过拟合，提高泛化
3. **数据增强**：扩充数据，提升鲁棒性
4. **GPU 训练**：加速计算，缩短周期
5. **深层架构**：提取层次化特征

CNN 的核心思想就是：**用卷积提取局部特征，用池化降低维度，用深层网络组合复杂特征**。

从边缘到纹理，从部件到物体，CNN 像人眼一样，层层递进地"看懂"图片！

![](https://cdn.xiaolincoding.com/gh/xiaolincoder/ImageHost4@main/深度学习/alexnet总结.png)

---

**下一篇预告**：第8篇我们将学习 Word2Vec，看看如何用神经网络学习词向量，让计算机理解语言的语义！

---

> **关注公众号「小林 coding」，回复「深度学习」获取完整代码和数据集！**
> 
> ![](https://cdn.xiaolincoding.com/gh/xiaolincoder/ImageHost3@main/其他/公众号二维码.jpg)