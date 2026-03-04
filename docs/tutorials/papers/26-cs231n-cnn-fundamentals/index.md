# CS231n 的 CNN 视觉入门到底是怎么从像素走到分类的?

问下大家,你第一次写图像分类时是不是也经历过这个心路历程:

"我现在手里只有 32x32x3 的像素,我要怎么让模型吐出 10 个类别概率?"

晓寒当年看 CS231n 的时候,最爽的一点是:它不直接甩你一个现成 CNN,而是一步步从 kNN → 线性分类器 → 两层神经网络 → CNN,把每一层抽象为什么有效讲得明明白白。

这篇按 notebook `26_cs231n_cnn_fundamentals.ipynb` 的纯 NumPy 实现,给你一条完整的“像素到预测”路线图:

1) 数据集(合成 CIFAR-10)
2) kNN:最近邻 baseline
3) 线性分类器:Softmax / SVM
4) 优化:SGD/Momentum/Adam + 学习率计划
5) 两层全连接网络:加入非线性
6) CNN:卷积/池化/层叠
7) 训练过程的 babysitting(最实用的工程技巧)

注意:这份教程主打“理解脉络 + 关键代码”,完整可运行代码与可视化在 `26_cs231n_cnn_fundamentals.ipynb`。

## 0. 你需要的那条主线

图像分类管线一句话:

**Data → Model → Loss → Optimizer → Update → Repeat**

你只要把这条线跑通,后面的 ResNet/ViT 也只是 Model 变复杂而已。

## 1) 数据集:先用合成 CIFAR-10 把实验跑起来

notebook 用合成图像(螺旋/棋盘/渐变/圆形等模式)模拟 CIFAR-10 的 10 类,好处是:

- 不用下载数据
- 模式清晰,便于观察模型学到了什么

关键代码(示意):

```python
def generate_synthetic_cifar(num_samples=1000, img_size=32, num_classes=10):
    X = np.zeros((num_samples, img_size, img_size, 3))
    y = np.random.randint(0, num_classes, num_samples)
    # ...按类别画出不同模式...
    return X, y


X_train, y_train = generate_synthetic_cifar(num_samples=2000)
X_val, y_val = generate_synthetic_cifar(num_samples=400)
X_test, y_test = generate_synthetic_cifar(num_samples=400)

# 传统分类器通常先 flatten
X_train_flat = X_train.reshape(len(X_train), -1)  # (N, 3072)
```

## 2) kNN:不训练的 baseline

kNN 的逻辑非常朴素:

- 对每个 test 样本,找训练集中最近的 k 个
- 投票

```python
class KNearestNeighbor:
    def __init__(self, k=5):
        self.k = k

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            dist = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))
            knn = np.argsort(dist)[:self.k]
            y_pred[i] = np.argmax(np.bincount(self.y_train[knn]))
        return y_pred
```

它的价值是:给你一个最低基线,同时让你体会“非参数方法 test 时很慢”。

## 3) 线性分类器:Softmax / SVM

线性分类器的形式:

\[
f(x) = Wx + b
\]

它像是在学“每一类一个模板”。

notebook 里实现了 SVM hinge loss 和 softmax cross-entropy,并手推了梯度,用 SGD 训练。

你需要记住的点:

- softmax 给你概率视角
- SVM 给你 margin 视角
- 都是凸的(对 W 来说),训练稳定

## 4) 优化:SGD/Momentum/Adam 与学习率计划

很多时候模型不行,其实是优化不行。

CS231n 的经典经验是:

- 最重要的超参是 learning rate
- momentum 让下降更平滑、更快
- Adam 常常开箱即用
- 学习率 schedule 能带来更好的收敛

notebook 给了一个最小实现:

```python
class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, param, grad, param_id='p'):
        # ...一阶/二阶矩估计 + bias correction...
        return param - self.lr * grad  # 这里省略细节
```

## 5) 两层神经网络:一旦有非线性,世界就变了

线性模型只能画直线/超平面。

加一层 + ReLU/sigmoid,就能拟合复杂决策边界。

notebook 实现了一个 TwoLayerNet,包含:

- forward cache
- softmax loss
- 手写 backprop

这部分是理解深度学习“为什么能学”的关键。

## 6) CNN:卷积层在干嘛?

卷积层的两个核心归纳偏置:

1) **局部连接**:图像的局部像素相关
2) **参数共享**:同一个滤波器在全图滑动

这让 CNN 在参数量远小于全连接的情况下,仍然能学到强特征。

notebook 用最朴素的循环写了 conv forward 和 maxpool forward:

```python
def conv2d_forward(X, W, b, stride=1, pad=0):
    # X: (N, C_in, H, W)
    # W: (C_out, C_in, K, K)
    # ...padding + sliding window...
    return out, cache


def maxpool2d_forward(X, pool_size=2, stride=2):
    # ...每个窗口取 max...
    return out, cache
```

然后堆一个 toy Mini-AlexNet:

Conv → ReLU → Pool → Conv → ReLU → Pool → FC → ReLU → FC

它把“空间结构”一路抽象成更高层语义。

## 7) Babysitting:训练神经网络最实用的 checklist

CS231n 最出圈的工程经验之一就是 babysitting:

1) 初始 loss 是否合理(10 类 softmax 初始大约是 -log(0.1)=2.303)
2) 能不能 overfit 50 个样本(能的话说明实现没大 bug)
3) 监控 train/val gap(过拟合信号)
4) 学习率是不是太大/太小(不下降先怀疑 lr)
5) 可视化权重/滤波器/显著图(saliency)

notebook 里给了完整的诊断图代码,建议你直接跑一遍。

## 小结

这篇的核心不是某个模型,而是你脑子里要形成“进化路线”:

- kNN:不训练 baseline
- 线性:学模板
- 两层 NN:学非线性特征
- CNN:利用空间结构,参数更高效

当你能用 NumPy 把这些都写出来,后面你用 PyTorch/JAX 只是换工具,不是换理解。

## 练习题

1) 把 kNN 的距离从 L2 改成 cosine,在合成数据上差异是什么?

2) 给 TwoLayerNet 加 dropout 或 L2,观察 train/val gap 变化。

3) 给 CNN 加第三个 conv block,再看参数量与效果(如果你实现了训练)如何变化。

4) (进阶) 把 conv forward 从四层循环优化成 im2col/矩阵乘,体会工程优化的威力。

## 延伸阅读

1) Stanford CS231n 课程讲义

2) AlexNet / VGG / ResNet(Paper 10/15)

3) Vision Transformer(Paper 13 相关思想扩展到视觉)

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 26 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
