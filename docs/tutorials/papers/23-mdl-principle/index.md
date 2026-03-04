# MDL(最小描述长度)原则是怎么用“压缩”来做模型选择的?

问下大家,你有没有遇到过这种纠结:

"我这个模型到底该不该更复杂一点?"
"多加几层/多加几个参数,训练误差确实更低,但会不会过拟合?"

晓寒以前也经常靠经验拍脑袋,直到接触到 **MDL(Minimum Description Length)** 才发现卧槽:

**模型选择可以变成一个“压缩比赛”。**

你把“模型本身”和“用模型解释数据后的剩余误差”都编码成比特数,总比特数最小的那个模型,往往就是泛化最好的那个。

这篇按 notebook `23_mdl_principle.ipynb` 的实现,用几个可跑的例子把 MDL 的核心直觉讲透:

1) 信息论里的编码长度
2) 多项式回归:用 MDL 选阶数
3) 神经网络:用 MDL 选隐藏层宽度
4) 剪枝(pruning):MDL 解释“为什么能删 90% 权重”
5) 压缩=理解:与 Kolmogorov complexity 的关系

## MDL 的一句话公式

MDL 的经典写法:

\[
MDL(M) = L(M) + L(D\,|\,M)
\]

其中:

- L(M):描述模型要多少比特(模型越复杂,越大)
- L(D|M):在模型已知的前提下,描述数据还需要多少比特(拟合越好,越小)

生活类比:

你要把一份数据打包寄给朋友:

- 你可以选择带一个“解压程序”(模型)
- 再带一份“补丁/残差”(数据在模型下还没解释掉的部分)

最好的选择就是:解压程序+补丁加起来最小。

## 1) 信息论基础:概率越小,码长越长

Shannon 给了一个非常硬核的结论:

\[
L(x) = -\log_2 p(x)
\]

p 越小的事件,最优编码长度越长。

notebook 里还实现了一个简单的 universal code length(整数的通用编码长度近似),用来解释“编码模型复杂度”。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln

np.random.seed(42)


def universal_code_length(n):
    # 简化的 Elias gamma code 近似
    if n <= 0:
        return float('inf')
    log_n = np.log2(n + 1)
    return log_n + np.log2(log_n + 1) + 2.865


def probability_code_length(p):
    if p <= 0 or p > 1:
        return float('inf')
    return -np.log2(p)
```

## 2) 例子 1:多项式回归用 MDL 选阶数

如果你只看 RSS(残差平方和),阶数越高越好,最后会过拟合。

MDL 会把两部分都算进去:

- L_model:参数个数(阶数+1)带来的复杂度
- L_data:残差在假设高斯噪声下的编码长度

```python
def generate_polynomial_data(n_points=50, noise_std=0.5):
    X = np.linspace(-2, 2, n_points)
    y_true = X**3 - 2*X**2 + X + 1
    y = y_true + np.random.randn(n_points) * noise_std
    return X, y, y_true


def fit_polynomial(X, y, degree):
    coeffs = np.polyfit(X, y, degree)
    y_pred = np.polyval(coeffs, X)
    rss = np.sum((y - y_pred) ** 2)
    return coeffs, y_pred, rss


def mdl_polynomial(X, y, degree):
    N = len(X)
    n_params = degree + 1
    _, _, rss = fit_polynomial(X, y, degree)

    # L(model):参数数 * log2(N)/2 (Fisher 信息近似)
    L_model = n_params * np.log2(N) / 2

    # L(data|model):残差的编码长度(高斯误差近似)
    if rss < 1e-10:
        L_data = 0
    else:
        L_data = N / 2 * np.log2(rss / N + 1e-10)

    return L_model + L_data, L_model, L_data
```

跑一遍 1~9 阶,你会看到:

- 低阶:拟合差 → L_data 大
- 高阶:模型复杂 → L_model 大
- 中间某个阶数:两者平衡 → MDL 最小(往往接近真实复杂度)

这就是 Occam's Razor(奥卡姆剃刀)的“比特化版本”。

## 3) 例子 2:神经网络结构选择

notebook 用一个简单的 2D spiral 分类数据,测试不同 hidden_dim 的网络:

- hidden_dim 越大,参数越多,L_model 越大
- hidden_dim 太小,拟合不动,L_data 越大

MDL 依然会找到一个折中点。

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        scale = 0.1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * scale
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        h = sigmoid(X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return softmax(logits)

    def compute_loss(self, X, y):
        probs = self.forward(X)
        N = len(X)
        y_onehot = np.zeros((N, self.output_dim))
        y_onehot[np.arange(N), y] = 1
        return -np.sum(y_onehot * np.log(probs + 1e-10)) / N

    def count_parameters(self):
        return (self.input_dim * self.hidden_dim + self.hidden_dim +
                self.hidden_dim * self.output_dim + self.output_dim)
```

这里训练部分为了速度做了简化(多次随机初始化选最优),但 MDL 的结构权衡点依然能体现出来。

## 4) 例子 3:MDL 解释剪枝(Pruning)

Paper 5 我们已经看到过“删掉大量权重,精度不掉”。

MDL 给了一个很漂亮的解释:

- 删权重会显著降低 L(model)
- 只要精度(拟合)没坏太多,L(data|model)涨得不大
- 总 MDL 反而下降 → 更好的模型

notebook 做了 magnitude pruning,并对不同 sparsity 计算 MDL,会出现一个 MDL-optimal sparsity。

## 5) 压缩=理解:与 Kolmogorov complexity 的关系

Kolmogorov complexity K(x) 是“生成 x 的最短程序长度”。

它很美,但不可计算。

MDL 你可以把它理解成:

**在你允许的模型族里,找一个可计算的 K(x) 近似。**

随机串几乎不可压缩,结构串可压缩,这就是算法随机性(algo randomness)的直觉。

## 小结

MDL 的威力在于:它把一堆看似不同的工程技巧统一起来了。

1) 模型选择:选阶数/选结构/选超参
2) 正则化:偏向简单模型
3) 剪枝与压缩:减少描述长度
4) 泛化:压缩得好往往泛化好

一句话: **最好的模型,是能把数据“压到最短”的那个。**

## 练习题

1) 把多项式数据的噪声调大/调小,MDL 选出来的最优阶数会怎么变?

2) 在 L_model 里把惩罚变强/变弱(系数乘 2 或除 2),会更偏向简单还是更偏向拟合?

3) 对剪枝实验:用不同 pruning 策略(随机剪/按层剪/全局剪),MDL-optimal sparsity 是否不同?

4) 思考题:为什么“越能压缩的数据越有规律”?这句话在深度学习里对应什么现象?

## 延伸阅读

1) Rissanen, 1978/1989, MDL 相关工作

2) AIC/BIC 与 MDL 的关系(BIC 基本就是 MDL 的近亲)

3) Kolmogorov complexity(Paper 25):MDL 的理论背景

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 23 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
