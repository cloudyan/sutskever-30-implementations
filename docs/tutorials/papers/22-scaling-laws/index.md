# Scaling Laws 是怎么“预测大模型越大越强”的?

问下大家,你有没有遇到过这种老板式问题:

"如果我把参数翻 10 倍,loss 能降多少?"
"如果我多喂 10 倍数据,收益还大吗?"
"同样算力预算,我是做更大的模型,还是刷更多的数据?"

晓寒以前觉得这些问题只能靠玄学+调参,直到看了 Scaling Laws(尤其 Kaplan/Chinchilla 一脉)才发现卧槽:

**很多指标在对数坐标下近似一条直线。**

也就是说:loss/perplexity 和 参数量 N、数据量 D、算力 C 之间,在很大范围内遵循幂律(power law)。

这篇按 notebook `22_scaling_laws.ipynb` 的实现,用一个 toy 实验把三件事讲清楚:

1) Loss vs 参数量 N 的幂律
2) Loss vs 数据量 D 的幂律
3) 同算力下的 compute-optimal(为什么 Chinchilla 说要“模型和数据一起变大”)

## 1) 幂律长什么样?

常见写法之一:

\[
L(x) = a\,x^{-b} + c
\]

或者更简化:

\[
L(x) \approx a\,x^{-b}
\]

把它放到 log-log 坐标下,就会变成近似直线:

\[
\log L \approx \log a - b\log x
\]

这也是 scaling law 最“工程化”的地方:你可以用小模型的数据拟合 b,再外推大模型。

## 2) 用 toy 模型模拟“训练损失随规模下降”

为了让代码可跑,notebook 用了一个非常简化的语言模型模拟器:

- 参数越多(capacity 越大) → loss 越低
- 数据越多 → loss 越低
- 训练步数越多 → 更接近收敛

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(42)


def scaling_law_params(x, a, b):
    # L = a * x^{-b}
    return a * np.power(x, -b)


class SimpleLanguageModel:
    def __init__(self, num_params, vocab_size=100):
        self.num_params = num_params
        self.vocab_size = vocab_size
        self.capacity = np.log(num_params) / 10.0

    def train(self, dataset_size, num_steps):
        base_loss = np.log(self.vocab_size)

        param_factor = 1.0 / (1.0 + self.capacity)
        data_factor = 1.0 / (1.0 + np.log(dataset_size) / 15.0)
        train_factor = np.exp(-num_steps / 1000.0)

        loss = base_loss * param_factor * data_factor * (0.5 + 0.5 * train_factor)
        loss += np.random.randn() * 0.05
        return max(loss, 1.0)
```

这个模型不是为了“真实”,而是为了让你在 1 分钟内把 scaling 的形状跑出来。

## 3) 实验 1:参数量 N 越大,loss 怎么变?

```python
dataset_size = 100_000
num_steps = 1000

param_counts = np.array([1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7])
losses = []

for N in param_counts:
    model = SimpleLanguageModel(num_params=int(N))
    losses.append(model.train(dataset_size, num_steps))

losses = np.array(losses)
fit, _ = curve_fit(scaling_law_params, param_counts, losses)
a, b = fit

plt.figure(figsize=(9, 5))
plt.loglog(param_counts, losses, 'o', label='measured')
plt.loglog(param_counts, scaling_law_params(param_counts, *fit), '--',
           label=f'fit: L ∝ N^{-b:.3f}')
plt.xlabel('params N')
plt.ylabel('loss L')
plt.grid(True, alpha=0.3, which='both')
plt.legend()
plt.show()
```

你会看到一条很“直”的曲线。

直觉类比:

这像是开“无限连锁店”:店越多(参数越多),覆盖能力越强,损失越低,但边际收益递减(因为幂律指数 b 通常不大)。

## 4) 实验 2:数据量 D 越大,loss 怎么变?

同样的方法,固定模型规模,只换数据量:

```python
num_params = 1_000_000
num_steps = 1000

dataset_sizes = np.array([1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7])
losses = []

for D in dataset_sizes:
    model = SimpleLanguageModel(num_params=int(num_params))
    losses.append(model.train(int(D), num_steps))

losses = np.array(losses)
fit, _ = curve_fit(scaling_law_params, dataset_sizes, losses)
a, b = fit

plt.figure(figsize=(9, 5))
plt.loglog(dataset_sizes, losses, 's', label='measured')
plt.loglog(dataset_sizes, scaling_law_params(dataset_sizes, *fit), '--',
           label=f'fit: L ∝ D^{-b:.3f}')
plt.xlabel('data D')
plt.ylabel('loss L')
plt.grid(True, alpha=0.3, which='both')
plt.legend()
plt.show()
```

你会得到另一个幂律指数(通常与参数幂律不同)。

## 5) 实验 3:同样算力预算,怎么分配 N 和 D 才划算?

一个非常工程化的问题:

"我只有 C 这么多算力,我该训练大模型少数据,还是小模型多数据?"

Chinchilla 的核心洞察可以粗略理解成:

**compute-optimal 的区域里,模型规模 N 和数据量 D 都应该随算力一起增长。**

notebook 用一个简化约束:

\[
C \approx 6ND
\]

并用 N≈D≈sqrt(C/6) 做平衡分配。

```python
compute_budgets = np.array([1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9])
results = []

for C in compute_budgets:
    N_opt = int(np.sqrt(C / 6))
    D_opt = int(np.sqrt(C / 6))
    model = SimpleLanguageModel(num_params=N_opt)
    loss = model.train(D_opt, num_steps=1000)
    results.append((C, N_opt, D_opt, loss))
```

你再做一个对比会更直观:

- 大模型+小数据
- 小模型+大数据
- 平衡(Chinchilla)

通常平衡策略最优。

生活类比:

你要开餐厅(模型)还要买菜(数据)。

只扩店不买菜,菜不够,做不出好菜;

只买菜不扩店,厨房太小,也做不出来。

## 6) 外推:用小模型曲线预测更大的模型

一旦你拟合出 b,就能在 log-log 直线上外推。

notebook 里演示了从 1e3 参数一路外推到 1e12 参数。

工程上这意味着:

- 你不必训练到天文级模型才知道趋势
- 但外推要小心分布变化/数据质量/优化策略等因素

## 小结

Scaling Laws 的价值可以总结成三句:

1) **loss 和规模常见幂律关系**:log-log 近似直线,可拟合可外推
2) **边际收益递减但可预测**:指数 b 通常不大,但趋势稳定
3) **算力最优不是“只堆参数”**:模型和数据要一起变大(Chinchilla)

## 练习题

1) 把 toy 模型里的 `capacity = log(num_params)/10` 改成别的形式,拟合出的 b 会怎么变?

2) 给 loss 加一个不可约项 c(比如数据噪声下限),再用 y=a*x^{-b}+c 拟合,看拟合是否更稳定。

3) 思考题:为什么现实里“数据质量”有时比“数据量”更重要?这在 scaling law 里怎么体现?

## 延伸阅读

1) Kaplan et al., 2020, "Scaling Laws for Neural Language Models"

2) Hoffmann et al., 2022, "Training Compute-Optimal Large Language Models" (Chinchilla)

3) 现代 LLM 训练工程:数据配比、去重、质量过滤与 scaling 的关系

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 22 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
