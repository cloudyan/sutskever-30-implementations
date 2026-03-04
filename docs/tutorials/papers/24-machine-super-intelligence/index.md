# 机器超级智能是怎么被“形式化”出来的?

问下大家,你有没有被这类问题困扰过:

"智能到底是什么?"
"我怎么比较两个 agent 谁更聪明?"
"有没有一种不依赖具体任务的通用衡量?"

晓寒以前看到“超级智能”四个字,脑子里全是科幻画面。直到我接触到 Legg & Hutter 那套 **Universal Intelligence(通用智能)** 形式化思路,才发现卧槽:

原来可以把“智能”写成一个数学对象,并且和 **Solomonoff induction(所罗门诺夫归纳)**、**AIXI** 这种“理论最优智能体”连在一起。

这篇按 notebook `24_machine_super_intelligence.ipynb` 的实现,做一份小林图解风格的“可跑的理论导读”:

1) 从心理测量(g-factor)到通用智能 Υ(π)
2) 用一个玩具版本近似 Solomonoff 先验
3) 用 toy GridWorld + MCTS 做一个 MC-AIXI 的影子
4) 讨论递归自我改进与“智能爆炸”的动力学直觉

注意:这篇偏理论,我们用 toy 代码做直觉演示,重点是把概念串起来。

## 1) 从 g-factor 到“通用智能”

心理测量学里有个经典现象:正相关流形(positive manifold)。

通俗说就是:

一个人在数学、记忆、空间想象、语言等测试上,往往会一起高或一起低。

把相关矩阵做 PCA,第一主成分经常被解释为 g-factor。

notebook 用合成数据演示了这一点:

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import spearmanr


def generate_cognitive_test_data(n_subjects=200, n_tests=8, g_variance=0.7):
    # 一个简单的生成模型:每个人有一个 g,每个测试还有独立噪声
    g = np.random.randn(n_subjects, 1)
    loadings = np.random.uniform(0.5, 1.0, size=(1, n_tests))
    noise = np.random.randn(n_subjects, n_tests)

    scores = np.sqrt(g_variance) * g @ loadings + np.sqrt(1 - g_variance) * noise
    return scores


def extract_g_factor(scores):
    # 用 PCA 的第一主成分做 g 的 toy 近似
    X = scores - scores.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / (len(X) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    w = eigvecs[:, -1]
    g_hat = X @ w
    return g_hat
```

这一步的意义是:它提醒我们“智能可以被当成某种跨任务的能力”。

但 g 仍然依赖“人类出的题”,它不是通用形式化。

## 2) 通用智能 Υ(π):把 agent 放进所有环境里考

Legg & Hutter 的通用智能有个非常硬核的定义(口语化版):

**一个 agent 的智能,等于它在所有可计算环境上的期望回报,并对环境复杂度做加权。**

写成一个常见形式:

\[
\Upsilon(\pi) = \sum_{\mu \in \mathcal{E}} 2^{-K(\mu)} V^{\pi}_{\mu}
\]

- \(\mu\):环境
- \(V^{\pi}_{\mu}\):agent \(\pi\) 在环境 \(\mu\) 的期望回报
- \(K(\mu)\):环境的 Kolmogorov complexity(越简单权重越大)

直觉类比:

你不是只考“高考数学”,而是把 agent 丢进很多世界里(迷宫、博弈、控制、预测…),再按“世界的简单程度”加权打分。

真正难点在于:K(\mu) 不可计算,所以我们只能做近似。

## 3) Solomonoff induction 的玩具近似:枚举短程序

Solomonoff 的核心直觉是:

**更短的程序(更简单的解释)有更大的先验权重。**

notebook 里用“枚举短程序/有限状态机”的 toy 方式近似:

```python
class SimpleProgramEnumerator:
    """玩具版 Solomonoff:枚举短程序,按 2^{-len} 加权"""

    def __init__(self, max_program_len=6):
        self.max_program_len = max_program_len
        self.programs = []  # (program, weight)

        for L in range(1, max_program_len + 1):
            # program 用 0/1 序列表示(玩具)
            for bits in itertools.product([0, 1], repeat=L):
                weight = 2 ** (-L)
                self.programs.append((list(bits), weight))

    def predict_next(self, observed):
        """根据观测前缀,对下一位做加权投票"""
        w0 = 0.0
        w1 = 0.0
        for prog, w in self.programs:
            # 简化:用循环模式生成序列
            generated = [prog[i % len(prog)] for i in range(len(observed) + 1)]
            if generated[:len(observed)] == observed:
                nxt = generated[len(observed)]
                if nxt == 0:
                    w0 += w
                else:
                    w1 += w
        return 1 if w1 >= w0 else 0
```

这段代码的意义不是“它很强”,而是让你亲眼看到:

- 简单模式(短程序)会更快占优势
- 随着观测变长,预测会越来越稳定

这就是“奥卡姆剃刀 + 归纳”的数学化直觉。

## 4) MC-AIXI 的影子:用 MCTS 在环境里做规划

AIXI 本体非常理论,无法直接算。

一个常见的近似路线是:

- 用某种方式近似环境分布
- 用规划算法(如 MCTS)在有限 horizon 内做决策

notebook 用一个 toy GridWorld + MCTS 展示“MC-AIXI-light”的影子结构。

下面给一个简化版环境:

```python
class ToyGridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        return self.pos

    def step(self, action):
        # 0/1/2/3 = up/down/left/right
        x, y = self.pos
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        else:
            y = min(self.size - 1, y + 1)

        self.pos = (x, y)
        done = self.pos == self.goal
        reward = 1.0 if done else -0.01
        return self.pos, reward, done
```

你可以把 MCTS 想成:

"在脑子里模拟未来很多种走法,选长期回报最大的那条。"

## 5) 智能爆炸:递归自我改进的动力学直觉

最后 notebook 用一个 toy 自我改进 agent 演示:

- agent 可以提升自己的计算预算(更多模拟)
- 或提升规划视野(horizon)
- 于是表现变好,更容易获得资源,继续提升...

这类正反馈如果没有约束,就会出现“加速增长”的曲线。

当然,现实里会有很多瓶颈:

- 算力/能耗/数据
- 物理世界的限制
- 安全与目标对齐

但这个 toy 模拟至少能帮你建立一个直觉:为什么“递归自我改进”会被认为可能带来非线性风险。

## 小结

这篇你不需要记住所有公式,记住四个关键词就够了:

1) **通用智能 Υ(π)**:跨环境加权的期望回报
2) **复杂度权重 2^{-K}**:更简单的环境更重要(奥卡姆)
3) **Solomonoff/AIXI**:把“归纳+规划”推到理论极限的理想化对象
4) **递归自我改进**:可能导致加速增长,也带来对齐与安全问题

如果你把 Paper 23(MDL)、Paper 25(Kolmogorov complexity) 和这篇连起来看,会发现它们都在围绕同一个主题:

**压缩/复杂度/归纳,是智能与学习的底层骨架。**

## 练习题

1) 用 `SimpleProgramEnumerator` 测试更多序列:随机序列 vs 规则序列,预测准确率曲线差异是什么?

2) 给 `ToyGridWorld` 加“陷阱格子”(负奖励),看 MCTS budget 增加时,策略是否更稳健。

3) 思考题:为什么 Υ(π) 要对环境复杂度加权?如果不加权会发生什么?

4) 思考题:现实世界的“环境分布”应该怎么定义?它和“人类价值”如何关联?

## 延伸阅读

1) Legg & Hutter, 2007, "Universal Intelligence: A Definition of Machine Intelligence"

2) Hutter, "Universal Artificial Intelligence" (AIXI)

3) Solomonoff induction(算法概率)

4) 超级智能与对齐:智能增长、目标错配、可控性

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 24 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
