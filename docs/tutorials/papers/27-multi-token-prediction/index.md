# Multi-token Prediction 是怎么让大模型“学得更快”的?

问下大家,你训练语言模型的时候,是不是默认在做这件事:

给定前缀 `x_1..x_t`,只预测下一个 token `x_{t+1}`。

这就是标准的 next-token prediction。

晓寒一开始也觉得这很自然,直到看到 Multi-token Prediction 的思路,才发现卧槽:

**同一个隐藏状态 h_t,其实可以顺便预测未来多个 token。**

这相当于在同样的数据和同样的前向计算里,给模型塞了更密集的监督信号,让训练更“划算”。

这篇按 notebook `27_multi_token_prediction.ipynb` 的 toy 实现,用一个最小 RNN 把核心直觉跑出来:

1) 单 token 预测 vs 多 token 预测的结构差异
2) 为什么多 token 会更 sample-efficient
3) 直觉上它和推理加速(例如 speculative decoding)的关系

## 1) 单 token 预测:每个位置只看一步

标准 RNN 在每个位置输出一个 softmax:

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SingleTokenRNN:
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.W_embed = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_xh = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros((hidden_dim, 1))

        self.W_out = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.b_out = np.zeros((vocab_size, 1))

    def forward(self, input_seq):
        h = np.zeros((self.hidden_dim, 1))
        preds = []

        for tok in input_seq:
            x = self.W_embed[tok].reshape(-1, 1)
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)

            logits = self.W_out @ h + self.b_out
            probs = softmax(logits.T)
            preds.append(probs.flatten())

        return preds
```

训练目标就是:

\[
\sum_t -\log p(x_{t+1}\,|\,x_{\le t})
\]

## 2) 多 token 预测:一个隐藏状态,多个输出头

Multi-token prediction 的最小改动就是:

**把输出层从 1 个 head 变成 K 个 head。**

在位置 t,同时预测:

- t+1
- t+2
- ...
- t+K

```python
class MultiTokenRNN:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_future_tokens=3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_future_tokens = num_future_tokens

        self.W_embed = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_xh = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros((hidden_dim, 1))

        # K 个输出头
        self.output_heads = []
        for _ in range(num_future_tokens):
            W_out = np.random.randn(vocab_size, hidden_dim) * 0.01
            b_out = np.zeros((vocab_size, 1))
            self.output_heads.append((W_out, b_out))

    def forward(self, input_seq):
        h = np.zeros((self.hidden_dim, 1))
        preds = []  # preds[t] 是一个 list,里面有 K 个未来分布

        for tok in input_seq:
            x = self.W_embed[tok].reshape(-1, 1)
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)

            position_preds = []
            for W_out, b_out in self.output_heads:
                logits = W_out @ h + b_out
                probs = softmax(logits.T)
                position_preds.append(probs.flatten())

            preds.append(position_preds)

        return preds
```

训练目标变成:

\[
\sum_t \sum_{j=1}^K -\log p(x_{t+j}\,|\,x_{\le t})
\]

直觉:

同一个 h_t,以前只收一个监督信号,现在收 K 个。

这很像“同一堂课,你不只做一道题,你做了 K 道”。

## 3) 玩具数据:等差序列(可预测的模式)

notebook 用等差序列造 synthetic text:

```python
def generate_synthetic_sequences(vocab_size=50, num_sequences=1000, seq_length=20):
    seqs = []
    for _ in range(num_sequences):
        start = np.random.randint(0, vocab_size // 2)
        step = np.random.randint(1, 3)
        seq = [(start + i * step) % vocab_size for i in range(seq_length)]
        seqs.append(seq)
    return seqs
```

这种数据的好处是:规律非常明确,你能更清楚看到“监督信号密度”对学习曲线的影响。

## 4) 为什么多 token 更 sample-efficient?

如果你只预测 t+1,那么每个位置只贡献 1 个 loss。

如果你预测 t+1..t+K,每个位置贡献 K 个 loss。

在数据量有限时,这相当于:

- 同样的序列长度
- 训练信号更密集
- 梯度(或更新方向)更稳定

notebook 用对比曲线展示了这一点(这里训练实现是 toy 版,核心是“loss 计算方式”的对比直觉)。

## 5) 和推理加速有什么关系?

Multi-token prediction 的另一个直觉收益是:

- 你不仅能更好预测下一个 token
- 还学会了对未来多个 token 的分布

这会自然联想到一些推理加速技巧(例如 speculative decoding 的思想):

"先快速提出一段候选,再用更强模型验证/修正"。

当然,真正的工程实现会复杂很多,但“多步预测”是一个很重要的组件。

## 小结

Multi-token prediction 你可以理解成:

**让同一个隐藏状态同时回答多道未来题,用更密的监督信号提高样本效率。**

当模型和数据规模越来越大时,任何“同样算力换来更多有效训练信号”的技巧,都会有很高的工程价值。

## 练习题

1) 把 num_future_tokens 从 3 改成 1/2/4/8,画出 loss 曲线随 K 的变化。

2) 把 synthetic 数据从等差序列换成更复杂的模式(例如周期+噪声),多 token 的优势是否仍然明显?

3) 思考题:K 取太大时,会不会把学习目标变得更难(远期更不可预测)?你会怎么加权(例如对远期 loss 降权)?

4) 思考题:在 transformer 里实现 multi-token prediction,输出头应该怎么设计?共享还是分离? 

## 延伸阅读

1) Multi-token prediction 相关论文与后续工作(训练效率/推理效率)

2) Speculative decoding / draft-verify 一类推理加速思路

3) Scaling laws(Paper 22):训练预算下的收益评估

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 27 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
