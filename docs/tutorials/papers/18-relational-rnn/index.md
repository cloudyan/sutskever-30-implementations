# Relational RNN 是怎么把“记忆”变成可推理的关系网络的?

问下大家,你有没有这种感觉:

LSTM 明明能记住东西,但一旦任务需要“比较”和“推理”(比如排序、找最大、对齐、跨步引用),它就像一个只会背书的人,背得很辛苦,推理却不够利索。

晓寒当年第一次做算法题风格的序列任务时也踩过坑:模型能记住片段,但要它在脑子里做“元素之间的关系运算”,总是差一口气。

直到我看到 **Relational RNN / Relational Memory** 这条思路,才发现卧槽,原来可以给 RNN 加一个“可交互的记忆矩阵”,让记忆槽(slot)之间通过 self-attention 互相“聊天”,从而把关系推理内置进循环结构里。

这篇我们按 notebook `18_relational_rnn.ipynb` 的实现,用纯 NumPy 搭一个最小可运行版本:

- 多头注意力(Multi-Head Attention)
- Relational Memory Core(多槽记忆 + 注意力交互 + 门控更新)
- Relational RNN Cell(LSTM proposal + 记忆更新 + 融合输出)
- 一个 toy 的“排序任务”做前向验证

注意:为了让教程长度可控,本文把“手写 1000+ 行反向传播训练”(notebook Section 11)作为扩展阅读,核心教程聚焦于**架构与数据流**。

## 先把直觉讲清楚:什么叫“可推理的记忆”? 

普通 RNN/LSTM 的记忆是一个向量 h/c,更像“一个人脑子里的一团综合印象”。

Relational Memory 则像“一个会议室里有很多座位(记忆槽)”:

- 每个槽存一类信息
- 槽与槽之间可以互相注意(谁更相关就多看两眼)
- 看完以后再用门控(类似 LSTM 的 gate)决定哪些写入、哪些遗忘

这样做的好处是:当任务需要“比较 A 和 B”“把 A 的信息搬到 B 的推理里”,模型可以通过 attention 在记忆槽之间做信息路由。

## 代码实现:Multi-Head Attention

我们先实现一个最小的多头自注意力,用于让记忆槽之间交互。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax, log_softmax


def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """
    Multi-head attention

    X: (N, d_model) 这里 N 通常是 mem_slots + 1(把当前输入也拼进去)
    每个 head 的投影: W_q[h], W_k[h], W_v[h] 形状 (d_model, d_k)
    W_o: (d_model, d_model)
    """
    N, d_model = X.shape
    d_k = d_model // num_heads

    heads = []
    for h in range(num_heads):
        Q = X @ W_q[h]                  # (N, d_k)
        K = X @ W_k[h]                  # (N, d_k)
        V = X @ W_v[h]                  # (N, d_k)

        scores = Q @ K.T / np.sqrt(d_k) # (N, N)
        if mask is not None:
            scores = scores + mask

        attn = softmax(scores, axis=-1)
        heads.append(attn @ V)          # (N, d_k)

    concat = np.concatenate(heads, axis=-1)  # (N, d_model)
    out = concat @ W_o                       # (N, d_model)
    return out, None
```

你可以把它当成“记忆槽之间做信息交换”的通信层。

## 代码实现:Relational Memory Core

Relational Memory 的关键流程是:

1) 把当前记忆槽 memory 和当前输入 input_vec 拼起来
2) 对拼接后的矩阵做 self-attention
3) residual + row-wise MLP
4) 用门控把结果写回每个记忆槽

```python
class RelationalMemory:
    """Relational Memory Core: 多槽记忆 + self-attention + 门控更新"""

    def __init__(self, mem_slots, head_size, num_heads=4):
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.d_model = head_size * num_heads

        self.W_q = [np.random.randn(self.d_model, head_size) * 0.1 for _ in range(num_heads)]
        self.W_k = [np.random.randn(self.d_model, head_size) * 0.1 for _ in range(num_heads)]
        self.W_v = [np.random.randn(self.d_model, head_size) * 0.1 for _ in range(num_heads)]
        self.W_o = np.random.randn(self.d_model, self.d_model) * 0.1

        # row-wise MLP
        self.W_mlp1 = np.random.randn(self.d_model, self.d_model * 2) * 0.1
        self.W_mlp2 = np.random.randn(self.d_model * 2, self.d_model) * 0.1

        # 门控(类 LSTM)
        self.W_gate_i = np.random.randn(self.d_model, self.d_model) * 0.1
        self.W_gate_f = np.random.randn(self.d_model, self.d_model) * 0.1
        self.W_gate_o = np.random.randn(self.d_model, self.d_model) * 0.1

        self.memory = np.random.randn(mem_slots, self.d_model) * 0.01

    def reset_state(self):
        self.memory = np.random.randn(self.mem_slots, self.d_model) * 0.01

    def step(self, input_vec):
        """input_vec: (d_model,)"""
        M_tilde = np.concatenate([self.memory, input_vec[None]], axis=0)  # (mem_slots+1, d_model)

        attended, _ = multi_head_attention(M_tilde, self.W_q, self.W_k, self.W_v, self.W_o, self.num_heads)
        gated = attended + M_tilde  # residual

        hidden = np.maximum(0, gated @ self.W_mlp1)
        mlp_out = hidden @ self.W_mlp2

        new_memory = []
        for i in range(self.mem_slots):
            m = mlp_out[i]
            i_gate = 1 / (1 + np.exp(-(m @ self.W_gate_i)))
            f_gate = 1 / (1 + np.exp(-(m @ self.W_gate_f)))
            o_gate = 1 / (1 + np.exp(-(m @ self.W_gate_o)))

            candidate = np.tanh(m)
            new_slot = f_gate * self.memory[i] + i_gate * candidate
            new_memory.append(o_gate * np.tanh(new_slot))

        self.memory = np.array(new_memory)
        return mlp_out[-1]  # 最后一行对应拼接进去的 input
```

这段代码可以当成本文的“灵魂”:**让记忆槽通过 attention 交互,再用门控写回。**

## 代码实现:Relational RNN Cell(LSTM + Relational Memory)

Relational RNN cell 的结构可以理解成:

- LSTM 先处理输入,给出一个 proposal hidden state
- proposal 去更新 relational memory
- 然后把两者融合成最终输出 hidden state

```python
class RelationalRNNCell:
    def __init__(self, input_size, hidden_size, mem_slots=4, num_heads=4):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = np.random.randn(input_size + hidden_size, 4 * hidden_size) * 0.1
        self.lstm_bias = np.zeros(4 * hidden_size)

        self.rm = RelationalMemory(
            mem_slots=mem_slots,
            head_size=hidden_size // num_heads,
            num_heads=num_heads,
        )

        self.W_combine = np.random.randn(2 * hidden_size, hidden_size) * 0.1
        self.b_combine = np.zeros(hidden_size)

        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)

    def reset_state(self):
        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)
        self.rm.reset_state()

    def forward(self, x):
        concat = np.concatenate([x, self.h])
        gates = concat @ self.lstm + self.lstm_bias
        i, f, o, g = np.split(gates, 4)

        i = 1 / (1 + np.exp(-i))
        f = 1 / (1 + np.exp(-f))
        o = 1 / (1 + np.exp(-o))
        g = np.tanh(g)

        self.c = f * self.c + i * g
        h_proposal = o * np.tanh(self.c)

        rm_out = self.rm.step(h_proposal)

        combined = np.concatenate([h_proposal, rm_out])
        self.h = np.tanh(combined @ self.W_combine + self.b_combine)
        return self.h
```

## 用一个 toy 任务做前向验证:序列排序

我们用一个很适合“记忆+比较”的任务:

输入:一串数字(用 one-hot 表示)

输出:把这串数字排序后的序列(同样用 one-hot)

这任务的难点不是识别数字,而是“比较大小并重排”。

```python
def generate_sorting_task(seq_len=10, max_digit=20, batch_size=64):
    x = np.random.randint(0, max_digit, size=(batch_size, seq_len))
    y = np.sort(x, axis=1)
    X = np.eye(max_digit)[x]
    Y = np.eye(max_digit)[y]
    return X.astype(np.float32), Y.astype(np.float32)
```

为了保持实现简单,notebook 里的 verification 是“前向验证”(不做参数更新):

- 每次生成一个新序列
- reset 状态
- 跑一遍 sequence
- 用一个随机 readout W_out 算交叉熵,确认 loss 能正常计算

```python
def run_model_verification(model, epochs=30, seq_len=10):
    max_digit = 30
    losses = []
    W_out = np.random.randn(model.hidden_size, max_digit) * 0.1

    for _ in range(epochs):
        X, Y = generate_sorting_task(seq_len, max_digit, batch_size=1)
        if isinstance(model, RelationalRNNCell):
            model.reset_state()
        else:
            model.reset()

        total = 0.0
        for t in range(seq_len):
            x_t = X[0, t]
            y_t = Y[0, t]

            h = model.forward(x_t) if isinstance(model, RelationalRNNCell) else model.step(x_t)
            logits = h @ W_out
            logp = log_softmax(logits)
            total += -np.sum(y_t * logp)

        losses.append(total / seq_len)

    return losses
```

同样提供一个 LSTM baseline 做对照:

```python
class LSTMBaseline:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.wx = np.random.randn(input_size, 4 * hidden_size) * 0.1
        self.wh = np.random.randn(hidden_size, 4 * hidden_size) * 0.1
        self.b = np.zeros(4 * hidden_size)
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)

    def reset(self):
        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)

    def step(self, x):
        gates = x @ self.wx + self.h @ self.wh + self.b
        i, f, o, g = np.split(gates, 4)
        i = 1 / (1 + np.exp(-i))
        f = 1 / (1 + np.exp(-f))
        o = 1 / (1 + np.exp(-o))
        g = np.tanh(g)
        self.c = f * self.c + i * g
        self.h = o * np.tanh(self.c)
        return self.h
```

## 关于“完整训练”:为什么会出现 1000+ 行手写反传?

Relational RNN 这种结构包含:

- LSTM 门控
- 多头注意力(softmax)
- 记忆槽的门控更新
- 融合层

如果只用 NumPy,想“完整训练”,就必须把每一步的梯度都手写出来。

notebook `18_relational_rnn.ipynb` 的 Section 11 就提供了完整的手写反向传播(约 1000+ 行),包含:

- 前向缓存(cache)
- 逐层反传
- 梯度检查(数值验证)
- 训练循环

如果你想直接跑完整版本,建议:

```bash
jupyter notebook 18_relational_rnn.ipynb
```

## 小结

Relational RNN 这篇你只要抓住一个主线就够了:

**把“记忆”从一个向量变成多个可交互的槽,让槽之间用 self-attention 做关系推理。**

当你后面学到:

- Transformer 的多头注意力
- Memory-augmented networks
- 甚至一些更现代的 RNN/State Space 模型

你会更容易看懂它们在“信息路由”和“结构归纳偏置(inductive bias)”上做了什么。

## 练习题

1) 把 mem_slots 从 2/4/8 改一改,观察 memory 形状和数值范围有什么变化。

2) 做一个 ablation:去掉门控(直接写回),看看前向 loss 分布是否更不稳定。

3) 把排序任务换成 "复制-延迟"(copy/delay) 任务,看这个结构是否更容易保持长期信息。

4) (进阶) 阅读 notebook Section 11,尝试只对 multi-head attention 那部分做数值梯度检查。

## 延伸阅读

1) Santoro et al., 2018, "Relational recurrent neural networks" (NeurIPS)

2) Relation Network(Paper 16):两两关系显式建模

3) Transformer(Paper 13):self-attention 的更通用形式

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 18 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
