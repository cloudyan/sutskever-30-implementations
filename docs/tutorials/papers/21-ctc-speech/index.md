# CTC 是怎么在“没有对齐标注”的情况下训练语音识别的?

问下大家,你有没有想过语音识别最难的一件事是什么?

不是把声音变成字母概率(模型可以学),而是这个更恶心的问题:

**对齐(alignment)到底在哪?**

一句话说清楚:

- 音频是 T 帧连续信号
- 文本是 U 个离散字符
- 训练时我们常常只有整句文本,并没有告诉你“第 t 帧对应哪个字符”

晓寒以前以为必须做强制对齐,后来看到 CTC(Connectionist Temporal Classification) 才发现卧槽:

**可以把所有可能的对齐路径都算一遍,把概率加起来。**

这篇按 notebook `21_ctc_speech.ipynb` 的实现,用纯 NumPy 写一个简化版 CTC forward algorithm + greedy decoding,把核心直觉讲透。

## 1. CTC 先解决“长度不一致”

CTC 引入一个特殊符号:blank(空白),我们用 ε 表示。

它允许模型在很多帧都输出 ε,表示“这一帧没有新字符”。

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

vocab = list('abcdefghijklmnopqrstuvwxyz ') + ['ε']
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
blank_idx = len(vocab) - 1
```

## 2. CTC 的折叠规则(collapse)

一条“帧级路径”会很长,CTC 用两条规则把它折叠成最终输出:

1) 去掉 blank
2) 合并连续重复字符

```python
def collapse_ctc(sequence, blank_idx):
    # 1) 去掉 blank
    no_blanks = [s for s in sequence if s != blank_idx]
    if len(no_blanks) == 0:
        return []

    # 2) 合并连续重复
    collapsed = [no_blanks[0]]
    for s in no_blanks[1:]:
        if s != collapsed[-1]:
            collapsed.append(s)
    return collapsed
```

比如路径:

`h ε e l l o` 折叠后就是 `hello`

这就像做字幕对齐:你可以在很多帧“沉默”(ε),也允许同一个字母拖长多帧,最后再统一合并。

## 3. 造一个玩具“声学特征”(模拟 MFCC)

真实语音会做特征提取,这里用随机向量模拟每个字符对应的一段连续帧。

```python
def generate_audio_features(text, frames_per_char=3, feature_dim=20):
    char_indices = [char_to_idx[c] for c in text]
    features = []

    for char_idx in char_indices:
        base = np.random.randn(feature_dim) + char_idx * 0.1
        num_frames = np.random.randint(frames_per_char - 1, frames_per_char + 2)
        for _ in range(num_frames):
            features.append(base + np.random.randn(feature_dim) * 0.3)

    return np.array(features)
```

## 4. 一个最小的“声学模型”:RNN 输出每帧字符分布

模型输入: (T, feature_dim)

模型输出: (T, vocab_size) 的 log_probs(每帧对每个字符的对数概率)

```python
class AcousticModel:
    def __init__(self, feature_dim, hidden_size, vocab_size):
        self.hidden_size = hidden_size

        self.W_xh = np.random.randn(hidden_size, feature_dim) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))

        self.W_out = np.random.randn(vocab_size, hidden_size) * 0.01
        self.b_out = np.zeros((vocab_size, 1))

    def forward(self, features):
        h = np.zeros((self.hidden_size, 1))
        outputs = []

        for t in range(len(features)):
            x = features[t:t+1].T
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)

            logits = self.W_out @ h + self.b_out
            # 简化 log-softmax
            log_probs = logits - np.log(np.sum(np.exp(logits)))
            outputs.append(log_probs.flatten())

        return np.array(outputs)
```

## 5. CTC 的 forward algorithm(动态规划)

核心思想:

**把 target 插入 blank 变成 extended_target,然后用 DP 累加所有合法路径的概率。**

比如 target="hi":

extended 变成 `ε h ε i ε`

DP 状态 alpha[t, s] 表示:时间 t 走到 extended_target 的位置 s 的概率(这里用 log 版本)。

```python
def ctc_loss_naive(log_probs, target, blank_idx):
    T = len(log_probs)
    U = len(target)

    extended = [blank_idx]
    for ch in target:
        extended.extend([ch, blank_idx])
    S = len(extended)

    log_alpha = np.ones((T, S)) * -np.inf

    # init
    log_alpha[0, 0] = log_probs[0, extended[0]]
    if S > 1:
        log_alpha[0, 1] = log_probs[0, extended[1]]

    # forward
    for t in range(1, T):
        for s in range(S):
            label = extended[s]
            candidates = [log_alpha[t-1, s]]
            if s > 0:
                candidates.append(log_alpha[t-1, s-1])
            if s > 1 and label != blank_idx and extended[s-2] != label:
                candidates.append(log_alpha[t-1, s-2])

            log_alpha[t, s] = np.logaddexp.reduce(candidates) + log_probs[t, label]

    # end: 最后两个位置求和
    log_p = np.logaddexp(
        log_alpha[T-1, S-1],
        log_alpha[T-1, S-2] if S > 1 else -np.inf,
    )
    return -log_p, log_alpha
```

如果你把 alpha 画出来,会看到 CTC 在时间-状态网格上“铺开”地探索各种对齐路径。

## 6. 解码:最简单的 greedy decoding

真实系统会用 beam search + 语言模型,这里我们只做 greedy:

1) 每帧选 argmax
2) 用 collapse 规则折叠

```python
def greedy_decode(log_probs, blank_idx):
    pred = np.argmax(log_probs, axis=1)
    decoded = collapse_ctc(pred.tolist(), blank_idx)
    return decoded, pred
```

## 小结

CTC 最重要的三件事:

1) **blank ε**:允许“这一帧没有新字符”
2) **collapse 规则**:去 blank + 合并重复
3) **forward DP**:把所有合法对齐路径的概率加起来,无需对齐标注

你后面再看 Deep Speech 2 这种端到端语音识别框架,就会发现 CTC 是把“对齐难题”绕开的一把钥匙。

## 练习题

1) 把 target 从 "hi" 改成 "hello",观察 extended_target 的长度和 alpha 网格的形状变化。

2) 实现一个更稳定的 log-softmax(按行减 max),避免数值溢出。

3) 写一个 beam search 解码(只要 top-k 即可),对比 greedy 的效果。

4) 思考题:CTC 为什么需要“跳过 s-2”的转移条件(extended[s-2] != label)?它在避免什么问题?

## 延伸阅读

1) Graves et al., 2006, "Connectionist Temporal Classification"

2) Deep Speech 2:Amodei et al., 2016, "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin"

3) CTC vs Attention-based ASR:两种对齐思路的对比

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 21 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
