# 神经图灵机(NTM)是怎么“读写外部记忆”的?

问下大家,你有没有这种体验:

脑子(工作记忆)一次只能记住很少的东西,但你拿个小本本/备忘录,就能把一大堆信息写下来,想用的时候再翻出来。

晓寒刚开始做算法题风格任务(复制、排序、括号匹配)的时候,经常觉得 RNN/LSTM 像是在“硬背”,记不住就全崩。

直到后来看到 **Neural Turing Machine(NTM)**,才发现卧槽,原来可以给网络配一个“外置记忆矩阵”,并且用**可微分**的方式去读写。

一句话概括 NTM:

**控制器(controller)负责算“怎么读写”,记忆(memory)负责存“写进去的内容”。**

这篇我们按 notebook `20_neural_turing_machine.ipynb` 的实现,用纯 NumPy 做一个最小可运行 NTM 读写头,把地址机制讲清楚。

## NTM 的核心组件

NTM 通常包含:

1) 外部记忆矩阵 M:形状 (N, M)
2) 读头/写头(head):输出一个对 N 个槽的注意力权重 w
3) 地址机制(addressing):既能按内容找(像检索),也能按位置移动(像指针)
4) 写入机制(write):先 erase 再 add,保证可微分

生活类比:

- 记忆矩阵像一排抽屉(每个抽屉一张卡片)
- w 像你手在抽屉上的“指向强度”(不是硬选一个,而是软分布)
- 内容寻址像“按关键词找”
- 位置寻址像“把指针往左/往右挪一格”

## 1. 外部记忆矩阵:读和写

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


class Memory:
    def __init__(self, num_slots, slot_size):
        """
        num_slots: N, 记忆槽数量
        slot_size: M, 每个槽的向量维度
        """
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.memory = np.random.randn(num_slots, slot_size) * 0.01

    def read(self, weights):
        """weights: (N,) -> 返回 (M,)"""
        return np.dot(weights, self.memory)

    def write(self, weights, erase_vector, add_vector):
        """
        写入分两步:
        1) erase:按权重把某些维度抹掉
        2) add:按权重把新内容加进去
        """
        erase = np.outer(weights, erase_vector)  # (N, M)
        self.memory = self.memory * (1 - erase)

        add = np.outer(weights, add_vector)      # (N, M)
        self.memory = self.memory + add

    def get_memory(self):
        return self.memory.copy()
```

注意这里最关键的一点:整个过程都是连续可微的(没有 hard index)。

## 2. 内容寻址(Content-based addressing):像检索一样找相似

内容寻址的经典做法:

- 用 key 向量去和每个记忆槽做相似度(常用 cosine)
- 用 softmax 得到注意力分布
- beta 控制“尖锐程度”(beta 越大,越接近硬选择)

```python
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8)


def softmax(x, beta=1.0):
    x = beta * x
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def content_addressing(memory, key, beta):
    similarities = np.array([
        cosine_similarity(key, memory[i])
        for i in range(len(memory))
    ])
    return softmax(similarities, beta=beta)
```

这就像你拿着一个“关键词向量”,去一排卡片里找最像的那张。

## 3. 位置寻址(Location-based addressing):像指针一样移动

NTM 不仅能“按内容找”,还希望能“沿着位置走”。

典型管道是三步:

1) interpolation:把当前权重 w_prev 和内容权重 w_content 做插值(门 g)
2) shift:用一个小卷积核把权重分布往左/往右挪(允许轻微移动)
3) sharpen:用 gamma 把分布变尖(防止越走越糊)

```python
def interpolation(weights_content, weights_prev, g):
    return g * weights_content + (1 - g) * weights_prev


def convolutional_shift(weights, shift_weights):
    """shift_weights 对应 [-1, 0, +1] 的概率"""
    shifted = np.zeros_like(weights)
    for shift_idx, shift_amount in enumerate([-1, 0, 1]):
        shifted += shift_weights[shift_idx] * np.roll(weights, shift_amount)
    return shifted


def sharpening(weights, gamma):
    weights = weights ** gamma
    return weights / (np.sum(weights) + 1e-8)
```

直觉类比:

- g 决定“这次更相信内容检索还是更相信上一步指针位置”
- shift 让指针能平滑移动
- sharpen 防止注意力分布扩散

## 4. 把所有东西打包成一个 NTM Head(读/写头)

真实 NTM 里这些参数来自 controller(通常是 RNN/MLP)。

我们这里做一个简化版:给定 controller_output(一个向量),通过线性层生成 key/beta/g/shift/gamma/erase/add。

```python
class NTMHead:
    def __init__(self, memory_slots, memory_size, controller_size):
        self.memory_slots = memory_slots
        self.memory_size = memory_size

        self.W_key = np.random.randn(memory_size, controller_size) * 0.1
        self.W_beta = np.random.randn(1, controller_size) * 0.1
        self.W_g = np.random.randn(1, controller_size) * 0.1
        self.W_shift = np.random.randn(3, controller_size) * 0.1
        self.W_gamma = np.random.randn(1, controller_size) * 0.1

        self.W_erase = np.random.randn(memory_size, controller_size) * 0.1
        self.W_add = np.random.randn(memory_size, controller_size) * 0.1

        self.weights_prev = np.ones(memory_slots) / memory_slots

    def address(self, memory_matrix, controller_output):
        key = np.tanh(self.W_key @ controller_output)
        beta = np.exp((self.W_beta @ controller_output)[0]) + 1e-4
        w_content = content_addressing(memory_matrix, key, beta)

        g = (1 / (1 + np.exp(-(self.W_g @ controller_output))))[0]
        w = interpolation(w_content, self.weights_prev, g)

        shift_logits = self.W_shift @ controller_output
        shift_w = softmax(shift_logits)
        w = convolutional_shift(w, shift_w)

        gamma = np.exp((self.W_gamma @ controller_output)[0]) + 1.0
        w = sharpening(w, gamma)

        self.weights_prev = w
        return w

    def read(self, memory: Memory, weights):
        return memory.read(weights)

    def write(self, memory: Memory, weights, controller_output):
        erase = 1 / (1 + np.exp(-(self.W_erase @ controller_output)))
        add = np.tanh(self.W_add @ controller_output)
        memory.write(weights, erase, add)
```

## 5. 玩具任务:Copy Sequence(写入再读出)

完整训练需要反向传播,这里我们只做“写入过程”的可视化:

- 给一个 4 维 one-hot 序列
- 每步产生一个随机 controller 输出(只是演示)
- 用 head.address 得到写权重
- head.write 写入记忆
- 观察记忆矩阵随时间变化

```python
memory = Memory(num_slots=8, slot_size=4)
head = NTMHead(memory_slots=8, memory_size=4, controller_size=16)

sequence = [
    np.array([1, 0, 0, 0]),
    np.array([0, 1, 0, 0]),
    np.array([0, 0, 1, 0]),
    np.array([0, 0, 0, 1]),
]

states = [memory.get_memory()]
weights_hist = []

for _ in sequence:
    controller_out = np.random.randn(16)
    w = head.address(memory.memory, controller_out)
    weights_hist.append(w)
    head.write(memory, w, controller_out)
    states.append(memory.get_memory())

# 记忆矩阵变化
fig, axes = plt.subplots(1, len(states), figsize=(16, 3))
for i, ax in enumerate(axes):
    ax.imshow(states[i], cmap='RdBu', aspect='auto')
    ax.set_title('t=%d' % i)
    ax.set_xlabel('dim')
axes[0].set_ylabel('slot')
plt.tight_layout()
plt.show()

# 写入注意力
W = np.array(weights_hist).T
plt.figure(figsize=(8, 4))
plt.imshow(W, cmap='viridis', aspect='auto')
plt.colorbar(label='write weight')
plt.xlabel('step')
plt.ylabel('slot')
plt.title('Write attention')
plt.show()
```

你会看到两个重要现象:

1) 写入是“软”的:同一步可能写到多个槽(只是权重不同)
2) 地址机制决定了“写到哪里”:内容相似 + 指针移动共同作用

## 小结

NTM 最核心的价值不是“它很强”,而是它把一个关键想法讲得特别清楚:

**把存储(记忆)和计算(控制器)解耦,并用可微分寻址把两者连接起来。**

后面你学到:

- Memory Network
- Neural RAM
- 甚至很多检索增强(RAG)系统

都会看到类似的影子:把“外部信息”变成一个可读写/可检索的结构。

## 练习题

1) 把 shift 的支持从 [-1,0,+1] 扩展到 [-2,-1,0,+1,+2],观察注意力移动的平滑性。

2) 把 beta 拉大/拉小,看看内容寻址的分布会不会变得更像 one-hot。

3) 自己实现一个“读阶段”:在写完之后,用同一个 head 地址并 read,看读取向量是否能回忆出写入的模式。

4) (进阶) 用一个简单 controller(如小 RNN)去学习 copy task,需要哪些梯度(地址机制/erase-add 都要可微)。

## 延伸阅读

1) Graves et al., 2014, "Neural Turing Machines"

2) Memory-augmented neural networks 的后续工作(DNC 等)

3) 现代 LLM 的检索与工具调用:从“外部记忆”到“外部知识”

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 20 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
