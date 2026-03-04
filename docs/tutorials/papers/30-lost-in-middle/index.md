# Lost in the Middle 是怎么解释“长上下文中间信息用不上”的?

问下大家,你有没有遇到过这种奇怪现象:

你把答案证据塞进一个很长的上下文里,

- 放在最前面,模型答对
- 放在最后面,模型也答对
- 但放在中间,它突然像瞎了一样

晓寒第一次看到这个现象时,第一反应是“是不是检索错了”。

后来读了 **Lost in the Middle** 才发现卧槽:

**很多模型对位置有偏置(position bias),长上下文里会出现典型的 U 型曲线:两头强,中间弱。**

这篇按 notebook `30_lost_in_middle.ipynb` 的 toy 实现,用一个“多文档 QA”模拟把这个现象讲清楚,并给出对 RAG 的直接工程建议。

## 1) 先造一个多文档 QA 场景

我们有:

- 1 个相关文档(包含答案)
- N-1 个干扰文档
- 一个问题

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


class Document:
    def __init__(self, content, is_relevant=False):
        self.content = content
        self.is_relevant = is_relevant


relevant_doc = Document(
    "The Eiffel Tower was completed in 1889 ...",
    is_relevant=True,
)

distractor_docs = [
    Document("The Great Wall of China is over 13,000 miles long ..."),
    Document("The Statue of Liberty was gifted by France ..."),
    # ...
]

query = "When was the Eiffel Tower completed?"
correct_answer = "1889"
```

## 2) 用一个 toy LM 模拟“位置偏置”

论文里观察到的一类典型模式是 U-shaped bias:

- 开头权重高(primacy)
- 结尾权重高(recency)
- 中间权重低

notebook 用一个简化模型:

"模型使用相关文档的概率 ≈ 该文档所在位置的注意力权重"

```python
class SimpleLM:
    def __init__(self, position_bias_type='u_shaped'):
        self.position_bias_type = position_bias_type

    def get_position_weights(self, num_positions):
        pos = np.arange(num_positions)

        if self.position_bias_type == 'uniform':
            w = np.ones(num_positions)
        elif self.position_bias_type == 'u_shaped':
            x = pos / (num_positions - 1)
            w = 4 * (x - 0.5) ** 2 + 0.3
        elif self.position_bias_type == 'recency':
            w = np.exp(pos * 0.2)
        elif self.position_bias_type == 'primacy':
            w = np.exp(-pos * 0.2)

        return w / np.sum(w)

    def answer_query(self, query, documents):
        w = self.get_position_weights(len(documents))
        rel_pos = next((i for i, d in enumerate(documents) if d.is_relevant), None)
        return 0.0 if rel_pos is None else float(w[rel_pos])
```

这当然不是一个真实 LM,但它能把“位置偏置 → 正确率曲线”的关系用最小数学表达出来。

## 3) 把相关文档放到不同位置,看性能曲线

```python
def test_all_positions(model, query, relevant_doc, distractor_docs):
    n = len(distractor_docs) + 1
    acc = []
    for pos in range(n):
        docs = distractor_docs[:pos] + [relevant_doc] + distractor_docs[pos:]
        docs = docs[:n]
        acc.append(model.answer_query(query, docs))
    return acc
```

你会得到一个非常“教科书”的 U 型:

- 相关文档在开头:高
- 相关文档在结尾:高
- 相关文档在中间:低

这就是“Lost in the Middle”。

## 4) 上下文越长,中间越惨

notebook 还测试了不同文档数量(length):

- length 越长
- 中间位置分到的注意力越少
- 中间惩罚更明显

直觉上也合理:

注意力预算是有限的,长上下文会稀释中间。

## 5) 对 RAG 的直接工程建议:把关键证据放在边上

如果你做的是 RAG/多文档 QA,这篇论文会给你一个非常实用的结论:

**把最重要的文档放在开头或结尾,不要让它落在中间。**

notebook 用一个 ordering 策略函数做了演示:

- most_relevant_first:把最相关放前面
- most_relevant_edges:把最相关放两头

你可以把它当作 prompt 组织的启发式。

## 6) 更现实的补救策略(除了排序)

1) Chunk 更合理:减少无关 token
2) 多轮检索:先粗检索,再针对子问题检索
3) 引用提示:让模型“必须引用证据”
4) 结构化上下文:标题/编号/摘要,让模型更容易定位
5) 长上下文模型/位置编码改进:从模型侧缓解偏置

## 小结

Lost in the Middle 这篇的核心结论可以直接写进你的 RAG 工程 checklist:

1) 长上下文不是越长越好,中间信息可能用不上
2) 存在位置偏置,U 型曲线常见
3) 证据组织很关键:优先把核心证据放在两头

如果你把 Paper 28(DPR) + Paper 29(RAG) + 这篇连起来,你会发现:

检索到证据只是第一步,**证据怎么放**同样决定成败。

## 练习题

1) 把 bias_type 从 u_shaped 改成 recency/primacy,画出不同偏置下的曲线,并解释现实中为什么常见 u_shaped。

2) 给 ordering 策略再加一个 "interleave"(高相关与低相关交替),看是否比直接堆边缘更稳。

3) 思考题:如果你的证据必须出现在中间(比如格式限制),你能用什么提示工程把它“拉出来”? 

4) 思考题:当上下文特别长时,你更倾向于用 RAG 多轮检索还是直接喂长文本?为什么?

## 延伸阅读

1) Liu et al., 2023, "Lost in the Middle: How Language Models Use Long Contexts"

2) RAG(Paper 29):证据组织与生成

3) 长上下文建模与位置编码改进(各种 RoPE/ALiBi/长上下文训练)

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 30 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
