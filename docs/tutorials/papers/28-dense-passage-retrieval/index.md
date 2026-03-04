# DPR(密集检索)是怎么把“问题-段落”映射到同一个向量空间的?

问下大家,你做过检索吗?

以前我们用 BM25 这种 sparse 检索,本质是“词匹配”:问句里出现的词,文档里也得出现,分数才高。

但现实里你经常会遇到:

- 问句和答案段落用的是不同措辞(同义改写)
- 关键词不重合,但语义是对的

晓寒第一次用 Dense Retrieval 时最大的感受就是:卧槽,它不是在找“同词”,而是在找“同义”。

**Dense Passage Retrieval(DPR)** 的核心做法很干净:

1) 用 question encoder 把问题编码成向量 q
2) 用 passage encoder 把每个段落编码成向量 p
3) 用内积/余弦相似度做检索:score(q,p)=q·p
4) 训练时用 in-batch negatives 做对比学习,把正确段落拉近,把其他段落推远

这篇按 notebook `28_dense_passage_retrieval.ipynb` 的纯 NumPy toy 实现,把 DPR 的关键机制跑通:

- Dual Encoder
- MIPS(max inner product search) 的检索
- InfoNCE/对比损失 + in-batch negatives
- 与 BM25 的直觉对比

## 1) Dual Encoder:两台编码器,一个空间

真实 DPR 用的是 BERT 编码器,这里用一个极简 RNN encoder 代替,重点是结构:

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)


class SimpleTextEncoder:
    """玩具 encoder:embedding + RNN + L2 normalize"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_xh = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros((hidden_dim, 1))
        self.W_out = np.random.randn(hidden_dim, hidden_dim) * 0.01

    def encode(self, token_ids):
        h = np.zeros((self.W_hh.shape[0], 1))
        for tid in token_ids:
            x = self.embeddings[tid].reshape(-1, 1)
            h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)

        out = (self.W_out @ h).flatten()
        return out / (np.linalg.norm(out) + 1e-8)
```

Dual encoder 的关键点是:问题和段落分别编码,但最终要落在同一空间里,相似的语义就会靠近。

## 2) 数据:一个最小 QA 语料

notebook 用少量 passage + question 的配对作为监督:

```python
passages = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "The Great Wall of China is a series of fortifications in northern China.",
    # ...
]

questions = [
    ("What is the Eiffel Tower?", 0),
    ("Where is the Great Wall located?", 1),
]
```

## 3) 检索:用内积做 MIPS

有了 q 和所有 p_i,检索就是:

\[
\arg\max_i\; q \cdot p_i
\]

```python
def retrieve_top_k(query_embedding, passage_embeddings, k=3):
    sims = passage_embeddings @ query_embedding
    top = np.argsort(sims)[::-1][:k]
    return top, sims[top]
```

这一步在工程里会变成 ANN/向量索引(FAISS/HNSW),但数学内核就是内积。

## 4) 训练:in-batch negatives(又强又省)

DPR 训练最经典的技巧之一是 in-batch negatives:

- 一个 batch 里有 B 个 (q_i, p_i^+) 正样本
- 对 q_i 来说,其它 p_j(j!=i) 都当负样本

这样你不用额外挖 hard negatives,就能得到不少“看起来很像”的负例。

notebook 用 InfoNCE(对比学习)形式写了一个最小 loss:

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def contrastive_loss(query_emb, positive_emb, negative_embs):
    pos = np.dot(query_emb, positive_emb)
    negs = [np.dot(query_emb, n) for n in negative_embs]
    scores = np.array([pos] + negs)
    probs = softmax(scores)
    return -np.log(probs[0] + 1e-8)
```

直觉:

你在教模型做一个分类问题:

"在这一堆候选段落里,哪个才是这道问题的正确段落?"

## 5) Dense vs BM25:语义匹配 vs 词匹配

BM25 的优点:

- 不需要训练
- 对关键词精确匹配非常强

Dense 的优点:

- 同义改写更鲁棒
- 能召回语义相关但用词不同的段落

notebook 里实现了一个简化 BM25,并输出 recall@k / MRR 等检索指标做对比(这里 encoder 未训练,结果只是结构演示)。

## 小结

DPR 你可以记成一个非常工程化的公式:

**检索 = 向量相似度,训练 = 对比学习把正例拉近、负例推远。**

后面你看 RAG(Paper 29),会发现 DPR 就是它的“检索发动机”。

## 练习题

1) 把 dot product 换成 cosine(其实 L2 normalize 后两者等价),看看实现差异在哪里。

2) 做一个真正的梯度更新(哪怕只更新 embeddings),让检索结果不再随机。

3) 加一个 hard negative:故意添加一个内容很像但答案不对的 passage,看 loss 是否更大。

4) 思考题:为什么 in-batch negatives 在大 batch 时效果更好?它和“对比学习的负样本数”是什么关系?

## 延伸阅读

1) Karpukhin et al., 2020, "Dense Passage Retrieval for Open-Domain Question Answering"

2) 对比学习(InfoNCE)与表示学习

3) 向量索引与 ANN(MIPS/HNSW/FAISS)

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 28 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
