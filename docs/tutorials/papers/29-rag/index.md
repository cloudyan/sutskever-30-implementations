# RAG(检索增强生成)是怎么把“搜索”和“生成”拼在一起的?

问下大家,你有没有遇到过这种 LLM 场景:

你问一个事实类问题,模型一本正经地胡说八道。

你再追问来源,它更尴尬。

晓寒以前把这类问题统称为“幻觉”,直到我系统理解了 RAG 才发现卧槽:

**很多知识密集型任务,核心不是模型不会说话,而是它没把外部知识带进来。**

RAG(Retrieval-Augmented Generation) 的核心思路就是:

1) 先检索出可能相关的文档(外部知识)
2) 再把这些文档喂给生成模型,让答案“有依据”

这篇按 notebook `29_rag.ipynb` 的 toy NumPy 实现,把 RAG 的两个经典变体讲清楚:

- RAG-Sequence:按“文档”做边缘化(marginalize)
- RAG-Token:按“token”做边缘化(每个 token 可以用不同文档)

并用最小代码把概率公式跑通。

## 1) RAG 的基本组件

RAG 通常由两部分组成:

- Retriever(检索器):P(z|x) 给出 query x 对文档 z 的分布
- Generator(生成器):P(y|x,z) 给出在文档条件下生成答案 y 的概率

直觉类比:

Retriever 像“先去图书馆把相关书翻出来”。

Generator 像“拿着书写答案”。

## 2) toy Retriever:像 DPR 一样做 dense 检索

notebook 用一个极简 retriever 来模拟 DPR:

- 把 query 编码成向量
- 和文档向量做点积
- top-k 后 softmax 得到 P(z|x)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


class SimpleRetriever:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.query_encoder_W = np.random.randn(embedding_dim, embedding_dim) * 0.01

    def encode_query(self, query_tokens):
        query_vec = np.mean(query_tokens, axis=0)
        encoded = self.query_encoder_W @ query_vec
        return encoded / (np.linalg.norm(encoded) + 1e-8)

    def retrieve(self, query_embedding, document_embeddings, k=5):
        sims = document_embeddings @ query_embedding
        top_idx = np.argsort(sims)[::-1][:k]
        top_scores = sims[top_idx]
        probs = softmax(top_scores)
        return top_idx, probs
```

真实工程里你会用 DPR(28) + 向量索引,但核心就是 P(z|x) 这一步。

## 3) toy Generator:给定文档条件的生成概率

真实 RAG 的 generator 通常是 seq2seq(比如 BART/T5) 或 decoder-only LLM。

notebook 用一个极简版本模拟 P(y|x,z):

```python
class SimpleGenerator:
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.encoder_W = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.decoder_W = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.output_W = np.random.randn(vocab_size, hidden_dim) * 0.01

    def generate_prob(self, query_tokens, doc_tokens, target_tokens):
        # 编码(query+doc)
        combined = np.concatenate([query_tokens, doc_tokens], axis=0)
        enc = np.tanh(self.encoder_W @ np.mean(combined, axis=0))

        log_prob = 0.0
        for tok in target_tokens:
            dec = np.tanh(self.decoder_W @ tok)
            h = enc + dec
            logits = self.output_W @ h
            probs = softmax(logits)
            target_idx = int(np.argmax(tok))
            log_prob += np.log(probs[target_idx] + 1e-8)

        return log_prob
```

这段代码不是为了真实生成,而是为了让你看到“条件在文档上”的概率结构。

## 4) RAG-Sequence:按文档边缘化

RAG-Sequence 的核心公式:

\[
P(y|x) = \sum_{z \in topK(x)} P(z|x)\,P(y|x,z)
\]

也就是:

先选文档(隐变量 z),再在该文档条件下生成整段答案 y。

```python
class RAGSequence:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def forward(self, query_tokens, target_tokens, document_embeddings, documents_tokens, k=5):
        q = self.retriever.encode_query(query_tokens)
        doc_idx, doc_p = self.retriever.retrieve(q, document_embeddings, k=k)

        total = 0.0
        for i, pz in zip(doc_idx, doc_p):
            log_py = self.generator.generate_prob(query_tokens, documents_tokens[i], target_tokens)
            total += pz * np.exp(log_py)

        return np.log(total + 1e-8), doc_idx, doc_p
```

直觉:

它更“整段一致”:整段答案由同一个文档支撑。

## 5) RAG-Token:按 token 边缘化

RAG-Token 的直觉更激进:

每个 token 都可以从不同文档吸收信息。

写成一类常见形式:

\[
P(y|x) = \prod_i \sum_z P(z|x)\,P(y_i|x,z,y_{<i})
\]

notebook 用 toy 版本演示“逐 token 混合文档”的结构:

```python
class RAGToken:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def forward_token(self, query_tokens, target_token, document_embeddings, documents_tokens, k=5):
        q = self.retriever.encode_query(query_tokens)
        doc_idx, doc_p = self.retriever.retrieve(q, document_embeddings, k=k)

        prob = 0.0
        for i, pz in zip(doc_idx, doc_p):
            log_py = self.generator.generate_prob(query_tokens, documents_tokens[i], [target_token])
            prob += pz * np.exp(log_py)

        return prob

    def forward(self, query_tokens, target_tokens, document_embeddings, documents_tokens, k=5):
        logp = 0.0
        for tok in target_tokens:
            p = self.forward_token(query_tokens, tok, document_embeddings, documents_tokens, k=k)
            logp += np.log(p + 1e-8)
        return logp
```

直觉:

- 更灵活:不同 token 可以“引用不同证据”
- 但也更可能带来“拼接式知识混合”(需要更强的训练与约束)

## 6) RAG 的工程关键点(比公式更重要)

1) Retriever 决定了你能不能拿到对的证据(DPR/索引/负样本)

2) Chunking 决定了文档粒度(段落太大噪声多,太小上下文不够)

3) Prompt/融合方式决定了 generator 是否会用证据

4) 评测要看“有无依据”:不仅看答案对不对,也看引用对不对

## 小结

RAG 你可以记成一句话:

**把“知识获取”交给检索,把“语言表达”交给生成。**

RAG-Sequence 更一致,RAG-Token 更灵活。

而真正决定效果的,往往是 retriever 的训练质量与检索数据管线。

## 练习题

1) 把 top-k 从 1/3/5/10 改一改,观察 RAG-Sequence 的 doc_probs 分布会不会更分散。

2) 把 retriever 的 softmax 温度调一下(更尖/更平),RAG 的边缘化会有什么变化?

3) 思考题:RAG-Token 为什么可能“每个 token 用不同文档”?它对事实一致性有什么利弊?

4) 思考题:如果 retriever 召回的文档里包含冲突信息,你希望 generator 怎么做?

## 延伸阅读

1) Lewis et al., 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

2) DPR(Paper 28):RAG 的检索底座

3) 长上下文与位置偏置(Paper 30):证据在长上下文里“放哪里”很关键

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 29 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
