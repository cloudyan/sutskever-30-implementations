# AI 基础入门学习指南

> 📅 创建日期：2026-03-22  
> 🎯 目标：从前端开发到 AI 基础入门  
> ⏱️ 周期：6 周（阶段 0 + 阶段 1）

---

## 📋 目录

1. [学习路径总览](#学习路径总览)
2. [第 1 周：数学基础 + 反向传播](#第 1 周数学基础--反向传播)
3. [第 2 周：RNN/LSTM](#第 2 周 rnnlstm)
4. [第 3 周：CNN/AlexNet](#第 3 周 cnnalexnet)
5. [第 4 周：词向量/Word2Vec](#第 4 周词向量 word2vec)
6. [第 5 周：Seq2Seq/Attention](#第 5 周 seq2seqattention)
7. [第 6 周：Transformer 预习](#第 6 周 transformer 预习)
8. [代码实践指南](#代码实践指南)
9. [常见问题解答](#常见问题解答)

---

## 学习路径总览

### 阶段 0：基础入门（3 周）⭐ 进行中

| 周次 | 主题 | 核心论文 | 实践项目 |
|------|------|----------|----------|
| 第 1 周 | 数学基础 + 反向传播 | 反向传播 (1986) | NumPy 实现神经网络 |
| 第 2 周 | RNN/LSTM | LSTM (1997) | Sutskever-30 Notebook 03 |
| 第 3 周 | CNN/图像识别 | AlexNet (2012) | Sutskever-30 Notebook 07 |

### 阶段 1：NLP 基础（3 周）

| 周次 | 主题 | 核心论文 | 实践项目 |
|------|------|----------|----------|
| 第 4 周 | 词向量 | Word2Vec (2013) | 训练词向量 |
| 第 5 周 | 序列建模 | Seq2Seq (2014) + Attention (2014) | 机器翻译 Demo |
| 第 6 周 | Transformer 预习 | Transformer (2017) | 跑通预训练模型 |

---

## 第 1 周：数学基础 + 反向传播

### 📅 每日计划

#### 第 1 天：微积分基础（导数）

**学习目标**：
- 理解什么是导数
- 理解导数的几何意义（斜率）
- 理解偏导数（多变量）

**学习资源**：
- 🎬 [3Blue1Brown - 微积分本质 第 1-3 集](https://www.bilibili.com/video/BV1qW411N7o8)（B 站，中文字幕）
- 📖 [知乎 - 如何理解导数](https://www.zhihu.com/question/24852119)

**检查清单**：
- [ ] 能解释导数是什么
- [ ] 理解导数 = 变化率 = 斜率
- [ ] 知道什么是偏导数

---

#### 第 2 天：微积分基础（链式法则）

**学习目标**：
- 理解复合函数求导
- 掌握链式法则

**学习资源**：
- 🎬 [3Blue1Brown - 微积分本质 第 4 集](https://www.bilibili.com/video/BV1qW411N7o8_p4)
- 📖 [知乎 - 链式法则详解](https://zhuanlan.zhihu.com/p/36497644)

**检查清单**：
- [ ] 能写出链式法则公式
- [ ] 理解为什么需要链式法则

---

#### 第 3 天：神经网络基础

**学习目标**：
- 理解什么是神经元
- 理解什么是权重、偏置
- 理解激活函数的作用

**学习资源**：
- 🎬 [3Blue1Brown - 神经网络与深度学习 第 1 集](https://www.bilibili.com/video/BV12x41137dT)
- 📖 [Michael Nielsen - 神经网络与深度学习 第 1 章](http://neuralnetworksanddeeplearning.com/chap1.html)

**检查清单**：
- [ ] 能画出单个神经元结构图
- [ ] 理解权重和偏置的作用
- [ ] 知道 sigmoid、ReLU 等激活函数

---

#### 第 4 天：前向传播

**学习目标**：
- 理解数据如何在神经网络中流动
- 理解多层网络的计算过程

**学习资源**：
- 🎬 [3Blue1Brown - 神经网络与深度学习 第 2 集](https://www.bilibili.com/video/BV12x41137dT_p2)
- 📖 [吴恩达 Coursera - 前向传播](https://www.coursera.org/learn/neural-networks-deep-learning)

**检查清单**：
- [ ] 能手算简单网络的前向传播
- [ ] 理解矩阵乘法在其中的作用

---

#### 第 5-6 天：反向传播（核心！）

**学习目标**：
- 理解反向传播的核心思想
- 理解梯度下降
- 理解如何更新权重

**学习资源**：
- 🎬 [3Blue1Brown - 神经网络与深度学习 第 3-4 集](https://www.bilibili.com/video/BV12x41137dT_p3)
- 📖 [吴恩达 Coursera - 反向传播](https://www.coursera.org/learn/neural-networks-deep-learning)
- 📖 [知乎 - 反向传播详解](https://zhuanlan.zhihu.com/p/24530447)

**检查清单**：
- [ ] 能解释什么是梯度
- [ ] 理解反向传播的三步：前向→计算误差→反向更新
- [ ] 知道什么是学习率

---

#### 第 7 天：代码实践

**实践任务**：
```bash
cd /Users/cloudyan/data/code/llm/sutskever-30-implementations

# 运行 RNN 基础示例
jupyter notebook 02_char_rnn_karpathy.ipynb
```

**检查清单**：
- [ ] 能跑通 Notebook
- [ ] 修改参数观察效果
- [ ] 理解每一行代码的作用

---

### 📚 核心概念速查

| 概念 | 通俗解释 | 公式 |
|------|----------|------|
| **导数** | 函数的变化率（斜率） | f'(x) = lim(h→0) [f(x+h)-f(x)]/h |
| **偏导数** | 多变量函数，对其中一个变量求导 | ∂f/∂x |
| **链式法则** | 复合函数求导 | (f∘g)' = f'∘g × g' |
| **梯度** | 多变量导数向量 | ∇f = [∂f/∂x₁, ∂f/∂x₂, ...] |
| **梯度下降** | 沿梯度反方向更新，最小化损失 | w_new = w_old - η × ∇L |
| **激活函数** | 引入非线性，让网络能学习复杂模式 | sigmoid, ReLU, tanh |

---

## 第 2 周：RNN/LSTM

### 📅 每日计划

#### 第 1 天：序列数据概念

**学习目标**：
- 理解什么是序列数据
- 理解 RNN 的基本结构

**学习资源**：
- 🎬 [李宏毅 - RNN 第 1 部分](https://www.bilibili.com/video/BV1JE411g7XF)
- 📖 [知乎 - RNN 入门](https://zhuanlan.zhihu.com/p/30844968)

**检查清单**：
- [ ] 能举例说明什么是序列数据
- [ ] 理解 RNN 为什么能处理序列

---

#### 第 2 天：RNN 详解

**学习目标**：
- 理解 RNN 的前向传播
- 理解 RNN 的隐藏状态

**学习资源**：
- 🎬 [3Blue1Brown - LSTM 视频 第 1 部分](https://www.bilibili.com/video/BV1t4411E7fF)
- 📖 [Colah's Blog - RNN 简介](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

**检查清单**：
- [ ] 能画出 RNN 展开图
- [ ] 理解隐藏状态的作用

---

#### 第 3 天：RNN 的问题

**学习目标**：
- 理解长程依赖问题
- 理解梯度消失问题

**学习资源**：
- 📖 [知乎 - RNN 梯度消失详解](https://zhuanlan.zhihu.com/p/32085878)

**检查清单**：
- [ ] 能解释为什么 RNN 记不住长序列
- [ ] 理解梯度消失的数学原因

---

#### 第 4-6 天：LSTM 详解（核心！）

**学习目标**：
- 理解 LSTM 的三个门
- 理解细胞状态
- 理解为什么能解决梯度消失

**学习资源**：
- 🎬 [3Blue1Brown - LSTM 完整视频](https://www.bilibili.com/video/BV1t4411E7fF)
- 📖 [Colah's Blog - Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)（经典！必读！）
- 📖 [知乎 - LSTM 图解](https://zhuanlan.zhihu.com/p/32424444)

**检查清单**：
- [ ] 能画出 LSTM 细胞结构图
- [ ] 理解遗忘门、输入门、输出门的作用
- [ ] 能写出 LSTM 核心公式
- [ ] 理解为什么能解决梯度消失

---

#### 第 7 天：代码实践

**实践任务**：
```bash
cd /Users/cloudyan/data/code/llm/sutskever-30-implementations

# 运行 LSTM 理解 Notebook
jupyter notebook 03_lstm_understanding.ipynb
```

**检查清单**：
- [ ] 能跑通 Notebook
- [ ] 理解每个门的计算过程
- [ ] 修改参数观察效果

---

### 📚 LSTM 核心公式速查

```
遗忘门：fₜ = σ(W_f · [hₜ₋₁, xₜ] + b_f)
输入门：iₜ = σ(W_i · [hₜ₋₁, xₜ] + b_i)
细胞候选：C̃ₜ = tanh(W_C · [hₜ₋₁, xₜ] + b_C)
细胞更新：Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ  ← 核心！
输出门：oₜ = σ(W_o · [hₜ₋₁, xₜ] + b_o)
隐藏状态：hₜ = oₜ × tanh(Cₜ)
```

---

## 第 3 周：CNN/AlexNet

### 📅 每日计划

#### 第 1 天：卷积神经网络基础

**学习目标**：
- 理解什么是卷积
- 理解卷积核/滤波器

**学习资源**：
- 🎬 [3Blue1Brown - 卷积](https://www.bilibili.com/video/BV1qW411N7o8_conv)
- 📖 [知乎 - CNN 入门](https://zhuanlan.zhihu.com/p/33005239)

**检查清单**：
- [ ] 能解释什么是卷积操作
- [ ] 理解卷积核的作用

---

#### 第 2 天：CNN 架构

**学习目标**：
- 理解卷积层
- 理解池化层
- 理解全连接层

**学习资源**：
- 🎬 [李飞飞 CS231n 第 1-3 讲](https://www.bilibili.com/video/BV1iJ411E7xW)
- 📖 [CS231n 课程笔记 - CNN](https://cs231n.github.io/convolutional-networks/)

**检查清单**：
- [ ] 能画出 CNN 基本架构
- [ ] 理解各层的作用

---

#### 第 3-4 天：AlexNet 论文精读

**学习目标**：
- 理解 AlexNet 的创新点
- 理解 ReLU、Dropout、数据增强

**学习资源**：
- 📖 [AlexNet 原论文](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- 📖 [知乎 - AlexNet 论文解读](https://zhuanlan.zhihu.com/p/34946836)

**检查清单**：
- [ ] 理解 AlexNet 的 8 层结构
- [ ] 知道 ReLU 的作用
- [ ] 理解 Dropout 防止过拟合

---

#### 第 5-6 天：实践

**实践任务**：
```bash
cd /Users/cloudyan/data/code/llm/sutskever-30-implementations

# 运行 CNN  Notebook
jupyter notebook 07_alexnet_cnn.ipynb
```

**检查清单**：
- [ ] 能跑通 Notebook
- [ ] 理解卷积操作
- [ ] 修改卷积核观察效果

---

#### 第 7 天：复习

**复习内容**：
- 反向传播
- LSTM
- CNN

**检查清单**：
- [ ] 能用自己的话解释三个核心概念
- [ ] 准备好进入下一阶段

---

## 第 4 周：词向量/Word2Vec

### 📅 每日计划

#### 第 1-2 天：词向量概念

**学习目标**：
- 理解为什么需要词向量
- 理解分布式表示

**学习资源**：
- 📖 [知乎 - Word2Vec 详解](https://zhuanlan.zhihu.com/p/35357486)

**检查清单**：
- [ ] 理解词向量的意义
- [ ] 知道 One-Hot 编码的缺点

---

#### 第 3-5 天：Word2Vec 详解

**学习目标**：
- 理解 Skip-gram 和 CBOW
- 理解负采样

**学习资源**：
- 📖 [Word2Vec 原论文](https://arxiv.org/abs/1301.3781)
- 📖 [知乎 - Word2Vec 原理](https://zhuanlan.zhihu.com/p/35357486)

**检查清单**：
- [ ] 理解 Skip-gram 和 CBOW 的区别
- [ ] 知道如何训练词向量

---

#### 第 6-7 天：实践

**实践任务**：
```bash
# 使用 gensim 训练词向量
pip install gensim
```

**检查清单**：
- [ ] 能训练自己的词向量
- [ ] 可视化词向量（t-SNE）

---

## 第 5 周：Seq2Seq/Attention

### 📅 每日计划

#### 第 1-2 天：Seq2Seq

**学习目标**：
- 理解编码器 - 解码器架构
- 理解序列到序列任务

**学习资源**：
- 📖 [Seq2Seq 原论文](https://arxiv.org/abs/1409.3215)
- 📖 [知乎 - Seq2Seq 详解](https://zhuanlan.zhihu.com/p/30039900)

**检查清单**：
- [ ] 理解 Encoder-Decoder 架构
- [ ] 知道应用场景（翻译、摘要等）

---

#### 第 3-5 天：Attention 机制

**学习目标**：
- 理解 Attention 的核心思想
- 理解 Bahdanau Attention

**学习资源**：
- 🎬 [李宏毅 - Attention](https://www.bilibili.com/video/BV1JE411g7XF_attention)
- 📖 [Bahdanau Attention 原论文](https://arxiv.org/abs/1409.0473)
- 📖 [知乎 - Attention 详解](https://zhuanlan.zhihu.com/p/31547842)

**检查清单**：
- [ ] 理解 Attention 为什么有效
- [ ] 能画出 Attention 计算流程图

---

#### 第 6-7 天：实践

**实践任务**：
```bash
cd /Users/cloudyan/data/code/llm/sutskever-30-implementations

# 运行 Attention Notebook
jupyter notebook 14_bahdanau_attention.ipynb
```

---

## 第 6 周：Transformer 预习

### 📅 每日计划

#### 第 1-3 天：Transformer 预习

**学习目标**：
- 理解 Self-Attention
- 理解多头注意力
- 理解位置编码

**学习资源**：
- 📖 [Transformer 原论文](https://arxiv.org/abs/1706.03762)
- 📖 [知乎 - Transformer 详解](https://zhuanlan.zhihu.com/p/33881762)
- 📖 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（图解，强烈推荐！）

**检查清单**：
- [ ] 理解 Self-Attention 计算过程
- [ ] 知道 Q、K、V 的含义
- [ ] 理解多头注意力的作用

---

#### 第 4-6 天：Transformer 实践

**实践任务**：
```bash
cd /Users/cloudyan/data/code/llm/sutskever-30-implementations

# 运行 Transformer Notebook
jupyter notebook 13_attention_is_all_you_need.ipynb
```

**检查清单**：
- [ ] 能跑通 Notebook
- [ ] 理解每个组件的作用

---

#### 第 7 天：总结与展望

**复习内容**：
- 6 周学习总结
- 准备进入大模型阶段

**检查清单**：
- [ ] 能画出完整的知识图谱
- [ ] 准备好学习 BERT/GPT

---

## 代码实践指南

### Sutskever-30 项目使用

```bash
# 进入项目目录
cd /Users/cloudyan/data/code/llm/sutskever-30-implementations

# 安装依赖
pip install numpy matplotlib jupyter scipy

# 启动 Jupyter
jupyter notebook
```

### 按周次执行 Notebook

| 周次 | Notebook | 说明 |
|------|----------|------|
| 第 1 周 | 02_char_rnn_karpathy.ipynb | RNN 基础 |
| 第 2 周 | 03_lstm_understanding.ipynb | LSTM 详解 |
| 第 3 周 | 07_alexnet_cnn.ipynb | CNN/AlexNet |
| 第 4 周 | - | Word2Vec（可用 gensim） |
| 第 5 周 | 14_bahdanau_attention.ipynb | Attention |
| 第 6 周 | 13_attention_is_all_you_need.ipynb | Transformer |

---

## 常见问题解答

### Q1: 数学基础差怎么办？
**A**: 先看 3Blue1Brown 的微积分和线性代数视频，直观理解概念，不需要深究证明。

### Q2: 英语不好怎么办？
**A**: 优先看中文字幕视频和知乎文章，英文论文可以等有一定基础后再看。

### Q3: 代码看不懂怎么办？
**A**: 先跑通 Notebook，修改参数观察效果，逐步理解每一行的作用。

### Q4: 学完记不住怎么办？
**A**: 正常！多做笔记，用自己的话总结，定期复习。

### Q5: 每天需要学多久？
**A**: 建议每天 1-2 小时，周末可以多一些。关键是坚持，不要一天学 10 小时然后放弃。

### Q6: 学完这些能做什么？
**A**: 
- 理解深度学习基础
- 能看懂大部分 AI 论文
- 能跑通常见模型
- 为学习大模型打下基础

---

## 学习进度追踪

### 阶段 0：基础入门

| 周次 | 主题 | 开始日期 | 完成日期 | 状态 |
|------|------|----------|----------|------|
| 第 1 周 | 反向传播 | 2026-03-22 | - | ⏳ 进行中 |
| 第 2 周 | LSTM | - | - | ⏳ 待开始 |
| 第 3 周 | CNN/AlexNet | - | - | ⏳ 待开始 |

### 阶段 1：NLP 基础

| 周次 | 主题 | 开始日期 | 完成日期 | 状态 |
|------|------|----------|----------|------|
| 第 4 周 | Word2Vec | - | - | ⏳ 待开始 |
| 第 5 周 | Seq2Seq/Attention | - | - | ⏳ 待开始 |
| 第 6 周 | Transformer | - | - | ⏳ 待开始 |

---

## 资源汇总

### 视频教程

| 资源 | 平台 | 链接 |
|------|------|------|
| 3Blue1Brown - 微积分 | B 站 | 搜索"3Blue1Brown 微积分" |
| 3Blue1Brown - 神经网络 | B 站 | 搜索"3Blue1Brown 神经网络" |
| 3Blue1Brown - LSTM | B 站 | 搜索"3Blue1Brown LSTM" |
| 李宏毅 - 机器学习 | B 站/YouTube | 搜索"李宏毅机器学习" |
| 李飞飞 - CS231n | B 站/YouTube | 搜索"CS231n" |
| 吴恩达 - 机器学习 | Coursera | [coursera.org](https://www.coursera.org/learn/machine-learning) |

### 图文教程

| 资源 | 说明 | 链接 |
|------|------|------|
| Michael Nielsen - 神经网络 | 免费在线书 | [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/) |
| Colah's Blog | LSTM 经典博文 | [colah.github.io](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| CS231n 笔记 | 计算机视觉 | [cs231n.github.io](https://cs231n.github.io/) |
| The Illustrated Transformer | Transformer 图解 | [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/) |
| 知乎 AI 专栏 | 中文高质量文章 | [zhihu.com](https://www.zhihu.com/topic/19574076) |

### 代码资源

| 资源 | 说明 | 位置 |
|------|------|------|
| Sutskever-30 | 30 篇论文实现 | `/data/code/llm/sutskever-30-implementations` |
| 本指南代码 | 反向传播/LSTM 实现 | `memory/2026-03-22-backprop.md` `memory/2026-03-22-lstm.md` |

---

## 学习建议

### ✅ 应该做的

1. **每天坚持**：哪怕只学 30 分钟
2. **做笔记**：用自己的话总结
3. **跑代码**：理论 + 实践结合
4. **提问**：不懂就问（问我！）
5. **复习**：定期回顾之前的内容

### ❌ 不应该做的

1. **不要一天学 10 小时**：容易 burnout
2. **不要只看不练**：一定要动手
3. **不要追求完美**：先理解大意，细节后面补
4. **不要跳过基础**：基础不牢，地动山摇
5. **不要孤军奋战**：有问题就问

---

## 联系方式

学习过程中有任何问题，随时找我（悟空 🐒）！

可以通过：
- 飞书消息
- 直接问"这个概念我不懂"
- 发代码截图问"这行为什么这样写"

---

*创建日期：2026-03-22 | 整理：悟空 🐒*

*最后更新：2026-03-22*
