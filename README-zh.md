# Sutskever 30 - 完整实现套件

**Ilya Sutskever 推荐的 30 篇奠基性论文的综合玩具实现**

[![实现数](https://img.shields.io/badge/Implementations-30%2F30-brightgreen)](https://github.com/pageman/sutskever-30-implementations)
[![覆盖率](https://img.shields.io/badge/Coverage-100%25-blue)](https://github.com/pageman/sutskever-30-implementations)
[![Python](https://img.shields.io/badge/Python-NumPy%20Only-yellow)](https://numpy.org/)

## 概览

这个仓库包含 Ilya Sutskever 著名阅读列表中论文的详细、教育性实现——这个集合是他告诉 John Carmack 可以教会你深度学习中"90% 重要内容"的资料。

**进度：30/30 篇论文 (100%) - 已完成！🎉**

每个实现：
- ✅ 仅使用 NumPy（无深度学习框架）以获得教育清晰度
- ✅ 包含合成/自举数据以便立即执行
- ✅ 提供丰富的可视化和解释
- ✅ 展示每篇论文的核心概念
- ✅ 在 Jupyter notebook 中运行以实现交互式学习

## 快速开始

```bash
# 进入目录
cd sutskever-30-implementations

# 安装依赖
pip install numpy matplotlib scipy

# 运行任意 notebook
jupyter notebook 02_char_rnn_karpathy.ipynb
```

## Sutskever 30 篇论文

### 基础概念（论文 1-5）

| # | 论文 | Notebook | 核心概念 |
|---|-------|----------|--------------|
| 1 | 复杂动力学的第一定律 | ✅ `01_complexity_dynamics.ipynb` | 熵、复杂度增长、元胞自动机 |
| 2 | RNN 的惊人有效性 | ✅ `02_char_rnn_karpathy.ipynb` | 字符级模型、RNN 基础、文本生成 |
| 3 | 理解 LSTM 网络 | ✅ `03_lstm_understanding.ipynb` | 门控、长期记忆、梯度流 |
| 4 | RNN 正则化 | ✅ `04_rnn_regularization.ipynb` | 序列的 Dropout、变分 Dropout |
| 5 | 保持神经网络简单 | ✅ `05_neural_network_pruning.ipynb` | MDL 原理、权重剪枝、90%+ 稀疏性 |

### 架构与机制（论文 6-15）

| # | 论文 | Notebook | 核心概念 |
|---|-------|----------|--------------|
| 6 | 指针网络 | ✅ `06_pointer_networks.ipynb` | 注意力作为指针、组合问题 |
| 7 | ImageNet/AlexNet | ✅ `07_alexnet_cnn.ipynb` | CNN、卷积、数据增强 |
| 8 | 顺序很重要：集合的 Seq2Seq | ✅ `08_seq2seq_for_sets.ipynb` | 集合编码、排列不变性、注意力池化 |
| 9 | GPipe | ✅ `09_gpipe.ipynb` | 流水线并行、微批处理、重新物化 |
| 10 | 深度残差学习 (ResNet) | ✅ `10_resnet_deep_residual.ipynb` | 跳跃连接、梯度高速公路 |
| 11 | 扩张卷积 | ✅ `11_dilated_convolutions.ipynb` | 感受野、多尺度 |
| 12 | 神经消息传递 (GNN) | ✅ `12_graph_neural_networks.ipynb` | 图网络、消息传递 |
| 13 | **注意力机制即你所需要的一切** | ✅ `13_attention_is_all_you_need.ipynb` | Transformer、自注意力、多头 |
| 14 | 神经机器翻译 | ✅ `14_bahdanau_attention.ipynb` | Seq2seq、Bahdanau 注意力 |
| 15 | ResNet 中的恒等映射 | ✅ `15_identity_mappings_resnet.ipynb` | 预激活、梯度流 |

### 高级主题（论文 16-22）

| # | 论文 | Notebook | 核心概念 |
|---|-------|----------|--------------|
| 16 | 关系推理 | ✅ `16_relational_reasoning.ipynb` | 关系网络、成对函数 |
| 17 | **变分有损自编码器** | ✅ `17_variational_autoencoder.ipynb` | VAE、ELBO、重参数化技巧 |
| 18 | **关系 RNN** | ✅ `18_relational_rnn.ipynb` | 关系记忆、多头自注意力、手动反向传播 (~1100 行) |
| 19 | 咖啡自动机 | ✅ `19_coffee_automaton.ipynb` | 不可逆性、熵、时间之箭、Landauer 原理 |
| 20 | **神经图灵机** | ✅ `20_neural_turing_machine.ipynb` | 外部记忆、可微分寻址 |
| 21 | Deep Speech 2 (CTC) | ✅ `21_ctc_speech.ipynb` | CTC 损失、语音识别 |
| 22 | **缩放定律** | ✅ `22_scaling_laws.ipynb` | 幂律、计算最优训练 |

### 理论与元学习（论文 23-30）

| # | 论文 | Notebook | 核心概念 |
|---|-------|----------|--------------|
| 23 | MDL 原理 | ✅ `23_mdl_principle.ipynb` | 信息论、模型选择、压缩 |
| 24 | **机器超级智能** | ✅ `24_machine_super_intelligence.ipynb` | 通用 AI、AIXI、Solomonoff 归纳、智能度量、自我改进 |
| 25 | Kolmogorov 复杂度 | ✅ `25_kolmogorov_complexity.ipynb` | 压缩、算法随机性、通用先验 |
| 26 | **CS231n：视觉识别的 CNN** | ✅ `26_cs231n_cnn_fundamentals.ipynb` | 图像分类流水线、kNN/线性/NN/CNN、反向传播、优化、神经网络的调参 |
| 27 | 多令牌预测 | ✅ `27_multi_token_prediction.ipynb` | 多个未来令牌、样本效率、2-3 倍更快 |
| 28 | 稠密段落检索 | ✅ `28_dense_passage_retrieval.ipynb` | 双编码器、MIPS、批内负样本 |
| 29 | 检索增强生成 | ✅ `29_rag.ipynb` | RAG-Sequence、RAG-Token、知识检索 |
| 30 | 迷失在中间 | ✅ `30_lost_in_middle.ipynb` | 位置偏差、长上下文、U 型曲线 |

## 精选实现

### 🌟 必读 Notebook

这些实现涵盖了最有影响力的论文，并展示了核心深度学习概念：

#### 基础
1. **`02_char_rnn_karpathy.ipynb`** - 字符级 RNN
   - 从零构建 RNN
   - 理解通过时间的反向传播
   - 生成文本

2. **`03_lstm_understanding.ipynb`** - LSTM 网络
   - 实现遗忘/输入/输出门
   - 可视化门激活
   - 与标准 RNN 比较

3. **`04_rnn_regularization.ipynb`** - RNN 正则化
   - RNN 的变分 Dropout
   - 正确的 Dropout 放置
   - 训练改进

4. **`05_neural_network_pruning.ipynb`** - 网络剪枝与 MDL
   - 基于幅度的剪枝
   - 带微调的迭代剪枝
   - 90%+ 稀疏性，损失最小
   - 最小描述长度原理

#### 计算机视觉
5. **`07_alexnet_cnn.ipynb`** - CNN 与 AlexNet
   - 从零实现卷积层
   - 最大池化和 ReLU
   - 数据增强技术

6. **`10_resnet_deep_residual.ipynb`** - ResNet
   - 跳跃连接解决退化
   - 梯度流可视化
   - 恒等映射直觉

7. **`15_identity_mappings_resnet.ipynb`** - 预激活 ResNet
   - 预激活 vs 后激活
   - 更好的梯度流
   - 训练 1000+ 层网络

8. **`11_dilated_convolutions.ipynb`** - 扩张卷积
   - 多尺度感受野
   - 无需池化
   - 语义分割

#### 注意力与 Transformer
9. **`14_bahdanau_attention.ipynb`** - 神经机器翻译
   - 原始注意力机制
   - 带对齐的 Seq2seq
   - 注意力可视化

10. **`13_attention_is_all_you_need.ipynb`** - Transformer
    - 缩放点积注意力
    - 多头注意力
    - 位置编码
    - 现代 LLM 的基础

11. **`06_pointer_networks.ipynb`** - 指针网络
    - 注意力作为选择
    - 组合优化
    - 可变输出大小

12. **`08_seq2seq_for_sets.ipynb`** - 集合的 Seq2Seq
    - 排列不变集合编码器
    - 读-处理-写架构
    - 无序元素的注意力
    - 排序和集合操作
    - 比较：顺序敏感 vs 顺序不变

13. **`09_gpipe.ipynb`** - GPipe 流水线并行
    - 跨设备模型分区
    - 流水线利用的微批处理
    - F-then-B 调度（先全部前向，再全部反向）
    - 重新物化（梯度检查点）
    - 气泡时间分析
    - 训练大于单设备内存的模型

#### 高级主题
14. **`12_graph_neural_networks.ipynb`** - 图神经网络
    - 消息传递框架
    - 图卷积
    - 分子属性预测

15. **`16_relational_reasoning.ipynb`** - 关系网络
    - 成对关系推理
    - 视觉问答
    - 排列不变性

16. **`18_relational_rnn.ipynb`** - 关系 RNN
    - 带关系记忆的 LSTM
    - 跨记忆槽的多头自注意力
    - 架构演示（前向传播）
    - 顺序推理任务
    - **第 11 节：手动反向传播实现 (~1100 行)**
    - 所有组件的完整梯度计算
    - 带数值验证的梯度检查

17. **`20_neural_turing_machine.ipynb`** - 记忆增强网络
    - 内容和位置寻址
    - 可微分读/写
    - 外部记忆

18. **`21_ctc_speech.ipynb`** - CTC 损失与语音识别
    - 连接主义时序分类
    - 无对齐训练
    - 前向算法

#### 生成模型
19. **`17_variational_autoencoder.ipynb`** - VAE
    - 生成建模
    - ELBO 损失
    - 潜空间可视化

#### 现代应用
20. **`27_multi_token_prediction.ipynb`** - 多令牌预测
    - 预测多个未来令牌
    - 2-3 倍样本效率
    - 推测解码
    - 更快的训练与推理

21. **`28_dense_passage_retrieval.ipynb`** - 稠密检索
    - 双编码器架构
    - 批内负样本
    - 语义搜索

22. **`29_rag.ipynb`** - 检索增强生成
    - RAG-Sequence vs RAG-Token
    - 结合检索 + 生成
    - 知识 grounded 输出

23. **`30_lost_in_middle.ipynb`** - 长上下文分析
    - LLM 中的位置偏差
    - U 型性能曲线
    - 文档排序策略

#### 缩放与理论
24. **`22_scaling_laws.ipynb`** - 缩放定律
    - 幂律关系
    - 计算最优训练
    - 性能预测

25. **`23_mdl_principle.ipynb`** - 最小描述长度
    - 信息论模型选择
    - 压缩 = 理解
    - MDL vs AIC/BIC 比较
    - 神经网络架构选择
    - 基于 MDL 的剪枝（连接到论文 5）
    - Kolmogorov 复杂度预览

26. **`25_kolmogorov_complexity.ipynb`** - Kolmogorov 复杂度
    - K(x) = 生成 x 的最短程序
    - 随机性 = 不可压缩性
    - 算法概率 (Solomonoff)
    - 归纳的通用先验
    - 与 Shannon 熵的联系
    - 奥卡姆剃刀的形式化
    - ML 的理论基础

27. **`24_machine_super_intelligence.ipynb`** - 通用人工智能
    - **智能的正式理论 (Legg & Hutter)**
    - 心理测量 g 因子和通用智能 Υ(π)
    - 用于序列预测的 Solomonoff 归纳
    - AIXI：理论最优 RL 智能体
    - 蒙特卡洛 AIXI (MC-AIXI) 近似
    - Kolmogorov 复杂度估计
    - 跨环境的智能测量
    - 递归自我改进动力学
    - 智能爆炸场景
    - **6 个章节：从心理测量学到超级智能**
    - 连接论文 #23 (MDL)、#25 (Kolmogorov)、#8 (DQN)

28. **`01_complexity_dynamics.ipynb`** - 复杂度与熵
    - 元胞自动机 (Rule 30)
    - 熵增长
    - 不可逆性（基本介绍）

28. **`19_coffee_automaton.ipynb`** - 咖啡自动机（深度探索）
    - **不可逆性的全面探索**
    - 咖啡混合和扩散过程
    - 熵增长和粗粒化
    - 相空间和 Liouville 定理
    - Poincaré 递归定理（将在 e^N 时间后重新混合！）
    - 麦克斯韦妖和 Landauer 原理
    - 计算不可逆性（单向函数、哈希）
    - 机器学习中的信息瓶颈
    - 生物不可逆性（生命和第二定律）
    - 时间之箭：基本 vs 涌现
    - **10 个全面章节探索所有尺度的不可逆性**

29. **`26_cs231n_cnn_fundamentals.ipynb`** - CS231n：从第一原理的视觉
    - **纯 NumPy 的完整视觉流水线**
    - k-最近邻基线
    - 线性分类器（SVM 和 Softmax）
    - 优化（SGD、Momentum、Adam、学习率调度）
    - 带反向传播的 2 层神经网络
    - 卷积层（conv、pool、ReLU）
    - 完整 CNN 架构（Mini-AlexNet）
    - 可视化技术（滤波器、显著性图）
    - 迁移学习原理
    - 调参技巧（完整性检查、超参数调优、监控）
    - **10 个章节涵盖整个 CS231n 课程**
    - 连接论文 #7 (AlexNet)、#10 (ResNet)、#11 (扩张卷积)

## 仓库结构

```
sutskever-30-implementations/
├── README.md                           # 本文件
├── PROGRESS.md                         # 实现进度跟踪
├── IMPLEMENTATION_TRACKS.md            # 所有 30 篇论文的详细轨道
│
├── 01_complexity_dynamics.ipynb        # 熵与复杂度
├── 02_char_rnn_karpathy.ipynb         # 标准 RNN
├── 03_lstm_understanding.ipynb         # LSTM 门控
├── 04_rnn_regularization.ipynb         # RNN 的 Dropout
├── 05_neural_network_pruning.ipynb     # 剪枝与 MDL
├── 06_pointer_networks.ipynb           # 注意力指针
├── 07_alexnet_cnn.ipynb               # CNN 与 AlexNet
├── 08_seq2seq_for_sets.ipynb          # 排列不变集合
├── 09_gpipe.ipynb                     # 流水线并行
├── 10_resnet_deep_residual.ipynb      # 残差连接
├── 11_dilated_convolutions.ipynb       # 多尺度卷积
├── 12_graph_neural_networks.ipynb      # 消息传递 GNN
├── 13_attention_is_all_you_need.ipynb # Transformer 架构
├── 14_bahdanau_attention.ipynb         # 原始注意力
├── 15_identity_mappings_resnet.ipynb   # 预激活 ResNet
├── 16_relational_reasoning.ipynb       # 关系网络
├── 17_variational_autoencoder.ipynb   # VAE
├── 18_relational_rnn.ipynb             # 关系 RNN
├── 19_coffee_automaton.ipynb           # 不可逆性深度探索
├── 20_neural_turing_machine.ipynb     # 外部记忆
├── 21_ctc_speech.ipynb                # CTC 损失
├── 22_scaling_laws.ipynb              # 实证缩放
├── 23_mdl_principle.ipynb             # MDL 与压缩
├── 24_machine_super_intelligence.ipynb # 通用 AI 与 AIXI
├── 25_kolmogorov_complexity.ipynb     # K(x) 与随机性
├── 26_cs231n_cnn_fundamentals.ipynb    # 从第一原理的视觉
├── 27_multi_token_prediction.ipynb     # 多令牌预测
├── 28_dense_passage_retrieval.ipynb    # 稠密检索
├── 29_rag.ipynb                       # RAG 架构
└── 30_lost_in_middle.ipynb            # 长上下文分析
```

**所有 30 篇论文已实现！(100% 完成！) 🎉**

## 学习路径

### 初学者路径（从这里开始！）
1. **字符 RNN** (`02_char_rnn_karpathy.ipynb`) - 学习基础 RNN
2. **LSTM** (`03_lstm_understanding.ipynb`) - 理解门控机制
3. **CNN** (`07_alexnet_cnn.ipynb`) - 计算机视觉基础
4. **ResNet** (`10_resnet_deep_residual.ipynb`) - 跳跃连接
5. **VAE** (`17_variational_autoencoder.ipynb`) - 生成模型

### 中级路径
6. **RNN 正则化** (`04_rnn_regularization.ipynb`) - 更好的训练
7. **Bahdanau 注意力** (`14_bahdanau_attention.ipynb`) - 注意力基础
8. **指针网络** (`06_pointer_networks.ipynb`) - 注意力作为选择
9. **集合的 Seq2Seq** (`08_seq2seq_for_sets.ipynb`) - 排列不变性
10. **CS231n** (`26_cs231n_cnn_fundamentals.ipynb`) - 完整视觉流水线 (kNN → CNNs)
11. **GPipe** (`09_gpipe.ipynb`) - 大模型的流水线并行
12. **Transformer** (`13_attention_is_all_you_need.ipynb`) - 现代架构
13. **扩张卷积** (`11_dilated_convolutions.ipynb`) - 感受野
14. **缩放定律** (`22_scaling_laws.ipynb`) - 理解缩放

### 高级路径
15. **预激活 ResNet** (`15_identity_mappings_resnet.ipynb`) - 架构细节
16. **图神经网络** (`12_graph_neural_networks.ipynb`) - 图学习
17. **关系网络** (`16_relational_reasoning.ipynb`) - 关系推理
18. **神经图灵机** (`20_neural_turing_machine.ipynb`) - 外部记忆
19. **CTC 损失** (`21_ctc_speech.ipynb`) - 语音识别
20. **稠密检索** (`28_dense_passage_retrieval.ipynb`) - 语义搜索
21. **RAG** (`29_rag.ipynb`) - 检索增强生成
22. **迷失在中间** (`30_lost_in_middle.ipynb`) - 长上下文分析

### 理论与基础
23. **MDL 原理** (`23_mdl_principle.ipynb`) - 通过压缩进行模型选择
24. **Kolmogorov 复杂度** (`25_kolmogorov_complexity.ipynb`) - 随机性与信息
25. **复杂度动力学** (`01_complexity_dynamics.ipynb`) - 熵与涌现
26. **咖啡自动机** (`19_coffee_automaton.ipynb`) - 深入探索不可逆性

## Sutskever 30 的关键洞察

### 架构演进
- **RNN → LSTM**：门控解决梯度消失
- **普通网络 → ResNet**：跳跃连接实现深度
- **RNN → Transformer**：注意力实现并行化
- **固定词汇 → 指针**：输出可以引用输入

### 基本机制
- **注意力**：可微分选择机制
- **残差连接**：梯度高速公路
- **门控**：学习的信息流控制
- **外部记忆**：存储与计算分离

### 训练洞察
- **缩放定律**：性能随规模可预测改善
- **正则化**：Dropout、权重衰减、数据增强
- **优化**：梯度裁剪、学习率调度
- **计算最优**：平衡模型大小和训练数据

### 理论基础
- **信息论**：压缩、熵、MDL
- **复杂度**：Kolmogorov 复杂度、幂律
- **生成建模**：VAE、ELBO、潜空间
- **记忆**：可微分数据结构

## 实现哲学

### 为什么只用 NumPy？

这些实现刻意避免 PyTorch/TensorFlow 以：
- **加深理解**：看到框架抽象掉了什么
- **教育清晰度**：没有魔法，每个操作都显式
- **核心概念**：专注于算法，而非框架 API
- **可转移知识**：原则适用于任何框架

### 合成数据方法

每个 notebook 生成自己的数据以：
- **立即执行**：无需数据集下载
- **受控实验**：理解简单情况下的行为
- **概念聚焦**：数据不会掩盖算法
- **快速迭代**：修改并立即重新运行

## 扩展与下一步

### 在这些实现基础上构建

理解核心概念后，尝试：

1. **扩展规模**：用 PyTorch/JAX 为真实数据集实现
2. **组合技术**：例如，ResNet + 注意力
3. **现代变体**：
   - RNN → GRU → Transformer
   - VAE → β-VAE → VQ-VAE
   - ResNet → ResNeXt → EfficientNet
4. **应用**：应用于实际问题

### 研究方向

Sutskever 30 指向：
- 缩放（更大的模型、更多数据）
- 效率（稀疏模型、量化）
- 能力（推理、多模态）
- 理解（可解释性、理论）

## 资源

### 原始论文
参见 `IMPLEMENTATION_TRACKS.md` 获取完整引用和链接

### 补充阅读
- [Ilya Sutskever 阅读列表 (GitHub)](https://github.com/dzyim/ilya-sutskever-recommended-reading)
- [Aman 的 AI 期刊 - Sutskever 30 入门](https://aman.ai/primers/ai/top-30-papers/)
- [注释版 Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Andrej Karpathy 的博客](http://karpathy.github.io/)

### 课程
- Stanford CS231n：卷积神经网络
- Stanford CS224n：深度学习自然语言处理
- MIT 6.S191：深度学习导论

## 贡献

这些实现是教育性的，可以改进！考虑：
- 添加更多可视化
- 实现缺失的论文
- 改进解释
- 发现 bug
- 添加与框架实现的比较

## 引用

如果您在工作或教学中使用这些实现：

```bibtex
@misc{sutskever30implementations,
  title={Sutskever 30: Complete Implementation Suite},
  author={Paul "The Pageman" Pajo, pageman@gmail.com},
  year={2025},
  note={Educational implementations of Ilya Sutskever's recommended reading list, inspired by https://papercode.vercel.app/}
}
```

## 许可证

教育用途。原始研究引用请参阅各篇论文。

## 致谢
- **Ilya Sutskever**：策划了这个必要的阅读列表
- **论文作者**：他们的奠基性贡献
- **社区**：让这些想法变得可访问

---

## 最新添加（2025 年 12 月）

### 最近实现（21 篇新论文！）
- ✅ **论文 4**：RNN 正则化（变分 Dropout）
- ✅ **论文 5**：神经网络剪枝（MDL、90%+ 稀疏性）
- ✅ **论文 7**：AlexNet（从零实现 CNN）
- ✅ **论文 8**：集合的 Seq2Seq（排列不变性、注意力池化）
- ✅ **论文 9**：GPipe（流水线并行、微批处理、重新物化）
- ✅ **论文 19**：咖啡自动机（深入探索不可逆性、熵、Landauer 原理）
- ✅ **论文 26**：CS231n（完整视觉流水线：kNN → CNN，全部用 NumPy）
- ✅ **论文 11**：扩张卷积（多尺度）
- ✅ **论文 12**：图神经网络（消息传递）
- ✅ **论文 14**：Bahdanau 注意力（原始注意力）
- ✅ **论文 15**：恒等映射 ResNet（预激活）
- ✅ **论文 16**：关系推理（关系网络）
- ✅ **论文 18**：关系 RNN（关系记忆 + 第 11 节：手动反向传播 ~1100 行）
- ✅ **论文 21**：Deep Speech 2（CTC 损失）
- ✅ **论文 23**：MDL 原理（压缩、模型选择，连接到论文 5 & 25）
- ✅ **论文 24**：机器超级智能（通用 AI、AIXI、Solomonoff 归纳、智能度量、递归自我改进）
- ✅ **论文 25**：Kolmogorov 复杂度（随机性、算法概率、理论基础）
- ✅ **论文 27**：多令牌预测（2-3 倍样本效率）
- ✅ **论文 28**：稠密段落检索（双编码器）
- ✅ **论文 29**：RAG（检索增强生成）
- ✅ **论文 30**：迷失在中间（长上下文）

## 快速参考：实现复杂度

### 可以在一个下午实现
- ✅ 字符 RNN
- ✅ LSTM
- ✅ ResNet
- ✅ 简单 VAE
- ✅ 扩张卷积

### 周末项目
- ✅ Transformer
- ✅ 指针网络
- ✅ 图神经网络
- ✅ 关系网络
- ✅ 神经图灵机
- ✅ CTC 损失
- ✅ 稠密检索

### 一周深度探索
- ✅ 完整 RAG 系统
- ⚠️ 大规模实验
- ⚠️ 超参数优化

---

**"如果你真的学会了所有这些，你就会掌握今天 90% 的重要内容。"** - Ilya Sutskever

祝你学习愉快！🚀
