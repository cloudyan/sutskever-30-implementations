# AI 专家成长路线图

_从数学基础到 AI 前沿研究的完整知识体系_

---

## 🎯 培养目标

培养具备以下能力的 AI 专家：

1. **扎实的理论基础** - 数学、算法、信息论
2. **深入的核心技术** - 机器学习、深度学习、Transformer、LLM
3. **工程实践能力** - 代码实现、系统优化、部署
4. **研究创新能力** - 论文阅读、实验设计、创新思维
5. **行业洞察力** - 技术趋势、产品化、创业思维

---

## 📚 完整知识体系

### 阶段 1：数学基础（1-30 天）

**目标**：掌握 AI 核心数学工具

```
01-math-foundation/
├── README.md                    # 阶段概览
├── 01-linear-algebra/           # 线性代数
│   ├── 01-vector.md             # 向量与向量空间
│   ├── 02-matrix.md             # 矩阵与矩阵运算
│   ├── 03-linear-transform.md   # 线性变换
│   ├── 04-determinant.md        # 行列式
│   ├── 05-inverse-pseudoinverse.md  # 逆矩阵与伪逆
│   ├── 06-eigenvalue.md         # 特征值与特征向量
│   ├── 07-eigendecomposition.md # 特征分解
│   ├── 08-svd.md                # 奇异值分解
│   ├── 09-pca.md                # 主成分分析
│   └── 10-norm.md               # 范数与距离
├── 02-calculus-optimization/    # 微积分与优化
│   ├── 01-derivative.md         # 导数与偏导数
│   ├── 02-gradient.md           # 梯度
│   ├── 03-chain-rule.md         # 链式法则
│   ├── 04-taylor.md             # 泰勒展开
│   ├── 05-gradient-descent.md   # 梯度下降
│   ├── 06-newton.md             # 牛顿法
│   ├── 07-lagrange.md           # 拉格朗日乘数法
│   ├── 08-convex.md             # 凸优化
│   ├── 09-sgd-momentum.md       # SGD 与动量
│   └── 10-adam.md               # Adam 优化器
├── 03-probability/              # 概率论
│   ├── 01-probability.md        # 概率与条件概率
│   ├── 02-bayes.md              # 贝叶斯定理
│   ├── 03-random-variable.md    # 随机变量
│   ├── 04-distribution.md       # 概率分布
│   ├── 05-expectation.md        # 期望与方差
│   ├── 06-covariance.md         # 协方差
│   ├── 07-lln.md                # 大数定律
│   ├── 08-clt.md                # 中心极限定理
│   ├── 09-joint-distribution.md # 联合分布
│   └── 10-conditional-independence.md # 条件独立
├── 04-statistics/               # 统计推断
│   ├── 01-mle.md                # 最大似然估计
│   ├── 02-map.md                # 最大后验估计
│   ├── 03-bayesian-inference.md # 贝叶斯推断
│   ├── 04-hypothesis-testing.md # 假设检验
│   ├── 05-confidence-interval.md # 置信区间
│   ├── 06-regression.md         # 回归分析
│   ├── 07-overfitting.md        # 过拟合与欠拟合
│   ├── 08-regularization.md     # 正则化
│   ├── 09-cross-validation.md   # 交叉验证
│   └── 10-bootstrap.md          # Bootstrap 方法
├── 05-information-theory/       # 信息论
│   ├── 01-self-information.md   # 信息量与自信息
│   ├── 02-entropy.md            # 熵
│   ├── 03-cross-entropy.md      # 交叉熵
│   ├── 04-kl-divergence.md      # KL 散度
│   └── 05-mutual-information.md # 互信息
└── 06-llm-math/                 # LLM 数学原理
    ├── 01-softmax.md            # Softmax 详解
    ├── 02-attention-math.md     # 注意力的数学
    ├── 03-positional-encoding.md # 位置编码
    ├── 04-normalization.md      # 归一化
    ├── 05-initialization.md     # 初始化
    ├── 06-learning-rate.md      # 学习率调度
    ├── 07-parameter-memory.md   # 参数记忆
    ├── 08-context-memory.md     # 上下文记忆
    └── 09-rag.md                # RAG 原理
```

**配套资源**：
- ✅ [AI 数学基础 70 讲](https://cloudyan.github.io/ai-math-docs/) - 完整视频教程
- 📝 每节配套代码实现（NumPy）
- 📝 每节配套习题与解答

---

### 阶段 2：Python 编程基础（7-14 天）

**目标**：掌握 AI 开发的编程工具

```
02-python-basics/
├── README.md
├── 01-python-core/              # Python 核心
│   ├── 01-basic-syntax.md       # 基础语法
│   ├── 02-data-structures.md    # 数据结构（列表、字典、集合）
│   ├── 03-functions.md          # 函数与 lambda
│   ├── 04-oop.md                # 面向对象编程
│   ├── 05-modules.md            # 模块与包
│   ├── 06-exception.md          # 异常处理
│   └── 07-iterator-generator.md # 迭代器与生成器
├── 02-numpy/                    # NumPy
│   ├── 01-array.md              # 数组基础
│   ├── 02-indexing.md           # 索引与切片
│   ├── 03-broadcasting.md       # 广播机制
│   ├── 04-linear-algebra.md     # 线性代数运算
│   ├── 05-statistics.md         # 统计运算
│   └── 06-performance.md        # 性能优化
├── 03-pandas/                   # Pandas
│   ├── 01-series-dataframe.md   # Series 与 DataFrame
│   ├── 02-data-cleaning.md      # 数据清洗
│   ├── 03-data-analysis.md      # 数据分析
│   └── 04-time-series.md        # 时间序列
├── 04-matplotlib-seaborn/       # 可视化
│   ├── 01-basic-plot.md         # 基础图表
│   ├── 02-advanced-plot.md      # 高级图表
│   └── 03-visualization-best-practices.md # 最佳实践
└── 05-project/                  # 实战项目
    ├── 01-data-analysis.md      # 数据分析项目
    └── 02-visualization.md      # 可视化项目
```

**配套资源**：
- 📝 每节配套代码示例
- 📝 Jupyter Notebook 练习
- 📝 实战项目代码

---

### 阶段 3：机器学习基础（30-45 天）

**目标**：掌握经典机器学习算法

```
03-machine-learning/
├── README.md
├── 01-introduction/             # 入门
│   ├── 01-what-is-ml.md         # 什么是机器学习
│   ├── 02-ml-types.md           # 机器学习类型
│   ├── 03-ml-workflow.md        # 机器学习工作流程
│   └── 04-evaluation-metrics.md # 评估指标
├── 02-supervised-learning/      # 监督学习
│   ├── 01-linear-regression.md  # 线性回归
│   ├── 02-logistic-regression.md # 逻辑回归
│   ├── 03-decision-tree.md      # 决策树
│   ├── 04-random-forest.md      # 随机森林
│   ├── 05-svm.md                # 支持向量机
│   ├── 06-knn.md                # K 近邻
│   ├── 07-naive-bayes.md        # 朴素贝叶斯
│   └── 08-ensemble.md           # 集成学习
├── 03-unsupervised-learning/    # 无监督学习
│   ├── 01-clustering.md         # 聚类
│   ├── 02-kmeans.md             # K-Means
│   ├── 03-hierarchical.md       # 层次聚类
│   ├── 04-dbscan.md             # DBSCAN
│   ├── 05-pca.md                # 主成分分析
│   ├── 06-autoencoder.md        # 自编码器
│   └── 07-gmm.md                # 高斯混合模型
├── 04-feature-engineering/      # 特征工程
│   ├── 01-feature-selection.md  # 特征选择
│   ├── 02-feature-extraction.md # 特征提取
│   ├── 03-feature-scaling.md    # 特征缩放
│   └── 04-handling-missing.md   # 处理缺失值
├── 05-model-selection/          # 模型选择
│   ├── 01-cross-validation.md   # 交叉验证
│   ├── 02-hyperparameter.md     # 超参数调优
│   ├── 03-grid-search.md        # 网格搜索
│   └── 04-random-search.md      # 随机搜索
└── 06-projects/                 # 实战项目
    ├── 01-house-price.md        # 房价预测
    ├── 02-classification.md     # 分类问题
    └── 03-clustering.md         # 聚类分析
```

**配套资源**：
- 📝 scikit-learn 代码实现
- 📝 算法可视化演示
- 📝 实战项目代码
- 📖 配套论文：Decision Trees, Random Forests, SVM

---

### 阶段 4：深度学习（45-60 天）

**目标**：深入理解神经网络原理与实现

```
04-deep-learning/
├── README.md
├── 01-neural-network-basics/    # 神经网络基础
│   ├── 01-perceptron.md         # 感知机
│   ├── 02-mlp.md                # 多层感知机
│   ├── 03-activation.md         # 激活函数
│   ├── 04-loss-function.md      # 损失函数
│   ├── 05-backpropagation.md    # 反向传播
│   └── 06-optimization.md       # 优化算法
├── 02-training-techniques/      # 训练技巧
│   ├── 01-initialization.md     # 权重初始化
│   ├── 02-batch-norm.md         # 批量归一化
│   ├── 03-dropout.md            # Dropout
│   ├── 04-regularization.md     # 正则化
│   ├── 05-learning-rate.md      # 学习率调整
│   └── 06-early-stopping.md     # 早停
├── 03-cnn/                      # 卷积神经网络
│   ├── 01-convolution.md        # 卷积操作
│   ├── 02-pooling.md            # 池化
│   ├── 03-architecture.md       # 经典架构
│   ├── 04-alexnet.md            # AlexNet
│   ├── 05-vgg.md                # VGG
│   ├── 06-resnet.md             # ResNet
│   ├── 07-inception.md          # Inception
│   └── 08-transfer-learning.md  # 迁移学习
├── 04-rnn/                      # 循环神经网络
│   ├── 01-rnn-basics.md         # RNN 基础
│   ├── 02-lstm.md               # LSTM
│   ├── 03-gru.md                # GRU
│   ├── 04-bidir-rnn.md          # 双向 RNN
│   ├── 05-seq2seq.md            # Seq2Seq
│   └── 06-attention.md          # 注意力机制
├── 05-gan/                      # 生成对抗网络
│   ├── 01-gan-basics.md         # GAN 基础
│   ├── 02-dcgan.md              # DCGAN
│   ├── 03-cgan.md               # CGAN
│   └── 04-wgan.md               # WGAN
├── 06-projects/                 # 实战项目
│   ├── 01-image-classification.md # 图像分类
│   ├── 02-object-detection.md   # 目标检测
│   └── 03-style-transfer.md     # 风格迁移
└── 07-papers/                   # 论文解读
    ├── 01-alexnet.md            # AlexNet 论文解读
    ├── 02-resnet.md             # ResNet 论文解读
    ├── 03-lstm.md               # LSTM 论文解读
    └── 04-gan.md                # GAN 论文解读
```

**配套资源**：
- 📝 PyTorch 代码实现
- 📝 网络架构可视化
- 📝 实战项目代码
- 📖 配套论文：AlexNet, VGG, ResNet, LSTM, GAN

---

### 阶段 5：Transformer 架构（30-45 天）⭐

**目标**：深入理解现代 AI 的核心架构

```
05-transformer/
├── README.md
├── 01-attention-mechanism/      # 注意力机制
│   ├── 01-introduction.md       # 注意力简介
│   ├── 02-self-attention.md     # 自注意力
│   ├── 03-multi-head.md         # 多头注意力
│   ├── 04-math-detail.md        # 数学详解
│   ├── 05-visualization.md      # 可视化
│   └── 06-code-implementation.md # 代码实现
├── 02-transformer-architecture/ # Transformer 架构
│   ├── 01-encoder.md            # 编码器
│   ├── 02-decoder.md            # 解码器
│   ├── 03-positional-encoding.md # 位置编码
│   ├── 04-layer-norm.md         # 层归一化
│   ├── 05-ffn.md                # 前馈网络
│   └── 06-complete-architecture.md # 完整架构
├── 03-implementation/           # 从零实现
│   ├── 01-pytorch-implementation.md # PyTorch 实现
│   ├── 02-training.md           # 训练
│   ├── 03-debugging.md          # 调试
│   └── 04-optimization.md       # 优化
├── 04-variants/                 # 变体
│   ├── 01-bert.md               # BERT
│   ├── 02-gpt.md                # GPT
│   ├── 03-t5.md                 # T5
│   ├── 04-albert.md             # ALBERT
│   ├── 05-roberta.md            # RoBERTa
│   └── 06-vit.md                # Vision Transformer
├── 05-papers/                   # 论文解读
│   ├── 01-attention-is-all-you-need.md # Transformer 原论文
│   ├── 02-bert.md               # BERT 论文
│   ├── 03-gpt.md                # GPT 论文
│   ├── 04-vit.md                # ViT 论文
│   └── 05-roformer.md           # RoFormer 论文解读
└── 06-projects/                 # 实战项目
    ├── 01-machine-translation.md # 机器翻译
    ├── 02-text-classification.md # 文本分类
    └── 03-image-classification.md # 图像分类（ViT）
```

**配套资源**：
- 📝 Transformer 从零实现代码
- 📝 注意力可视化
- 📝 论文精读笔记
- 📖 配套论文：Attention Is All You Need, BERT, GPT, ViT

---

### 阶段 6：LLM 应用（30-45 天）

**目标**：掌握大语言模型的应用与开发

```
06-llm-application/
├── README.md
├── 01-llm-basics/               # LLM 基础
│   ├── 01-what-is-llm.md        # 什么是 LLM
│   ├── 02-llm-architecture.md   # LLM 架构
│   ├── 03-llm-training.md       # LLM 训练
│   ├── 04-llm-inference.md      # LLM 推理
│   └── 05-llm-evaluation.md     # LLM 评估
├── 02-prompt-engineering/       # Prompt 工程
│   ├── 01-introduction.md       # Prompt 简介
│   ├── 02-basic-techniques.md   # 基础技巧
│   ├── 03-few-shot.md           # Few-Shot
│   ├── 04-cot.md                # Chain of Thought
│   ├── 05-advanced.md           # 高级技巧
│   └── 06-best-practices.md     # 最佳实践
├── 03-rag/                      # RAG 检索增强生成
│   ├── 01-introduction.md       # RAG 简介
│   ├── 02-embedding.md          # Embedding
│   ├── 03-vector-database.md    # 向量数据库
│   ├── 04-retrieval.md          # 检索
│   ├── 05-generation.md         # 生成
│   └── 06-optimization.md       # 优化
├── 04-fine-tuning/              # 微调
│   ├── 01-introduction.md       # 微调简介
│   ├── 02-full-finetuning.md    # 全量微调
│   ├── 03-lora.md               # LoRA
│   ├── 04-qlora.md              # QLoRA
│   ├── 05-rlhf.md               # RLHF
│   └── 06-dpo.md                # DPO
├── 05-agent/                    # Agent
│   ├── 01-introduction.md       # Agent 简介
│   ├── 02-tool-use.md           # 工具使用
│   ├── 03-planning.md           # 规划
│   ├── 04-memory.md             # 记忆
│   └── 05-multi-agent.md        # 多 Agent
├── 06-application/              # 应用开发
│   ├── 01-chatbot.md            # 聊天机器人
│   ├── 02-qa-system.md          # 问答系统
│   ├── 03-code-assistant.md     # 代码助手
│   └── 04-knowledge-base.md     # 知识库
└── 07-papers/                   # 论文解读
    ├── 01-gpt3.md               # GPT-3 论文
    ├── 02-llama.md              # LLaMA 论文
    ├── 03-rlhf.md               # RLHF 论文
    └── 04-rag.md                # RAG 论文
```

**配套资源**：
- 📝 Prompt 工程实战
- 📝 RAG 系统实现
- 📝 微调代码示例
- 📖 配套论文：GPT-3, LLaMA, RLHF

---

### 阶段 7：AI Agent（30-45 天）

**目标**：掌握智能体系统的设计与实现

```
07-ai-agent/
├── README.md
├── 01-introduction/             # 入门
│   ├── 01-what-is-agent.md      # 什么是 Agent
│   ├── 02-agent-architecture.md # Agent 架构
│   ├── 03-agent-types.md        # Agent 类型
│   └── 04-agent-history.md      # 发展历史
├── 02-core-components/          # 核心组件
│   ├── 01-planning.md           # 规划
│   ├── 02-memory.md             # 记忆
│   ├── 03-tool-use.md           # 工具使用
│   ├── 04-reflection.md         # 反思
│   └── 05-collaboration.md      # 协作
├── 03-frameworks/               # 框架
│   ├── 01-langchain.md          # LangChain
│   ├── 02-auto-gen.md           # AutoGen
│   ├── 03-crew-ai.md            # CrewAI
│   └── 04-litellm.md            # LiteLLM
├── 04-advanced-topics/          # 高级主题
│   ├── 01-multi-agent.md        # 多 Agent
│   ├── 02-agent-swarm.md        # Agent 群体
│   ├── 03-human-in-loop.md      # 人在回路
│   └── 04-agent-evaluation.md   # Agent 评估
├── 05-projects/                 # 实战项目
│   ├── 01-research-assistant.md # 研究助手
│   ├── 02-coding-agent.md       # 编程 Agent
│   └── 03-business-agent.md     # 商业 Agent
└── 06-papers/                   # 论文解读
    ├── 01-react.md              # ReAct 论文
    ├── 02-cot.md                # CoT 论文
    └── 03-agent-survey.md       # Agent 综述
```

---

### 阶段 8：系统优化（30 天）

**目标**：掌握模型优化与部署技术

```
08-system-optimization/
├── README.md
├── 01-model-compression/        # 模型压缩
│   ├── 01-pruning.md            # 剪枝
│   ├── 02-quantization.md       # 量化
│   ├── 03-distillation.md       # 蒸馏
│   └── 04-low-rank.md           # 低秩分解
├── 02-inference-optimization/   # 推理优化
│   ├── 01-batch-inference.md    # 批量推理
│   ├── 02-kv-cache.md           # KV Cache
│   ├── 03-speculative.md        # 推测解码
│   └── 04-continuous-batching.md # 连续批处理
├── 03-hardware/                 # 硬件
│   ├── 01-gpu.md                # GPU
│   ├── 02-tpu.md                # TPU
│   ├── 03-npu.md                # NPU
│   └── 04-edge-device.md        # 边缘设备
├── 04-deployment/               # 部署
│   ├── 01-onnx.md               # ONNX
│   ├── 02-tensorrt.md           # TensorRT
│   ├── 03-openvino.md           # OpenVINO
│   └── 04-tflite.md             # TFLite
└── 05-monitoring/               # 监控
    ├── 01-performance.md        # 性能监控
    ├── 02-drift.md              # 数据漂移
    └── 03-logging.md            # 日志
```

---

### 阶段 9：高级架构（30 天）

**目标**：了解前沿架构与研究

```
09-advanced-architecture/
├── README.md
├── 01-moe/                      # MoE 混合专家
│   ├── 01-introduction.md       # MoE 简介
│   ├── 02-switch-transformer.md # Switch Transformer
│   └── 03-mixtral.md            # Mixtral
├── 02-multimodal/               # 多模态
│   ├── 01-clip.md               # CLIP
│   ├── 02-flamingo.md           # Flamingo
│   ├── 03-lvm.md                # LVM
│   └── 04-sora.md               # Sora
├── 03-world-model/              # 世界模型
│   ├── 01-introduction.md       # 世界模型简介
│   ├── 02-jePA.md               # JePA
│   └── 03-genie.md              # Genie
├── 04-efficient-architecture/   # 高效架构
│   ├── 01-mamba.md              # Mamba
│   ├── 02-rwkv.md               # RWKV
│   └── 03-hyena.md              # Hyena
└── 05-papers/                   # 论文解读
    ├── 01-moe.md                # MoE 论文
    ├── 02-clip.md               # CLIP 论文
    ├── 03-mamba.md              # Mamba 论文
    └── 04-sora.md               # Sora 论文
```

---

### 阶段 10：研究与创新（持续）

**目标**：掌握研究方法与创新能力

```
10-research/
├── README.md
├── 01-paper-reading/            # 论文阅读
│   ├── 01-how-to-read.md        # 如何读论文
│   ├── 02-note-taking.md        # 笔记方法
│   ├── 03-paper-management.md   # 论文管理
│   └── 04-key-papers.md         # 关键论文列表
├── 02-experiment-design/        # 实验设计
│   ├── 01-hypothesis.md         # 假设形成
│   ├── 02-ab-test.md            # A/B 测试
│   ├── 03-statistical-significance.md # 统计显著性
│   └── 04-reproducibility.md    # 可复现性
├── 03-writing/                  # 论文写作
│   ├── 01-structure.md          # 论文结构
│   ├── 02-latex.md              # LaTeX
│   ├── 03-figures.md            # 图表制作
│   └── 04-review.md             # 审稿回复
├── 04-innovation/               # 创新思维
│   ├── 01-idea-generation.md    # 创意产生
│   ├── 02-literature-review.md  # 文献综述
│   └── 03-research-gap.md       # 研究空白
└── 05-career/                   # 职业发展
    ├── 01-phd.md                # 博士申请
    ├── 02-industry-research.md  # 工业界研究
    └── 03-academic-career.md    # 学术生涯
```

---

### 阶段 11：工程实践（30 天）

**目标**：掌握工程化与部署技能

```
11-engineering/
├── README.md
├── 01-mlops/                    # MLOps
│   ├── 01-introduction.md       # MLOps 简介
│   ├── 02-ci-cd.md              # CI/CD
│   ├── 03-model-registry.md     # 模型注册
│   └── 04-monitoring.md         # 监控
├── 02-deployment/               # 部署
│   ├── 01-api-design.md         # API 设计
│   ├── 02-scaling.md            # 扩展
│   ├── 03-container.md          # 容器化
│   └── 04-kubernetes.md         # Kubernetes
├── 03-best-practices/           # 最佳实践
│   ├── 01-code-quality.md       # 代码质量
│   ├── 02-testing.md            # 测试
│   ├── 03-documentation.md      # 文档
│   └── 04-security.md           # 安全
└── 04-tools/                    # 工具
    ├── 01-git.md                # Git
    ├── 02-docker.md             # Docker
    ├── 03-vscode.md             # VSCode
    └── 04-jupyter.md            # Jupyter
```

---

### 阶段 12：创新与创业（持续）

**目标**：从技术到产品的思维转变

```
12-innovation/
├── README.md
├── 01-product-thinking/         # 产品思维
│   ├── 01-user-need.md          # 用户需求
│   ├── 02-mvp.md                # MVP
│   ├── 03-product-market-fit.md # PMF
│   └── 04-iteration.md          # 迭代
├── 02-startup/                  # 创业
│   ├── 01-idea-validation.md    # 创意验证
│   ├── 02-team-building.md      # 团队建设
│   ├── 03-funding.md            # 融资
│   └── 04-growth.md             # 增长
├── 03-industry/                 # 行业
│   ├── 01-trends.md             # 趋势
│   ├── 02-competitive-analysis.md # 竞品分析
│   └── 03-business-model.md     # 商业模式
└── 04-case-studies/             # 案例研究
    ├── 01-openai.md             # OpenAI
    ├── 02-anthropic.md          # Anthropic
    ├── 03-google-deepmind.md    # Google DeepMind
    └── 04-china-ai.md           # 中国 AI 公司
```

---

## 🎯 学习路径建议

### 不同背景的学习者

| 背景 | 建议起点 | 重点阶段 | 预计时间 |
|------|---------|---------|---------|
| **零基础** | 阶段 1（数学） | 1→2→3→4→5 | 12-18 个月 |
| **数学/统计专业** | 阶段 2（Python） | 2→3→4→5→6 | 9-12 个月 |
| **计算机专业** | 阶段 3（ML） | 3→4→5→6→7 | 6-9 个月 |
| **已有 AI 经验** | 阶段 5（Transformer） | 5→6→7→9→10 | 3-6 个月 |
| **研究者** | 阶段 10（研究） | 10→5→6→9 | 持续学习 |

### 推荐学习顺序

```
阶段 1（数学）→ 阶段 2（Python）→ 阶段 3（ML）→ 阶段 4（DL）
                                              ↓
    阶段 12（创新）← 阶段 11（工程）← 阶段 10（研究）← 阶段 9（高级架构）
                                              ↑
阶段 5（Transformer）→ 阶段 6（LLM）→ 阶段 7（Agent）→ 阶段 8（优化）
```

---

## 📖 配套资源

### 在线课程

- [AI 数学基础 70 讲](https://cloudyan.github.io/ai-math-docs/) - 前置课程

### 代码仓库

- 每个章节配套代码实现
- 从论文到代码的完整复现
- 实战项目代码

### 论文列表

每个阶段配套关键论文，详见各章节 `papers/` 目录。

### 社区

- GitHub Discussions
- 学习群组
- 代码 Review

---

*最后更新：2026-04-22*
*创建者：悟空 AI 助手*
