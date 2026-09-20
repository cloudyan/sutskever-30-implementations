# 学习路径与知识依赖图

_70 天数学+LLM 原理卡片体系的完整知识图谱_

---

## 📊 知识依赖图

```mermaid
graph TD
    subgraph S1[阶段 1：线性代数]
        D1[Day 1-2<br>向量矩阵] --> D3[Day 3-4<br>变换行列式]
        D3 --> D5[Day 5-7<br>特征值分解]
        D5 --> D8[Day 8-9<br>SVD PCA]
        D8 --> D10[Day 10<br>范数]
    end
    
    subgraph S2[阶段 2：微积分优化]
        D11[Day 11-12<br>导数梯度] --> D13[Day 13<br>链式法则]
        D13 --> D15[Day 15<br>梯度下降]
        D15 --> D20[Day 20<br>Adam]
    end
    
    subgraph S3[阶段 3：概率论]
        D21[Day 21-22<br>概率贝叶斯] --> D24[Day 24<br>分布]
        D24 --> D25[Day 25<br>期望方差]
        D25 --> D28[Day 28<br>中心极限]
    end
    
    subgraph S4[阶段 4：统计推断]
        D31[Day 31<br>MLE] --> D32[Day 32<br>MAP]
        D31 --> D47[Day 47<br>最小二乘]
        D37[Day 37<br>过拟合] --> D38[Day 38<br>正则化]
    end
    
    subgraph S5[阶段 5：信息论]
        D42[Day 42<br>熵] --> D43[Day 43<br>交叉熵]
        D43 --> D44[Day 44<br>KL 散度]
        D44 --> D55[Day 55<br>变分推断]
    end
    
    subgraph S6[阶段 6：综合应用]
        D46[Day 46<br>softmax] --> D56[Day 56<br>注意力]
        D50[Day 50<br>EM] --> D55
        D56 --> D57[Day 57<br>位置编码]
    end
    
    subgraph S7[阶段 7：LLM 原理]
        D56 --> D63[Day 63<br>Q-K-V 详解]
        D61[Day 61<br>参数记忆] --> D66[Day 66<br>微调 LoRA]
        D62[Day 62<br>上下文记忆] --> D68[Day 68<br>少样本学习]
        D63 --> D64[Day 64<br>Prompt 工程]
        D65[Day 65<br>RAG] --> D64
        D67[Day 67<br>灾难性遗忘] --> D66
        D69[Day 69<br>链式思维] --> D70[Day 70<br>Prompt 实战]
    end
    
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S5 --> S6
    S6 --> S7
    
    D10 -.-> D38
    D31 -.-> D47
    D43 -.-> D46
    D56 -.-> D63
```

---

## 📚 学习路径建议

### 🎯 标准路径（70 天）

| 阶段 | Day 范围 | 天数 | 每天投入 | 总时长 |
|------|---------|------|---------|--------|
| 阶段 1 | Day 1-10 | 10 天 | 30-60 分钟 | 5-10 小时 |
| 阶段 2 | Day 11-20 | 10 天 | 30-60 分钟 | 5-10 小时 |
| 阶段 3 | Day 21-30 | 10 天 | 30-60 分钟 | 5-10 小时 |
| 阶段 4 | Day 31-40 | 10 天 | 30-60 分钟 | 5-10 小时 |
| 阶段 5 | Day 41-45 | 5 天 | 30-60 分钟 | 2.5-5 小时 |
| 阶段 6 | Day 46-60 | 15 天 | 30-60 分钟 | 7.5-15 小时 |
| 阶段 7 | Day 61-70 | 10 天 | 30-60 分钟 | 5-10 小时 |

**总计**：70 天，35-55 小时

---

### 🚀 加速路径（35 天）

每 2 天学习一个知识点，适合有基础的学习者。

---

### 🐢 深入路径（105 天）

每知识点 1.5 天，额外时间用于推导练习和代码实现。

---

## 🔗 关键依赖关系

| 知识点 | 前置依赖 | 后续应用 |
|--------|---------|---------|
| **Day 38 正则化** | Day 10 范数、Day 37 过拟合 | Day 59 初始化、Day 67 灾难性遗忘 |
| **Day 46 softmax** | Day 43 交叉熵、Day 13 链式法则 | Day 56 注意力 |
| **Day 47 最小二乘** | Day 31 MLE、Day 24 高斯分布 | Day 58 归一化 |
| **Day 50 EM 算法** | Day 31 MLE、Day 29 联合分布 | Day 55 变分推断 |
| **Day 55 变分推断** | Day 44 KL 散度、Day 50 EM | - |
| **Day 56 注意力** | Day 46 softmax、Day 1-2 向量矩阵 | Day 57 位置编码、Day 63 Q-K-V |
| **Day 63 Q-K-V** | Day 56 注意力、Day 1 向量点积 | Day 64 Prompt 工程 |
| **Day 64 Prompt** | Day 63 Q-K-V、Day 42 熵 | Day 70 Prompt 实战 |
| **Day 65 RAG** | Day 1 向量相似度、Day 9 PCA | Day 64 Prompt 工程 |
| **Day 66 LoRA** | Day 15 梯度下降、Day 5 伪逆 | Day 67 灾难性遗忘 |

---

## 📖 推荐教材

### 综合教材
- 《深度学习》花书 - 第 3-6 章
- 《Pattern Recognition and Machine Learning》Bishop

### 线性代数
- 《Introduction to Linear Algebra》Gilbert Strang
- 3Blue1Brown 线性代数本质

### 微积分与优化
- 《Convex Optimization》Stephen Boyd

### 概率与统计
- 《概率论与数理统计》陈希孺
- 《All of Statistics》Larry Wasserman

### 信息论
- 《Elements of Information Theory》Cover

### LLM 原理
- 《Attention Is All You Need》Transformer 原论文
- 《Language Models are Few-Shot Learners》GPT-3 论文

---

## 💡 学习建议

<div class="callout callout-tip">

✅ **应该做的**

1. 按顺序学习，不要跳跃
2. 理解优先，不要死记硬背
3. 动手推导关键公式
4. 联系 AI 应用场景
5. 间隔复习

</div>

<div class="callout callout-warning">

❌ **应该避免的**

1. 前面没理解就往后看
2. 只看不练
3. 追求完美
4. 孤立学习
5. 急于求成

</div>

---

*最后更新：2026-04-22*
