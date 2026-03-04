# 注意力机制如何让翻译更准确?

问下大家,有没有想过早期的机器翻译系统是怎么工作的?为什么以前翻译质量那么差?

晓寒刚开始学 NLP 的时候,就听说过 Seq2Seq 模型,它把整个句子压缩成一个固定长度的向量,然后生成翻译。但问题是,句子一长,这个向量就记不住了!

直到 2014 年,Bahdanau 等人提出了注意力机制,才发现卧槽,原来可以让模型"看着原文"来翻译,而不是只靠一个压缩的向量!

## Seq2Seq 的问题

### 固定长度瓶颈

```
传统 Seq2Seq:

编码器: "我爱北京天安门" → [固定长度向量 c]
解码器: [c] → "I love Beijing Tiananmen"

问题:
1. 长句子压缩到一个向量,信息丢失严重
2. 解码时看不到原文,只能靠记忆
3. 句子越长,翻译质量越差

就像让学生读完一篇文章后立即复述,
学生可能只记得大概意思,细节都忘了!
```

### 需要解决的问题

```
翻译时的真实情况:

翻译"I love"时,应该关注"我"
翻译"Beijing"时,应该关注"北京"
翻译"Tiananmen"时,应该关注"天安门"

不同时刻需要关注原文的不同部分!
```

## Bahdanau 注意力机制

### 核心思想

**让解码器在生成每个词时,都能"回头看"原文的不同部分!**

```
结构:

编码器: 双向 RNN,生成所有位置的表示 h_1, h_2, ..., h_T

解码器: 每个时间步:
1. 计算当前状态与所有编码器状态的"相关性"
2. 用相关性加权求和得到上下文向量
3. 结合上下文向量生成输出
```

### 数学公式

**1. 注意力分数**:

$$e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$$

**2. 注意力权重**:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

**3. 上下文向量**:

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

其中:
- $s_{i-1}$: 前一个解码器状态
- $h_j$: 编码器第 j 个位置的表示
- $c_i$: 当前时刻的上下文向量

## 用 NumPy 实现

### 双向 RNN 编码器

```python
import numpy as np

def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class EncoderRNN:
    """双向 RNN 编码器"""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # 前向 RNN
        self.W_fwd = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_fwd = np.zeros((hidden_size, 1))
        
        # 后向 RNN
        self.W_bwd = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_bwd = np.zeros((hidden_size, 1))
    
    def forward(self, inputs):
        """
        参数:
            inputs: 输入序列,每个元素形状 (input_size, 1)
        
        返回:
            annotations: 双向隐藏状态列表,每个形状 (2*hidden_size, 1)
        """
        seq_len = len(inputs)
        
        # 前向传播
        h_fwd = []
        h = np.zeros((self.hidden_size, 1))
        for x in inputs:
            concat = np.vstack([x, h])
            h = np.tanh(np.dot(self.W_fwd, concat) + self.b_fwd)
            h_fwd.append(h)
        
        # 后向传播
        h_bwd = []
        h = np.zeros((self.hidden_size, 1))
        for x in reversed(inputs):
            concat = np.vstack([x, h])
            h = np.tanh(np.dot(self.W_bwd, concat) + self.b_bwd)
            h_bwd.append(h)
        h_bwd = list(reversed(h_bwd))
        
        # 拼接前向和后向
        annotations = [np.vstack([h_f, h_b]) for h_f, h_b in zip(h_fwd, h_bwd)]
        
        return annotations

print("双向编码器创建完成")
```

### Bahdanau 注意力

```python
class BahdanauAttention:
    """
    Bahdanau 注意力机制 (加性注意力)
    
    公式: e_ij = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)
    """
    def __init__(self, hidden_size, annotation_size):
        self.hidden_size = hidden_size
        
        # 注意力参数
        self.W_a = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_a = np.random.randn(hidden_size, annotation_size) * 0.01
        self.v_a = np.random.randn(1, hidden_size) * 0.01
    
    def forward(self, decoder_hidden, encoder_annotations):
        """
        参数:
            decoder_hidden: 解码器当前状态 (hidden_size, 1)
            encoder_annotations: 编码器所有状态列表
        
        返回:
            context: 上下文向量 (annotation_size, 1)
            attention_weights: 注意力权重
        """
        scores = []
        
        # 计算每个位置的注意力分数
        for h_j in encoder_annotations:
            # e_ij = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)
            score = np.dot(self.v_a, np.tanh(
                np.dot(self.W_a, decoder_hidden) + 
                np.dot(self.U_a, h_j)
            ))
            scores.append(score[0, 0])
        
        # Softmax 得到注意力权重
        scores = np.array(scores)
        attention_weights = softmax(scores)
        
        # 加权求和得到上下文向量
        context = sum(alpha * h for alpha, h in zip(attention_weights, encoder_annotations))
        
        return context, attention_weights

print("Bahdanau 注意力机制创建完成")
```

### 带注意力的解码器

```python
class AttentionDecoder:
    """带注意力的 RNN 解码器"""
    def __init__(self, output_size, hidden_size, annotation_size):
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 注意力机制
        self.attention = BahdanauAttention(hidden_size, annotation_size)
        
        # RNN: 输入 = 前一个输出 + 上下文向量
        input_size = output_size + annotation_size
        self.W_dec = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.b_dec = np.zeros((hidden_size, 1))
        
        # 输出层
        self.W_out = np.random.randn(output_size, hidden_size + annotation_size + output_size) * 0.01
        self.b_out = np.zeros((output_size, 1))
    
    def step(self, prev_output, decoder_hidden, encoder_annotations):
        """
        单步解码
        
        参数:
            prev_output: 前一个输出词 (output_size, 1)
            decoder_hidden: 解码器状态 (hidden_size, 1)
            encoder_annotations: 编码器状态列表
        
        返回:
            output: 输出分布
            new_hidden: 新的解码器状态
            attention_weights: 注意力权重
        """
        # 1. 计算注意力和上下文向量
        context, attention_weights = self.attention.forward(decoder_hidden, encoder_annotations)
        
        # 2. RNN 更新
        rnn_input = np.vstack([prev_output, context])
        concat = np.vstack([rnn_input, decoder_hidden])
        new_hidden = np.tanh(np.dot(self.W_dec, concat) + self.b_dec)
        
        # 3. 输出预测
        output_input = np.vstack([new_hidden, context, prev_output])
        output = softmax(np.dot(self.W_out, output_input) + self.b_out)
        
        return output, new_hidden, attention_weights

print("注意力解码器创建完成")
```

### 完整的 Seq2Seq + Attention

```python
class Seq2SeqWithAttention:
    """完整的序列到序列模型(带注意力)"""
    def __init__(self, input_size, output_size, hidden_size):
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttentionDecoder(output_size, hidden_size, hidden_size * 2)
    
    def forward(self, inputs, max_output_len):
        """
        前向传播
        
        参数:
            inputs: 输入序列
            max_output_len: 最大输出长度
        
        返回:
            outputs: 输出序列
            all_attention_weights: 所有时间步的注意力权重
        """
        # 1. 编码
        encoder_annotations = self.encoder.forward(inputs)
        
        # 2. 解码
        outputs = []
        all_attention_weights = []
        
        # 初始状态
        decoder_hidden = np.zeros((self.decoder.hidden_size, 1))
        prev_output = np.zeros((self.decoder.output_size, 1))
        prev_output[0] = 1  # <START> token
        
        for _ in range(max_output_len):
            output, decoder_hidden, attention_weights = self.decoder.step(
                prev_output, decoder_hidden, encoder_annotations
            )
            
            outputs.append(output)
            all_attention_weights.append(attention_weights)
            
            # 使用预测的输出作为下一步输入
            prev_output = output
        
        return outputs, all_attention_weights

print("完整的 Seq2Seq + Attention 模型创建完成")
```

## 注意力可视化

```python
import matplotlib.pyplot as plt

def visualize_attention(attention_weights, input_words, output_words):
    """
    可视化注意力权重
    
    参数:
        attention_weights: (output_len, input_len) 注意力矩阵
        input_words: 输入词列表
        output_words: 输出词列表
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    attention_matrix = np.array(attention_weights)
    plt.imshow(attention_matrix, cmap='YlOrRd', aspect='auto')
    
    # 设置坐标轴标签
    plt.xticks(range(len(input_words)), input_words, rotation=45)
    plt.yticks(range(len(output_words)), output_words)
    
    plt.xlabel('输入序列')
    plt.ylabel('输出序列')
    plt.title('注意力权重可视化')
    plt.colorbar(label='注意力权重')
    
    plt.tight_layout()
    plt.show()

# 示例
print("注意力可视化示例:")
print("- X轴: 输入序列(源语言)")
print("- Y轴: 输出序列(目标语言)")
print("- 颜色深浅: 注意力权重大小")
print("\n理想情况:")
print("- 对角线模式: 正确的对齐")
print("- 翻译每个词时关注对应的源词")
```

## 注意力机制的优势

### 1. 解决信息瓶颈

```
传统 Seq2Seq:
- 所有信息压缩到固定向量
- 长句子信息丢失严重

注意力机制:
- 解码时动态"查看"原文
- 每个时刻关注不同部分
- 长句子翻译质量显著提升
```

### 2. 提供可解释性

```
注意力权重的价值:
1. 可视化对齐关系
2. 理解模型的推理过程
3. 发现翻译错误的原因
4. 调试和分析更容易

这就像翻译官在翻译时会用手指着原文,
告诉我们他正在看哪里!
```

### 3. 改善长距离依赖

```
RNN 的长距离依赖问题:
- 信息要一步步传递
- 远距离信息容易丢失

注意力机制:
- 每个位置可以直接看到所有位置
- 距离为 1,不存在长距离问题
```

## 实际应用效果

### 机器翻译

```
BLEU 分数提升:

英德翻译:
- 传统 Seq2Seq: 22.5
- + 注意力: 26.8 (+4.3)

英法翻译:
- 传统 Seq2Seq: 28.5
- + 注意力: 32.1 (+3.6)

显著改进!
```

### 其他应用

```python
# 1. 文本摘要
print("文本摘要:")
print("- 输入: 长文章")
print("- 输出: 摘要")
print("- 注意力: 生成每个摘要词时关注原文相关部分")

# 2. 问答系统
print("\n问答系统:")
print("- 输入: 文章 + 问题")
print("- 输出: 答案")
print("- 注意力: 回答问题时关注文章相关段落")

# 3. 图像描述
print("\n图像描述:")
print("- 输入: 图像特征")
print("- 输出: 描述文本")
print("- 注意力: 生成每个词时关注图像不同区域")
```

## 小结

今天我们深入理解了 Bahdanau 注意力机制:

### 核心创新

1. **动态上下文**: 每个时刻计算不同的上下文向量
2. **软对齐**: 自动学习源语言和目标语言的对齐关系
3. **可解释性**: 注意力权重可视化理解模型行为

### 为什么有效

1. **打破瓶颈**: 不再依赖固定长度向量
2. **长距离依赖**: 直接访问所有位置
3. **对齐学习**: 自动学习翻译对齐

### 历史意义

Bahdanau 注意力开创了注意力机制的先河:
- 2014: Bahdanau 注意力
- 2015: 各种注意力变体
- 2017: Transformer (纯注意力架构)
- 2018+: BERT、GPT 等大模型

**注意力机制成为现代深度学习的基石!**

## 练习题

### 1. 概念理解

**问题 1**: Bahdanau 注意力和 Transformer 的自注意力有什么区别?

**问题 2**: 为什么双向 RNN 编码器比单向 RNN 更好?

**问题 3**: 注意力机制的复杂度是多少?如何处理超长序列?

### 2. 编程实践

**练习 1**: 实现其他类型的注意力:

```python
class DotProductAttention:
    """点积注意力"""
    def forward(self, query, keys, values):
        # TODO: 实现
        pass

class LuongAttention:
    """Luong 注意力 (乘性注意力)"""
    def forward(self, decoder_hidden, encoder_annotations):
        # TODO: 实现
        pass
```

**练习 2**: 在真实数据集上训练:

```python
# 使用 WMT 或 IWSLT 数据集
# 实现完整的训练和评估流程
# TODO: 实现
```

### 3. 深度思考

**思考 1**: 注意力机制是否总是有益的?在什么情况下可能不必要?

**思考 2**: 如何将 Bahdanau 注意力应用到图像描述生成任务?

**思考 3**: 注意力机制和人类的视觉注意力有什么异同?

## 延伸阅读

### 经典论文

1. **Bahdanau 注意力**: Bahdanau et al. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate"
2. **Luong 注意力**: Luong et al. (2015). "Effective Approaches to Attention-based Neural Machine Translation"
3. **Transformer**: Vaswani et al. (2017). "Attention Is All You Need"

### 教程资源

- "Visualizing A Neural Machine Translation Model" - Jay Alammar
- "Attention? Attention!" - Lilian Weng
- "The Annotated Encoder-Decoder" - Harvard NLP

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 14 篇。上一篇我们学习了 Transformer 的纯注意力架构,下一篇我们将探讨预激活 ResNet 如何改善梯度流。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!** 🚀