# 阶段 6：LLM 应用

_大语言模型的应用与开发_

---

## 📖 学习指南

**前置知识**：
- ✅ Transformer 架构
- ✅ Python 编程
- ✅ 深度学习基础

**学习目标**：
- ✅ 理解 LLM 基本原理
- ✅ 掌握 Prompt 工程技巧
- ✅ 掌握 RAG 检索增强生成
- ✅ 掌握微调技术（LoRA、QLoRA）
- ✅ 能开发 LLM 应用

**预计时间**：45 天

---

## 6.1 LLM 基础

### 什么是 LLM？

<div class="formula-box">

```
LLM（Large Language Model）= 大规模语言模型

核心特点：
- 参数量巨大（10 亿 - 万亿级）
- 在海量文本上预训练
- 具有涌现能力（Emergent Abilities）
- 可完成多种任务（零样本、少样本）

代表模型：
- GPT 系列（OpenAI）
- LLaMA 系列（Meta）
- Claude 系列（Anthropic）
- 通义千问（阿里）
- 文心一言（百度）
```

</div>

### LLM 架构演进

<div class="formula-box">

```
GPT (2018)：
- Decoder-only
- 1.17B 参数
- 单向注意力

GPT-2 (2019)：
- 1.5B 参数
- 零样本能力

GPT-3 (2020)：
- 175B 参数
- 少样本学习（Few-shot）

LLaMA (2023)：
- 7B-65B 参数
- 开源、高效

LLaMA 2 (2023)：
- 免费商用
- 更好的对齐

LLaMA 3 (2024)：
- 8B-70B 参数
- 更强的推理能力
```

</div>

---

## 6.2 Prompt 工程

### 什么是 Prompt 工程？

<div class="formula-box">

```
Prompt 工程 = 设计最优的输入提示，引导 LLM 输出期望结果

核心思想：
- 不修改模型参数
- 通过文本指令控制模型
- 低成本、高效率
```

</div>

### 基础技巧

<div class="formula-box">

```python
# 1. 清晰指令
❌ "写点什么"
✅ "写一篇 500 字的文章，介绍人工智能的发展历史"

# 2. 提供示例（Few-shot）
❌ "分类：这部电影太好看了"
✅ 
"""
示例 1: "这部电影太棒了" → 正面
示例 2: "我讨厌这个电影" → 负面
分类："这部电影太好看了" →
"""

# 3. 指定输出格式
✅ "请用 JSON 格式输出，包含 title、content、tags 三个字段"

# 4. 分步思考（Chain of Thought）
✅ "请逐步思考：首先...其次...最后..."

# 5. 角色扮演
✅ "你是一位资深软件工程师，请审查以下代码..."
```

</div>

### 高级技巧

<div class="formula-box">

```
1. Chain of Thought (CoT)
让模型展示推理过程

Prompt:
"小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
请逐步推理。"

效果：
- 减少计算错误
- 提高复杂任务准确率

2. Self-Consistency
多次采样，选择最一致的答案

3. ReAct（Reasoning + Acting）
推理与行动交替

4. Tree of Thoughts
多路径探索，选择最优解
```

</div>

### 实战示例

<div class="formula-box">

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# 1. 文本分类
def classify_sentiment(text):
    prompt = f"""
请判断以下文本的情感倾向（正面/负面/中性）：

文本："{text}"

情感倾向：
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 2. 代码生成
def generate_code(description):
    prompt = f"""
请编写 Python 代码实现以下功能：

{description}

要求：
1. 代码简洁高效
2. 添加必要的注释
3. 包含使用示例

代码：
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 3. 文档摘要
def summarize_document(text, max_length=200):
    prompt = f"""
请用{max_length}字以内总结以下文档的核心内容：

{text}

摘要：
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

</div>

---

## 6.3 RAG（检索增强生成）

### 什么是 RAG？

<div class="formula-box">

```
RAG（Retrieval-Augmented Generation）= 检索 + 生成

流程：
1. 用户提问
2. 从知识库检索相关文档
3. 将文档注入 Prompt
4. LLM 生成回答

优势：
- 减少幻觉
- 知识可更新
- 可追溯来源
```

</div>

### RAG 架构

<div class="formula-box">

```
用户问题
    ↓
Embedding 模型
    ↓
问题向量
    ↓
向量数据库（检索 Top-K 相关文档）
    ↓
文档片段 + 问题 → Prompt 模板
    ↓
LLM
    ↓
回答
```

</div>

### 实战实现

<div class="formula-box">

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. 加载文档
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# 2. 分割文档
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# 3. 创建向量数据库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# 4. 创建检索链
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 5. 使用
query = "什么是机器学习？"
result = qa_chain({"query": query})

print(f"回答：{result['result']}")
print(f"来源：{result['source_documents']}")
```

</div>

### 优化技巧

<div class="formula-box">

```
1. 文档分割策略
- 按段落分割
- 按语义分割
- 重叠窗口（避免信息丢失）

2. 检索优化
- 多路召回（关键词 + 向量）
- 重排序（Re-ranking）
- 元数据过滤

3. Prompt 优化
- 添加系统指令
- 指定回答格式
- 要求引用来源
```

</div>

---

## 6.4 微调技术

### 为什么需要微调？

<div class="formula-box">

```
预训练模型的问题：
- 通用知识强，领域知识弱
- 指令遵循能力不足
- 输出风格不可控

微调的作用：
- 适应特定领域
- 提升任务性能
- 控制输出风格
```

</div>

### 全量微调 vs 参数高效微调

<div class="formula-box">

```
全量微调：
- 更新所有参数
- 计算资源需求大
- 可能灾难性遗忘

参数高效微调（PEFT）：
- 只更新少量参数
- 计算资源需求小
- 保持预训练知识
- 代表：LoRA、QLoRA、Adapter
```

</div>

### LoRA（Low-Rank Adaptation）

<div class="formula-box">

```
原理：
W' = W + ΔW = W + BA

其中：
- W：预训练权重（冻结）
- B ∈ R^(d×r)：低秩矩阵（可学习）
- A ∈ R^(r×k)：低秩矩阵（可学习）
- r << d（通常 r=8, 16, 32）

优势：
- 参数量减少 10000 倍
- 训练速度提升 3 倍
- 效果接近全量微调
```

</div>

<div class="formula-box">

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 1. 加载基础模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=16,                      # 秩
    lora_alpha=32,             # 缩放系数
    target_modules=["q_proj", "v_proj"],  # 目标模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出：trainable params: 0.1% || all params: 100%

# 4. 训练（与普通训练相同）
trainer = Trainer(model=model, ...)
trainer.train()

# 5. 保存 LoRA 权重
model.save_pretrained("lora_weights")
```

</div>

### QLoRA（Quantized LoRA）

<div class="formula-box">

```
原理：
- 4-bit 量化预训练模型
- 结合 LoRA 微调

优势：
- 显存需求降低 4 倍
- 7B 模型只需 8GB 显存
- 效果接近 LoRA
```

</div>

<div class="formula-box">

```python
from transformers import BitsAndBytesConfig

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 应用 LoRA（同上）
```

</div>

---

## 6.5 LLM Agent

### 什么是 Agent？

<div class="formula-box">

```
LLM Agent = LLM + 规划 + 工具使用 + 记忆

核心能力：
1. 规划（Planning）
   - 任务分解
   - 多步推理

2. 工具使用（Tool Use）
   - 调用 API
   - 执行代码
   - 查询数据库

3. 记忆（Memory）
   - 短期记忆（上下文）
   - 长期记忆（向量数据库）
```

</div>

### LangChain 框架

<div class="formula-box">

```python
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper

# 1. 定义工具
search = DuckDuckGoSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="搜索互联网获取最新信息"
    ),
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="执行数学计算"
    )
]

# 2. 初始化 Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 3. 使用
response = agent.run("今天北京的天气如何？气温是多少度？")
print(response)
```

</div>

### AutoGen 框架

<div class="formula-box">

```python
from autogen import AssistantAgent, UserProxyAgent

# 1. 配置
config_list = [{"model": "gpt-4", "api_key": "your-api-key"}]

# 2. 创建 Agent
assistant = AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"}
)

# 3. 对话
user_proxy.initiate_chat(
    assistant,
    message="请帮我写一个 Python 脚本，计算斐波那契数列"
)
```

</div>

---

## 6.6 实战项目

### 项目 1：智能客服系统

<div class="formula-box">

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. 准备知识库（RAG）
vectorstore = Chroma.from_documents(docs, embeddings)

# 2. 创建对话链
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# 3. 对话
chat_history = []
while True:
    query = input("用户：")
    if query == "exit":
        break
    
    result = qa_chain({"question": query, "chat_history": chat_history})
    print(f"客服：{result['answer']}")
    
    chat_history.append((query, result['answer']))
```

</div>

### 项目 2：代码助手

<div class="formula-box">

```python
from langchain.agents import create_python_agent
from langchain.python import PythonREPL

# 创建 Python Agent
agent = create_python_agent(
    llm=ChatOpenAI(model="gpt-4"),
    tool=PythonREPL(),
    verbose=True
)

# 使用
response = agent.run("请计算 1 到 100 的平方和")
print(response)
```

</div>

### 项目 3：文档问答系统

<div class="formula-box">

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. 加载 PDF
loader = PyPDFLoader("manual.pdf")
documents = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# 3. 创建向量库（使用本地 Embedding）
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)

# 4. 创建问答链
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever()
)

# 5. 使用
query = "如何安装这个软件？"
answer = qa.run(query)
print(answer)
```

</div>

---

## 📚 学习资源

### 官方文档

- [OpenAI API](https://platform.openai.com/docs)
- [LangChain](https://python.langchain.com/)
- [HuggingFace](https://huggingface.co/docs)

### 开源项目

- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [AutoGen](https://github.com/microsoft/autogen)

### 模型资源

- [HuggingFace Model Hub](https://huggingface.co/models)
- [LLaMA 下载](https://ai.meta.com/llama/)

---

## ✅ 学习检查清单

- [ ] 理解 LLM 基本原理
- [ ] 掌握 Prompt 工程技巧
- [ ] 掌握 RAG 原理与实现
- [ ] 理解微调技术（LoRA、QLoRA）
- [ ] 了解 Agent 架构
- [ ] 掌握 LangChain 框架
- [ ] 完成至少 2 个实战项目

---

*最后更新：2026-04-22*
