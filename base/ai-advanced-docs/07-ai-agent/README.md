# 阶段 7：AI Agent

_智能体系统的设计与实现_

---

## 📖 学习指南

**前置知识**：
- ✅ LLM 应用基础
- ✅ Python 编程
- ✅ API 调用基础

**学习目标**：
- ✅ 理解 Agent 的核心架构
- ✅ 掌握规划与推理能力
- ✅ 掌握工具使用（Function Calling）
- ✅ 掌握记忆机制
- ✅ 能开发多 Agent 系统

**预计时间**：30 天

---

## 7.1 Agent 核心架构

### 什么是 Agent？

<div class="formula-box">

```
LLM Agent = LLM（大脑）+ 规划 + 工具使用 + 记忆

核心能力：
1. 感知（Perception）
   - 理解用户输入
   - 感知环境状态

2. 规划（Planning）
   - 任务分解
   - 多步推理
   - 反思与修正

3. 行动（Action）
   - 工具使用
   - API 调用
   - 代码执行

4. 记忆（Memory）
   - 短期记忆（上下文）
   - 长期记忆（向量数据库）
```

</div>

### Agent 架构演进

<div class="formula-box">

```
ReAct (2022)：
Reason + Act
推理与行动交替进行

Reflexion (2023)：
添加反思机制
从失败中学习

Tree of Thoughts (2023)：
多路径探索
选择最优解

AutoGen (2023)：
多 Agent 协作
角色分工
```

</div>

---

## 7.2 规划能力

### 任务分解

<div class="formula-box">

```
复杂任务 → 子任务 1 → 子任务 2 → ... → 结果

示例：
"帮我规划一次北京旅行"
  ↓
1. 查询北京天气
2. 预订机票
3. 预订酒店
4. 规划景点路线
5. 推荐餐厅
```

</div>

<div class="formula-box">

```python
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI

# 定义工具
def search_weather(city):
    """查询天气"""
    return f"{city}今天晴朗，气温 25°C"

def book_flight(from_city, to_city):
    """预订机票"""
    return f"已预订从{from_city}到{to_city}的机票"

tools = [
    Tool(name="Weather", func=search_weather, description="查询城市天气"),
    Tool(name="Flight", func=book_flight, description="预订机票")
]

# 初始化 Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 使用
response = agent.run("帮我规划从上海到北京的旅行，包括天气和机票")
print(response)
```

</div>

### Chain of Thought（CoT）

<div class="formula-box">

```
Prompt 示例：

"小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
请逐步思考。"

效果：
- 减少计算错误
- 提高复杂任务准确率
- 让推理过程可解释
```

</div>

### Tree of Thoughts（ToT）

<div class="formula-box">

```
核心思想：
1. 生成多个思考路径
2. 评估每条路径
3. 选择最优路径
4. 深度优先或广度优先搜索

示例：
写作任务：
- 思路 1：按时间顺序
- 思路 2：按重要性
- 思路 3：按主题分类
  ↓
评估 → 选择思路 2 → 展开写作
```

</div>

---

## 7.3 工具使用

### Function Calling

<div class="formula-box">

```
OpenAI Function Calling：

{
  "name": "get_weather",
  "description": "查询城市天气",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "城市名"}
    },
    "required": ["city"]
  }
}
```

</div>

<div class="formula-box">

```python
from openai import OpenAI
import json

client = OpenAI(api_key="your-api-key")

# 定义函数
def get_weather(city):
    """查询城市天气"""
    weather_data = {
        "北京": "晴朗，25°C",
        "上海": "多云，28°C",
        "广州": "小雨，30°C"
    }
    return weather_data.get(city, "未知城市")

# 定义函数 schema
functions = [
    {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名，如'北京'、'上海'"
                }
            },
            "required": ["city"]
        }
    }
]

# 调用 LLM
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    functions=functions,
    function_call="auto"
)

# 处理函数调用
if response.choices[0].finish_reason == "function_call":
    function_name = response.choices[0].message.function_call.name
    function_args = json.loads(response.choices[0].message.function_call.arguments)
    
    # 执行函数
    result = get_weather(**function_args)
    
    # 将结果返回给 LLM
    second_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "北京今天天气怎么样？"},
            {"role": "assistant", "content": response.choices[0].message.content},
            {"role": "function", "name": "get_weather", "content": result},
        ]
    )
    
    print(second_response.choices[0].message.content)
```

</div>

### 代码执行

<div class="formula-box">

```python
from langchain.agents import create_python_agent
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

# 创建 Python Agent
agent = create_python_agent(
    llm=ChatOpenAI(model="gpt-4"),
    tool=PythonREPL(),
    verbose=True
)

# 使用
response = agent.run("请计算 1 到 100 的平方和，并画出折线图")
print(response)
```

</div>

### API 调用

<div class="formula-box">

```python
from langchain.tools import Tool
import requests

# 自定义 API 工具
def search_web(query):
    """搜索互联网"""
    url = f"https://api.example.com/search?q={query}"
    response = requests.get(url)
    return response.json()["result"]

def calculate(expression):
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

tools = [
    Tool(
        name="WebSearch",
        func=search_web,
        description="搜索互联网获取信息"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="计算数学表达式"
    )
]
```

</div>

---

## 7.4 记忆机制

### 短期记忆

<div class="formula-box">

```
短期记忆 = 对话上下文

实现：
- 保存历史对话
- 限制上下文长度
- 滑动窗口
```

</div>

<div class="formula-box">

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# 对话记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建对话链
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# 对话
chat_history = []
while True:
    query = input("用户：")
    if query == "exit":
        break
    
    result = chain({"question": query, "chat_history": chat_history})
    print(f"助手：{result['answer']}")
    
    chat_history.append((query, result['answer']))
```

</div>

### 长期记忆

<div class="formula-box">

```
长期记忆 = 向量数据库

实现：
- 将对话嵌入向量
- 存储到向量数据库
- 检索相关记忆
```

</div>

<div class="formula-box">

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 创建向量数据库记忆
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

memory = VectorStoreRetrieverMemory(retriever=retriever)

# 保存记忆
memory.save_context(
    {"input": "用户喜欢 Python"},
    {"output": "好的，我记住了"}
)

memory.save_context(
    {"input": "用户是软件工程师"},
    {"output": "明白了"}
)

# 检索记忆
memories = memory.load_memory_variables({"input": "用户想学什么编程"})
print(memories)
```

</div>

---

## 7.5 多 Agent 协作

### AutoGen 框架

<div class="formula-box">

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 配置
config_list = [{"model": "gpt-4", "api_key": "your-api-key"}]

# 创建 Agent
coder = AssistantAgent(
    name="Coder",
    system_message="你是资深软件工程师，负责编写代码。",
    llm_config={"config_list": config_list}
)

reviewer = AssistantAgent(
    name="Reviewer",
    system_message="你是代码审查专家，负责审查代码质量。",
    llm_config={"config_list": config_list}
)

user_proxy = UserProxyAgent(
    name="User",
    code_execution_config={"work_dir": "coding"}
)

# 群聊
groupchat = GroupChat(
    agents=[coder, reviewer, user_proxy],
    messages=[],
    max_round=10
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

# 开始协作
user_proxy.initiate_chat(
    manager,
    message="请帮我写一个快速排序算法，并审查代码质量"
)
```

</div>

### 角色分工

<div class="formula-box">

```
典型角色：

1. Manager（管理者）
   - 任务分配
   - 进度控制
   - 质量把关

2. Coder（程序员）
   - 编写代码
   - 调试修复

3. Reviewer（审查者）
   - 代码审查
   - 提出改进建议

4. Researcher（研究员）
   - 信息搜集
   - 资料整理

5. Writer（写作者）
   - 文档编写
   - 内容创作
```

</div>

---

## 7.6 实战项目

### 项目 1：智能研究助手

<div class="formula-box">

```python
from langchain.agents import Tool
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.chat_models import ChatOpenAI

# 工具定义
search = DuckDuckGoSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="搜索互联网获取最新信息"
    ),
    Tool(
        name="Summarize",
        func=lambda x: llm.predict(f"总结以下内容：{x}"),
        description="总结长文本"
    )
]

# 创建研究 Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
researcher = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 使用
topic = "人工智能的最新进展"
research_plan = f"""
请研究"{topic}"，包括：
1. 搜索最新论文
2. 总结核心贡献
3. 分析技术趋势
"""

result = researcher.run(research_plan)
print(result)
```

</div>

### 项目 2：自动化数据分析

<div class="formula-box">

```python
from autogen import AssistantAgent, UserProxyAgent

# 配置
config_list = [{"model": "gpt-4", "api_key": "your-api-key"}]

# 数据分析师 Agent
data_analyst = AssistantAgent(
    name="DataAnalyst",
    system_message="""你是数据科学家，负责数据分析。
你可以执行 Python 代码进行分析。
请使用 pandas、matplotlib 等库。""",
    llm_config={"config_list": config_list}
)

# 用户代理
user_proxy = UserProxyAgent(
    name="User",
    code_execution_config={"work_dir": "data_analysis"}
)

# 任务
task = """
请分析 sales_data.csv 文件：
1. 加载数据
2. 数据探索（形状、列名、统计信息）
3. 可视化销售趋势
4. 找出最畅销的产品
"""

# 执行
user_proxy.initiate_chat(data_analyst, message=task)
```

</div>

### 项目 3：智能客服系统

<div class="formula-box">

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory

# 工具
tools = [
    Tool(
        name="KnowledgeBase",
        func=search_knowledge_base,
        description="搜索知识库获取产品信息"
    ),
    Tool(
        name="OrderLookup",
        func=lookup_order,
        description="查询订单状态"
    ),
    Tool(
        name="TicketCreate",
        func=create_support_ticket,
        description="创建客服工单"
    )
]

# 记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 客服对话
while True:
    user_input = input("客户：")
    if user_input == "exit":
        break
    
    response = agent_executor({"input": user_input})
    print(f"客服：{response['output']}")
```

</div>

---

## 📚 学习资源

### 框架文档

- [LangChain](https://python.langchain.com/) - Agent 开发框架
- [AutoGen](https://microsoft.github.io/autogen/) - 多 Agent 协作
- [LlamaIndex](https://www.llamaindex.ai/) - 数据连接

### 论文

- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Tree of Thoughts: Deliberate Problem Solving](https://arxiv.org/abs/2305.10601)

---

## ✅ 学习检查清单

- [ ] 理解 Agent 核心架构
- [ ] 掌握任务分解方法
- [ ] 掌握 Function Calling
- [ ] 掌握代码执行
- [ ] 掌握短期记忆与长期记忆
- [ ] 了解多 Agent 协作
- [ ] 掌握 LangChain 框架
- [ ] 完成至少 2 个实战项目

---

*最后更新：2026-04-22*
