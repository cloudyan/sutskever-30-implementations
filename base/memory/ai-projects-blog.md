# 从前端架构师到 AI Agent 工程师：我的研发效能实践之路

> **摘要**：本文分享了我在众安在线从前端架构师转型 AI Agent 方向的实践经历，详细拆解了三个核心 AI 项目：基于 RAG 的前端知识库、AI 代码审查系统、AI Commit 工具。希望能给想转型 AI 的前端同学一些参考。

---

## 一、背景：为什么做 AI 研发效能？

2024 年 11 月加入众安在线后，我负责前端研发效能相关工作。接手后发现几个痛点：

1. **文档缺失严重** - 核心业务代码缺乏文档，新人上手成本高
2. **代码审查深度不足** - CR 流于形式，关键 Bug 和合规问题容易遗漏
3. **Commit 信息低效** - 提交信息不规范，后续追溯困难

作为有 13 年前端经验的架构师，我意识到这正是 AI Agent 可以发挥价值的场景。于是从 2025 年 8 月开始，我主导了 AI 研发效能的探索与实践。

---

## 二、项目一：基于 RAG 的前端知识库

### 2.1 技术方案

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  代码仓库   │ ──→ │  文档生成器  │ ──→ │  Milvus     │
│  (Git)      │     │  (DeepWiki)  │     │  向量数据库  │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ↓
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  用户提问   │ ←── │  LLM 回答    │ ←── │  检索模块   │
│  (Chat)     │     │  (Qwen/GPT)  │     │  (RAG)      │
└─────────────┘     └──────────────┘     └─────────────┘
```

### 2.2 核心实现

**技术栈**：
- DeepWiki-Open（代码解析）
- Milvus（向量数据库）
- LangChain（RAG 编排）
- Qwen/GPT-4（LLM）

**关键代码片段**：

```typescript
// 向量检索核心逻辑
async function retrieveRelevantCode(query: string, topK: number = 5) {
  // 1. 查询向量化
  const queryEmbedding = await embedder.embed(query);
  
  // 2. Milvus 相似度检索
  const results = await milvusClient.search({
    collection_name: 'frontend_code',
    vector: queryEmbedding,
    limit: topK,
    params: { metric_type: 'COSINE' }
  });
  
  // 3. 返回上下文
  return results.map(r => ({
    code: r.entity.code,
    filePath: r.entity.file_path,
    score: r.score
  }));
}
```

### 2.3 成果

- ✅ 覆盖司内 100+ 前端项目
- ✅ 自动生成 API 文档、组件文档、架构说明
- ✅ 智能问答准确率 85%+
- ✅ 新人上手时间缩短 40%

---

## 三、项目二：AI 代码审查系统

### 3.1 设计思路

传统代码审查的问题：
- 人工 CR 耗时，容易遗漏
- 审查标准不统一
- 新人看不出问题

我们的解决方案：**启发式核心变更高优审查策略**

```
┌─────────────┐
│  Git Diff   │
└──────┬──────┘
       ↓
┌─────────────┐     ┌──────────────┐
│  变更分析   │ ──→ │  风险评分    │
│  (AST 解析) │     │  (核心/普通) │
└──────┬──────┘     └──────┬───────┘
       ↓                   ↓
   普通变更            核心变更
       │                   │
       ↓                   ↓
┌─────────────┐     ┌──────────────┐
│  基础检查   │     │  分层 Prompt  │
│  (规则引擎) │     │  (深度审查)  │
└─────────────┘     └──────────────┘
```

### 3.2 分层 Prompt 设计

```python
# Layer 1: Bug 检测
BUG_DETECTION_PROMPT = """
分析以下代码变更，识别潜在的 Bug：
1. 空指针/未定义访问
2. 类型不匹配
3. 边界条件遗漏
4. 异步处理错误

变更内容：
{diff}

请按以下格式输出：
- 问题描述
- 风险等级 (High/Medium/Low)
- 修复建议
"""

# Layer 2: 合规检查
COMPLIANCE_PROMPT = """
检查代码是否符合安全合规要求：
1. 敏感信息硬编码
2. 数据加密处理
3. 权限校验
4. 日志脱钩

变更内容：
{diff}
"""

# Layer 3: 性能优化
PERFORMANCE_PROMPT = """
分析代码性能问题：
1. 不必要的循环/递归
2. 内存泄漏风险
3. 异步并发优化
4. 缓存策略

变更内容：
{diff}
"""
```

### 3.3 成果

| 指标 | 数值 |
|------|------|
| 审查准确率 | 65% |
| 关键 Bug 检出率 | 70% |
| 审查耗时 | 降低 80% |
| 团队采纳率 | 90%+ |

---

## 四、项目三：AI Commit 工具

### 4.1 问题与方案

**问题**：
- Commit 信息不规范（"fix bug"、"update"）
- 后续追溯困难
- Changelog 生成困难

**方案**：Diff 提取 → 语义理解 → 规范 Message 生成

```typescript
// AI Commit 核心流程
async function generateCommitMessage(diff: string): Promise<string> {
  // 1. Diff 解析
  const changes = parseDiff(diff);
  
  // 2. 语义分析
  const analysis = await llm.analyze({
    prompt: `分析以下代码变更的语义：
    - 变更类型 (feat/fix/refactor/chore...)
    - 影响模块
    - 变更描述
    
    Diff: ${diff}`
  });
  
  // 3. 生成规范 Message
  const message = formatConventionalCommit({
    type: analysis.type,
    scope: analysis.scope,
    description: analysis.description,
    body: analysis.body
  });
  
  return message;
}
```

### 4.2 成果

- ✅ 团队采纳率 90%
- ✅ Commit 规范率从 30% 提升至 95%
- ✅ 自动生成 Changelog
- ✅ 代码追溯效率提升 60%

---

## 五、开源贡献：Neovate Bug 修复

在研究 Neovate（AI 编程助手）源码时，我发现了一个 RAG 检索的 Bug：

```typescript
// 问题代码
const results = await vectorStore.similaritySearch(query, k);
// 未处理空结果情况，导致后续处理崩溃

// 修复
const results = await vectorStore.similaritySearch(query, k);
if (!results || results.length === 0) {
  return handleNoResults(query);
}
```

已提 PR 并合并。这也是我转型 AI 后的第一个开源贡献。

---

## 六、经验总结

### 6.1 前端转型 AI 的优势

1. **工程能力强** - 前端架构经验能落地，不是纸上谈兵
2. **产品思维** - 懂用户体验，Agent 交互设计更友好
3. **全栈视角** - 能打通前后端，做完整的 Agent 系统

### 6.2 需要补的课

1. **模型底层** - Transformer、Attention 机制需要深入理解
2. **向量检索** - Milvus、FAISS 等需要系统学习
3. **Agent 框架** - LangChain、AutoGen 需要熟练掌握

### 6.3 给想转型的同学的建议

1. **从现有工作切入** - 找到能用 AI 优化的痛点
2. **做能落地的项目** - 不要追求高大上，解决实际问题
3. **持续学习** - AI 技术迭代快，保持好奇心
4. **输出倒逼输入** - 写博客、做分享、参与开源

---

## 七、下一步计划

1. **技术深度** - 深入学习 Transformer、RAG 优化、Agent 规划
2. **项目广度** - 做 1-2 个 C 端可见的 Agent 产品
3. **职业发展** - 探索 AI Agent 架构师/技术专家方向

---

## 关于我

- **闫战军**，13 年前端开发经验
- 现任职于众安在线，负责 AI 研发效能
- 技术栈：React/Vue/Taro + RAG/LangChain/LLM
- GitHub：github.com/cloudyan

**欢迎交流 AI Agent、前端架构、研发效能相关话题！**

---

*本文首发于 2026 年 3 月，如有转载需求请联系作者。*
