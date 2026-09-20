# 🚀 AI 进阶教程 - 部署指南

---

## 📋 完成内容

✅ **文档体系设计完成**

| 项目 | 状态 |
|------|------|
| 完整知识体系设计 | ✅ 完成（12 个阶段） |
| README.md（首页） | ✅ 完成（18KB） |
| _sidebar.md（导航） | ✅ 完成 |
| index.html（Docsify 配置） | ✅ 完成 |
| deploy.sh（部署脚本） | ✅ 完成 |

---

## 🎯 文档体系概览

### 12 个阶段，涵盖 AI 专家成长全路径

| 阶段 | 主题 | 预计时间 | 核心内容 |
|------|------|---------|---------|
| 1 | 数学基础 | 30 天 | 线性代数、微积分、概率论、统计、信息论 |
| 2 | Python 基础 | 14 天 | Python、NumPy、Pandas、可视化 |
| 3 | 机器学习 | 45 天 | 监督学习、无监督学习、特征工程 |
| 4 | 深度学习 | 60 天 | 神经网络、CNN、RNN、GAN |
| 5 | Transformer ⭐ | 45 天 | 注意力机制、Transformer、BERT、GPT、ViT |
| 6 | LLM 应用 | 45 天 | Prompt 工程、RAG、微调、Agent |
| 7 | AI Agent | 45 天 | Agent 架构、工具使用、多 Agent |
| 8 | 系统优化 | 30 天 | 模型压缩、量化、推理优化、部署 |
| 9 | 高级架构 | 30 天 | MoE、多模态、世界模型、Mamba |
| 10 | 研究与创新 | 持续 | 论文阅读、实验设计、创新思维 |
| 11 | 工程实践 | 30 天 | MLOps、部署、最佳实践 |
| 12 | 创新与创业 | 持续 | 产品思维、创业、行业洞察 |

**总计**：约 375 天（1 年 +）系统学习

---

## 📁 目录结构

```
ai-advanced-docs/
├── index.html              # Docsify 主配置
├── README.md               # 首页（完整知识体系）
├── _sidebar.md             # 侧边栏导航
├── deploy.sh               # 部署脚本
├── DEPLOY.md               # 部署说明
│
├── 01-math-foundation/     # 阶段 1：数学基础
│   ├── README.md
│   ├── day-01-vector.md
│   ├── day-02-matrix.md
│   ├── day-03-linear-transform.md
│   └── day-04-determinant.md
│
├── 02-python-basics/       # 阶段 2：Python 基础
├── 03-machine-learning/    # 阶段 3：机器学习
├── 04-deep-learning/       # 阶段 4：深度学习
├── 05-transformer/         # 阶段 5：Transformer ⭐
├── 06-llm-application/     # 阶段 6：LLM 应用
├── 07-ai-agent/            # 阶段 7：AI Agent
├── 08-system-optimization/ # 阶段 8：系统优化
├── 09-advanced-architecture/ # 阶段 9：高级架构
├── 10-research/            # 阶段 10：研究
├── 11-engineering/         # 阶段 11：工程
└── 12-innovation/          # 阶段 12：创新
```

---

## 🔧 部署步骤

### 步骤 1：创建 GitHub 仓库

1. 访问 https://github.com/new
2. 仓库名：`ai-advanced-docs`
3. 可见性：**Public**（公开）
4. 点击 **Create repository**

### 步骤 2：推送代码

在终端执行：

```bash
cd /Users/cloudyan/.openclaw/workspace/ai-advanced-docs

# 配置远程仓库
git remote add origin git@github.com:cloudyan/ai-advanced-docs.git

# 推送
git push -u origin main
```

### 步骤 3：启用 GitHub Pages

1. 访问：https://github.com/cloudyan/ai-advanced-docs/settings/pages
2. **Source**: Deploy from a branch
3. **Branch**: main / (root)
4. 点击 **Save**

**等待 1-2 分钟**即可访问！

---

## 🌐 访问地址

```
https://cloudyan.github.io/ai-advanced-docs/
```

---

## 📊 与 AI 数学基础的关系

| 项目 | AI 数学基础 | AI 进阶教程 |
|------|-----------|-----------|
| **定位** | 数学基础 | AI 核心原理与实现 |
| **内容** | 70 张数学卡片 | 12 阶段 AI 知识体系 |
| **目标** | 数学工具掌握 | AI 专家培养 |
| **后续** | 前置课程 | 结合论文、代码实现 |
| **状态** | ✅ 已部署 | ⏳ 待部署 |

---

## 🎯 下一步工作

### 短期（1-2 周）

- [ ] 创建 GitHub 仓库并部署
- [ ] 补充阶段 2-4 的详细内容
- [ ] 添加代码示例

### 中期（1-2 月）

- [ ] 补充阶段 5（Transformer）完整内容
- [ ] 添加论文解读
- [ ] 实现从零到一的代码复现

### 长期（3-6 月）

- [ ] 完成所有 12 个阶段
- [ ] 每章配套实战项目
- [ ] 建立学习社区

---

## 💡 特色亮点

1. **系统化** - 从数学到创新，完整路径
2. **结合论文** - 每章配套经典论文解读
3. **代码实现** - 从论文到代码完整复现
4. **实战导向** - 每阶段配套实战项目
5. **持续更新** - 跟进最新研究（Mamba、Sora 等）

---

*最后更新：2026-04-22*
