# AI 数学基础 Docsify 文档

_从线性代数到 Transformer 的完整数学体系 - 在线文档预览_

---

## 🚀 快速启动

### 方式 1：使用 docsify CLI（推荐）

```bash
# 安装 docsify CLI
npm install -g docsify-cli

# 进入文档目录
cd /Users/cloudyan/.openclaw/workspace/ai-math-docs

# 启动本地服务器
docsify serve .

# 访问 http://localhost:3000
```

### 方式 2：使用 Python HTTP 服务器

```bash
cd /Users/cloudyan/.openclaw/workspace/ai-math-docs
python3 -m http.server 8000

# 访问 http://localhost:8000
```

### 方式 3：直接用浏览器打开

```bash
open /Users/cloudyan/.openclaw/workspace/ai-math-docs/index.html
```

---

## 📁 文档结构

```
ai-math-docs/
├── index.html              # Docsify 主配置
├── README.md               # 首页
├── _sidebar.md             # 侧边栏导航
└── docs/
    ├── stage1.md           # 阶段 1：线性代数
    ├── stage2.md           # 阶段 2：微积分与优化
    ├── stage3.md           # 阶段 3-6 概览
    ├── stage4.md           # 阶段 4-6 详细
    ├── stage5.md           # 阶段 5：信息论
    ├── stage6.md           # 阶段 6：综合应用
    └── learning-path.md    # 学习路径
```

---

## 📖 文档内容

### 阶段 1：线性代数基础（Day 1-10）

- 向量与向量空间
- 矩阵与矩阵运算
- 线性变换
- 行列式
- 逆矩阵与伪逆
- 特征值与特征向量
- 特征分解
- 奇异值分解 SVD
- 主成分分析 PCA
- 范数与距离

### 阶段 2：微积分与优化（Day 11-20）

- 导数与偏导数
- 梯度
- 链式法则
- 泰勒展开
- 梯度下降法
- 牛顿法与拟牛顿法
- 拉格朗日乘数法
- 凸优化基础
- SGD 与动量
- Adam 优化器

### 阶段 3：概率论基础（Day 21-30）

- 概率与条件概率
- 贝叶斯定理
- 随机变量
- 概率分布
- 期望与方差
- 协方差与相关系数
- 大数定律
- 中心极限定理
- 联合分布与边缘分布
- 条件独立性

### 阶段 4：统计与推断（Day 31-40）

- 最大似然估计 MLE
- 最大后验估计 MAP
- 贝叶斯推断
- 假设检验
- 置信区间
- 回归分析
- 过拟合与欠拟合
- 正则化 L1/L2
- 交叉验证
- Bootstrap 方法

### 阶段 5：信息论基础（Day 41-45）

- 信息量与自信息
- 熵 Entropy
- 交叉熵
- KL 散度
- 互信息

### 阶段 6：综合与应用（Day 46-60）

- softmax 与交叉熵
- 高斯分布与最小二乘
- sigmoid 与逻辑回归
- 拉普拉斯平滑
- EM 算法
- 马尔可夫链
- 隐马尔可夫模型 HMM
- 蒙特卡洛方法
- MCMC 采样
- 变分推断
- 注意力机制的数学
- 位置编码的数学
- 归一化技术
- 初始化方法
- 学习率调度

---

## 🎨 特性

- ✅ **响应式设计**：支持手机、平板、电脑
- ✅ **代码高亮**：支持 Python、JavaScript 等多种语言
- ✅ **数学公式**：支持 LaTeX 公式渲染
- ✅ **全文搜索**：快速查找知识点
- ✅ **分页导航**：上一篇/下一篇切换
- ✅ **知识依赖图**：Mermaid 流程图展示依赖关系

---

## 🛠️ 自定义主题

编辑 `index.html` 中的 `--theme-color` 变量：

```css
:root {
  --theme-color: #4A90E2;  /* 修改为主题色 */
}
```

---

## 📝 更新文档

1. 编辑对应的 `.md` 文件
2. 刷新浏览器即可看到更新（无需重新构建）

---

## 🌐 部署到 GitHub Pages

```bash
# 安装 docsify-cli
npm install -g docsify-cli

# 初始化（如果还没有）
docsify init ./docs

# 推送到 GitHub
git add .
git commit -m "Add AI Math docs"
git push origin main

# 在 GitHub 仓库设置中启用 GitHub Pages
# Settings -> Pages -> Source: main branch -> Save
```

---

## 📞 反馈

如有问题或建议，欢迎交流！

---

*最后更新：2026-04-22*
