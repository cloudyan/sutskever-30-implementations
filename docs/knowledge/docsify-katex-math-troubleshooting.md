---
title: Docsify + KaTeX 公式渲染排障
aliases:
  - Docsify 数学公式不显示
  - KaTeX 在 docsify 中不渲染
tags:
  - docsify
  - katex
  - markdown
  - obsidian
  - troubleshooting
---

# Docsify + KaTeX 公式渲染排障

## 核心结论

这次遇到的公式不渲染问题，主要有两个根因：

1. `docsify-katex` 插件依赖没有补全
2. 数学模式里用了不符合 LaTeX 语法的字符串写法

## 问题现象

### 现象 1：块级公式不渲染

下面这种写法在 Markdown 中是正确的：

```markdown
$$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$$
```

但在 docsify 页面中没有正确预览。

### 现象 2：行内公式部分内容显示成普通文本

下面这种写法会出问题：

```markdown
$K(\"0\"^{10000}) \approx \log_2(10000) + O(1) \approx 14$
```

## 根因分析

### 根因 1：`docsify-katex` 需要 `marked@4`

虽然页面里已经引入了：

```html
<script src="https://cdn.jsdelivr.net/npm/docsify@4"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/docsify-katex@latest/dist/docsify-katex.js"></script>
```

但 `docsify-katex` 在当前组合下还依赖 `marked@4` 的全局对象。缺失后，公式插件可能加载了，但解析链路不完整，导致 `$$...$$` 公式无法稳定渲染。

### 根因 2：数学模式里不能把 `\"` 当作引号转义

在 Markdown 里，`\"` 看起来像是转义双引号。

但在 KaTeX / LaTeX 数学模式里，`\"` 不是双引号，而是重音命令，所以：

```markdown
$K(\"0\"^{10000})$
```

不是合法的“字符串 0 的幂次”写法，容易直接渲染失败或局部失效。

## 修复方案

### 修复 1：补齐 docsify 数学插件依赖

推荐写法：

```html
<script src="https://cdn.jsdelivr.net/npm/docsify@4"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@4"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/docsify-katex@2.0.2/dist/docsify-katex.js"></script>
```

说明：

- 显式引入 `marked@4`
- 不再使用 `docsify-katex@latest`
- 固定插件版本，避免未来 CDN 最新版带来兼容漂移

## 修复 2：把“字符串样式”改成数学友好的写法

错误写法：

```markdown
$K(\"0\"^{10000})$
```

推荐写法：

```markdown
$K(\mathtt{0}^{10000})$
```

如果必须保留引号，也可以写：

```markdown
$K(\text{"0"}^{10000})$
```

但一般更推荐 `\mathtt{...}`，语义更清楚，也更稳定。

## 本次实际修复

### 配置层

修复了 [docs/index.html](/Users/cloudyan/data/code/llm/sutskever-30-implementations/docs/index.html) 中的数学公式脚本加载：

- 增加 `marked@4`
- 将 `docsify-katex` 固定为 `2.0.2`

### 文档层

将以下页面中的 `\"0\"`、`\"ab\"` 等数学模式字符串写法改为 `\mathtt{...}`：

- [02-kolmogorov-complexity.md](/Users/cloudyan/data/code/llm/sutskever-30-implementations/docs/tutorials/papers/01-complexity-dynamics/02-kolmogorov-complexity.md)
- [06-sophistication.md](/Users/cloudyan/data/code/llm/sutskever-30-implementations/docs/tutorials/papers/01-complexity-dynamics/06-sophistication.md)
- [07-resource-bounded-kc.md](/Users/cloudyan/data/code/llm/sutskever-30-implementations/docs/tutorials/papers/01-complexity-dynamics/07-resource-bounded-kc.md)

## 可复用检查清单

遇到 docsify 公式不显示时，优先检查：

1. 是否已经加载 `katex.min.css`
2. 是否已经加载 `katex.min.js`
3. 是否已经加载 `docsify-katex`
4. 是否补了 `marked@4`
5. 是否把插件版本固定，而不是使用 `@latest`
6. 行内数学里是否混入了非法 LaTeX 语法
7. 是否在 `$...$` 里写了 `\"`
8. 字符串、代码样式是否应该改成 `\mathtt{...}` 或 `\text{...}`

## 经验总结

一句话记忆：

> Docsify 里的公式问题，先查插件依赖，再查 LaTeX 语法，不要把 Markdown 转义习惯带进数学模式。

## 相关知识点

- [[Markdown 数学公式]]
- [[KaTeX]]
- [[Docsify 插件兼容性]]
- [[LaTeX 数学模式]]
