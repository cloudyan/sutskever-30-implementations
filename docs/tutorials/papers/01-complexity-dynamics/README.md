# 图解复杂动力学 (Complexity Dynamics)

大家好，我是云言，是《图解复杂动力学》教程的作者。

这个系列教程将带你深入理解 Scott Aaronson 的经典论文 **"The First Law of Complexodynamics"**，探索为什么物理系统的"复杂度"会随时间先增后减，而熵却单调递增。

## 适合什么群体？

- 对信息论、计算复杂性理论感兴趣的同学
- 想了解 Kolmogorov 复杂度实际应用的同学
- 学习机器学习中熵和复杂度概念的同学
- 对元胞自动机和复杂系统感兴趣的同学

## 要怎么阅读？

建议按顺序阅读：
1. **基础概念篇** - 先理解熵、复杂度、Kolmogorov 复杂度的基本概念
2. **核心问题篇** - 理解"复杂动力学第一定律"的核心思想
3. **数学工具篇** - 掌握 Sophistication、资源受限复杂度等数学工具
4. **应用实践篇** - 通过元胞自动机和咖啡混合模拟来验证理论

## 目录列表

### 1. 基础概念篇 :point_down:
- [01-熵是什么？熵和复杂度有什么区别？](01-entropy-vs-complexity.md)
- [02-Kolmogorov 复杂度：什么是最短描述？](02-kolmogorov-complexity.md)
- [03-香农熵 vs 算法熵：两种度量信息的方式](03-shannon-vs-algorithmic-entropy.md)

### 2. 核心问题篇 :point_down:
- [04-Sean Carroll 的困惑：为什么复杂度会先增后减？](04-sean-carroll-question.md)
- [05-复杂动力学的第一定律：Aaronson 的猜想](05-first-law-complexodynamics.md)

### 3. 数学工具篇 :point_down:
- [06-Sophistication：如何度量"精致复杂度"？](06-sophistication.md)
- [07-资源受限 Kolmogorov 复杂度：时间也有代价](07-resource-bounded-kc.md)
- [08-Complextropy：定义"复杂熵"的尝试](08-complextropy.md)

### 4. 应用实践篇 :point_down:
- [09-Rule 30 元胞自动机：复杂性的可视化](09-rule-30-cellular-automata.md)
- [10-咖啡混合模拟：离散化的复杂动力学](10-coffee-cup-simulation.md)
- [11-熵增长测量：从理论到代码](11-entropy-measurement.md)

### 5. 深入拓展篇 :point_down:
- [12-算法统计学：随机性的结构](12-algorithmic-statistics.md)
- [13-开放问题与未来方向](13-open-problems.md)

## 参考资源

### 原始论文
- **Scott Aaronson**, "The First Law of Complexodynamics", 2011
  - 博客原文: https://scottaaronson.blog/?p=762
  - 这是理解复杂动力学的起点

### 相关论文
- **Claude Shannon**, "A Mathematical Theory of Communication", 1948
  - 信息论的奠基之作
- **Andrey Kolmogorov**, "Three Approaches to the Quantitative Definition of Information", 1965
  - 算法信息论的开创性论文
- **Gács, Tromp, Vitányi**, "Algorithmic Statistics", 2001
  - 算法统计学的权威综述

### 推荐书籍
- Sean Carroll, "From Eternity to Here" - 关于时间和熵的科普佳作
- Stephen Wolfram, "A New Kind of Science" - 元胞自动机与复杂性
- Ming Li & Paul Vitányi, "An Introduction to Kolmogorov Complexity and Its Applications"

## 有错误怎么办？

如果发现教程中有错误或不清晰的地方，欢迎：
1. 提交 Issue 到项目仓库
2. 直接联系作者进行讨论

## 公众号推广

关注「云言 AI」公众号，获取更多 AI 和深度学习相关的图解教程！

![公众号二维码](https://cdn.example.com/qrcode.png)

---

*本教程遵循 Creative Commons BY-NC-SA 4.0 协议*
