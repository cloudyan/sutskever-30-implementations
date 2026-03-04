# Kolmogorov 复杂度 K(x) 是怎么衡量“信息量”的?

问下大家,你有没有见过这两串东西:

- `00000000000000000000...`
- `10110010111001011100...`

直觉上第一串“很简单”,第二串“更随机”。

那问题来了:

**我能不能用一个统一的量,来衡量一个对象到底有多“复杂/信息量多大”?**

晓寒第一次学到 Kolmogorov complexity 的定义时,直接被震住了:

> K(x) = 生成 x 的最短程序长度

卧槽,这句话把“复杂度/压缩/随机性/学习”全部连成了一条线。

这篇按 notebook `25_kolmogorov_complexity.ipynb` 的实现,用可跑的压缩实验建立直觉,并解释三件事:

1) K(x) 为什么是“终极的压缩定义”
2) 为什么 K(x) 一般不可计算(uncomputable)
3) 为什么它会变成机器学习里 Occam's Razor、MDL、正则化的理论底座

## 1) K(x) 的定义:最短程序

形式化定义(直觉版):

\[
K(x) = \min_{p: U(p)=x} |p|
\]

- U:某个固定的通用图灵机(通用解释器)
- p:程序
- |p|:程序长度(比特数)

所以:

- 结构强的数据 → 可以用短程序生成 → K(x) 小
- 随机的数据 → 只能“原样打印” → K(x) 接近 |x|

生活类比:

"ABCABCABC..." 你可以说“重复 ABC 333 次”。

但一串真正随机的 0/1,你基本只能把它整串抄下来。

## 2) 现实里怎么“估计” K(x)?用压缩算法做上界

K(x) 的理论最优压缩不可实现,但我们可以用 gzip/zlib 这类压缩器做近似:

- 压缩后字节数是 K(x) 的一个上界(upper bound)
- 压缩率越接近 1,越随机(不可压缩)

```python
import numpy as np
import zlib
import gzip
import io

np.random.seed(42)


def estimate_kolmogorov_via_compression(s, method='zlib'):
    if isinstance(s, str):
        s = s.encode('utf-8')

    if method == 'zlib':
        compressed = zlib.compress(s, level=9)
    elif method == 'gzip':
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
            f.write(s)
        compressed = buf.getvalue()
    else:
        raise ValueError('unknown method')

    return len(compressed)


def compression_ratio(s, method='zlib'):
    b = s.encode('utf-8') if isinstance(s, str) else s
    if len(b) == 0:
        return 0
    return estimate_kolmogorov_via_compression(b, method) / len(b)
```

你可以拿几种典型字符串测一下:

- 全 0
- 重复模式
- 英文文本
- 随机二进制

通常会看到:

- 模式越强 → 压缩率越小
- 越随机 → 压缩率越接近 1

## 3) 为什么 K(x) 不可计算?

核心原因是它和停机问题(halting problem)同级别。

直觉说法:

如果你能计算 K(x),你就能判断“是否存在一个更短程序输出 x”,这会让你能解决一类不可判定问题。

notebook 里用 Berry paradox 的直觉做了一个非严格演示:

"找一个压缩率最高(最不可压缩)的字符串" 这件事本身,如果能被一个短算法稳定做到,就会产生自指悖论味道。

结论:

**K(x) 是“理想信息度量”,但它一般不可计算。**

所以我们才需要 MDL(可计算近似)、gzip(工程近似)、L1/L2(更粗的近似)。

## 4) 随机性=不可压缩(算法随机性)

算法信息论里一个很漂亮的结论:

**一个对象越随机,就越不可压缩。**

所以“随机性”不再是“看起来乱”,而是“任何短程序都生成不了它”。

压缩实验就能给你非常强的直觉支撑。

## 5) 不变性定理:换一台通用机,差一个常数

你可能会担心:

"K(x) 依赖具体的编程语言/解释器,那不就不客观了?"

不变性定理(invariance theorem)说:

对任意两台通用机 U1/U2,存在常数 c,使得:

\[
|K_{U1}(x) - K_{U2}(x)| \le c
\]

直觉:

换语言只会多一个“翻译器”的固定开销。

所以对足够长的 x,差异是常数项,不会改变“谁更复杂”的排序。

notebook 用 zlib/gzip 的对比做了一个经验验证:它们的估计差异通常像常数偏移。

## 6) Shannon 熵 vs Kolmogorov 复杂度:一个是平均,一个是个体

- Shannon entropy H(X) 需要分布 p(x),衡量的是随机变量的平均信息量
- K(x) 是“单个样本”的信息量

理论上对典型样本有关系:

\[
\mathbb{E}[K(X)] \approx H(X)\cdot |x| + O(\log|x|)
\]

notebook 用不同偏置的二进制串做了散点图,能看到 K(x)/|x| 和 H(X) 的相关性。

## 7) 算法概率 P(x)=2^{-K(x)}:Occam's Razor 的数学化

Solomonoff 的一个核心洞察:

\[
P(x) \approx 2^{-K(x)}
\]

意思是:

- 简单的序列(K 小)有更高先验概率
- 复杂的序列(K 大)先验更低

这句话基本就是 Occam's Razor 的“硬核版本”。

```python
def algorithmic_probability_approximation(x):
    K = estimate_kolmogorov_via_compression(x)
    return 2 ** (-K)
```

## 8) 和机器学习的关系:为什么正则化/剪枝/MDL有效?

一旦你接受“学习=压缩/找到短描述”,很多工程现象就突然合理了:

- 正则化(L1/L2):惩罚复杂度,偏向低 K 的模型
- 剪枝/量化/蒸馏:让模型更可压缩,也常常泛化更好
- MDL(Paper 23):在可计算范围内近似 K(x)

一句话:

**能压缩数据的模型,往往就是理解了数据结构的模型。**

## 小结

Kolmogorov complexity 这篇你记住 4 件事就够了:

1) K(x)=最短程序长度 → “信息量”的终极定义
2) K(x) 一般不可计算 → 只能做近似
3) 随机性=不可压缩 → 随机性有了可操作定义
4) Occam/MDL/正则化/泛化 → 都能用 K 的视角串起来

## 练习题

1) 用 zlib/gzip 测试更多对象:JSON、代码、日志、图片(先转 bytes),压缩率差异说明了什么?

2) 自己实现一个简单的 RLE(游程编码)压缩器,看看它对哪些模式有效,对哪些无效。

3) 思考题:为什么“圆周率的数字”看起来随机,但 K(π) 很小?

4) 思考题:现代 LLM 的参数很多,看起来很复杂,它的 K(model) 真的是很大吗?预训练在“搜索空间”上起了什么作用?

## 延伸阅读

1) Kolmogorov, Chaitin, Solomonoff 的算法信息论经典工作

2) Paper 23:MDL(可计算的 K 近似)

3) Universal Intelligence/AIXI(Paper 24):P(x)=2^{-K(x)} 的归纳基础

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 25 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
