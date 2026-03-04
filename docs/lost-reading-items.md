---
created: 2026-03-04T09:23:42 (UTC +08:00)
tags: []
source: https://tensorlabbet.com/2024/11/11/lost-reading-items/
author:
---

# 遗失的阅读材料 · Tensor Labbet

> ## Excerpt
> 2024年11月11日，太郎·朗纳

---
## 遗失的阅读物品

2024年11月11日，太郎·朗纳

本文内容：尝试重构伊利亚·苏茨克维尔2020年的人工智能阅读清单

_（阅读需8分钟）_

我最近[分享了一份据称由伊利亚·苏茨克维尔 (Ilya Sutskever) 撰写的](https://tensorlabbet.com/2024/09/24/ai-reading-list/)[病毒式传播的人工智能阅读](https://arc.net/folder/D0472A20-9C20-4D3F-B145-D2865C0A9FEE)清单的摘要，该清单声称涵盖了 2020 年“ _90% 的重要内容_”。这份摘要将阅读材料精简到原书字数的不到 1%，形成了我在阅读之前就希望看到的 TL;DR（太长不看版）。

网上流传的这份书单版本并不完整，仅包含[约40篇](https://dallasinnovates.com/exclusive-qa-john-carmacks-different-path-to-artificial-general-intelligence/?utm_source=www.turingpost.com&utm_medium=referral&utm_campaign=the-mysterious-ai-reading-list-ilya-sutskever-s-recommendations)原始阅读材料中的[27篇](https://x.com/andrew_n_carr/status/1752526711311507526)。其余材料[据称](https://news.ycombinator.com/item?id=34641359)因Meta¹的电子邮件删除政策而被删除。这些缺失的阅读材料曾引发过一些有益的讨论，人们提出了许多不同的想法，认为哪些论文应该被收录。[](https://dallasinnovates.com/exclusive-qa-john-carmacks-different-path-to-artificial-general-intelligence/?utm_source=www.turingpost.com&utm_medium=referral&utm_campaign=the-mysterious-ai-reading-list-ilya-sutskever-s-recommendations)[](https://news.ycombinator.com/item?id=34641359)

本文旨在寻找这些遗失的阅读材料。文章基于从病毒式传播的书单、伊利亚·苏茨克维尔的近期演讲、OpenAI分享的资源等渠道收集的线索。

¹_更正：之前的版本错误地将此处提及的是OpenAI而非Meta。_

## 填补空白

主要证据是[与列表一起分享的一项声明，](https://x.com/andrew_n_carr/status/1752526711311507526)该声明称，一整套元_学习_论文都丢失了。

元学习通常被认为追求_“学习如何学习”_，即训练神经网络使其具备更强的通用能力，从而更容易地适应仅有少量训练样本的新任务。因此，网络应该能够利用其现有的权重，而无需在新数据上从头开始进行全新的训练。_单样本学习_仅向模型提供一个训练样本，模型需要从中学习新的下游任务；而_零样本_学习则完全不提供任何带标注的训练样本。

对于以下列出的部分候选论文，OpenAI 官方的认可进一步增强了其说服力。Ilya Sutskever 曾任 OpenAI 首席科学家，当时 OpenAI 发布了教育资源[《深度强化学习入门》（Spinning Up in Deep RL），](https://spinningup.openai.com/en/latest/index.html)其中收录了部分候选论文，并将其列入一份独立的“[深度强化学习关键论文”阅读清单，该清单包含 105 篇论文](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#meta-rl)。以下列出的论文中，也出现在该清单中的论文用符号 (⚛) 标记。

### 从保存下来的阅读材料中寻找线索

即使在已知文献列表中，也能找到一些元学习概念。保留下来的阅读材料可以围绕[记忆增强神经网络（MANN）](https://arxiv.org/abs/1410.3916)这一相关研究分支，构建一个叙事弧。继[《神经图灵机》（NTM）](https://tensorlabbet.com/2024/09/24/ai-reading-list/#NTM)论文之后，[《Set2Set》](https://tensorlabbet.com/2024/09/24/ai-reading-list/#Set2Set)和[《关系型RNN》](https://tensorlabbet.com/2024/09/24/ai-reading-list/#RelationalRNN)尝试使用外部存储器，供RNN读写信息。它们直接引用或密切相关于多篇论文，而这些论文很可能原本就包含在原始列表中：

**潜在阅读材料（第一部分）：**

-   [](http://proceedings.mlr.press/v48/santoro16.pdf)_2016 年发表的_[《基于记忆增强神经网络的元学习》](http://proceedings.mlr.press/v48/santoro16.pdf)

-   [](https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)_2017 年发表的_[《少样本学习的原型网络》](https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)

-   [《面向深度网络快速适应的模型无关元学习》](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf) ⚛

    _2017年_

### 从当代演讲中寻找线索

在这一时期，伊利亚·苏茨克维尔 (Ilya Sutskever) 的一系列演讲中也反复出现了一些关于元学习和_竞争性自我博弈的论文，这些论文很可能最终也被列入了阅读清单。_

**录制演讲：**

[\- 元学习与自博弈 - Ilya Sutskever，OpenAI（YouTube）](https://www.youtube.com/watch?v=BCzFs9Xb9_o)，2017

[\- OpenAI - 元学习与自博弈 - Ilya Sutskever（YouTube）](https://www.youtube.com/watch?v=AopSlxNYqX8)，2018

[\- Ilya Sutskever：OpenAI 元学习与自博弈（YouTube）](https://www.youtube.com/watch?v=9EN_HoEk3KY)，2018

这些报告的内容大多重叠，并反复引用阅读清单中的已知内容。报告首先阐述深度学习的根本原理，将神经网络的反向传播视为寻找符合[最小描述长度原则的](https://tensorlabbet.com/2024/09/24/ai-reading-list/#MDL)_小型电路的过程_。根据该原则，能够解释给定数据的最短程序将达到最佳泛化能力。[](https://tensorlabbet.com/2024/09/24/ai-reading-list/#MDL)

接下来，这三篇报告都引用了以下元学习论文：

**潜在阅读材料（第二部分）：**

-   [](https://amygdala.psychdept.arizona.edu/labspace/JclubLabMeetings/Lijuan-Science-2015-Lake-1332-8.pdf)_Lake等人（2016）_

    提出的[“通过概率程序归纳进行人类水平概念学习”](https://amygdala.psychdept.arizona.edu/labspace/JclubLabMeetings/Lijuan-Science-2015-Lake-1332-8.pdf)
-   [](https://arxiv.org/pdf/1611.01578)_Zoph 和 Le 在 2017 年_

    发表的论文[《基于强化学习的神经架构搜索》](https://arxiv.org/pdf/1611.01578)
-   [“一个简单的神经注意力元学习器”](https://arxiv.org/pdf/1707.03141) ⚛

    （_Mishra等人，2017）_

强化学习（RL）在所有三场报告中都占据了重要地位，并与元学习密切相关。其中一个关键概念是_竞争性自我博弈_，即智能体在模拟环境中进行互动，以达成特定的、通常具有对抗性的目标。作为一种_“将计算转化为数据”的_方法，这种方法使模拟智能体能够超越人类冠军，并在基于规则的博弈中创造出新的招式。伊利亚·苏茨克维尔（Ilya Sutskever）从进化生物学的角度出发，将竞争性自我博弈与社会互动对大脑容量的影响联系起来[（付费链接）](https://www.science.org/doi/10.1126/science.1098410)。他进一步指出，在模拟的_“智能体社会”_中快速提升能力，最终可能为实现某种形式的通用人工智能（AGI）提供[一条可行的途径](https://www.youtube.com/watch?v=9EN_HoEk3KY&t=2325s)。

鉴于他对这些概念的重视程度，一些被引用的关于自我博弈的论文后来可能也被纳入了阅读清单。这些论文可能构成缺失条目的相当一部分，尤其是在其他已保留的阅读条目中，[只有一篇提到了强化学习（RL）。](https://tensorlabbet.com/2024/09/24/ai-reading-list/#MachineSuperIntelligence)

**潜在阅读材料（第三部分）：**

-   [“事后经验重现”](https://proceedings.neurips.cc/paper_files/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf) ⚛

    （_Andrychowicz 等人，2017 年）_
-   [“基于深度强化学习的连续控制”](https://arxiv.org/abs/1509.02971) ⚛

    简称_DDPG：深度确定性策略梯度，2015_
-   [](https://arxiv.org/abs/1710.06537)_Peng等人于2017年_

    发表了题为[“基于动力学随机化的机器人控制仿真到实际迁移”的研究论文。](https://arxiv.org/abs/1710.06537)
-   [“元学习共享层级结构”](https://arxiv.org/abs/1710.09767)

    如_Frans 等人，2017 年_
-   [“时间差分学习和TD-Gammon \[1995\]”](https://www.csd.uwo.ca/~xling/cs346a/extra/tdgammon.pdf)

    如_Tesauro等人，1992年_
-   [《卡尔·西姆斯 - 进化虚拟生物，进化模拟，1994》](https://www.youtube.com/watch?v=JBgG_VSP7f8&t=2s)

    作为_卡尔·西姆斯，1994 年（YouTube 视频 \[4:09\]）_
-   [](https://arxiv.org/abs/1710.03748)_Bansal等人于2017年_

    发表的论文[《通过多智能体竞争涌现的复杂性》](https://arxiv.org/abs/1710.03748)
-   [“基于人类偏好的深度强化学习”](https://arxiv.org/abs/1706.03741) ⚛

    （_Christiano 等人，2017 年_）（注：引入了 RLHF）

即使在今天，这些大约在2018年左右的演讲仍然值得一看。除了引人入胜的知识点之外，它们还包含一些精彩的论述，例如：

> [“就像在人类世界一样：人类之所以觉得生活艰难，是因为其他人。”](https://www.youtube.com/watch?v=BCzFs9Xb9_o&t=2532s)
>
> 伊利亚·苏茨克维尔

因此，计算机科学中的一些概念似乎永不过时，但其他一些观点在今天看来可能会令人惊讶，例如一位听众在问答环节中的随意发言：

> [“看来，通往通用人工智能（AGI）道路上的一个重要子问题是理解语言，而目前生成式语言建模的现状相当糟糕。”](https://www.youtube.com/watch?v=9EN_HoEk3KY&t=3082s)
>
> \-观众

对此，伊利亚·苏茨克维尔回应道：

> [“即使没有超越现有模型的任何特别创新，仅仅将现有模型扩展到更大的数据集上，也能取得令人惊讶的成效。”](https://www.youtube.com/watch?v=9EN_HoEk3KY&t=3106s)
>
> \-伊利亚·苏茨克维尔（2018年）

这一观点后来在阅读材料[《神经语言模型的缩放定律》](https://tensorlabbet.com/2024/09/24/ai-reading-list/#ScalingLaws) （与 Rich Sutton 的[《惨痛的教训》](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)遥相呼应）中的实验结果得到了证实。最终，事实证明他的观点是正确的，因为他负责监督 Transformer 架构的扩展[，使其参数数量估计达到 1.8 万亿，在 128 个 GPU 上训练的成本超过 6000 万美元，](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/)从而构建出大型语言模型 (LLM)。如今，这些 LLM 能够生成越来越难以与人类书写区分开来的文本。

## 荣誉提名

最初的名单上可能还有许多其他作品和作者，但从这里开始，证据就越来越不足了。

总体而言，保留下来的阅读材料在涵盖不同模型类别、应用和理论方面取得了令人印象深刻的平衡，同时也囊括了该领域众多知名作者的作品。或许值得一提的是，其中一些例外情况也值得关注，即便它们可能被遗漏在了最初书单中那_“真正重要的10%”之列。_

因此，似乎可以合理地将其纳入：

-   [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun)在 CNN 的实际应用方面做出了[开创性工作](https://hal.science/hal-03926082/document)
-   [伊恩·古德费洛（Ian Goodfellow）](https://en.wikipedia.org/wiki/Ian_Goodfellow)及其[开发的生成对抗网络（GAN）](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html)在当时主导了图像生成领域。
-   [Demis Hassabis](https://en.wikipedia.org/wiki/Demis_Hassabis)因其在[AlphaFold方面的](https://ccsp.hms.harvard.edu/wp-content/uploads/2020/11/AlphaFold-at-CASP13-AlQuraishi.pdf)[强化学习研究](https://daiwk.github.io/assets/dqn.pdf)而荣获诺贝尔奖。[](https://ccsp.hms.harvard.edu/wp-content/uploads/2020/11/AlphaFold-at-CASP13-AlQuraishi.pdf)

## 结论

在获得更多信息之前，这篇文章仍将主要停留在推测阶段。毕竟，就连那份广为流传的书单本身也从未被官方证实为真。尽管如此，上面列出的那些可能遗失的阅读材料似乎值得分享。综合来看，它们或许能够填补那份广为流传的书单中的一个空白，用作者的话来说，这大概相当于当时缺失的_“30%的重要内容” 。_
