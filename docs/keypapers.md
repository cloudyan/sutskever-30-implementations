---
created: 2026-03-04T09:21:01 (UTC +08:00)
tags: []
source: https://spinningup.openai.com/en/latest/spinningup/keypapers.html#meta-rl
author:
---

# Key Papers in Deep RL — Spinning Up

> ## Excerpt
> What follows is a list of papers in deep RL that are worth reading. This is far from comprehensive, but should provide a useful starting point for someone looking to do research in the field.

---
[Spinning Up](https://spinningup.openai.com/en/latest/index.html)

What follows is a list of papers in deep RL that are worth reading. This is _far_ from comprehensive, but should provide a useful starting point for someone looking to do research in the field.

Table of Contents

-   [Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#key-papers-in-deep-rl)
    -   [1\. Model-Free RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#model-free-rl)
    -   [2\. Exploration](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#exploration)
    -   [3\. Transfer and Multitask RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#transfer-and-multitask-rl)
    -   [4\. Hierarchy](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#hierarchy)
    -   [5\. Memory](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#memory)
    -   [6\. Model-Based RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#model-based-rl)
    -   [7\. Meta-RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#meta-rl)
    -   [8\. Scaling RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#scaling-rl)
    -   [9\. RL in the Real World](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#rl-in-the-real-world)
    -   [10\. Safety](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#safety)
    -   [11\. Imitation Learning and Inverse Reinforcement Learning](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#imitation-learning-and-inverse-reinforcement-learning)
    -   [12\. Reproducibility, Analysis, and Critique](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#reproducibility-analysis-and-critique)
    -   [13\. Bonus: Classic Papers in RL Theory or Review](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#bonus-classic-papers-in-rl-theory-or-review)

## [1\. Model-Free RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id107)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#model-free-rl "Permalink to this headline")

### a. Deep Q-Learning[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#a-deep-q-learning "Permalink to this headline")

|\[1\]|[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih et al, 2013. **Algorithm: DQN.**|
|---|---|

|\[2\]|[Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527), Hausknecht and Stone, 2015. **Algorithm: Deep Recurrent Q-Learning.**|
|---|---|

|\[3\]|[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), Wang et al, 2015. **Algorithm: Dueling DQN.**|
|---|---|

|\[4\]|[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), Hasselt et al 2015. **Algorithm: Double DQN.**|
|---|---|

|\[5\]|[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), Schaul et al, 2015. **Algorithm: Prioritized Experience Replay (PER).**|
|---|---|

|\[6\]|[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298), Hessel et al, 2017. **Algorithm: Rainbow DQN.**|
|---|---|

### b. Policy Gradients[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#b-policy-gradients "Permalink to this headline")

|\[7\]|[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih et al, 2016. **Algorithm: A3C.**|
|---|---|

|\[8\]|[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman et al, 2015. **Algorithm: TRPO.**|
|---|---|

|\[9\]|[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), Schulman et al, 2015. **Algorithm: GAE.**|
|---|---|

|\[10\]|[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al, 2017. **Algorithm: PPO-Clip, PPO-Penalty.**|
|---|---|

|\[11\]|[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286), Heess et al, 2017. **Algorithm: PPO-Penalty.**|
|---|---|

|\[12\]|[Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144), Wu et al, 2017. **Algorithm: ACKTR.**|
|---|---|

|\[13\]|[Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224), Wang et al, 2016. **Algorithm: ACER.**|
|---|---|

|\[14\]|[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja et al, 2018. **Algorithm: SAC.**|
|---|---|

### c. Deterministic Policy Gradients[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#c-deterministic-policy-gradients "Permalink to this headline")

|\[15\]|[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf), Silver et al, 2014. **Algorithm: DPG.**|
|---|---|

|\[16\]|[Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971), Lillicrap et al, 2015. **Algorithm: DDPG.**|
|---|---|

|\[17\]|[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477), Fujimoto et al, 2018. **Algorithm: TD3.**|
|---|---|

### d. Distributional RL[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#d-distributional-rl "Permalink to this headline")

|\[18\]|[A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), Bellemare et al, 2017. **Algorithm: C51.**|
|---|---|

|\[19\]|[Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044), Dabney et al, 2017. **Algorithm: QR-DQN.**|
|---|---|

|\[20\]|[Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923), Dabney et al, 2018. **Algorithm: IQN.**|
|---|---|

|\[21\]|[Dopamine: A Research Framework for Deep Reinforcement Learning](https://openreview.net/forum?id=ByG_3s09KX), Anonymous, 2018. **Contribution:** Introduces Dopamine, a code repository containing implementations of DQN, C51, IQN, and Rainbow. [Code link.](https://github.com/google/dopamine)|
|---|---|

### e. Policy Gradients with Action-Dependent Baselines[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#e-policy-gradients-with-action-dependent-baselines "Permalink to this headline")

|\[22\]|[Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](https://arxiv.org/abs/1611.02247), Gu et al, 2016. **Algorithm: Q-Prop.**|
|---|---|

|\[23\]|[Action-depedent Control Variates for Policy Optimization via Stein’s Identity](https://arxiv.org/abs/1710.11198), Liu et al, 2017. **Algorithm: Stein Control Variates.**|
|---|---|

|\[24\]|[The Mirage of Action-Dependent Baselines in Reinforcement Learning](https://arxiv.org/abs/1802.10031), Tucker et al, 2018. **Contribution:** interestingly, critiques and reevaluates claims from earlier papers (including Q-Prop and stein control variates) and finds important methodological errors in them.|
|---|---|

### f. Path-Consistency Learning[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#f-path-consistency-learning "Permalink to this headline")

|\[25\]|[Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892), Nachum et al, 2017. **Algorithm: PCL.**|
|---|---|

|\[26\]|[Trust-PCL: An Off-Policy Trust Region Method for Continuous Control](https://arxiv.org/abs/1707.01891), Nachum et al, 2017. **Algorithm: Trust-PCL.**|
|---|---|

### g. Other Directions for Combining Policy-Learning and Q-Learning[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#g-other-directions-for-combining-policy-learning-and-q-learning "Permalink to this headline")

|\[27\]|[Combining Policy Gradient and Q-learning](https://arxiv.org/abs/1611.01626), O’Donoghue et al, 2016. **Algorithm: PGQL.**|
|---|---|

|\[28\]|[The Reactor: A Fast and Sample-Efficient Actor-Critic Agent for Reinforcement Learning](https://arxiv.org/abs/1704.04651), Gruslys et al, 2017. **Algorithm: Reactor.**|
|---|---|

|\[29\]|[Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning](http://papers.nips.cc/paper/6974-interpolated-policy-gradient-merging-on-policy-and-off-policy-gradient-estimation-for-deep-reinforcement-learning), Gu et al, 2017. **Algorithm: IPG.**|
|---|---|

|\[30\]|[Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/abs/1704.06440), Schulman et al, 2017. **Contribution:** Reveals a theoretical link between these two families of RL algorithms.|
|---|---|

### h. Evolutionary Algorithms[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#h-evolutionary-algorithms "Permalink to this headline")

|\[31\]|[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864), Salimans et al, 2017. **Algorithm: ES.**|
|---|---|

## [2\. Exploration](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id108)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#exploration "Permalink to this headline")

### a. Intrinsic Motivation[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#a-intrinsic-motivation "Permalink to this headline")

|\[32\]|[VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674), Houthooft et al, 2016. **Algorithm: VIME.**|
|---|---|

|\[33\]|[Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868), Bellemare et al, 2016. **Algorithm: CTS-based Pseudocounts.**|
|---|---|

|\[34\]|[Count-Based Exploration with Neural Density Models](https://arxiv.org/abs/1703.01310), Ostrovski et al, 2017. **Algorithm: PixelCNN-based Pseudocounts.**|
|---|---|

|\[35\]|[#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](https://arxiv.org/abs/1611.04717), Tang et al, 2016. **Algorithm: Hash-based Counts.**|
|---|---|

|\[36\]|[EX2: Exploration with Exemplar Models for Deep Reinforcement Learning](https://arxiv.org/abs/1703.01260), Fu et al, 2017. **Algorithm: EX2.**|
|---|---|

|\[37\]|[Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363), Pathak et al, 2017. **Algorithm: Intrinsic Curiosity Module (ICM).**|
|---|---|

|\[38\]|[Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355), Burda et al, 2018. **Contribution:** Systematic analysis of how surprisal-based intrinsic motivation performs in a wide variety of environments.|
|---|---|

|\[39\]|[Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894), Burda et al, 2018. **Algorithm: RND.**|
|---|---|

### b. Unsupervised RL[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#b-unsupervised-rl "Permalink to this headline")

|\[40\]|[Variational Intrinsic Control](https://arxiv.org/abs/1611.07507), Gregor et al, 2016. **Algorithm: VIC.**|
|---|---|

|\[41\]|[Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070), Eysenbach et al, 2018. **Algorithm: DIAYN.**|
|---|---|

|\[42\]|[Variational Option Discovery Algorithms](https://arxiv.org/abs/1807.10299), Achiam et al, 2018. **Algorithm: VALOR.**|
|---|---|

## [3\. Transfer and Multitask RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id109)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#transfer-and-multitask-rl "Permalink to this headline")

|\[43\]|[Progressive Neural Networks](https://arxiv.org/abs/1606.04671), Rusu et al, 2016. **Algorithm: Progressive Networks.**|
|---|---|

|\[44\]|[Universal Value Function Approximators](http://proceedings.mlr.press/v37/schaul15.pdf), Schaul et al, 2015. **Algorithm: UVFA.**|
|---|---|

|\[45\]|[Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397), Jaderberg et al, 2016. **Algorithm: UNREAL.**|
|---|---|

|\[46\]|[The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously](https://arxiv.org/abs/1707.03300), Cabi et al, 2017. **Algorithm: IU Agent.**|
|---|---|

|\[47\]|[PathNet: Evolution Channels Gradient Descent in Super Neural Networks](https://arxiv.org/abs/1701.08734), Fernando et al, 2017. **Algorithm: PathNet.**|
|---|---|

|\[48\]|[Mutual Alignment Transfer Learning](https://arxiv.org/abs/1707.07907), Wulfmeier et al, 2017. **Algorithm: MATL.**|
|---|---|

|\[49\]|[Learning an Embedding Space for Transferable Robot Skills](https://openreview.net/forum?id=rk07ZXZRb&noteId=rk07ZXZRb), Hausman et al, 2018.|
|---|---|

|\[50\]|[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495), Andrychowicz et al, 2017. **Algorithm: Hindsight Experience Replay (HER).**|
|---|---|

## [4\. Hierarchy](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id110)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#hierarchy "Permalink to this headline")

|\[51\]|[Strategic Attentive Writer for Learning Macro-Actions](https://arxiv.org/abs/1606.04695), Vezhnevets et al, 2016. **Algorithm: STRAW.**|
|---|---|

|\[52\]|[FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161), Vezhnevets et al, 2017. **Algorithm: Feudal Networks**|
|---|---|

|\[53\]|[Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296), Nachum et al, 2018. **Algorithm: HIRO.**|
|---|---|

## [5\. Memory](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id111)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#memory "Permalink to this headline")

|\[54\]|[Model-Free Episodic Control](https://arxiv.org/abs/1606.04460), Blundell et al, 2016. **Algorithm: MFEC.**|
|---|---|

|\[55\]|[Neural Episodic Control](https://arxiv.org/abs/1703.01988), Pritzel et al, 2017. **Algorithm: NEC.**|
|---|---|

|\[56\]|[Neural Map: Structured Memory for Deep Reinforcement Learning](https://arxiv.org/abs/1702.08360), Parisotto and Salakhutdinov, 2017. **Algorithm: Neural Map.**|
|---|---|

|\[57\]|[Unsupervised Predictive Memory in a Goal-Directed Agent](https://arxiv.org/abs/1803.10760), Wayne et al, 2018. **Algorithm: MERLIN.**|
|---|---|

|\[58\]|[Relational Recurrent Neural Networks](https://arxiv.org/abs/1806.01822), Santoro et al, 2018. **Algorithm: RMC.**|
|---|---|

## [6\. Model-Based RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id112)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#model-based-rl "Permalink to this headline")

### a. Model is Learned[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#a-model-is-learned "Permalink to this headline")

|\[59\]|[Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/abs/1707.06203), Weber et al, 2017. **Algorithm: I2A.**|
|---|---|

|\[60\]|[Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning](https://arxiv.org/abs/1708.02596), Nagabandi et al, 2017. **Algorithm: MBMF.**|
|---|---|

|\[61\]|[Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning](https://arxiv.org/abs/1803.00101), Feinberg et al, 2018. **Algorithm: MVE.**|
|---|---|

|\[62\]|[Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675), Buckman et al, 2018. **Algorithm: STEVE.**|
|---|---|

|\[63\]|[Model-Ensemble Trust-Region Policy Optimization](https://openreview.net/forum?id=SJJinbWRZ&noteId=SJJinbWRZ), Kurutach et al, 2018. **Algorithm: ME-TRPO.**|
|---|---|

|\[64\]|[Model-Based Reinforcement Learning via Meta-Policy Optimization](https://arxiv.org/abs/1809.05214), Clavera et al, 2018. **Algorithm: MB-MPO.**|
|---|---|

|\[65\]|[Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/abs/1809.01999), Ha and Schmidhuber, 2018.|
|---|---|

### b. Model is Given[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#b-model-is-given "Permalink to this headline")

|\[66\]|[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815), Silver et al, 2017. **Algorithm: AlphaZero.**|
|---|---|

|\[67\]|[Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439), Anthony et al, 2017. **Algorithm: ExIt.**|
|---|---|

## [7\. Meta-RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id113)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#meta-rl "Permalink to this headline")

|\[68\]|[RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779), Duan et al, 2016. **Algorithm: RL^2.**|
|---|---|

|\[69\]|[Learning to Reinforcement Learn](https://arxiv.org/abs/1611.05763), Wang et al, 2016.|
|---|---|

|\[70\]|[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400), Finn et al, 2017. **Algorithm: MAML.**|
|---|---|

|\[71\]|[A Simple Neural Attentive Meta-Learner](https://openreview.net/forum?id=B1DmUzWAW&noteId=B1DmUzWAW), Mishra et al, 2018. **Algorithm: SNAIL.**|
|---|---|

## [8\. Scaling RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id114)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#scaling-rl "Permalink to this headline")

|\[72\]|[Accelerated Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1803.02811), Stooke and Abbeel, 2018. **Contribution:** Systematic analysis of parallelization in deep RL across algorithms.|
|---|---|

|\[73\]|[IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561), Espeholt et al, 2018. **Algorithm: IMPALA.**|
|---|---|

|\[74\]|[Distributed Prioritized Experience Replay](https://openreview.net/forum?id=H1Dy---0Z), Horgan et al, 2018. **Algorithm: Ape-X.**|

|\[75\]|[Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX), Anonymous, 2018. **Algorithm: R2D2.**|
|---|---|

|\[76\]|[RLlib: Abstractions for Distributed Reinforcement Learning](https://arxiv.org/abs/1712.09381), Liang et al, 2017. **Contribution:** A scalable library of RL algorithm implementations. [Documentation link.](https://ray.readthedocs.io/en/latest/rllib.html)|
|---|---|

## [9\. RL in the Real World](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id115)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#rl-in-the-real-world "Permalink to this headline")

|\[77\]|[Benchmarking Reinforcement Learning Algorithms on Real-World Robots](https://arxiv.org/abs/1809.07731), Mahmood et al, 2018.|
|---|---|

|\[78\]|[Learning Dexterous In-Hand Manipulation](https://arxiv.org/abs/1808.00177), OpenAI, 2018.|
|---|---|

|\[79\]|[QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://arxiv.org/abs/1806.10293), Kalashnikov et al, 2018. **Algorithm: QT-Opt.**|
|---|---|

|\[80\]|[Horizon: Facebook’s Open Source Applied Reinforcement Learning Platform](https://arxiv.org/abs/1811.00260), Gauci et al, 2018.|
|---|---|

## [10\. Safety](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id116)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#safety "Permalink to this headline")

|\[81\]|[Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565), Amodei et al, 2016. **Contribution:** establishes a taxonomy of safety problems, serving as an important jumping-off point for future research. We need to solve these!|
|---|---|

|\[82\]|[Deep Reinforcement Learning From Human Preferences](https://arxiv.org/abs/1706.03741), Christiano et al, 2017. **Algorithm: LFP.**|
|---|---|

|\[83\]|[Constrained Policy Optimization](https://arxiv.org/abs/1705.10528), Achiam et al, 2017. **Algorithm: CPO.**|
|---|---|

|\[84\]|[Safe Exploration in Continuous Action Spaces](https://arxiv.org/abs/1801.08757), Dalal et al, 2018. **Algorithm: DDPG+Safety Layer.**|
|---|---|

|\[85\]|[Trial without Error: Towards Safe Reinforcement Learning via Human Intervention](https://arxiv.org/abs/1707.05173), Saunders et al, 2017. **Algorithm: HIRL.**|
|---|---|

|\[86\]|[Leave No Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning](https://arxiv.org/abs/1711.06782), Eysenbach et al, 2017. **Algorithm: Leave No Trace.**|
|---|---|

## [11\. Imitation Learning and Inverse Reinforcement Learning](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id117)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#imitation-learning-and-inverse-reinforcement-learning "Permalink to this headline")

|\[87\]|[Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy](http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf), Ziebart 2010. **Contributions:** Crisp formulation of maximum entropy IRL.|
|---|---|

|\[88\]|[Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/abs/1603.00448), Finn et al, 2016. **Algorithm: GCL.**|
|---|---|

|\[89\]|[Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476), Ho and Ermon, 2016. **Algorithm: GAIL.**|
|---|---|

|\[90\]|[DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/2018_TOG_DeepMimic.pdf), Peng et al, 2018. **Algorithm: DeepMimic.**|
|---|---|

|\[91\]|[Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](https://arxiv.org/abs/1810.00821), Peng et al, 2018. **Algorithm: VAIL.**|
|---|---|

|\[92\]|[One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL](https://arxiv.org/abs/1810.05017), Le Paine et al, 2018. **Algorithm: MetaMimic.**|
|---|---|

## [12\. Reproducibility, Analysis, and Critique](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id118)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#reproducibility-analysis-and-critique "Permalink to this headline")

|\[93\]|[Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/abs/1604.06778), Duan et al, 2016. **Contribution: rllab.**|
|---|---|

|\[94\]|[Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control](https://arxiv.org/abs/1708.04133), Islam et al, 2017.|
|---|---|

|\[95\]|[Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), Henderson et al, 2017.|
|---|---|

|\[96\]|[Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods](https://arxiv.org/abs/1810.02525), Henderson et al, 2018.|
|---|---|

|\[97\]|[Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms?](https://arxiv.org/abs/1811.02553), Ilyas et al, 2018.|
|---|---|

|\[98\]|[Simple Random Search Provides a Competitive Approach to Reinforcement Learning](https://arxiv.org/abs/1803.07055), Mania et al, 2018.|
|---|---|

|\[99\]|[Benchmarking Model-Based Reinforcement Learning](https://arxiv.org/abs/1907.02057), Wang et al, 2019.|
|---|---|

## [13\. Bonus: Classic Papers in RL Theory or Review](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#id119)[¶](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#bonus-classic-papers-in-rl-theory-or-review "Permalink to this headline")

|\[100\]|[Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), Sutton et al, 2000. **Contributions:** Established policy gradient theorem and showed convergence of policy gradient algorithm for arbitrary policy classes.|
|---|---|

|\[101\]|[An Analysis of Temporal-Difference Learning with Function Approximation](http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf), Tsitsiklis and Van Roy, 1997. **Contributions:** Variety of convergence results and counter-examples for value-learning methods in RL.|
|---|---|

|\[102\]|[Reinforcement Learning of Motor Skills with Policy Gradients](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Netw-2008-21-682_4867%5b0%5d.pdf), Peters and Schaal, 2008. **Contributions:** Thorough review of policy gradient methods at the time, many of which are still serviceable descriptions of deep RL methods.|
|---|---|

|\[103\]|[Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf), Kakade and Langford, 2002. **Contributions:** Early roots for monotonic improvement theory, later leading to theoretical justification for TRPO and other algorithms.|
|---|---|

|\[104\]|[A Natural Policy Gradient](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf), Kakade, 2002. **Contributions:** Brought natural gradients into RL, later leading to TRPO, ACKTR, and several other methods in deep RL.|
|---|---|

|\[105\]|[Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf), Szepesvari, 2009. **Contributions:** Unbeatable reference on RL before deep RL, containing foundations and theoretical background.|
|---|---|
