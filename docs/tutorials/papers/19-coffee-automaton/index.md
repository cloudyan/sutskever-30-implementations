# 咖啡自动机(The Coffee Automaton)是怎么解释“不可逆”的?

问下大家,你有没有干过这件事:

往一杯黑咖啡里倒一滴牛奶,搅一搅,它很快就变成一杯浅棕色的“拿铁”。

然后你盯着杯子发呆,心里冒出一个非常不科学的想法:

**它能不能自己再分回去?**

晓寒第一次认真想这个问题的时候,差点被自己绕晕:

- 微观物理定律(牛顿力学/量子力学)很多是可逆的
- 但宏观现象(混合、摩擦、扩散)几乎都是不可逆的

卧槽,那“时间方向”到底是从哪来的?

这篇我们按 notebook `19_coffee_automaton.ipynb` 的思路,用一系列小模拟把不可逆性拆开讲:

1) 扩散为什么天然“只会混,不会自己分”
2) 熵为什么会增长(以及怎么量化)
3) 微观可逆 vs 宏观不可逆(粗粒化是关键)
4) 庞加莱复现:理论上会复原,但要等到宇宙寿命的 10^(10^23) 倍
5) 麦克斯韦妖:信息是物理的(擦除信息要付熵的代价)
6) 计算不可逆:哈希/单向函数就是“信息粗粒化”
7) 机器学习的信息瓶颈:学习=压缩=不可逆

## 1. 一滴牛奶的扩散:最朴素的不可逆

我们用一个 2D 网格模拟“牛奶浓度场”:

- 1 表示牛奶,0 表示咖啡
- 初始:中心一个小圆点是牛奶
- 每一步做一次离散拉普拉斯扩散

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.stats import entropy as scipy_entropy


def initialize_coffee_cup(size: int = 64) -> np.ndarray:
    cup = np.zeros((size, size))
    center = size // 2
    radius = size // 8

    y, x = np.ogrid[:size, :size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    cup[mask] = 1.0
    return cup


def diffusion_step(concentration: np.ndarray, D: float = 0.1) -> np.ndarray:
    # 离散拉普拉斯核(∇²)
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

    laplacian = convolve(concentration, kernel, mode='constant', cval=0.0)

    dt = 0.1
    new_c = concentration + D * dt * laplacian
    return np.clip(new_c, 0, 1)


def simulate_coffee_mixing(steps: int = 200, D: float = 0.1):
    cup = initialize_coffee_cup()
    history = [cup.copy()]
    for _ in range(steps):
        cup = diffusion_step(cup, D)
        history.append(cup.copy())
    return history
```

跑一下关键帧,你会看到“牛奶只会越扩越散”。

这时候你可能会问:

"那我把时间反过来算不就行了?"

注意:扩散方程本质上是一个带耗散的宏观方程,它已经把微观信息丢掉了,所以天然带了“箭头”。

## 2. 熵怎么量化?我们用两个视角

一个最直观的做法:看浓度分布的 Shannon entropy(信息熵)。混得越均匀,熵越高。

```python
def compute_shannon_entropy(concentration: np.ndarray, num_bins: int = 10) -> float:
    flat = concentration.flatten()
    hist, _ = np.histogram(flat, bins=num_bins, range=(0, 1), density=True)
    hist = hist / hist.sum()
    return scipy_entropy(hist, base=2)


def compute_mixing_quality(concentration: np.ndarray) -> float:
    # 1 表示完全均匀,0 表示极不均匀
    mean = concentration.mean()
    var = np.var(concentration)
    max_var = mean * (1 - mean)
    if max_var == 0:
        return 1.0
    return 1 - (var / max_var)
```

你会观察到:

- Shannon entropy 随时间上升(更“无序/更均匀”)
- mixing quality 也随时间逼近 1(趋向平衡)

这就是热力学第二定律在一个玩具世界里的投影。

## 3. 微观可逆,为什么宏观不可逆?粗粒化(coarse-graining)是关键

接下来我们用粒子在盒子里弹来弹去做模拟:

- 微观态(microstate):每个粒子的精确位置/速度(理论上可逆)
- 宏观态(macrostate):把空间分成几个格子,只记录每格有多少粒子(这就是粗粒化)

一旦你只看宏观态,你就把大量微观信息“丢掉”了。

丢信息这件事,就是不可逆的根源。

```python
from dataclasses import dataclass
from typing import List


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float


def initialize_particles(num_particles: int = 200, region: str = 'left') -> List[Particle]:
    particles = []
    for _ in range(num_particles):
        x = np.random.uniform(0.1, 0.4) if region == 'left' else np.random.uniform(0.6, 0.9)
        y = np.random.uniform(0.1, 0.9)

        speed = 0.02
        angle = np.random.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        particles.append(Particle(x, y, vx, vy))
    return particles


def update_particles(particles: List[Particle], dt: float = 1.0) -> List[Particle]:
    out = []
    for p in particles:
        x = p.x + p.vx * dt
        y = p.y + p.vy * dt
        vx, vy = p.vx, p.vy

        if x < 0 or x > 1:
            vx = -vx
            x = np.clip(x, 0, 1)
        if y < 0 or y > 1:
            vy = -vy
            y = np.clip(y, 0, 1)

        out.append(Particle(float(x), float(y), float(vx), float(vy)))
    return out


def compute_macrostate(particles: List[Particle], num_bins: int = 4) -> np.ndarray:
    pos = np.array([[p.x, p.y] for p in particles])
    hist, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=num_bins, range=[[0, 1], [0, 1]])
    return hist


def compute_macrostate_entropy(macrostate: np.ndarray) -> float:
    counts = macrostate.flatten()
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))
```

你会看到一个很反直觉但非常关键的现象:

**粒子的运动规则是可逆的(你知道所有位置/速度就能倒推),但宏观熵还是会涨。**

原因就是:宏观态把很多微观态压扁到同一个桶里(多对一映射),而“多对一”天然不可逆。

## 4. 庞加莱复现:理论上会复原,但你等不起

很多人会说:"既然可逆,那总有一天会回到最初状态吧?"

对,这叫 Poincaré recurrence(庞加莱复现)。

但关键在于:复现时间随系统规模指数爆炸。

你可以先用一个很小的有限状态系统做演示:

```python
from typing import Tuple, List


def simple_phase_space_system(num_states: int = 16, num_steps: int = 200) -> Tuple[List[int], List[int]]:
    a, b = 3, 1
    initial = 0
    s = initial
    states = [s]
    rec = []
    for step in range(1, num_steps):
        s = (a * s + b) % num_states
        states.append(s)
        if s == initial:
            rec.append(step)
    return states, rec
```

对咖啡这种 N~10^23 的系统来说,复现时间大到离谱(常见的写法是 e^N 量级)。

所以结论是:

**不可逆不是“绝对”不可逆,而是“在宇宙寿命尺度上不可逆”。**

## 5. 麦克斯韦妖:智能能不能“逆熵”?

麦克斯韦妖的故事你可能听过:

- 有个小妖精守着门
- 它让快粒子去右边,慢粒子去左边
- 看起来像凭空制造了温差(降低熵)

真正的反转点在于 Landauer principle(朗道尔原理):

**擦除 1 bit 信息,至少要付出 k_B T ln(2) 的热(熵)代价。**

也就是说:妖精必须测量、记录、擦除记忆,而擦除会把熵补回去。

notebook 里给了一个简化模拟,核心结构是:

- 记录测量(speed)
- 记忆有限,需要定期擦除
- 擦除累积“熵成本”

你可以直接在 `19_coffee_automaton.ipynb` 里跑完整可视化。

## 6. 计算不可逆:哈希/单向函数就是“信息粗粒化”

你把一个 64-bit 输入哈希成 16-bit 输出,发生了什么?

- 很多输入映射到同一个输出(碰撞)
- 你丢掉了 48 bits 信息
- 丢信息=不可逆

```python
import hashlib


def cryptographic_hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def compute_information_loss(input_bits: int, output_bits: int) -> int:
    return max(0, input_bits - output_bits)
```

把这件事和“粗粒化”对上,你会突然发现:

**哈希就是一种离散的粗粒化。**

它把一堆微观输入归并成同一个宏观输出,因此反推几乎不可能。

## 7. 机器学习的信息瓶颈:学习=压缩=不可逆

神经网络为什么能泛化?

一个常见观点是 Information Bottleneck:

- 输入里有很多噪声/细节
- 好模型会把输入压缩成“任务相关”的表示
- 压缩本质上就是丢信息

notebook 里用一个简单的自编码器层堆叠,用激活分布的熵来粗略估计“信息量”,你会看到层越深(尤其瓶颈层),信息量会下降。

直觉总结一句:

**学会忽略细节,才有可能学到规律。**

## 8. 时间之箭:定律里没有方向,方向在初始条件里

把上面所有东西串起来,你会得到一个更大的结论:

- 微观定律多数可逆
- 宏观不可逆来自统计与粗粒化
- 时间方向来自宇宙极低熵的初始条件(大爆炸为什么低熵?这是深谜)

## 小结

咖啡自动机想告诉你的核心其实很统一:

1) **不可逆=信息丢失**(粗粒化、多对一映射)
2) **熵增长=系统更可能落入“宏观更常见”的状态**
3) **复现存在但等不起**(规模一大,时间指数爆炸)
4) **信息是物理的**(Landauer, 麦克斯韦妖)
5) **学习是不可逆的压缩**(信息瓶颈)

## 练习题

1) 扩散模拟里把边界条件从 `constant` 改成 `reflect` 或 `wrap`,混合速度和最终分布会怎么变?

2) 把宏观态的网格从 4x4 改成 8x8/16x16,宏观熵曲线会更“接近微观可逆”还是更“不可逆”?为什么?

3) 自己实现一个“可逆计算”小例子(例如 bijection 的 bit 操作),并对比哈希的不可逆。

4) 思考题:为什么神经网络的泛化常常依赖丢信息?有没有任务需要“尽量不丢信息”? 

## 延伸阅读

1) Landauer's principle(信息擦除的最低热耗)

2) Maxwell's demon(信息与热力学第二定律)

3) Poincaré recurrence(有限系统的复现)

4) Information Bottleneck(机器学习中的压缩视角)

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 19 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
