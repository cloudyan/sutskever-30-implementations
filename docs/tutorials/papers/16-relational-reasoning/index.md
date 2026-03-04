# 关系网络(Relation Network)是怎么做关系推理的?

问下大家,你有没有遇到过这种题:

"红色物体旁边最近的是哪个形状?"、"A 离谁最近?"、"谁和谁是一对?"

晓寒刚开始做视觉问答(VQA)的时候,经常卡在一个点:网络明明能看懂每个物体是什么,但一问到**物体之间的关系**(最近/最远/左边/同色/同形),它就开始胡说八道。

直到后来看到 **Relation Network(RN)** 这个结构,才发现卧槽,原来把“关系”显式地算一遍,模型就更容易学会推理!

这篇我们用纯 NumPy 写一个最小可运行的 Relation Network,并用一个简化版 Sort-of-CLEVR 生成数据,把 RN 的关键直觉一次讲透。

## 为什么“看见”不等于“会推理”? 

生活里你判断“谁离谁最近”,通常不会把所有人揉成一团再猜,而是会下意识做两步:

1) 先把人一个个识别出来(每个人的位置、衣服颜色、身高…)

2) 再把“某两个人之间”的关系拿出来比(距离/相似度/是否同组…)

标准的 CNN/MLP 很擅长第 1 步(提特征),但第 2 步如果不显式建模,它就只能“自己在特征里偷偷学”,学习难度会陡增。

RN 的核心思想很朴素:

把“所有物体两两配对”,对每一对算一次关系,然后把所有关系加起来,再输出答案。

## Relation Network 的核心公式

RN 通常写成:

\[
RN(O) = f_\phi\Big( \sum_{i,j} g_\theta(o_i, o_j, q) \Big)
\]

其中:

- \(o_i\): 第 i 个物体的表示(位置+属性)
- \(q\): 问题/查询(比如“离红色最近的那个”)的表示
- \(g_\theta\): “关系函数”,输入一对物体+问题,输出这对关系的向量
- \(f_\phi\): “聚合函数”,把所有关系向量求和后再映射成最终答案

你可以把它想成:

"先让一个小脑袋 g 去评估每一对人的关系,再让另一个小脑袋 f 汇总所有关系做决定。"

关键点有两个:

1) **显式两两比较**:推理从“隐式”变成“显式”
2) **求和聚合**:对物体顺序不敏感(天然 permutation invariant)

## 代码实现(纯 NumPy)

下面的实现完全对应上面的公式:用两个小 MLP 分别当作 \(g_\theta\) 和 \(f_\phi\)。

### 1) MLP 基础组件

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def relu(x):
    return np.maximum(0, x)


class MLP:
    """一个最小的多层感知机(只做前向),用于 g_theta / f_phi"""

    def __init__(self, input_dim, hidden_dims, output_dim):
        self.layers = []

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        for i in range(len(dims) - 1):
            # 小尺度初始化,避免一开始数值爆炸
            W = np.random.randn(dims[i + 1], dims[i]) * 0.01
            b = np.zeros((dims[i + 1], 1))
            self.layers.append((W, b))

    def forward(self, x):
        """x: (input_dim,) 或 (input_dim, 1)"""
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            # 最后一层不加 ReLU,让输出更自由
            if i < len(self.layers) - 1:
                x = relu(x)

        return x.flatten()
```

### 2) Relation Network 模块

```python
class RelationNetwork:
    """Relation Network: RN(O) = f_phi( sum_{i,j} g_theta(o_i, o_j, q) )"""

    def __init__(self, object_dim, query_dim, g_hidden_dims, f_hidden_dims, output_dim):
        # g_theta 输入是 [o_i, o_j, q] 拼起来
        g_input_dim = object_dim * 2 + query_dim
        g_output_dim = g_hidden_dims[-1] if len(g_hidden_dims) > 0 else 256
        self.g_theta = MLP(g_input_dim, g_hidden_dims[:-1], g_output_dim)

        # f_phi 接收 sum 后的关系向量
        self.f_phi = MLP(g_output_dim, f_hidden_dims, output_dim)

    def forward(self, objects, query):
        """
        objects: list[np.ndarray], 每个 shape=(object_dim,)
        query: np.ndarray, shape=(query_dim,)
        """
        n_objects = len(objects)
        relations = []

        # 这里按论文写法:枚举所有 (i, j)
        # 复杂度 O(n^2),但换来明确的关系计算
        for i in range(n_objects):
            for j in range(n_objects):
                pair_input = np.concatenate([objects[i], objects[j], query])
                relations.append(self.g_theta.forward(pair_input))

        aggregated = np.sum(relations, axis=0)
        return self.f_phi.forward(aggregated)
```

## 用一个玩具数据集:Sort-of-CLEVR

真正的 CLEVR 很大,我们这里用简化版本:

- 场景里有多个物体
- 每个物体:位置(x,y)+颜色+形状+大小
- 问题分两类:
  - 非关系问题: "红色物体是什么形状?"
  - 关系问题: "离红色物体最近的那个是什么形状?"

### 3) 生成场景和问题

```python
class SortOfCLEVR:
    """生成一个简化版 Sort-of-CLEVR"""

    def __init__(self):
        self.colors = ['red', 'blue', 'green', 'orange', 'yellow', 'purple']
        self.shapes = ['circle', 'square', 'triangle']
        self.sizes = ['small', 'large']

    def generate_scene(self, n_objects=6):
        """每个物体: (x, y, color, shape, size)"""
        objects = []
        used_colors = set()

        for _ in range(n_objects):
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)

            # 用不同颜色避免歧义(更像“指代某个物体”)
            available = [c for c in range(len(self.colors)) if c not in used_colors]
            if not available:
                break
            color_idx = int(np.random.choice(available))
            used_colors.add(color_idx)

            shape_idx = int(np.random.randint(len(self.shapes)))
            size_idx = int(np.random.randint(len(self.sizes)))

            objects.append({
                'x': float(x),
                'y': float(y),
                'color': color_idx,
                'shape': shape_idx,
                'size': size_idx,
            })

        return objects

    def generate_question(self, scene, question_type='relational'):
        """返回: (question_text, answer_idx, question_type)"""
        if question_type == 'relational':
            ref_obj = np.random.choice(scene)

            # 找最近的另一个物体
            min_dist = float('inf')
            closest_obj = None
            for obj in scene:
                if obj is ref_obj:
                    continue
                dist = np.sqrt((obj['x'] - ref_obj['x'])**2 + (obj['y'] - ref_obj['y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_obj = obj

            question = f"Shape of object closest to {self.colors[ref_obj['color']]}?"
            answer = closest_obj['shape']
            return question, answer, 'relational'

        # 非关系问题:直接问某个物体属性
        obj = np.random.choice(scene)
        question = f"What is the shape of the {self.colors[obj['color']]} object?"
        answer = obj['shape']
        return question, answer, 'non-relational'
```

### 4) 可视化场景(方便直觉理解)

```python
def visualize_scene(scene, dataset):
    fig, ax = plt.subplots(figsize=(7, 7))

    color_map = {
        'red': 'red',
        'blue': 'blue',
        'green': 'green',
        'orange': 'orange',
        'yellow': 'yellow',
        'purple': 'purple',
    }

    for obj in scene:
        x, y = obj['x'], obj['y']
        color = color_map[dataset.colors[obj['color']]]
        shape = dataset.shapes[obj['shape']]
        size = 300 if obj['size'] == 1 else 150

        marker = 'o' if shape == 'circle' else ('s' if shape == 'square' else '^')
        ax.scatter([x], [y], s=size, c=color, marker=marker,
                   edgecolors='black', linewidths=2)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('Sort-of-CLEVR Scene', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.show()
```

## 把“物体”和“问题”编码成向量

RN 不关心你用什么编码器(论文里是 CNN + question embedding),我们这里用最朴素的 one-hot:

- 物体编码: `[x, y, color_one_hot, shape_one_hot, size_one_hot]`
- 问题编码: `[ref_color_one_hot, is_relational]`

```python
def encode_object(obj, dataset):
    pos = np.array([obj['x'], obj['y']])

    color_oh = np.zeros(len(dataset.colors))
    color_oh[obj['color']] = 1

    shape_oh = np.zeros(len(dataset.shapes))
    shape_oh[obj['shape']] = 1

    size_oh = np.zeros(len(dataset.sizes))
    size_oh[obj['size']] = 1

    return np.concatenate([pos, color_oh, shape_oh, size_oh])


def encode_question(question_text, ref_color, dataset):
    color_oh = np.zeros(len(dataset.colors))
    if ref_color is not None:
        color_oh[ref_color] = 1

    is_relational = 1.0 if 'closest' in question_text else 0.0
    return np.concatenate([color_oh, [is_relational]])
```

## 端到端跑一遍:Scene → Objects → RN → Answer

注意:这里模型没有训练,所以预测是随机的;我们的重点是把结构和数据流跑通。

```python
dataset = SortOfCLEVR()
scene = dataset.generate_scene(n_objects=6)
visualize_scene(scene, dataset)

# 编码场景
encoded_objects = [encode_object(obj, dataset) for obj in scene]

# 生成一个关系问题
question, answer, _ = dataset.generate_question(scene, 'relational')

# 从问题里找被指代的颜色(玩具做法)
ref_color = None
for i, color in enumerate(dataset.colors):
    if color in question.lower():
        ref_color = i
        break

encoded_question = encode_question(question, ref_color, dataset)

object_dim = encoded_objects[0].shape[0]
query_dim = encoded_question.shape[0]

rn = RelationNetwork(
    object_dim=object_dim,
    query_dim=query_dim,
    g_hidden_dims=[64, 64, 32],
    f_hidden_dims=[64, 32],
    output_dim=len(dataset.shapes),
)

logits = rn.forward(encoded_objects, encoded_question)
pred = int(np.argmax(logits))

print('Question:', question)
print('True answer:', dataset.shapes[answer])
print('Predicted (untrained):', dataset.shapes[pred])
```

## RN 的一个隐藏大优点:天然“对顺序不敏感”

你把物体列表打乱,只要物体集合不变,\(\sum_{i,j}\) 的结果就不变。

```python
# 置换不变性测试
test_objects = [np.random.randn(object_dim) for _ in range(4)]
test_query = np.random.randn(query_dim)

out1 = rn.forward(test_objects, test_query)
shuffled = test_objects.copy()
np.random.shuffle(shuffled)
out2 = rn.forward(shuffled, test_query)

diff = np.linalg.norm(out1 - out2)
print('Difference:', diff)
print('PASSED' if diff < 1e-10 else 'FAILED')
```

这点在“输入是集合(set)”的任务里非常香:你不需要额外做排序/对齐,结构上就支持。

## 对比一个不显式建模关系的 Baseline

Baseline 思路很常见:把所有物体向量拼起来再过 MLP。

问题是:关系被强行塞进一个大向量里,学习会更依赖数据量和优化运气。

```python
class BaselineNetwork:
    """Baseline: concat(all objects, query) -> MLP"""

    def __init__(self, object_dim, query_dim, max_objects, output_dim):
        input_dim = object_dim * max_objects + query_dim
        self.mlp = MLP(input_dim, [128, 64], output_dim)
        self.max_objects = max_objects
        self.object_dim = object_dim

    def forward(self, objects, query):
        padded = []
        for i in range(self.max_objects):
            if i < len(objects):
                padded.append(objects[i])
            else:
                padded.append(np.zeros(self.object_dim))

        concat = np.concatenate(padded + [query])
        return self.mlp.forward(concat)
```

RN 和 Baseline 的差异你可以记成一句话:

RN 是“先两两比关系再汇总”,Baseline 是“先揉成一团再让 MLP 自己悟”。

## 小结

今天我们把 Relation Network 的核心直觉拆成了三句话:

1) **推理=关系**:很多任务不是识别物体,而是识别物体之间的关系
2) **显式两两比较**:用 \(g_\theta(o_i,o_j,q)\) 把关系算出来
3) **求和聚合带来置换不变性**:集合输入不怕顺序变化

如果你后面学到 Graph Neural Network(GNN) 或 Transformer 的 self-attention,会发现它们都在用不同方式做“关系建模”。RN 可以当成一个非常直接的启蒙版本。

## 练习题

1) 把关系聚合从 `sum` 改成 `mean` / `max` / `logsumexp`,观察输出是否仍然置换不变。

2) 只枚举 `i < j` 的无序对(组合)而不是所有有序对(i,j),复杂度怎么变?表达能力会损失吗?

3) 给 Sort-of-CLEVR 加一个新问题类型: "离红色最远的那个是什么形状?" 并扩展编码。

4) 给 RN 加一个训练循环(交叉熵),让它在关系问题上显著超过 Baseline。

## 延伸阅读

1) Santoro et al., 2017, "A simple neural network module for relational reasoning" (Relation Networks)

2) CLEVR 数据集与视觉问答(VQA)相关工作

3) 图神经网络(GNN)的消息传递框架(你会看到“关系”如何在图上流动)

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 16 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
