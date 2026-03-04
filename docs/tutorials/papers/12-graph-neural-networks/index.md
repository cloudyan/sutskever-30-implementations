# 图神经网络如何传递消息?

问下大家,有没有想过为什么传统神经网络处理不了社交网络、分子结构这样的图数据?

晓寒刚开始学深度学习的时候,就遇到了这样的困惑:图像是规则的网格,RNN 是规则的序列,但现实世界中很多数据是不规则的图结构!

直到后来遇到了**图神经网络**(Graph Neural Networks, GNN),才发现卧槽,原来用消息传递的方式处理图数据这么自然!

## 图数据的挑战

### 图 vs 传统数据

```
传统数据:
- 图像: 规则的 2D 网格 (CNN)
- 文本: 规则的 1D 序列 (RNN)
- 特征: 固定大小的向量 (MLP)

图数据:
- 节点数量: 不固定
- 节点连接: 不规则
- 节点顺序: 无顺序 (排列不变)

问题: 如何用神经网络处理不规则图结构?
```

### 现实中的图数据

```
社交网络: 用户是节点,关注关系是边
分子结构: 原子是节点,化学键是边
知识图谱: 实体是节点,关系是边
交通网络: 地点是节点,道路是边
引用网络: 论文是节点,引用关系是边
```

## 消息传递框架

### 核心思想:邻居传递信息

图神经网络的核心是**消息传递**(Message Passing):

```
直觉:
- 每个节点有自己的特征
- 通过边与邻居节点交换信息
- 聚合邻居的信息更新自己
- 多次迭代,信息传播更远
```

这就像朋友圈传播信息:
1. 你听到朋友的消息
2. 把多个朋友的消息综合起来
3. 形成自己的看法
4. 再传给其他人

### 数学公式

**消息传递神经网络**(MPNN)的两个关键步骤:

**1. 消息生成**:
$$m_v^{(k)} = \sum_{u \in \mathcal{N}(v)} M_k(h_v^{(k-1)}, h_u^{(k-1)}, e_{uv})$$

**2. 节点更新**:
$$h_v^{(k)} = U_k(h_v^{(k-1)}, m_v^{(k)})$$

其中:
- $h_v^{(k)}$: 节点 $v$ 在第 $k$ 层的表示
- $\mathcal{N}(v)$: 节点 $v$ 的邻居集合
- $e_{uv}$: 边 $(u, v)$ 的特征
- $M_k$: 消息函数
- $U_k$: 更新函数

## 用 NumPy 实现 GNN

### 图的表示

首先,我们需要表示图结构:

```python
import numpy as np

class Graph:
    """
    简单的图表示
    
    属性:
        num_nodes: 节点数量
        edges: 边列表 [(源节点, 目标节点), ...]
        node_features: 节点特征列表
        edge_features: 边特征字典 {(源, 目标): 特征}
    """
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.edges = []  # 边列表
        self.node_features = []  # 节点特征
        self.edge_features = {}  # 边特征
    
    def add_edge(self, src, tgt, features=None):
        """添加边"""
        self.edges.append((src, tgt))
        if features is not None:
            self.edge_features[(src, tgt)] = features
    
    def set_node_features(self, features):
        """设置节点特征"""
        self.node_features = features
    
    def get_neighbors(self, node):
        """获取节点的所有邻居"""
        neighbors = []
        for src, tgt in self.edges:
            if src == node:
                neighbors.append(tgt)
        return neighbors
    
    def visualize(self):
        """可视化图结构"""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_edges_from(self.edges)
        
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, 
                node_color='lightblue',
                node_size=800, 
                font_size=12,
                arrows=True,
                edge_color='gray',
                width=2)
        plt.title("图结构")
        plt.axis('off')
        plt.show()

# 示例:水分子 H2O
water = Graph(num_nodes=3)
water.add_edge(0, 1)  # O -> H
water.add_edge(0, 2)  # O -> H
water.add_edge(1, 0)  # H -> O (无向图)
water.add_edge(2, 0)  # H -> O

# 节点特征: [原子序数, 化合价]
water.set_node_features([
    np.array([8, 2]),  # 氧原子
    np.array([1, 1]),  # 氢原子
    np.array([1, 1]),  # 氢原子
])

print("水分子图:")
print(f"节点数: {water.num_nodes}")
print(f"边数: {len(water.edges)}")
print(f"氧原子(节点0)的邻居: {water.get_neighbors(0)}")
```

### 消息传递层

```python
class MessagePassingLayer:
    """
    单层消息传递
    
    实现核心的 Message-Aggregate-Update 流程
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # 消息函数: M(h_source, h_target, e_features)
        self.W_msg = np.random.randn(hidden_dim, 2*node_dim + edge_dim) * 0.01
        self.b_msg = np.zeros(hidden_dim)
        
        # 更新函数: U(h_node, aggregated_message)
        self.W_update = np.random.randn(node_dim, node_dim + hidden_dim) * 0.01
        self.b_update = np.zeros(node_dim)
    
    def message(self, h_source, h_target, e_features):
        """
        生成消息
        
        参数:
            h_source: 源节点特征
            h_target: 目标节点特征
            e_features: 边特征
        
        返回:
            消息向量
        """
        # 拼接源节点、目标节点、边特征
        if e_features is None:
            e_features = np.zeros(self.edge_dim)
        
        concat = np.concatenate([h_source, h_target, e_features])
        
        # 通过神经网络生成消息
        message = np.tanh(np.dot(self.W_msg, concat) + self.b_msg)
        return message
    
    def aggregate(self, messages):
        """
        聚合消息 (求和)
        
        参数:
            messages: 消息列表
        
        返回:
            聚合后的消息
        """
        if len(messages) == 0:
            return np.zeros(self.hidden_dim)
        return np.sum(messages, axis=0)
    
    def update(self, h_node, aggregated_message):
        """
        更新节点表示
        
        参数:
            h_node: 当前节点特征
            aggregated_message: 聚合后的消息
        
        返回:
            更新后的节点特征
        """
        concat = np.concatenate([h_node, aggregated_message])
        h_new = np.tanh(np.dot(self.W_update, concat) + self.b_update)
        return h_new
    
    def forward(self, graph, node_states):
        """
        前向传播: 对所有节点执行消息传递
        
        参数:
            graph: 图对象
            node_states: 当前节点状态列表
        
        返回:
            更新后的节点状态列表
        """
        new_states = []
        
        for v in range(graph.num_nodes):
            # 1. 收集邻居消息
            messages = []
            for w in graph.get_neighbors(v):
                # 获取边特征
                edge_feat = graph.edge_features.get((w, v), None)
                
                # 生成消息
                msg = self.message(node_states[w], node_states[v], edge_feat)
                messages.append(msg)
            
            # 2. 聚合消息
            aggregated = self.aggregate(messages)
            
            # 3. 更新节点状态
            h_new = self.update(node_states[v], aggregated)
            new_states.append(h_new)
        
        return new_states

# 测试消息传递层
node_dim = 4
edge_dim = 2
hidden_dim = 8

mp_layer = MessagePassingLayer(node_dim, edge_dim, hidden_dim)

# 初始化节点状态
initial_states = []
for feat in water.node_features:
    # 嵌入到更高维度
    state = np.concatenate([feat, np.zeros(node_dim - len(feat))])
    initial_states.append(state)

# 执行消息传递
updated_states = mp_layer.forward(water, initial_states)

print("初始状态 (氧原子):", initial_states[0])
print("更新状态 (氧原子):", updated_states[0])
print("\n节点状态已通过邻居信息更新!")
```

### 完整的 MPNN

```python
class MPNN:
    """
    完整的消息传递神经网络
    
    结构:
    1. 节点特征嵌入
    2. 多层消息传递
    3. 读出函数 (图级别预测)
    """
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, 
                 num_layers, output_dim):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 节点特征嵌入层
        self.embed_W = np.random.randn(hidden_dim, node_feat_dim) * 0.01
        
        # 多层消息传递
        self.mp_layers = [
            MessagePassingLayer(hidden_dim, edge_feat_dim, hidden_dim*2)
            for _ in range(num_layers)
        ]
        
        # 读出函数 (图级别预测)
        self.readout_W = np.random.randn(output_dim, hidden_dim) * 0.01
        self.readout_b = np.zeros(output_dim)
    
    def forward(self, graph):
        """
        前向传播
        
        参数:
            graph: 输入图
        
        返回:
            output: 图级别预测
            history: 各层节点状态历史
        """
        # 1. 嵌入节点特征
        node_states = []
        for feat in graph.node_features:
            embedded = np.tanh(np.dot(self.embed_W, feat))
            node_states.append(embedded)
        
        # 2. 多层消息传递
        states_history = [node_states]
        for layer in self.mp_layers:
            node_states = layer.forward(graph, node_states)
            states_history.append(node_states)
        
        # 3. 读出: 聚合节点状态得到图表示
        graph_repr = np.sum(node_states, axis=0)  # 简单求和池化
        
        # 4. 最终预测
        output = np.dot(self.readout_W, graph_repr) + self.readout_b
        
        return output, states_history

# 创建 MPNN
mpnn = MPNN(
    node_feat_dim=2,    # 节点特征维度
    edge_feat_dim=2,    # 边特征维度
    hidden_dim=8,       # 隐藏层维度
    num_layers=3,       # 消息传递层数
    output_dim=1        # 输出维度 (如预测分子能量)
)

# 前向传播
prediction, history = mpnn.forward(water)

print(f"图级别预测: {prediction[0]:.4f}")
print("(例如: 分子属性预测,如能量、溶解度等)")
```

## 消息传递的可视化

```python
def visualize_message_passing(graph, states_history):
    """
    可视化消息传递过程
    """
    import matplotlib.pyplot as plt
    
    num_layers = len(states_history)
    
    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))
    
    if num_layers == 1:
        axes = [axes]
    
    for i, (ax, states) in enumerate(zip(axes, states_history)):
        # 将节点状态转换为颜色
        colors = [np.linalg.norm(s) for s in states]
        colors = (np.array(colors) - min(colors)) / (max(colors) - min(colors) + 1e-8)
        
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(range(graph.num_nodes))
        G.add_edges_from(graph.edges)
        
        pos = nx.spring_layout(G, seed=42)
        
        nx.draw(G, pos, ax=ax,
                node_color=colors,
                cmap='YlOrRd',
                node_size=800,
                font_size=12,
                with_labels=True,
                arrows=True,
                edge_color='gray')
        
        ax.set_title(f'Layer {i}')
    
    plt.tight_layout()
    plt.show()

# 可视化消息传递过程
visualize_message_passing(water, history[:4])

print("观察:")
print("1. 初始层: 节点状态主要由自己的特征决定")
print("2. 中间层: 信息从邻居传播过来")
print("3. 最终层: 节点状态融合了多跳邻居的信息")
```

## GNN 的关键特性

### 1. 排列不变性

```python
# GNN 对节点顺序不敏感
graph1 = Graph(num_nodes=3)
graph1.add_edge(0, 1)
graph1.add_edge(1, 0)
graph1.set_node_features([np.array([1.0]), np.array([2.0]), np.array([3.0])])

graph2 = Graph(num_nodes=3)
graph2.add_edge(1, 0)  # 顺序不同
graph2.add_edge(0, 1)
graph2.set_node_features([np.array([3.0]), np.array([1.0]), np.array([2.0])])  # 顺序不同

# 但如果图结构相同,预测结果应该相似
print("GNN 具有排列不变性:")
print("节点顺序改变,只要图结构不变,结果保持一致")
```

### 2. 感受野随层数增加

```python
def compute_receptive_field(num_layers):
    """
    计算 GNN 的感受野
    
    每多一层,感受野扩展 1 跳
    """
    return 2 * num_layers + 1

print("GNN 感受野随层数增长:")
for L in [1, 2, 3, 4, 5]:
    rf = compute_receptive_field(L)
    print(f"  {L} 层: {rf} 跳范围")

print("\n类比 CNN:")
print("  CNN: 感受野随层数和卷积核大小增长")
print("  GNN: 感受野随层数增长,每层扩展 1 跳")
```

### 3. 参数共享

```python
# 所有节点共享同一套消息传递参数
print("GNN 的参数共享:")
print("1. 所有节点使用相同的消息函数")
print("2. 所有边使用相同的聚合方式")
print("3. 参数数量与图大小无关")

# 类比 CNN
print("\n类比 CNN:")
print("  CNN: 所有位置共享卷积核参数")
print("  GNN: 所有节点共享消息传递参数")
```

## GNN 的变体

### 1. GCN (Graph Convolutional Network)

```python
class GCNLayer:
    """
    图卷积网络层
    
    简化的消息传递:
    - 消息: 邻居特征的加权和
    - 聚合: 归一化求和
    - 更新: 线性变换 + 激活
    """
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
    
    def forward(self, X, A):
        """
        X: 节点特征 (num_nodes, in_features)
        A: 邻接矩阵 (num_nodes, num_nodes)
        """
        # 度矩阵
        D = np.diag(np.sum(A, axis=1))
        
        # 归一化邻接矩阵
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        
        # 图卷积
        output = A_norm @ X @ self.W
        
        return np.tanh(output)

print("GCN 特点:")
print("1. 使用归一化邻接矩阵")
print("2. 简单的线性消息传递")
print("3. 计算高效,易于训练")
```

### 2. GAT (Graph Attention Network)

```python
class GATLayer:
    """
    图注意力网络层
    
    使用注意力机制聚合邻居信息
    """
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.a = np.random.randn(2 * out_features, 1) * 0.01
    
    def attention(self, h_i, h_j):
        """计算注意力系数"""
        concat = np.concatenate([h_i, h_j])
        e = np.dot(concat, self.a)
        return np.exp(e)
    
    def forward(self, X, A):
        """前向传播"""
        # 线性变换
        H = X @ self.W
        
        num_nodes = X.shape[0]
        output = np.zeros_like(H)
        
        for i in range(num_nodes):
            # 计算注意力系数
            neighbors = np.where(A[i] > 0)[0]
            if len(neighbors) == 0:
                continue
            
            # 注意力分数
            scores = np.array([self.attention(H[i], H[j]) for j in neighbors])
            scores = scores / (scores.sum() + 1e-8)  # Softmax
            
            # 加权聚合
            output[i] = np.sum(scores[:, None] * H[neighbors], axis=0)
        
        return output

print("GAT 特点:")
print("1. 自动学习邻居的重要性")
print("2. 注意力机制提供可解释性")
print("3. 适合异质图(不同邻居重要性不同)")
```

## 实际应用

### 1. 分子属性预测

```python
# 分子图: 节点=原子, 边=化学键
# 任务: 预测分子属性(溶解度、毒性等)

class MolecularGNN:
    """分子属性预测 GNN"""
    def __init__(self):
        self.mpnn = MPNN(
            node_feat_dim=...,
            edge_feat_dim=...,
            hidden_dim=64,
            num_layers=5,
            output_dim=1
        )
    
    def predict_property(self, molecule_graph):
        """预测分子属性"""
        prediction, _ = self.mpnn.forward(molecule_graph)
        return prediction

print("分子属性预测:")
print("- 输入: 分子图(原子作为节点,化学键作为边)")
print("- 输出: 分子属性(能量、溶解度、毒性等)")
print("- 应用: 药物发现、材料设计")
```

### 2. 社交网络分析

```python
# 社交图: 节点=用户, 边=关注关系
# 任务: 预测用户属性、推荐好友

print("社交网络分析:")
print("- 节点分类: 预测用户兴趣、职业")
print("- 链接预测: 推荐好友")
print("- 社区发现: 识别用户群体")
```

### 3. 知识图谱

```python
# 知识图谱: 节点=实体, 边=关系
# 任务: 链接预测、实体分类

print("知识图谱应用:")
print("- 链接预测: 预测缺失的关系")
print("- 实体分类: 实体类型识别")
print("- 推理: 基于图结构进行推理")
```

## 小结

今天我们深入理解了图神经网络的核心机制:

### 核心概念

1. **消息传递**: 节点通过边传递和聚合信息
2. **排列不变性**: 对节点顺序不敏感
3. **感受野**: 随层数增长,扩展多跳邻居

### 关键步骤

```
Message → Aggregate → Update
   ↓          ↓         ↓
生成消息   聚合消息   更新节点
```

### 为什么有效

1. **自然建模**: 图结构数据的天生建模方式
2. **信息传播**: 多层消息传递实现信息长距离传播
3. **参数共享**: 高效处理变长图结构

### 应用场景

1. **分子属性预测**: 药物发现
2. **社交网络**: 推荐系统
3. **知识图谱**: 推理和问答

## 练习题

### 1. 概念理解

**问题 1**: GNN 的消息传递与 CNN 的卷积有什么相似之处?有什么区别?

**问题 2**: 为什么 GNN 具有排列不变性?这对实际应用有什么意义?

**问题 3**: 比较 GCN 和 GAT 的优缺点,分别适用于什么场景?

### 2. 编程实践

**练习 1**: 实现一个带边特征的 GNN:

```python
class EdgeGNN:
    """考虑边特征的 GNN"""
    # TODO: 实现
    pass
```

**练习 2**: 实现图级别的读出函数:

```python
def graph_readout(node_states, method='sum'):
    """
    图级别读出函数
    
    method: 'sum', 'mean', 'max', 'attention'
    """
    # TODO: 实现
    pass
```

**练习 3**: 在真实数据集上训练 GNN:

```python
# 使用 Cora 数据集 (论文引用网络)
# 任务: 论文分类
# TODO: 实现数据加载、训练、评估
```

### 3. 深度思考

**思考 1**: GNN 的过平滑(Over-smoothing)问题是什么?如何缓解?

**思考 2**: 如何设计适合大规模图的 GNN?有哪些采样策略?

**思考 3**: GNN 与 Transformer 的注意力机制有什么联系?能否结合?

## 延伸阅读

### 经典论文

1. **MPNN**: Gilmer et al. (2017). "Neural Message Passing for Quantum Chemistry"
2. **GCN**: Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks"
3. **GAT**: Veličković et al. (2018). "Graph Attention Networks"

### 教程和资源

- Stanford CS224W: Machine Learning with Graphs
- "A Comprehensive Survey on Graph Neural Networks"
- PyTorch Geometric: 图神经网络库

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 12 篇。上一篇我们学习了膨胀卷积如何扩大感受野,下一篇我们将深入理解 Transformer 的注意力机制。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!** 🚀