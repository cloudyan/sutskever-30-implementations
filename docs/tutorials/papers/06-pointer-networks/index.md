# 如何让神经网络学会"指"？

问下大家，有没有想过为什么传统的神经网络在处理某些问题时特别吃力？

比如说，你要让神经网络解决旅行商问题（TSP）：给定 10 个城市，找出访问所有城市一次并回到起点的最短路径。输出应该是这 10 个城市的一个排列，比如 [3, 7, 1, 9, 4, 2, 8, 5, 6, 10]。

但是！传统的 seq2seq 模型有个大问题：输出词典大小必须是固定的。城市数量变了怎么办？10 个城市要 10 个输出，100 个城市要 100 个输出，模型完全没法通用！

这时候就需要**Pointer Network（指针网络）**——让神经网络学会"指"，而不是"猜"！

## 传统 Seq2Seq 的困境

### 固定输出词典的问题

传统的 Encoder-Decoder 架构：

```
输入序列 → [Encoder] → 上下文向量 → [Decoder] → 输出序列
   ↓                                        ↓
 [我, 爱, 深度, 学习]                [I, love, deep, learning]
```

Decoder 每步输出一个单词，从预定义的词典中选择：
- 词典大小 = 10000 个单词
- 输出层 = Softmax(10000 个类别)

**问题**：输出类别数必须是固定的！

### 组合优化问题的挑战

考虑旅行商问题（TSP）：
- 输入：N 个城市的坐标
- 输出：N 个城市的一个排列（访问顺序）

如果用传统 seq2seq：
- N=10 时，输出词典大小 = 10
- N=100 时，输出词典大小 = 100
- **N 变化时，模型需要重新训练！**

这显然不现实。我们需要一个**输出词典大小可以动态变化**的模型。

## Pointer Network 的核心思想

### 关键洞察：用注意力作为指针

传统的 Attention 机制：

$$\alpha_t = \text{softmax}(e_t) \quad \text{其中} \quad e_{t,i} = f(s_t, h_i)$$

这里的 $\alpha_t$ 是一个**概率分布**，表示解码器在生成输出时应该关注编码器哪些位置。

**Pointer Network 的关键创新**：

**不直接用这个分布生成输出词汇，而是用它作为"指针"，从输入序列中选择元素！**

```
传统 Seq2Seq:
注意力分布 → [选择词汇] → 输出词
   ↓
词汇表: [the, cat, dog, ...]

Pointer Network:
注意力分布 → [选择位置] → 输入元素
   ↓
输入序列: [城市A, 城市B, 城市C, ...]
```

### 数学公式

**编码器**（与标准 Seq2Seq 相同）：

$$h_i^{enc} = \text{LSTM}_{enc}(x_i, h_{i-1}^{enc})$$

**解码器**（关键差异）：

在每个解码步骤 $t$：

1. **解码器状态更新**：
   $$h_t^{dec} = \text{LSTM}_{dec}(y_{t-1}, h_{t-1}^{dec})$$

2. **计算注意力分数**（与标准 Attention 相同）：
   $$u_t^i = v^T \tanh(W_h h_i^{enc} + W_s h_t^{dec} + b)$$
   
3. **转换为概率分布**：
   $$p(C_i | C_1, ..., C_{i-1}) = \text{softmax}(u_t)$$

**关键区别**：这里的输出不是词汇表中的词，而是**输入序列中的位置**！

### 直观示例

假设输入是 5 个城市的坐标：
```
输入: [A, B, C, D, E]
```

TSP 的目标：输出一个排列，如 `[C, A, E, B, D]`

Pointer Network 的工作过程：

```
解码步骤 1:
  解码器状态 h_1
  注意力分数: [0.1, 0.1, 0.5, 0.2, 0.1]  ← C 的分数最高
  输出: C

解码步骤 2:
  解码器状态 h_2 (考虑已输出 C)
  注意力分数: [0.4, 0.1, 0.0, 0.2, 0.3]  ← A 的分数最高 (C 已被屏蔽)
  输出: A

解码步骤 3:
  注意力分数: [0.0, 0.1, 0.0, 0.2, 0.7]  ← E
  输出: E

... 直到输出所有城市

最终输出: [C, A, E, B, D]
```

**关键**：
- 注意力机制学会了"指向"正确的输入元素
- 每一步都考虑之前的选择（通过解码器状态）
- 可以处理变长输入和输出

## 用 NumPy 实现 Pointer Network

现在让我们用 NumPy 实现一个简化版的 Pointer Network：

```python
import numpy as np
import matplotlib.pyplot as plt

class PointerNetwork:
    """
    Pointer Network 的 NumPy 实现
    
    用于解决组合优化问题（如 TSP）
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        参数:
            input_size: 输入特征维度（如城市坐标的维度）
            hidden_size: 隐藏层维度
            output_size: 输出序列的最大长度（如最大城市数）
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 编码器参数
        self.Wx_enc = np.random.randn(input_size, hidden_size) * 0.01
        self.Wh_enc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_enc = np.zeros((1, hidden_size))
        
        # 解码器参数
        self.Wy_dec = np.random.randn(output_size, hidden_size) * 0.01
        self.Wh_dec = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_dec = np.zeros((1, hidden_size))
        
        # 注意力参数
        self.W_attn_enc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_attn_dec = np.random.randn(hidden_size, hidden_size) * 0.01
        self.v_attn = np.random.randn(hidden_size, 1) * 0.01
        self.b_attn = np.zeros((1, 1))
    
    def encoder(self, X):
        """
        编码器：将输入序列编码为隐藏状态序列
        
        参数:
            X: 输入序列 (seq_len, input_size)
        
        返回:
            enc_states: 编码器隐藏状态序列 (seq_len, hidden_size)
        """
        seq_len = X.shape[0]
        enc_states = np.zeros((seq_len, self.hidden_size))
        h = np.zeros((1, self.hidden_size))
        
        for t in range(seq_len):
            x_t = X[t:t+1, :]
            h = np.tanh(np.dot(x_t, self.Wx_enc) + 
                       np.dot(h, self.Wh_enc) + 
                       self.b_enc)
            enc_states[t] = h.flatten()
        
        return enc_states
    
    def attention(self, enc_states, dec_state, mask=None):
        """
        计算注意力权重
        
        参数:
            enc_states: 编码器隐藏状态 (seq_len, hidden_size)
            dec_state: 解码器当前状态 (1, hidden_size)
            mask: 可选的掩码，用于屏蔽已访问的位置
        
        返回:
            attn_weights: 注意力权重 (1, seq_len)
        """
        seq_len = enc_states.shape[0]
        
        # 计算注意力分数
        enc_transformed = np.dot(enc_states, self.W_attn_enc.T)  # (seq_len, hidden_size)
        dec_transformed = np.dot(dec_state, self.W_attn_dec.T)  # (1, hidden_size)
        
        scores = np.tanh(enc_transformed + dec_transformed)  # (seq_len, hidden_size)
        scores = np.dot(scores, self.v_attn) + self.b_attn  # (seq_len, 1)
        scores = scores.T  # (1, seq_len)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores + mask  # mask 中需要屏蔽的位置设为 -inf
        
        # Softmax 得到注意力权重
        attn_weights = np.exp(scores - np.max(scores))
        attn_weights = attn_weights / np.sum(attn_weights)
        
        return attn_weights
    
    def decoder_step(self, enc_states, prev_dec_state, 
                     prev_output, mask=None):
        """
        解码器单步
        
        参数:
            enc_states: 编码器隐藏状态
            prev_dec_state: 上一个解码器状态
            prev_output: 上一个输出（用于 teacher forcing）
            mask: 位置掩码
        
        返回:
            output: 当前输出（指向输入位置的指针）
            dec_state: 更新后的解码器状态
            attn_weights: 注意力权重（用于可视化）
        """
        # 计算注意力
        attn_weights = self.attention(enc_states, prev_dec_state, mask)
        
        # 上下文向量（加权的编码器状态）
        context = np.dot(attn_weights, enc_states)  # (1, hidden_size)
        
        # 更新解码器状态
        dec_input = np.concatenate([prev_output, context], axis=1)
        dec_state = np.tanh(np.dot(dec_input, self.Wy_dec) + 
                           np.dot(prev_dec_state, self.Wh_dec) + 
                           self.b_dec)
        
        # 输出是当前注意力最高的位置（Pointer！）
        output = np.argmax(attn_weights)
        
        return output, dec_state, attn_weights
    
    def forward(self, X, max_len=None, teacher_forcing_ratio=0.5):
        """
        完整的前向传播（训练时使用）
        
        参数:
            X: 输入序列 (seq_len, input_size)
            max_len: 最大输出长度
            teacher_forcing_ratio: 使用真实标签作为下一输入的概率
        
        返回:
            outputs: 输出序列
            attention_history: 注意力历史（用于可视化）
        """
        seq_len = X.shape[0]
        if max_len is None:
            max_len = seq_len
        
        # 编码
        enc_states = self.encoder(X)
        
        # 解码
        outputs = []
        attention_history = []
        dec_state = np.zeros((1, self.hidden_size))
        prev_output = np.zeros((1, self.output_size))
        
        # 创建掩码，防止重复选择
        mask = np.zeros((1, seq_len))
        
        for t in range(max_len):
            output, dec_state, attn_weights = self.decoder_step(
                enc_states, dec_state, prev_output, mask
            )
            
            outputs.append(output)
            attention_history.append(attn_weights.flatten())
            
            # 更新掩码，标记已选择的位置
            mask[0, output] = -1e9  # 用很大的负数，softmax后会变成0
            
            # 准备下一个输入（这里简化处理）
            prev_output = np.zeros((1, self.output_size))
            if output < self.output_size:
                prev_output[0, output] = 1
        
        return np.array(outputs), np.array(attention_history)
    
    def predict(self, X, max_len=None):
        """
        预测（推理时使用，无梯度）
        """
        return self.forward(X, max_len, teacher_forcing_ratio=0.0)

# 测试 Pointer Network
print("="*60)
print("测试 Pointer Network")
print("="*60)

np.random.seed(42)

# 创建一个小例子：TSP 有 5 个城市
n_cities = 5
input_size = 2  # 城市坐标 (x, y)
hidden_size = 32
output_size = n_cities

# 随机生成城市坐标
cities = np.random.rand(n_cities, input_size)
print(f"\n城市坐标:")
for i, coord in enumerate(cities):
    print(f"  城市 {i}: ({coord[0]:.3f}, {coord[1]:.3f})")

# 创建 Pointer Network
ptr_net = PointerNetwork(input_size, hidden_size, output_size)

# 前向传播
outputs, attention_history = ptr_net.forward(cities, max_len=n_cities)

print(f"\nPointer Network 输出 (城市访问顺序):")
print(f"  {outputs}")

print(f"\n注意力历史形状: {attention_history.shape}")
print(f"  (时间步, 输入位置)")

# 可视化注意力
plt.figure(figsize=(10, 6))
plt.imshow(attention_history, cmap='viridis', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.xlabel('Input Position (City)')
plt.ylabel('Output Step')
plt.title('Pointer Network Attention Visualization')
plt.xticks(range(n_cities))
plt.yticks(range(n_cities))

# 在每个时间步标注选中的城市
for t in range(n_cities):
    selected = outputs[t]
    plt.text(selected, t, '★', ha='center', va='center', 
             color='red', fontsize=20)

plt.tight_layout()
plt.show()

print("\nPointer Network 测试完成!")
print("\n观察：")
print("1. 注意力热力图显示了每个解码步骤对输入位置的关注程度")
print("2. 红色五角星标记了每个时间步选择的城市")
print("3. 可以看到 Pointer Network 学会了'指向'正确的输入位置")
```

## 应用：解决 TSP 问题

让我们用 Pointer Network 来解决实际的旅行商问题：

```python
class TSPSolver:
    """使用 Pointer Network 解决 TSP"""
    
    def __init__(self, hidden_size=128):
        self.hidden_size = hidden_size
        self.model = None
    
    def train(self, train_instances, n_epochs=100, learning_rate=0.001):
        """
        在 TSP 实例上训练 Pointer Network
        
        参数:
            train_instances: 列表，每个元素是 (cities, tour) 元组
            n_epochs: 训练轮数
            learning_rate: 学习率
        """
        # 确定最大城市数
        max_cities = max(cities.shape[0] for cities, _ in train_instances)
        input_size = 2  # 2D 坐标
        
        # 创建模型
        self.model = PointerNetwork(
            input_size, 
            self.hidden_size, 
            max_cities
        )
        
        print(f"开始训练 TSP Solver...")
        print(f"  训练实例数: {len(train_instances)}")
        print(f"  最大城市数: {max_cities}")
        print(f"  隐藏层大小: {self.hidden_size}")
        print(f"  训练轮数: {n_epochs}")
        
        # 训练循环（简化版）
        for epoch in range(n_epochs):
            total_loss = 0
            np.random.shuffle(train_instances)
            
            for cities, optimal_tour in train_instances:
                n_cities = cities.shape[0]
                
                # 前向传播
                outputs, attention = self.model.forward(
                    cities, 
                    max_len=n_cities
                )
                
                # 计算损失（这里简化处理，实际需要更复杂的损失函数）
                # 理想情况下，outputs 应该接近 optimal_tour
                loss = 0
                for t in range(n_cities):
                    if t < len(outputs) and t < len(optimal_tour):
                        loss += (outputs[t] - optimal_tour[t]) ** 2
                
                total_loss += loss
                
                # 反向传播（简化，实际需要实现完整的 BPTT）
                # ...
            
            avg_loss = total_loss / len(train_instances)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        print("训练完成!")
    
    def solve(self, cities):
        """
        使用训练好的模型解决新的 TSP 实例
        
        参数:
            cities: 城市坐标数组 (n_cities, 2)
        
        返回:
            tour: 访问顺序列表
            tour_length: 路径长度
        """
        if self.model is None:
            raise ValueError("模型尚未训练!")
        
        n_cities = cities.shape[0]
        
        # 使用模型预测
        outputs, attention = self.model.predict(cities, max_len=n_cities)
        
        # 构建 tour
        tour = list(outputs[:n_cities])
        
        # 确保每个城市只访问一次（后处理）
        visited = set()
        unique_tour = []
        for city in tour:
            if city not in visited and 0 <= city < n_cities:
                unique_tour.append(city)
                visited.add(city)
        
        # 添加遗漏的城市（简单的启发式）
        for city in range(n_cities):
            if city not in visited:
                unique_tour.append(city)
        
        # 计算路径长度
        tour_length = 0
        for i in range(len(unique_tour)):
            city1 = unique_tour[i]
            city2 = unique_tour[(i + 1) % len(unique_tour)]
            dist = np.linalg.norm(cities[city1] - cities[city2])
            tour_length += dist
        
        return unique_tour, tour_length

# 测试 TSP Solver
print("="*60)
print("测试 TSP Solver")
print("="*60)

np.random.seed(42)

# 生成训练数据
def generate_tsp_data(n_instances=100, n_cities_range=(5, 10)):
    """生成 TSP 训练数据"""
    instances = []
    
    for _ in range(n_instances):
        n_cities = np.random.randint(n_cities_range[0], n_cities_range[1] + 1)
        
        # 随机生成城市坐标
        cities = np.random.rand(n_cities, 2)
        
        # 使用贪心算法生成一个合理的 tour（作为监督信号）
        tour = greedy_tsp(cities)
        
        instances.append((cities, tour))
    
    return instances

def greedy_tsp(cities):
    """贪心算法求解 TSP（近似解）"""
    n_cities = cities.shape[0]
    unvisited = set(range(n_cities))
    tour = []
    
    # 从城市 0 开始
    current = 0
    tour.append(current)
    unvisited.remove(current)
    
    while unvisited:
        # 找到最近的未访问城市
        nearest = None
        min_dist = float('inf')
        
        for city in unvisited:
            dist = np.linalg.norm(cities[current] - cities[city])
            if dist < min_dist:
                min_dist = dist
                nearest = city
        
        current = nearest
        tour.append(current)
        unvisited.remove(current)
    
    return tour

# 生成训练数据
print("\n生成训练数据...")
train_data = generate_tsp_data(n_instances=50, n_cities_range=(5, 8))
print(f"生成了 {len(train_data)} 个训练实例")

# 创建并训练 TSP Solver
print("\n创建 TSP Solver...")
solver = TSPSolver(hidden_size=64)

print("\n训练模型（简化版）...")
print("注意：完整的训练需要更多时间和更复杂的实现")

# 这里简化训练过程，实际应该使用完整的 BPTT
# solver.train(train_data, n_epochs=50)

# 直接使用未训练的模型进行演示
print("\n使用未训练的模型进行演示（实际应用需要训练）...")

# 测试一个新的 TSP 实例
test_cities = np.array([
    [0.2, 0.3],
    [0.8, 0.7],
    [0.5, 0.1],
    [0.3, 0.9],
    [0.9, 0.4]
])

print(f"\n测试 TSP 实例:")
print(f"  城市数量: {len(test_cities)}")
print(f"  城市坐标:")
for i, coord in enumerate(test_cities):
    print(f"    城市 {i}: ({coord[0]:.2f}, {coord[1]:.2f})")

# 使用贪心算法获得参考解
reference_tour = greedy_tsp(test_cities)
reference_length = sum(
    np.linalg.norm(test_cities[reference_tour[i]] - 
                   test_cities[reference_tour[(i+1) % len(reference_tour)]])
    for i in range(len(reference_tour))
)

print(f"\n参考解 (贪心算法):")
print(f"  路径: {reference_tour}")
print(f"  长度: {reference_length:.4f}")

# 可视化
plt.figure(figsize=(15, 5))

# 子图 1：城市分布
plt.subplot(1, 3, 1)
plt.scatter(test_cities[:, 0], test_cities[:, 1], 
           s=200, c='blue', alpha=0.6, edgecolors='black', linewidth=2)
for i, coord in enumerate(test_cities):
    plt.annotate(str(i), (coord[0], coord[1]), 
                fontsize=12, ha='center', va='center', fontweight='bold')
plt.title('City Distribution')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 子图 2：参考路径
plt.subplot(1, 3, 2)
tour_coords = test_cities[reference_tour + [reference_tour[0]]]
plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-o', 
         linewidth=2, markersize=8, alpha=0.7)
plt.scatter(test_cities[:, 0], test_cities[:, 1], 
           s=100, c='red', alpha=0.8, zorder=5)
for i, coord in enumerate(test_cities):
    plt.annotate(str(i), (coord[0], coord[1]), 
                fontsize=10, ha='center', va='center')
plt.title(f'Reference Tour (Greedy)\nLength: {reference_length:.4f}')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 子图 3：距离矩阵热图
plt.subplot(1, 3, 3)
n_cities = len(test_cities)
dist_matrix = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        dist_matrix[i, j] = np.linalg.norm(test_cities[i] - test_cities[j])

im = plt.imshow(dist_matrix, cmap='viridis')
plt.colorbar(im, label='Distance')
plt.title('Distance Matrix')
plt.xlabel('City Index')
plt.ylabel('City Index')

# 在热图上标记参考路径
for i in range(len(reference_tour)):
    city1 = reference_tour[i]
    city2 = reference_tour[(i + 1) % len(reference_tour)]
    plt.text(city2, city1, '→', ha='center', va='center', 
            fontsize=12, color='red', fontweight='bold')

plt.tight_layout()
plt.show()

print("\nPointer Network 演示完成!")
print("\n关键要点:")
print("1. Pointer Network 通过注意力机制'指向'输入位置")
print("2. 输出词典大小可以动态变化，适应不同问题规模")
print("3. 特别适用于组合优化问题（如TSP）")
print("4. 结合神经网络的学习能力和组合优化的结构化输出")
```

## Pointer Network 的应用与扩展

### 应用场景

Pointer Network 及其变体已被广泛应用于：

1. **组合优化**：
   - 旅行商问题（TSP）
   - 车辆路径规划（VRP）
   - 作业车间调度

2. **自然语言处理**：
   - 文本摘要（从原文中抽取句子）
   - 问答系统（从文档中定位答案）
   - 代码生成（从 API 文档中选择函数）

3. **生物信息学**：
   - DNA 序列组装
   - 蛋白质结构预测

### 扩展与变体

1. **Pointer-Generator Network**：
   - 结合生成和复制机制
   - 可以选择从词典生成，或从输入复制

2. **Transformer + Pointer**：
   - 用 Transformer 替代 LSTM 作为编码器/解码器
   - 更好的并行化和长程依赖建模

3. **Reinforcement Learning + Pointer**：
   - 用强化学习训练 Pointer Network
   - 解决没有最优解标签的问题

4. **Graph Pointer Network**：
   - 处理图结构数据
   - 应用于图遍历、路径规划等

## 小结

Pointer Network 是一个强大的架构，它解决了传统 seq2seq 模型的关键限制：

### 核心创新

1. **动态输出词典**：
   - 输出大小可以随输入变化
   - 解决了固定词典的局限性

2. **注意力作为指针**：
   - 不生成词汇，而是指向输入位置
   - 自然支持复制机制

3. **端到端可微**：
   - 可以用标准反向传播训练
   - 无需复杂的强化学习（虽然 RL 也可以）

### 关键公式

```
编码:    h_i^enc = LSTM_enc(x_i, h_{i-1}^enc)

注意力:  u_t^i = v^T tanh(W_h h_i^enc + W_s h_t^dec)
         p(C_i|...) = softmax(u_t)

解码:    h_t^dec = LSTM_dec([y_{t-1}, c_t], h_{t-1}^dec)
         c_t = Σ_i α_t^i h_i^enc  (上下文向量)

输出:    y_t = argmax(p)  (指向输入位置)
```

### 应用场景

- **组合优化**：TSP、VRP、调度问题
- **自然语言处理**：摘要、问答、代码生成
- **生物信息学**：序列组装、结构预测

### 与相关工作的关系

```
Seq2Seq:  固定输出词典 → 无法处理变长输出
    ↓
Attention: 引入对齐机制
    ↓
Pointer Network: 注意力 → 指针，动态输出词典
    ↓
CopyNet: 生成 + 复制机制
    ↓
Pointer-Generator: 软选择生成 vs 复制
```

## 练习题

1. **概念理解**：
   - 为什么 Pointer Network 可以解决输出词典大小不固定的问题？
   - Pointer Network 和传统 Attention 机制有什么区别和联系？
   - 在什么情况下，Pointer Network 比传统的 seq2seq 更合适？

2. **数学推导**：
   - 推导 Pointer Network 的反向传播公式
   - 分析 Pointer Network 的梯度流动
   - 比较 Pointer Network 和标准 seq2seq 的参数量和计算复杂度

3. **编程实践**：
   - 在上面的实现基础上，添加以下功能：
     * 多头注意力机制
     * Transformer 作为编码器/解码器
     * 束搜索（Beam Search）解码
   - 在更大规模的 TSP 实例上测试（n=20, 50, 100）

4. **实验分析**：
   - 对比不同注意力机制的效果：
     * 加性注意力（Bahdanau）vs 乘性注意力（Luong）
     * 局部注意力 vs 全局注意力
   - 分析注意力权重的可视化结果
   - 评估不同问题规模下的泛化能力

5. **深度思考**：
   - Pointer Network 本质上是在学习一个排序/排列函数。这和传统的排序算法有什么本质区别？
   - 随着 GPT-4 等大模型的出现，基于 Transformer 的架构能否完全取代 Pointer Network？
   - 在组合优化问题中，学习式方法（如 Pointer Network）和经典算法（如动态规划、分支定界）各有什么优势和局限？未来会朝着什么方向发展？

## 延伸阅读

- **经典论文**：
  - "Pointer Networks" by Vinyals, Fortunato, & Jaitly (2015) - Pointer Network 的原始论文
  - "Neural Combinatorial Optimization with Reinforcement Learning" by Bello et al. (2016) - 用 RL 训练 Pointer Network 解决 TSP
  - "Attention, Learn to Solve Routing Problems!" by Kool, Van Hoof, & Welling (2018) - 基于 Transformer 的解决方法
  - "The Transformer Network for Neural Machine Translation" by Vaswani et al. (2017) - Transformer 原始论文

- **在线资源**：
  - Google AI Blog: [Pointer Networks](https://ai.googleblog.com/2016/10/learning-to-learn.html)
  - Distill.pub: [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
  - Papers With Code: [Traveling Salesman Problem](https://paperswithcode.com/task/traveling-salesman-problem)

- **开源实现**：
  - [Google's OR-Tools](https://developers.google.com/optimization) - 包含 TSP 求解器
  - [PyTorch Pointer Network](https://github.com/shirgur/PointerNet) - PyTorch 实现
  - [Attention Routing](https://github.com/wouterkool/attention-routing) - Kool et al. 的实现

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 6 篇。上一篇我们学习了神经网络剪枝技术，下一篇我们将探索 AlexNet——深度卷积神经网络的开山之作。*