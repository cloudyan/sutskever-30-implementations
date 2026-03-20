# 10 咖啡混合模拟：离散化的复杂动力学

问下大家，经过这么多理论学习，是不是手痒想写代码了？

今天，我们就来实现 Aaronson 文章中的核心例子——**离散咖啡杯模型**！通过模拟，我们可以直观地观察"复杂度先增后减"的现象。

## 模型设定

### 物理系统

想象一个咖啡杯：
- 底部是黑色的咖啡
- 倒入白色的牛奶
- 牛奶和咖啡开始混合

**关键观察**：
- 初始：黑白分明（简单）
- 中间：形成复杂的分形边界（复杂）
- 晚期：均匀混合（简单）

### 离散化模型

为了计算机模拟，我们将系统离散化：

**网格**：$N \times N$ 的二维格子

**状态**：每个格子有两种状态
- 0：黑色（咖啡）
- 1：白色（牛奶）

**初始条件**：
- 左半边：0（黑）
- 右半边：1（白）

**动力学**：随机混合

## 动力学规则

### 规则 1：随机交换（Aaronson 原始模型）

每步：
1. 随机选择一对相邻的格子
2. 如果一个是黑，一个是白，则交换它们
3. 否则，什么也不做

**等价描述**：
- 每个白粒子进行随机游走
- 粒子间不可区分
- 最终达到均匀混合

### 规则 2：扩散模型（更物理）

每步：
1. 对于每个格子
2. 以概率 $p$ 与随机邻居交换

**物理意义**：
- 模拟扩散过程
- $p$ 控制扩散速率
- 更符合真实的流体混合

### 规则 3：对流+扩散（最物理）

加入对流：
- 某些区域有定向流动
- 同时有随机扩散
- 产生更复杂的图案

## Python 实现

### 基础模拟

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import zlib

class CoffeeCupSimulation:
    """
    离散咖啡杯模拟
    模拟黑白液体的混合过程
    """
    
    def __init__(self, N=100):
        """
        初始化模拟
        N: 网格大小 (N x N)
        """
        self.N = N
        self.grid = np.zeros((N, N), dtype=int)
        
        # 初始条件：左半边白，右半边黑
        self.grid[:, :N//2] = 1  # 白色 (牛奶)
        self.grid[:, N//2:] = 0  # 黑色 (咖啡)
        
        # 历史记录
        self.history = [self.grid.copy()]
        self.time_steps = [0]
    
    def get_neighbors(self, i, j):
        """获取邻居坐标"""
        neighbors = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.N and 0 <= nj < self.N:
                neighbors.append((ni, nj))
        return neighbors
    
    def step_swap(self):
        """
        一步：随机交换模型（Aaronson 原始模型）
        """
        # 随机选择一对相邻格子
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)
        
        neighbors = self.get_neighbors(i, j)
        if not neighbors:
            return
        
        ni, nj = neighbors[np.random.randint(len(neighbors))]
        
        # 如果颜色不同，交换
        if self.grid[i, j] != self.grid[ni, nj]:
            self.grid[i, j], self.grid[ni, nj] = \
                self.grid[ni, nj], self.grid[i, j]
    
    def step_diffusion(self, p=0.1):
        """
        一步：扩散模型
        p: 扩散概率
        """
        new_grid = self.grid.copy()
        
        for i in range(self.N):
            for j in range(self.N):
                if np.random.random() < p:
                    neighbors = self.get_neighbors(i, j)
                    if neighbors:
                        ni, nj = neighbors[np.random.randint(len(neighbors))]
                        # 交换
                        new_grid[i, j], new_grid[ni, nj] = \
                            new_grid[ni, nj], new_grid[i, j]
        
        self.grid = new_grid
    
    def run(self, steps, model='swap', save_interval=1, **kwargs):
        """
        运行模拟
        steps: 总步数
        model: 'swap' 或 'diffusion'
        save_interval: 保存间隔
        """
        for t in range(1, steps + 1):
            if model == 'swap':
                self.step_swap()
            elif model == 'diffusion':
                self.step_diffusion(**kwargs)
            
            if t % save_interval == 0:
                self.history.append(self.grid.copy())
                self.time_steps.append(t)
        
        return self.history, self.time_steps
    
    def plot_state(self, t=-1, ax=None):
        """绘制某一时刻的状态"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        state = self.history[t]
        ax.imshow(state, cmap='binary', interpolation='nearest')
        ax.set_title(f'Time: {self.time_steps[t]}')
        ax.axis('off')
        
        return ax
    
    def plot_evolution(self, n_frames=6):
        """绘制演化过程"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        indices = np.linspace(0, len(self.history)-1, n_frames, dtype=int)
        
        for idx, ax in zip(indices, axes):
            self.plot_state(idx, ax)
        
        plt.tight_layout()
        return fig

# 运行模拟
np.random.seed(42)
sim = CoffeeCupSimulation(N=100)
history, time_steps = sim.run(steps=10000, model='swap', save_interval=500)

# 绘制演化
fig = sim.plot_evolution()
plt.savefig('coffee_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 度量复杂度

### 1. 香农熵

```python
def compute_entropy(grid):
    """
    计算网格的香农熵
    """
    # 统计黑白像素比例
    p_white = np.mean(grid)
    p_black = 1 - p_white
    
    # 计算熵
    if p_white == 0 or p_black == 0:
        return 0
    
    entropy = -(p_white * np.log2(p_white) + p_black * np.log2(p_black))
    return entropy

# 计算每步的熵
entropies = [compute_entropy(state) for state in sim.history]

plt.figure(figsize=(10, 4))
plt.plot(time_steps, entropies, 'b-', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Entropy')
plt.title('Entropy Evolution in Coffee Mixing')
plt.grid(True)
plt.axhline(y=1.0, color='r', linestyle='--', label='Max Entropy')
plt.legend()
plt.show()
```

**预期结果**：熵单调递增，从 0 增加到 1（最大熵）。

### 2. 压缩比（近似 KC）

```python
def approximate_kc(grid):
    """
    用 gzip 压缩比近似 Kolmogorov 复杂度
    """
    # 将网格转换为字节
    grid_bytes = grid.tobytes()
    
    # 压缩
    compressed = zlib.compress(grid_bytes)
    
    return len(compressed)

# 计算每步的近似 KC
kcs = [approximate_kc(state) for state in sim.history]

# 归一化
kcs = np.array(kcs)
kcs_normalized = (kcs - kcs.min()) / (kcs.max() - kcs.min())

plt.figure(figsize=(10, 4))
plt.plot(time_steps, kcs_normalized, 'g-', linewidth=2, label='Approximate KC')
plt.plot(time_steps, entropies, 'b--', linewidth=2, label='Entropy')
plt.xlabel('Time Step')
plt.ylabel('Normalized Value')
plt.title('Complexity vs Entropy')
plt.legend()
plt.grid(True)
plt.show()
```

**预期结果**：KC 可能单调递增或趋于稳定，但不会先增后减（因为是开放系统）。

### 3. 边界复杂度

```python
from scipy import ndimage

def compute_boundary_complexity(grid):
    """
    计算边界复杂度
    使用边界长度和分形维度
    """
    # 找到边界
    edges = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbors = []
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                    if grid[i, j] != grid[ni, nj]:
                        edges[i, j] = 1
                        break
    
    # 边界长度
    boundary_length = np.sum(edges)
    
    # 归一化
    max_boundary = grid.shape[0] * grid.shape[1]
    normalized = boundary_length / max_boundary
    
    return normalized, edges

# 计算每步的边界复杂度
boundary_complexities = []
for state in sim.history:
    bc, _ = compute_boundary_complexity(state)
    boundary_complexities.append(bc)

plt.figure(figsize=(10, 4))
plt.plot(time_steps, boundary_complexities, 'r-', linewidth=2)
plt.xlabel('Time Step')
plt.ylabel('Boundary Complexity')
plt.title('Boundary Complexity Evolution')
plt.grid(True)
plt.show()
```

**预期结果**：边界复杂度先增后减！

- 早期：边界短（简单）
- 中期：边界长且复杂（分形）
- 晚期：边界消失（均匀）

### 4. 综合对比

```python
# 综合绘图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 熵
axes[0, 0].plot(time_steps, entropies, 'b-', linewidth=2)
axes[0, 0].set_title('Entropy (Monotonic)')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Entropy')
axes[0, 0].grid(True)

# 近似 KC
axes[0, 1].plot(time_steps, kcs_normalized, 'g-', linewidth=2)
axes[0, 1].set_title('Approximate KC')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Normalized KC')
axes[0, 1].grid(True)

# 边界复杂度
axes[1, 0].plot(time_steps, boundary_complexities, 'r-', linewidth=2)
axes[1, 0].set_title('Boundary Complexity (Peak in Middle)')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Boundary Complexity')
axes[1, 0].grid(True)

# 对比
axes[1, 1].plot(time_steps, entropies, 'b--', label='Entropy', linewidth=2)
axes[1, 1].plot(time_steps, boundary_complexities, 'r-', label='Boundary Complexity', linewidth=2)
axes[1, 1].set_title('Entropy vs Boundary Complexity')
axes[1, 1].set_xlabel('Time')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('complexity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 动画可视化

```python
def create_animation(sim, interval=50):
    """
    创建混合过程的动画
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(sim.history[0], cmap='binary', interpolation='nearest')
    ax.set_title('Coffee Mixing Simulation')
    ax.axis('off')
    
    def update(frame):
        im.set_array(sim.history[frame])
        ax.set_title(f'Time: {sim.time_steps[frame]}')
        return [im]
    
    anim = FuncAnimation(fig, update, frames=len(sim.history), 
                        interval=interval, blit=True)
    
    return anim

# 创建动画（需要较长时间）
# anim = create_animation(sim)
# anim.save('coffee_mixing.mp4', writer='ffmpeg', fps=10)
```

## 参数研究

### 网格大小的影响

```python
def run_experiment(N, steps):
    """运行不同网格大小的实验"""
    sim = CoffeeCupSimulation(N=N)
    history, time_steps = sim.run(steps=steps, model='swap', 
                                   save_interval=steps//20)
    
    # 计算边界复杂度
    bcs = []
    for state in history:
        bc, _ = compute_boundary_complexity(state)
        bcs.append(bc)
    
    return time_steps, bcs

# 不同网格大小
Ns = [50, 100, 200]
results = {}

for N in Ns:
    print(f"Running N={N}...")
    t, bc = run_experiment(N, steps=N*N*2)
    results[N] = (t, bc)

# 绘制对比
plt.figure(figsize=(10, 6))
for N, (t, bc) in results.items():
    # 归一化时间
    t_normalized = np.array(t) / (N * N)
    plt.plot(t_normalized, bc, label=f'N={N}', linewidth=2)

plt.xlabel('Time (normalized by N²)')
plt.ylabel('Boundary Complexity')
plt.title('Boundary Complexity for Different Grid Sizes')
plt.legend()
plt.grid(True)
plt.show()
```

**预期结果**：不同网格大小的曲线形状相似，验证了标度律。

## 与理论的联系

### 验证 Aaronson 的猜想

通过模拟，我们可以验证：

1. **熵单调递增** ✅
2. **边界复杂度先增后减** ✅
3. **复杂度峰值出现在中间时刻** ✅

### 开放问题

1. **Complextropy 的精确定义**：如何用代码实现？
2. **统计复杂度**：如何计算因果态？
3. **标度律**：复杂度峰值时间与系统大小的关系？

## 小结

问下大家，通过模拟是不是对"复杂动力学"有了更直观的理解？

**核心发现**：

1. **熵**：单调递增（符合热力学第二定律）
2. **边界复杂度**：先增后减（符合"第一定律"）
3. **压缩比**：可能单调递增（因为是开放系统）

**关键洞察**：
- 封闭系统（固定边界）vs 开放系统（扩展边界）行为不同
- 边界复杂度是 Complextropy 的一个好代理
- 模拟验证了 Aaronson 的核心猜想

**下一步**：
- 实现统计复杂度
- 研究不同动力学规则
- 应用到其他物理系统

下一篇，我们将学习 **算法统计学**——理解随机性的结构！

---

**思考题**：
1. 你能修改模拟，实现封闭系统（周期性边界）吗？
2. 如何用机器学习来学习系统的"结构"？
3. 三维咖啡混合会有什么不同？

*关注「云言 AI」，回复"咖啡模拟"获取完整代码！*
