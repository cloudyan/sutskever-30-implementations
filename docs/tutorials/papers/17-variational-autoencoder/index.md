# 变分自编码器(VAE)到底是怎么“又压缩又生成”的?

问下大家,你有没有想过一个很离谱的事情:

我能不能把一张图(比如 4x4 的小图)压缩成**两个数字**,然后再从这两个数字把图“还原”出来?

晓寒刚开始学 AutoEncoder 的时候,一度以为它就是个“学会复制输入”的网络。直到我第一次看懂 VAE 的那句核心话:

**编码器输出的不是一个点,而是一个分布。**

卧槽,这一下就不一样了:你不仅能压缩,还能在“压缩空间”里采样生成,还能做插值、做连续的语义变化。

这篇我们用纯 NumPy 写一个最小可运行的 VAE(前向+损失+可视化),把 ELBO + 重参数化技巧(reparameterization trick)讲明白。

## 先把直觉立起来:从“确定的编码”到“概率的编码”

普通 AutoEncoder:

- 编码器: x -> z(一个确定的向量)
- 解码器: z -> x_hat

VAE:

- 编码器: x -> (mu(x), log_var(x))
- 然后从 q(z|x)=N(mu, diag(sigma^2)) 里采样 z
- 解码器: z -> x_hat

生活类比一下:

普通 AE 像“把文件压成一个固定压缩包”。

VAE 像“给文件一个概率编码”:同一个文件,每次解码出来可能略有不同,但总体保持相似,而且整个潜变量空间是连续可走的。

## VAE 的目标函数:ELBO(重构 + KL)

VAE 常见的训练目标可以写成:

\[
\mathcal{L}(x) = \underbrace{\mathbb{E}_{q(z|x)}[-\log p(x|z)]}_{\text{重构损失}} + \underbrace{KL\big(q(z|x)\;||\;p(z)\big)}_{\text{正则:把潜空间拉回标准正态}}
\]

你可以把它理解成:

- 重构项:别瞎编,要能还原输入
- KL 项:别把每个样本的 z 都塞到天涯海角,要让整体潜空间“规整”“可采样”

很多教程会把它叫 ELBO 的负号版本,本质就是同一件事。

## 最关键的技巧:重参数化(reparameterization trick)

问题:采样 z 是随机操作,怎么反向传播?

答案:把随机性“挪出去”:

\[
z = \mu + \sigma \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
\]

直觉类比:

你不是直接随机抽 z,而是先抽一个标准噪声 \(\epsilon\),再用 \(\mu,\sigma\) 去“缩放+平移”它。

这样 \(\mu,\sigma\) 都在确定计算图里,梯度就能传回去。

## 纯 NumPy 代码实现(最小可运行)

下面的实现和 notebook `17_variational_autoencoder.ipynb` 一致:我们用一个小 MLP 编码器/解码器,做前向和损失,再做潜空间可视化。

### 1) 工具函数 + VAE 类

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    # clip 防止 exp 溢出
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class VAE:
    """最小 VAE: encoder 输出 (mu, log_var), decoder 输出 x_recon"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: x -> h -> (mu, log_var)
        self.W_enc_h = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b_enc_h = np.zeros(hidden_dim)

        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_mu = np.zeros(latent_dim)

        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_logvar = np.zeros(latent_dim)

        # Decoder: z -> h -> x_recon
        self.W_dec_h = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b_dec_h = np.zeros(hidden_dim)

        self.W_recon = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_recon = np.zeros(input_dim)

    def encode(self, x):
        """x: (batch, input_dim) -> mu/log_var: (batch, latent_dim)"""
        h = relu(x @ self.W_enc_h + self.b_enc_h)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """z = mu + sigma * eps, eps ~ N(0, I)"""
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z):
        """z: (batch, latent_dim) -> x_recon: (batch, input_dim)"""
        h = relu(z @ self.W_dec_h + self.b_dec_h)
        x_recon = sigmoid(h @ self.W_recon + self.b_recon)
        return x_recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z

    def loss(self, x, x_recon, mu, log_var):
        """VAE loss = recon(BCE) + KL(q(z|x)||p(z))"""
        recon = -np.sum(
            x * np.log(x_recon + 1e-8) + (1 - x) * np.log(1 - x_recon + 1e-8)
        )

        kl = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        return recon + kl, recon, kl
```

### 2) 造一个“可控”的小数据集(4x4 图案)

我们生成四种简单图案(横线/竖线/对角线/角块),再加一点噪声。

```python
def generate_patterns(num_samples=200):
    data = []

    for i in range(num_samples):
        pattern = np.zeros((4, 4))

        if i % 4 == 0:
            pattern[1:2, :] = 1
        elif i % 4 == 1:
            pattern[:, 2:3] = 1
        elif i % 4 == 2:
            np.fill_diagonal(pattern, 1)
        else:
            pattern[:2, :2] = 1

        noise = np.random.randn(4, 4) * 0.05
        pattern = np.clip(pattern + noise, 0, 1)

        data.append(pattern.flatten())

    return np.array(data)


X_train = generate_patterns(200)

fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i].reshape(4, 4), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'pattern {i}')
    ax.axis('off')
plt.suptitle('Toy data')
plt.show()
```

### 3) 跑一次前向+看损失(未训练,只是验证管道)

```python
vae = VAE(input_dim=16, hidden_dim=32, latent_dim=2)

x = X_train[0:1]
x_recon, mu, log_var, z = vae.forward(x)

total, recon, kl = vae.loss(x, x_recon, mu, log_var)
print('mu:', mu)
print('log_var:', log_var)
print('z:', z)
print('loss total/recon/kl:', total, recon, kl)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
ax1.imshow(x.reshape(4, 4), cmap='gray', vmin=0, vmax=1)
ax1.set_title('original')
ax1.axis('off')

ax2.imshow(x_recon.reshape(4, 4), cmap='gray', vmin=0, vmax=1)
ax2.set_title('recon (untrained)')
ax2.axis('off')
plt.show()
```

## 潜空间到底在干嘛?三种经典可视化

### 1) 把所有样本编码到 2D 潜空间

```python
latent = []
types = []

for i, x in enumerate(X_train):
    mu, _ = vae.encode(x.reshape(1, -1))
    latent.append(mu[0])
    types.append(i % 4)

latent = np.array(latent)
types = np.array(types)

plt.figure(figsize=(7, 6))
sc = plt.scatter(latent[:, 0], latent[:, 1], c=types, cmap='tab10', alpha=0.6)
plt.colorbar(sc, label='pattern type')
plt.grid(True, alpha=0.3)
plt.title('Latent space (untrained)')
plt.show()
```

直觉上:训练好之后,同类图案会在潜空间里聚成团,不同类分开,并且团与团之间的路径是连续的。

### 2) 从先验 p(z)=N(0,I) 采样生成

```python
z = np.random.randn(8, 2)
gen = vae.decode(z)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(gen[i].reshape(4, 4), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.suptitle('Samples from prior (untrained)')
plt.show()
```

### 3) 在潜空间里插值(看“语义连续性”)

```python
x1 = X_train[0:1]
x2 = X_train[1:2]

mu1, _ = vae.encode(x1)
mu2, _ = vae.encode(x2)

alphas = np.linspace(0, 1, 8)
imgs = []
for a in alphas:
    z = (1 - a) * mu1 + a * mu2
    imgs.append(vae.decode(z)[0])

fig, axes = plt.subplots(1, 8, figsize=(16, 2))
for i, ax in enumerate(axes):
    ax.imshow(imgs[i].reshape(4, 4), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'a={alphas[i]:.2f}')
    ax.axis('off')
plt.suptitle('Latent interpolation (untrained)')
plt.show()
```

## 重参数化技巧的“可视化直觉”

同一个 x 对应的是一个高斯分布 q(z|x)=N(mu, sigma^2)。你多采样几次,会围着 mu 抖动。

```python
x = X_train[0:1]
mu, log_var = vae.encode(x)

zs = []
for _ in range(200):
    zs.append(vae.reparameterize(mu, log_var)[0])
zs = np.array(zs)

plt.figure(figsize=(6, 6))
plt.scatter(zs[:, 0], zs[:, 1], alpha=0.3, s=15)
plt.scatter(mu[0, 0], mu[0, 1], color='red', s=200, marker='*', label='mu')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.title('z samples around mu')
plt.show()
```

## 小结

VAE 你只要记住三个关键词,基本就不会学歪:

1) **分布编码**:编码器输出 (mu, log_var),不是一个确定 z
2) **ELBO/损失**:重构项 + KL 项(把潜空间拉回标准正态)
3) **重参数化**:z = mu + sigma * eps,把随机性挪到 eps

再往后你会看到一堆变体:beta-VAE、VQ-VAE、IWAE、diffusion 等,但“概率潜变量 + 可采样 + 可训练”这套骨架非常重要。

## 练习题

1) 给这个 VAE 加一个最小训练循环(例如 SGD),观察重构图像和潜空间聚类如何变化。

2) 把潜变量维度 latent_dim 从 2 改成 8,再用 PCA/t-SNE 做可视化,看看聚类是否更明显。

3) 实现 beta-VAE:把 KL 项乘一个系数 beta,试试 beta=0.1/1/4 对生成质量和可解释性的影响。

4) 把重构损失从 BCE 换成 MSE,在这个 toy 数据上差异是什么?什么时候更合适?

## 延伸阅读

1) Kingma & Welling, 2014, "Auto-Encoding Variational Bayes" (VAE)

2) Rezende et al., 2014, "Stochastic Backpropagation and Approximate Inference in Deep Generative Models"

3) beta-VAE: Higgins et al., 2017, "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"

---

*本文是《Sutskever 30 论文纯 NumPy 实现》系列教程的第 17 篇。*

---

**如果觉得有帮助,欢迎关注公众号「图解AI」,获取更多深度学习教程!**
