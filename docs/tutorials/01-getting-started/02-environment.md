# 1.2 环境搭建：5 分钟搞定开发环境！

问下大家，你有没有遇到过这种情况？

想跑一个深度学习项目，结果光是搭环境就花了好几个小时，各种依赖冲突、版本不兼容，最后搞得心态都崩了！

晓寒刚开始学的时候也踩过无数坑，今天我就来教你如何 5 分钟快速搞定开发环境！

## Python 环境配置

首先，我们需要 Python 环境。推荐两个选择：

### 选项 1：用 Anaconda（推荐新手）

Anaconda 是一个 Python 发行版，自带了很多科学计算库，非常适合新手。

**安装步骤：**

1. 去 [Anaconda 官网](https://www.anaconda.com/) 下载对应系统的安装包
2. 双击安装，一路"下一步"就行
3. 安装完成后，打开终端（Windows 用 Anaconda Prompt）

验证安装：
```bash
conda --version
```

如果能看到版本号，说明安装成功！

### 选项 2：用 venv（推荐老手）

如果你已经有 Python 了，可以用 venv 创建虚拟环境：

```bash
# 创建虚拟环境
python -m venv sutskever-env

# 激活虚拟环境
# macOS/Linux:
source sutskever-env/bin/activate
# Windows:
sutskever-env\Scripts\activate
```

### 选项 3：使用 uv 管理环境

使用 [uv](https://docs.astral.sh/uv/)【推荐】, 因为它能提供更快的安装速度和更好的依赖管理

```bash
# 确保 Python ≥ 3.11
python --version

cd paper-python
# 使用 uv 创建虚拟环境并安装依赖
uv venv --python 3.11
# 也可以指定提示符名称
uv venv --python 3.11 --prompt paper_311

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows
# 关闭
deactivate

# 同步依赖
uv sync
# 如果项目和缓存目录不同，可以修改，然后同步依赖更快（用 reflink）
uv cache dir
echo 'export UV_CACHE_DIR="/Volumes/data/.cache/uv"' >> ~/.zshrc

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置你的 API 密钥

# 运行验证
python 00-env/simple_check.py
```

## 必要依赖安装

好，环境准备好了，现在安装必要的依赖。

本项目只需要三个核心库：

```bash
pip install numpy matplotlib scipy jupyter
```

就这么简单！没有 PyTorch，没有 TensorFlow，就用纯 NumPy！

我们来逐个看看这些库是做什么的：

| 库名 | 用途 | 为什么需要 |
|------|------|-----------|
| **numpy** | 数值计算 | 我们整个项目的核心！所有算法都用它实现 |
| **matplotlib** | 绘图 | 可视化算法结果，直观理解 |
| **scipy** | 科学计算 | 提供一些额外的科学计算函数 |
| **jupyter** | 交互式笔记本 | 运行我们的 notebook，边学边练 |

## Jupyter Notebook 使用指南

Jupyter Notebook 是我们学习这个项目的主要工具。如果你还没用过，别担心，超级简单！

### 启动 Jupyter

在项目目录下运行：

```bash
jupyter notebook
```

这会自动打开浏览器，你会看到项目的文件列表。

### 基本操作

打开一个 notebook 后，你会看到很多"单元格"。单元格有两种类型：

1. **代码单元格**：可以写 Python 代码，按 `Shift+Enter` 运行
2. **Markdown 单元格**：可以写文档，用 Markdown 格式

**常用快捷键：**
- `Shift+Enter`：运行当前单元格并跳到下一个
- `Ctrl+Enter`：运行当前单元格，保持在当前位置
- `A`：在当前单元格上方插入新单元格
- `B`：在当前单元格下方插入新单元格
- `DD`：删除当前单元格（按两次 D）
- `M`：切换到 Markdown 模式
- `Y`：切换到代码模式

### 推荐的工作流程

1. **从头到尾运行**：第一次打开 notebook，建议从头到尾运行一遍
2. **仔细看注释**：我们的代码有详细注释，理解每一步在做什么
3. **动手修改**：别只看，试着修改参数，观察变化
4. **添加自己的笔记**：在 notebook 中添加你自己的理解

## 快速验证环境

安装完成后，我们来快速验证一下环境是否正常：

```python
import numpy as np
import matplotlib.pyplot as plt

# 测试 NumPy
x = np.array([1, 2, 3, 4, 5])
print(f"NumPy 数组: {x}")
print(f"NumPy 版本: {np.__version__}")

# 测试 Matplotlib
plt.plot(x, x ** 2)
plt.title("测试图")
plt.show()

print("✅ 环境验证成功！")
```

你可以把这段代码复制到一个新 notebook 中运行，如果能看到输出和图表，说明环境没问题！

## 常见问题解决

### 问题 1：import 报错

**问题**：运行时提示 `ModuleNotFoundError: No module named 'xxx'`

**解决**：确保你在正确的环境中安装了依赖，用 `pip list` 检查已安装的包

### 问题 2：Jupyter 打不开

**问题**：运行 `jupyter notebook` 后浏览器没反应

**解决**：
1. 检查终端输出，看有没有报错
2. 尝试手动访问终端显示的 URL（通常是 `http://localhost:8888`）
3. 确保端口 8888 没被占用

### 问题 3：Matplotlib 绘图不显示

**问题**：调用 `plt.show()` 后没有图表弹出

**解决**：
- 在 Jupyter 中，确保你用了 `%matplotlib inline` 或 `%matplotlib notebook`
- 在脚本中，确保最后调用了 `plt.show()`

### 问题 4：NumPy 版本过旧

**问题**：某些新特性用不了

**解决**：升级 NumPy
```bash
pip install --upgrade numpy
```

## 项目运行方式

现在环境搭好了，我们来看看怎么运行项目：

### 方式 1：Jupyter Notebook（推荐）

```bash
# 进入项目目录
cd sutskever-30-implementations

# 启动 Jupyter
jupyter notebook
```

然后在浏览器中打开任意 notebook 开始学习！

### 方式 2：直接运行 Python 脚本

某些模块可以直接作为脚本运行：

```bash
# 运行测试
python test_training_utils_quick.py

# 运行训练
python train_lstm_baseline.py
```

### 方式 3：在 VS Code 中运行

如果你用 VS Code，可以安装 Jupyter 扩展，直接在 VS Code 中打开和运行 notebook！

## 推荐的开发工具

虽然不是必须的，但这些工具能让你的学习更顺畅：

### 编辑器
- **VS Code**：免费、功能强大，推荐安装 Python 和 Jupyter 扩展
- **PyCharm**：Python 开发利器，对 Jupyter 支持也很好
- **Jupyter Lab**：Jupyter Notebook 的升级版，界面更现代

### 终端
- **iTerm2**（macOS）：比系统自带终端好用
- **Windows Terminal**（Windows）：现代化的终端工具
- **Oh My Zsh**：让你的终端更酷炫、更高效

## 总结

综上所述，搭建开发环境其实很简单：

1. 选择 Anaconda 或 venv 配置 Python 环境
2. 安装 numpy、matplotlib、scipy、jupyter 四个依赖
3. 启动 Jupyter Notebook 开始学习

别让环境搭建成为你学习的障碍，5 分钟搞定，马上开始你的深度学习之旅！

下一节，我们来回顾一下 NumPy 的核心操作，为后面的实现打基础！
