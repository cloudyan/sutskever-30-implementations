# 依赖管理文档

本项目使用 `uv` 工具进行依赖管理。`uv` 是一个极快的 Python 包安装工具，兼容 pip 和虚拟环境管理。

## 安装 uv

```bash
# 使用 pipx 安装 uv
pipx install uv

# 或使用 curl 安装
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 项目依赖

本项目需要以下核心依赖：
- `numpy` >= 1.21.0 - 数值计算库，用于所有深度学习实现
- `matplotlib` >= 3.3.0 - 数据可视化和绘图
- `scipy` >= 1.7.0 - 科学计算扩展

## 依赖管理命令

### 安装依赖

```bash
# 安装所有依赖（包括开发依赖）
uv sync

# 只安装项目依赖
uv sync --no-dev
```

### 添加新依赖

```bash
# 添加运行时依赖
uv add package_name

# 添加开发依赖
uv add --dev package_name

# 添加带版本约束的依赖
uv add "package_name>=1.0.0,<2.0.0"
```

示例：添加本项目所需的依赖
```bash
uv add numpy matplotlib scipy
```

### 移除依赖

```bash
# 移除依赖
uv remove package_name
```

### 更新依赖

```bash
# 更新所有依赖到最新兼容版本
uv sync --upgrade

# 更新特定依赖
uv add --upgrade package_name
```

### 创建虚拟环境

```bash
# 创建虚拟环境（通常 uv sync 会自动处理）
uv venv

# 激活虚拟环境
source .venv/bin/activate
```

### 导出依赖列表

```bash
# 导出为 requirements.txt 格式
uv export -o requirements.txt
```

## 项目特定设置

### 虚拟环境激活

```bash
# 进入项目目录
cd sutskever-30-implementations

# 确保虚拟环境是最新的
uv sync

# 激活虚拟环境
source .venv/bin/activate

# 运行任意 notebook
jupyter notebook 02_char_rnn_karpathy.ipynb
```

### 安装后的验证

安装依赖后，可以通过以下命令验证：

```bash
python -c "import numpy, matplotlib, scipy; print('所有依赖安装成功')"
```

## 依赖添加流程

1. 将依赖添加到 `pyproject.toml`
2. 运行 `uv sync` 同步依赖
3. 在 notebook 或 Python 脚本中导入测试

注意：由于本项目是教育性质的深度学习实现套件，核心依赖 `numpy`、`matplotlib`、`scipy` 已配置在 `pyproject.toml` 中，通常不需要手动添加。

## 故障排除

如果遇到虚拟环境问题，可以重新创建：

```bash
# 删除现有虚拟环境
rm -rf .venv

# 重新创建并安装依赖
uv venv
uv sync
```

如果遇到 Python 路径配置错误（如 "Could not find platform independent libraries"），通常是因为虚拟环境损坏，按上述步骤删除并重新创建即可解决。

常见错误信息：
```
Fatal Python error: init_fs_encoding: failed to get the Python codec of the filesystem encoding
ModuleNotFoundError: No module named 'encodings'
```

这种情况下只需删除 `.venv` 目录并重新运行 `uv venv` 和 `uv sync` 命令即可。

## 依赖兼容性

本项目使用 Python 3.13+，所有依赖都已针对此版本进行测试，确保在深度学习实现中有最佳兼容性。
