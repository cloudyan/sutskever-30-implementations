# Claude Code CLI 安装指南

_Windows Git Bash / macOS / Linux 完整安装教程_

---

## 🚀 快速安装

### Windows (Git Bash) - pnpm 方式（推荐）

```bash
# 1. 如果已安装过，先完全卸载
pnpm remove -g @anthropic-ai/claude-code
npm uninstall -g @anthropic-ai/claude-code  # 如果之前用 npm 安装过

# 2. 清理缓存
rm -rf $(pnpm root -g)/@anthropic-ai
rm -rf ~/.cache/claude-code

# 3. 全新安装（pnpm v10+ 必须加这两个参数）
pnpm add -g @anthropic-ai/claude-code --allow-build=@anthropic-ai/claude-code --ignore-scripts=false

# 4. 验证安装
claude --version

# 5. 启动
claude
```

### Windows (Git Bash) - npm 方式（最简单）

```bash
# 如果之前安装过，先卸载
npm uninstall -g @anthropic-ai/claude-code

# 安装
npm install -g @anthropic-ai/claude-code

# 验证
claude --version

# 启动
claude
```

### macOS / Linux

```bash
# npm 安装（推荐）
npm install -g @anthropic-ai/claude-code

# 或使用 pnpm
pnpm add -g @anthropic-ai/claude-code --allow-build=@anthropic-ai/claude-code --ignore-scripts=false

# 验证
claude --version
```

---

## ⚠️ 常见问题与解决

### 1. `claude: command not found`

**原因**：全局安装目录不在 PATH 中

**解决**：

```bash
# 查看 pnpm 全局目录
pnpm root -g

# 添加到 PATH（~/.bashrc 或 ~/.zshrc）
export PATH=$(pnpm root -g)/.bin:$PATH
source ~/.bashrc  # 或 source ~/.zshrc
```

### 2. `Error: claude native binary not installed`

**原因**：pnpm v10+ 默认阻止构建脚本

**解决**：

```bash
# 完全卸载
pnpm remove -g @anthropic-ai/claude-code
rm -rf $(pnpm root -g)/@anthropic-ai

# 重新安装（必须加参数）
pnpm add -g @anthropic-ai/claude-code --allow-build=@anthropic-ai/claude-code --ignore-scripts=false
```

### 3. `ERR_PNPM_PUBLIC_HOIST_PATTERN_DIFF`

**原因**：全局目录配置冲突

**解决**：

```bash
# 清理全局目录
rm -rf $(pnpm root -g)

# 重新安装
pnpm add -g @anthropic-ai/claude-code --allow-build=@anthropic-ai/claude-code --ignore-scripts=false
```

### 4. Git Bash 路径问题（Windows）

**现象**：Claude Code 无法找到 Git Bash

**解决**：

```bash
# 获取真实路径
cygpath -w /usr/bin/bash
# 输出示例：C:\Users\xxx\AppData\Local\Programs\Git\usr\bin\bash.exe

# 设置环境变量
export CLAUDE_CODE_GIT_BASH_PATH="C:\Users\xxx\AppData\Local\Programs\Git\usr\bin\bash.exe"

# 永久设置（写入 ~/.bashrc）
echo 'export CLAUDE_CODE_GIT_BASH_PATH="C:\Users\xxx\AppData\Local\Programs\Git\usr\bin\bash.exe"' >> ~/.bashrc
source ~/.bashrc
```

### 5. 安装插件时报错：`git` not found（Windows）⭐ 常见

**现象**：

```
Failed to install: Failed to clone repository:
Command 'git' not found or is in an unsafe location
```

**根本原因**：

Claude Code 的配置文件 `~/.claude/settings.json` 中的 `env.PATH` 环境变量不包含 Git 安装路径。

> 虽然 Git 在 Windows cmd、PowerShell、Git Bash 中都能正常运行（已添加到用户级 PATH），但 Claude Code 使用的是自定义 PATH，缺少 Git 路径。

**解决方案**：

编辑 `~/.claude/settings.json`，在 `env.PATH` 中添加 Git 路径：

```json
{
  "env": {
    "PATH": "/c/Users/Zhanjun.Yan/AppData/Local/Programs/Git/cmd:/c/Users/Zhanjun.Yan/AppData/Local/Programs/Git/bin:/mingw64/bin:/usr/bin:/bin:/cmd:$PATH"
  }
}
```

> ⚠️ 注意：将 `Zhanjun.Yan` 替换为你的实际 Windows 用户名。

**关键要点**：

1. **问题出在 Claude Code 的配置，而非系统 PATH** — 系统 PATH 正常不代表 Claude Code 能找到 Git
2. **必须同时添加 `cmd` 和 `bin` 目录** — `cmd` 包含 `git.exe`，`bin` 包含其他依赖
3. **修改配置后必须完全退出 Claude Code 并重启** — 仅重启终端不够

**排查步骤**：

```bash
# 1. 确认 Git 在 Git Bash 中可用
which git
# 应输出：/c/Users/xxx/AppData/Local/Programs/Git/cmd/git

# 2. 查看 Claude Code 配置
cat ~/.claude/settings.json

# 3. 如果 env.PATH 不存在或不包含 Git 路径，添加之

# 4. 完全退出 Claude Code（关闭终端窗口），重新打开
```

---

## 🌐 国内网络配置

### 配置代理端点

编辑配置文件 `~/.claude-code-config.json`：

```json
{
  "mcpServers": {},
  "anthropicBaseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "model": "sonnet",
  "availableModels": ["opus", "sonnet", "haiku"],
  "env": {
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "kimi-k2.5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.5-plus",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.5"
  }
}
```

### 设置 API Key

```bash
export ANTHROPIC_API_KEY="your-proxy-api-key"
```

或添加到 `~/.bashrc`：

```bash
echo 'export ANTHROPIC_API_KEY="your-api-key"' >> ~/.bashrc
source ~/.bashrc
```

---

## 📋 安装参数说明

### pnpm v10+ 关键参数

| 参数 | 作用 | 必需性 |
|------|------|--------|
| `--allow-build=@anthropic-ai/claude-code` | 授权原生包执行构建脚本 | ⭐⭐⭐ 必需 |
| `--ignore-scripts=false` | 确保不忽略脚本 | ⭐⭐ 推荐 |
| `-g` | 全局安装 | ⭐⭐⭐ 必需 |

### 为什么需要这些参数？

**pnpm v10+ 安全机制变化**：
- 默认阻止所有包的 `postinstall` 脚本
- Claude Code 需要执行脚本下载原生二进制文件
- 不授权会导致安装不完整

---

## 🔍 验证安装

```bash
# 检查版本
claude --version

# 检查安装位置
which claude

# 查看配置
claude --help

# 测试连接
claude --version
```

**成功输出示例**：
```
@anthropic-ai/claude-code@x.x.x
```

---

## 🧹 完全卸载

```bash
# pnpm 安装
pnpm remove -g @anthropic-ai/claude-code

# npm 安装
npm uninstall -g @anthropic-ai/claude-code

# 清理缓存
rm -rf $(pnpm root -g)/@anthropic-ai
rm -rf ~/.cache/claude-code

# 清理配置（可选）
rm ~/.claude-code-config.json
```

---

## 📚 相关资源

- **官方文档**：https://docs.anthropic.com/claude-code
- **GitHub 仓库**：https://github.com/anthropics/claude-code
- **npm 包**：https://www.npmjs.com/package/@anthropic-ai/claude-code

---

## 💡 最佳实践

### 1. 选择安装方式

| 场景 | 推荐方式 |
|------|---------|
| 已用 pnpm 管理全局包 | pnpm（加参数） |
| 第一次安装 CLI | npm（最简单） |
| 临时使用 | npx（不安装） |

### 2. 网络优化

- 使用稳定的代理服务
- 配置正确的 API 端点
- 避免频繁切换代理

### 3. 配置管理

- 使用环境变量管理 API Key
- 配置文件备份到安全位置
- 定期更新到最新版本

---

*最后更新：2026-04-29*
