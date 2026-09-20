# AI 数学基础文档 - 快速部署指南

_3 步部署到互联网，获得可分享的 URL_

---

## ⚡ 快速部署（3 步）

### 步骤 1️⃣：创建 GitHub 仓库

1. 访问 https://github.com/new
2. 仓库名：`ai-math-docs`
3. 可见性：**Public**（公开）
4. 点击 **Create repository**

---

### 步骤 2️⃣：推送文档

在终端执行：

```bash
# 进入文档目录
cd /Users/cloudyan/.openclaw/workspace/ai-math-docs

# 初始化并推送
git init
git add .
git commit -m "📐 AI 数学基础文档"
git branch -M main

# 替换 YOUR_USERNAME 为你的 GitHub 用户名
git remote add origin https://github.com/YOUR_USERNAME/ai-math-docs.git
git push -u origin main
```

---

### 步骤 3️⃣：启用 GitHub Pages

1. 访问：`https://github.com/YOUR_USERNAME/ai-math-docs/settings/pages`
2. **Source** 选择：`Deploy from a branch`
3. **Branch** 选择：`main` / `(root)`
4. 点击 **Save**

等待 1-2 分钟，刷新页面即可看到访问地址！

---

## 🎉 完成！

你的文档现在可以通过以下地址访问：

```
https://YOUR_USERNAME.github.io/ai-math-docs/
```

**示例**：
- https://cloudyan.github.io/ai-math-docs/

---

## 📱 分享

现在你可以：

✅ 将 URL 发送到飞书/微信
✅ 在手机上查看
✅ 分享给朋友和同事
✅ 更新文档后重新推送（自动更新）

---

## 🔄 更新文档

修改文档后，重新部署：

```bash
cd /Users/cloudyan/.openclaw/workspace/ai-math-docs
./deploy.sh
```

或手动：

```bash
git add .
git commit -m "更新内容"
git push
```

---

## 🌐 自定义域名（可选）

### 有域名？

1. 在仓库根目录创建 `CNAME` 文件：
   ```bash
   echo "aimath.fun" > CNAME
   git add CNAME
   git commit -m "添加自定义域名"
   git push
   ```

2. 在域名服务商添加 CNAME 记录：
   ```
   类型：CNAME
   主机：www
   值：YOUR_USERNAME.github.io
   ```

3. 在 GitHub Pages 设置中验证

---

## ❓ 遇到问题？

### 页面显示 404

- 等待 1-2 分钟（GitHub 需要时间构建）
- 检查 `index.html` 是否在根目录
- 确认 Pages 设置正确

### CSS/JS 加载失败

- 检查网络连接
- CDN 可能需要代理（中国大陆）

### 数学公式不显示

- MathJax CDN 可能需要代理
- 尝试更换 CDN 源

---

## 📊 部署状态

访问：`https://github.com/YOUR_USERNAME/ai-math-docs/deployments`

查看部署历史和状态。

---

*最后更新：2026-04-22*
