# GitHub Pages 部署指南

_将 AI 数学基础文档部署到 GitHub Pages，提供互联网访问_

---

## 📋 部署方式对比

| 方式 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **GitHub Pages** | 免费、稳定、自定义域名 | 需要 GitHub 账号 | ⭐⭐⭐⭐⭐ |
| **Vercel** | 自动部署、速度快 | 需要注册 | ⭐⭐⭐⭐ |
| **Netlify** | 拖拽部署、表单功能 | 需要注册 | ⭐⭐⭐⭐ |
| **Cloudflare Pages** | 速度快、免费 | 较新 | ⭐⭐⭐ |

---

## 方式 1：GitHub Pages（推荐）

### 步骤 1：准备 GitHub 仓库

```bash
# 如果还没有 GitHub 账号，先注册 https://github.com

# 创建新仓库（名字随意，比如 ai-math-docs）
# 访问 https://github.com/new
# 仓库名：ai-math-docs
# 可见性：Public（公开）
# 不要初始化 README
```

### 步骤 2：推送文档到 GitHub

```bash
cd /Users/cloudyan/.openclaw/workspace/ai-math-docs

# 初始化 git
git init

# 添加所有文件
git add .

# 提交
git commit -m "📐 AI 数学基础文档 - 初始版本"

# 添加远程仓库（替换为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/ai-math-docs.git

# 推送
git branch -M main
git push -u origin main
```

### 步骤 3：启用 GitHub Pages

1. 访问你的仓库：`https://github.com/YOUR_USERNAME/ai-math-docs`
2. 点击 **Settings**（设置）
3. 左侧菜单点击 **Pages**
4. **Source** 选择：`Deploy from a branch`
5. **Branch** 选择：`main` / `(root)`
6. 点击 **Save**

### 步骤 4：等待部署

等待 1-2 分钟，刷新页面，你会看到：

```
Your site is live at `https://YOUR_USERNAME.github.io/ai-math-docs/`
```

### 🎉 完成！

访问地址：`https://YOUR_USERNAME.github.io/ai-math-docs/`

---

## 方式 2：Vercel 部署

### 步骤 1：注册 Vercel

访问 https://vercel.com/signup

### 步骤 2：导入 GitHub 仓库

1. 登录 Vercel
2. 点击 **Add New Project**
3. 选择 **Import Git Repository**
4. 选择你的 `ai-math-docs` 仓库
5. 点击 **Deploy**

### 步骤 3：获取访问地址

部署完成后，你会得到类似地址：
```
https://ai-math-docs-xxx.vercel.app
```

---

## 方式 3：Netlify 部署

### 步骤 1：拖拽部署（最简单）

1. 访问 https://app.netlify.com/drop
2. 将 `ai-math-docs` 文件夹拖到页面上
3. 等待上传完成
4. 获得访问地址

### 步骤 2：Git 部署（推荐）

1. 注册 Netlify：https://www.netlify.com/
2. 点击 **Add new site** → **Import an existing project**
3. 选择 GitHub
4. 选择 `ai-math-docs` 仓库
5. 点击 **Deploy site**

---

## 🌐 自定义域名（可选）

### GitHub Pages 自定义域名

1. 购买域名（如：aimath.fun）
2. 在域名服务商添加 CNAME 记录：
   ```
   类型：CNAME
   主机：@ 或 www
   值：YOUR_USERNAME.github.io
   ```
3. 在仓库根目录创建 `CNAME` 文件：
   ```bash
   echo "aimath.fun" > /Users/cloudyan/.openclaw/workspace/ai-math-docs/CNAME
   git add CNAME
   git commit -m "Add custom domain"
   git push
   ```
4. 在 GitHub Pages 设置中验证

---

## 📝 自动化部署脚本

创建 `deploy.sh` 脚本：

```bash
#!/bin/bash
# GitHub Pages 部署脚本

echo "🚀 部署 AI 数学基础文档到 GitHub Pages..."

# 检查是否在正确的目录
if [ ! -f "index.html" ]; then
    echo "❌ 请在 ai-math-docs 目录运行此脚本"
    exit 1
fi

# Git 操作
git add .
git commit -m "📝 更新文档 - $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main

echo ""
echo "✅ 部署完成！"
echo "📖 访问地址：https://YOUR_USERNAME.github.io/ai-math-docs/"
echo ""
echo "注意：GitHub Pages 可能需要 1-2 分钟更新"
```

使用方法：
```bash
chmod +x deploy.sh
./deploy.sh
```

---

## 🔧 常见问题

### Q1: 页面显示 404

**解决**：
- 等待 1-2 分钟（GitHub 需要时间构建）
- 检查 `index.html` 是否在根目录
- 确认 Pages 设置正确

### Q2: CSS/JS 加载失败

**解决**：
- 检查 `_sidebar.md` 路径是否正确
- 确保 CDN 链接可访问（中国大陆可能需要代理）

### Q3: 数学公式不显示

**解决**：
- 检查 MathJax CDN 是否可访问
- 尝试更换 CDN 源（如使用 jsDelivr）

---

## 📊 部署后优化

### 1. 添加 SEO 元标签

编辑 `index.html`：

```html
<meta name="description" content="AI 数学基础 60 天学习计划 - 从线性代数到 Transformer 的完整数学体系">
<meta name="keywords" content="AI, 数学，机器学习，深度学习，线性代数，概率论">
<meta name="author" content="悟空 AI 助手">

<!-- Open Graph for social sharing -->
<meta property="og:title" content="AI 数学基础 60 天">
<meta property="og:description" content="从线性代数到 Transformer 的完整数学体系">
<meta property="og:image" content="https://YOUR_URL/og-image.png">
```

### 2. 添加统计代码（可选）

编辑 `index.html`，在 `</body>` 前添加：

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### 3. 添加 favicon

将 favicon.ico 放到根目录，在 `index.html` 添加：

```html
<link rel="icon" href="favicon.ico" type="image/x-icon">
```

---

## 🎯 推荐方案

**最佳实践**：

1. **GitHub 仓库**：`github.com/YOUR_USERNAME/ai-math-docs`
2. **GitHub Pages**：`YOUR_USERNAME.github.io/ai-math-docs`
3. **自定义域名**：`aimath.fun`（可选）
4. **自动部署**：使用 `deploy.sh` 脚本

---

*最后更新：2026-04-22*
