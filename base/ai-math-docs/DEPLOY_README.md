# 🚀 部署到 GitHub Pages - 一键脚本

---

## 📋 部署前准备

1. **GitHub 账号**：如果没有，访问 https://github.com/signup 注册
2. **Git 已安装**：Mac 自带，Windows 需安装 https://git-scm.com

---

## ⚡ 自动部署（推荐）

### 1. 运行部署脚本

```bash
cd /Users/cloudyan/.openclaw/workspace/ai-math-docs
./deploy.sh
```

脚本会：
- ✅ 初始化 Git 仓库
- ✅ 提交所有文件
- ✅ 推送到 GitHub
- ✅ 提示启用 GitHub Pages

### 2. 启用 GitHub Pages

访问脚本输出的 URL，然后：

1. 进入 **Settings** → **Pages**
2. 设置 **Source**: `Deploy from a branch`
3. 设置 **Branch**: `main` / `(root)`
4. 点击 **Save**

### 3. 等待部署

1-2 分钟后，访问：
```
https://YOUR_USERNAME.github.io/ai-math-docs/
```

---

## 📝 手动部署

```bash
cd /Users/cloudyan/.openclaw/workspace/ai-math-docs

# 初始化
git init
git add .
git commit -m "📐 AI 数学基础文档"

# 关联远程仓库（替换用户名）
git remote add origin https://github.com/YOUR_USERNAME/ai-math-docs.git

# 推送
git branch -M main
git push -u origin main
```

---

## 🔄 更新文档

修改文档后重新部署：

```bash
cd /Users/cloudyan/.openclaw/workspace/ai-math-docs
./deploy.sh
```

---

## 🌐 自定义域名

1. 编辑 `CNAME` 文件，填入你的域名：
   ```
   aimath.fun
   ```

2. 提交：
   ```bash
   git add CNAME
   git commit -m "添加自定义域名"
   git push
   ```

3. 在域名服务商添加 CNAME 记录：
   ```
   类型：CNAME
   主机：www
   值：YOUR_USERNAME.github.io
   ```

---

## 📱 分享

部署完成后，你可以：

- ✅ 将 URL 发送到飞书/微信
- ✅ 在手机上查看
- ✅ 分享给朋友和同事

---

## ❓ 常见问题

**Q: 页面显示 404？**
- 等待 1-2 分钟（GitHub 需要时间构建）
- 检查是否正确启用 Pages

**Q: 样式加载失败？**
- CDN 可能需要代理（中国大陆）
- 尝试直接打开 `index.html`

**Q: 如何删除部署？**
- 在 GitHub 仓库 Settings 删除仓库即可

---

*最后更新：2026-04-22*
