#!/bin/bash
# GitHub Pages 部署脚本

echo "🚀 部署 AI 数学基础文档到 GitHub Pages..."
echo ""

# 检查是否在正确的目录
if [ ! -f "index.html" ]; then
    echo "❌ 请在 ai-math-docs 目录运行此脚本"
    exit 1
fi

# 检查 git 是否配置
if ! command -v git &> /dev/null; then
    echo "❌ Git 未安装，请先安装 Git"
    exit 1
fi

# 检查是否已初始化 git
if [ ! -d ".git" ]; then
    echo "📦 初始化 Git 仓库..."
    git init
fi

# 输入 GitHub 用户名（如果还没有配置远程仓库）
if ! git remote get-url origin &> /dev/null; then
    read -p "请输入你的 GitHub 用户名：" username
    echo ""
    echo "🔗 添加远程仓库..."
    git remote add origin https://github.com/${username}/ai-math-docs.git
fi

# Git 操作
echo "📝 提交更改..."
git add .
git commit -m "📐 更新文档 - $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "⬆️  推送到 GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "✅ 部署完成！"
echo ""
echo "📖 访问地址："
echo "   https://YOUR_USERNAME.github.io/ai-math-docs/"
echo ""
echo "⚠️  注意："
echo "   1. GitHub Pages 可能需要 1-2 分钟更新"
echo "   2. 首次部署需要在 GitHub 仓库 Settings -> Pages 启用"
echo "   3. 确保仓库是 Public（公开）"
echo ""
echo "🔗 下一步："
echo "   1. 访问 https://github.com/YOUR_USERNAME/ai-math-docs/settings/pages"
echo "   2. 设置 Source: Deploy from a branch"
echo "   3. 设置 Branch: main / (root)"
echo "   4. 点击 Save"
echo ""
