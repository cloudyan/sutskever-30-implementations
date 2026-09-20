#!/bin/bash
# AI Math Docs 启动脚本

echo "🚀 启动 AI 数学基础文档站点..."
echo ""

# 检查是否安装了 docsify
if command -v docsify &> /dev/null; then
    echo "✅ 使用 docsify CLI 启动..."
    echo "📖 访问地址：http://localhost:3000"
    echo "按 Ctrl+C 停止服务"
    echo ""
    cd /Users/cloudyan/.openclaw/workspace/ai-math-docs
    docsify serve .
else
    echo "⚠️  docsify CLI 未安装"
    echo ""
    echo "请选择启动方式："
    echo ""
    echo "1️⃣  使用 Python HTTP 服务器（需要 Python 3）"
    echo "2️⃣  直接用浏览器打开 index.html"
    echo "3️⃣  安装 docsify CLI（推荐）"
    echo ""
    read -p "请选择 (1/2/3): " choice
    
    case $choice in
        1)
            echo ""
            echo "📖 使用 Python HTTP 服务器..."
            echo "🌐 访问地址：http://localhost:8000"
            echo "按 Ctrl+C 停止服务"
            echo ""
            cd /Users/cloudyan/.openclaw/workspace/ai-math-docs
            python3 -m http.server 8000
            ;;
        2)
            echo ""
            echo "📖 用浏览器打开..."
            open /Users/cloudyan/.openclaw/workspace/ai-math-docs/index.html
            ;;
        3)
            echo ""
            echo "📦 安装 docsify CLI..."
            npm install -g docsify-cli
            echo ""
            echo "✅ 安装完成！重新启动脚本即可。"
            echo ""
            ;;
        *)
            echo "❌ 无效选择"
            ;;
    esac
fi
