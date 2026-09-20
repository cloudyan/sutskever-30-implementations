#!/bin/bash

# TikZ 图形编译脚本
# 使用方法: ./compile.sh [文件名]

FILE=${1:-partial-derivative-visualization}

echo "🎨 编译 TikZ 图形: $FILE.tex"

# 检查是否安装了 pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo "❌ 错误: 未找到 pdflatex"
    echo "请安装 TeX Live 或 MiKTeX:"
    echo "  macOS: brew install --cask mactex"
    echo "  Ubuntu: sudo apt-get install texlive-full"
    echo "  Windows: 下载安装 MiKTeX"
    exit 1
fi

# 编译 PDF
echo "📄 编译 PDF..."
pdflatex -interaction=nonstopmode "$FILE.tex"

if [ $? -eq 0 ]; then
    echo "✅ PDF 编译成功: $FILE.pdf"
    
    # 检查是否安装了 ImageMagick
    if command -v convert &> /dev/null; then
        echo "🖼️  转换为 PNG 图片..."
        convert -density 300 "$FILE.pdf" -quality 90 "$FILE.png"
        echo "✅ 图片生成成功: $FILE.png"
    else
        echo "⚠️  未找到 ImageMagick，跳过图片转换"
        echo "如需转换，请安装: brew install imagemagick"
    fi
else
    echo "❌ 编译失败，请检查错误信息"
    exit 1
fi

# 清理临时文件
echo "🧹 清理临时文件..."
rm -f "$FILE.aux" "$FILE.log" "$FILE.out" "$FILE.toc"

echo "🎉 完成!"
