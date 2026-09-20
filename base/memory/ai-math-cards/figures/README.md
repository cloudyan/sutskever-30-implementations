# TikZ 数学图形绘制指南

## 📁 文件说明

- `partial-derivative-visualization.tex` - 完整的 TikZ 图形代码
- `compile.sh` - 编译脚本（生成 PDF 和图片）

## 🎨 包含的图形

1. **偏导数的几何意义** - 3D曲面与切面
2. **梯度下降过程** - 在损失函数曲面上的优化路径
3. **链式法则分解图** - 复合函数求导的分解
4. **神经网络反向传播** - 链式法则在神经网络中的应用
5. **导数与微分的几何对比** - 真实变化 vs 线性近似

## 🚀 使用方法

### 方法1：在线编译（推荐）

访问 [Overleaf](https://www.overleaf.com/)，创建新项目，上传 `.tex` 文件，直接编译查看。

### 方法2：本地编译

需要安装 TeX Live 或 MiKTeX：

```bash
# 编译为 PDF
pdflatex partial-derivative-visualization.tex

# 或者使用 latexmk 自动编译
latexmk -pdf partial-derivative-visualization.tex

# 转换为图片（需要 ImageMagick）
convert -density 300 partial-derivative-visualization.pdf \
    -quality 90 partial-derivative.png
```

### 方法3：生成单独的图片

修改 `.tex` 文件，使用 `standalone` 类生成单独的图形文件。

## 📦 依赖包

- `tikz` - 基础绘图
- `pgfplots` - 3D 绘图
- `3d` - 3D 库
- `calc` - 计算库

## 💡 学习资源

- [TikZ 官方手册](https://www.ctan.org/pkg/pgf)
- [PGFPlots 手册](http://pgfplots.sourceforge.net/pgfplots.pdf)
- [Overleaf TikZ 教程](https://www.overleaf.com/learn/latex/TikZ_package)
- [3Blue1Brown 的 Manim](https://github.com/3b1b/manim) - 动画制作

## 🔧 常用代码片段

### 绘制函数曲线
```latex
\draw[thick, blue, domain=0:5, smooth, variable=\x] 
    plot (\x, {\x*\x});
```

### 绘制 3D 曲面
```latex
\addplot3[
    surf,
    shader=interp,
    domain=-2:2,
    domain y=-2:2,
] {x^2 + y^2};
```

### 绘制节点和箭头
```latex
\node[draw, fill=blue!20] (A) at (0,0) {A};
\node[draw, fill=red!20] (B) at (3,0) {B};
\draw[->, thick] (A) -- (B);
```
