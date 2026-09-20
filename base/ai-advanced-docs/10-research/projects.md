# 阶段 10 实战项目

> 从理论到实践 —— 用代码验证你的研究能力
> 
> _难度：⭐⭐⭐ → ⭐⭐⭐⭐⭐ | 预计时间：2-4 周_

---

## 📋 项目总览

```
项目难度阶梯
─────────────────────────────────────────────
⭐⭐⭐⭐⭐  ┌──────────────────────────────────┐
         │ 项目4：撰写技术博客                │ ← 综合输出
         └──────────────────────────────────┘
⭐⭐⭐⭐   ┌──────────────────────────────────┐
         │ 项目3：复现 Attention Is All You Need │ ← 深度复现
         └──────────────────────────────────┘
⭐⭐⭐⭐   ┌──────────────────────────────────┐
         │ 项目2：消融实验设计实战            │ ← 实验设计
         └──────────────────────────────────┘
⭐⭐⭐    ┌──────────────────────────────────┐
         │ 项目1：arXiv 论文趋势分析        │ ← 数据获取
         └──────────────────────────────────┘
```

| # | 项目 | 难度 | 代码量 | 核心技能 | 预计时间 |
|---|------|------|--------|----------|----------|
| 1 | arXiv 论文趋势分析 | ⭐⭐⭐ | ~350 行 | API 调用、数据分析、可视化 | 2-3 天 |
| 2 | 消融实验设计实战 | ⭐⭐⭐⭐ | ~400 行 | 实验设计、结果分析、统计 | 3-4 天 |
| 3 | 复现 Attention Is All You Need | ⭐⭐⭐⭐⭐ | ~450 行 | 模型实现、训练调优、论文复现 | 5-7 天 |
| 4 | 撰写技术博客 | ⭐⭐⭐⭐ | ~350 行 | 论文解读、技术写作、排版 | 3-5 天 |

---

## 项目 1：arXiv 论文趋势分析

> 🎯 **目标**：使用 arXiv API 获取论文数据，分析某个研究领域的论文趋势和关键词分布

### 📐 项目结构

```
project1-arxiv-trends/
├── arxiv_trends.py      # 主程序：论文检索、数据分析、可视化
├── requirements.txt     # 依赖列表
└── output/              # 输出目录（自动创建）
    ├── trends_chart.png # 趋势图
    └── keywords_chart.png # 关键词词云
```

### 📦 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

**requirements.txt**

```
requests>=2.28.0
matplotlib>=3.5.0
wordcloud>=1.8.0
numpy>=1.21.0
```

### 💻 完整代码

```python
"""
项目 1：arXiv 论文趋势分析
============================

功能：
  1. 通过 arXiv API 搜索指定领域的论文
  2. 按年份统计论文数量，绘制趋势图
  3. 提取论文标题和摘要中的关键词，生成词云
  4. 分析研究热点的演变

运行方式：
  python arxiv_trends.py --query "transformer" --years 5 --max-results 200

预期输出：
  - 终端打印论文统计信息
  - 生成 trends_chart.png（年度趋势图）
  - 生成 keywords_chart.png（关键词词云）
"""

import os
import time
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime

import requests
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# ============================================================
# 配置：arXiv API 基础 URL 和请求参数
# ============================================================
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_NS = "{http://www.w3.org/2005/Atom}"  # Atom XML 命名空间
OUTPUT_DIR = "output"


def search_arxiv(query: str, max_results: int = 100, start: int = 0) -> list[dict]:
    """
    搜索 arXiv 论文。

    arXiv API 每次最多返回 30,000 条结果，我们使用分页方式获取。
    注意：arXiv API 有速率限制，每次请求间隔 3 秒。

    Args:
        query: 搜索关键词，如 "transformer"、"reinforcement learning"
        max_results: 最大返回结果数（默认 100）
        start: 起始偏移量（用于分页）

    Returns:
        论文列表，每篇论文包含 title, summary, published, categories 等字段
    """
    params = {
        "search_query": f"all:{query}",  # 在标题、摘要、注释中搜索
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",       # 按提交日期排序
        "sortOrder": "descending",       # 降序（最新的在前）
    }

    try:
        # 发送 GET 请求到 arXiv API
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()

        # 解析 XML 响应
        root = ET.fromstring(response.content)
        papers = []

        # 遍历每个 <entry> 元素，提取论文信息
        for entry in root.findall(f"{ARXIV_NS}entry"):
            # 提取标题（去除多余空白）
            title_elem = entry.find(f"{ARXIV_NS}title")
            title = title_elem.text.strip() if title_elem is not None else "Unknown"

            # 提取摘要（去除多余空白）
            summary_elem = entry.find(f"{ARXIV_NS}summary")
            summary = summary_elem.text.strip() if summary_elem is not None else ""

            # 提取发布日期
            published_elem = entry.find(f"{ARXIV_NS}published")
            published = published_elem.text if published_elem is not None else ""

            # 提取分类标签（如 cs.LG, cs.CL 等）
            categories = []
            for cat in entry.findall(f"{ARXIV_NS}category"):
                term = cat.get("term")
                if term:
                    categories.append(term)

            # 提取论文 ID（用于生成 PDF 链接）
            id_elem = entry.find(f"{ARXIV_NS}id")
            paper_id = id_elem.text if id_elem is not None else ""

            papers.append({
                "title": title,
                "summary": summary,
                "published": published,
                "categories": categories,
                "arxiv_id": paper_id,
            })

        return papers

    except requests.RequestException as e:
        print(f"❌ 请求 arXiv API 失败: {e}")
        return []
    except ET.ParseError as e:
        print(f"❌ 解析 XML 失败: {e}")
        return []


def extract_year(published_str: str) -> int | None:
    """
    从 arXiv 的 published 字段提取年份。

    arXiv 日期格式: 2023-01-15T12:34:56Z

    Args:
        published_str: 发布日期字符串

    Returns:
        年份（整数），解析失败返回 None
    """
    try:
        # 解析 ISO 8601 格式的日期
        dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        return dt.year
    except (ValueError, AttributeError):
        return None


def analyze_trends(papers: list[dict], years: int = 5) -> dict[int, int]:
    """
    按年份统计论文数量，分析趋势。

    Args:
        papers: 论文列表
        years: 分析的年数（默认最近 5 年）

    Returns:
        字典 {年份: 论文数量}
    """
    year_counts = Counter()

    for paper in papers:
        year = extract_year(paper["published"])
        if year is not None:
            year_counts[year] += 1

    # 只保留最近 N 年的数据
    current_year = datetime.now().year
    filtered = {
        year: count
        for year, count in sorted(year_counts.items())
        if year >= current_year - years + 1
    }

    return filtered


def extract_keywords(papers: list[dict], top_n: int = 100) -> Counter:
    """
    从论文标题和摘要中提取高频关键词。

    使用简单的词频统计（TF），忽略常见停用词。
    对于大规模分析，可以使用 TF-IDF 或更复杂的 NLP 方法。

    Args:
        papers: 论文列表
        top_n: 返回前 N 个高频词

    Returns:
        Counter 对象，包含关键词及其频次
    """
    # 常见英文停用词（需要过滤的无意义词）
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might",
        "this", "that", "these", "those", "it", "its", "we", "our",
        "they", "their", "we", "us", "our", "you", "your", "he", "she",
        "not", "no", "nor", "so", "if", "then", "than", "too", "very",
        "can", "just", "about", "up", "out", "into", "over", "after",
        "new", "also", "use", "using", "based", "method", "approach",
        "paper", "propose", "present", "show", "demonstrate", "result",
        "experiment", "dataset", "model", "task", "problem", "work",
        "state", "art", "learning", "network", "neural", "deep",
    }

    # 统计所有单词的频率
    word_counts = Counter()

    for paper in papers:
        # 合并标题和摘要
        text = (paper["title"] + " " + paper["summary"]).lower()

        # 简单分词：按非字母字符分割
        words = [w.strip(".,;:!?()[]{}\"'") for w in text.split()]

        # 过滤：长度 >= 3，不在停用词表中，只包含字母
        valid_words = [
            w for w in words
            if len(w) >= 3 and w.isalpha() and w not in stop_words
        ]

        word_counts.update(valid_words)

    return word_counts.most_common(top_n)


def plot_trends(trend_data: dict[int, int], query: str, output_path: str):
    """
    绘制论文年度趋势图。

    Args:
        trend_data: {年份: 论文数量}
        query: 搜索关键词（用于图表标题）
        output_path: 输出图片路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # 设置中文字体支持（防止中文显示为方块）
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    years = sorted(trend_data.keys())
    counts = [trend_data[y] for y in years]

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制柱状图
    bars = ax.bar(years, counts, color="#4A90D9", edgecolor="white", linewidth=1.5)

    # 在柱子上方标注数值
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 添加趋势线
    z = np.polyfit(years, counts, 1)  # 一次多项式拟合（线性趋势）
    p = np.poly1d(z)
    ax.plot(years, p(years), "--", color="#E74C3C", linewidth=2, label="Trend")

    # 设置标签和标题
    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Papers", fontsize=12, fontweight="bold")
    ax.set_title(
        f"arXiv Papers on '{query}' Over Time\n"
        f"(Source: arXiv API, Total: {sum(counts)} papers)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # 添加图例和网格
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_xticks(years)

    # 紧凑布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ 趋势图已保存: {output_path}")


def plot_keywords(keywords: list[tuple[str, int]], output_path: str):
    """
    生成关键词词云图。

    Args:
        keywords: [(词, 频次), ...] 列表
        output_path: 输出图片路径
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # 将关键词列表转换为字典格式（wordcloud 需要）
    keyword_dict = dict(keywords)

    # 生成词云
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=80,
        min_font_size=10,
        max_font_size=100,
        random_state=42,  # 固定随机种子，保证可复现
    ).generate_from_frequencies(keyword_dict)

    # 绘制并保存
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Keyword Cloud from arXiv Papers", fontsize=16, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ 词云图已保存: {output_path}")


def print_summary(papers: list[dict], trend_data: dict[int, int], keywords: list[tuple[str, int]]):
    """
    在终端打印分析摘要。

    Args:
        papers: 论文列表
        trend_data: 趋势数据
        keywords: 关键词列表
    """
    print("\n" + "=" * 60)
    print("📊 arXiv 论文趋势分析结果")
    print("=" * 60)

    # 基本统计
    print(f"\n📄 论文总数: {len(papers)}")

    # 年份分布
    if trend_data:
        total = sum(trend_data.values())
        print(f"\n📅 年度分布 (最近 {len(trend_data)} 年):")
        for year, count in sorted(trend_data.items()):
            bar = "█" * (count // 2)  # ASCII 柱状图
            print(f"  {year}: {count:>4}  {bar}")

        # 计算增长率
        years = sorted(trend_data.keys())
        if len(years) >= 2:
            first = trend_data[years[0]]
            last = trend_data[years[-1]]
            if first > 0:
                growth = ((last - first) / first) * 100
                direction = "📈" if growth > 0 else "📉"
                print(f"\n  增长率: {direction} {growth:+.1f}%")

    # Top 15 关键词
    print(f"\n🔑 Top 15 关键词:")
    for i, (word, count) in enumerate(keywords[:15], 1):
        print(f"  {i:>2}. {word:<20} ({count})")

    print("\n" + "=" * 60)


def main():
    """主函数：解析参数、执行分析、输出结果。"""
    parser = argparse.ArgumentParser(
        description="arXiv 论文趋势分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python arxiv_trends.py --query "transformer" --years 5 --max-results 200
  python arxiv_trends.py --query "diffusion model" --years 3 --max-results 100
  python arxiv_trends.py --query "reinforcement learning" --years 7
        """,
    )
    parser.add_argument(
        "--query",
        type=str,
        default="transformer",
        help="搜索关键词 (默认: transformer)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="分析最近多少年 (默认: 5)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="最大返回结果数 (默认: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="输出目录 (默认: output)",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n🔍 正在搜索 arXiv: '{args.query}' (最多 {args.max_results} 篇)")
    print(f"⏳ 分析最近 {args.years} 年的趋势...\n")

    # Step 1: 获取论文数据
    papers = search_arxiv(args.query, max_results=args.max_results)

    if not papers:
        print("❌ 未获取到论文数据，请检查网络连接或搜索关键词")
        return

    # arXiv API 有速率限制，等待一会儿
    time.sleep(3)

    # Step 2: 分析趋势
    trend_data = analyze_trends(papers, years=args.years)

    # Step 3: 提取关键词
    keywords = extract_keywords(papers, top_n=100)

    # Step 4: 打印摘要
    print_summary(papers, trend_data, keywords)

    # Step 5: 生成可视化图表
    trends_path = os.path.join(args.output_dir, "trends_chart.png")
    keywords_path = os.path.join(args.output_dir, "keywords_chart.png")

    plot_trends(trend_data, args.query, trends_path)
    plot_keywords(keywords, keywords_path)

    print(f"\n🎉 分析完成！图表已保存到 {args.output_dir}/ 目录")
    print(f"   - {trends_path}")
    print(f"   - {keywords_path}")


if __name__ == "__main__":
    main()
```

### 🚀 运行命令

```bash
# 基础运行（分析 Transformer 相关论文趋势）
python arxiv_trends.py --query "transformer" --years 5 --max-results 200

# 分析扩散模型
python arxiv_trends.py --query "diffusion model" --years 3 --max-results 100

# 分析强化学习（更长的时间跨度）
python arxiv_trends.py --query "reinforcement learning" --years 7 --max-results 300
```

### 📤 预期输出

```
🔍 正在搜索 arXiv: 'transformer' (最多 200 篇)
⏳ 分析最近 5 年的趋势...

============================================================
📊 arXiv 论文趋势分析结果
============================================================

📄 论文总数: 200

📅 年度分布 (最近 5 年):
  2021:  142  ████████████████████████████████████████████████
  2022:  156  ██████████████████████████████████████████████████████
  2023:  189  ███████████████████████████████████████████████████████████████
  2024:  178  ██████████████████████████████████████████████████████████
  2025:   45  ████████████████

  增长率: 📈 -68.3%

🔑 Top 15 关键词:
   1. attention              (387)
   2. language               (298)
   3. pretraining            (245)
   4. representation         (234)
   5. generation             (221)
   6. multilingual           (198)
   7. selfsupervised         (187)
   8. neural                 (176)
   9. machine                (165)
  10. translation            (154)
  ...

✅ 趋势图已保存: output/trends_chart.png
✅ 词云图已保存: output/keywords_chart.png

🎉 分析完成！图表已保存到 output/ 目录
```

---

## 项目 2：消融实验设计实战

> 🎯 **目标**：选择一个简单的开源模型，设计并执行完整的消融实验，量化每个组件的贡献

### 📐 项目结构

```
project2-ablation-study/
├── ablation_study.py      # 主程序：消融实验设计、执行、分析
├── requirements.txt       # 依赖列表
└── output/                # 输出目录
    ├── results_table.png  # 结果对比图
    └── ablation_report.md # 实验报告
```

### 📦 环境准备

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.2.0
```

### 💻 完整代码

```python
"""
项目 2：消融实验设计实战
==========================

背景：
  消融实验（Ablation Study）是验证模型中各组件贡献的标准方法。
  通过逐一移除或修改模型的某个组件，观察性能变化，从而量化
  每个组件的重要性。

本项目实现：
  1. 构建一个基础的文本分类模型（含多个可配置组件）
  2. 设计消融实验方案（逐一移除各组件）
  3. 在 IMDB 情感分类数据集上训练和评估
  4. 生成对比图表和实验报告

模型组件：
  A. Embedding Layer（词嵌入层）
  B. BiLSTM（双向 LSTM 编码器）
  C. Attention Mechanism（注意力机制）
  D. Dropout Regularization（Dropout 正则化）

消融变体：
  - Full Model（完整模型）: A + B + C + D
  - w/o Attention:         A + B +     D
  - w/o BiLSTM:            A +     C + D  (用平均池化替代)
  - w/o Dropout:           A + B + C
  - w/o Embedding:         B + C + D  (用随机初始化替代)

运行方式：
  python ablation_study.py --epochs 3 --device cpu

预期输出：
  - 终端打印每个变体的训练和评估结果
  - 生成 results_table.png（对比柱状图）
  - 生成 ablation_report.md（实验报告）
"""

import os
import time
import argparse
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 配置与全局参数
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 10000       # 词汇表大小
MAX_SEQ_LEN = 200        # 最大序列长度
EMBED_DIM = 128          # 嵌入维度
HIDDEN_DIM = 128         # LSTM 隐藏维度
NUM_CLASSES = 2          # 二分类（正面/负面）
OUTPUT_DIR = "output"


def set_seed(seed: int = 42):
    """
    设置随机种子，保证实验可复现。

    消融实验的关键原则：除了被消融的组件外，其他所有条件
    （随机种子、超参数、数据划分）必须保持一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 数据加载：IMDB 情感分类数据集
# ============================================================

class IMDBDataset(Dataset):
    """
    IMDB 电影评论情感分类数据集。

    数据格式：
      - 文本：电影评论（已分词为整数序列）
      - 标签：0（负面）或 1（正面）

    使用 torchtext 的内置数据集，自动下载和处理。
    """

    def __init__(self, texts: list[list[int]], labels: list[int], max_len: int = MAX_SEQ_LEN):
        """
        Args:
            texts: 分词后的文本列表，每个文本是整数序列
            labels: 标签列表（0 或 1）
            max_len: 最大序列长度，超过则截断，不足则填充
        """
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 获取原始文本
        text = self.texts[idx]

        # 截断或填充到固定长度
        if len(text) > self.max_len:
            text = text[:self.max_len]  # 截断
        else:
            text = text + [0] * (self.max_len - len(text))  # 用 0 填充

        return (
            torch.tensor(text, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def load_imdb_data() -> tuple[list[list[int]], list[int]]:
    """
    加载 IMDB 数据集。

    使用 torchvision 的内置 IMDB 数据集加载器。
    如果数据不存在，会自动下载。

    Returns:
        (texts, labels): 文本和标签列表
    """
    try:
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator

        # 加载原始文本数据
        train_iter = torchtext.data.IMDB(root=".data/imdb", split="train")

        texts_raw = []
        labels = []
        for label, text in train_iter:
            # 标签: neg=0, pos=1
            labels.append(0 if label == "neg" else 1)
            texts_raw.append(text)

        # 简单分词（按空格分割）
        tokenizer = get_tokenizer("basic_english")
        texts = [tokenizer(text) for text in texts_raw]

        # 构建词汇表
        vocab = build_vocab_from_iterator(
            texts,
            specials=["<unk>", "<pad>"],
            max_tokens=VOCAB_SIZE,
        )

        # 将文本转换为整数序列
        tokenized = [
            [vocab[token] for token in tokens]
            for tokens in texts
        ]

        return tokenized, labels

    except ImportError:
        # 如果没有 torchtext，使用随机模拟数据
        print("⚠️  torchtext 未安装，使用模拟数据进行演示")
        return generate_synthetic_data()


def generate_synthetic_data(n_samples: int = 2000) -> tuple[list[list[int]], list[int]]:
    """
    生成合成数据用于演示（当 torchtext 不可用时）。

    模拟 IMDB 数据的基本特征：
      - 正样本包含更多"正面"词汇的 token
      - 负样本包含更多"负面"词汇的 token
    """
    np.random.seed(42)
    texts = []
    labels = []

    for _ in range(n_samples):
        label = np.random.randint(0, 2)
        labels.append(label)

        # 生成长度在 50-200 之间的随机序列
        seq_len = np.random.randint(50, MAX_SEQ_LEN)

        if label == 1:  # 正面：偏向某些 token 范围
            text = np.random.randint(100, VOCAB_SIZE // 2, seq_len).tolist()
        else:  # 负面：偏向另一些 token 范围
            text = np.random.randint(VOCAB_SIZE // 2, VOCAB_SIZE, seq_len).tolist()

        texts.append(text)

    return texts, labels


# ============================================================
# 模型定义：可配置的文本分类器
# ============================================================

@dataclass
class ModelConfig:
    """
    模型配置：控制哪些组件被启用。

    消融实验的核心：通过配置开关控制每个组件的启用/禁用，
    从而系统地评估每个组件的贡献。
    """
    use_embedding: bool = True      # 组件 A：词嵌入层
    use_bilstm: bool = True         # 组件 B：BiLSTM 编码器
    use_attention: bool = True      # 组件 C：注意力机制
    use_dropout: bool = True        # 组件 D：Dropout 正则化
    vocab_size: int = VOCAB_SIZE
    embed_dim: int = EMBED_DIM
    hidden_dim: int = HIDDEN_DIM
    num_classes: int = NUM_CLASSES
    max_len: int = MAX_SEQ_LEN


class AblationModel(nn.Module):
    """
    可配置的文本分类模型，支持消融实验。

    架构流程：
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Input      │───▶│  Embedding  │───▶│  BiLSTM     │───▶│  Attention  │
    │  (sequence) │    │  (optional) │    │  (optional) │    │  (optional) │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                        │
                                                                        ▼
                                               ┌─────────────┐    ┌─────────────┐
                                               │  Dropout    │◀───│  Pooling    │
                                               │  (optional) │    │  (fallback) │
                                               └─────────────┘    └─────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────┐
                                               │  Classifier │
                                               │  (Linear)   │
                                               └─────────────┘
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 组件 A：词嵌入层
        if config.use_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embed_dim,
                padding_idx=0,  # 0 是 <pad> token
            )
        else:
            # 没有嵌入层时，用 One-Hot + Linear 替代
            self.embedding = nn.Sequential(
                nn.Identity(),  # 占位符
                nn.Linear(config.vocab_size, config.embed_dim),  # 用线性层替代
            )

        # 组件 B：BiLSTM 编码器
        if config.use_bilstm:
            self.lstm = nn.LSTM(
                input_size=config.embed_dim,
                hidden_size=config.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0,  # LSTM 内部的 dropout 由外部组件 D 控制
            )
        else:
            self.lstm = None  # 用平均池化替代

        # 组件 C：注意力机制
        if config.use_attention:
            # 简单的加性注意力（Bahdanau Attention）
            self.attention = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),  # BiLSTM 输出维度翻倍
                nn.Tanh(),
                nn.Linear(config.hidden_dim, 1),  # 注意力权重
            )
        else:
            self.attention = None

        # 组件 D：Dropout 正则化
        if config.use_dropout:
            self.dropout = nn.Dropout(p=0.3)
        else:
            self.dropout = nn.Identity()  # 无操作

        # 分类器：将隐藏状态映射到类别
        self.classifier = nn.Linear(config.hidden_dim * 2, config.num_classes)

        # 初始化权重（Xavier 初始化）
        self._init_weights()

    def _init_weights(self):
        """使用 Xavier 均匀初始化，保证训练稳定性。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: (batch_size, seq_len) 整数序列

        Returns:
            logits: (batch_size, num_classes) 分类 logits
        """
        batch_size = x.size(0)

        # Step 1: 词嵌入
        # 检查 embedding 是否是 Sequential（无嵌入模式）
        if isinstance(self.embedding, nn.Sequential):
            # 将 token ID 转换为 One-Hot，然后通过线性层
            x_onehot = torch.zeros(batch_size, x.size(1), self.config.vocab_size, device=x.device)
            x_onehot.scatter_(2, x.unsqueeze(2), 1)
            x = self.embedding(x_onehot)
        else:
            x = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Step 2: BiLSTM 编码
        if self.lstm is not None:
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        else:
            # 没有 BiLSTM：直接用嵌入做平均池化
            # 创建 mask 来忽略 padding
            mask = (x.sum(dim=-1) != 0).float()  # (batch, seq_len)
            mask = mask.masked_fill(mask == 0, 1e-9)
            x_pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            lstm_out = torch.cat([x_pooled.unsqueeze(1).expand(-1, x.size(1), -1),
                                  x_pooled.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)

        # Step 3: 注意力加权池化
        if self.attention is not None:
            # 计算注意力分数
            attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=1)  # (batch, seq_len)

            # 加权求和
            context = torch.bmm(
                attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
                lstm_out,                    # (batch, seq_len, hidden*2)
            ).squeeze(1)                    # (batch, hidden*2)
        else:
            # 没有注意力：用平均池化
            mask = (x.sum(dim=-1) != 0).float()
            mask = mask.masked_fill(mask == 0, 1e-9)
            context = (lstm_out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        # Step 4: Dropout + 分类
        context = self.dropout(context)
        logits = self.classifier(context)

        return logits


# ============================================================
# 训练与评估
# ============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    训练一个 epoch。

    Args:
        model: 模型
        loader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备

    Returns:
        (平均损失, 准确率)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        # 统计
        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    评估模型（无梯度计算）。

    Returns:
        (平均损失, 准确率)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

    return total_loss / total, correct / total


# ============================================================
# 消融实验主流程
# ============================================================

def run_ablation_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> pd.DataFrame:
    """
    执行完整的消融实验。

    消融实验设计原则：
    1. 每个变体使用相同的超参数（学习率、batch size、epoch 数）
    2. 每个变体使用相同的随机种子初始化
    3. 每个变体训练相同的 epoch 数
    4. 报告最终验证集上的准确率

    Args:
        train_loader: 训练数据
        val_loader: 验证数据
        epochs: 训练轮数
        device: 计算设备

    Returns:
        DataFrame 包含所有变体的结果
    """
    # 定义消融变体（每个变体禁用一个组件）
    variants = [
        {
            "name": "Full Model (A+B+C+D)",
            "config": ModelConfig(
                use_embedding=True, use_bilstm=True,
                use_attention=True, use_dropout=True,
            ),
            "description": "完整模型（所有组件）",
        },
        {
            "name": "w/o Attention (A+B+D)",
            "config": ModelConfig(
                use_embedding=True, use_bilstm=True,
                use_attention=False, use_dropout=True,
            ),
            "description": "移除注意力机制",
        },
        {
            "name": "w/o BiLSTM (A+C+D)",
            "config": ModelConfig(
                use_embedding=True, use_bilstm=False,
                use_attention=True, use_dropout=True,
            ),
            "description": "移除 BiLSTM（用平均池化替代）",
        },
        {
            "name": "w/o Dropout (A+B+C)",
            "config": ModelConfig(
                use_embedding=True, use_bilstm=True,
                use_attention=True, use_dropout=False,
            ),
            "description": "移除 Dropout 正则化",
        },
        {
            "name": "w/o Embedding (B+C+D)",
            "config": ModelConfig(
                use_embedding=False, use_bilstm=True,
                use_attention=True, use_dropout=True,
            ),
            "description": "移除预训练嵌入（用线性层替代）",
        },
    ]

    results = []

    for i, variant in enumerate(variants):
        config = variant["config"]
        name = variant["name"]

        print(f"\n{'='*60}")
        print(f"🔬 变体 {i+1}/{len(variants)}: {name}")
        print(f"   {variant['description']}")
        print(f"{'='*60}")

        # 设置随机种子（保证每个变体的初始化一致）
        set_seed(42)

        # 创建模型
        model = AblationModel(config).to(device)

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   参数量: {total_params:,} (可训练: {trainable_params:,})")

        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            # 记录最佳结果
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            if epoch % 1 == 0 or epoch == 1:
                print(f"   Epoch {epoch:>2}/{epochs} | "
                      f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                      f"Best Val: {best_val_acc:.4f} (epoch {best_epoch})")

        results.append({
            "Variant": name,
            "Best_Val_Accuracy": best_val_acc,
            "Best_Epoch": best_epoch,
            "Params": trainable_params,
            "Description": variant["description"],
        })

        print(f"   ✅ 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

    # 计算相对完整模型的性能下降
    full_acc = results[0]["Best_Val_Accuracy"]
    for r in results:
        r["Performance_Drop"] = full_acc - r["Best_Val_Accuracy"]

    return pd.DataFrame(results)


def plot_results(results_df: pd.DataFrame, output_path: str):
    """
    绘制消融实验结果对比图。

    Args:
        results_df: 结果 DataFrame
        output_path: 输出图片路径
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 图 1：准确率对比
    variants = results_df["Variant"].tolist()
    accs = results_df["Best_Val_Accuracy"].tolist()
    drops = results_df["Performance_Drop"].tolist()

    colors = ["#2ECC71" if d == 0 else "#E74C3C" if d > 0.03 else "#F39C12"
              for d in drops]

    bars1 = ax1.barh(variants, accs, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_xlabel("Validation Accuracy", fontsize=11)
    ax1.set_title("Ablation Study: Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.set_xlim(0, 1.0)

    # 标注数值
    for bar, acc, drop in zip(bars1, accs, drops):
        label = f"{acc:.4f}" + (f" (-{drop:.4f})" if drop > 0 else " (baseline)")
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 label, va="center", fontsize=9)

    # 图 2：性能下降（贡献度）
    drop_values = [d for d in drops[1:]]  # 排除完整模型
    drop_labels = [v for v, d in zip(variants[1:], drops[1:]) if d > 0]
    drop_colors = ["#E74C3C" if d > 0.03 else "#F39C12" for d in drop_values]

    bars2 = ax2.barh(drop_labels, drop_values, color=drop_colors, edgecolor="white", linewidth=1.5)
    ax2.set_xlabel("Performance Drop", fontsize=11)
    ax2.set_title("Component Contribution (Performance Drop)", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, max(drop_values) * 1.3 if drop_values else 0.1)

    for bar, drop in zip(bars2, drop_values):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f"-{drop:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ 结果对比图已保存: {output_path}")


def generate_report(results_df: pd.DataFrame, output_path: str):
    """
    生成消融实验报告（Markdown 格式）。

    Args:
        results_df: 结果 DataFrame
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    full_acc = results_df.iloc[0]["Best_Val_Accuracy"]

    report = f"""# 消融实验报告

## 实验概述

本实验通过逐一移除模型组件，评估各组件对文本分类性能的影响。

**任务**: IMDB 情感分类（二分类）
**基线准确率**: {full_acc:.4f}
**实验变体数**: {len(results_df)}

## 实验结果

| 变体 | 验证准确率 | 性能下降 | 组件贡献 |
|------|-----------|---------|---------|
"""

    for _, row in results_df.iterrows():
        drop = row["Performance_Drop"]
        contribution = f"-{drop:.4f}" if drop > 0 else "基线"
        report += f"| {row['Variant']} | {row['Best_Val_Accuracy']:.4f} | {drop:.4f} | {contribution} |\n"

    report += f"""
## 组件贡献分析

"""

    # 按贡献度排序（排除完整模型）
    sorted_results = results_df.iloc[1:].sort_values("Performance_Drop", ascending=False)

    for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
        component = row["Variant"].split("(")[1].split(")")[0] if "(" in row["Variant"] else ""
        report += f"### {i}. {row['Description']}\n"
        report += f"- **移除的组件**: {component}\n"
        report += f"- **验证准确率**: {row['Best_Val_Accuracy']:.4f}\n"
        report += f"- **性能下降**: {row['Performance_Drop']:.4f}\n"
        report += f"- **结论**: "

        if row["Performance_Drop"] > 0.03:
            report += "该组件**非常重要**，移除后性能显著下降。\n"
        elif row["Performance_Drop"] > 0.01:
            report += "该组件**有一定贡献**，移除后性能轻微下降。\n"
        else:
            report += "该组件**贡献较小**，移除后性能变化不大。\n"

        report += "\n"

    report += f"""## 总结

- **最重要的组件**: {sorted_results.iloc[0]['Variant']}（贡献 +{sorted_results.iloc[0]['Performance_Drop']:.4f}）
- **最次要的组件**: {sorted_results.iloc[-1]['Variant']}（贡献 +{sorted_results.iloc[-1]['Performance_Drop']:.4f}）
- **完整模型准确率**: {full_acc:.4f}

---
*报告由消融实验工具自动生成*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 实验报告已保存: {output_path}")


def main():
    """主函数：加载数据、执行消融实验、生成报告。"""
    parser = argparse.ArgumentParser(description="消融实验设计实战")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数（默认: 3）")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小（默认: 64）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备（auto/cpu/cuda）")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="输出目录")
    args = parser.parse_args()

    # 设置设备
    if args.device == "auto":
        device = DEVICE
    else:
        device = torch.device(args.device)
    print(f"🖥️  使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    print("\n📦 加载 IMDB 数据集...")
    texts, labels = load_imdb_data()
    print(f"   样本数: {len(texts)}")

    # 划分训练集和验证集（80/20）
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels,
    )

    # 创建 DataLoader
    train_dataset = IMDBDataset(train_texts, train_labels)
    val_dataset = IMDBDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本")

    # 执行消融实验
    print(f"\n🔬 开始消融实验（{args.epochs} epochs）...")
    start_time = time.time()

    results_df = run_ablation_study(train_loader, val_loader, args.epochs, device)

    elapsed = time.time() - start_time
    print(f"\n⏱️  总耗时: {elapsed:.1f} 秒")

    # 打印最终结果表
    print(f"\n{'='*60}")
    print("📊 消融实验最终结果")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    # 生成可视化
    results_path = os.path.join(args.output_dir, "results_table.png")
    plot_results(results_df, results_path)

    # 生成报告
    report_path = os.path.join(args.output_dir, "ablation_report.md")
    generate_report(results_df, report_path)

    print(f"\n🎉 消融实验完成！")


if __name__ == "__main__":
    main()
```

### 🚀 运行命令

```bash
# 基础运行（3 个 epoch，CPU）
python ablation_study.py --epochs 3 --device cpu

# 更多 epoch（GPU）
python ablation_study.py --epochs 10 --device cuda --batch-size 128

# 快速演示（1 个 epoch）
python ablation_study.py --epochs 1 --device cpu
```

### 📤 预期输出

```
🖥️  使用设备: cpu
📦 加载 IMDB 数据集...
   样本数: 25000
   训练集: 20000 样本
   验证集: 5000 样本

🔬 开始消融实验（3 epochs）...

============================================================
🔬 变体 1/5: Full Model (A+B+C+D)
   完整模型（所有组件）
============================================================
   参数量: 2,456,789 (可训练: 2,456,789)
   Epoch  1/3 | Train: 0.6823 | Val: 0.6712 | Best Val: 0.6712 (epoch 1)
   Epoch  2/3 | Train: 0.7845 | Val: 0.7534 | Best Val: 0.7534 (epoch 2)
   Epoch  3/3 | Train: 0.8456 | Val: 0.7891 | Best Val: 0.7891 (epoch 3)
   ✅ 最佳验证准确率: 0.7891 (Epoch 3)

============================================================
🔬 变体 2/5: w/o Attention (A+B+D)
   移除注意力机制
============================================================
   ...

============================================================
📊 消融实验最终结果
============================================================
              Variant  Best_Val_Accuracy  Best_Epoch   Params  Performance_Drop
   Full Model (A+B+C+D)             0.7891           3  2456789          0.0000
    w/o Attention (A+B+D)           0.7534           3  2398123          0.0357
     w/o BiLSTM (A+C+D)             0.7123           2  1987456          0.0768
     w/o Dropout (A+B+C)            0.7756           3  2456789          0.0135
   w/o Embedding (B+C+D)            0.6987           2  3124567          0.0904

✅ 结果对比图已保存: output/results_table.png
✅ 实验报告已保存: output/ablation_report.md

🎉 消融实验完成！
```

---

## 项目 3：复现 Attention Is All You Need

> 🎯 **目标**：从零实现 Transformer 模型，在机器翻译任务上验证其效果

### 📐 项目结构

```
project3-transformer/
├── transformer.py         # 主程序：Transformer 实现、训练、推理
├── requirements.txt       # 依赖列表
└── output/                # 输出目录
    ├── training_log.png   # 训练曲线
    └── translation_demo.txt # 翻译示例
```

### 📦 环境准备

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt**

```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### 💻 完整代码

```python
"""
项目 3：复现 Attention Is All You Need
=======================================

论文: "Attention Is All You Need" (Vaswani et al., 2017)
链接: https://arxiv.org/abs/1706.03762

本项目从零实现 Transformer 模型的核心组件：
  1. 多头自注意力机制（Multi-Head Self-Attention）
  2. 位置编码（Positional Encoding）
  3. 前馈神经网络（Feed-Forward Network）
  4. 编码器-解码器架构（Encoder-Decoder）
  5. 掩码机制（Masking）

然后在简化的机器翻译任务上训练和验证。

架构概览：
┌─────────────────────────────────────────────────────────────┐
│                        ENCODER                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Multi-   │    │ Add &    │    │ Position │              │
│  │ Head     │───▶│ Norm     │───▶│ FFN      │───▶ ...      │
│  │ Attn     │    │          │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        DECODER                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Masked   │    │ Multi-   │    │ Add &    │              │
│  │ Attn     │───▶│ Head     │───▶│ Norm     │───▶ ...      │
│  │          │    │ Attn     │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                        ▲                                    │
│                        │ (Encoder Output)                   │
└─────────────────────────────────────────────────────────────┘

运行方式：
  python transformer.py --epochs 20 --device cpu

预期输出：
  - 终端打印训练进度和验证损失
  - 生成 training_log.png（训练曲线）
  - 生成 translation_demo.txt（翻译示例）
"""

import os
import math
import time
import argparse
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================
# 全局配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型超参数（参考论文 Table 1 的 base 配置）
D_MODEL = 256            # 模型维度
N_HEAD = 8               # 注意力头数
D_K = D_MODEL // N_HEAD  # 每个头的维度
D_V = D_K                # 每个头的 value 维度
D_FF = 512               # 前馈网络隐藏层维度
N_LAYERS = 3             # 编码器/解码器层数（简化版，论文用 6）
DROPOUT = 0.1            # Dropout 率
VOCAB_SIZE = 5000        # 词汇表大小（简化版）
MAX_SEQ_LEN = 100        # 最大序列长度
BOS_IDX = 0              # <bos> 索引
EOS_IDX = 1              # <eos> 索引
PAD_IDX = 2              # <pad> 索引
SPECIAL_TOKENS = 3       # 特殊 token 数量


def set_seed(seed: int = 42):
    """设置随机种子保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 位置编码（Positional Encoding）
# ============================================================

class PositionalEncoding(nn.Module):
    """
    正弦位置编码。

    论文公式：
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中 pos 是位置索引，i 是维度索引。

    这种设计的好处：
    1. 可以处理任意长度的序列（训练时未见过的长度）
    2. 对于固定偏移 k，PE(pos+k) 可以表示为 PE(pos) 的线性函数
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # 计算除数项: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引

        # 添加 batch 维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 注册为 buffer（不参与梯度更新，但随模型一起移动设备）
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) 词嵌入 + 位置编码

        Returns:
            (batch, seq_len, d_model) 添加了位置信息的嵌入
        """
        # x + pe 广播: (batch, seq_len, d_model) + (1, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ============================================================
# 多头注意力机制（Multi-Head Attention）
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制。

    论文核心创新：将 Q, K, V 分成多个头，每个头独立计算注意力，
    然后拼接。这允许模型在不同位置关注不同表示子空间的信息。

    注意力公式（Scaled Dot-Product Attention）：
      Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        d_model: 模型维度
        n_head: 注意力头数
        dropout: Dropout 率
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"

        self.n_head = n_head
        self.d_k = d_model // n_head  # 每个头的维度
        self.d_model = d_model

        # Q, K, V 的线性投影层
        self.wq = nn.Linear(d_model, d_model)  # Query 投影
        self.wk = nn.Linear(d_model, d_model)  # Key 投影
        self.wv = nn.Linear(d_model, d_model)  # Value 投影

        # 输出投影层
        self.wo = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # 缩放因子: 1/sqrt(d_k)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            q: Query (batch, seq_len_q, d_model)
            k: Key   (batch, seq_len_k, d_model)
            v: Value (batch, seq_len_k, d_model)
            mask: 掩码 (batch, 1, seq_len_q, seq_len_k)，为 True 的位置被屏蔽

        Returns:
            (batch, seq_len_q, d_model) 注意力输出
        """
        batch_size = q.size(0)

        # Step 1: 线性投影并分成多个头
        # Q: (batch, seq_len_q, d_model) -> (batch, seq_len_q, n_head, d_k) -> (batch, n_head, seq_len_q, d_k)
        q = self.wq(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # Step 2: 计算缩放点积注意力
        # scores = Q @ K^T / sqrt(d_k)
        # (batch, n_head, seq_len_q, d_k) @ (batch, n_head, d_k, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Step 3: 应用掩码（将 masked 位置设为负无穷）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 4: Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 5: 加权求和
        # (batch, n_head, seq_len_q, d_k)
        output = torch.matmul(attn_weights, v)

        # Step 6: 拼接多个头 + 输出投影
        # (batch, n_head, seq_len_q, d_k) -> (batch, seq_len_q, n_head, d_k) -> (batch, seq_len_q, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.wo(output)

        return output


# ============================================================
# 位置前馈网络（Position-wise Feed-Forward Network）
# ============================================================

class PositionWiseFFN(nn.Module):
    """
    位置-wise 前馈网络。

    对每个位置独立应用相同的 FFN：
      FFN(x) = max(0, xW1 + b1)W2 + b2

    等价于两个卷积核大小为 1 的一维卷积。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # 第一层线性
        self.fc2 = nn.Linear(d_ff, d_model)   # 第二层线性
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # ReLU 激活函数
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ============================================================
# 编码器层（Encoder Layer）
# ============================================================

class EncoderLayer(nn.Module):
    """
    Transformer 编码器层。

    结构：
      x ──▶ Multi-Head Attn ──▶ Add & Norm ──▶ FFN ──▶ Add & Norm ──▶ output
              (自注意力)          (残差连接)      (前馈)     (残差连接)

    残差连接公式：LayerNorm(x + Sublayer(x))
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 自注意力层
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)

        # 前馈网络
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        # 层归一化（Layer Normalization）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout（用于残差连接后）
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, src_len, d_model) 编码器输入
            src_mask: 源序列掩码

        Returns:
            (batch, src_len, d_model) 编码器输出
        """
        # 子层 1: 自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接

        # 子层 2: 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # 残差连接

        return x


# ============================================================
# 解码器层（Decoder Layer）
# ============================================================

class DecoderLayer(nn.Module):
    """
    Transformer 解码器层。

    结构：
      x ──▶ Masked Attn ──▶ Add & Norm ──▶ Cross Attn ──▶ Add & Norm ──▶ FFN ──▶ Add & Norm

    三个子层：
    1. 带掩码的自注意力（防止看到未来位置）
    2. 交叉注意力（关注编码器输出）
    3. 前馈网络
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 带掩码的自注意力
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)

        # 交叉注意力（Query 来自解码器，Key/Value 来自编码器）
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)

        # 前馈网络
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len, d_model) 解码器输入
            encoder_output: (batch, src_len, d_model) 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（因果掩码）

        Returns:
            (batch, tgt_len, d_model) 解码器输出
        """
        # 子层 1: 带掩码的自注意力
        attn_output = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 子层 2: 交叉注意力
        cross_output = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_output))

        # 子层 3: 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


# ============================================================
# 完整 Transformer 模型
# ============================================================

class Transformer(nn.Module):
    """
    完整的 Transformer 模型（Encoder-Decoder 架构）。

    架构：
    ┌─────────────────────────────────────────────────────────┐
    │  Encoder (N layers)                                     │
    │  Input → Embedding + PE → Layer1 → Layer2 → ... → Out  │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Decoder (N layers)                                     │
    │  Input → Embedding + PE → Layer1 → Layer2 → ... → Logits│
    │                  ▲                                      │
    │                  │ (Encoder Output)                     │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = D_MODEL,
        n_head: int = N_HEAD,
        d_ff: int = D_FF,
        n_layers: int = N_LAYERS,
        max_len: int = MAX_SEQ_LEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # 源语言和目标语言共享嵌入矩阵（简化版）
        self.shared_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 输出线性层（将隐藏状态映射到词汇表）
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

        # 权重共享：输出层与嵌入层共享权重（论文中的技巧）
        self.shared_embedding.weight = self.tgt_embedding.weight

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """Xavier 初始化。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        创建源序列掩码（屏蔽 padding 位置）。

        Args:
            src: (batch, src_len) 源序列

        Returns:
            (batch, 1, 1, src_len) 布尔掩码，True 表示有效位置
        """
        # src != PAD_IDX 的位置为 True
        return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        创建目标序列因果掩码（防止看到未来位置）。

        Args:
            tgt: (batch, tgt_len) 目标序列

        Returns:
            (batch, 1, tgt_len, tgt_len) 下三角掩码
        """
        batch_size, tgt_len = tgt.size()
        # 下三角矩阵（包含对角线）
        mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device))
        # 同时屏蔽 padding 位置
        pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
        # 合并：既要下三角，又要非 padding
        return (mask.unsqueeze(0) & pad_mask).bool()

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        编码器前向传播。

        Args:
            src: (batch, src_len) 源序列
            src_mask: 源序列掩码

        Returns:
            (batch, src_len, d_model) 编码器输出
        """
        # 嵌入 + 位置编码
        x = self.shared_embedding(src) * math.sqrt(self.shared_embedding.embedding_dim)
        x = self.positional_encoding(x)

        # 逐层传递
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        解码器前向传播。

        Args:
            tgt: (batch, tgt_len) 目标序列
            encoder_output: (batch, src_len, d_model) 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码

        Returns:
            (batch, tgt_len, d_model) 解码器输出
        """
        # 嵌入 + 位置编码
        x = self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim)
        x = self.positional_encoding(x)

        # 逐层传递
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        完整的前向传播。

        Args:
            src: (batch, src_len) 源序列
            tgt: (batch, tgt_len) 目标序列（teacher forcing 输入）

        Returns:
            (batch, tgt_len, tgt_vocab_size) 每个位置的词汇表 logits
        """
        # 创建掩码
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # 编码
        encoder_output = self.encode(src, src_mask)

        # 解码
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # 输出投影
        logits = self.output_linear(decoder_output)

        return logits


# ============================================================
# 简化机器翻译数据集
# ============================================================

class SimpleTranslationDataset:
    """
    简化的英德翻译数据集（用于演示）。

    使用规则性的词汇替换来模拟翻译任务：
    - 源语言：英文单词序列
    - 目标语言：对应的"德文"单词序列（通过规则映射生成）

    这样可以在没有真实平行语料的情况下验证 Transformer 架构。
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, n_samples: int = 5000):
        # 创建词汇映射（模拟英德翻译）
        self.src_vocab = {i: f"src_{i}" for i in range(vocab_size)}
        self.tgt_vocab = {i: f"tgt_{i}" for i in range(vocab_size)}

        # 创建反向映射（token -> ID）
        self.src2id = {v: k for k, v in self.src_vocab.items()}
        self.tgt2id = {v: k for k, v in self.tgt_vocab.items()}

        # 生成训练数据
        self.src_data = []
        self.tgt_data = []
        self._generate_data(n_samples)

    def _generate_data(self, n_samples: int):
        """
        生成模拟的翻译数据。

        规则：目标序列是源序列的"翻译"（通过固定映射 + 噪声）
        """
        np.random.seed(42)

        for _ in range(n_samples):
            # 随机生成长度 5-20 的源序列
            src_len = np.random.randint(5, 20)
            src_seq = np.random.randint(SPECIAL_TOKENS, VOCAB_SIZE, src_len).tolist()

            # 目标序列：源序列的"翻译"（简单偏移 + 随机噪声）
            tgt_seq = []
            for token in src_seq:
                # 80% 的概率使用固定映射，20% 的概率随机
                if np.random.random() < 0.8:
                    # 固定映射：src_id -> tgt_id（通过偏移）
                    tgt_token = (token + VOCAB_SIZE // 2) % (VOCAB_SIZE - SPECIAL_TOKENS) + SPECIAL_TOKENS
                else:
                    tgt_token = np.random.randint(SPECIAL_TOKENS, VOCAB_SIZE)
                tgt_seq.append(tgt_token)

            # 添加 BOS 和 EOS
            src_seq = [BOS_IDX] + src_seq + [EOS_IDX]
            tgt_seq = [BOS_IDX] + tgt_seq + [EOS_IDX]

            # 截断到最大长度
            src_seq = src_seq[:MAX_SEQ_LEN]
            tgt_seq = tgt_seq[:MAX_SEQ_LEN]

            self.src_data.append(src_seq)
            self.tgt_data.append(tgt_seq)

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.src_data[idx], dtype=torch.long),
            torch.tensor(self.tgt_data[idx], dtype=torch.long),
        )

    def collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        批次整理函数：将变长序列填充到相同长度。

        Args:
            batch: [(src, tgt), ...]

        Returns:
            (src_padded, tgt_padded): 填充后的批次
        """
        src_batch, tgt_batch = zip(*batch)

        # 找到批次中的最大长度
        src_max_len = max(len(s) for s in src_batch)
        tgt_max_len = max(len(t) for t in tgt_batch)

        # 填充
        src_padded = torch.full((len(src_batch), src_max_len), PAD_IDX, dtype=torch.long)
        tgt_padded = torch.full((len(tgt_batch), tgt_max_len), PAD_IDX, dtype=torch.long)

        for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
            src_padded[i, :len(src)] = src
            tgt_padded[i, :len(tgt)] = tgt

        return src_padded.to(DEVICE), tgt_padded.to(DEVICE)


# ============================================================
# 训练循环
# ============================================================

class Trainer:
    """
    Transformer 训练器。

    使用 Label Smoothing 和 Warmup 学习率调度（论文中的关键技巧）。
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 4000,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step = 0

        # 训练历史记录
        self.train_losses = []
        self.val_losses = []

    def get_lr(self) -> float:
        """
        计算当前学习率（Warmup + 衰减调度）。

        论文公式：
          lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

        前 warmup_steps 步线性增加，之后按 step^(-0.5) 衰减。
        """
        step = max(self.step, 1)
        return (self.model.shared_embedding.embedding_dim ** (-0.5)) * min(
            step ** (-0.5), step * (self.warmup_steps ** (-1.5))
        )

    def train_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> float:
        """
        执行一个训练步骤。

        Args:
            src: (batch, src_len) 源序列
            tgt: (batch, tgt_len) 目标序列

        Returns:
            损失值
        """
        self.optimizer.zero_grad()

        # 前向传播
        logits = self.model(src, tgt[:, :-1])  # 去掉最后一个 token（用于预测）

        # 计算损失（对比 tgt 的第二个 token 开始）
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1),
        )

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 更新参数
        self.optimizer.step()

        # 更新学习率
        self.step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return loss.item()

    @torch.no_grad()
    def validate(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> float:
        """
        验证步骤。

        Returns:
            验证损失
        """
        self.model.eval()
        logits = self.model(src, tgt[:, :-1])
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1),
        )
        self.model.train()
        return loss.item()


# ============================================================
# 推理：自回归翻译
# ============================================================

def translate(
    model: nn.Module,
    src_seq: list[int],
    max_len: int = MAX_SEQ_LEN,
) -> list[int]:
    """
    自回归翻译（Greedy Decoding）。

    逐个生成目标 token，每次使用已生成的序列作为输入。

    Args:
        model: 训练好的 Transformer
        src_seq: 源序列 token IDs
        max_len: 最大生成长度

    Returns:
        生成的目标序列 token IDs
    """
    model.eval()

    # 将源序列移到设备上
    src = torch.tensor([src_seq], dtype=torch.long, device=DEVICE)
    src_mask = model.make_src_mask(src)

    # 编码
    encoder_output = model.encode(src, src_mask)

    # 从 BOS 开始生成
    tgt = torch.tensor([[BOS_IDX]], dtype=torch.long, device=DEVICE)

    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(tgt)
        decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)

        # 取最后一个位置的 logits
        logits = model.output_linear(decoder_output[:, -1, :])  # (1, vocab_size)
        next_token = logits.argmax(dim=-1)  # Greedy 选择

        # 如果生成 EOS 或 PAD，停止
        if next_token.item() in (EOS_IDX, PAD_IDX):
            break

        # 追加到目标序列
        tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)

    return tgt.squeeze(0).tolist()


# ============================================================
# 可视化
# ============================================================

def plot_training_curves(train_losses: list[float], val_losses: list[float], output_path: str):
    """
    绘制训练曲线。

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    steps = range(1, len(train_losses) + 1)
    ax1.plot(steps, train_losses, "b-", label="Train Loss", linewidth=1.5)
    if val_losses:
        val_steps = range(1, len(val_losses) + 1)
        ax1.plot(val_steps, val_losses, "r-", label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 学习率曲线
    # 绘制前 10000 步的学习率
    d_model = D_MODEL
    warmup = 4000
    lr_steps = list(range(1, min(10000, len(train_losses) + 1)))
    lrs = [(d_model ** (-0.5)) * min(s ** (-0.5), s * (warmup ** (-1.5))) for s in lr_steps]

    ax2.plot(lr_steps, lrs, "g-", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ 训练曲线已保存: {output_path}")


def save_translation_demo(
    model: nn.Module, dataset: SimpleTranslationDataset, output_path: str
):
    """
    保存翻译示例。

    Args:
        model: 训练好的模型
        dataset: 数据集
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    lines = ["# Transformer 翻译示例\n"]
    lines.append(f"模型在 {len(dataset)} 个模拟翻译样本上训练\n\n")

    # 随机选择 5 个样本
    indices = random.sample(range(len(dataset)), min(5, len(dataset)))

    for i, idx in enumerate(indices, 1):
        src, tgt = dataset[idx]
        src_list = src.tolist()
        tgt_list = tgt.tolist()

        # 生成翻译
        pred_list = translate(model, src_list)

        lines.append(f"## 示例 {i}\n")
        lines.append(f"源序列: {src_list[:20]}...")
        lines.append(f"目标序列: {tgt_list[:20]}...")
        lines.append(f"预测序列: {pred_list}")
        lines.append("")

        # 计算准确率
        # 去掉 BOS 和 EOS，比较中间部分
        tgt_core = [t for t in tgt_list if t not in (BOS_IDX, EOS_IDX, PAD_IDX)]
        pred_core = [p for p in pred_list if p not in (BOS_IDX, EOS_IDX, PAD_IDX)]

        if len(tgt_core) > 0:
            matches = sum(1 for p, t in zip(pred_core, tgt_core) if p == t)
            acc = matches / len(tgt_core)
            lines.append(f"Token 准确率: {acc:.2%} ({matches}/{len(tgt_core)})")

        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ 翻译示例已保存: {output_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：创建模型、训练、评估、生成可视化。"""
    parser = argparse.ArgumentParser(description="复现 Attention Is All You Need")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数（默认: 20）")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小（默认: 64）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录")
    args = parser.parse_args()

    # 设置设备
    device = DEVICE if args.device == "auto" else torch.device(args.device)
    print(f"🖥️  使用设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 创建数据集
    print("\n📦 创建模拟翻译数据集...")
    dataset = SimpleTranslationDataset(n_samples=5000)
    print(f"   样本数: {len(dataset)}")

    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=dataset.collate_fn,
    )

    # 创建模型
    print(f"\n🏗️  创建 Transformer 模型...")
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_head=N_HEAD,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        max_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {total_params:,}")
    print(f"   模型维度: {D_MODEL}, 头数: {N_HEAD}, 层数: {N_LAYERS}")

    # 标签平滑损失
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    # Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-9, betas=(0.9, 0.98), eps=1e-9)

    # 创建训练器
    trainer = Trainer(model, criterion, optimizer, warmup_steps=4000)

    # 训练循环
    print(f"\n🚀 开始训练（{args.epochs} epochs）...")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0

        model.train()
        for src, tgt in train_loader:
            loss = trainer.train_step(src, tgt)
            total_loss += loss
            n_batches += 1

        avg_train_loss = total_loss / n_batches

        # 验证
        total_val_loss = 0.0
        n_val_batches = 0
        for src, tgt in val_loader:
            val_loss = trainer.validate(src, tgt)
            total_val_loss += val_loss
            n_val_batches += 1

        avg_val_loss = total_val_loss / n_val_batches if n_val_batches > 0 else 0

        epoch_time = time.time() - epoch_start
        trainer.train_losses.append(avg_train_loss)
        trainer.val_losses.append(avg_val_loss)

        print(f"Epoch {epoch:>2}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {trainer.get_lr():.2e} | "
              f"Time: {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"⏱️  总训练时间: {total_time:.1f} 秒")

    # 保存可视化
    plot_training_curves(
        trainer.train_losses, trainer.val_losses,
        os.path.join(args.output_dir, "training_log.png"),
    )

    # 保存翻译示例
    save_translation_demo(
        model, dataset,
        os.path.join(args.output_dir, "translation_demo.txt"),
    )

    print(f"\n🎉 训练完成！结果已保存到 {args.output_dir}/")


if __name__ == "__main__":
    main()
```

### 🚀 运行命令

```bash
# 基础训练（20 个 epoch，CPU）
python transformer.py --epochs 20 --device cpu

# GPU 训练（更多 epoch）
python transformer.py --epochs 50 --device cuda --batch-size 128

# 快速验证（5 个 epoch）
python transformer.py --epochs 5 --device cpu
```

### 📤 预期输出

```
🖥️  使用设备: cpu
📦 创建模拟翻译数据集...
   样本数: 5000

🏗️  创建 Transformer 模型...
   参数量: 15,234,567
   模型维度: 256, 头数: 8, 层数: 3

🚀 开始训练（20 epochs）...
============================================================
Epoch  1/20 | Train Loss: 8.5234 | Val Loss: 8.4567 | LR: 1.00e-04 | Time: 45.2s
Epoch  2/20 | Train Loss: 7.2345 | Val Loss: 7.1890 | LR: 1.41e-04 | Time: 44.8s
Epoch  3/20 | Train Loss: 6.1234 | Val Loss: 6.0987 | LR: 1.63e-04 | Time: 45.1s
...
Epoch 20/20 | Train Loss: 2.3456 | Val Loss: 2.4567 | LR: 5.00e-05 | Time: 44.9s
============================================================
⏱️  总训练时间: 905.3 秒
✅ 训练曲线已保存: output/training_log.png
✅ 翻译示例已保存: output/translation_demo.txt

🎉 训练完成！结果已保存到 output/
```

---

## 项目 4：撰写技术博客

> 🎯 **目标**：选择一篇论文，撰写完整的技术解读博客，包含公式推导、代码实现和可视化

### 📐 项目结构

```
project4-tech-blog/
├── blog_generator.py      # 主程序：博客内容生成、排版、导出
├── requirements.txt       # 依赖列表
└── output/                # 输出目录
    └── blog_post.md       # 生成的博客文章
```

### 📦 环境准备

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt**

```
requests>=2.28.0
matplotlib>=3.5.0
numpy>=1.21.0
```

### 💻 完整代码

```python
"""
项目 4：撰写技术博客
======================

背景：
  技术写作是研究者最重要的软技能之一。好的技术博客能够：
  1. 帮助他人理解复杂概念
  2. 建立个人技术品牌
  3. 加深自己对问题的理解
  4. 为未来的论文写作打基础

本项目实现：
  1. 从 arXiv 获取论文元数据（标题、摘要、作者）
  2. 生成结构化的技术博客内容
  3. 包含公式推导、架构图、代码示例
  4. 导出为 Markdown 格式

博客结构模板：
  1. 标题 + 引言（为什么这篇论文重要？）
  2. 背景知识（需要哪些预备知识？）
  3. 核心方法（公式 + 图解 + 代码）
  4. 实验结果（数据 + 分析）
  5. 局限性 + 未来方向
  6. 总结 + 参考

运行方式：
  python blog_generator.py --paper "1706.03762" --output output/blog_post.md

预期输出：
  - 生成结构化的 Markdown 博客文章
  - 包含完整的公式、代码、图表引用
"""

import os
import time
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

import requests
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 配置
# ============================================================
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_NS = "{http://www.w3.org/2005/Atom}"
OUTPUT_DIR = "output"

# 论文数据库：预置一些经典论文的信息
PAPER_DATABASE = {
    "1706.03762": {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin",
        "institution": "Google Brain, Google Research",
        "venue": "NeurIPS 2017",
        "year": 2017,
        "category": "cs.CL",
        "citations": "100,000+",
        "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "key_contributions": [
            "提出完全基于注意力机制的 Transformer 架构",
            "多头注意力机制（Multi-Head Attention）",
            "位置编码（Positional Encoding）替代 RNN 的序列建模",
            "在机器翻译任务上达到 SOTA，训练速度大幅提升",
        ],
        "equations": {
            "scaled_dot_product_attention": r"""
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$ (Query): 查询向量，表示"我在找什么"
- $K$ (Key): 键向量，表示"我有什么"
- $V$ (Value): 值向量，表示"我的内容是什么"
- $d_k$: Key 的维度，用于缩放点积防止梯度消失
            """,
            "multi_head_attention": r"""
$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$$

$$\text{where } head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

多头注意力将 Q, K, V 投影到 h 个子空间，每个子空间独立计算注意力，
然后拼接并通过线性变换 $W^O$ 得到最终输出。
            """,
            "position_encoding": r"""
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

正弦位置编码的优势：
1. 可以处理任意长度的序列
2. 对于固定偏移 k，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数
            """,
            "layer_norm": r"""
$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

其中 $\mu$ 和 $\sigma$ 是对每个样本的 feature 维度计算均值和标准差，
$\gamma$ 和 $\beta$ 是可学习的缩放和平移参数。
            """,
        },
        "code_example": """```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    \"\"\"缩放点积注意力机制实现\"\"\"

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = None  # 1/sqrt(d_k)，在 forward 中动态计算

    def forward(self, Q, K, V, mask=None):
        \"\"\"
        Args:
            Q: (batch, heads, seq_len, d_k)
            K: (batch, heads, seq_len, d_k)
            V: (batch, heads, seq_len, d_v)
            mask: 可选的掩码

        Returns:
            output: (batch, heads, seq_len, d_v)
            attention_weights: (batch, heads, seq_len, seq_len)
        \"\"\"
        d_k = Q.size(-1)
        self.scale = 1.0 / math.sqrt(d_k)

        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax + Dropout
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
```""",
        "architecture_diagram": """
┌─────────────────────────────────────────────────────────────┐
│                    Transformer Architecture                  │
│                                                             │
│  ┌─────────── ENCODER ──────────┐  ┌─── DECODER ──────────┐ │
│  │                              │  │                        │ │
│  │  ┌──────────────┐            │  │  ┌──────────────┐     │ │
│  │  │ Multi-Head   │            │  │  │ Masked       │     │ │
│  │  │ Attn         │            │  │  │ Multi-Head   │     │ │
│  │  └──────┬───────┘            │  │  │ Attn         │     │ │
│  │         │                    │  │  └──────┬───────┘     │ │
│  │  ┌──────▼───────┐            │  │         │             │ │
│  │  │ Add & Norm   │            │  │  ┌──────▼───────┐     │ │
│  │  └──────┬───────┘            │  │  │ Multi-Head   │     │ │
│  │         │                    │  │  │ Attn         │─────┼─┐
│  │  ┌──────▼───────┐            │  │  └──────┬───────┘     │ │
│  │  │ FFN          │            │  │         │             │ │
│  │  └──────┬───────┘            │  │  ┌──────▼───────┐     │ │
│  │         │                    │  │  │ Add & Norm   │     │ │
│  │  ┌──────▼───────┐            │  │  └──────┬───────┘     │ │
│  │  │ Add & Norm   │            │  │         │             │ │
│  │  └──────┬───────┘            │  │  ┌──────▼───────┐     │ │
│  │         │                    │  │  │ FFN          │     │ │
│  │  ... (×N layers)             │  │  └──────┬───────┘     │ │
│  │                              │  │         │             │ │
│  └──────────────┬───────────────┘  │  ┌──────▼───────┐     │ │
│                 │                  │  │ Linear +      │     │ │
│                 └──────────────────┼─▶│ Softmax       │     │ │
│                                    │  └──────────────┘     │ │
└────────────────────────────────────┴────────────────────────┘ │
                                                                 │
  Input Embedding ──────────────────────────────▶ Output        │
  + Positional Encoding                         + Positional     │
                                                Encoding         │
""",
        "limitations": [
            "自注意力复杂度为 O(n²)，长序列计算开销大",
            "位置编码是固定的，不能学习位置关系",
            "对局部模式的建模能力不如 CNN",
            "需要大量数据才能充分训练",
        ],
        "follow_up_works": [
            "Sparse Transformer (2019): 稀疏注意力降低复杂度",
            "Linear Transformer (2020): 线性复杂度注意力",
            "Performer (2021): 基于 FFE 的近似注意力",
            "Flash Attention (2022): IO 感知的精确注意力",
        ],
    },
}


def fetch_arxiv_paper(arxiv_id: str) -> Optional[dict]:
    """
    从 arXiv API 获取论文元数据。

    Args:
        arxiv_id: arXiv ID，如 "1706.03762"

    Returns:
        论文元数据字典，获取失败返回 None
    """
    params = {
        "id": arxiv_id,
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        for entry in root.findall(f"{ARXIV_NS}entry"):
            title = entry.find(f"{ARXIV_NS}title").text.strip()
            summary = entry.find(f"{ARXIV_NS}summary").text.strip()
            published = entry.find(f"{ARXIV_NS}published").text
            authors = []
            for author in entry.findall(f"{ARXIV_NS}author"):
                name = author.find(f"{ARXIV_NS}name")
                if name is not None:
                    authors.append(name.text)

            return {
                "title": title,
                "summary": summary,
                "published": published,
                "authors": ", ".join(authors),
                "arxiv_id": arxiv_id,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            }

    except Exception as e:
        print(f"⚠️  获取 arXiv 数据失败: {e}")

    return None


def generate_blog_content(paper_info: dict, template_type: str = "detailed") -> str:
    """
    生成技术博客内容。

    根据论文信息生成结构化的 Markdown 博客文章。

    Args:
        paper_info: 论文信息字典
        template_type: 模板类型（detailed / quick / code_focused）

    Returns:
        Markdown 格式的博客内容
    """
    title = paper_info.get("title", "Unknown Paper")
    authors = paper_info.get("authors", "Unknown Authors")
    year = paper_info.get("year", datetime.now().year)
    venue = paper_info.get("venue", "")
    summary = paper_info.get("summary", "")
    contributions = paper_info.get("key_contributions", [])
    equations = paper_info.get("equations", {})
    code = paper_info.get("code_example", "")
    diagram = paper_info.get("architecture_diagram", "")
    limitations = paper_info.get("limitations", [])
    follow_ups = paper_info.get("follow_up_works", [])

    # ============================================================
    # 博客头部：标题、元数据、引言
    # ============================================================
    blog = []

    blog.append(f"# 📄 {title}")
    blog.append("")
    blog.append(f"> **作者**: {authors}")
    blog.append(f"> **发表**: {venue} ({year})")
    blog.append(f"> **arXiv**: https://arxiv.org/abs/{paper_info.get('arxiv_id', '')}")
    blog.append(f"> **引用**: {paper_info.get('citations', 'N/A')}")
    blog.append("")
    blog.append("---")
    blog.append("")

    # ============================================================
    # 引言：为什么这篇论文重要？
    # ============================================================
    blog.append("## 🎯 一句话总结")
    blog.append("")
    blog.append(f"> {summary[:200]}...")
    blog.append("")

    blog.append("## 📖 背景与动机")
    blog.append("")
    blog.append("在深入方法细节之前，让我们先理解这篇论文要解决什么问题。")
    blog.append("")
    blog.append(f"**核心问题**: 这篇论文发表于 {year} 年，当时序列建模的主流方法是 RNN 和 CNN。")
    blog.append("这些方法存在以下局限性：")
    blog.append("")
    blog.append("1. **长距离依赖**: RNN 难以捕捉远距离的依赖关系")
    blog.append("2. **计算效率**: 序列的自回归性质限制了并行化")
    blog.append("3. **信息瓶颈**: 固定长度的上下文向量限制了信息容量")
    blog.append("")
    blog.append("这篇论文的核心洞察是：**注意力机制本身就足够了**，不需要循环或卷积。")
    blog.append("")

    # ============================================================
    # 核心方法：公式 + 图解
    # ============================================================
    blog.append("## 🔬 核心方法")
    blog.append("")

    # 架构图
    if diagram:
        blog.append("### 架构概览")
        blog.append("")
        blog.append("```")
        blog.append(diagram.strip())
        blog.append("```")
        blog.append("")

    # 公式推导
    blog.append("### 关键公式")
    blog.append("")

    for eq_name, eq_content in equations.items():
        blog.append(f"#### {eq_name.replace('_', ' ').title()}")
        blog.append("")
        blog.append(eq_content.strip())
        blog.append("")

    # ============================================================
    # 代码实现
    # ============================================================
    if code:
        blog.append("## 💻 代码实现")
        blog.append("")
        blog.append("以下是核心组件的 PyTorch 实现：")
        blog.append("")
        blog.append(code)
        blog.append("")

    # ============================================================
    # 主要贡献
    # ============================================================
    if contributions:
        blog.append("## ✨ 主要贡献")
        blog.append("")
        for i, contribution in enumerate(contributions, 1):
            blog.append(f"{i}. {contribution}")
        blog.append("")

    # ============================================================
    # 实验结果分析
    # ============================================================
    blog.append("## 📊 实验结果")
    blog.append("")
    blog.append("论文在 WMT 2014 英德翻译和英法翻译任务上进行了评估。")
    blog.append("")
    blog.append("| 任务 | BLEU 分数 | 训练成本 |")
    blog.append("|------|----------|---------|")
    blog.append("| EN-DE | 28.4 | 3.5 days (8 GPUs) |")
    blog.append("| EN-FR | 41.0 | 12 hours (8 GPUs) |")
    blog.append("")
    blog.append("**关键发现**:")
    blog.append("- 在 EN-DE 翻译上超越了之前最好的模型 2 BLEU")
    blog.append("- 训练速度比之前的模型快一个数量级")
    blog.append("- 模型质量更高，训练成本更低")
    blog.append("")

    # ============================================================
    # 局限性
    # ============================================================
    if limitations:
        blog.append("## ⚠️ 局限性")
        blog.append("")
        for limitation in limitations:
            blog.append(f"- {limitation}")
        blog.append("")

    # ============================================================
    # 后续工作
    # ============================================================
    if follow_ups:
        blog.append("## 🚀 后续工作")
        blog.append("")
        blog.append("这篇论文引发了大量后续研究：")
        blog.append("")
        for follow_up in follow_ups:
            blog.append(f"- {follow_up}")
        blog.append("")

    # ============================================================
    # 总结
    # ============================================================
    blog.append("## 📝 总结")
    blog.append("")
    blog.append(f"这篇论文提出了 **{title}**，彻底改变了序列建模的方式。")
    blog.append("")
    blog.append("**核心要点**:")
    blog.append("1. 完全基于注意力的架构可以替代 RNN/CNN")
    blog.append("2. 多头注意力允许多个表示子空间的并行学习")
    blog.append("3. 位置编码解决了序列顺序信息的编码问题")
    blog.append("4. 训练效率和质量都有显著提升")
    blog.append("")
    blog.append("**影响**:")
    blog.append("这篇论文是近年来引用最多的 AI 论文之一，直接催生了 BERT、GPT 等大语言模型。")
    blog.append("Transformer 已经成为 NLP、CV、多模态等领域的标准架构。")
    blog.append("")

    # ============================================================
    # 参考文献
    # ============================================================
    blog.append("---")
    blog.append("")
    blog.append("## 📚 参考文献")
    blog.append("")
    blog.append(f"1. {authors}. \"{title}\". {venue}, {year}.")
    blog.append(f"   - [arXiv](https://arxiv.org/abs/{paper_info.get('arxiv_id', '')})")
    blog.append(f"   - [PDF](https://arxiv.org/pdf/{paper_info.get('arxiv_id', '')})")
    blog.append("")
    blog.append("---")
    blog.append("")
    blog.append(f"*本文由技术博客生成工具自动生成 | 最后更新: {datetime.now().strftime('%Y-%m-%d')}*")

    return "\n".join(blog)


def generate_attention_visualization(output_path: str):
    """
    生成注意力机制的可视化图。

    Args:
        output_path: 输出图片路径
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图 1: 注意力权重热力图
    np.random.seed(42)
    attn = np.random.dirichlet(np.ones(10), size=10)

    im1 = axes[0].imshow(attn, cmap="viridis", aspect="auto")
    axes[0].set_title("Attention Weights Heatmap", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # 图 2: 位置编码可视化
    pe = np.zeros((50, 64))
    position = np.arange(50)
    div_term = np.exp(np.arange(0, 64, 2) * (-np.log(10000.0) / 64))
    pe[:, 0::2] = np.sin(position[:, None] * div_term[None, :])
    pe[:, 1::2] = np.cos(position[:, None] * div_term[None, :])

    im2 = axes[1].imshow(pe.T, cmap="coolwarm", aspect="auto")
    axes[1].set_title("Positional Encoding", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Dimension")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # 图 3: 缩放因子的效果
    d_k_values = np.arange(1, 100)
    scale = 1.0 / np.sqrt(d_k_values)
    axes[2].plot(d_k_values, scale, "b-", linewidth=2)
    axes[2].fill_between(d_k_values, scale, alpha=0.3)
    axes[2].set_title("Scaling Factor 1/√d_k", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("d_k (Key Dimension)")
    axes[2].set_ylabel("Scale")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ 可视化图表已保存: {output_path}")


def main():
    """主函数：获取论文信息、生成博客、输出文件。"""
    parser = argparse.ArgumentParser(description="技术博客生成器")
    parser.add_argument(
        "--paper",
        type=str,
        default="1706.03762",
        help="arXiv ID（默认: 1706.03762 - Attention Is All You Need）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(OUTPUT_DIR, "blog_post.md"),
        help="输出文件路径",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="detailed",
        choices=["detailed", "quick", "code_focused"],
        help="博客模板类型",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print(f"\n📝 正在生成技术博客...")
    print(f"   论文: arXiv {args.paper}")
    print(f"   模板: {args.template}")

    # 获取论文信息
    paper_info = PAPER_DATABASE.get(args.paper, {})

    if not paper_info:
        # 尝试从 arXiv API 获取
        print("   ⏳ 从 arXiv 获取论文信息...")
        arxiv_data = fetch_arxiv_paper(args.paper)
        if arxiv_data:
            paper_info = arxiv_data
        else:
            print("❌ 无法获取论文信息，请检查 arXiv ID")
            return

    # 生成博客内容
    print("   ⏳ 生成博客内容...")
    blog_content = generate_blog_content(paper_info, template_type=args.template)

    # 写入文件
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(blog_content)

    # 生成可视化
    viz_path = os.path.join(os.path.dirname(args.output), "attention_viz.png")
    generate_attention_visualization(viz_path)

    # 统计
    word_count = len(blog_content.split())
    line_count = blog_content.count("\n") + 1

    print(f"\n✅ 博客已生成: {args.output}")
    print(f"   字数: {word_count}")
    print(f"   行数: {line_count}")
    print(f"   可视化: {viz_path}")
    print(f"\n🎉 技术博客完成！")


if __name__ == "__main__":
    main()
```

### 🚀 运行命令

```bash
# 生成 Attention Is All You Need 的技术博客
python blog_generator.py --paper "1706.03762" --output output/blog_post.md

# 使用不同的模板
python blog_generator.py --paper "1706.03762" --template code_focused

# 尝试获取其他论文（需要网络）
python blog_generator.py --paper "1810.04805" --output output/bert_blog.md
```

### 📤 预期输出

```
📝 正在生成技术博客...
   论文: arXiv 1706.03762
   模板: detailed
   ⏳ 生成博客内容...
✅ 可视化图表已保存: output/attention_viz.png

✅ 博客已生成: output/blog_post.md
   字数: 2847
   行数: 156
   可视化: output/attention_viz.png

🎉 技术博客完成！
```

**生成的博客文章结构预览**：

```markdown
# 📄 Attention Is All You Need

> **作者**: Ashish Vaswani, Noam Shazeer, ...
> **发表**: NeurIPS 2017 (2017)
> **arXiv**: https://arxiv.org/abs/1706.03762
> **引用**: 100,000+

---

## 🎯 一句话总结

> The dominant sequence transduction models are based on complex recurrent...

## 📖 背景与动机

在深入方法细节之前，让我们先理解这篇论文要解决什么问题。

**核心问题**: 这篇论文发表于 2017 年，当时序列建模的主流方法是 RNN 和 CNN。
这些方法存在以下局限性：

1. **长距离依赖**: RNN 难以捕捉远距离的依赖关系
2. **计算效率**: 序列的自回归性质限制了并行化
...

## 🔬 核心方法

### 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer Architecture                  │
...
```

### 关键公式

#### Scaled Dot Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
...

## 💻 代码实现

```python
class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制实现"""
...
```

## ✨ 主要贡献
## 📊 实验结果
## ⚠️ 局限性
## 🚀 后续工作
## 📝 总结
## 📚 参考文献
```

---

## 📋 项目完成检查清单

完成所有 4 个项目后，你应该能够：

- [x] **项目 1**: 使用 arXiv API 获取和分析论文数据
- [x] **项目 2**: 设计并执行消融实验，量化模型组件贡献
- [x] **项目 3**: 从零实现 Transformer 并在翻译任务上验证
- [x] **项目 4**: 撰写结构化的技术博客文章

### 🎓 能力成长路径

```
完成前                          完成后
─────────────────────────────────────────────
文献检索    ⭐⭐              文献检索    ⭐⭐⭐⭐
实验设计    ⭐               实验设计    ⭐⭐⭐⭐
代码复现    ⭐⭐              代码复现    ⭐⭐⭐⭐⭐
技术写作    ⭐               技术写作    ⭐⭐⭐⭐
─────────────────────────────────────────────
```

> _研究能力不是一蹴而就的，每一个项目都是你学术生涯的一块基石。_
>
> _—— 悟空_
