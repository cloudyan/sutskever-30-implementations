# 第 8 章实战项目：大模型系统优化

> 🏗️ 动手实践 —— 让大模型跑得更快、更省、更稳
> 
> _难度：⭐⭐⭐⭐⭐ | 预计耗时：4-6 周_

---

## 📋 项目清单

| # | 项目名称 | 难度 | 预计时间 | 核心技能 |
|---|---------|------|---------|---------|
| 1 | LLM INT4 量化部署 | ⭐⭐⭐ | 3-5 天 | AutoGPTQ、bitsandbytes、精度评估 |
| 2 | vLLM 推理服务搭建 | ⭐⭐⭐⭐ | 3-5 天 | vLLM、吞吐量对比、服务部署 |
| 3 | DeepSpeed ZeRO-3 多卡训练 | ⭐⭐⭐⭐⭐ | 5-7 天 | DeepSpeed、分布式训练、ZeRO |
| 4 | 模型蒸馏实战 | ⭐⭐⭐⭐⭐ | 7-10 天 | 知识蒸馏、教师-学生模型 |

---

## 📦 环境准备

```bash
# 基础依赖
pip install torch transformers accelerate bitsandbytes auto-gptq vllm deepspeed

# 评估依赖
pip install evaluate rouge-score nltk perplexity

# 监控依赖（项目 2）
pip install psutil pynvml

# DeepSpeed 依赖（项目 3）
pip install deepspeed
python -m deepspeed.check_env  # 检查环境
```

---

## 项目 1：LLM INT4 量化部署

### 🎯 项目目标

使用 bitsandbytes 和 AutoGPTQ 对 LLaMA-2-7B 进行 INT4 量化，对比量化前后的：
- 模型大小
- 推理速度（token/s）
- 生成质量（困惑度 PPL）
- 显存占用

### 📁 项目结构

```
project-1-quantization/
├── requirements.txt
├── quantize_bnb.py        # bitsandbytes 量化
├── quantize_autogptq.py   # AutoGPTQ 量化
├── benchmark.py           # 性能对比测试
└── README.md
```

### 代码 1：bitsandbytes INT4 量化

```python
"""
项目 1 - 代码 1：使用 bitsandbytes 进行 INT4 量化

功能：
1. 加载原始 FP16 模型
2. 使用 bitsandbytes 的 4bit 量化加载
3. 对比量化前后的显存占用和推理速度
4. 生成文本并对比质量

依赖：
    pip install torch transformers bitsandbytes accelerate

运行：
    python quantize_bnb.py
"""

import torch
import time
import gc
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)


def clear_memory():
    """清理 GPU 显存"""
    gc.collect()
    torch.cuda.empty_cache()


def get_gpu_memory():
    """获取当前 GPU 显存使用量（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def benchmark_inference(model, tokenizer, prompts, max_new_tokens=100):
    """
    推理性能基准测试
    
    Args:
        model: 量化后的模型
        tokenizer: 分词器
        prompts: 测试提示列表
        max_new_tokens: 最大生成 token 数
    
    Returns:
        dict: 包含平均延迟、吞吐量等指标
    """
    total_tokens = 0
    total_time = 0
    
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 预热（第一次推理通常较慢）
        if total_time == 0:
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            torch.cuda.synchronize()
        
        # 正式计时
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        # 计算生成的 token 数
        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += generated_tokens
        total_time += elapsed
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "output": generated_text,
            "tokens": generated_tokens,
            "time": elapsed,
            "tokens_per_sec": generated_tokens / elapsed
        })
    
    avg_latency = total_time / len(prompts)
    throughput = total_tokens / total_time
    
    return {
        "avg_latency": avg_latency,
        "throughput": throughput,  # tokens per second
        "total_tokens": total_tokens,
        "total_time": total_time,
        "results": results
    }


def main():
    """主函数：对比 FP16 和 INT4 量化模型"""
    
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    # 测试提示
    test_prompts = [
        "请解释什么是机器学习，用简单易懂的方式。",
        "Python 中列表和元组有什么区别？",
        "写一首关于春天的四行诗。",
    ]
    
    print("=" * 60)
    print("📊 LLM INT4 量化对比测试")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"测试提示数: {len(test_prompts)}")
    print()
    
    # ========================================
    # 第一部分：FP16 原始模型基准
    # ========================================
    print("\n" + "=" * 60)
    print("🔵 测试 FP16 原始模型...")
    print("=" * 60)
    
    clear_memory()
    fp16_memory_before = get_gpu_memory()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    fp16_memory_after = get_gpu_memory()
    fp16_memory_used = fp16_memory_after - fp16_memory_before
    
    print(f"FP16 模型显存占用: {fp16_memory_used:.0f} MB ({fp16_memory_used/1024:.2f} GB)")
    
    # FP16 推理测试
    fp16_results = benchmark_inference(model_fp16, tokenizer, test_prompts)
    
    print(f"\nFP16 推理结果:")
    print(f"  平均延迟: {fp16_results['avg_latency']:.2f} 秒")
    print(f"  吞吐量: {fp16_results['throughput']:.2f} tokens/s")
    
    # 打印生成结果
    for i, r in enumerate(fp16_results["results"]):
        print(f"\n  提示 {i+1}: {r['prompt']}")
        print(f"  输出: {r['output'][-200:]}")  # 只显示最后 200 字符
        print(f"  速度: {r['tokens_per_sec']:.2f} tokens/s")
    
    # 清理 FP16 模型
    del model_fp16
    clear_memory()
    
    # ========================================
    # 第二部分：INT4 量化模型
    # ========================================
    print("\n" + "=" * 60)
    print("🟢 测试 INT4 量化模型（bitsandbytes）...")
    print("=" * 60)
    
    # 配置 4bit 量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,              # 启用 4bit 量化
        bnb_4bit_quant_type="nf4",      # NF4 量化类型（正态浮点 4bit）
        bnb_4bit_compute_dtype=torch.float16,  # 计算精度
        bnb_4bit_use_double_quant=True, # 双重量化（进一步压缩）
    )
    
    int4_memory_before = get_gpu_memory()
    
    model_int4 = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    int4_memory_after = get_gpu_memory()
    int4_memory_used = int4_memory_after - int4_memory_before
    
    print(f"INT4 模型显存占用: {int4_memory_used:.0f} MB ({int4_memory_used/1024:.2f} GB)")
    
    # INT4 推理测试
    int4_results = benchmark_inference(model_int4, tokenizer, test_prompts)
    
    print(f"\nINT4 推理结果:")
    print(f"  平均延迟: {int4_results['avg_latency']:.2f} 秒")
    print(f"  吞吐量: {int4_results['throughput']:.2f} tokens/s")
    
    for i, r in enumerate(int4_results["results"]):
        print(f"\n  提示 {i+1}: {r['prompt']}")
        print(f"  输出: {r['output'][-200:]}")
        print(f"  速度: {r['tokens_per_sec']:.2f} tokens/s")
    
    del model_int4
    clear_memory()
    
    # ========================================
    # 第三部分：对比总结
    # ========================================
    print("\n" + "=" * 60)
    print("📊 量化对比总结")
    print("=" * 60)
    
    memory_saving = (1 - int4_memory_used / fp16_memory_used) * 100
    speed_ratio = int4_results["throughput"] / fp16_results["throughput"]
    
    print(f"""
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│ 指标             │ FP16         │ INT4 (NF4)   │ 变化         │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 显存占用         │ {fp16_memory_used/1024:8.2f} GB │ {int4_memory_used/1024:8.2f} GB │ {memory_saving:+.1f}%      │
│ 平均延迟         │ {fp16_results['avg_latency']:8.2f} s   │ {int4_results['avg_latency']:8.2f} s   │                │
│ 吞吐量           │ {fp16_results['throughput']:8.2f} t/s  │ {int4_results['throughput']:8.2f} t/s  │ {speed_ratio:+.2f}x      │
└──────────────────┴──────────────┴──────────────┴──────────────┘
    """)
    
    print("💡 结论：")
    print(f"  - INT4 量化节省了 {memory_saving:.1f}% 的显存")
    print(f"  - 吞吐量变化为 {speed_ratio:.2f}x")
    print(f"  - 量化后模型可在更小显存的 GPU 上运行")
    print(f"  - 对于 LLaMA-2-7B：FP16 需要 ~14GB，INT4 仅需 ~4GB")


if __name__ == "__main__":
    main()
```

### 代码 2：AutoGPTQ 量化 + 精度评估

```python
"""
项目 1 - 代码 2：使用 AutoGPTQ 进行 INT4 量化 + 精度评估

功能：
1. 使用 AutoGPTQ 对模型进行 GPTQ 量化
2. 计算量化前后的困惑度（PPL）
3. 对比生成质量

依赖：
    pip install auto-gptq optimum

运行：
    python quantize_autogptq.py

注意：GPTQ 量化需要较长时间（约 2-4 小时），建议使用 GPU
"""

import torch
import json
import time
import math
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# ========================================
# 校准数据集（用于 GPTQ 量化）
# ========================================

def get_calibration_dataset(tokenizer, dataset_size=128):
    """
    获取校准数据集
    
    GPTQ 需要一小部分数据来校准量化参数。
    这里使用简单的文本生成提示作为校准数据。
    
    Args:
        tokenizer: 分词器
        dataset_size: 校准样本数量
    
    Returns:
        list: 校准文本列表
    """
    calibration_texts = [
        "机器学习是人工智能的一个重要分支，它通过算法让计算机从数据中学习规律。",
        "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的层次化表示。",
        "自然语言处理（NLP）是人工智能的一个关键领域，致力于让计算机理解和生成人类语言。",
        "Transformer 架构通过自注意力机制实现了高效的序列建模，成为现代大模型的基础。",
        "大语言模型通过在大规模文本数据上预训练，学习到了丰富的语言知识和推理能力。",
        "强化学习通过与环境的交互来学习最优策略，在棋类游戏和机器人控制中取得了巨大成功。",
        "计算机视觉利用深度学习技术实现了图像分类、目标检测和语义分割等任务。",
        "生成对抗网络（GAN）通过生成器和判别器的对抗训练来生成逼真的数据。",
        "扩散模型是近年来最成功的生成模型之一，在图像生成领域表现出色。",
        "模型压缩技术包括量化、剪枝和知识蒸馏，可以有效减小模型大小并加速推理。",
    ]
    
    # 扩展数据集
    while len(calibration_texts) < dataset_size:
        calibration_texts.extend(calibration_texts[:dataset_size - len(calibration_texts)])
    
    # Tokenize
    tokenized_texts = []
    for text in calibration_texts[:dataset_size]:
        tokens = tokenizer(text, return_tensors="pt")
        tokenized_texts.append(tokens)
    
    return tokenized_texts


def calculate_perplexity(model, tokenizer, text, max_length=512):
    """
    计算文本的困惑度（Perplexity, PPL）
    
    PPL 是衡量语言模型质量的核心指标：
    - PPL 越低，模型对文本的预测越准确
    - PPL = exp(-1/N * Σ log P(w_i | w_<i))
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 待评估文本
        max_length: 最大处理长度
    
    Returns:
        float: 困惑度值
    """
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)
    
    # 计算 loss
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        neg_log_likelihood = outputs.loss
    
    # PPL = exp(loss)
    perplexity = math.exp(neg_log_likelihood.item())
    
    return perplexity


def evaluate_generation(model, tokenizer, prompts, max_new_tokens=100):
    """
    评估模型的生成质量
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        prompts: 测试提示列表
        max_new_tokens: 最大生成 token 数
    
    Returns:
        list: 生成结果列表
    """
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        elapsed = time.time() - start_time
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            "prompt": prompt,
            "generated": generated_text,
            "time": elapsed
        })
    
    return results


def main():
    """主函数：GPTQ 量化 + 评估"""
    
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    print("=" * 60)
    print("📊 AutoGPTQ INT4 量化 + 精度评估")
    print("=" * 60)
    
    # 加载分词器
    print("\n📥 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 评估文本（用于计算 PPL）
    evaluation_texts = [
        "人工智能正在改变我们的世界。从智能手机到自动驾驶汽车，"
        "AI 技术已经渗透到我们生活的方方面面。自然语言处理技术让计算机"
        "能够理解和生成人类语言，计算机视觉技术让计算机能够'看'懂图像。"
        "这些技术的进步离不开深度学习的发展，特别是 Transformer 架构的提出，"
        "为大语言模型的诞生奠定了基础。",
        
        "量子计算是一种利用量子力学原理进行计算的技术。"
        "与经典计算机使用比特（0 或 1）不同，量子计算机使用量子比特（qubit），"
        "可以同时处于 0 和 1 的叠加态。这使得量子计算机在某些特定问题上"
        "具有指数级的加速能力，如大整数分解、量子化学模拟等。"
    ]
    
    # ========================================
    # 第一部分：FP16 基准评估
    # ========================================
    print("\n🔵 评估 FP16 原始模型...")
    
    model_fp16 = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 计算 PPL
    fp16_ppls = []
    for i, text in enumerate(evaluation_texts):
        ppl = calculate_perplexity(model_fp16, tokenizer, text)
        fp16_ppls.append(ppl)
        print(f"  文本 {i+1} PPL: {ppl:.2f}")
    
    avg_fp16_ppl = sum(fp16_ppls) / len(fp16_ppls)
    print(f"\n  FP16 平均 PPL: {avg_fp16_ppl:.2f}")
    
    # 生成质量评估
    test_prompts = [
        "请解释什么是深度学习：",
        "机器学习的主要应用包括：",
    ]
    
    fp16_gen = evaluate_generation(model_fp16, tokenizer, test_prompts)
    print("\n  FP16 生成结果:")
    for r in fp16_gen:
        print(f"    提示: {r['prompt']}")
        print(f"    输出: {r['generated'][-150:]}")
    
    del model_fp16
    torch.cuda.empty_cache()
    
    # ========================================
    # 第二部分：GPTQ INT4 量化
    # ========================================
    print("\n" + "=" * 60)
    print("🟢 开始 GPTQ INT4 量化...")
    print("=" * 60)
    
    # 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=4,                   # 4bit 量化
        group_size=128,           # 量化分组大小
        desc_act=False,           # 不使用激活重排序
        damp_percent=0.01,        # 阻尼系数
    )
    
    # 获取校准数据
    print("📊 准备校准数据...")
    calibration_data = get_calibration_dataset(tokenizer, dataset_size=128)
    
    # 加载原始模型并量化
    print("⏳ 加载模型并量化（这可能需要 2-4 小时）...")
    start_time = time.time()
    
    model_gptq = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 执行量化
    model_gptq.quantize(calibration_data)
    
    quantize_time = time.time() - start_time
    print(f"✅ 量化完成！耗时: {quantize_time/60:.1f} 分钟")
    
    # 保存量化模型
    output_dir = "llama-2-7b-chat-gptq-int4"
    model_gptq.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"💾 量化模型已保存到: {output_dir}")
    
    # ========================================
    # 第三部分：量化模型评估
    # ========================================
    print("\n" + "=" * 60)
    print("📊 量化模型评估")
    print("=" * 60)
    
    # 计算 PPL
    gptq_ppls = []
    for i, text in enumerate(evaluation_texts):
        ppl = calculate_perplexity(model_gptq, tokenizer, text)
        gptq_ppls.append(ppl)
        print(f"  文本 {i+1} PPL: {ppl:.2f}")
    
    avg_gptq_ppl = sum(gptq_ppls) / len(gptq_ppls)
    print(f"\n  GPTQ INT4 平均 PPL: {avg_gptq_ppl:.2f}")
    
    ppl_increase = ((avg_gptq_ppl - avg_fp16_ppl) / avg_fp16_ppl) * 100
    print(f"  PPL 变化: {ppl_increase:+.2f}%")
    
    # 生成质量评估
    gptq_gen = evaluate_generation(model_gptq, tokenizer, test_prompts)
    print("\n  GPTQ INT4 生成结果:")
    for r in gptq_gen:
        print(f"    提示: {r['prompt']}")
        print(f"    输出: {r['generated'][-150:]}")
    
    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 60)
    print("📊 量化评估总结")
    print("=" * 60)
    
    print(f"""
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│ 指标             │ FP16         │ GPTQ INT4    │ 变化         │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 平均 PPL         │ {avg_fp16_ppl:12.2f} │ {avg_gptq_ppl:12.2f} │ {ppl_increase:+.2f}%      │
│ 量化时间         │ -            │ {quantize_time/60:10.1f} min │                │
│ 模型大小         │ ~14 GB       │ ~3.5 GB      │ -75%         │
└──────────────────┴──────────────┴──────────────┴──────────────┘
    """)
    
    print("💡 结论：")
    print(f"  - GPTQ INT4 量化将模型大小从 ~14GB 减小到 ~3.5GB")
    print(f"  - PPL 变化 {ppl_increase:+.2f}%（越小越好，通常 < 5% 可接受）")
    print(f"  - 量化后的模型在单张消费级 GPU 上即可运行")


if __name__ == "__main__":
    main()
```

---

## 项目 2：vLLM 推理服务搭建

### 🎯 项目目标

1. 使用 vLLM 搭建推理服务
2. 对比 vLLM 与原生 HuggingFace 的吞吐量差异
3. 实现 OpenAI 兼容的 API 服务

### 代码：vLLM 吞吐量对比 + API 服务

```python
"""
项目 2：vLLM 推理服务搭建 + 吞吐量对比

功能：
1. 对比 vLLM 与 HuggingFace 原生推理的吞吐量
2. 启动 OpenAI 兼容的 API 服务
3. 测试不同并发数下的性能表现

依赖：
    pip install vllm openai requests

运行对比测试：
    python project2_vllm_benchmark.py

启动 API 服务：
    python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf
"""

import time
import asyncio
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ========================================
# 第一部分：吞吐量对比测试
# ========================================

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    framework: str
    total_requests: int
    successful_requests: int
    total_tokens: int
    total_time: float
    avg_latency: float
    p50_latency: float
    p90_latency: float
    p99_latency: float
    throughput: float  # requests per second
    token_throughput: float  # tokens per second
    
    def print_summary(self):
        """打印结果摘要"""
        print(f"\n{'='*60}")
        print(f"📊 {self.framework} 基准测试结果")
        print(f"{'='*60}")
        print(f"""
┌──────────────────┬──────────────┐
│ 指标             │ 值           │
├──────────────────┼──────────────┤
│ 总请求数         │ {self.total_requests:12d} │
│ 成功请求数       │ {self.successful_requests:12d} │
│ 总生成 Token     │ {self.total_tokens:12d} │
│ 总耗时           │ {self.total_time:10.2f} s  │
│ 平均延迟         │ {self.avg_latency:10.2f} s  │
│ P50 延迟         │ {self.p50_latency:10.2f} s  │
│ P90 延迟         │ {self.p90_latency:10.2f} s  │
│ P99 延迟         │ {self.p99_latency:10.2f} s  │
│ 吞吐量 (RPS)     │ {self.throughput:10.2f} req/s│
│ Token 吞吐量     │ {self.token_throughput:10.2f} t/s │
└──────────────────┴──────────────┘
        """)


def benchmark_huggingface(model_name, prompts, max_tokens=100):
    """
    使用 HuggingFace 原生 API 进行基准测试
    
    Args:
        model_name: 模型名称
        prompts: 测试提示列表
        max_tokens: 最大生成 token 数
    
    Returns:
        BenchmarkResult: 测试结果
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n🔵 开始 HuggingFace 基准测试...")
    
    # 加载模型
    print("  📥 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    latencies = []
    total_tokens = 0
    successful = 0
    
    start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        req_start = time.time()
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False
                )
            
            req_elapsed = time.time() - req_start
            generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            
            latencies.append(req_elapsed)
            total_tokens += generated_tokens
            successful += 1
            
            if (i + 1) % 10 == 0:
                print(f"  已处理: {i+1}/{len(prompts)}")
        
        except Exception as e:
            print(f"  ❌ 请求 {i+1} 失败: {e}")
    
    total_time = time.time() - start_time
    
    latencies.sort()
    n = len(latencies)
    
    result = BenchmarkResult(
        framework="HuggingFace 原生",
        total_requests=len(prompts),
        successful_requests=successful,
        total_tokens=total_tokens,
        total_time=total_time,
        avg_latency=sum(latencies) / n if n > 0 else 0,
        p50_latency=latencies[int(n * 0.5)] if n > 0 else 0,
        p90_latency=latencies[int(n * 0.9)] if n > 0 else 0,
        p99_latency=latencies[int(n * 0.99)] if n > 0 else 0,
        throughput=successful / total_time if total_time > 0 else 0,
        token_throughput=total_tokens / total_time if total_time > 0 else 0
    )
    
    result.print_summary()
    return result


def benchmark_vllm(model_name, prompts, max_tokens=100):
    """
    使用 vLLM 进行基准测试
    
    Args:
        model_name: 模型名称
        prompts: 测试提示列表
        max_tokens: 最大生成 token 数
    
    Returns:
        BenchmarkResult: 测试结果
    """
    from vllm import LLM, SamplingParams
    
    print(f"\n🟢 开始 vLLM 基准测试...")
    
    # 创建 vLLM 实例
    print("  📥 加载模型...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens
    )
    
    # vLLM 支持批量推理（核心优势）
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    total_time = time.time() - start_time
    
    # 计算统计信息
    latencies = []
    total_tokens = 0
    
    for output in outputs:
        generated_tokens = len(output.outputs[0].token_ids)
        total_tokens += generated_tokens
        # vLLM 批量处理，无法精确到单个请求的延迟
        # 这里用总时间/请求数作为近似
    
    n = len(outputs)
    avg_latency = total_time / n if n > 0 else 0
    
    result = BenchmarkResult(
        framework="vLLM",
        total_requests=len(prompts),
        successful_requests=n,
        total_tokens=total_tokens,
        total_time=total_time,
        avg_latency=avg_latency,
        p50_latency=avg_latency,  # 批量处理，近似
        p90_latency=avg_latency,
        p99_latency=avg_latency,
        throughput=n / total_time if total_time > 0 else 0,
        token_throughput=total_tokens / total_time if total_time > 0 else 0
    )
    
    result.print_summary()
    return result


def run_comparison(model_name="meta-llama/Llama-2-7b-chat-hf", num_prompts=50):
    """
    运行完整的对比测试
    
    Args:
        model_name: 模型名称
        num_prompts: 测试提示数量
    """
    import torch
    
    # 生成测试提示
    base_prompts = [
        "请解释什么是机器学习：",
        "Python 中装饰器的作用是什么：",
        "请写一个快速排序的实现：",
        "什么是深度学习中的过拟合：",
        "解释一下 Transformer 的注意力机制：",
        "如何评估一个分类模型的性能：",
        "请解释反向传播算法：",
        "什么是迁移学习：",
        "解释一下梯度下降的原理：",
        "如何防止神经网络过拟合：",
    ]
    
    # 扩展提示列表
    prompts = []
    for i in range(num_prompts):
        prompts.append(base_prompts[i % len(base_prompts)])
    
    print("=" * 60)
    print("📊 vLLM vs HuggingFace 吞吐量对比")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"提示数量: {num_prompts}")
    
    # 分别测试
    hf_result = benchmark_huggingface(model_name, prompts)
    
    # 清理
    del hf_result
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    time.sleep(5)  # 等待显存释放
    
    vllm_result = benchmark_vllm(model_name, prompts)
    
    # 对比总结
    print("\n" + "=" * 60)
    print("📊 对比总结")
    print("=" * 60)
    
    throughput_ratio = vllm_result.throughput / hf_result.throughput if hf_result.throughput > 0 else 0
    token_ratio = vllm_result.token_throughput / hf_result.token_throughput if hf_result.token_throughput > 0 else 0
    
    print(f"""
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│ 指标             │ HuggingFace  │ vLLM         │ 加速比       │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 吞吐量 (RPS)     │ {hf_result.throughput:12.2f} │ {vllm_result.throughput:12.2f} │ {throughput_ratio:8.2f}x  │
│ Token 吞吐量     │ {hf_result.token_throughput:12.2f} │ {vllm_result.token_throughput:12.2f} │ {token_ratio:8.2f}x  │
│ 平均延迟         │ {hf_result.avg_latency:12.2f} │ {vllm_result.avg_latency:12.2f} │                │
└──────────────────┴──────────────┴──────────────┴──────────────┘
    """)
    
    print("💡 结论：")
    print(f"  - vLLM 吞吐量是 HuggingFace 的 {throughput_ratio:.2f}x")
    print(f"  - vLLM 的 PagedAttention 和连续批处理带来了显著提升")
    print(f"  - 并发越高，vLLM 的优势越明显")


# ========================================
# 第二部分：OpenAI 兼容 API 服务
# ========================================

def start_vllm_api_server(model_name="meta-llama/Llama-2-7b-chat-hf", port=8000):
    """
    启动 vLLM OpenAI 兼容 API 服务
    
    命令行启动（推荐）：
        python -m vllm.entrypoints.openai.api_server \\
            --model {model_name} \\
            --port {port} \\
            --gpu-memory-utilization 0.9
    
    或使用 serve.py:
        vllm serve {model_name} --port {port}
    """
    print(f"""
🚀 启动 vLLM API 服务

命令：
    python -m vllm.entrypoints.openai.api_server \\
        --model {model_name} \\
        --port {port} \\
        --gpu-memory-utilization 0.9 \\
        --max-model-len 2048

API 端点：
    POST /v1/chat/completions  - 聊天补全
    POST /v1/completions       - 文本补全
    GET  /v1/models            - 模型列表

测试命令：
    curl http://localhost:{port}/v1/models
    curl http://localhost:{port}/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{{
            "model": "{model_name}",
            "messages": [{{"role": "user", "content": "你好"}}],
            "max_tokens": 100
        }}'
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        start_vllm_api_server()
    else:
        run_comparison()
```

---

## 项目 3：DeepSpeed ZeRO-3 多卡训练

### 🎯 项目目标

1. 配置 DeepSpeed ZeRO-3 训练环境
2. 训练一个 7B 语言模型（指令微调）
3. 对比 ZeRO-1/2/3 的显存使用和训练速度

### 代码：DeepSpeed ZeRO-3 训练

```python
"""
项目 3：DeepSpeed ZeRO-3 多卡训练

功能：
1. 配置 DeepSpeed ZeRO-3 训练
2. 在指令数据集上微调 7B 模型
3. 支持多卡分布式训练

依赖：
    pip install deepspeed transformers accelerate datasets

运行（单卡）：
    deepspeed --num_gpus 1 train_deepspeed.py --deepspeed ds_config.json

运行（多卡）：
    deepspeed --num_gpus 4 train_deepspeed.py --deepspeed ds_config.json

注意：
    - ZeRO-3 需要 DeepSpeed >= 0.10
    - 建议使用 80GB 显存的 GPU（如 A100/H100）
    - 如果显存不足，可以启用 CPU offload
"""

import os
import json
import time
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)


# ========================================
# 配置文件
# ========================================

# DeepSpeed ZeRO-3 配置
DS_CONFIG = {
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,
    
    # ZeRO-3 配置
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",        # 优化器状态卸载到 CPU
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",        # 参数卸载到 CPU（可选，节省更多显存）
            "pin_memory": True
        },
        "overlap_comm": True,       # 通信和计算重叠
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    
    # 混合精度
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2
    },
    
    # 梯度裁剪
    "gradient_clipping": 1.0,
    
    # 日志
    "steps_per_print": 10,
    "wall_clock_breakdown": False
}


# ========================================
# 数据集
# ========================================

class InstructionDataset(Dataset):
    """
    指令微调数据集
    
    格式：
    {
        "instruction": "用户的指令",
        "input": "可选的输入内容",
        "output": "期望的输出"
    }
    """
    
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Args:
            data_path: 数据文件路径（JSON 格式）
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"📊 加载了 {len(self.data)} 条指令数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        # 拼接指令格式
        if input_text:
            prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 响应:\n"
        else:
            prompt = f"### 指令:\n{instruction}\n\n### 响应:\n"
        
        # Tokenize 输入
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True
        )
        
        # Tokenize 输出
        response_ids = self.tokenizer.encode(
            output,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        
        # 添加结束符
        response_ids = response_ids + [self.tokenizer.eos_token_id]
        
        # 拼接
        input_ids = prompt_ids + response_ids
        
        # 截断到最大长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # 创建 label（prompt 部分用 -100 填充，不计算 loss）
        label_ids = [-100] * len(prompt_ids) + response_ids
        if len(label_ids) > self.max_length:
            label_ids = label_ids[:self.max_length]
        
        # 填充
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        label_ids = label_ids + [-100] * padding_length
        attention_mask = [1 if x != self.tokenizer.pad_token_id else 0 for x in input_ids]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }


# ========================================
# 生成示例数据（用于测试）
# ========================================

def generate_sample_data(output_path="sample_instructions.json", num_samples=1000):
    """
    生成示例指令数据（用于测试）
    
    实际使用时替换为你的指令数据集
    """
    samples = [
        {
            "instruction": "请解释什么是机器学习",
            "input": "",
            "output": "机器学习是人工智能的一个子领域，它使计算机能够从数据中自动学习并改进性能，而无需明确编程。核心思想是让算法从历史数据中发现模式和规律，然后用这些规律来预测新数据。常见的机器学习方法包括监督学习（如分类和回归）、无监督学习（如聚类）和强化学习。"
        },
        {
            "instruction": "将以下句子翻译成英文",
            "input": "人工智能正在改变我们的世界",
            "output": "Artificial intelligence is changing our world."
        },
        {
            "instruction": "请写一个 Python 函数来计算斐波那契数列",
            "input": "",
            "output": "```python\ndef fibonacci(n):\n    \"\"\"计算第 n 个斐波那契数\"\"\"\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n# 测试\nfor i in range(10):\n    print(f\"F({i}) = {fibonacci(i)}\")\n```"
        },
        {
            "instruction": "请总结以下文章的主要内容",
            "input": "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的层次化表示。从图像识别到自然语言处理，深度学习已经在许多领域取得了突破性进展。卷积神经网络（CNN）在计算机视觉中表现出色，循环神经网络（RNN）和 Transformer 在序列处理中占据主导地位。",
            "output": "这篇文章主要介绍了深度学习的基本概念和应用。深度学习通过使用多层神经网络学习数据的层次化表示，在图像识别和自然语言处理等领域取得了显著成果。CNN 在视觉任务中表现优异，而 RNN 和 Transformer 则在序列处理中占据主导地位。"
        },
        {
            "instruction": "请解释 Transformer 中的自注意力机制",
            "input": "",
            "output": "自注意力机制（Self-Attention）是 Transformer 的核心组件。它允许序列中的每个位置关注序列中的所有其他位置，从而捕获长距离依赖关系。\n\n具体计算过程：\n1. 对每个输入向量，生成三个向量：Query（查询）、Key（键）、Value（值）\n2. 计算 Query 和所有 Key 的点积，得到注意力分数\n3. 对分数进行缩放（除以 √d_k）和 Softmax 归一化\n4. 用归一化的权重对 Value 进行加权求和\n\n公式：Attention(Q, K, V) = softmax(QK^T / √d_k) V"
        },
    ]
    
    # 扩展数据集
    while len(samples) < num_samples:
        samples.extend(samples[:num_samples - len(samples)])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples[:num_samples], f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已生成 {num_samples} 条示例数据到 {output_path}")


# ========================================
# 训练函数
# ========================================

def train(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    data_path: str = "sample_instructions.json",
    output_dir: str = "./output",
    epochs: int = 3,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    local_rank: int = 0
):
    """
    DeepSpeed ZeRO-3 训练函数
    
    Args:
        model_name: 预训练模型名称
        data_path: 训练数据路径
        output_dir: 输出目录
        epochs: 训练轮数
        learning_rate: 学习率
        max_length: 最大序列长度
        local_rank: 本地 GPU 排名
    """
    
    set_seed(42)
    
    print("=" * 60)
    print("🚀 DeepSpeed ZeRO-3 训练")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"数据: {data_path}")
    print(f"轮数: {epochs}")
    print(f"学习率: {learning_rate}")
    
    # 初始化 DeepSpeed
    deepspeed.init_distributed()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    dataset = InstructionDataset(data_path, tokenizer, max_length)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 初始化 DeepSpeed engine
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=DS_CONFIG
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=DS_CONFIG["train_micro_batch_size_per_gpu"],
        shuffle=True,
        drop_last=True
    )
    
    # 训练循环
    print(f"\n📈 开始训练...")
    global_step = 0
    total_train_time = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model_engine.train()
        epoch_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            batch_start = time.time()
            
            # 将数据移到 GPU
            input_ids = batch["input_ids"].to(local_rank)
            attention_mask = batch["attention_mask"].to(local_rank)
            labels = batch["labels"].to(local_rank)
            
            # 前向传播
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
            
            batch_time = time.time() - batch_start
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # 打印进度
            if local_rank == 0 and global_step % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Step {global_step} | Epoch {epoch+1}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Batch time: {batch_time:.2f}s")
        
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        avg_epoch_loss = epoch_loss / num_batches
        
        if local_rank == 0:
            print(f"\n✅ Epoch {epoch+1}/{epochs} 完成")
            print(f"   平均 Loss: {avg_epoch_loss:.4f}")
            print(f"   耗时: {epoch_time:.2f}s ({epoch_time/60:.1f} min)")
    
    # 保存模型
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存权重（收集完整权重）
        model_engine.save_16bit_model(os.path.join(output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n💾 模型已保存到: {output_dir}")
    
    # 打印训练总结
    if local_rank == 0:
        print("\n" + "=" * 60)
        print("📊 训练总结")
        print("=" * 60)
        print(f"""
┌──────────────────┬──────────────┐
│ 指标             │ 值           │
├──────────────────┼──────────────┤
│ 总训练时间       │ {total_train_time:10.2f} s  │
│ 总步数           │ {global_step:12d} │
│ 平均每步时间     │ {total_train_time/global_step:10.2f} s  │
│ 最终 Loss        │ {avg_epoch_loss:12.4f} │
└──────────────────┴──────────────┘
        """)


# ========================================
# 主函数
# ========================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, 
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--data_path", type=str, 
                        default="sample_instructions.json")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--generate_data", action="store_true",
                        help="生成示例数据")
    parser.add_argument("--local_rank", type=int, default=0)
    
    args = parser.parse_args()
    
    # 生成示例数据（如果需要）
    if args.generate_data or not os.path.exists(args.data_path):
        generate_sample_data(args.data_path)
    
    # 开始训练
    train(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        local_rank=args.local_rank
    )


if __name__ == "__main__":
    main()
```

### DeepSpeed 配置文件

```json
{
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": { "device": "cpu", "pin_memory": true },
        "offload_param": { "device": "cpu", "pin_memory": true },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "fp16": { "enabled": true },
    "gradient_clipping": 1.0,
    "steps_per_print": 10
}
```

---

## 项目 4：模型蒸馏实战

### 🎯 项目目标

1. 实现教师-学生模型的知识蒸馏
2. 将大模型（LLaMA-2-13B）的知识蒸馏到小模型（LLaMA-2-7B）
3. 对比蒸馏前后小模型的性能

### 代码：知识蒸馏

```python
"""
项目 4：模型蒸馏实战

功能：
1. 实现基于 logits 的知识蒸馏（Dark Knowledge）
2. 教师模型：LLaMA-2-13B（或更小的可用模型）
3. 学生模型：LLaMA-2-7B
4. 对比蒸馏前后学生模型的性能

依赖：
    pip install transformers accelerate datasets

运行：
    python distillation.py

知识蒸馏原理：
    教师模型（大）→ 软标签（Soft Targets）→ 学生模型（小）
    
    损失函数：
    L = α * L_hard(y, student) + (1-α) * L_soft(teacher_soft, student_soft)
    
    其中：
    - L_hard: 标准交叉熵（硬标签）
    - L_soft: KL 散度（软标签，温度缩放）
    - α: 硬标签损失权重
    - 温度 T: 控制软标签的平滑程度
"""

import os
import time
import math
import json
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)


# ========================================
# 蒸馏损失函数
# ========================================

class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    
    结合了硬标签损失（标准交叉熵）和软标签损失（KL 散度）
    
    公式：
    L = α * CE(y, z_s) + (1-α) * T² * KL(softmax(z_t/T), softmax(z_s/T))
    
    其中：
    - z_s: 学生模型的 logits
    - z_t: 教师模型的 logits
    - T: 温度参数
    - α: 硬标签损失权重
    """
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        """
        Args:
            temperature: 温度参数，控制软标签的平滑程度
            alpha: 硬标签损失权重（1-alpha 是软标签损失权重）
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型 logits [batch, seq_len, vocab_size]
            teacher_logits: 教师模型 logits [batch, seq_len, vocab_size]
            labels: 真实标签 [batch, seq_len]
        
        Returns:
            loss: 总损失
            metrics: 各项损失的字典
        """
        # 展平 logits 和 labels
        batch_size, seq_len, vocab_size = student_logits.shape
        student_logits_flat = student_logits.view(-1, vocab_size)
        teacher_logits_flat = teacher_logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # 硬标签损失（标准交叉熵）
        hard_loss = self.ce_loss(student_logits_flat, labels_flat)
        
        # 软标签损失（KL 散度）
        # 温度缩放
        student_soft = F.log_softmax(student_logits_flat / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits_flat / self.temperature, dim=-1)
        
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 总损失
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        metrics = {
            "total_loss": total_loss.item(),
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
            "hard_weight": self.alpha,
            "soft_weight": 1 - self.alpha
        }
        
        return total_loss, metrics


# ========================================
# 蒸馏数据集
# ========================================

class DistillationDataset(Dataset):
    """
    蒸馏数据集
    
    使用通用文本数据，教师模型生成软标签
    """
    
    def __init__(self, texts, tokenizer, max_length=256):
        """
        Args:
            texts: 文本列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # labels 是 input_ids 右移一位（用于因果语言建模）
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # padding 位置不计算 loss
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ========================================
# 生成示例数据
# ========================================

def generate_distillation_data(num_samples=500):
    """
    生成示例蒸馏数据
    
    实际使用时替换为大规模通用文本数据
    """
    texts = [
        "机器学习是人工智能的核心领域之一，它研究如何让计算机从数据中自动学习。"
        "通过算法，计算机可以识别模式、做出决策，并不断改进性能。",
        
        "深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的层次化表示。"
        "从图像到文本，从语音到视频，深度学习已经在各个领域取得了突破性进展。",
        
        "自然语言处理（NLP）致力于让计算机理解和生成人类语言。"
        "近年来，大语言模型的出现极大地推动了 NLP 的发展，"
        "机器翻译、文本摘要、问答系统等任务都取得了显著进步。",
        
        "计算机视觉是人工智能的另一个重要分支，它让计算机能够'看'懂图像和视频。"
        "卷积神经网络（CNN）是计算机视觉的核心技术，"
        "在图像分类、目标检测、语义分割等任务中表现出色。",
        
        "强化学习通过与环境的交互来学习最优策略。"
        "AlphaGo、AlphaStar 等系统在棋类游戏和实时战略游戏中超越了人类水平，"
        "展示了强化学习的强大能力。",
        
        "Transformer 架构通过自注意力机制实现了高效的序列建模。"
        "它摒弃了传统的循环和卷积结构，完全依赖注意力机制来捕获序列中的依赖关系。"
        "这一创新使得训练更大规模的模型成为可能。",
        
        "模型压缩技术包括量化、剪枝和知识蒸馏，目标是减小模型大小并加速推理。"
        "知识蒸馏通过让一个小模型学习大模型的行为，"
        "在保持性能的同时显著减小模型规模。",
        
        "预训练-微调范式已经成为 NLP 的标准方法。"
        "模型首先在大规模无标注数据上进行预训练，学习通用的语言表示，"
        "然后在特定任务上进行微调，适应下游任务的需求。",
        
        "生成式 AI 正在改变内容创作的方式。"
        "从文本生成到图像生成，从代码编写到音乐创作，"
        "生成式模型展现出了惊人的创造力。",
        
        "AI 伦理和安全是随着 AI 技术发展而日益重要的话题。"
        "包括偏见和公平性、隐私保护、可解释性、安全性等方面，"
        "需要在技术发展的同时给予充分关注。",
    ]
    
    # 扩展数据集
    while len(texts) < num_samples:
        texts.extend(texts[:num_samples - len(texts)])
    
    return texts[:num_samples]


# ========================================
# 蒸馏训练
# ========================================

def distill(
    teacher_model_name: str,
    student_model_name: str,
    output_dir: str = "./distilled_model",
    epochs: int = 3,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    max_length: int = 256,
    temperature: float = 2.0,
    alpha: float = 0.5,
    device: str = "cuda"
):
    """
    知识蒸馏训练
    
    Args:
        teacher_model_name: 教师模型名称
        student_model_name: 学生模型名称
        output_dir: 输出目录
        epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批次大小
        max_length: 最大序列长度
        temperature: 蒸馏温度
        alpha: 硬标签损失权重
        device: 设备
    """
    
    set_seed(42)
    
    print("=" * 60)
    print("🧪 模型蒸馏训练")
    print("=" * 60)
    print(f"教师模型: {teacher_model_name}")
    print(f"学生模型: {student_model_name}")
    print(f"温度: {temperature}")
    print(f"硬标签权重: {alpha}")
    
    # 加载分词器（教师和学生使用相同的分词器）
    print("\n📥 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载教师模型（冻结参数）
    print("📥 加载教师模型...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    teacher_model.eval()
    teacher_model.requires_grad_(False)  # 冻结教师模型
    
    # 加载学生模型
    print("📥 加载学生模型...")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 创建数据集
    print("\n📊 准备数据...")
    texts = generate_distillation_data(num_samples=1000)
    dataset = DistillationDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建蒸馏损失
    distillation_loss_fn = DistillationLoss(
        temperature=temperature,
        alpha=alpha
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # 学习率调度
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # 训练循环
    print(f"\n📈 开始蒸馏训练...")
    global_step = 0
    
    for epoch in range(epochs):
        student_model.train()
        epoch_metrics = {
            "total_loss": 0,
            "hard_loss": 0,
            "soft_loss": 0
        }
        num_batches = 0
        
        epoch_start = time.time()
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 获取教师模型的 logits（不计算梯度）
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
            
            # 获取学生模型的 logits
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = student_outputs.logits
            
            # 计算蒸馏损失
            loss, metrics = distillation_loss_fn(
                student_logits, teacher_logits, labels
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # 累积指标
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            num_batches += 1
            global_step += 1
            
            # 打印进度
            if global_step % 20 == 0:
                avg_total = epoch_metrics["total_loss"] / num_batches
                avg_hard = epoch_metrics["hard_loss"] / num_batches
                avg_soft = epoch_metrics["soft_loss"] / num_batches
                print(f"  Step {global_step}/{total_steps} | "
                      f"Total: {avg_total:.4f} | "
                      f"Hard: {avg_hard:.4f} | "
                      f"Soft: {avg_soft:.4f}")
        
        epoch_time = time.time() - epoch_start
        
        # 打印 epoch 结果
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        print(f"\n✅ Epoch {epoch+1}/{epochs} 完成")
        print(f"   总损失: {avg_metrics['total_loss']:.4f}")
        print(f"   硬标签损失: {avg_metrics['hard_loss']:.4f}")
        print(f"   软标签损失: {avg_metrics['soft_loss']:.4f}")
        print(f"   耗时: {epoch_time:.2f}s")
    
    # 保存学生模型
    os.makedirs(output_dir, exist_ok=True)
    student_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n💾 蒸馏后的学生模型已保存到: {output_dir}")
    
    return student_model, tokenizer


# ========================================
# 评估函数
# ========================================

def evaluate_model(model, tokenizer, texts, device="cuda", max_length=256):
    """
    评估模型的困惑度
    
    Args:
        model: 模型
        tokenizer: 分词器
        texts: 评估文本列表
        device: 设备
        max_length: 最大长度
    
    Returns:
        float: 平均困惑度
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            total_loss += loss.item() * (input_ids.shape[1] - 1)
            total_tokens += input_ids.shape[1] - 1
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def compare_models(
    original_model_name: str,
    distilled_model_path: str,
    device: str = "cuda"
):
    """
    对比原始学生模型和蒸馏后模型的性能
    
    Args:
        original_model_name: 原始学生模型名称
        distilled_model_path: 蒸馏模型路径
        device: 设备
    """
    print("\n" + "=" * 60)
    print("📊 模型性能对比")
    print("=" * 60)
    
    # 评估文本
    eval_texts = generate_distillation_data(num_samples=50)
    
    # 评估原始模型
    print("\n🔵 评估原始学生模型...")
    orig_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    if orig_tokenizer.pad_token is None:
        orig_tokenizer.pad_token = orig_tokenizer.eos_token
    
    orig_model = AutoModelForCausalLM.from_pretrained(
        original_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    orig_ppl = evaluate_model(orig_model, orig_tokenizer, eval_texts, device)
    print(f"  原始模型 PPL: {orig_ppl:.2f}")
    
    del orig_model
    torch.cuda.empty_cache()
    
    # 评估蒸馏模型
    print("\n🟢 评估蒸馏后模型...")
    dist_tokenizer = AutoTokenizer.from_pretrained(distilled_model_path)
    dist_model = AutoModelForCausalLM.from_pretrained(
        distilled_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    dist_ppl = evaluate_model(dist_model, dist_tokenizer, eval_texts, device)
    print(f"  蒸馏模型 PPL: {dist_ppl:.2f}")
    
    # 对比总结
    ppl_change = ((dist_ppl - orig_ppl) / orig_ppl) * 100
    
    print("\n" + "=" * 60)
    print("📊 对比总结")
    print("=" * 60)
    
    print(f"""
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│ 指标             │ 原始学生模型 │ 蒸馏后模型   │ 变化         │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 困惑度 (PPL)     │ {orig_ppl:12.2f} │ {dist_ppl:12.2f} │ {ppl_change:+.2f}%      │
└──────────────────┴──────────────┴──────────────┴──────────────┘
    """)
    
    print("💡 结论：")
    if ppl_change < 0:
        print(f"  - 蒸馏后 PPL 降低了 {abs(ppl_change):.2f}%，性能提升！")
    else:
        print(f"  - 蒸馏后 PPL 增加了 {ppl_change:.2f}%（小幅增加是正常的）")
    print(f"  - 蒸馏模型保留了教师模型的部分知识")
    print(f"  - 学生模型更小、更快，适合部署")


# ========================================
# 主函数
# ========================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型蒸馏")
    parser.add_argument("--teacher", type=str, 
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="教师模型（使用 7B 作为示例，实际可用更大的模型）")
    parser.add_argument("--student", type=str,
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="学生模型")
    parser.add_argument("--output_dir", type=str, default="./distilled_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--evaluate", action="store_true",
                        help="评估蒸馏效果")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 执行蒸馏
    student_model, tokenizer = distill(
        teacher_model_name=args.teacher,
        student_model_name=args.student,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        temperature=args.temperature,
        alpha=args.alpha,
        device=device
    )
    
    # 评估
    if args.evaluate:
        compare_models(args.student, args.output_dir, device)


if __name__ == "__main__":
    main()
```

---

## 📝 总结

### 项目完成度检查

| 项目 | 核心技能 | 完成状态 |
|------|---------|---------|
| 项目 1：LLM INT4 量化 | AutoGPTQ、bitsandbytes、精度评估 | ⬜ |
| 项目 2：vLLM 推理服务 | vLLM、吞吐量对比、API 服务 | ⬜ |
| 项目 3：DeepSpeed ZeRO-3 | DeepSpeed、分布式训练、ZeRO | ⬜ |
| 项目 4：模型蒸馏 | 知识蒸馏、教师-学生模型 | ⬜ |

### 关键收获

1. **量化**：INT4 量化可将模型大小减少 75%，在消费级 GPU 上运行大模型
2. **推理加速**：vLLM 的 PagedAttention 和连续批处理可提升 3-10x 吞吐量
3. **分布式训练**：DeepSpeed ZeRO-3 让多卡训练大模型变得简单高效
4. **知识蒸馏**：将大模型知识迁移到小模型，兼顾性能和效率

---

> _优化无止境，每一字节显存都珍贵，每一毫秒延迟都值得优化。_
> 
> _—— 悟空_
