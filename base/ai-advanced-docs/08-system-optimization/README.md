# 第 8 章：大模型系统优化（8-10 周）

> 让大模型跑得更快、更省、更稳 —— 从分布式训练到推理加速
> 
> _学习周期：8-10 周 | 难度：⭐⭐⭐⭐⭐ | 重要性：⭐⭐⭐⭐⭐_

---

## 📖 本章概述

### 为什么需要系统优化？

```
大模型训练挑战：

┌─────────────────────────────────────────────────────────────────┐
│ 模型规模        │ 训练时间    │ 显存需求   │ 成本          │
├─────────────────────────────────────────────────────────────────┤
│ GPT-3 (175B)    │ 35 天       │ 多节点     │ $4.6M+        │
│ LLaMA-2 (70B)   │ 18 天       │ 多节点     │ $2M+          │
│ ChatGLM3 (6B)   │ 数天        │ 多卡       │ 数十万        │
└─────────────────────────────────────────────────────────────────┘

优化目标：
1. 训练更快：从月到天
2. 显存更省：从多卡到单卡
3. 推理更快：从高延迟到实时
4. 成本更低：从百万到万元
```

### 本章学习目标

学完本章后，你将能够：
- ✅ 掌握分布式训练策略（数据/模型/流水线并行）
- ✅ 使用 DeepSpeed 进行 ZeRO 优化
- ✅ 实现模型量化（INT8/INT4）
- ✅ 使用 vLLM/TGI 加速推理
- ✅ 分析和优化训练瓶颈

---

## 📚 学习大纲

### 8.1 分布式训练（3 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 并行策略对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    分布式训练并行策略                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. 数据并行 (Data Parallelism)                                  │
│    ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                         │
│    │ GPU0│  │ GPU1│  │ GPU2│  │ GPU3│                         │
│    │全模型│  │全模型│  │全模型│  │全模型│                         │
│    │数据 1│  │数据 2│  │数据 3│  │数据 4│                         │
│    └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                         │
│       └────────┴───┬───┴────────┘                               │
│                    ▼                                            │
│              梯度同步 (All-Reduce)                               │
│                                                                 │
│ 优点：实现简单，通信量小                                         │
│ 缺点：单卡必须能放下整个模型                                     │
│                                                                 │
│ 2. 模型并行 (Model Parallelism)                                 │
│    ┌─────────────┐  ┌─────────────┐                             │
│    │ GPU0        │  │ GPU1        │                             │
│    │ 层 1-6      │  │ 层 7-12     │                             │
│    │ (1/2 模型)   │  │ (1/2 模型)   │                             │
│    └──────┬──────┘  └──────┬──────┘                             │
│           │                │                                     │
│           └───────┬────────┘                                     │
│              激活/梯度传递                                        │
│                                                                 │
│ 优点：可训练超大模型                                             │
│ 缺点：通信量大，GPU 利用率低                                       │
│                                                                 │
│ 3. 流水线并行 (Pipeline Parallelism)                            │
│    ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                         │
│    │ GPU0│  │ GPU1│  │ GPU2│  │ GPU3│                         │
│    │层 1-3│→ │层 4-6│→ │层 7-9│→ │层 10-12│                    │
│    └─────┘  └─────┘  └─────┘  └─────┘                         │
│      ↑                              │                          │
│      └──────────────────────────────┘                          │
│                    │                                            │
│              气泡等待（空闲时间）                                 │
│                                                                 │
│ 优点：减少单卡显存占用                                           │
│ 缺点：气泡问题，GPU 利用率低                                       │
│                                                                 │
│ 4. 3D 并行（混合并行）                                           │
│    数据并行 + 模型并行 + 流水线并行                               │
│    大规模训练的标准方案                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 数据并行实现（PyTorch DDP）

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup_ddp(rank, world_size):
    """初始化分布式环境"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """清理分布式环境"""
    dist.destroy_process_group()

def ddp_example(rank, world_size):
    """数据并行训练示例"""
    # 1. 初始化
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 2. 创建模型（每卡一份）
    model = YourModel().to(rank)
    
    # 3. 包装为 DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # 4. 创建数据加载器（数据分片）
    dataset = YourDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    # 5. 训练
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        sampler.set_epoch(epoch)  # 每轮重新打乱
        
        for batch in dataloader:
            x, y = batch
            x, y = x.to(rank), y.to(rank)
            
            optimizer.zero_grad()
            output = ddp_model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        if rank == 0:  # 只在主进程打印
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    cleanup_ddp()

# 运行（使用 torchrun）
# torchrun --nproc_per_node=4 train_ddp.py
```

#### DeepSpeed ZeRO 优化

```python
"""
ZeRO（Zero Redundancy Optimizer）优化：

ZeRO-1: 优化器状态分片
  - 每卡只保存 1/N 的优化器状态
  - 显存节省：N 倍

ZeRO-2: 优化器状态 + 梯度分片
  - 每卡只保存 1/N 的梯度
  - 显存节省：2N 倍

ZeRO-3: 优化器状态 + 梯度 + 参数分片
  - 每卡只保存 1/N 的参数
  - 显存节省：3N 倍
  - 可训练超大模型
"""

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamClass
import deepspeed

# ZeRO 配置文件（ds_config.json）
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    
    # ZeRO 配置
    "zero_optimization": {
        "stage": 3,              # ZeRO-3
        "offload_optimizer": {
            "device": "cpu",     # 优化器卸载到 CPU
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",     # 参数卸载到 CPU
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    
    # 混合精度训练
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    
    # 优化器
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    
    # 学习率调度
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 1000
        }
    }
}

# 使用 DeepSpeed 训练
import deepspeed

model = YourLargeModel()

# 初始化 DeepSpeed
model_engine, optimizer, trainloader, scheduler = deepspeed.initialize(
    model=model,
    config=ds_config,
    training_data=dataset
)

# 训练循环
for epoch in range(10):
    for batch in trainloader:
        loss = model_engine.train_batch()
        # DeepSpeed 自动处理梯度累积、ZeRO 分片等
```

</details>

---

### 8.2 推理优化（3 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 模型量化

```
量化原理：
FP32 (32 位浮点) → INT8 (8 位整数) → INT4 (4 位整数)

优势：
- 模型大小：FP32 的 1/4 (INT8) 或 1/8 (INT4)
- 推理速度：提升 2-4 倍
- 显存占用：大幅降低

精度损失：
- INT8: 通常 < 1%
- INT4: 通常 1-3%（需要校准）
```

#### Post-Training Quantization (PTQ)

```python
import torch
from torch.ao.quantization import quantize_dynamic

# 1. 动态量化（最简单）
model_fp32 = YourModel()
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # 量化哪些层
    dtype=torch.qint8
)

# 保存
torch.save(model_int8.state_dict(), "model_int8.pth")

# 2. 静态量化（需要校准）
from torch.ao.quantization import (
    QuantStub, DeQuantStub, 
    prepare, convert, 
    QConfig, default_qconfig
)

class QuantizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()      # FP32 → INT8
        self.dequant = DeQuantStub()  # INT8 → FP32
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, 10)
    
    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

model = QuantizedModel()
model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

# 准备量化
model_prepared = prepare(model)

# 校准（用少量数据）
for batch in calibration_dataloader:
    model_prepared(batch)

# 转换为量化模型
model_quantized = convert(model_prepared)
```

#### GPTQ 量化（大模型专用）

```python
# 使用 AutoGPTQ 量化大模型
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,              # 4bit 量化
    group_size=128,      # 分组大小
    damp_percent=0.01,   # 阻尼系数
    desc_act=False,      # 是否激活排序
)

# 加载模型并量化
model = AutoGPTQForCausalLM.from_pretrained(
    "THUDM/chatglm3-6b",
    quantize_config=quantize_config
)

# 量化（需要校准数据）
model.quantize(calibration_data, cache_examples_on_gpu=False)

# 保存量化模型
model.save_quantized("chatglm3-6b-int4")
```

#### 推理框架对比

```python
"""
推理框架对比：

┌─────────────┬───────────┬───────────┬───────────┬───────────┐
│ 框架        │ vLLM      │ TGI       │ TensorRT  │ ONNX      │
├─────────────┼───────────┼───────────┼───────────┼───────────┤
│ 开发者      │ UC Berkeley│ HuggingFace│ NVIDIA   │ Microsoft │
├─────────────┼───────────┼───────────┼───────────┼───────────┤
│ 核心优化    │ Paged     │ Continuous│ Layer     │ Graph     │
│             │ Attention │ Batching  │ Fusion    │ Optimize  │
├─────────────┼───────────┼───────────┼───────────┼───────────┤
│ 吞吐量      │ ⭐⭐⭐⭐⭐    │ ⭐⭐⭐⭐     │ ⭐⭐⭐⭐⭐    │ ⭐⭐⭐       │
├─────────────┼───────────┼───────────┼───────────┼───────────┤
│ 易用性      │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐⭐     │ ⭐⭐        │ ⭐⭐⭐⭐      │
├─────────────┼───────────┼───────────┼───────────┼───────────┤
│ 模型支持    │ 主流 LLM  │ 主流 LLM  │ 需转换    │ 广泛      │
└─────────────┴───────────┴───────────┴───────────┴───────────┘
"""

# vLLM 使用示例
from vllm import LLM, SamplingParams

# 创建 LLM 实例
llm = LLM(
    model="THUDM/chatglm3-6b",
    tensor_parallel_size=1,  # GPU 数量
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

# 采样配置
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# 批量推理
prompts = [
    "什么是机器学习？",
    "Python 中如何实现快速排序？",
    "请解释量子力学的基本原理"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```

</details>

---

### 8.3 显存优化（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 显存分析工具

```python
import torch

def print_memory_usage():
    """打印显存使用情况"""
    if torch.cuda.is_available():
        print(f"已分配：{torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"已缓存：{torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"最大分配：{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# 使用示例
print_memory_usage()

# 使用 torch.cuda.mem_get_info 获取详细信息
free_mem, total_mem = torch.cuda.mem_get_info()
print(f"总显存：{total_mem / 1024**3:.2f} GB")
print(f"可用显存：{free_mem / 1024**3:.2f} GB")
```

#### 梯度检查点（Gradient Checkpointing）

```python
"""
梯度检查点原理：

正常训练：
- 保存所有中间激活值
- 显存占用：O(n)
- 反向传播：直接使用激活值

梯度检查点：
- 只保存部分激活值
- 显存占用：O(√n)
- 反向传播：重新计算缺失的激活值
- 时间开销：增加 20-30%
"""

import torch
from torch.utils.checkpoint import checkpoint

class CheckpointModel(torch.nn.Module):
    """使用梯度检查点的模型"""
    
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(768, 768) for _ in range(24)
        ])
    
    def forward(self, x):
        # 每 4 层使用一次检查点
        for i, layer in enumerate(self.layers):
            if i % 4 == 0:
                x = checkpoint(layer, x)  # 重新计算
            else:
                x = layer(x)
        return x

# PyTorch 内置支持
model = YourLargeModel()
model.gradient_checkpointing_enable()  # 启用梯度检查点
```

#### 混合精度训练

```python
"""
混合精度训练：

FP32（单精度）：32 位，精度高，显存占用大
FP16（半精度）：16 位，精度低，显存占用小，速度快

混合精度：
- 权重：FP32（保持精度）
- 激活/梯度：FP16（节省显存）
- 损失缩放：防止梯度下溢
"""

from torch.cuda.amp import autocast, GradScaler

model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

for batch in dataloader:
    x, y = batch.cuda()
    
    optimizer.zero_grad()
    
    # 自动混合精度
    with autocast():
        output = model(x)
        loss = criterion(output, y)
    
    # 缩放梯度
    scaler.scale(loss).backward()
    
    # 更新权重（自动 unscale）
    scaler.step(optimizer)
    scaler.update()
```

</details>

---

### 8.4 性能调优（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### PyTorch Profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# 性能分析
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for batch in dataloader:
        with record_function("model_inference"):
            output = model(batch)

# 打印分析结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出为 Chrome Trace 格式
prof.export_chrome_trace("trace.json")
# 在 Chrome 浏览器中打开 trace.json 查看可视化结果
```

#### 批处理优化

```python
"""
批处理优化技巧：

1. 动态批处理（Dynamic Batching）
   - 根据请求数量动态调整 batch size
   - 平衡延迟和吞吐量

2. 序列打包（Sequence Packing）
   - 将多个短序列打包成一个 batch
   - 减少 padding 浪费

3. 连续批处理（Continuous Batching）
   - 请求完成立即加入新请求
   - vLLM/TGI 的核心优化
"""

# 序列打包示例
def pack_sequences(sequences, max_length):
    """将多个短序列打包"""
    packed = []
    current_pack = []
    current_length = 0
    
    for seq in sequences:
        if current_length + len(seq) <= max_length:
            current_pack.append(seq)
            current_length += len(seq)
        else:
            # 填充并保存当前包
            packed_batch = pad_sequence(current_pack)
            packed.append(packed_batch)
            current_pack = [seq]
            current_length = len(seq)
    
    return packed
```

</details>

---

## 📊 进度追踪

### 打卡表

| 章节 | 周数 | 已完成 | 进度 | 状态 |
|------|------|--------|------|------|
| 8.1 分布式训练 | 3 周 | - | 0% | ⏳ |
| 8.2 推理优化 | 3 周 | - | 0% | ⏳ |
| 8.3 显存优化 | 2 周 | - | 0% | ⏳ |
| 8.4 性能调优 | 1 周 | - | 0% | ⏳ |

### 项目清单

- [ ] DDP 数据并行训练
- [ ] DeepSpeed ZeRO-3 配置
- [ ] INT4 量化部署
- [ ] vLLM 推理服务
- [ ] 性能 Profiling 分析

---

> _优化是永恒的主题，让每一字节显存都发挥价值，让每一毫秒延迟都有意义。_
> 
> _—— 悟空_
