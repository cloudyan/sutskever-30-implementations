# 第 11 章：工程实践（持续）

> 从模型到产品 —— 生产环境部署、MLOps、安全合规
> 
> _学习周期：持续 | 难度：⭐⭐⭐⭐ | 重要性：⭐⭐⭐⭐⭐_

---

## 📖 本章概述

### AI 工程化挑战

```
实验室 → 生产环境的差距：

┌─────────────────────────────────────────────────────────────────┐
│ 维度        │ 实验室          │ 生产环境        │ 差距          │
├─────────────────────────────────────────────────────────────────┤
│ 用户量      │ 1-10 人          │ 1 万 -1000 万 +    │ 100 万倍       │
│ 请求量      │ 手动测试        │ 1000+ QPS       │ 无限倍       │
│ 可用性      │ 随时重启        │ 99.9% SLA       │ 严格         │
│ 延迟要求    │ 秒级可接受      │ <100ms          │ 10 倍         │
│ 监控        │ 基本日志        │ 全链路追踪      │ 复杂         │
│ 成本        │ 不太关注        │ 严格优化        │ 必须         │
└─────────────────────────────────────────────────────────────────┘
```

### 本章学习目标

学完本章后，你将能够：
- ✅ 部署生产级模型服务
- ✅ 搭建 MLOps 流水线
- ✅ 实现监控和告警
- ✅ 保证数据隐私和模型安全
- ✅ 完成从原型到产品的转化

---

## 📚 学习大纲

### 11.1 生产环境部署（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 模型服务化

```python
"""
模型服务化架构：

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│   API       │───▶│   Model     │
│  (Web/App)  │◀───│   Gateway   │◀───│   Service   │
└─────────────┘    └─────────────┘    └─────────────┘
                        │                    │
                        ▼                    ▼
                  ┌─────────────┐    ┌─────────────┐
                  │   Auth      │    │   Cache     │
                  │   Rate Limit│    │   Queue     │
                  └─────────────┘    └─────────────┘
"""

# FastAPI 模型服务示例
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn

app = FastAPI(title="AI Model API")

# 请求/响应模型
class PredictionRequest(BaseModel):
    text: str
    max_length: int = 512

class PredictionResponse(BaseModel):
    result: str
    confidence: float
    latency_ms: float

# 加载模型
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "THUDM/chatglm3-6b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    import time
    start_time = time.time()
    
    try:
        # Tokenize
        inputs = tokenizer(request.text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                do_sample=True,
                temperature=0.7
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        latency = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            result=result,
            confidence=0.95,  # 实际应计算
            latency_ms=latency
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 运行
# uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 负载均衡配置

```yaml
# Nginx 负载均衡配置
upstream model_servers {
    least_conn;  # 最少连接优先
    server 127.0.0.1:8001 weight=3;
    server 127.0.0.1:8002 weight=3;
    server 127.0.0.1:8003 weight=3;
    server 127.0.0.1:8004 backup;  # 备用服务器
}

server {
    listen 80;
    server_name api.example.com;
    
    # 请求限制
    limit_req_zone $binary_remote_addr zone=one:10m rate=10r/s;
    
    location / {
        limit_req zone=one burst=20 nodelay;
        
        proxy_pass http://model_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # 健康检查
    location /health {
        access_log off;
        return 200 "healthy\n";
    }
}
```

#### 监控告警

```python
"""
监控指标：

1. 系统指标
   - CPU 使用率
   - GPU 使用率
   - 内存使用率
   - 磁盘 IO

2. 服务指标
   - QPS (Queries Per Second)
   - 延迟 (P50/P90/P99)
   - 错误率
   - 成功率

3. 业务指标
   - 活跃用户数
   - 平均响应长度
   - Token 消耗量
   - 成本

工具栈：
- Prometheus: 指标收集
- Grafana: 可视化
- AlertManager: 告警
- ELK: 日志分析
"""

# Prometheus 指标集成
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import random

# 定义指标
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API request latency')
GPU_MEMORY = Gauge('gpu_memory_used_bytes', 'GPU memory used', ['gpu_id'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')

# 启动指标服务器
start_http_server(8000)

# 在 API 中使用
@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()
    
    try:
        # ... 处理逻辑
        latency = time.time() - start_time
        
        # 记录指标
        REQUEST_COUNT.labels(method='predict', status='success').inc()
        REQUEST_LATENCY.observe(latency)
        
        return result
    
    except Exception as e:
        REQUEST_COUNT.labels(method='predict', status='error').inc()
        raise
    
    finally:
        ACTIVE_CONNECTIONS.dec()

# GPU 监控
def monitor_gpu():
    import pynvml
    pynvml.nvmlInit()
    
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        GPU_MEMORY.labels(gpu_id=i).set(info.used)

# 定期运行
import threading
threading.Thread(target=lambda: [monitor_gpu(), time.sleep(60)]*1000, daemon=True).start()
```

</details>

---

### 11.2 MLOps（2 周）

<details>
<summary>📋 查看详细知识点</summary>

#### MLOps 流水线

```
完整 MLOps 流程：

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  数据 → 特征工程 → 训练 → 评估 → 部署 → 监控 → 迭代              │
│   │         │        │      │      │      │      │             │
│   ▼         ▼        ▼      ▼      ▼      ▼      ▼             │
│  DVC      特征     MLflow  测试   CI/CD  监控   重新训练         │
│  版本     存储     实验   验证   部署   告警   触发              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 实验追踪（MLflow）

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 设置追踪服务器
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris_classification")

# 开始实验
with mlflow.start_run():
    # 参数
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)
    
    # 数据
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2
    )
    
    # 训练
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 评估
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # 记录模型
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# 查看实验
# mlflow ui --port 5000
```

#### CI/CD for ML

```yaml
# GitHub Actions MLOps 流水线
name: MLOps Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=src
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Train model
        run: python src/train.py
      
      - name: Evaluate model
        run: python src/evaluate.py
      
      - name: Register model (if accuracy > threshold)
        run: python src/register_model.py

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # 部署逻辑
          echo "Deploying to production..."
```

</details>

---

### 11.3 安全与合规（1 周）

<details>
<summary>📋 查看详细知识点</summary>

#### 数据隐私

```python
"""
数据隐私保护措施：

1. 数据脱敏
   - 移除 PII（个人身份信息）
   - 数据匿名化
   - k-匿名性

2. 差分隐私
   - 添加噪声保护个体
   - 保证统计准确性

3. 联邦学习
   - 数据不出本地
   - 只共享模型更新

4. 加密
   - 传输加密（TLS）
   - 存储加密
   - 同态加密（前沿）
"""

# 数据脱敏示例
import re
from faker import Faker

fake = Faker('zh_CN')

def anonymize_text(text):
    """脱敏文本中的个人信息"""
    # 替换手机号
    text = re.sub(r'1[3-9]\d{9}', '[PHONE]', text)
    
    # 替换邮箱
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # 替换身份证号
    text = re.sub(r'\d{17}[\dXx]', '[ID_CARD]', text)
    
    return text

# 差分隐私示例
import numpy as np

def laplace_mechanism(value, epsilon):
    """拉普拉斯机制实现差分隐私"""
    noise = np.random.laplace(0, 1/epsilon)
    return value + noise

# 使用
true_count = 1000
private_count = laplace_mechanism(true_count, epsilon=0.1)
print(f"真实值：{true_count}, 隐私保护后：{private_count:.0f}")
```

#### 模型安全

```python
"""
模型安全威胁：

1. 对抗攻击
   - 输入微小扰动导致错误预测
   - 防御：对抗训练

2. 模型窃取
   - 通过 API 查询复制模型
   - 防御：查询限制、水印

3. 成员推断
   - 判断某样本是否在训练集
   - 防御：差分隐私

4. 模型投毒
   - 训练数据被恶意污染
   - 防御：数据验证、鲁棒训练
"""

# 对抗攻击示例（FGSM）
import torch

def fgsm_attack(model, data, target, epsilon):
    """FGSM 对抗攻击"""
    data.requires_grad = True
    
    output = model(data)
    loss = torch.nn.functional.cross_entropy(output, target)
    
    # 计算梯度
    loss.backward()
    
    # 生成对抗样本
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    
    return perturbed_data

# 防御：对抗训练
def adversarial_training(model, dataloader, epsilon=0.01):
    """对抗训练"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    
    for data, target in dataloader:
        # 正常训练
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        
        # 对抗训练
        adv_data = fgsm_attack(model, data, target, epsilon)
        adv_output = model(adv_data)
        adv_loss = torch.nn.functional.cross_entropy(adv_output, target)
        adv_loss.backward()
        
        optimizer.step()
```

</details>

---

## 📊 进度追踪

### 项目清单

- [x] FastAPI 模型服务部署（详见 [projects.md](projects.md) 项目 1）
- [x] Nginx 负载均衡配置
- [x] Prometheus+Grafana 监控（详见 [projects.md](projects.md) 项目 1）
- [x] MLflow 实验追踪（详见 [projects.md](projects.md) 项目 2）
- [x] GitHub Actions CI/CD（详见 [projects.md](projects.md) 项目 2）
- [x] 模型 A/B 测试系统（详见 [projects.md](projects.md) 项目 3）
- [x] RAG 生产部署（详见 [projects.md](projects.md) 项目 4）
- [ ] 数据脱敏工具
- [ ] 对抗训练实现

---

> _工程是将科学转化为艺术的桥梁，好的工程让 AI 真正服务于人。_
> 
> _—— 悟空_
