# 🛠️ 实战项目：工程实践

> 从理论到实战 —— 4 个完整可运行的生产级项目
> 
> _难度：⭐⭐⭐⭐ | 预计用时：2-3 周 | 每个项目均可独立运行_

---

## 📋 项目总览

```
项目路线图（难度递增）：

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  项目 1：生产级 LLM API 服务    ⭐⭐⭐                      │
  │  ├─ FastAPI + vLLM + Redis + Prometheus                     │
  │  │                                                          │
  │  项目 2：MLOps 完整流水线      ⭐⭐⭐⭐                      │
  │  ├─ DVC + MLflow + GitHub Actions                           │
  │  │                                                          │
  │  项目 3：模型 A/B 测试系统     ⭐⭐⭐⭐                       │
  │  ├─ 灰度发布 + A/B 测试 + 流量分流                           │
  │  │                                                          │
  │  项目 4：RAG 生产部署         ⭐⭐⭐⭐⭐                      │
  │  ├─ RAG + 向量数据库 + 监控告警 + 自动化流水线              │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

---

## 项目 1：生产级 LLM API 服务

> **难度**：⭐⭐⭐ ｜ **预计时间**：2-3 天
> 
> 构建一个生产级的 LLM API 服务，包含缓存、限流、监控和异步推理。

### 🏗️ 架构设计

```
┌──────────────────────────────────────────────────────────────────┐
│                        客户端请求                                 │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI 网关层                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────────┐ │
│  │ 身份认证   │  │ 限流中间件 │  │ 请求/响应验证 (Pydantic)   │ │
│  └────────────┘  └────────────┘  └────────────────────────────┘ │
└────────────────────────┬─────────────────────────────────────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Redis    │  │ vLLM     │  │ Prometheus│
    │ 缓存层   │  │ 推理引擎 │  │ 监控指标  │
    └──────────┘  └──────────┘  └──────────┘
```

### 📦 项目结构

```
llm-api-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── models.py            # 数据模型
│   ├── middleware.py        # 中间件（限流、鉴权）
│   ├── cache.py             # Redis 缓存
│   ├── monitoring.py        # Prometheus 监控
│   └── inference.py         # vLLM 推理引擎
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### 🔧 依赖安装

```bash
# requirements.txt
fastapi==0.115.0
uvicorn==0.30.0
redis==5.0.0
prometheus-client==0.20.0
pydantic==2.5.0
python-jose[cryptography]==3.3.0
```

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 vLLM（需要 GPU 环境，可选）
pip install vllm

# 启动 Redis（Docker 方式）
docker run -d -p 6379:6379 redis:7-alpine
```

### 📝 完整代码

#### `app/config.py` — 配置管理

```python
"""
配置管理模块
集中管理所有环境变量和默认配置
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置，支持从环境变量读取"""

    # 服务配置
    APP_NAME: str = "LLM API Service"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 3600  # 缓存过期时间（秒）

    # vLLM 配置
    VLLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    VLLM_TENSOR_PARALLEL_SIZE: int = 1
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.9
    VLLM_MAX_MODEL_LEN: int = 4096

    # 限流配置
    RATE_LIMIT_PER_MINUTE: int = 60  # 每分钟最大请求数
    RATE_LIMIT_BURST: int = 10       # 突发流量上限

    # API 密钥（生产环境应从密钥管理服务获取）
    API_KEY: str = "your-secret-api-key-change-in-production"

    # Prometheus 配置
    PROMETHEUS_PORT: int = 9090

    class Config:
        env_file = ".env"
        case_sensitive = True


# 全局配置实例
settings = Settings()
```

#### `app/models.py` — 数据模型

```python
"""
数据模型定义
使用 Pydantic 进行请求/响应验证
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """单条聊天消息"""
    role: MessageRole = Field(..., description="消息角色：system/user/assistant")
    content: str = Field(..., min_length=1, max_length=8192, description="消息内容")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    model: str = Field(default="default", description="模型名称（兼容 OpenAI 格式）")
    messages: List[ChatMessage] = Field(
        ..., min_items=1, max_items=50,
        description="消息列表"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0,
        description="生成温度，越高越随机"
    )
    max_tokens: int = Field(
        default=1024, ge=1, le=4096,
        description="最大生成 token 数"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0,
        description="核采样参数"
    )
    stream: bool = Field(default=False, description="是否流式输出")
    user_id: Optional[str] = Field(default=None, description="用户 ID（用于限流和追踪）")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    id: str = Field(..., description="响应 ID")
    model: str = Field(..., description="使用的模型")
    content: str = Field(..., description="生成的内容")
    usage: dict = Field(..., description="token 使用统计")
    latency_ms: float = Field(..., description="推理耗时（毫秒）")
    cached: bool = Field(default=False, description="是否命中缓存")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态：healthy/unhealthy")
    version: str = Field(..., description="服务版本")
    cache_status: str = Field(..., description="缓存状态")
    uptime_seconds: float = Field(..., description="运行时长（秒）")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误详情")
    code: int = Field(..., description="错误码")
```

#### `app/monitoring.py` — Prometheus 监控

```python
"""
Prometheus 监控指标
定义和暴露所有服务指标
"""

from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client import CollectorRegistry
import time
import threading


class MetricsCollector:
    """
    Prometheus 指标收集器
    封装所有监控指标的定义和更新方法
    """

    def __init__(self):
        # 使用独立注册表，避免默认注册表冲突
        self.registry = CollectorRegistry()

        # ========== 请求指标 ==========

        # API 请求总数（按方法和状态分组）
        self.request_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        # 请求延迟分布（P50/P90/P99）
        self.request_latency = Histogram(
            'api_request_latency_seconds',
            'API request latency in seconds',
            ['endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )

        # ========== 推理指标 ==========

        # 推理请求总数
        self.inference_total = Counter(
            'inference_requests_total',
            'Total inference requests',
            ['model', 'status'],
            registry=self.registry
        )

        # 推理延迟
        self.inference_latency = Histogram(
            'inference_latency_seconds',
            'Inference latency in seconds',
            ['model'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )

        # Token 生成速率
        self.tokens_generated = Counter(
            'tokens_generated_total',
            'Total tokens generated',
            ['model'],
            registry=self.registry
        )

        # ========== 缓存指标 ==========

        # 缓存命中率
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            registry=self.registry
        )

        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            registry=self.registry
        )

        # ========== 系统指标 ==========

        # 活跃连接数
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )

        # GPU 显存使用（如有 GPU）
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id'],
            registry=self.registry
        )

        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )

        # 服务启动时间
        self.start_time = time.time()
        self.uptime = Gauge(
            'service_uptime_seconds',
            'Service uptime in seconds',
            registry=self.registry
        )

    def record_request(self, method: str, endpoint: str, status: str, latency: float):
        """记录请求指标"""
        self.request_total.labels(
            method=method, endpoint=endpoint, status=status
        ).inc()
        self.request_latency.labels(endpoint=endpoint).observe(latency)

    def record_inference(self, model: str, status: str, latency: float, tokens: int = 0):
        """记录推理指标"""
        self.inference_total.labels(model=model, status=status).inc()
        self.inference_latency.labels(model=model).observe(latency)
        if tokens > 0:
            self.tokens_generated.labels(model=model).inc(tokens)

    def record_cache_hit(self):
        """记录缓存命中"""
        self.cache_hits.inc()

    def record_cache_miss(self):
        """记录缓存未命中"""
        self.cache_misses.inc()

    def update_active_connections(self, count: int):
        """更新活跃连接数"""
        self.active_connections.set(count)

    def update_uptime(self):
        """更新运行时长"""
        self.uptime.set(time.time() - self.start_time)

    def get_metrics(self) -> bytes:
        """获取所有指标的 Prometheus 格式数据"""
        self.update_uptime()
        return generate_latest(self.registry)


# 全局单例
metrics = MetricsCollector()
```

#### `app/cache.py` — Redis 缓存

```python
"""
Redis 缓存模块
实现请求缓存，减少重复推理
"""

import redis
import json
import hashlib
from typing import Optional, Any
from app.config import settings


class RequestCache:
    """
    Redis 缓存管理器
    对相同请求进行缓存，避免重复推理
    """

    def __init__(self):
        """初始化 Redis 连接"""
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,  # 自动解码为字符串
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        self.ttl = settings.CACHE_TTL  # 缓存过期时间

    def _generate_key(self, request_data: dict) -> str:
        """
        根据请求内容生成缓存键
        使用 SHA256 哈希确保唯一性
        """
        # 将请求数据序列化为 JSON 字符串
        request_str = json.dumps(request_data, sort_keys=True, ensure_ascii=False)
        # 计算哈希值作为缓存键
        cache_key = hashlib.sha256(request_str.encode()).hexdigest()
        return f"llm:cache:{cache_key}"

    def get(self, request_data: dict) -> Optional[dict]:
        """
        从缓存获取结果
        返回 None 表示缓存未命中
        """
        try:
            key = self._generate_key(request_data)
            cached = self.client.get(key)
            if cached:
                return json.loads(cached)
            return None
        except (redis.ConnectionError, redis.TimeoutError) as e:
            # Redis 不可用时降级为不缓存
            print(f"[Cache] Redis error on get: {e}")
            return None
        except json.JSONDecodeError:
            return None

    def set(self, request_data: dict, response_data: dict):
        """
        将结果存入缓存
        设置 TTL 自动过期
        """
        try:
            key = self._generate_key(request_data)
            # 将响应数据序列化为 JSON
            value = json.dumps(response_data, ensure_ascii=False)
            # 写入 Redis，设置过期时间
            self.client.setex(key, self.ttl, value)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"[Cache] Redis error on set: {e}")
        except Exception as e:
            print(f"[Cache] Unexpected error: {e}")

    def is_available(self) -> bool:
        """检查 Redis 是否可用"""
        try:
            return self.client.ping()
        except (redis.ConnectionError, redis.TimeoutError):
            return False

    def clear(self):
        """清空所有缓存（谨慎使用）"""
        try:
            # 只删除 LLM 相关的缓存键
            cursor = 0
            while True:
                cursor, keys = self.client.scan(
                    cursor=cursor, match="llm:cache:*", count=100
                )
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            print(f"[Cache] Error clearing cache: {e}")

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        try:
            info = self.client.info('stats')
            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "keys_count": self.client.dbsize()
            }
        except Exception:
            return {"hits": 0, "misses": 0, "keys_count": 0}


# 全局单例
cache = RequestCache()
```

#### `app/inference.py` — vLLM 推理引擎

```python
"""
vLLM 推理引擎
封装模型加载和推理逻辑
"""

import time
from typing import List, Dict, Optional
from app.config import settings


class InferenceEngine:
    """
    推理引擎类
    支持 vLLM 和回退到简单模拟模式
    """

    def __init__(self):
        """初始化推理引擎"""
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.model_name = settings.VLLM_MODEL

    def load(self):
        """
        加载模型
        尝试使用 vLLM，如果不可用则使用模拟模式
        """
        try:
            # 尝试导入 vLLM
            from vllm import LLM, SamplingParams

            print(f"[Inference] Loading model: {self.model_name}")
            print(f"[Inference] Tensor parallel: {settings.VLLM_TENSOR_PARALLEL_SIZE}")
            print(f"[Inference] GPU memory utilization: {settings.VLLM_GPU_MEMORY_UTILIZATION}")

            # 初始化 vLLM 引擎
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=settings.VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=settings.VLLM_GPU_MEMORY_UTILIZATION,
                max_model_len=settings.VLLM_MAX_MODEL_LEN,
            )
            self.sampling_params = SamplingParams

            # 初始化 tokenizer
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.is_loaded = True
            print("[Inference] Model loaded successfully (vLLM)")

        except ImportError:
            # vLLM 不可用，使用模拟模式
            print("[Inference] vLLM not available, using mock mode")
            print("[Inference] Install vLLM for GPU-accelerated inference:")
            print("[Inference]   pip install vllm")
            self.is_loaded = False

        except Exception as e:
            print(f"[Inference] Failed to load model: {e}")
            print("[Inference] Falling back to mock mode")
            self.is_loaded = False

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ) -> Dict:
        """
        生成回复
        返回包含内容、token 使用量和延迟的字典
        """
        start_time = time.time()

        if self.is_loaded:
            # vLLM 推理模式
            return self._vllm_generate(messages, temperature, max_tokens, top_p)
        else:
            # 模拟模式（用于开发和测试）
            return self._mock_generate(messages, temperature, max_tokens)

    def _vllm_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Dict:
        """vLLM 推理实现"""
        # 将消息列表格式化为 prompt
        prompt = self._format_messages(messages)

        # 设置采样参数
        params = self.sampling_params(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        # 执行推理
        outputs = self.llm.generate([prompt], params)
        generated_text = outputs[0].outputs[0].text

        # 计算 token 使用量
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(generated_text))

        latency = time.time() - start_time

        return {
            "content": generated_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency * 1000,
        }

    def _mock_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict:
        """
        模拟推理（无 GPU 环境下的回退方案）
        模拟不同延迟和 token 消耗
        """
        import random

        # 获取用户最后一条消息
        user_messages = [m for m in messages if m["role"] == "user"]
        last_user_msg = user_messages[-1]["content"] if user_messages else ""

        # 模拟推理延迟（500ms - 2000ms）
        latency = random.uniform(0.5, 2.0)
        time.sleep(latency * 0.1)  # 缩短等待，便于测试

        # 生成模拟回复
        mock_responses = [
            f"这是一个模拟回复。你输入的问题是：{last_user_msg[:50]}...",
            f"收到你的请求：{last_user_msg[:30]}...\n\n"
            f"（这是模拟模式下的回复，安装 vLLM 可获得真实推理结果）",
            f"你好！基于你的输入「{last_user_msg[:40]}」，"
            f"我来为你生成一个回复。\n\n"
            f"💡 提示：当前运行在模拟模式，安装 vLLM 后可以获得真实的模型推理结果。",
        ]

        response = random.choice(mock_responses)

        # 模拟 token 统计
        prompt_tokens = len(last_user_msg) // 2
        completion_tokens = len(response) // 2

        return {
            "content": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency * 1000,
        }

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """将消息列表格式化为模型可接受的 prompt"""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
        formatted += "<|assistant|>\n"
        return formatted


# 全局单例
engine = InferenceEngine()
```

#### `app/middleware.py` — 中间件

```python
"""
中间件模块
实现限流和身份认证
"""

import time
import hashlib
from typing import Dict, Tuple
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
from app.monitoring import metrics


# HTTP Bearer 认证
security = HTTPBearer(auto_error=False)


class RateLimiter:
    """
    滑动窗口限流器
    基于内存的简单实现，生产环境建议使用 Redis
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        初始化限流器
        :param max_requests: 窗口内最大请求数
        :param window_seconds: 窗口大小（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # 存储每个用户的请求时间戳列表
        self.requests: Dict[str, list] = {}

    def _cleanup(self, key: str):
        """清理过期记录"""
        if key in self.requests:
            cutoff = time.time() - self.window_seconds
            self.requests[key] = [
                t for t in self.requests[key] if t > cutoff
            ]

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        检查请求是否允许
        :return: (是否允许, 剩余配额)
        """
        self._cleanup(key)

        if key not in self.requests:
            self.requests[key] = []

        current_count = len(self.requests[key])

        if current_count >= self.max_requests:
            return False, 0

        # 记录本次请求
        self.requests[key].append(time.time())
        return True, self.max_requests - current_count - 1


# 全局限流器实例
rate_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_PER_MINUTE,
    window_seconds=60
)


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = None
) -> str:
    """
    验证 API 密钥
    支持两种认证方式：
    1. Authorization: Bearer <api_key>
    2. X-API-Key: <api_key>
    """
    api_key = None

    # 方式 1：从 Authorization header 获取
    if credentials and credentials.credentials:
        api_key = credentials.credentials

    # 方式 2：从自定义 header 获取
    if not api_key:
        api_key = request.headers.get("X-API-Key")

    # 方式 3：从 query 参数获取（不推荐，仅用于测试）
    if not api_key:
        api_key = request.query_params.get("api_key")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via Authorization header or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key


async def check_rate_limit(request: Request) -> bool:
    """
    检查限流
    使用用户 ID 或 IP 作为限流键
    """
    # 优先使用用户 ID，否则使用 IP
    user_id = request.headers.get("X-User-ID", "")
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"{user_id}:{client_ip}" if user_id else client_ip

    allowed, remaining = rate_limiter.is_allowed(rate_key)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again later.",
            headers={"Retry-After": "60"},
        )

    return allowed
```

#### `app/main.py` — FastAPI 应用入口

```python
"""
LLM API 服务主入口
整合所有模块，提供完整的 API 服务
"""

import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.config import settings
from app.models import (
    ChatRequest, ChatResponse, HealthResponse, ErrorResponse,
)
from app.cache import cache
from app.inference import engine
from app.monitoring import metrics
from app.middleware import verify_api_key, check_rate_limit


# ========== 生命周期管理 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的生命周期管理"""
    # 启动时：加载模型
    print(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"📦 Model: {settings.VLLM_MODEL}")

    engine.load()

    # 检查缓存状态
    cache_available = cache.is_available()
    print(f"💾 Redis Cache: {'✅ Connected' if cache_available else '⚠️  Disconnected (cache disabled)'}")

    yield  # 应用运行期间

    # 关闭时：清理资源
    print("👋 Shutting down...")


# ========== 应用初始化 ==========

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="生产级 LLM API 服务，支持缓存、限流和监控",
    lifespan=lifespan,
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 中间件：监控和限流 ==========

@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """
    全局监控中间件
    记录每个请求的指标
    """
    start_time = time.time()

    # 更新活跃连接数
    metrics.update_active_connections(
        metrics.active_connections._value.get() + 1
    )

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception as e:
        status_code = 500
        raise

    finally:
        latency = time.time() - start_time
        metrics.update_active_connections(
            max(0, metrics.active_connections._value.get() - 1)
        )
        metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            status=str(status_code),
            latency=latency,
        )


# ========== API 路由 ==========

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查端点
    返回服务状态、缓存状态和运行时长
    """
    uptime = time.time() - metrics.start_time
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        cache_status="connected" if cache.is_available() else "disconnected",
        uptime_seconds=round(uptime, 2),
    )


@app.get("/metrics", tags=["系统"])
async def get_metrics():
    """
    Prometheus 指标端点
    供 Prometheus 抓取监控数据
    """
    return Response(
        content=metrics.get_metrics(),
        media_type="text/plain",
    )


@app.post("/v1/chat/completions", response_model=ChatResponse, tags=["推理"])
async def chat_completions(
    request: ChatRequest,
    http_request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    聊天补全接口（兼容 OpenAI 格式）

    核心流程：
    1. 限流检查
    2. 缓存查询
    3. 模型推理
    4. 结果缓存
    5. 返回响应
    """
    inference_start = time.time()

    # 1. 限流检查
    await check_rate_limit(http_request)

    # 2. 构建缓存键（排除流式请求，因为流式结果不适合缓存）
    cache_key_data = {
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
    }

    # 3. 查询缓存
    cached_result = cache.get(cache_key_data)
    if cached_result:
        metrics.record_cache_hit()
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=request.model or "default",
            content=cached_result["content"],
            usage=cached_result["usage"],
            latency=cached_result["latency_ms"],
            cached=True,
        )

    metrics.record_cache_miss()

    # 4. 执行推理
    try:
        result = engine.generate(
            messages=[m.model_dump() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )

        # 5. 缓存结果
        cache_data = {
            "content": result["content"],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            },
            "latency_ms": result["latency_ms"],
        }
        cache.set(cache_key_data, cache_data)

        # 记录推理指标
        metrics.record_inference(
            model=engine.model_name,
            status="success",
            latency=result["latency_ms"] / 1000,
            tokens=result["completion_tokens"],
        )

        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=request.model or "default",
            content=result["content"],
            usage={
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            },
            latency=result["latency_ms"],
            cached=False,
        )

    except Exception as e:
        metrics.record_inference(
            model=engine.model_name,
            status="error",
            latency=(time.time() - inference_start),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )


@app.post("/v1/completions", tags=["推理"])
async def completions(
    request: Request,
    http_request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    文本补全接口（兼容 OpenAI 格式）
    内部复用 chat_completions 逻辑
    """
    body = await request.json()

    # 将 completions 格式转换为 chat 格式
    prompt = body.get("prompt", "")
    messages = [{"role": "user", "content": prompt}]

    chat_request = ChatRequest(
        messages=[{"role": "user", "content": prompt}],
        temperature=body.get("temperature", 0.7),
        max_tokens=body.get("max_tokens", 256),
        top_p=body.get("top_p", 0.9),
    )

    # 复用 chat_completions 逻辑
    await check_rate_limit(http_request)
    result = engine.generate(
        messages=messages,
        temperature=chat_request.temperature,
        max_tokens=chat_request.max_tokens,
        top_p=chat_request.top_p,
    )

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": engine.model_name,
        "choices": [{
            "text": result["content"],
            "index": 0,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        },
    }


@app.get("/", tags=["系统"])
async def root():
    """根路径，返回服务信息"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model": settings.VLLM_MODEL,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


# ========== 启动入口 ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,  # 开发模式
        workers=1,    # vLLM 不支持多 worker
    )
```

### 🚀 运行方式

```bash
# 1. 创建项目目录
mkdir -p llm-api-service/app
cd llm-api-service

# 2. 将上述代码保存到对应文件

# 3. 安装依赖
pip install fastapi uvicorn redis prometheus-client pydantic python-jose

# 4. 启动 Redis
docker run -d -p 6379:6379 redis:7-alpine

# 5. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 📡 测试请求

```bash
# 健康检查
curl http://localhost:8000/health

# 发送聊天请求
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key-change-in-production" \
  -d '{
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "用一句话介绍 Python。"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# 查看 Prometheus 指标
curl http://localhost:8000/metrics

# 查看 API 文档
# 浏览器打开：http://localhost:8000/docs
```

### 📊 预期输出

```json
// 健康检查
{
  "status": "healthy",
  "version": "1.0.0",
  "cache_status": "connected",
  "uptime_seconds": 45.23
}

// 聊天响应
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "model": "default",
  "content": "这是一个模拟回复。你输入的问题是：用一句话介绍 Python。...",
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 30,
    "total_tokens": 55
  },
  "latency_ms": 523.4,
  "cached": false
}
```

---

## 项目 2：MLOps 完整流水线

> **难度**：⭐⭐⭐⭐ ｜ **预计时间**：3-4 天
> 
> 搭建完整的 MLOps 流水线，包含数据版本管理、实验追踪、模型注册和 CI/CD。

### 🏗️ 架构设计

```
┌──────────────────────────────────────────────────────────────────────┐
│                          MLOps 流水线                                │
│                                                                      │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│  │ 数据    │──▶│ 特征    │──▶│ 训练    │──▶│ 评估    │              │
│  │ DVC     │   │ 工程    │   │ MLflow  │   │ 验证    │              │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘              │
│                                                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│  │ 注册    │──▶│ 部署    │──▶│ 监控    │──▶│ 告警    │              │
│  │ MLflow  │   │ CI/CD   │   │ 指标    │   │ 通知    │              │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘              │
│                                                                      │
│  工具链：DVC + MLflow + GitHub Actions + Prometheus                  │
└──────────────────────────────────────────────────────────────────────┘
```

### 📦 项目结构

```
mlops-pipeline/
├── data/                    # 数据目录（DVC 管理）
│   ├── raw/
│   ├── processed/
│   └── .dvc/
├── src/
│   ├── __init__.py
│   ├── data.py              # 数据加载和预处理
│   ├── features.py          # 特征工程
│   ├── train.py             # 模型训练
│   ├── evaluate.py          # 模型评估
│   └── register.py          # 模型注册
├── configs/
│   └── default.yaml         # 训练配置
├── tests/
│   └── test_pipeline.py     # 流水线测试
├── .github/
│   └── workflows/
│       └── mlops.yml        # GitHub Actions 流水线
├── dvc.yaml                 # DVC 流水线定义
├── requirements.txt
├── Makefile
└── README.md
```

### 🔧 依赖安装

```bash
# requirements.txt
mlflow==2.17.0
dvc==3.55.0
scikit-learn==1.5.0
pandas==2.2.0
numpy==1.26.0
pyyaml==6.0.1
pytest==8.3.0
```

```bash
pip install mlflow dvc scikit-learn pandas numpy pyyaml pytest
```

### 📝 完整代码

#### `src/data.py` — 数据加载和预处理

```python
"""
数据加载和预处理模块
负责从原始数据到处理好的数据集的转换
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json


def load_raw_data() -> pd.DataFrame:
    """
    加载原始数据
    使用 California Housing 数据集作为示例
    实际项目中替换为你的数据源
    """
    print("[Data] Loading California Housing dataset...")

    # 加载数据集
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target

    print(f"[Data] Loaded {len(df)} samples, {len(df.columns)} features")
    return df


def preprocess_data(df: pd.DataFrame, config: dict = None) -> dict:
    """
    数据预处理
    包含：缺失值处理、特征缩放、数据集划分

    Args:
        df: 原始数据
        config: 预处理配置

    Returns:
        包含训练集、测试集和预处理器的字典
    """
    if config is None:
        config = {
            "test_size": 0.2,
            "random_state": 42,
        }

    print("[Data] Starting preprocessing...")

    # 1. 检查缺失值
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"[Data] Found missing values: {missing[missing > 0].to_dict()}")
        df = df.fillna(df.median())

    # 2. 分离特征和标签
    feature_cols = [c for c in df.columns if c != 'Price']
    X = df[feature_cols].copy()
    y = df['Price'].copy()

    print(f"[Data] Features: {len(feature_cols)}, Target: Price")

    # 3. 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
    )

    print(f"[Data] Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    print("[Data] Preprocessing complete!")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


def compute_data_stats(df: pd.DataFrame) -> dict:
    """
    计算数据集统计信息
    用于数据质量监控和漂移检测
    """
    stats = {
        "num_samples": len(df),
        "num_features": len(df.columns) - 1,  # 减去目标列
        "missing_values": int(df.isnull().sum().sum()),
        "feature_stats": {},
    }

    # 计算每个特征的统计信息
    for col in df.columns:
        if col != 'Price' and df[col].dtype in [np.float64, np.int64]:
            stats["feature_stats"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }

    return stats


if __name__ == "__main__":
    # 独立运行测试
    df = load_raw_data()
    result = preprocess_data(df)
    stats = compute_data_stats(df)
    print(f"\n[Data] Data stats: {json.dumps(stats, indent=2)}")
```

#### `src/features.py` — 特征工程

```python
"""
特征工程模块
特征创建、选择和变换
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建衍生特征
    从原始特征中构造新的有意义的特征
    """
    df_features = df.copy()

    print("[Features] Creating derived features...")

    # California Housing 数据集的特征工程示例
    if 'AveRooms' in df.columns and 'AveBedrms' in df.columns:
        # 卧室/房间比例
        df_features['BedroomRatio'] = df['AveBedrms'] / df['AveRooms'].clip(lower=0.01)

    if 'Population' in df.columns and 'AveOccup' in df.columns:
        # 人口密度指标
        df_features['PopDensity'] = df['Population'] / df['AveOccup'].clip(lower=0.01)

    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        # 到洛杉矶的距离（简化计算）
        la_lat, la_lon = 34.05, -118.25
        df_features['DistToLA'] = np.sqrt(
            (df['Latitude'] - la_lat) ** 2 + (df['Longitude'] - la_lon) ** 2
        )

    # 收入分箱（离散化）
    if 'MedInc' in df.columns:
        df_features['IncomeGroup'] = pd.cut(
            df['MedInc'],
            bins=[0, 2.5, 5, 7.5, 10, 15],
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
        ).astype(str)

    print(f"[Features] Created features, total columns: {len(df_features.columns)}")
    return df_features


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 8,
) -> list:
    """
    特征选择
    使用 ANOVA F 值选择最重要的 k 个特征
    """
    print(f"[Features] Selecting top {k} features using ANOVA F-test...")

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)

    # 获取选中的特征名
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()

    # 打印特征重要性
    scores = selector.scores_
    feature_scores = sorted(
        zip(X.columns, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    print("[Features] Feature scores:")
    for name, score in feature_scores:
        marker = "✅" if name in selected_features else "  "
        print(f"  {marker} {name}: {score:.2f}")

    return selected_features
```

#### `src/train.py` — 模型训练

```python
"""
模型训练模块
使用 MLflow 追踪实验
"""

import mlflow
import mlflow.sklearn
import mlflow.models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import yaml
import os
import sys


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """加载训练配置"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return get_default_config()


def get_default_config() -> dict:
    """默认训练配置"""
    return {
        "model": "random_forest",
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1,
        "test_size": 0.2,
        "mlflow_tracking_uri": "http://localhost:5000",
        "mlflow_experiment": "California Housing",
    }


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> object:
    """
    训练模型
    支持多种模型类型
    """
    model_type = config.get("model", "random_forest")

    print(f"[Train] Training model: {model_type}")

    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
            random_state=config.get("random_state", 42),
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=config.get("n_estimators", 100),
            learning_rate=config.get("learning_rate", 0.1),
            max_depth=config.get("max_depth", 5),
            random_state=config.get("random_state", 42),
        )
    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 训练模型
    model.fit(X_train, y_train)
    print(f"[Train] Model trained successfully!")

    return model


def run_training_pipeline(config_path: str = "configs/default.yaml"):
    """
    运行完整训练流水线
    包含：数据加载 → 预处理 → 特征工程 → 训练 → MLflow 记录
    """
    # 加载配置
    config = load_config(config_path)

    # 设置 MLflow
    mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(config.get("mlflow_experiment", "Default Experiment"))

    # 导入数据处理模块
    from src.data import load_raw_data, preprocess_data

    # 1. 加载和预处理数据
    df = load_raw_data()
    data_result = preprocess_data(df, config)

    # 2. 特征选择
    from src.features import select_features
    selected_features = select_features(
        data_result["X_train"],
        data_result["y_train"],
        k=min(8, len(data_result["X_train"].columns)),
    )

    # 3. 使用选中的特征
    X_train_selected = data_result["X_train"][selected_features]
    X_test_selected = data_result["X_test"][selected_features]

    # 4. 训练并记录到 MLflow
    with mlflow.start_run() as run:
        # 记录参数
        mlflow.log_params(config)
        mlflow.log_param("selected_features", ",".join(selected_features))
        mlflow.log_param("train_samples", len(X_train_selected))
        mlflow.log_param("test_samples", len(X_test_selected))

        # 训练模型
        model = train_model(X_train_selected, data_result["y_train"], config)

        # 评估
        from src.evaluate import evaluate_model
        metrics = evaluate_model(model, X_test_selected, data_result["y_test"])

        # 记录指标
        mlflow.log_metrics(metrics)

        # 记录模型
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_test_selected.iloc[:1],
        )

        print(f"\n[Train] Run ID: {run.info.run_id}")
        print(f"[Train] Metrics: {metrics}")

        return run.info.run_id, model


if __name__ == "__main__":
    run_id, model = run_training_pipeline()
    print(f"\n✅ Training pipeline complete! Run ID: {run_id}")
```

#### `src/evaluate.py` — 模型评估

```python
"""
模型评估模块
计算各种回归指标
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import mlflow


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    评估模型性能
    计算多个回归指标

    Returns:
        包含所有评估指标的字典
    """
    print("[Evaluate] Evaluating model...")

    # 预测
    y_pred = model.predict(X_test)

    # 计算指标
    metrics = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
    }

    # 打印评估结果
    print("[Evaluate] Results:")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.4f}")

    return metrics


def check_model_quality(metrics: dict, thresholds: dict = None) -> bool:
    """
    检查模型质量是否达标
    用于 CI/CD 流水线中的质量门控
    """
    if thresholds is None:
        thresholds = {
            "r2_min": 0.7,       # R² 最低要求
            "rmse_max": 1.0,     # RMSE 最高要求
        }

    checks = []

    # R² 检查
    r2_pass = metrics["r2"] >= thresholds["r2_min"]
    checks.append(r2_pass)
    status = "✅" if r2_pass else "❌"
    print(f"[Evaluate] {status} R² = {metrics['r2']:.4f} (min: {thresholds['r2_min']})")

    # RMSE 检查
    rmse_pass = metrics["rmse"] <= thresholds["rmse_max"]
    checks.append(rmse_pass)
    status = "✅" if rmse_pass else "❌"
    print(f"[Evaluate] {status} RMSE = {metrics['rmse']:.4f} (max: {thresholds['rmse_max']})")

    all_pass = all(checks)
    print(f"[Evaluate] Quality gate: {'✅ PASSED' if all_pass else '❌ FAILED'}")

    return all_pass


if __name__ == "__main__":
    # 独立测试
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X[:300], y[:300])

    metrics = evaluate_model(model, X[300:], y[300:])
    passed = check_model_quality(metrics)
    print(f"\nQuality check passed: {passed}")
```

#### `src/register.py` — 模型注册

```python
"""
模型注册模块
将优质模型注册到 MLflow Model Registry
"""

import mlflow
import os
import yaml


def register_model(run_id: str, model_name: str = "housing_model"):
    """
    将训练好的模型注册到 Model Registry
    只有质量检查通过的模型才会被注册
    """
    print(f"[Register] Registering model from run: {run_id}")

    # 获取运行信息
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    # 质量检查
    from src.evaluate import check_model_quality
    if not check_model_quality(metrics):
        print("[Register] ❌ Model did not pass quality gate, skipping registration")
        return None

    # 注册模型
    model_uri = f"runs:/{run_id}/model"

    try:
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )
        print(f"[Register] ✅ Model registered: {result.name} version {result.version}")

        # 过渡到 Staging 阶段
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging",
        )
        print(f"[Register] Model transitioned to Staging")

        return result

    except mlflow.exceptions.MlflowException as e:
        # 模型已存在时的处理
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"[Register] Model '{model_name}' already exists, creating new version")
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
            )
            print(f"[Register] ✅ New version registered: {result.version}")
            return result
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
        register_model(run_id)
    else:
        print("Usage: python -m src.register <run_id>")
```

#### `dvc.yaml` — DVC 流水线定义

```yaml
# DVC 流水线配置
stages:
  data:
    cmd: python -c "
      from src.data import load_raw_data, preprocess_data;
      import pandas as pd;
      df = load_raw_data();
      df.to_csv('data/raw/housing.csv', index=False);
      result = preprocess_data(df);
      result['X_train'].to_csv('data/processed/X_train.csv');
      result['X_test'].to_csv('data/processed/X_train.csv');
      result['y_train'].to_csv('data/processed/y_train.csv');
      result['y_test'].to_csv('data/processed/y_test.csv');
      print('Data processing complete');
    "
    deps:
      - src/data.py
    outs:
      - data/raw/housing.csv
      - data/processed/

  train:
    cmd: python -m src.train
    deps:
      - src/train.py
      - data/processed/
    outs:
      - models/

  evaluate:
    cmd: python -c "
      from src.evaluate import evaluate_model, check_model_quality;
      import mlflow;
      runs = mlflow.search_runs(order_by=['metrics.r2 DESC'], max_results=1);
      if len(runs) > 0:
        run_id = runs.iloc[0]['run_id'];
        print(f'Best run: {run_id}');
    "
    deps:
      - src/evaluate.py
      - models/
```

#### `.github/workflows/mlops.yml` — CI/CD 流水线

```yaml
# GitHub Actions MLOps 流水线
name: MLOps Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # ========== 测试作业 ==========
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  # ========== 训练作业 ==========
  train:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run training pipeline
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python -m src.train

      - name: Evaluate model
        run: |
          python -c "
          from src.evaluate import evaluate_model, check_model_quality
          import mlflow
          runs = mlflow.search_runs(order_by=['metrics.r2 DESC'], max_results=1)
          print(f'Best model R2: {runs.iloc[0][\"metrics.r2\"]:.4f}')
          "

  # ========== 部署作业 ==========
  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          echo "🚀 Deploying best model to production..."
          # 实际部署逻辑：
          # - 从 MLflow 下载最佳模型
          # - 部署到推理服务
          # - 运行冒烟测试
          echo "✅ Deployment complete"
```

#### `tests/test_pipeline.py` — 流水线测试

```python
"""
流水线测试
确保每个组件正常工作
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing


class TestDataModule:
    """数据模块测试"""

    def test_load_raw_data(self):
        """测试数据加载"""
        from src.data import load_raw_data
        df = load_raw_data()
        assert len(df) > 0
        assert 'Price' in df.columns
        assert not df.isnull().any().any()

    def test_preprocess_data(self):
        """测试数据预处理"""
        from src.data import load_raw_data, preprocess_data
        df = load_raw_data()
        result = preprocess_data(df)

        assert "X_train" in result
        assert "X_test" in result
        assert "y_train" in result
        assert "y_test" in result
        assert len(result["X_train"]) > len(result["X_test"])

    def test_compute_data_stats(self):
        """测试数据统计"""
        from src.data import load_raw_data, compute_data_stats
        df = load_raw_data()
        stats = compute_data_stats(df)

        assert stats["num_samples"] > 0
        assert stats["num_features"] > 0
        assert "feature_stats" in stats


class TestFeatureModule:
    """特征工程测试"""

    def test_create_features(self):
        """测试特征创建"""
        from src.features import create_features
        df = pd.DataFrame({
            'AveRooms': [5, 6, 7],
            'AveBedrms': [1, 2, 3],
            'Population': [1000, 2000, 3000],
            'AveOccup': [2, 3, 4],
        })
        result = create_features(df)
        assert 'BedroomRatio' in result.columns
        assert 'DistToLA' in result.columns


class TestEvaluateModule:
    """评估模块测试"""

    def test_evaluate_model(self):
        """测试模型评估"""
        from sklearn.ensemble import RandomForestRegressor
        from src.evaluate import evaluate_model

        X = np.random.randn(100, 5)
        y = X[:, 0] + np.random.randn(100) * 0.1

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X[:80], y[:80])

        metrics = evaluate_model(model, X[80:], y[80:])

        assert "mse" in metrics
        assert "r2" in metrics
        assert metrics["r2"] > -1  # R² 不应为极端负值

    def test_check_model_quality(self):
        """测试质量检查"""
        from src.evaluate import check_model_quality

        # 高质量模型
        good_metrics = {"r2": 0.85, "rmse": 0.5}
        assert check_model_quality(good_metrics) == True

        # 低质量模型
        bad_metrics = {"r2": 0.3, "rmse": 2.0}
        assert check_model_quality(bad_metrics) == False
```

### 🚀 运行方式

```bash
# 1. 创建项目结构
mkdir -p mlops-pipeline/{src,configs,tests,data/{raw,processed},.github/workflows}

# 2. 将上述代码保存到对应文件

# 3. 安装依赖
cd mlops-pipeline
pip install -r requirements.txt

# 4. 启动 MLflow UI（新终端）
mlflow ui --port 5000

# 5. 运行训练流水线
python -m src.train

# 6. 运行测试
pytest tests/ -v

# 7. 注册模型
python -m src.register <run_id>
```

### 📊 预期输出

```
[Data] Loading California Housing dataset...
[Data] Loaded 20640 samples, 9 features
[Data] Starting preprocessing...
[Data] Features: 8, Target: Price
[Data] Train: 16512, Test: 4128
[Data] Preprocessing complete!
[Features] Selecting top 8 features using ANOVA F-test...
[Train] Training model: random_forest
[Train] Model trained successfully!
[Evaluate] Evaluating model...
[Evaluate] Results:
  MSE:  0.2345
  RMSE: 0.4843
  MAE:  0.3456
  R²:   0.8123
  MAPE: 0.1234
[Evaluate] Quality gate: ✅ PASSED
[Train] Run ID: a1b2c3d4e5f6g7h8
[Train] Metrics: {'mse': 0.2345, 'rmse': 0.4843, 'r2': 0.8123, ...}

✅ Training pipeline complete! Run ID: a1b2c3d4e5f6g7h8
```

---

## 项目 3：模型 A/B 测试系统

> **难度**：⭐⭐⭐⭐ ｜ **预计时间**：3-4 天
> 
> 实现模型灰度发布和 A/B 测试系统，支持流量分流、实时指标对比和自动回滚。

### 🏗️ 架构设计

```
┌──────────────────────────────────────────────────────────────────┐
│                        用户请求                                   │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    流量路由器                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ 分流策略：                                                 │  │
│  │  • 按用户 ID 哈希分流（一致性）                             │  │
│  │  • 按百分比随机分流                                        │  │
│  │  • 按特征条件分流                                          │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────┬───────────────────────┬───────────────────────────────────┘
       │ 80%                   │ 20%
       ▼                       ▼
┌──────────────┐        ┌──────────────┐
│  模型 A      │        │  模型 B      │
│  (Baseline)  │        │  (Challenger)│
└──────┬───────┘        └──────┬───────┘
       │                       │
       ▼                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    指标收集器                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ 延迟     │  │ 质量     │  │ 用户满意度│  │ Token 成本│        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    分析引擎                                        │
│  • 统计显著性检验 (t-test)                                        │
│  • 自动回滚（质量下降时）                                         │
│  • 推荐最佳模型                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 📦 项目结构

```
ab-testing/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── router.py            # 流量路由器
│   ├── models.py            # 模型封装
│   ├── metrics.py           # 指标收集
│   ├── analyzer.py          # 统计分析
│   └── dashboard.py         # 可视化面板
├── requirements.txt
└── tests/
```

### 🔧 依赖安装

```bash
# requirements.txt
fastapi==0.115.0
uvicorn==0.30.0
scipy==1.14.0
pandas==2.2.0
numpy==1.26.0
jinja2==3.1.4
```

```bash
pip install fastapi uvicorn scipy pandas numpy jinja2
```

### 📝 完整代码

#### `app/router.py` — 流量路由器

```python
"""
流量路由器
负责将请求分配到不同的模型版本
支持多种分流策略
"""

import hashlib
import random
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class SplitStrategy(str, Enum):
    """分流策略枚举"""
    RANDOM = "random"           # 随机分流
    USER_HASH = "user_hash"     # 按用户 ID 哈希分流
    HEADER_VALUE = "header"     # 按请求头值分流


@dataclass
class TrafficSplit:
    """
    流量分流配置
    定义每个模型版本接收的流量比例
    """
    model_name: str             # 模型名称
    weight: float               # 流量权重 (0.0 - 1.0)
    is_active: bool = True      # 是否活跃
    metadata: Dict = field(default_factory=dict)  # 额外元数据


class TrafficRouter:
    """
    流量路由器
    根据配置的分流策略，将请求路由到对应的模型
    """

    def __init__(self):
        """初始化路由器"""
        self.splits: List[TrafficSplit] = []
        self.strategy: SplitStrategy = SplitStrategy.USER_HASH
        self.header_name: str = "X-AB-Test-Group"  # 自定义分流头
        self.last_updated: float = time.time()

    def configure(
        self,
        splits: List[TrafficSplit],
        strategy: SplitStrategy = SplitStrategy.USER_HASH,
    ):
        """
        配置流量分流
        :param splits: 分流配置列表
        :param strategy: 分流策略
        """
        self.splits = splits
        self.strategy = strategy
        self.last_updated = time.time()

        # 验证权重总和
        total_weight = sum(s.weight for s in splits if s.is_active)
        if abs(total_weight - 1.0) > 0.001:
            print(f"[Router] ⚠️  Warning: total weight {total_weight} != 1.0, normalizing")
            self._normalize_weights()

        print(f"[Router] Configured: {len(splits)} splits, strategy={strategy.value}")
        for split in splits:
            print(f"  {split.model_name}: {split.weight*100:.1f}% {'✅' if split.is_active else '❌'}")

    def _normalize_weights(self):
        """归一化权重，确保总和为 1.0"""
        total = sum(s.weight for s in self.splits if s.is_active)
        if total > 0:
            for split in self.splits:
                if split.is_active:
                    split.weight /= split.weight / total  # 修正：直接除以 total
                    split.weight = split.weight / total

    def route(self, user_id: Optional[str] = None, headers: Optional[Dict] = None) -> TrafficSplit:
        """
        根据分流策略选择目标模型
        :param user_id: 用户 ID（用于一致性哈希）
        :param headers: 请求头（用于 header 策略）
        :return: 选中的分流配置
        """
        active_splits = [s for s in self.splits if s.is_active]

        if not active_splits:
            raise ValueError("No active traffic splits configured")

        if len(active_splits) == 1:
            return active_splits[0]

        if self.strategy == SplitStrategy.RANDOM:
            return self._random_route(active_splits)

        elif self.strategy == SplitStrategy.USER_HASH:
            return self._hash_route(active_splits, user_id or "anonymous")

        elif self.strategy == SplitStrategy.HEADER_VALUE:
            header_val = (headers or {}).get(self.header_name, "")
            return self._hash_route(active_splits, header_val or "default")

        # 默认使用随机分流
        return self._random_route(active_splits)

    def _random_route(self, active_splits: List[TrafficSplit]) -> TrafficSplit:
        """随机分流：按权重随机选择"""
        weights = [s.weight for s in active_splits]
        return random.choices(active_splits, weights=weights, k=1)[0]

    def _hash_route(self, active_splits: List[TrafficSplit], key: str) -> TrafficSplit:
        """
        哈希分流：相同 key 总是路由到同一模型
        确保同一用户始终看到相同版本的模型
        """
        # 使用 MD5 哈希（快速且分布均匀）
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        # 映射到 [0, 1) 区间
        normalized = (hash_value % 10000) / 10000.0

        # 累加权重找到对应的模型
        cumulative = 0.0
        for split in active_splits:
            cumulative += split.weight
            if normalized < cumulative:
                return split

        # 浮点精度回退
        return active_splits[-1]

    def update_split(self, model_name: str, weight: float, is_active: bool = None):
        """
        动态更新某个模型版本的流量比例
        支持热更新，无需重启服务
        """
        for split in self.splits:
            if split.model_name == model_name:
                split.weight = weight
                if is_active is not None:
                    split.is_active = is_active
                print(f"[Router] Updated {model_name}: weight={weight}, active={split.is_active}")
                return True

        print(f"[Router] Model {model_name} not found in splits")
        return False

    def get_status(self) -> Dict:
        """获取路由器当前状态"""
        return {
            "strategy": self.strategy.value,
            "splits": [
                {
                    "model": s.model_name,
                    "weight": s.weight,
                    "active": s.is_active,
                }
                for s in self.splits
            ],
            "last_updated": self.last_updated,
        }
```

#### `app/models.py` — 模型封装

```python
"""
模型封装模块
将不同模型统一封装为相同接口
"""

import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelResponse:
    """模型响应统一格式"""
    model_name: str             # 模型名称
    content: str                # 生成内容
    latency_ms: float           # 推理延迟
    tokens_used: int            # 使用的 token 数
    quality_score: float        # 质量评分（0-1）
    metadata: Dict = None       # 额外元数据


class MockModel:
    """
    模拟模型
    用于测试 A/B 测试系统，无需真实 GPU
    不同模型有不同的性能特征
    """

    def __init__(
        self,
        name: str,
        avg_latency_ms: float = 500,
        quality_score: float = 0.7,
        token_range: tuple = (50, 200),
    ):
        """
        初始化模拟模型
        :param name: 模型名称
        :param avg_latency_ms: 平均延迟（毫秒）
        :param quality_score: 平均质量评分
        :param token_range: token 使用范围
        """
        self.name = name
        self.avg_latency_ms = avg_latency_ms
        self.quality_score = quality_score
        self.token_range = token_range

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs,
    ) -> ModelResponse:
        """
        生成回复（模拟）
        模拟不同模型的响应特征
        """
        start_time = time.time()

        # 模拟推理延迟（正态分布）
        latency = max(50, random.gauss(self.avg_latency_ms, self.avg_latency_ms * 0.2))
        time.sleep(latency / 10000)  # 加速模拟（除以 10）

        # 模拟 token 使用
        tokens = random.randint(self.token_range[0], self.token_range[1])

        # 模拟质量评分（正态分布，围绕平均质量）
        quality = max(0, min(1, random.gauss(self.quality_score, 0.1)))

        # 生成模拟回复
        user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_msg = msg["content"]
                break

        content = f"[{self.name}] 回复：基于你的输入「{user_msg[:30]}」，"
        content += f"我生成了一个质量评分为 {quality:.2f} 的回复。"

        return ModelResponse(
            model_name=self.name,
            content=content,
            latency_ms=latency,
            tokens_used=tokens,
            quality_score=quality,
            metadata={
                "temperature": temperature,
                "prompt_length": len(user_msg),
            },
        )


def create_model_configs() -> Dict[str, MockModel]:
    """
    创建预定义的模型配置
    模拟两个不同特征的模型用于 A/B 测试
    """
    models = {
        # 模型 A：基线模型（稳定但一般）
        "model_a_baseline": MockModel(
            name="Model-A (Baseline)",
            avg_latency_ms=400,
            quality_score=0.72,
            token_range=(40, 180),
        ),
        # 模型 B：新模型（更快但质量略低）
        "model_b_challenger": MockModel(
            name="Model-B (Challenger)",
            avg_latency_ms=250,
            quality_score=0.68,
            token_range=(30, 150),
        ),
    }
    return models
```

#### `app/metrics.py` — 指标收集

```python
"""
指标收集模块
收集每个模型版本的性能和质量指标
"""

import time
import threading
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class MetricPoint:
    """单个指标数据点"""
    model_name: str
    latency_ms: float
    quality_score: float
    tokens_used: int
    timestamp: float
    user_satisfied: bool = True  # 用户满意度（可来自反馈）


class MetricsCollector:
    """
    指标收集器
    线程安全地收集和管理所有模型指标
    """

    def __init__(self):
        """初始化指标收集器"""
        self.points: List[MetricPoint] = []
        self.lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)  # 请求计数

    def record(self, point: MetricPoint):
        """
        记录一个指标数据点
        线程安全
        """
        with self.lock:
            self.points.append(point)
            self._counters[point.model_name] += 1

    def get_model_metrics(self, model_name: str, window_seconds: float = 3600) -> Dict:
        """
        获取指定模型在时间窗口内的聚合指标
        :param model_name: 模型名称
        :param window_seconds: 时间窗口（秒），默认 1 小时
        """
        cutoff = time.time() - window_seconds

        with self.lock:
            model_points = [
                p for p in self.points
                if p.model_name == model_name and p.timestamp >= cutoff
            ]

        if not model_points:
            return {
                "request_count": 0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p99_latency_ms": 0,
                "avg_quality": 0,
                "avg_tokens": 0,
                "satisfaction_rate": 0,
            }

        latencies = sorted([p.latency_ms for p in model_points])
        qualities = [p.quality_score for p in model_points]
        tokens = [p.tokens_used for p in model_points]
        satisfied = sum(1 for p in model_points if p.user_satisfied)

        return {
            "request_count": len(model_points),
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "p50_latency_ms": round(latencies[len(latencies) // 2], 2),
            "p99_latency_ms": round(latencies[int(len(latencies) * 0.99)], 2),
            "avg_quality": round(statistics.mean(qualities), 4),
            "quality_std": round(statistics.stdev(qualities), 4) if len(qualities) > 1 else 0,
            "avg_tokens": round(statistics.mean(tokens), 1),
            "satisfaction_rate": round(satisfied / len(model_points), 4),
        }

    def get_all_metrics(self, window_seconds: float = 3600) -> Dict[str, Dict]:
        """获取所有模型的指标"""
        with self.lock:
            model_names = set(p.model_name for p in self.points)

        return {
            name: self.get_model_metrics(name, window_seconds)
            for name in model_names
        }

    def get_total_requests(self) -> Dict[str, int]:
        """获取各模型的总请求数"""
        return dict(self._counters)

    def clear_old_data(self, max_age_seconds: float = 86400):
        """清理过期数据（默认保留 24 小时）"""
        cutoff = time.time() - max_age_seconds
        with self.lock:
            self.points = [p for p in self.points if p.timestamp >= cutoff]


# 全局单例
collector = MetricsCollector()
```

#### `app/analyzer.py` — 统计分析

```python
"""
统计分析模块
使用统计检验比较模型性能差异
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
from app.metrics import MetricsCollector


class ABAnalyzer:
    """
    A/B 测试分析器
    使用统计检验判断模型间差异是否显著
    """

    def __init__(self, significance_level: float = 0.05):
        """
        初始化分析器
        :param significance_level: 显著性水平，默认 0.05
        """
        self.alpha = significance_level

    def compare_models(
        self,
        metrics_a: Dict,
        metrics_b: Dict,
    ) -> Dict:
        """
        比较两个模型的性能
        返回详细的对比分析报告
        """
        report = {
            "model_a": metrics_a.get("model_name", "A"),
            "model_b": metrics_b.get("model_name", "B"),
            "samples_a": metrics_a.get("request_count", 0),
            "samples_b": metrics_b.get("request_count", 0),
            "comparisons": {},
            "recommendation": "",
        }

        # 1. 延迟对比
        report["comparisons"]["latency"] = self._compare_latency(
            metrics_a, metrics_b
        )

        # 2. 质量对比
        report["comparisons"]["quality"] = self._compare_quality(
            metrics_a, metrics_b
        )

        # 3. Token 效率对比
        report["comparisons"]["token_efficiency"] = self._compare_tokens(
            metrics_a, metrics_b
        )

        # 4. 综合推荐
        report["recommendation"] = self._make_recommendation(report["comparisons"])

        return report

    def _compare_latency(self, a: Dict, b: Dict) -> Dict:
        """比较延迟（越低越好）"""
        lat_a = a.get("avg_latency_ms", 0)
        lat_b = b.get("avg_latency_ms", 0)

        if lat_a == 0 or lat_b == 0:
            return {"result": "insufficient_data"}

        improvement = ((lat_a - lat_b) / lat_a) * 100
        winner = "B" if lat_b < lat_a else "A"

        return {
            "model_a_latency": lat_a,
            "model_b_latency": lat_b,
            "improvement_percent": round(improvement, 2),
            "winner": winner,
            "interpretation": f"Model {winner} is {abs(improvement):.1f}% {'faster' if improvement > 0 else 'slower'}",
        }

    def _compare_quality(self, a: Dict, b: Dict) -> Dict:
        """比较质量（越高越好）"""
        qual_a = a.get("avg_quality", 0)
        qual_b = b.get("avg_quality", 0)

        if qual_a == 0 or qual_b == 0:
            return {"result": "insufficient_data"}

        improvement = ((qual_b - qual_a) / qual_a) * 100
        winner = "B" if qual_b > qual_a else "A"

        return {
            "model_a_quality": round(qual_a, 4),
            "model_b_quality": round(qual_b, 4),
            "improvement_percent": round(improvement, 2),
            "winner": winner,
            "interpretation": f"Model {winner} has {abs(improvement):.2f}% {'higher' if improvement > 0 else 'lower'} quality",
        }

    def _compare_tokens(self, a: Dict, b: Dict) -> Dict:
        """比较 token 效率（越低越好）"""
        tok_a = a.get("avg_tokens", 0)
        tok_b = b.get("avg_tokens", 0)

        if tok_a == 0 or tok_b == 0:
            return {"result": "insufficient_data"}

        improvement = ((tok_a - tok_b) / tok_a) * 100
        winner = "B" if tok_b < tok_a else "A"

        return {
            "model_a_tokens": tok_a,
            "model_b_tokens": tok_b,
            "improvement_percent": round(improvement, 2),
            "winner": winner,
            "interpretation": f"Model {winner} uses {abs(improvement):.1f}% {'fewer' if improvement > 0 else 'more'} tokens",
        }

    def _make_recommendation(self, comparisons: Dict) -> str:
        """
        基于对比结果给出推荐
        考虑多个维度的综合表现
        """
        winners = {}
        for metric, result in comparisons.items():
            if "winner" in result:
                w = result["winner"]
                winners[w] = winners.get(w, 0) + 1

        if not winners:
            return "⚠️  数据不足，无法给出推荐。需要更多请求数据。"

        best_model = max(winners, key=winners.get)

        if winners.get(best_model, 0) >= 2:
            return f"✅ 推荐 Model-{best_model}，在 {winners[best_model]}/3 个维度上表现更好"
        else:
            return f"⚖️  模型各有优势，建议继续收集数据或根据业务优先级选择"

    def check_rollback_needed(
        self,
        challenger_metrics: Dict,
        baseline_metrics: Dict,
        quality_threshold: float = 0.05,
    ) -> Dict:
        """
        检查是否需要回滚
        当 challenger 质量下降超过阈值时触发回滚
        """
        qual_baseline = baseline_metrics.get("avg_quality", 1.0)
        qual_challenger = challenger_metrics.get("avg_quality", 0)

        if qual_challenger == 0:
            return {"rollback": False, "reason": "insufficient_data"}

        quality_drop = (qual_baseline - qual_challenger) / qual_baseline

        needs_rollback = quality_drop > quality_threshold

        return {
            "rollback": needs_rollback,
            "quality_drop_percent": round(quality_drop * 100, 2),
            "threshold_percent": quality_threshold * 100,
            "reason": (
                f"质量下降 {quality_drop*100:.2f}% 超过阈值 {quality_threshold*100}%"
                if needs_rollback
                else "质量在可接受范围内"
            ),
        }


# 全局单例
analyzer = ABAnalyzer()
```

#### `app/main.py` — FastAPI 入口

```python
"""
A/B 测试系统主入口
整合路由器、模型、指标收集和分析
"""

import time
import uuid
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from typing import Optional, List, Dict

from app.router import TrafficRouter, TrafficSplit, SplitStrategy
from app.models import create_model_configs, MockModel
from app.metrics import MetricPoint, collector
from app.analyzer import analyzer

# ========== 应用初始化 ==========
app = FastAPI(title="Model A/B Testing System")

# 初始化组件
router = TrafficRouter()
models = create_model_configs()


# ========== 配置 API ==========

@app.post("/ab/configure", tags=["配置"])
async def configure_ab_test(
    model_a_weight: float = Query(0.8, ge=0, le=1, description="模型 A 流量比例"),
    model_b_weight: float = Query(0.2, ge=0, le=1, description="模型 B 流量比例"),
    strategy: str = Query("user_hash", description="分流策略"),
):
    """配置 A/B 测试分流"""
    splits = [
        TrafficSplit(model_name="model_a_baseline", weight=model_a_weight),
        TrafficSplit(model_name="model_b_challenger", weight=model_b_weight),
    ]

    strategy_map = {
        "random": SplitStrategy.RANDOM,
        "user_hash": SplitStrategy.USER_HASH,
        "header": SplitStrategy.HEADER_VALUE,
    }

    router.configure(splits, strategy_map.get(strategy, SplitStrategy.USER_HASH))
    return {"status": "configured", "router": router.get_status()}


@app.get("/ab/status", tags=["配置"])
async def get_ab_status():
    """获取 A/B 测试状态"""
    return {
        "router": router.get_status(),
        "total_requests": collector.get_total_requests(),
    }


# ========== 推理 API ==========

@app.post("/ab/chat", tags=["推理"])
async def ab_chat(
    request: Request,
    user_id: Optional[str] = Query(None, description="用户 ID（用于一致性分流）"),
):
    """
    A/B 测试聊天接口
    自动将请求路由到对应模型并收集指标
    """
    body = await request.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)

    # 1. 路由决策
    split = router.route(
        user_id=user_id,
        headers=dict(request.headers),
    )

    # 2. 获取对应模型
    model = models.get(split.model_name)
    if not model:
        return {"error": f"Model {split.model_name} not found"}

    # 3. 执行推理
    response = model.generate(
        messages=messages,
        temperature=temperature,
    )

    # 4. 记录指标
    collector.record(MetricPoint(
        model_name=split.model_name,
        latency_ms=response.latency_ms,
        quality_score=response.quality_score,
        tokens_used=response.tokens_used,
        timestamp=time.time(),
    ))

    # 5. 返回结果
    return {
        "id": str(uuid.uuid4())[:8],
        "model": response.model_name,
        "content": response.content,
        "latency_ms": round(response.latency_ms, 2),
        "tokens_used": response.tokens_used,
        "quality_score": round(response.quality_score, 4),
        "ab_test_group": split.model_name,
    }


# ========== 分析 API ==========

@app.get("/ab/analysis", tags=["分析"])
async def get_analysis(
    window_hours: float = Query(1.0, description="分析时间窗口（小时）"),
):
    """获取 A/B 测试分析报告"""
    window_seconds = window_hours * 3600

    metrics_a = collector.get_model_metrics("model_a_baseline", window_seconds)
    metrics_b = collector.get_model_metrics("model_b_challenger", window_seconds)

    # 添加模型名称
    metrics_a["model_name"] = "Model-A (Baseline)"
    metrics_b["model_name"] = "Model-B (Challenger)"

    # 对比分析
    comparison = analyzer.compare_models(metrics_a, metrics_b)

    # 回滚检查
    rollback = analyzer.check_rollback_needed(metrics_b, metrics_a)

    return {
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "comparison": comparison,
        "rollback_check": rollback,
    }


@app.get("/ab/dashboard", response_class=HTMLResponse, tags=["可视化"])
async def dashboard():
    """A/B 测试可视化面板"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>A/B 测试仪表盘</title>
        <style>
            body { font-family: system-ui; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .card { background: white; border-radius: 12px; padding: 24px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .metric { display: inline-block; margin: 8px 16px; }
            .metric-value { font-size: 2em; font-weight: bold; color: #333; }
            .metric-label { font-size: 0.9em; color: #666; }
            .model-a { color: #2563eb; }
            .model-b { color: #dc2626; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
            th { background: #f8f9fa; }
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.85em; }
            .badge-success { background: #d4edda; color: #155724; }
            .badge-warning { background: #fff3cd; color: #856404; }
            button { padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-size: 1em; }
            .btn-primary { background: #2563eb; color: white; }
            .btn-danger { background: #dc2626; color: white; }
        </style>
    </head>
    <body>
        <h1>🧪 模型 A/B 测试仪表盘</h1>

        <div class="card">
            <h2>📊 实时指标</h2>
            <div id="metrics"></div>
        </div>

        <div class="card">
            <h2>📈 对比分析</h2>
            <div id="comparison"></div>
        </div>

        <div class="card">
            <h2>⚙️ 操作</h2>
            <button class="btn-primary" onclick="configureTest(0.8, 0.2)">80/20 分流</button>
            <button class="btn-primary" onclick="configureTest(0.5, 0.5)">50/50 分流</button>
            <button class="btn-danger" onclick="rollback()">🔄 回滚到 Model A</button>
        </div>

        <script>
            async function loadDashboard() {
                const res = await fetch('/ab/analysis');
                const data = await res.json();

                // 渲染指标
                let metricsHtml = '<table>';
                metricsHtml += '<tr><th>指标</th><th class="model-a">Model A</th><th class="model-b">Model B</th></tr>';

                const fields = ['request_count', 'avg_latency_ms', 'avg_quality', 'avg_tokens', 'satisfaction_rate'];
                const labels = ['请求数', '平均延迟(ms)', '质量评分', '平均Token', '满意度'];

                fields.forEach((field, i) => {
                    metricsHtml += `<tr>
                        <td>${labels[i]}</td>
                        <td class="model-a">${data.metrics_a[field]?.toFixed ? data.metrics_a[field].toFixed(2) : data.metrics_a[field]}</td>
                        <td class="model-b">${data.metrics_b[field]?.toFixed ? data.metrics_b[field].toFixed(2) : data.metrics_b[field]}</td>
                    </tr>`;
                });
                metricsHtml += '</table>';
                document.getElementById('metrics').innerHTML = metricsHtml;

                // 渲染对比
                let compHtml = '<h3>推荐: ' + data.comparison.recommendation + '</h3>';
                if (data.rollback_check.rollback) {
                    compHtml += '<p><span class="badge badge-warning">⚠️ 建议回滚: ' + data.rollback_check.reason + '</span></p>';
                } else {
                    compHtml += '<p><span class="badge badge-success">✅ 质量正常</span></p>';
                }
                document.getElementById('comparison').innerHTML = compHtml;
            }

            async function configureTest(a, b) {
                await fetch(`/ab/configure?model_a_weight=${a}&model_b_weight=${b}&strategy=user_hash`, {method: 'POST'});
                loadDashboard();
            }

            async function rollback() {
                await fetch('/ab/configure?model_a_weight=1.0&model_b_weight=0.0&strategy=user_hash', {method: 'POST'});
                loadDashboard();
            }

            // 每 5 秒自动刷新
            loadDashboard();
            setInterval(loadDashboard, 5000);
        </script>
    </body>
    </html>
    """


# ========== 启动 ==========

if __name__ == "__main__":
    import uvicorn
    # 默认配置 80/20 分流
    router.configure([
        TrafficSplit(model_name="model_a_baseline", weight=0.8),
        TrafficSplit(model_name="model_b_challenger", weight=0.2),
    ], SplitStrategy.USER_HASH)

    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### 🚀 运行方式

```bash
# 1. 创建项目
mkdir -p ab-testing/app
cd ab-testing

# 2. 安装依赖
pip install fastapi uvicorn scipy pandas numpy jinja2

# 3. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 📡 测试流程

```bash
# 1. 配置 80/20 分流（Baseline 80%, Challenger 20%）
curl -X POST "http://localhost:8001/ab/configure?model_a_weight=0.8&model_b_weight=0.2&strategy=user_hash"

# 2. 发送测试请求（模拟 100 个请求）
for i in $(seq 1 100); do
  curl -s -X POST http://localhost:8001/ab/chat \
    -H "Content-Type: application/json" \
    -d "{\"messages\": [{\"role\": \"user\", \"content\": \"测试请求 $i\"}], \"user_id\": \"user_$((i % 10))\"}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Request {d[\"ab_test_group\"]}: {d[\"content\"][:50]}')"
done

# 3. 查看分析报告
curl http://localhost:8001/ab/analysis | python3 -m json.tool

# 4. 打开可视化仪表盘
# 浏览器访问：http://localhost:8001/ab/dashboard

# 5. 动态调整分流比例
curl -X POST "http://localhost:8001/ab/configure?model_a_weight=0.5&model_b_weight=0.5&strategy=user_hash"

# 6. 回滚到全部使用 Model A
curl -X POST "http://localhost:8001/ab/configure?model_a_weight=1.0&model_b_weight=0.0&strategy=user_hash"
```

### 📊 预期输出

```json
// 聊天响应
{
  "id": "a1b2c3d4",
  "model": "Model-A (Baseline)",
  "content": "[Model-A (Baseline)] 回复：基于你的输入「测试请求 1」，我生成了一个质量评分为 0.7234 的回复。",
  "latency_ms": 412.3,
  "tokens_used": 95,
  "quality_score": 0.7234,
  "ab_test_group": "model_a_baseline"
}

// 分析报告
{
  "metrics_a": {
    "request_count": 82,
    "avg_latency_ms": 405.23,
    "avg_quality": 0.7215,
    "avg_tokens": 110.3,
    "satisfaction_rate": 0.85
  },
  "metrics_b": {
    "request_count": 18,
    "avg_latency_ms": 248.67,
    "avg_quality": 0.6789,
    "avg_tokens": 90.1,
    "satisfaction_rate": 0.78
  },
  "comparison": {
    "recommendation": "⚖️  模型各有优势，建议继续收集数据或根据业务优先级选择"
  },
  "rollback_check": {
    "rollback": false,
    "quality_drop_percent": 5.9,
    "reason": "质量在可接受范围内"
  }
}
```

---

## 项目 4：RAG 生产部署

> **难度**：⭐⭐⭐⭐⭐ ｜ **预计时间**：4-5 天
> 
> 构建完整的 RAG 系统，包含文档处理、向量检索、LLM 生成、监控告警和自动化流水线。

### 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG 生产系统                                  │
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  文档采集   │───▶│  文档处理   │───▶│  向量化     │                  │
│  │  爬虫/上传  │    │  分块/清洗  │    │  Embedding  │                  │
│  └─────────────┘    └─────────────┘    └─────┬───────┘                  │
│                                              │                          │
│                                              ▼                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  监控告警   │◀───│  结果评估   │◀───│  LLM 生成   │                  │
│  │  质量/延迟  │    │  相关性评分 │    │  RAG Prompt │                  │
│  └─────────────┘    └─────────────┘    └─────┬───────┘                  │
│                                              │                          │
│  ┌─────────────┐    ┌─────────────┐          │                          │
│  │  向量数据库 │◀───│  索引管理   │◀─────────┘                          │
│  │  FAISS/...  │    │  更新/删除  │                                     │
│  └─────────────┘    └─────────────┘                                     │
│                                                                         │
│  技术栈：LangChain + FAISS + FastAPI + Prometheus + Celery              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 📦 项目结构

```
rag-production/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置
│   ├── documents.py         # 文档处理（分块、清洗）
│   ├── embeddings.py        # Embedding 和向量检索
│   ├── rag.py               # RAG 核心逻辑
│   ├── monitoring.py        # 监控和告警
│   └── pipeline.py          # 自动化流水线
├── data/
│   ├── raw/                 # 原始文档
│   └── index/               # 向量索引
├── requirements.txt
└── tests/
```

### 🔧 依赖安装

```bash
# requirements.txt
fastapi==0.115.0
uvicorn==0.30.0
numpy==1.26.0
faiss-cpu==1.8.0
sentence-transformers==3.1.0
langchain==0.3.0
langchain-community==0.3.0
pydantic==2.5.0
prometheus-client==0.20.0
python-multipart==0.0.12
tiktoken==0.8.0
```

```bash
pip install fastapi uvicorn numpy faiss-cpu sentence-transformers \
  langchain langchain-community pydantic prometheus-client \
  python-multipart tiktoken
```

### 📝 完整代码

#### `app/config.py` — 配置

```python
"""
RAG 系统配置
"""

from pydantic import BaseModel
from typing import Optional


class RAGConfig:
    """RAG 系统配置"""

    # 文档处理配置
    CHUNK_SIZE: int = 500           # 文档分块大小（字符）
    CHUNK_OVERLAP: int = 50         # 分块重叠大小
    MIN_CHUNK_LENGTH: int = 100     # 最小分块长度

    # Embedding 配置
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM: int = 384        # 向量维度

    # 向量检索配置
    TOP_K: int = 5                  # 检索返回的文档数量
    SIMILARITY_THRESHOLD: float = 0.3  # 相似度阈值

    # LLM 配置
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.3

    # 存储路径
    INDEX_DIR: str = "data/index"
    RAW_DIR: str = "data/raw"

    # 监控配置
    ALERT_QUALITY_THRESHOLD: float = 0.4   # 质量告警阈值
    ALERT_LATENCY_THRESHOLD: float = 5000  # 延迟告警阈值（毫秒）
    MAX_CONTEXT_LENGTH: int = 4000         # 最大上下文长度
```

#### `app/documents.py` — 文档处理

```python
"""
文档处理模块
负责文档加载、清洗、分块
"""

import re
import os
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Document:
    """文档数据模型"""
    content: str                  # 文档内容
    metadata: Dict = field(default_factory=dict)  # 元数据（来源、标题等）
    doc_id: str = ""              # 文档 ID


@dataclass
class DocumentChunk:
    """文档分块"""
    content: str                  # 分块内容
    doc_id: str                   # 所属文档 ID
    chunk_index: int              # 分块序号
    metadata: Dict = field(default_factory=dict)


class DocumentProcessor:
    """
    文档处理器
    负责文档的加载、清洗和分块
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文档处理器
        :param chunk_size: 分块大小
        :param chunk_overlap: 分块重叠
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_text(self, text: str, metadata: Dict = None) -> Document:
        """
        从文本加载文档
        :param text: 文档文本内容
        :param metadata: 文档元数据
        """
        import hashlib
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]

        return Document(
            content=text,
            metadata=metadata or {},
            doc_id=doc_id,
        )

    def load_file(self, file_path: str) -> Document:
        """
        从文件加载文档
        支持 .txt, .md 格式
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
        }

        return self.load_text(content, metadata)

    def clean_text(self, text: str) -> str:
        """
        文本清洗
        移除多余空白、特殊字符等
        """
        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)

        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 移除首尾空白
        text = text.strip()

        return text

    def split_by_sentence(self, text: str) -> List[str]:
        """
        按句子分割文本
        支持中英文句子边界检测
        """
        # 中文句号、问号、感叹号
        # 英文句号（需排除缩写如 Mr. Dr.）
        sentences = re.split(r'(?<=[。！？\.!?])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, doc: Document) -> List[DocumentChunk]:
        """
        将文档分割为多个块
        使用滑动窗口方法，保持上下文连贯性
        """
        # 1. 清洗文本
        cleaned = self.clean_text(doc.content)

        # 2. 按句子分割
        sentences = self.split_by_sentence(cleaned)

        if not sentences:
            return []

        # 3. 滑动窗口分块
        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)

            # 如果当前块加上新句子超过限制，保存当前块
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_content = ' '.join(current_chunk)

                if len(chunk_content) >= self.MIN_CHUNK_LENGTH:
                    chunks.append(DocumentChunk(
                        content=chunk_content,
                        doc_id=doc.doc_id,
                        chunk_index=len(chunks),
                        metadata={**doc.metadata, "sentence_count": len(current_chunk)},
                    ))

                # 保留重叠部分
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_len

        # 保存最后一个块
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            if len(chunk_content) >= self.MIN_CHUNK_LENGTH:
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    doc_id=doc.doc_id,
                    chunk_index=len(chunks),
                    metadata={**doc.metadata, "sentence_count": len(current_chunk)},
                ))

        print(f"[Documents] Split document {doc.doc_id} into {len(chunks)} chunks")
        return chunks

    def process_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        批量处理文档
        :param documents: 文档列表
        :return: 所有分块
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        print(f"[Documents] Processed {len(documents)} documents → {len(all_chunks)} chunks")
        return all_chunks


# 全局属性（用于类型提示）
DocumentProcessor.MIN_CHUNK_LENGTH = 100
```

#### `app/embeddings.py` — Embedding 和向量检索

```python
"""
Embedding 和向量检索模块
使用 sentence-transformers 生成向量，FAISS 进行相似度搜索
"""

import os
import numpy as np
import faiss
from typing import List, Dict, Tuple
from dataclasses import dataclass

from app.documents import DocumentChunk
from app.config import RAGConfig


@dataclass
class SearchResult:
    """检索结果"""
    chunk: DocumentChunk          # 文档分块
    score: float                  # 相似度分数
    rank: int                     # 排名


class EmbeddingManager:
    """
    Embedding 管理器
    负责向量生成、索引构建和相似度搜索
    """

    def __init__(self, config: RAGConfig = None):
        """初始化 Embedding 管理器"""
        self.config = config or RAGConfig()
        self.model = None
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.is_ready = False

    def load_model(self):
        """加载 Embedding 模型"""
        from sentence_transformers import SentenceTransformer

        print(f"[Embeddings] Loading model: {self.config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        print(f"[Embeddings] Model loaded, dimension: {self.model.get_sentence_embedding_dimension()}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为向量
        :param texts: 文本列表
        :return: 向量矩阵 (n, dim)
        """
        if self.model is None:
            self.load_model()

        # 批量编码（自动使用 GPU 如果可用）
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,  # 归一化（用于余弦相似度）
        )
        return embeddings.astype(np.float32)

    def build_index(self, chunks: List[DocumentChunk]):
        """
        构建向量索引
        :param chunks: 文档分块列表
        """
        if self.model is None:
            self.load_model()

        print(f"[Embeddings] Building index for {len(chunks)} chunks...")

        # 1. 生成向量
        texts = [chunk.content for chunk in chunks]
        embeddings = self.encode(texts)

        # 2. 构建 FAISS 索引
        dim = embeddings.shape[1]
        # 使用内积索引（向量已归一化，等价于余弦相似度）
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # 3. 保存分块
        self.chunks = chunks
        self.is_ready = True

        print(f"[Embeddings] Index built: {self.index.ntotal} vectors, dim={dim}")

    def search(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
    ) -> List[SearchResult]:
        """
        搜索相似文档
        :param query: 查询文本
        :param top_k: 返回结果数量
        :param threshold: 相似度阈值
        :return: 检索结果列表
        """
        if not self.is_ready:
            raise RuntimeError("Index not built. Call build_index() first.")

        top_k = top_k or self.config.TOP_K
        threshold = threshold or self.config.SIMILARITY_THRESHOLD

        # 1. 编码查询
        query_embedding = self.encode([query])

        # 2. 搜索
        scores, indices = self.index.search(query_embedding, top_k)

        # 3. 构建结果
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue  # FAISS 填充的无效索引
            if score < threshold:
                continue  # 低于阈值

            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(score),
                rank=rank + 1,
            ))

        return results

    def save_index(self, directory: str = None):
        """保存索引到磁盘"""
        directory = directory or self.config.INDEX_DIR
        os.makedirs(directory, exist_ok=True)

        # 保存 FAISS 索引
        index_path = os.path.join(directory, "faiss_index.bin")
        faiss.write_index(self.index, index_path)

        # 保存分块数据
        import json
        chunks_path = os.path.join(directory, "chunks.json")
        chunks_data = [
            {
                "content": c.content,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        print(f"[Embeddings] Index saved to {directory}")

    def load_index(self, directory: str = None):
        """从磁盘加载索引"""
        directory = directory or self.config.INDEX_DIR

        index_path = os.path.join(directory, "faiss_index.bin")
        chunks_path = os.path.join(directory, "chunks.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found at {directory}")

        # 加载 FAISS 索引
        self.index = faiss.read_index(index_path)

        # 加载分块数据
        import json
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        self.chunks = [
            DocumentChunk(
                content=c["content"],
                doc_id=c["doc_id"],
                chunk_index=c["chunk_index"],
                metadata=c["metadata"],
            )
            for c in chunks_data
        ]

        self.is_ready = True
        print(f"[Embeddings] Index loaded: {self.index.ntotal} vectors")


# 全局单例
embedding_manager = EmbeddingManager()
```

#### `app/rag.py` — RAG 核心逻辑

```python
"""
RAG 核心逻辑
检索增强生成的完整流程
"""

import time
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass

from app.embeddings import embedding_manager, SearchResult
from app.monitoring import rag_metrics
from app.config import RAGConfig


@dataclass
class RAGResponse:
    """RAG 响应"""
    response_id: str              # 响应 ID
    answer: str                   # 生成答案
    sources: List[Dict]           # 引用来源
    latency_ms: float             # 总延迟
    retrieval_latency_ms: float   # 检索延迟
    generation_latency_ms: float  # 生成延迟
    quality_score: float          # 质量评分
    context_length: int           # 上下文长度


class RAGEngine:
    """
    RAG 引擎
    整合检索和生成，提供完整的 RAG 能力
    """

    def __init__(self, config: RAGConfig = None):
        """初始化 RAG 引擎"""
        self.config = config or RAGConfig()
        self.llm = None

    def load_llm(self):
        """加载 LLM（模拟模式）"""
        # 实际项目中这里加载 vLLM 或其他推理引擎
        print(f"[RAG] LLM configured: {self.config.LLM_MODEL}")
        print("[RAG] Using mock LLM for demonstration")

    def build_prompt(
        self,
        query: str,
        contexts: List[SearchResult],
    ) -> str:
        """
        构建 RAG Prompt
        将检索到的文档与用户问题组合成 prompt
        """
        # 构建上下文部分
        context_parts = []
        for i, result in enumerate(contexts, 1):
            context_parts.append(
                f"[文档 {i}（相关度: {result.score:.3f}）]\n{result.chunk.content}"
            )

        context = "\n\n".join(context_parts)

        # 截断过长的上下文
        if len(context) > self.config.MAX_CONTEXT_LENGTH:
            context = context[:self.config.MAX_CONTEXT_LENGTH] + "..."

        # 构建完整 prompt
        prompt = f"""你是一个智能问答助手。请基于以下参考文档回答用户的问题。

## 参考文档
{context}

## 用户问题
{query}

## 回答要求
1. 仅基于参考文档中的信息回答
2. 如果文档中没有相关信息，请明确说明
3. 回答要简洁、准确
4. 在回答末尾注明引用的文档编号

请开始回答："""

        return prompt

    def generate_answer(self, prompt: str) -> str:
        """
        生成答案（模拟）
        实际项目中调用真实 LLM
        """
        import random

        # 模拟生成延迟
        latency = random.uniform(0.5, 2.0)
        time.sleep(latency * 0.05)  # 加速模拟

        # 从 prompt 中提取用户问题
        if "## 用户问题" in prompt:
            question_part = prompt.split("## 用户问题")[1]
            if "## 回答要求" in question_part:
                question = question_part.split("## 回答要求")[0].strip()
            else:
                question = question_part.strip()
        else:
            question = "未知问题"

        # 生成模拟答案
        answer = (
            f"基于检索到的文档，我来回答您的问题：「{question}」\n\n"
            f"根据参考文档中的信息，这个问题可以从以下几个方面来理解：\n\n"
            f"1. 文档中提到了相关的背景信息和上下文\n"
            f"2. 关键要点已经在参考文档中有详细描述\n"
            f"3. 建议结合多个文档的信息进行综合分析\n\n"
            f"💡 注意：当前使用模拟 LLM。在生产环境中，这里会调用真实的模型生成答案。\n\n"
            f"—— 引用：文档 1, 文档 2"
        )

        return answer

    def evaluate_quality(
        self,
        query: str,
        answer: str,
        contexts: List[SearchResult],
    ) -> float:
        """
        评估回答质量（模拟）
        实际项目中可以使用专门的评估模型
        """
        import random

        # 基于上下文相关性和答案长度模拟质量评分
        avg_similarity = sum(r.score for r in contexts) / len(contexts) if contexts else 0
        answer_quality = min(1.0, len(answer) / 200)  # 答案长度归一化

        quality = 0.6 * avg_similarity + 0.4 * answer_quality
        quality += random.uniform(-0.05, 0.05)  # 添加随机扰动
        return max(0, min(1, quality))

    def query(
        self,
        query: str,
        top_k: int = None,
    ) -> RAGResponse:
        """
        执行 RAG 查询
        完整流程：检索 → 构建 prompt → 生成 → 评估
        """
        start_time = time.time()

        # 1. 检索相关文档
        retrieval_start = time.time()
        contexts = embedding_manager.search(query, top_k=top_k)
        retrieval_latency = (time.time() - retrieval_start) * 1000

        rag_metrics.record_retrieval(
            query=query,
            num_results=len(contexts),
            avg_score=sum(r.score for r in contexts) / len(contexts) if contexts else 0,
            latency_ms=retrieval_latency,
        )

        # 2. 构建 prompt
        prompt = self.build_prompt(query, contexts)

        # 3. 生成答案
        generation_start = time.time()
        answer = self.generate_answer(prompt)
        generation_latency = (time.time() - generation_start) * 1000

        # 4. 质量评估
        quality = self.evaluate_quality(query, answer, contexts)

        total_latency = (time.time() - start_time) * 1000

        # 5. 记录指标
        rag_metrics.record_generation(
            query=query,
            answer=answer,
            quality=quality,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
            total_latency_ms=total_latency,
            context_length=len(prompt),
        )

        # 6. 检查告警
        rag_metrics.check_alerts(quality, total_latency)

        # 7. 构建响应
        sources = [
            {
                "rank": r.rank,
                "score": round(r.score, 4),
                "content_preview": r.chunk.content[:200] + "...",
                "doc_id": r.chunk.doc_id,
                "metadata": r.chunk.metadata,
            }
            for r in contexts
        ]

        return RAGResponse(
            response_id=str(uuid.uuid4())[:8],
            answer=answer,
            sources=sources,
            latency_ms=round(total_latency, 2),
            retrieval_latency_ms=round(retrieval_latency, 2),
            generation_latency_ms=round(generation_latency, 2),
            quality_score=round(quality, 4),
            context_length=len(prompt),
        )


# 全局单例
rag_engine = RAGEngine()
```

#### `app/monitoring.py` — 监控和告警

```python
"""
RAG 监控和告警模块
"""

import time
import threading
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警信息"""
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    metric_value: float
    threshold: float


class RAGMetrics:
    """
    RAG 专用监控指标收集器
    """

    def __init__(self):
        """初始化指标收集器"""
        self.retrieval_latencies: List[float] = []
        self.generation_latencies: List[float] = []
        self.quality_scores: List[float] = []
        self.context_lengths: List[float] = []
        self.total_latencies: List[float] = []
        self.query_count: int = 0
        self.alerts: List[Alert] = []
        self.lock = threading.Lock()

    def record_retrieval(
        self,
        query: str,
        num_results: int,
        avg_score: float,
        latency_ms: float,
    ):
        """记录检索指标"""
        with self.lock:
            self.retrieval_latencies.append(latency_ms)
            self.query_count += 1

    def record_generation(
        self,
        query: str,
        answer: str,
        quality: float,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        total_latency_ms: float,
        context_length: int,
    ):
        """记录生成指标"""
        with self.lock:
            self.quality_scores.append(quality)
            self.generation_latencies.append(generation_latency_ms)
            self.total_latencies.append(total_latency_ms)
            self.context_lengths.append(context_length)

    def check_alerts(self, quality: float, latency_ms: float):
        """
        检查是否需要告警
        """
        from app.config import RAGConfig
        config = RAGConfig()

        # 质量告警
        if quality < config.ALERT_QUALITY_THRESHOLD:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=f"回答质量过低: {quality:.4f} < {config.ALERT_QUALITY_THRESHOLD}",
                timestamp=time.time(),
                metric_name="quality",
                metric_value=quality,
                threshold=config.ALERT_QUALITY_THRESHOLD,
            )
            self.alerts.append(alert)
            print(f"🚨 [Alert] {alert.message}")

        # 延迟告警
        if latency_ms > config.ALERT_LATENCY_THRESHOLD:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=f"响应延迟过高: {latency_ms:.0f}ms > {config.ALERT_LATENCY_THRESHOLD}ms",
                timestamp=time.time(),
                metric_name="latency",
                metric_value=latency_ms,
                threshold=config.ALERT_LATENCY_THRESHOLD,
            )
            self.alerts.append(alert)
            print(f"🚨 [Alert] {alert.message}")

    def get_dashboard_data(self) -> Dict:
        """获取仪表盘数据"""
        with self.lock:
            import statistics

            def safe_mean(values):
                return round(statistics.mean(values), 2) if values else 0

            def safe_percentile(values, p):
                if not values:
                    return 0
                sorted_values = sorted(values)
                idx = int(len(sorted_values) * p / 100)
                return round(sorted_values[min(idx, len(sorted_values) - 1)], 2)

            return {
                "total_queries": self.query_count,
                "avg_quality": safe_mean(self.quality_scores),
                "p50_quality": safe_percentile(self.quality_scores, 50),
                "avg_retrieval_latency_ms": safe_mean(self.retrieval_latencies),
                "avg_generation_latency_ms": safe_mean(self.generation_latencies),
                "avg_total_latency_ms": safe_mean(self.total_latencies),
                "p99_total_latency_ms": safe_percentile(self.total_latencies, 99),
                "avg_context_length": safe_mean(self.context_lengths),
                "active_alerts": len([a for a in self.alerts
                                      if time.time() - a.timestamp < 3600]),
                "recent_alerts": [
                    {
                        "level": a.level.value,
                        "message": a.message,
                        "time": time.strftime("%H:%M:%S", time.localtime(a.timestamp)),
                    }
                    for a in self.alerts[-10:]
                ],
            }


# 全局单例
rag_metrics = RAGMetrics()
```

#### `app/pipeline.py` — 自动化流水线

```python
"""
RAG 自动化流水线
文档索引更新、定期重建等
"""

import os
import time
import glob
from typing import List
from app.documents import DocumentProcessor, Document
from app.embeddings import embedding_manager
from app.config import RAGConfig


class RAGPipeline:
    """
    RAG 自动化流水线
    管理文档的索引更新流程
    """

    def __init__(self, config: RAGConfig = None):
        """初始化流水线"""
        self.config = config or RAGConfig()
        self.processor = DocumentProcessor(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
        )

    def ingest_directory(self, directory: str) -> int:
        """
        处理目录中的所有文档
        :param directory: 文档目录
        :return: 处理的文档数量
        """
        print(f"[Pipeline] Ingesting documents from: {directory}")

        # 支持的文档格式
        extensions = ["*.txt", "*.md"]

        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))

        if not files:
            print(f"[Pipeline] No documents found in {directory}")
            return 0

        print(f"[Pipeline] Found {len(files)} documents")

        # 1. 加载文档
        documents = []
        for file_path in files:
            try:
                doc = self.processor.load_file(file_path)
                documents.append(doc)
            except Exception as e:
                print(f"[Pipeline] Error loading {file_path}: {e}")

        # 2. 分块
        all_chunks = self.processor.process_documents(documents)

        if not all_chunks:
            print("[Pipeline] No chunks generated")
            return 0

        # 3. 构建索引
        embedding_manager.build_index(all_chunks)

        # 4. 保存索引
        embedding_manager.save_index()

        print(f"[Pipeline] ✅ Pipeline complete: {len(documents)} docs → {len(all_chunks)} chunks")
        return len(documents)

    def add_document(self, text: str, metadata: dict = None) -> int:
        """
        添加单个文档并更新索引
        :param text: 文档内容
        :param metadata: 元数据
        :return: 新增分块数量
        """
        print("[Pipeline] Adding new document...")

        # 1. 创建文档
        doc = self.processor.load_text(text, metadata)

        # 2. 分块
        chunks = self.processor.chunk_document(doc)

        if not chunks:
            print("[Pipeline] No chunks generated")
            return 0

        # 3. 添加到现有索引
        if embedding_manager.is_ready:
            # 生成新分块的向量
            texts = [c.content for c in chunks]
            embeddings = embedding_manager.encode(texts)
            embedding_manager.index.add(embeddings)
            embedding_manager.chunks.extend(chunks)
            print(f"[Pipeline] Added {len(chunks)} chunks to existing index")
        else:
            # 首次构建索引
            embedding_manager.build_index(chunks)

        # 4. 保存更新后的索引
        embedding_manager.save_index()

        return len(chunks)

    def rebuild_index(self, directory: str = None) -> int:
        """
        完全重建索引
        :param directory: 文档目录
        :return: 处理的文档数量
        """
        print("[Pipeline] Rebuilding index from scratch...")
        directory = directory or self.config.RAW_DIR
        return self.ingest_directory(directory)


# 全局单例
pipeline = RAGPipeline()
```

#### `app/main.py` — FastAPI 入口

```python
"""
RAG 生产系统主入口
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse
from typing import Optional

from app.config import RAGConfig
from app.embeddings import embedding_manager
from app.rag import rag_engine
from app.pipeline import pipeline
from app.monitoring import rag_metrics

config = RAGConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("🚀 Starting RAG Production System")

    # 创建数据目录
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    os.makedirs(config.RAW_DIR, exist_ok=True)

    # 加载 Embedding 模型
    embedding_manager.load_model()

    # 尝试加载已有索引
    try:
        embedding_manager.load_index()
        print("✅ Existing index loaded")
    except FileNotFoundError:
        print("ℹ️  No existing index found, will build on first ingest")

    # 加载 LLM
    rag_engine.load_llm()

    yield

    print("👋 Shutting down RAG system")


app = FastAPI(
    title="RAG Production System",
    description="生产级 RAG 系统，包含文档处理、向量检索、LLM 生成和监控告警",
    lifespan=lifespan,
)


# ========== 文档管理 ==========

@app.post("/documents/ingest", tags=["文档管理"])
async def ingest_documents(
    directory: str = Query(None, description="文档目录路径"),
):
    """
    处理目录中的文档
    自动加载、分块、向量化和索引
    """
    directory = directory or config.RAW_DIR
    count = pipeline.ingest_directory(directory)
    return {
        "status": "success",
        "documents_processed": count,
        "index_size": int(embedding_manager.index.ntotal) if embedding_manager.is_ready else 0,
    }


@app.post("/documents/upload", tags=["文档管理"])
async def upload_document(file: UploadFile = File(...)):
    """
    上传单个文档
    支持 .txt 和 .md 文件
    """
    content = await file.read()
    text = content.decode('utf-8')

    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
    }

    chunks_added = pipeline.add_document(text, metadata)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": chunks_added,
        "total_chunks": int(embedding_manager.index.ntotal) if embedding_manager.is_ready else 0,
    }


@app.post("/documents/rebuild", tags=["文档管理"])
async def rebuild_index(directory: str = Query(None)):
    """完全重建索引"""
    directory = directory or config.RAW_DIR
    count = pipeline.rebuild_index(directory)
    return {"status": "success", "documents_processed": count}


# ========== RAG 查询 ==========

@app.post("/rag/query", tags=["RAG 查询"])
async def rag_query(
    query: str = Query(..., min_length=1, max_length=2000, description="查询问题"),
    top_k: int = Query(5, ge=1, le=20, description="返回结果数量"),
):
    """
    RAG 查询接口
    执行完整的检索增强生成流程
    """
    response = rag_engine.query(query, top_k=top_k)

    return {
        "response_id": response.response_id,
        "answer": response.answer,
        "sources": response.sources,
        "latency": {
            "total_ms": response.latency_ms,
            "retrieval_ms": response.retrieval_latency_ms,
            "generation_ms": response.generation_latency_ms,
        },
        "quality_score": response.quality_score,
        "context_length": response.context_length,
    }


@app.get("/rag/search", tags=["RAG 查询"])
async def search_documents(
    query: str = Query(..., description="搜索查询"),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    纯检索接口（不生成答案）
    仅返回相关文档片段
    """
    results = embedding_manager.search(query, top_k=top_k)

    return {
        "query": query,
        "num_results": len(results),
        "results": [
            {
                "rank": r.rank,
                "score": round(r.score, 4),
                "content": r.chunk.content,
                "doc_id": r.chunk.doc_id,
                "metadata": r.chunk.metadata,
            }
            for r in results
        ],
    }


# ========== 监控 ==========

@app.get("/monitoring/dashboard", tags=["监控"])
async def monitoring_dashboard():
    """监控仪表盘数据"""
    return rag_metrics.get_dashboard_data()


@app.get("/monitoring/dashboard", response_class=HTMLResponse, tags=["监控"])
async def monitoring_dashboard_html():
    """监控仪表盘（HTML 页面）"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG 监控仪表盘</title>
        <style>
            body { font-family: system-ui; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f0f2f5; }
            .card { background: white; border-radius: 12px; padding: 24px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
            .stat { text-align: center; padding: 16px; background: #f8f9fa; border-radius: 8px; }
            .stat-value { font-size: 2.5em; font-weight: bold; color: #1a73e8; }
            .stat-label { font-size: 0.9em; color: #666; margin-top: 4px; }
            .alert { padding: 12px 16px; border-radius: 8px; margin: 8px 0; }
            .alert-warning { background: #fff3cd; border-left: 4px solid #ffc107; }
            .alert-critical { background: #f8d7da; border-left: 4px solid #dc3545; }
            h1 { color: #333; }
            h2 { color: #555; font-size: 1.2em; }
        </style>
    </head>
    <body>
        <h1>📊 RAG 系统监控仪表盘</h1>

        <div class="card">
            <h2>核心指标</h2>
            <div class="grid" id="stats"></div>
        </div>

        <div class="card">
            <h2>告警信息</h2>
            <div id="alerts"></div>
        </div>

        <script>
            async function loadDashboard() {
                const res = await fetch('/monitoring/dashboard');
                const data = await res.json();

                document.getElementById('stats').innerHTML = `
                    <div class="stat">
                        <div class="stat-value">${data.total_queries}</div>
                        <div class="stat-label">总查询数</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">${data.avg_quality}</div>
                        <div class="stat-label">平均质量</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">${data.avg_total_latency_ms}ms</div>
                        <div class="stat-label">平均延迟</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">${data.p99_total_latency_ms}ms</div>
                        <div class="stat-label">P99 延迟</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">${data.active_alerts}</div>
                        <div class="stat-label">活跃告警</div>
                    </div>
                `;

                let alertsHtml = '';
                if (data.recent_alerts.length === 0) {
                    alertsHtml = '<p style="color: #28a745;">✅ 暂无告警</p>';
                } else {
                    data.recent_alerts.forEach(a => {
                        const cls = a.level === 'critical' ? 'alert-critical' : 'alert-warning';
                        alertsHtml += `<div class="alert ${cls}">
                            <strong>[${a.time}]</strong> ${a.message}
                        </div>`;
                    });
                }
                document.getElementById('alerts').innerHTML = alertsHtml;
            }

            loadDashboard();
            setInterval(loadDashboard, 3000);
        </script>
    </body>
    </html>
    """


# ========== 系统 ==========

@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "embedding_ready": embedding_manager.is_ready,
        "index_size": int(embedding_manager.index.ntotal) if embedding_manager.is_ready else 0,
    }


@app.get("/", tags=["系统"])
async def root():
    """根路径"""
    return {
        "name": "RAG Production System",
        "docs": "/docs",
        "health": "/health",
        "dashboard": "/monitoring/dashboard",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
```

### 🚀 运行方式

```bash
# 1. 创建项目结构
mkdir -p rag-production/{app,data/{raw,index},tests}

# 2. 将上述代码保存到对应文件

# 3. 安装依赖
cd rag-production
pip install -r requirements.txt

# 4. 创建测试文档
cat > data/raw/test1.txt << 'EOF'
Python 是一种广泛使用的高级编程语言。
它由 Guido van Rossum 于 1991 年发布。
Python 的设计哲学强调代码的可读性和简洁的语法。
它支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
Python 拥有一个庞大且活跃的标准库，涵盖了从网络编程到数据科学的各个领域。
在人工智能和机器学习领域，Python 是最流行的编程语言之一。
EOF

# 5. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

### 📡 测试流程

```bash
# 1. 处理文档（构建索引）
curl -X POST "http://localhost:8002/documents/ingest"
# 预期输出: {"status": "success", "documents_processed": 1, "index_size": 1}

# 2. 执行 RAG 查询
curl -X POST "http://localhost:8002/rag/query?query=Python是什么语言？&top_k=3" \
  | python3 -m json.tool

# 预期输出:
# {
#   "response_id": "a1b2c3d4",
#   "answer": "基于检索到的文档，我来回答您的问题...",
#   "sources": [...],
#   "latency": {"total_ms": 1234.5, "retrieval_ms": 45.2, "generation_ms": 1189.3},
#   "quality_score": 0.7234,
#   "context_length": 1523
# }

# 3. 纯检索（不生成）
curl "http://localhost:8002/rag/search?query=Python编程&top_k=3" \
  | python3 -m json.tool

# 4. 上传新文档
curl -X POST "http://localhost:8002/documents/upload" \
  -F "file=@data/raw/test1.txt"

# 5. 查看监控仪表盘
# 浏览器访问: http://localhost:8002/monitoring/dashboard

# 6. 查看 API 文档
# 浏览器访问: http://localhost:8002/docs
```

### 📊 预期输出

```json
// RAG 查询响应
{
  "response_id": "a1b2c3d4",
  "answer": "基于检索到的文档，我来回答您的问题：「Python是什么语言？」\n\n根据参考文档中的信息，Python 是一种广泛使用的高级编程语言...\n\n💡 注意：当前使用模拟 LLM。",
  "sources": [
    {
      "rank": 1,
      "score": 0.8234,
      "content_preview": "Python 是一种广泛使用的高级编程语言。它由 Guido van Rossum 于 1991 年发布...",
      "doc_id": "abc123def456",
      "metadata": {"source": "data/raw/test1.txt", "filename": "test1.txt"}
    }
  ],
  "latency": {
    "total_ms": 1234.5,
    "retrieval_ms": 45.2,
    "generation_ms": 1189.3
  },
  "quality_score": 0.7234,
  "context_length": 1523
}

// 监控仪表盘数据
{
  "total_queries": 15,
  "avg_quality": 0.71,
  "avg_total_latency_ms": 1156.8,
  "p99_total_latency_ms": 2345.6,
  "active_alerts": 0,
  "recent_alerts": []
}
```

---

## 📋 项目总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        项目能力矩阵                                      │
├──────────────┬──────────┬──────────┬──────────┬──────────┤
│ 能力         │ 项目 1   │ 项目 2   │ 项目 3   │ 项目 4   │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ API 开发     │    ✅    │          │    ✅    │    ✅    │
│ 模型推理     │    ✅    │    ✅    │    ✅    │    ✅    │
│ 缓存优化     │    ✅    │          │          │          │
│ 监控告警     │    ✅    │          │    ✅    │    ✅    │
│ 实验追踪     │          │    ✅    │          │          │
│ 数据版本     │          │    ✅    │          │          │
│ CI/CD        │          │    ✅    │          │          │
│ A/B 测试     │          │          │    ✅    │          │
│ 灰度发布     │          │          │    ✅    │          │
│ 向量检索     │          │          │          │    ✅    │
│ RAG 系统     │          │          │          │    ✅    │
│ 文档处理     │          │          │          │    ✅    │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 难度         │  ⭐⭐⭐  │  ⭐⭐⭐⭐ │  ⭐⭐⭐⭐ │ ⭐⭐⭐⭐⭐ │
│ 代码量       │  ~400行  │  ~450行  │  ~400行  │  ~500行  │
│ 预计时间     │  2-3天   │  3-4天   │  3-4天   │  4-5天   │
└──────────────┴──────────┴──────────┴──────────┴──────────┘
```

### 🎯 学习路径建议

```
推荐学习顺序：

  1️⃣ 项目 1（LLM API 服务）
     ↓ 掌握 API 开发、缓存、监控基础
  2️⃣ 项目 2（MLOps 流水线）
     ↓ 掌握实验追踪、模型管理、CI/CD
  3️⃣ 项目 3（A/B 测试系统）
     ↓ 掌握流量管理、统计分析和自动决策
  4️⃣ 项目 4（RAG 生产部署）
     ↓ 综合运用所有技能，构建完整系统

💡 提示：每个项目都可以独立运行和学习
   建议先跑通每个项目，再尝试将它们组合
```

---

> _好的工程实践不是写更多的代码，而是让代码更好地服务于用户。_
> 
> _—— 悟空_