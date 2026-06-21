# UnifiedAPI

Anthropic ↔ OpenAI 协议转换网关。

让使用 Anthropic Messages API 的客户端（主要是 [Claude Code](https://docs.anthropic.com/claude/docs/claude-code) 和 Anthropic SDK）透明地接入 OpenAI 兼容的上游服务。一期针对港科大（广州）AIGC 服务（`https://aigc-api.hkust-gz.edu.cn/v1`，模型 `DeepSeek-V4-Pro`）。

## 功能

- **协议转换**：Anthropic Messages API（`POST /v1/messages`）→ OpenAI Chat Completions
- **流式 SSE**：OpenAI delta 格式 → Anthropic 事件序列（`message_start` → `content_block_*` → `message_delta` → `message_stop`）
- **工具调用**：上游不原生支持 OpenAI `tools` 参数，改用 prompt 注入式 XML（`<function_calls><invoke name="...">`）
- **限流 / 队列 / 并发控制**：全局限流 + 按客户端限流 + 等待室 + 自动重试
- **多模型路由**：客户端可见的 alias（如 `claude-sonnet-4-5`）映射到上游实际模型
- **客户端 key 透传**：通过 `x-api-key` 或 `Authorization: Bearer` 头识别客户端，每个客户端独立限流额度

## 安装

```bash
git clone <repo> UnifiedAPI
cd UnifiedAPI
conda create -n UnifiedAPI python=3.12 -y
conda activate UnifiedAPI
pip install -e ".[dev]"
```

## 配置

### `.env`（项目根目录）

```bash
OPENAI_BASE_URL=https://aigc-api.hkust-gz.edu.cn/v1
OPENAI_KEY=your-upstream-api-key-here
OPENAI_MODEL=DeepSeek-V4-Pro
```

### `config.yaml`

完整配置见 [`config.yaml`](config.yaml)。关键字段：

| 路径 | 说明 | 默认 |
|---|---|---|
| `server.host` / `server.port` | 监听地址 | `0.0.0.0:8000` |
| `upstream.timeout_seconds` | 上游调用超时 | `300` |
| `limits.global_concurrency` | 全局并发上限（保护上游） | `10` |
| `limits.global_rpm` | 全局每分钟请求数 | `60` |
| `limits.per_client_concurrency` | 单客户端并发上限 | `5` |
| `limits.per_client_rpm` | 单客户端每分钟请求数 | `30` |
| `limits.queue_max_size` | 等待室容量（超出返回 503） | `100` |
| `retry.max_attempts` | 最大重试次数（含首次） | `3` |
| `retry.base_backoff_ms` / `max_backoff_ms` | 指数退避区间 | `500` / `10000` |
| `thinking.return_by_default` | 是否默认返回 reasoning_content | `false` |
| `thinking.return_when_client_enables` | 客户端发 `thinking:enabled` 时是否返回 | `true` |
| `models` | alias → upstream_model 映射列表 | 见下 |

模型路由示例：

```yaml
models:
  - alias: claude-sonnet-4-5
    upstream_model: DeepSeek-V4-Pro
    max_tokens_default: 8192
  - alias: claude-opus-4-6
    upstream_model: DeepSeek-V4-Pro
    max_tokens_default: 16384
  - alias: claude-haiku-4-5
    upstream_model: DeepSeek-V4-Pro
    max_tokens_default: 4096
  # 用户可自由加自定义 alias
```

`${OPENAI_BASE_URL}` 和 `${OPENAI_KEY}` 在 `config.yaml` 里会自动从 `.env` 插值。

## 启动

```bash
uvicorn unified_api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**必须 `--workers 1`**：限流 / 队列 / 并发统计都是单进程内存态，多 worker 会让每个进程独立计数，破坏语义。I/O bound 代理单进程 asyncio 足够撑住几百到上千并发。

## Claude Code 接入

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=any-string   # 我们透传，不校验内容
claude
```

非交互式冒烟测试：

```bash
claude -p "say hello"
claude -p "read README.md and summarize"   # 触发工具调用
```

## 健康检查

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## 测试

```bash
# 全部（含真实上游 E2E，需要 .env 配置好 OPENAI_KEY）
pytest

# 仅单元测试（不打上游）
pytest --ignore=tests/test_e2e.py

# 单个模块
pytest tests/test_xml_parser.py -v
pytest tests/test_stream_convert.py -v
```

测试覆盖：

| 文件 | 覆盖 |
|---|---|
| `test_xml_parser.py` | `<function_calls>` XML 增量解析（含 tag 跨 chunk 切分、并行 invoke、XML 实体转义、未闭合容错） |
| `test_think_splitter.py` | `<think>...</think>` 增量剥离（含 tag 跨 chunk 切分） |
| `test_stream_convert.py` | OpenAI SSE → Anthropic SSE 状态机（事件序列、block 切换、tool_use 增量、finish_reason 映射） |
| `test_request_convert.py` | Anthropic → OpenAI 请求转换（system / tools 注入 / tool_use 重放 / tool_result 重放） |
| `test_response_convert.py` | OpenAI → Anthropic 非流式响应转换（thinking / tool_use / stop_reason） |
| `test_rate_limiter.py` | 令牌桶（"go negative" 并发等待、per-client 隔离、全局共享） |
| `test_concurrency.py` | AdmissionControl（全局 / per-client 并发上限、等待室满载、异常 / 取消时槽释放） |
| `test_retry.py` | tenacity 包装器（5xx / 429 / 网络错误重试，4xx 不重试，指数退避） |
| `test_e2e.py` | 端到端：真实上游 + FastAPI app（非流式 / 流式 / 工具调用 / 多轮对话） |

## 架构概览

```
Claude Code ──POST /v1/messages──►  FastAPI (uvicorn workers=1)
                                      │
                                      ▼
                            [Request converter]
                            Anthropic req → OpenAI req
                            · system 拼接
                            · tools → XML prompt 注入
                            · tool_use 历史 → XML 重放
                            · tool_result → <tool_result> 文本
                                      │
                                      ▼
                            [AdmissionControl]
                            · 全局 + per-client 并发上限
                            · 等待室（满载返 503 overloaded_error）
                                      │
                                      ▼
                            [RateLimiter]
                            · 全局 + per-client RPM 令牌桶
                                      │
                                      ▼
                            [Retry] (tenacity)
                            · 指数退避 + jitter
                            · 仅对 5xx / 429 / 网络错误重试
                                      │
                                      ▼
                            [UpstreamClient] (httpx.AsyncClient)
                            · 错误归一化（混合格式 → typed exception）
                                      │
                                      ▼
                            [Response converter]
                            OpenAI → Anthropic
                            · <function_calls> XML → tool_use block
                            · <think>...</think> 剥离
                            · reasoning_content → thinking block（可选）
                            · 流式 SSE 状态机
```

## 上游已知行为（已适配）

| 现象 | 对策 |
|---|---|
| HTTP 永远 200，错误在 body 里（`{"code","msg"}` 或 `{"error":{...}}`） | `_looks_like_error` body 检测 + typed exception 映射 |
| 流式无 `data: [DONE]` 终止符 | 迭代到 HTTP body 结束即视为结束 |
| 没有 `X-RateLimit-*` / `Retry-After` header | 自维护令牌桶 |
| 忽略 OpenAI `tools` 参数 | prompt 注入 XML 工具说明 |
| `reasoning_content` 字段（思维链） | 默认丢弃；客户端 `thinking:enabled` 时转 thinking block |
| `<think>...</think>` 散落在 content 里 | 增量剥离 |
| 非流式 `created` 是字符串时间戳，流式是 int | 字段类型联合，不依赖 |

## 风险与限制

| 风险 | 状态 |
|---|---|
| 进程重启丢失等待队列 | 一期接受；二期可换 Redis |
| thinking block 无 signature | 多数客户端容忍；已默认不返回 |
| 多 worker 会破坏限流语义 | 强制 `--workers 1`，文档明示 |
| 工具调用依赖模型遵守 prompt 注入 | 实测稳定（6/6 场景），但理论上有幻觉风险 |

## 项目结构

```
src/unified_api/
├── main.py              # FastAPI app + lifespan
├── config.py            # .env + config.yaml 加载
├── models.py            # Pydantic schemas
├── errors.py            # AnthropicError + ConversionError
├── routes/
│   └── messages.py      # POST /v1/messages（非流式 + 流式）
├── converters/
│   ├── request.py       # Anthropic → OpenAI
│   ├── response.py      # OpenAI → Anthropic（非流式）
│   └── stream.py        # OpenAI SSE → Anthropic SSE（状态机）
├── tools/
│   ├── prompt_builder.py  # tools schema → XML system prompt
│   ├── xml_parser.py      # 增量 <function_calls> XML 解析
│   └── think_splitter.py  # 增量 <think>...</think> 剥离
├── upstream/
│   ├── client.py        # httpx.AsyncClient + 错误归一化
│   └── errors.py        # typed exception 层级
└── control/
    ├── rate_limiter.py  # 令牌桶（global + per-client RPM）
    ├── concurrency.py   # AdmissionControl（并发 + 等待室）
    └── retry.py         # tenacity 包装器
```

## License

MIT
# UnifiedAPI
