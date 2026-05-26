# Codex Proxy

OpenAI Codex 使用国内大模型的中间代理，过滤 Codex `/responses` API 中的非标准参数和工具类型，使其能通过 LiteLLM 正常连接国内模型。

## 架构

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────────┐
│ Codex CLI │───▶│ ccswitch  │───▶│ Codex Proxy  │───▶│  LiteLLM  │───▶│ 国内大模型 API │
│           │    │  :4001/v1 │    │   :4001      │    │  :4000    │    │             │
└──────────┘    └──────────┘    └──────────────┘    └──────────┘    └─────────────┘

Codex CLI      发起 /responses API 请求
ccswitch        Codex 模型切换工具，将 base URL 指向本代理
Codex Proxy    过滤非标准参数和工具类型
LiteLLM        模型路由、API Key 管理、多模型负载均衡
国内大模型 API   DeepSeek、智谱 GLM、Minimax 等，提供 /chat/completions 接口
```

ccswitch 将 Codex 的请求地址指向本代理（而非直接指向 LiteLLM），本代理过滤后再转发给 LiteLLM。

## 使用方法

纯 Python，无需安装依赖。

### 1. 启动 LiteLLM

配置好国内模型的 API Key 和地址，启动 LiteLLM。

### 2. 启动本代理

```bash
python3 proxy.py --mode filter
```

### 3. 配置 ccswitch

将 base URL 指向本代理：

```
http://localhost:4001/v1
```

或通过环境变量：

```bash
export OPENAI_BASE_URL=http://localhost:4001/v1
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `passthrough` | `passthrough` / `drop-params` / `filter` |
| `--litellm` | `http://localhost:4000` | LiteLLM 地址 |
| `--port` | `4001` | 代理监听端口 |
| `--log-dir` | `logs` | 日志目录 |

## 三种模式

| 模式 | 参数过滤 | Tool 过滤 | 降级转换 | 适用场景 |
|------|---------|----------|---------|---------|
| `passthrough` | 不过滤 | 不过滤 | 不做 | 排查问题，观察原始请求 |
| `drop-params` | 过滤 `client_metadata` 等 | 不过滤 | 不做 | 只解决参数报错，保留所有 tool 类型观察 |
| `filter` | 过滤 `client_metadata` 等 | 过滤 `web_search` 等 | `/responses` 失败时降级到 `/chat/completions` | 日常使用，推荐模式 |

### 过滤内容

**参数过滤（drop-params 和 filter 模式生效）：**

- `client_metadata` — Codex 安装标识
- `prompt_cache_key` — OpenAI 缓存机制
- `store` — OpenAI 存储选项
- `include` — OpenAI 响应包含字段
- `reasoning` — OpenAI 推理配置
- `parallel_tool_calls` — OpenAI 并行调用配置
- `vector_store_ids` — OpenAI 向量存储
- `vector_store_request_metadata` — OpenAI 向量存储元数据

**Tool 类型过滤（filter 模式生效）：**

- `web_search` — OpenAI 内置联网搜索
- `namespace` — OpenAI 多 agent 协作
- `file_search` — OpenAI 文件搜索

保留的 `function` 类型 tools（Codex 核心能力）：

- `exec_command` — 执行 shell 命令
- `write_stdin` — 向进程写入
- `update_plan` / `get_goal` / `create_goal` / `update_goal` — 任务管理
- `request_user_input` — 请求用户输入
- `view_image` — 查看图片

## 问题背景

Codex 使用 OpenAI 的 `/responses` API，国内模型只支持 `/chat/completions`。两者存在以下不兼容：

- `client_metadata` 等非标准参数导致 `AsyncCompletions.create() got an unexpected keyword argument` 报错
- `web_search`、`namespace` 等非标准 tool 类型导致国内模型返回 `model engine error`
- 部分模型平台对非标准 tool 类型的报错方式不同

## 日志

所有日志保存在 `logs/` 目录：

| 文件 | 说明 |
|------|------|
| `proxy.log` | 运行日志（同时输出到控制台和文件） |
| `req_*.json` | 完整原始请求体 |
| `req_dropped_*.json` | drop-params 模式过滤后的请求 |
| `req_filtered_*.json` | filter 模式过滤后的请求 |
| `resp_*.json` | 响应体 |
| `err_*.json` | 错误响应 |