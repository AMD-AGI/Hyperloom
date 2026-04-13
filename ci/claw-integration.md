# Claw 调用流程（给 Claw 团队对接用）

## 概述

Hyperloom CI 通过 Claw REST API 创建 Session、发送 prompt、监听 SSE 事件流来执行推理优化任务。

## 完整调用序列

```
CI Runner                              Claw API                           Claw Backend
   │                                      │                                    │
   │  1. POST /v1/sessions                │                                    │
   │  Body: {name, agent_id}              │                                    │
   │ ────────────────────────────────────→ │                                    │
   │  Response: {session_id}              │                                    │
   │ ←──────────────────────────────────── │                                    │
   │                                      │                                    │
   │  2. GET /v1/chat/sessions/{id}/messages  (SSE 长连接)                     │
   │  Accept: text/event-stream           │                                    │
   │ ────────────────────────────────────→ │                                    │
   │  (连接保持, 开始接收事件)              │                                    │
   │ ←═══════════════════════════════════  │                                    │
   │                                      │                                    │
   │  3. POST /v1/sessions/{id}/messages  │                                    │
   │  Body: {content, contents,           │                                    │
   │         messageType: "text",         │  → start_executor_sandbox()        │
   │         taskMode: "agent",           │  → simulate_session_run()          │
   │         tools: [], attachments: []}  │                                    │
   │ ────────────────────────────────────→ │                                    │
   │  Response: {accepted: true}          │                                    │
   │ ←──────────────────────────────────── │                                    │
   │                                      │                                    │
   │  4. SSE 事件流 (持续接收)             │                                    │
   │                                      │  publish_session_event →           │
   │  event: status_update                │                                    │
   │  data: {type: "statusUpdate",        │  (Agent 开始运行)                  │
   │         agentStatus: "running"}      │                                    │
   │ ←═══════════════════════════════════  │                                    │
   │                                      │                                    │
   │  event: chat                         │                                    │
   │  data: {type: "chat",                │  (用户消息回显)                     │
   │         role: "user", content: ...}  │                                    │
   │ ←═══════════════════════════════════  │                                    │
   │                                      │                                    │
   │  event: live_status                  │                                    │
   │  data: {type: "liveStatus",          │  (Agent 正在思考)                  │
   │         text: "Thinking"}            │                                    │
   │ ←═══════════════════════════════════  │                                    │
   │                                      │                                    │
   │  event: tool_used                    │                                    │
   │  data: {type: "toolUsed",            │  (调用 MCP 工具)                   │
   │         tool: "shell",               │                                    │
   │         status: "start",             │                                    │
   │         actionId: "xxx"}             │                                    │
   │ ←═══════════════════════════════════  │                                    │
   │                                      │                                    │
   │  event: tool_used                    │                                    │
   │  data: {type: "toolUsed",            │  (工具执行结果)                     │
   │         actionId: "xxx",             │                                    │
   │         status: "success"}           │                                    │
   │ ←═══════════════════════════════════  │                                    │
   │                                      │                                    │
   │  event: chat_delta                   │                                    │
   │  data: {type: "chatDelta",           │  (Agent 流式输出)                  │
   │         delta: {content: "...",       │                                    │
   │                 thought: "..."}}     │                                    │
   │ ←═══════════════════════════════════  │                                    │
   │                                      │                                    │
   │  ... (重复 tool_used / chat_delta)   │                                    │
   │                                      │                                    │
   │  event: chat                         │                                    │
   │  data: {type: "chat",                │  (Agent 最终回复)                  │
   │         role: "assistant",           │                                    │
   │         content: "...完整回复..."}    │                                    │
   │ ←═══════════════════════════════════  │                                    │
   │                                      │                                    │
   │  event: status_update                │                                    │
   │  data: {type: "statusUpdate",        │  (Agent 执行结束)                  │
   │         agentStatus: "stopped",      │                                    │
   │         brief: "PrimusClaw completed"}│                                   │
   │ ←═══════════════════════════════════  │                                    │
   │                                      │                                    │
   │  (CI 关闭 SSE 连接, 收集结果)         │                                    │
```

## API 详细说明

### Step 1: 创建 Session

```
POST /v1/sessions
Content-Type: application/json
Authorization: Bearer ak-xxx  (可选, 不传则 user_id=default)
```

**Request Body:**

```json
{
  "name": "ci-Qwen3.5-397B-A17B-20260408-0200",
  "agent_id": "agent_default"
}
```

`SessionCreateRequest` 完整字段（均有默认值）：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| name | str | "" | Session 名称 |
| agent_id | str | "agent_default" | Agent ID |
| system_prompt | str | "" | 系统 prompt |
| config | dict | {} | 额外配置 |
| private | int | 0 | 0=公开, 1=私有 |

**Response:**

```json
{
  "code": 200,
  "data": {
    "session_id": "5717783f-c4a6-4ca2-b876-b4a89162421c",
    "name": "ci-Qwen3.5-397B-A17B-20260408-0200",
    "agent_id": "agent_default",
    "status": "active",
    "user_id": "1b028c9fa3819bb971f80bc74a90e21d",
    "created_at": "2026-04-08T02:00:00.000000Z",
    "updated_at": "2026-04-08T02:00:00.000000Z"
  }
}
```

### Step 2: 订阅 SSE（先于发消息）

```
GET /v1/chat/sessions/{session_id}/messages
Accept: text/event-stream
```

长连接，服务端每 15 秒发 keepalive（`: keepalive\n\n`）。

SSE 格式：

```
id: event_xxxxxxxxxxxx
event: <event_name>
data: <JSON>

```

### Step 3: 发送消息

```
POST /v1/sessions/{session_id}/messages
Content-Type: application/json
```

**Request Body:**

```json
{
  "content": "Use the inference-optimization skill to optimize ...",
  "contents": [
    {"type": "text", "value": "Use the inference-optimization skill to optimize ..."}
  ],
  "messageType": "text",
  "taskMode": "agent",
  "attachments": [],
  "tools": []
}
```

`SessionMessageRequest` 完整字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| content | str | "" | 消息文本 |
| contents | [{type, value}] | [] | 富文本消息体（与 content 保持一致） |
| messageType | str | "text" | 消息类型: text / decision / interrupt / inject |
| taskMode | str | "agent" | 任务模式: agent |
| attachments | [str] | [] | 附件列表 |
| tools | [int] | [] | MCP tool IDs |
| extData | dict | {} | 扩展数据 |

**Response:**

```json
{
  "code": 200,
  "data": {
    "session_id": "5717783f-...",
    "accepted": true,
    "message": "Message accepted"
  }
}
```

发送后 Claw 后台异步执行：
1. 如果启用 Sandbox → `start_executor_sandbox()` 创建 SaFE Pod
2. `simulate_session_run()` 启动 Agent 执行

**注意**：如果该 session 已经有 Agent 在运行（`session_id in _running_sessions`），
消息会被当作 `inject` 发送给正在运行的 Agent，而不是启动新任务。

### Step 4: SSE 事件类型

| event 名 | data.type | 含义 | CI 处理方式 |
|-----------|-----------|------|------------|
| `status_update` | `statusUpdate` | Agent 状态变更 | `agentStatus=stopped` → 结束监控 |
| `sandbox_status` | `sandboxStatus` | Sandbox 生命周期 | 记录日志（creating→ready） |
| `chat` | `chat` | 完整消息（user/assistant） | 记录 Agent 回复 |
| `chat_delta` | `chatDelta` | 流式输出片段 | 可忽略（chat 事件包含完整内容） |
| `tool_used` | `toolUsed` | 工具调用（start/success/error） | 记录工具调用日志 |
| `live_status` | `liveStatus` | Agent 状态文本（Thinking 等） | 可忽略 |
| `error` | `error` | 执行错误 | 标记 failed，结束监控 |
| `permissionRequest` | `permissionRequest` | HITL 权限请求 | CI 不支持交互，需确认 Agent 不会触发 |

`sandboxStatus` 的 `phase` 取值：

| phase | status | 含义 |
|-------|--------|------|
| `creating` | `started` | Sandbox Pod 正在创建 |
| `ready` | `completed` | Sandbox 就绪，Agent 即将启动 |

**注意**：`chat` 事件的 `content` 字段不是纯文本，而是结构化 JSON：

```json
{
  "_type": "AssistantMessage",
  "content": [{"_type": "TextBlock", "text": "实际回复内容"}]
}
```

`statusUpdate` 的 `agentStatus` 取值：

| agentStatus | 含义 |
|-------------|------|
| `running` | Agent 开始执行 |
| `stopped` | Agent 执行结束 |

`brief` 字段说明结果：`"PrimusClaw completed"` 或 `"PrimusClaw failed"`。

### 结束条件（三种）

CI 监控 SSE 时，以下任一条件满足即视为结束：

| 条件 | 说明 |
|------|------|
| `statusUpdate` + `agentStatus=stopped` | **主结束信号**，`brief` 区分成功/失败 |
| `chatDelta` + `finished=true` | 最后一个流式输出片段，Agent 即将停止 |
| SSE data = `[DONE]` | 标准 SSE 终止标记 |

实际观测到的事件顺序：`chatDelta(finished=true)` → `chat(assistant 完整回复)` → `statusUpdate(stopped)`。
CI 代码以 `statusUpdate(stopped)` 为准结束，`chatDelta(finished)` 仅记录日志。

### Step 5: 下载结果文件

Agent 完成后，通过文件 API 获取产出。

```
GET /v1/sessions/{session_id}/files
→ {code: 200, data: [{path: "claw-1/optimization_report.md", size: 10743, ...}, ...]}

GET /v1/sessions/{session_id}/files/{path}/stream
→ 文件内容（二进制）
```

注意：`/download` 返回的是 `stream` 路径的重定向，直接用 `/stream` 即可。

典型文件列表：

| 文件 | 说明 |
|------|------|
| `claw-1/optimization_report.md` | 优化报告 |
| `claw-1/kernel_tasks.json` | GEAK/Claude kernel 任务 ID |
| `claw-1/results/baseline_*/baseline_*.json` | baseline benchmark 结果 |
| `claw-1/results/sweep/results.tsv` | sweep 结果 |
| `claw-1/results/*/server.log` | 各阶段 server 日志 |

## CI 关注的问题

1. **Session 超时**：如果 Agent 执行时间超过 sandbox_timeout（默认 4h），CI 侧主动关闭 SSE 连接。Claw 侧 Sandbox 是否有对应的超时清理机制？

2. **HITL（人工介入）**：Skill 执行过程中如果触发 `permissionRequest`（如需要确认删除操作），CI 无法交互。Agent 的 Skill 配置需要确保不触发 HITL，或者 Claw 侧有自动 approve 机制。

3. **结果获取**：Agent 产出的文件（优化报告等）写在 NFS 上，CI 通过 NFS 路径直接读取。不通过 Claw API 获取结果。

4. **并发 Session**：V1 阶段串行执行模型（一个 session 完成后再创建下一个）。后续如果需要并行，需要确认 Claw 侧的 Sandbox 资源限制。

5. **认证**：CI 使用 `Authorization: Bearer ak-xxx`（SaFE API Key）。不带认证也能调用（`user_id=default`），但建议带上以便追踪归属。
