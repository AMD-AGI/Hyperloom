# Hyperloom 本地模式

在本地机器上运行完整的 Hyperloom 推理优化闭环。

## 核心组件

| 组件 | 端口 | 说明 |
|------|------|------|
| **TraceLens MCP** | 8001 | 推理 Profiling 服务，采集并分析 GPU kernel 执行轨迹，为优化提供数据依据 |
| **GEAK REST API** | 8000 | GEAK 后端 HTTP 接口，负责 kernel 优化任务的调度与结果存储 |
| **GEAK MCP** | 8002 | Kernel 优化后端之一：在远程 GPU Pod 上完成代码生成与硬件验证，耗时 10–30 min |
| **OOB Agent MCP** | 8003 | Kernel 优化后端之一：集成 Claude Code / Codex, 耗时 2–6 min；可完全替代 GEAK |

## 前置条件

- 支持 Docker 或 K8s 集群
- Cursor IDE（安装 Remote SSH 插件）
- LLM API Key（用于 GEAK 内核优化）
- Anthropic / OpenAI API Key（用于 OOB Agent MCP 的 Claude Code / Codex 后端）

## 快速开始（Docker）

### 1. 启动容器

```bash
docker run -d --shm-size=16g \
  --device=/dev/kfd --device=/dev/dri \
  -v /path/to/models:/models \
  -p 20022:22 -p 20001:8001 -p 20002:8002 -p 20003:8003 \
  -e LLM_API_KEY=<your-api-key> \
  -e LLM_API_BASE=https://api.openai.com/v1 \
  hyperloom-local:sglang-latest
```

**可选环境变量**（按需添加）：

| 环境变量 | 用途 |
|---------|------|
| `HIP_VISIBLE_DEVICES=0,1` | 限制使用指定 GPU |
| `ANTHROPIC_API_KEY=<key>` | 启用 Claude Code 内核优化后端 |
| `ANTHROPIC_BASE_URL=<url>` | Claude API 端点（不填则使用 api.anthropic.com）|
| `OPENAI_API_KEY=<key>` | 启用 Codex 内核优化后端 |
| `OPENAI_BASE_URL=<url>` | OpenAI API 端点（不填则使用 api.openai.com）|

> `--shm-size=16g` 是多 GPU 推理的必要参数（RCCL 使用共享内存进行 GPU 间通信），默认的 64MB 会导致报错。

### 2. 配置 SSH

在 `~/.ssh/config`（Linux/macOS）或 `C:\Users\<你的用户名>\.ssh\config`（Windows）中添加：

```
Host hyperloom
    HostName <gpu-machine-ip>
    Port 20022
    User root
```

> 如果 Docker 运行在本机，将 `HostName` 设为 `localhost`。

### 3. 使用 Cursor 连接

1. 打开 Cursor → Remote SSH → Connect to Host → `hyperloom`（用户名：`root`，密码：`root`）
2. 打开目录：`/opt/hyperloom`
3. Skills 和 MCP 服务器自动加载

### 4. 运行优化

在 Cursor 聊天框中输入：

```
@inference-optimization Optimize /models/Qwen3-30B-A3B
```

Agent 会自动从容器环境中检测模式、框架、GPU 数量和 InferenceX 路径。仅在需要时指定额外参数：

```
@inference-optimization Optimize /models/Qwen3-30B-A3B

TP=8, CONC=64, ISL=1024, OSL=1024
Precision: FP8
GPU type: MI355X
Must optimize at least 5 kernels.
Execute the full skill pipeline (Phase 0-10), including parameter sweep.
Save results to /opt/hyperloom/results/
```

**内核优化后端** — 通过 `KERNEL_OPT_BACKENDS` 或提示词控制，支持并发跑多个后端取最优结果：

| 后端 | 说明 | 耗时 | 依赖 |
|------|------|------|------|
| `geak` | 本地子进程，有 GPU 访问，硬件验证 | 10–30 min | `LLM_API_KEY` |
| `codex` | Codex 代码生成，本地 benchmark | 2–6 min | `OPENAI_API_KEY` |
| `claude` | Claude Code 代码生成，本地 benchmark | 3–15 min | `ANTHROPIC_API_KEY` |

在提示词中指定后端（默认 `geak`，可通过提示词或环境变量 `KERNEL_OPT_BACKENDS` 修改）：

```
@inference-optimization Optimize /models/Qwen3-30B-A3B

# 用 Codex 后端（需要 OPENAI_API_KEY）
Use only codex as the kernel optimization backend.

# 用 Claude Code 后端（需要 ANTHROPIC_API_KEY）
Use only claude as the kernel optimization backend.
```

## Kubernetes 部署

同一镜像可作为 K8s Pod 的基础镜像。当 K8s 覆盖容器 CMD 时，MCP 服务会在首次登录时通过 `/etc/profile.d/hyperloom.sh` 自动启动。

Pod 配置示例：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hyperloom
  labels:
    app: hyperloom
spec:
  containers:
  - name: hyperloom
    image: hyperloom-local:sglang-latest
    ports:
    - containerPort: 22
      name: ssh
    - containerPort: 8001
      name: tracelens
    - containerPort: 8002
      name: geak
    - containerPort: 8003
      name: oob-agent
    env:
    - name: LLM_API_KEY
      valueFrom:
        secretKeyRef:
          name: hyperloom-secrets
          key: llm-api-key
    - name: LLM_API_BASE
      value: "https://api.deepseek.com/v1"
    - name: ANTHROPIC_API_KEY
      valueFrom:
        secretKeyRef:
          name: hyperloom-secrets
          key: anthropic-api-key
          optional: true
    resources:
      limits:
        amd.com/gpu: 1
    volumeMounts:
    - name: models
      mountPath: /models
  volumes:
  - name: models
    hostPath:
      path: /shared/models
---
apiVersion: v1
kind: Service
metadata:
  name: hyperloom-svc
spec:
  type: NodePort
  selector:
    app: hyperloom
  ports:
  - name: ssh
    port: 22
    targetPort: 22
    nodePort: 30022
  - name: tracelens
    port: 8001
    targetPort: 8001
    nodePort: 30001
  - name: geak
    port: 8002
    targetPort: 8002
    nodePort: 30002
  - name: oob-agent
    port: 8003
    targetPort: 8003
    nodePort: 30003
```

通过 Cursor Remote SSH 连接 `<node-ip>:30022`，打开 `/opt/hyperloom`。

## 容器端口

| 内部端口 | 服务 |
|---------|------|
| 22   | SSH（Cursor Remote SSH） |
| 8001 | TraceLens MCP |
| 8002 | GEAK MCP |
| 8003 | OOB Agent MCP（Claude Code / Codex） |

> 通过 `docker run -p` 或 K8s NodePort/LoadBalancer 映射到任意外部端口。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | — | GEAK 内核优化使用的 LLM API Key |
| `LLM_API_BASE` | — | LLM API 端点 URL |
| `FRAMEWORK` | `sglang` | 推理框架（`sglang` 或 `vllm`） |
| `TRACELENS_PORT` | `8001` | TraceLens MCP 端口 |
| `GEAK_MCP_PORT` | `8002` | GEAK MCP 端口 |
| `OOB_MCP_PORT` | `8003` | OOB Agent MCP 端口 |
| `ANTHROPIC_API_KEY` | — | Anthropic API Key（Claude Code 后端） |
| `ANTHROPIC_BASE_URL` | — | Anthropic API 端点（企业 LLM 网关时使用） |
| `OPENAI_API_KEY` | — | OpenAI API Key（Codex 后端） |
| `OPENAI_BASE_URL` | — | OpenAI API 端点 |
| `HIP_VISIBLE_DEVICES` | — | 逗号分隔的 GPU 索引（如 `0,1,2`） |
| `GPUS_PER_NODE` | — | 覆盖 entrypoint 显示的 GPU 数量 |

## 日志

服务日志写入 `/var/log/hyperloom/`：

```bash
tail -f /var/log/hyperloom/tracelens.log
tail -f /var/log/hyperloom/geak-api.log
tail -f /var/log/hyperloom/geak-mcp.log
tail -f /var/log/hyperloom/oob-mcp.log
```

## 安全

默认 SSH 密码为 `root`，首次登录后请修改：

```bash
passwd root
```

或挂载 SSH 公钥：

```bash
docker run ... -v ~/.ssh/id_rsa.pub:/root/.ssh/authorized_keys:ro ...
```

## 故障排查

**K8s Pod 中 MCP 服务未启动**

服务在首次登录时启动。如果没有自动启动，手动执行：

```bash
source /etc/profile.d/hyperloom.sh
```

检查服务状态：

```bash
curl -s http://localhost:8001/mcp > /dev/null && echo "TraceLens OK" || echo "TraceLens 未运行"
curl -s http://localhost:8000/health > /dev/null && echo "GEAK API OK" || echo "GEAK API 未运行"
curl -s http://localhost:8002/ > /dev/null && echo "GEAK MCP OK" || echo "GEAK MCP 未运行"
curl -s http://localhost:8003/ > /dev/null && echo "OOB Agent OK" || echo "OOB Agent 未运行"
```

**GPU 数量显示不正确**

entrypoint 按以下顺序检测：`GPUS_PER_NODE` → `HIP_VISIBLE_DEVICES` → `ROCR_VISIBLE_DEVICES` → `amd-smi` → `rocm-smi`。设置环境变量可覆盖硬件检测结果。
