# Hyperloom 全本地模式

> **面向用户自有基础设施的本地节点支持** — 在自有 GPU 节点上（Docker 或 K8s）运行完整的 Hyperloom 推理优化流程，无需依赖 AMD 托管的 PrimusClaw 沙箱或 Primus-SaFE 创作 Pod。架构详情参见 [DESIGN.zh.md](DESIGN.zh.md)。

## 前置条件

- 支持 AMD ROCm 的 Docker，或具备 AMD GPU 节点的 K8s 集群
- 带有 Remote SSH 扩展的 Cursor IDE
- 用于 GEAK 内核优化的 LLM API 密钥
- 用于 `oob` CLI（Claude Code / Codex 后端）的 OOB API 密钥和 base URL

## 快速开始（Docker）

### 1. 启动容器

```bash
docker run -d --shm-size=16g \
  --device=/dev/kfd --device=/dev/dri \
  -v /path/to/models:/models \
  -p 20022:22 \
  -e LLM_API_KEY=<your-geak-api-key> \
  -e LLM_API_BASE=https://<your-openai-compatible-endpoint>/v1 \
  -e GEAK_MODEL_NAME=<model-supported-by-that-endpoint> \
  primussafe/hyperloom-fully-local:sglang-423-4
```
> vllm 镜像：docker.io/primussafe/hyperloom-fully-local:vllm-423-1

> `LLM_API_KEY` 和 `LLM_API_BASE` 仅供 `geak` 内核优化后端使用。`GEAK_MODEL_NAME` 需设置为你的端点实际提供的模型；若不设置，默认为 `claude-opus-4-7`。如果使用 OOB 的 `codex` / `claude` 后端，请配置 `OOB_API_KEY` 和 `OOB_BASE_URL`。

**可选环境变量**（按需添加）：

| 环境变量 | 用途 |
|---------|------|
| `HIP_VISIBLE_DEVICES=0,1` | 限制可用 GPU |
| `GEAK_MODEL_NAME=<model>` | 覆盖渲染到本地 LiteLLM 配置中的 GEAK 模型 |
| `OOB_API_KEY=<key>` | 统一 OOB API 密钥（Claude/Codex 共用） |
| `OOB_BASE_URL=<url>` | 统一 OOB API 端点（推荐） |

> `--shm-size=16g` 对于多 GPU 推理是必需的（RCCL 使用共享内存），默认的 64MB 会导致错误。

### 2. 配置 SSH

添加到 `~/.ssh/config`（Linux/macOS）或 `C:\Users\<you>\.ssh\config`（Windows）：

```
Host hyperloom
    HostName <gpu-machine-ip>
    Port 20022
    User root
```

> 如果 Docker 运行在本地机器上，将 `HostName` 设为 `localhost`。

### 3. 使用 Cursor 连接

1. 打开 Cursor → Remote SSH → Connect to Host → `hyperloom`（用户：`root`，密码：`root`）
2. 打开文件夹：`/opt/hyperloom`
3. Skills 自动加载

> 全本地模式**不运行持久化 MCP 服务** — TraceLens、GEAK 和 OOB 均以容器内 CLI 方式调用（`tracelens-*`、`geak` 通过 Ray、`oob run`）。无需启用任何 MCP 开关。

### 4. 运行优化

在 Cursor 聊天中输入：

```
@inference-optimization Optimize /models/Qwen3-30B-A3B
```

Agent 会自动检测模式、框架、GPU 数量和 InferenceX 路径。仅在需要时指定额外参数：

```
@inference-optimization Optimize /models/Qwen3-30B-A3B

TP=8, CONC=64, ISL=1024, OSL=1024
Precision: FP8
GPU type: MI300X
Must optimize at least 5 kernels.
Execute the full skill pipeline (Phase 0-10), including parameter sweep.
Save results to /opt/hyperloom/results/
```

**内核优化后端** — 使用 `KERNEL_OPT_BACKENDS` 或提示词指定后端；多个后端可并行运行，保留最优结果：

| 后端 | 说明 | 耗时 | 依赖 |
|------|------|------|------|
| `geak` | 本地子进程，具有 GPU 访问和硬件验证能力 | 2-3 小时 | `LLM_API_KEY` + 匹配的 `GEAK_MODEL_NAME` / `LLM_API_BASE` |
| `codex` | Codex 代码生成 + 本地基准测试 | ~1 小时 | `OOB_API_KEY` + `OOB_BASE_URL` |
| `claude` | Claude Code 生成 + 本地基准测试 | ~1 小时 | `OOB_API_KEY` + `OOB_BASE_URL` |

在提示词中指定后端（默认 `geak`，也可通过 `KERNEL_OPT_BACKENDS` 修改）：

```
@inference-optimization Optimize /models/Qwen3-30B-A3B

# 使用 Codex 后端（需要 OOB_API_KEY + OOB_BASE_URL）
Use only codex as the kernel optimization backend.

# 使用 Claude 后端（需要 OOB_API_KEY + OOB_BASE_URL）
Use only claude as the kernel optimization backend.
```

## Kubernetes 部署

同一容器镜像可作为 K8s Pod 的基础镜像。当 K8s 覆盖容器 CMD 时，`/etc/profile.d/hyperloom.sh` 会渲染 GEAK 配置，在首次 SSH 登录时启动 Ray，并在设置了 `OOB_BASE_URL` 时启动本地 auth-proxy，将 `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` 重写为 `http://127.0.0.1:4002/...`。

Pod 示例配置：

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
    env:
    - name: LLM_API_KEY
      valueFrom:
        secretKeyRef:
          name: hyperloom-secrets
          key: llm-api-key
    - name: LLM_API_BASE
      value: "https://<your-openai-compatible-endpoint>/v1"
    - name: GEAK_MODEL_NAME
      value: "<model-supported-by-that-endpoint>"
    - name: OOB_API_KEY
      valueFrom:
        secretKeyRef:
          name: hyperloom-secrets
          key: oob-api-key
          optional: true
    - name: OOB_BASE_URL
      value: "https://<your-oob-endpoint>/v1"
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
```

通过 Cursor Remote SSH 连接 → `<node-ip>:30022`，打开 `/opt/hyperloom`。

## 容器端口

| 内部端口 | 服务 |
|---------|------|
| 22   | SSH（Cursor Remote SSH） — 唯一对外暴露的端口 |
| 6379 | Ray head（GEAK GPU 调度，内部） |
| 8265 | Ray dashboard（内部） |
| 4002 | OOB auth-proxy — 仅当设置 `OOB_BASE_URL` 时存在（内部） |

> TraceLens、GEAK 和 OOB **不监听任何端口** — 它们以 CLI 方式调用（`tracelens-*`、`geak` 通过 Ray、`oob run`）。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | — | GEAK 内核优化的 LLM API 密钥 |
| `LLM_API_BASE` | — | LLM API 端点 URL |
| `GEAK_MODEL_NAME` | `claude-opus-4-7` | 渲染到生成的 LiteLLM 配置中的 GEAK 模型名称 |
| `GEAK_API_KEY` | 回退到 `LLM_API_KEY` | 可选的 GEAK 专用 API 密钥覆盖 |
| `GEAK_BASE_URL` | 回退到 `LLM_API_BASE` | 可选的 GEAK 专用端点覆盖 |
| `FRAMEWORK` | `sglang` | 推理框架（`sglang` 或 `vllm`） |
| `OOB_API_KEY` | — | 统一 OOB API 密钥（Claude/Codex 的 `oob run` 调用共用） |
| `OOB_BASE_URL` | — | 统一 OOB API 端点（设置后，容器内 `:4002` auth-proxy 重写 Bearer 认证） |
| `OOB_CLI` | `oob` | OOB CLI 可执行文件名；仅在 `pip install` 到其他位置时覆盖 |
| `OOB_HOME` | `~/.oob` | `oob run` 存储任务工作区和 SQLite 数据库的根目录 |
| `HIP_VISIBLE_DEVICES` | — | 逗号分隔的 GPU 索引（如 `0,1,2`） |
| `GPUS_PER_NODE` | — | 覆盖 entrypoint 显示的 GPU 数量 |

## 日志

服务日志写入 `/var/log/hyperloom/`：

```bash
tail -f /var/log/hyperloom/ray-head.log         # Ray（GEAK GPU 调度器）
tail -f /var/log/hyperloom/oob-auth-proxy.log   # OOB auth proxy（仅当设置 OOB_BASE_URL 时）
```

> 每个任务的 CLI 日志不写入 `/var/log/hyperloom`。`oob run` 将文件存储在 `${OOB_HOME:-~/.oob}/tasks/cli/<task_id>/workspace/` 下（如 `execution.log`），而 `geak` 将结果写入其自身的输出目录。

## 安全

默认 SSH 密码为 `root`。首次登录后请修改：

```bash
passwd root
```

或挂载你的 SSH 密钥：

```bash
docker run ... -v ~/.ssh/id_rsa.pub:/root/.ssh/authorized_keys:ro ...
```

## 故障排查

**后台服务未运行（K8s Pod）**

Ray 和可选的 OOB auth-proxy 在首次 SSH 登录时初始化。同一启动脚本还会渲染 `GEAK_CONFIG`，并在设置 `OOB_BASE_URL` 时将 OOB base URL 重写到本地 `:4002` 代理。如果未启动，手动执行：

```bash
source /etc/profile.d/hyperloom.sh
```

检查状态：

```bash
ray status                                           # Ray head
ss -tlnp | grep -E ':6379|:8265|:4002' || true       # 监听端口
command -v oob && oob --help | head -5               # OOB CLI 已安装
command -v geak                                      # GEAK CLI 已安装
python3 -c "import TraceLens" && echo "TraceLens OK" # TraceLens 可导入
printf 'OPENAI_BASE_URL=%s\nANTHROPIC_BASE_URL=%s\n' "$OPENAI_BASE_URL" "$ANTHROPIC_BASE_URL"
```

**GPU 数量显示错误**

entrypoint 按以下顺序检查：`GPUS_PER_NODE` → `HIP_VISIBLE_DEVICES` → `ROCR_VISIBLE_DEVICES` → `amd-smi` → `rocm-smi`。设置环境变量可覆盖硬件扫描结果。
