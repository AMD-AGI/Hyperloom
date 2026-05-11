# 远程模式 — 完整执行参考

本文档包含所有远程模式专属说明。使用远程客户端连接 SaFE 集群时，**开始前必须阅读**。

**Agent：** 阅读 `SKILL.md` 了解编排循环和共享铁律（IR-1 到 IR-7b）。本文档定义远程模式专属铁律（IR-8 到 IR-11）、常量、架构，以及每个 action 的远程执行覆盖说明。

## 环境

- **Client**：远程客户端（外部编排环境）
- **Runtime**：带多节点 GPU 的 SaFE 集群
- **MCP Servers**：只使用 SaFE MCP（RayJob 生命周期）
- **OOB**：CLI（`oob_ray_submit.py run`），用于 Codex/Claude 优化任务
- **TraceLens**：CLI（`pip install -e /hyperloom/TraceLens-internal`）
- **Storage**：共享 NFS（Pod 内为 `/wekafs/`，映射到 NFS 根）

## 模式检测

当存在远程客户端上下文，或用户指定 `Mode: remote` 时，自动检测为远程模式。

```bash
MODE="remote"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/wekafs/inference-optimization}"
```

---

## 远程模式铁律

### IR-8：所有 GPU 侧命令都使用 Ray Dashboard REST

所有运行在 Ray 集群上的命令都必须通过 Ray Dashboard REST（`POST /api/jobs/`）提交。**绝不要**从 sandbox 用 Ray Client、`ray://<head>:10001` 或手写 Ray client submit wrapper 驱动集群。

```bash
# 正确
curl -sS -X POST "$RAY_DASHBOARD_URL/api/jobs/" \
  -H "Content-Type: application/json" \
  -d '{"entrypoint":"bash -lc '\''source /etc/profile.d/hyperloom-env.sh && bash $SCRIPTS_DIR/run_baseline.sh'\''"}'

# 错误：使用 sandbox 侧 Ray client 或自定义 Ray submit wrapper
```

### IR-9：主推理 workload 必须使用 `kind: "RayJob"`

持久推理集群**必须**是 `kind: "RayJob"`。内核优化后端通过 RayJob 上的 OOB CLI wrapper（`oob_ray_submit.py`）运行。skill 本身绝不能创建 PyTorchJob workload。

### IR-10：SaFE MCP 只允许 `workload_create(kind="RayJob")` 和 `workload_stop`

远程模式下，skill 只能将 SaFE MCP 用于：

- **`workload_create`** 且 `kind: "RayJob"`：创建推理集群
- **`workload_get`** / **`workload_list`**：检查 workload 状态
- **`workload_stop`**：优化完成后停止 RayJob

**禁止的 SaFE MCP 操作：**

- **`workload_delete`**：绝不要删除 workload；使用 `workload_stop`
- **`workload_create` 使用 `"RayJob"` 以外的任何 kind**：不要 PyTorchJob，也不要其他类型。内核优化由 Ray 上的 OOB CLI wrapper 处理；skill 不能直接创建其他 workload 类型。

违规 = 立即判定本次运行无效。

### IR-10a：创建新 RayJob，不使用已有 RayJob

每次 skill 执行开始时，都必须通过 `workload_create` 创建一个新的 RayJob。不要连接、恢复或复用之前运行留下的已有 RayJob，即使它看起来是健康的；陈旧状态会使优化结果无效。

### IR-11：OOB 仅 CLI 使用，不使用 service API

与 `SKILL.md` 中 IR-7 相同：绝不要修改 OOB 的配置、认证文件、测试数据或设置。只允许使用批准的 CLI 工具：

- OOB：`oob_ray_submit.py run`
- TraceLens：`TraceLens_generate_perf_report_pytorch_inference` 和 `orchestrator_prepare.py`

---

## 远程模式常量

这些常量补充 `SKILL.md` 中的共享常量。

| 常量 | 值 | 说明 |
|----------|-------|-------------|
| `RAY_DASHBOARD_PORT` | 8265 | RayJob head node 上的 Ray Dashboard REST 端口 |

### 镜像选择（远程模式）

使用 `KERNEL_OPT_IMAGE`（由 CI 或用户提供）。远程模式下，CI 应提供带 Ray patch 的 SGLang 镜像，例如包含 Ray 2.44.1 修复的 `harbor.../custom/lmsysorg/sglang:202603270958`。RayJob 和 kernel-opt 后端使用同一个镜像。

### 镜像构建（Dockerfile）

自定义 SGLang 镜像基于上游 SGLang，并包含 Ray 兼容性修复：

```dockerfile
FROM harbor.oci-slc.primus-safe.amd.com/proxy/lmsysorg/sglang:v0.5.9-rocm700-mi35x
RUN python -m pip install ray[default]==2.44.1 click==8.1.7
```

**原因：** 上游 SGLang 镜像自带的 Ray 版本存在 `copy.deepcopy` / `Sentinel` enum 崩溃问题，会破坏 RayJob 的 `ray start` 和 `ray-job-submitter`。`click==8.1.7` 用于固定 click 版本，避免 Ray CLI API 不兼容。

---

## RayJob 架构

```
Remote Client --> Skill (SKILL.md)
                 |-> SaFE MCP (workload_create kind="RayJob")
                 |     |-> 创建 Ray Cluster：head（1 个 pod）+ workers（N 个 pod）
                 |     |-> 暴露 Ray Dashboard REST 端口（8265）
                 |     \-> env.RAY_JOB_ENTRYPOINT = "tail -f /dev/null"（保持集群存活）
                 |
                 |-> Ray Dashboard REST
                 |     \-> POST http://<head>:8265/api/jobs/ -> 在 RayJob 镜像内运行
                 |
                 |-> oob_ray_submit.py run -a {claude,codex} -p "..." -f kernel.py -o <work_dir>
                 |     |-> Ray head 使用 GPU 隔离调度任务
                 |     |-> worker 启动 claude/codex CLI 子进程，并阻塞等待完成
                 |     \-> 优化后的文件写入 <work_dir>/<task_id>/workspace/
                 |
                 |-> TraceLens CLI
                 |     \-> 从共享 NFS 离线分析 trace
                 |
                 \-> benchmark 结果写入共享 NFS
```

---

## RayJob 生命周期

### 创建新的 RayJob

**每次 skill 执行严格只创建一个 RayJob。** 开始时创建一个新的 RayJob，并在整个运行过程中复用它。不要在执行中途创建第二个 RayJob。不要复用之前 skill 执行留下的 RayJob，因为它们可能有陈旧状态。优化完成后，使用 `workload_stop` 停止 RayJob。

> **关键：`RAY_JOB_ENTRYPOINT` 必须 base64 编码。** 明文会导致
> `base64: invalid input` → exit code 127。
> 编码方式：`echo -n "tail -f /dev/null" | base64` → `dGFpbCAtZiAvZGV2L251bGw=`

**单节点（NUM_NODES=1）：**

```
Tool: workload_create
Args: {
    "display_name": "inference-opt-<model_short_name>",
    "workspace_id": "<user_workspace_id>",
    "kind": "RayJob",
    "images": ["KERNEL_OPT_IMAGE"],
    "resources": [
        {"replica": 1, "cpu": "96", "gpu": "8", "memory": "1024Gi", "sharedMemory": "500Gi", "ephemeralStorage": "500Gi"}
    ],
    "env": { "RAY_JOB_ENTRYPOINT": "dGFpbCAtZiAvZGV2L251bGw=" },
    "is_tolerate_all": true,
    "ttl_seconds_after_finished": 600
}
```

**多节点（NUM_NODES>1）：**

```
Tool: workload_create
Args: {
    "display_name": "inference-opt-<model_short_name>",
    "workspace_id": "<user_workspace_id>",
    "kind": "RayJob",
    "images": ["KERNEL_OPT_IMAGE", "KERNEL_OPT_IMAGE"],
    "resources": [
        {"replica": 1, "cpu": "96", "gpu": "8", "memory": "1024Gi", "sharedMemory": "256Gi", "ephemeralStorage": "500Gi"},
        {"replica": 1, "cpu": "96", "gpu": "8", "memory": "1024Gi", "sharedMemory": "256Gi", "ephemeralStorage": "500Gi"}
    ],
    "env": { "RAY_JOB_ENTRYPOINT": "dGFpbCAtZiAvZGV2L251bGw=" },
    "is_tolerate_all": true,
    "ttl_seconds_after_finished": 7200
}
```

对于 N>2 个节点，将 `resources[1].replica` 设置为 `N-1`。TP = GPU_PER_NODE × NUM_NODES。

### 等待 RayJob Ready 并获取 Ray Dashboard 地址

```python
import time

RAYJOB_ID = "<workload_id from create response>"

for attempt in range(60):
    result = workload_get(workload_id=RAYJOB_ID)
    phase = result.get("status", {}).get("phase", "")
    if phase == "Running":
        break
    elif phase in ("Failed", "Stopped"):
        raise RuntimeError(f"RayJob failed: {phase}")
    time.sleep(30)

pods = result.get("status", {}).get("pods", [])
head_pod = next((p for p in pods if "head" in p.get("name", "")), pods[0] if pods else None)
HEAD_IP = head_pod.get("ip", head_pod.get("podIP", ""))
RAY_DASHBOARD_URL = f"http://{HEAD_IP}:8265"
```

### 设置远程执行环境

```bash
export MODE=remote
export HEAD_IP="<HEAD_IP>"
export RAY_DASHBOARD_URL="http://<HEAD_IP>:8265"
export MODEL="<model_path_on_shared_nfs>"
export TP=<total_gpu_count>
export CONC=<concurrency>
export ISL=1024
export OSL=256
export FRAMEWORK=<sglang|vllm>
export INFERENCEX_PATH="<path_on_shared_nfs>"
export SKILL_ROOT="${SKILL_ROOT:-/wekafs/yunkai/Hyperloom/.cursor/skills/inference-optimization}"
export SCRIPTS_DIR="$SKILL_ROOT/scripts"
export OOB_RAY_CLI="python3 $SCRIPTS_DIR/oob_ray_submit.py"
export OOB_CLI="${OOB_CLI:-/opt/venv/bin/oob}"
export TRACELENS_ROOT="${TRACELENS_ROOT:-/opt/hyperloom/TraceLens}"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
export RESULT_DIR="/wekafs/inference-optimization/results/${TIMESTAMP}"
export TRACE_DIR="/wekafs/inference-optimization/traces/${TIMESTAMP}"

# 验证 Ray 集群
curl -sS -X POST "$RAY_DASHBOARD_URL/api/jobs/" \
  -H "Content-Type: application/json" \
  -d '{"entrypoint":"bash -lc '\''source /etc/profile.d/hyperloom-env.sh && python3 -c \"import ray; ray.init(); print(ray.cluster_resources()); ray.shutdown()\"'\''"}'
```

### 远程路径约定

远程模式区分两个文件系统写入域：

- **RayJob 侧运行产物必须使用 `/wekafs/...` 路径。** 这包括 `$RESULT_DIR`、
  `$TRACE_DIR`、RayJob 内生成的 TraceLens 输出、OOB 输入/输出 workspace、
  kernel candidate、re-baseline 产物和 sweep 输出。这些路径必须对 RayJob
  head/worker pod 可见。
- **sandbox 侧生成的文件必须使用 `/workspace/hyperloom/...` 路径。** Claw
  sandbox 可以读取 `/wekafs` 作为输入，但不能写入。sandbox 创建的摘要、
  manifest、临时 prompt、复制的 metadata 或最终用户可见产物，都必须写到
  `/workspace/hyperloom/`。
- **最终交付物必须出现在 `/workspace/hyperloom/` 下。** RayJob 工作结束后，
  将相关 `/wekafs` RayJob 产物复制或汇总到
  `/workspace/hyperloom/optimization_report.md`、可选的
  `/workspace/hyperloom/ci_metrics.json`，以及必要的结果摘要/manifest。
  Claw 会在 session 结束时上传 `/workspace/hyperloom/`。

### RayJob 内 bootstrap / CLI 预检查

RayJob 进入 Running 后，复用 `scripts/bootstrap.sh` 中的 BYOI bootstrap 逻辑，确保 CLI 依赖存在于 RayJob 镜像内。它会安装或验证 OOB、TraceLens、Ray/click 兼容性，以及 `codex` / `claude` CLI。生成的环境文件可复用，但 source 后要再次强制 `MODE=remote`。

```bash
Ray Dashboard REST job entrypoint:
export PATH='/opt/venv/bin:'\"\$PATH\"
export MODE=remote
export SKILL_ROOT='$SKILL_ROOT'
export HYPERLOOM_BUNDLE='\${HYPERLOOM_BUNDLE:-/wekafs/fully-local}'

if [ ! -f /opt/hyperloom/.bootstrap_done ] && [ -d \"\$HYPERLOOM_BUNDLE\" ]; then
  bash '$SCRIPTS_DIR/bootstrap.sh'
fi

[ -f /etc/profile.d/hyperloom-env.sh ] && . /etc/profile.d/hyperloom-env.sh
export MODE=remote
export OOB_CLI='\${OOB_CLI:-/opt/venv/bin/oob}'

command -v oob || echo 'WARN: oob CLI missing'
command -v codex || command -v claude || echo 'WARN: codex/claude CLI missing'
command -v TraceLens_generate_perf_report_pytorch_inference || echo 'WARN: TraceLens CLI missing'
"
```

### 清理：停止 RayJob

优化完成并生成报告后：

```
Tool: workload_stop
Args: { "workload_id": "<RAYJOB_ID>" }
```

如果使用过 SaFE Option B 的并行 sweep，也停止这些并行 sweep workload：

```python
sweep_workloads = workload_list(workspace_id=WORKSPACE_ID, kind="RayJob")
for wl in sweep_workloads:
    if wl["displayName"].startswith("sweep-"):
        workload_stop(wl["workloadId"])
```

---

## 每个 Action 的远程覆盖说明

`actions/*.md` 中包含共享 action 逻辑。下面是每个 action 在远程模式下的执行细节。

### Setup（`actions/setup.md`）

远程模式下，在环境检测后先创建 RayJob（见上方 “RayJob Lifecycle”），然后再进入 classify/baseline。

### 远程模式中的 OOB

OOB 没有持久服务；每个 Codex / Claude 任务都是一次阻塞的 `oob_ray_submit.py run` 调用。Ray 通过 `HIP_VISIBLE_DEVICES` 分配 GPU 隔离。

```bash
# $OOB_RAY_CLI = "python3 $SCRIPTS_DIR/oob_ray_submit.py"
$OOB_RAY_CLI status

$OOB_RAY_CLI run \
  -a codex \
  -p "Optimize this Triton kernel ... (see prompt template in kernel-opt/codex.md)" \
  -f $WORK_DIR/kernel.py \
  -o $WORK_DIR/oob_codex_${KERNEL_NAME} \
  --max-turns 20 \
  --timeout 1200 \
  --no-live --json
```

输出文件位于：

```
<output-dir>/tasks/<user>/<task_id>/workspace/optimized_kernel.py
<output-dir>/tasks/<user>/<task_id>/workspace/execution.log
<output-dir>/tasks/<user>/<task_id>/workspace/trajectory_round_*.jsonl
```

使用 JSON 的 `.workspace` 字段直接获取 workspace 目录。没有单独的 `output/` 子目录。

RayJob 镜像必须在 `PATH` 中包含 `/opt/venv/bin/oob` 和所选后端 CLI（`codex` 或 `claude`）。如果 `oob_ray_submit.py` 转发的客户端 `PATH` 隐藏了 `/opt/venv/bin`，设置 `OOB_CLI=/opt/venv/bin/oob`。

额外的 DSR1 RayJob 陷阱：

- 始终将 pod 可见的 `/wekafs/...` 路径传给 `-f`、`-o` 和运行时文件引用。客户端路径如 `/mnt/weka/...` 在 worker 内无效。
- 从外部 Cursor 启动时使用 `OOB_CLI=/opt/venv/bin/oob`；否则转发的客户端 `PATH` 可能隐藏 pod venv。
- 后端 CLI 必须安装并完成认证。Codex 可能需要 `/root/.codex/auth.json` 或等效 `--api-key` 路径；仅 API key env 可能不够。
- `OPENAI_BASE_URL` 应指向 Core42 proxy：
  `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1`。
- `ANTHROPIC_BASE_URL` 必须指向 Core42 Anthropic-compatible direct gateway，
  不能指向本地 OOB auth proxy：
  `https://core42.primus-safe.amd.com/api/v1/llm-proxy`。
- `ANTHROPIC_CUSTOM_HEADERS` 必须包含 bearer header：
  `Authorization: Bearer ${ANTHROPIC_API_KEY}`。
- 在 Core42 上不要让 Claude/OOB 使用
  `ANTHROPIC_BASE_URL=http://127.0.0.1:4002/...`。该 auth-proxy 路径已知会对有效 Claude model 返回 `404`。
- **Core42 Claude/OOB 认证 smoke test：** 如果 `oob run -a claude` 或 `claude --print`
  对已知有效模型（如 `claude-opus-4-6` 或 `claude-opus-4-7`）报 model-not-found/access，
  只用下面这个一次性请求验证 Core42 Anthropic-compatible endpoint、key、headers 和模型名是否有效：
  ```bash
  curl -sS "${ANTHROPIC_BASE_URL}/v1/messages" \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -H "content-type: application/json" \
    -d '{"model":"claude-opus-4-6","max_tokens":8,"messages":[{"role":"user","content":"Reply OK only."}]}'
  ```
  这个请求**不是**内核优化路径，绝不能替代 OOB。如果它成功，应该修复 OOB/Claude 环境，然后重新运行 `oob_ray_submit.py run`。
  在 Core42 上，Claude Code 可能因 bootstrap auth proxy 的 path/header 适配返回 `404`，即使上游模型有效。修复 RayJob 环境时，应使用不带 `/v1` 的 Core42 Anthropic 风格 base path，并注入 Bearer header：
  ```bash
  export ANTHROPIC_BASE_URL="https://core42.primus-safe.amd.com/api/v1/llm-proxy"
  export ANTHROPIC_CUSTOM_HEADERS="Authorization: Bearer ${ANTHROPIC_API_KEY}"

  claude --bare --print --model claude-opus-4-6 "Reply with OK only."
  ```
  通过 `oob_ray_submit.py` 调用 Claude 时，确保 `ANTHROPIC_BASE_URL` 和
  `ANTHROPIC_CUSTOM_HEADERS` 被转发到 Ray workers。`oob_ray_submit.py` 已经会转发这些变量；如果 smoke test 仍显示 `127.0.0.1:4002`，必须带 Core42 direct gateway env 重新运行 `bootstrap.sh --force`，再提交 OOB 任务。
- 成功的 OOB smoke 应返回 `status: completed`，并在 `<output-dir>/tasks/cli/<task_id>/workspace/` 下产生 workspace。

### 远程模式中的 TraceLens

TraceLens 是 CLI 工具，没有持久服务。直接在远程客户端侧，对共享 NFS 上的 trace 文件运行：

```bash
mkdir -p "$RESULT_DIR/tracelens/perf_report_csvs"
TraceLens_generate_perf_report_pytorch_inference \
  --profile_json_path "$TRACE_PATH" \
  --output_xlsx_path "$RESULT_DIR/tracelens/perf_report.xlsx" \
  --output_csvs_dir "$RESULT_DIR/tracelens/perf_report_csvs" \
  --gpu_arch_json_path "$TRACELENS_ROOT/TraceLens/AgenticMode/Standalone/utils/arch/$GPU_TYPE.json" \
  --enable_pseudo_ops \
  --group_by_num_kernels \
  --enable_kernel_summary
if [ -f "$RESULT_DIR/tracelens/perf_report_csvs/ops_summary.csv" ]; then
  python3 "$TRACELENS_ROOT/TraceLens/AgenticMode/Standalone/orchestrator_prepare.py" \
    --trace-path "$TRACE_PATH" \
    --platform "$GPU_TYPE" \
    --output-dir "$RESULT_DIR/tracelens"
else
  echo "TraceLens produced GPU-only output; use kernel_summary.csv as fallback"
fi
```

vLLM worker trace 可能只有 GPU 信息。这种情况下 inference CLI 可能只生成 `gpu_timeline.csv` 和 `kernel_summary.csv`；使用 `kernel_summary.csv` 作为 fallback candidate source。

RayJob 验证中见过的 TraceLens 陷阱：

- 对 Hyperloom CLI 验证，使用 CLI，不要使用 TraceLens MCP。
- 如果客户端侧缺少 `TraceLens_generate_perf_report_pytorch_inference`，安装：
  `pip install -e /mnt/weka/yunkai/TraceLens-internal`，或依赖 RayJob bootstrap 安装的 CLI。
- `pandas 3.x` 可能触发 dtype assignment 失败；如果 TraceLens CLI 生成报告失败，固定 `pandas<3`。

### Baseline（`actions/baseline.md`）

所有命令都必须通过 Ray Dashboard REST 提交：

```bash
POST /api/jobs entrypoint: "export MODEL='$MODEL' TP=$TP CONC=$CONC FRAMEWORK=sglang \
  SGLANG_EXTRA_ARGS='--enable-torch-compile --mem-fraction-static 0.6' \
  RESULT_DIR='$RESULT_DIR' TRACE_DIR='$TRACE_DIR' INFERENCEX_PATH='$INFERENCEX_PATH' && \
  bash $SCRIPTS_DIR/run_baseline.sh"
```

Trace 文件和结果会写入共享 NFS，远程客户端和 Ray 集群都能访问。

### Profile（`actions/profile.md`）

profiling 命令通过 Ray Dashboard REST 运行：

profile 控制必须遵循 `actions/profile.md`：如果使用 `/start_profile` 或
`/stop_profile`，它必须是被 profile 的 SGLang backend 进程（PD/MoRI 通常是
prefill）内建暴露的 endpoint。不要实现、启动或模拟任何名为
`start_profile` / `stop_profile` 的自定义 endpoint 或 sidecar 服务。
当 `run_profile.sh` 无法表达 PD/MoRI 的 prefill/decode/router 拓扑时，
应提交 `actions/profile.md` 中记录的 Ray Dashboard REST Python driver，
而不是强行套用单 server 脚本路径。

PD/MoRI 的标准远程 profile 流程是：

1. 启动被 profile 的 backend 前，环境里必须已经有
   `SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR"`。
2. benchmark traffic 打 router / public endpoint。
3. `/start_profile` 和 `/stop_profile` 只打被 profile 的 backend
   （通常是 prefill），不要打 router。
4. 默认使用立即开始的 `/start_profile` payload：
   ```json
   {"output_dir":"$TRACE_DIR","activities":["CPU","GPU"],"with_stack":true,"record_shapes":true,"profile_prefix":"prefill"}
   ```
   默认不要传 `start_step` / `num_steps`；只有 immediate profile 已经成功、
   且需要降低 trace 大小时，才使用有界 step window。
5. `/stop_profile` 后轮询 `$TRACE_DIR`，直到 trace 数量和总大小稳定。
   如果目录为空，先检查 backend server log 里的 `Traces are saved to:`，
   再判定失败。

```bash
POST /api/jobs entrypoint: "export RUN_CONTEXT_FILE='$RESULT_DIR/run_context.env' && \
  bash $SCRIPTS_DIR/run_profile.sh"
```

profiling 后，在 Ray 集群内 unset profiler 环境变量：

```bash
POST /api/jobs entrypoint: "unset PROFILE SGLANG_TORCH_PROFILER_DIR VLLM_TORCH_PROFILER_DIR"
```

共享 NFS 上的 trace 文件可被远程客户端和 TraceLens CLI 访问。

用于查找内核源码的文件系统搜索也必须通过 Ray Dashboard REST：

```bash
POST /api/jobs entrypoint: "find /sgl-workspace /opt/venv /tmp/torchinductor_root -maxdepth 4 -name '*.py' \
  -exec grep -l 'KERNEL_NAME' {} \\; 2>/dev/null"
POST /api/jobs entrypoint: "python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('/path/to/standalone_kernel.py').read_text())\nPY"
```

trace 解析可以在远程客户端侧完成（trace 在共享 NFS 上）。

### Backends（`actions/backends.md`）

`ServerArgs` 检查和所有 backend test 命令都通过 Ray Dashboard REST 运行：

```bash
POST /api/jobs entrypoint: "python3 -c \"
from sglang.srt.server_args import ServerArgs
import inspect
src = inspect.getsource(ServerArgs.__init__)
for line in src.split('\\\\n'):
    if '--' in line and ('backend' in line.lower() or 'enable' in line.lower()):
        print(line.strip())
\""
```

每个 backend switch 的完整测试循环：

```bash
POST /api/jobs entrypoint: "
# Kill server (safe pattern)
ps aux | grep 'python3 -m sglang' | grep -v grep | grep -v bash | awk '{print \$2}' | xargs -r kill -9 2>/dev/null
sleep $SERVER_KILL_WAIT_S

# Restart with new backend
export MODEL='$MODEL' TP=$TP CONC=$CONC FRAMEWORK=$FRAMEWORK
export SGLANG_EXTRA_ARGS='$BASELINE_ARGS $NEW_BACKEND_SWITCH'
export RESULT_DIR='$RESULT_DIR/backend_test_$SWITCH_NAME'
bash $SCRIPTS_DIR/run_baseline.sh
"
```

kill 后验证 Ray cluster 仍存活：
通过 Ray Dashboard REST 提交 `curl -s http://localhost:8265/api/cluster_status`

### Server Params（`actions/params.md`）

所有 server kill/restart + benchmark 命令都使用 Ray Dashboard REST：

```bash
POST /api/jobs entrypoint: "
# Kill server (safe pattern — do NOT pkill -f sglang)
ps aux | grep 'python3 -m sglang' | grep -v grep | grep -v bash | awk '{print \$2}' | xargs -r kill -9 2>/dev/null
sleep $SERVER_KILL_WAIT_S

# Restart with new param
export MODEL='$MODEL' TP=$TP CONC=$CONC FRAMEWORK=$FRAMEWORK
export SGLANG_EXTRA_ARGS='$BASELINE_ARGS $NEW_PARAM'
export RESULT_DIR='$RESULT_DIR/param_test_$PARAM_NAME'
export INFERENCEX_PATH='$INFERENCEX_PATH'
bash $SCRIPTS_DIR/run_baseline.sh
"
```

kill 后通过 Ray Dashboard REST 提交 `curl -s http://localhost:8265/api/cluster_status` 验证 Ray cluster。

### Kernel Optimization（`actions/kernel-opt.md`）

内核源码位于 RayJob 上。所有 find/cat 命令都使用 Ray Dashboard REST：

```bash
POST /api/jobs entrypoint: "find /tmp/torchinductor_root -maxdepth 4 -name '*.py'"
POST /api/jobs entrypoint: "python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('/path/to/standalone_kernel.py').read_text())\nPY"
```

镜像选择：使用 `KERNEL_OPT_IMAGE`（由 CI 或用户提供）。

只通过 OOB 提交内核候选：

```bash
# OOB Codex / Claude
$OOB_RAY_CLI run -a codex -p "$PROMPT" -f "$WORK_DIR/kernel.py" \
  -o "$WORK_DIR/oob_codex_${KERNEL_NAME}" --max-turns 20 --timeout 1200 \
  --no-live --json
```

### Integrate（`actions/integrate.md`）

所有 patch 命令都通过 Ray Dashboard REST：

```bash
POST /api/jobs entrypoint: "python3 $SCRIPTS_DIR/patch_inductor.py patch \
    --kernel-name $KERNEL_NAME \
    --optimized-file $OOB_OUTPUT_PATH \
    --target-file $STANDALONE_FILE_PATH \
    --best-config '{\"XBLOCK\": 4, \"R0_BLOCK\": 2048, \"num_warps\": 4}'"
```

OOB 输出位于共享 NFS 或 CLI 报告的 workspace 路径。使用 `oob_ray_submit.py run --json` 的 JSON `.workspace` 字段。

**Re-Baseline：**

```bash
POST /api/jobs entrypoint: "
export HEALTH_TIMEOUT=1800
export MODEL='$MODEL' TP=$TP CONC=$CONC FRAMEWORK=$FRAMEWORK
export SGLANG_EXTRA_ARGS='$TUNED_SERVER_ARGS'
export RESULT_DIR='$RESULT_DIR/optimized_$KERNEL_NAME'
bash $SCRIPTS_DIR/run_baseline.sh
"
```

**在 Ray cluster 上回滚：**

```bash
POST /api/jobs entrypoint: "
# Strategy A (Inductor cache): restore .bak files
find /tmp/torchinductor_root -name '*.bak' -exec sh -c 'cp \"\$1\" \"\${1%.bak}\"' _ {} \\;

# Strategy B (framework source): restore backup
cp /path/to/kernel.py.bak /path/to/kernel.py
find /sgl-workspace/aiter -name '__pycache__' -exec rm -rf {} + 2>/dev/null
"
```

#### 多节点内核 patch

多节点 RayJob 中，kernel patch 需要在所有节点上执行：
- **Inductor cache**（Strategy A）：cache 是节点本地的。必须 patch 所有节点，或在 NFS 上使用共享 Inductor cache 目录（`TORCHINDUCTOR_CACHE_DIR=/wekafs/inductor_cache`）。
- **Framework source**（Strategy B）：如果 framework 安装在共享 NFS 上，只 patch head 通常足够。

多节点 patch 时，通过 Ray 将 patch 命令提交到每个节点：

```python
import ray

@ray.remote
def patch_on_node(patch_script):
    import subprocess
    return subprocess.run(patch_script, shell=True, capture_output=True, text=True)

nodes = ray.nodes()
futures = [patch_on_node.options(
    resources={f"node:{node['NodeManagerAddress'].split(':')[0]}": 0.001}
).remote(patch_script) for node in nodes if node['Alive']]
results = ray.get(futures)
```

多节点 RayJob 中，回滚也使用相同的 `patch_on_node` 模式在所有节点执行。

### Sweep（`actions/sweep.md`）

**Option A：通过 Ray Dashboard REST 在 RayJob 上串行 sweep：**

```bash
POST /api/jobs entrypoint: "export MODEL='$MODEL' TP=$TP INFERENCEX_PATH='$INFERENCEX_PATH' \
  CONC_VALUES='4 16 64' \
  ISL_OSL_CONFIGS='1024:1024 8192:1024 1024:8192' \
  RESULT_DIR='$RESULT_DIR/sweep' \
  SGLANG_EXTRA_ARGS='$TUNED_SERVER_ARGS' && \
  bash $SCRIPTS_DIR/run_sweep.sh"
```

**~~Option B：SaFE MCP 并行 sweep~~ — 已废弃（违反 IR-10）**

> 根据 IR-10，skill 不能创建主 RayJob 以外的 SaFE workload。
> 使用 Option A（串行）或 Option B（在现有 RayJob 内部运行 Ray tasks）。

**Option B：通过 RayJob 内部 Ray tasks 并行 sweep：**

通过 Ray Dashboard REST 提交一个 Python driver。该 driver 在 RayJob
镜像内部执行，可以使用集群内 `ray.init()`（不带 `address=`）在现有集群上调度任务：

```python
import ray

ray.init()

@ray.remote(num_gpus=8)
def run_sweep_config(model, tp, conc, isl, osl, result_dir, extra_args):
    import subprocess, os
    env = {
        "MODEL": model, "TP": str(tp), "CONC": str(conc),
        "ISL": str(isl), "OSL": str(osl),
        "RESULT_DIR": result_dir,
        "SGLANG_EXTRA_ARGS": extra_args,
    }
    cmd = f"bash {SCRIPTS_DIR}/run_baseline.sh"
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, **env})

configs = [(64, 1024, 1024), (16, 1024, 1024), (4, 1024, 1024),
           (64, 8192, 1024), (16, 8192, 1024)]
futures = [run_sweep_config.remote(MODEL, TP, c, i, o,
    f"/wekafs/inference-optimization/results/sweep_{TIMESTAMP}/c{c}_i{i}_o{o}",
    TUNED_SERVER_ARGS) for c, i, o in configs]
results = ray.get(futures)
```

**注意：** 需要 Ray 集群有足够 GPU 才能并行运行这些配置。单个 RayJob（8 GPU）下，配置会顺序运行。

### Report（`actions/report.md`）

报告生成后，停止 RayJob（见上方 “Cleanup”）。

---

## 多节点 TP

SGLang 和 vLLM 都支持多节点 tensor parallelism：

```bash
# SGLang multi-node (Ray-based)
python3 -m sglang.launch_server --model $MODEL --tp $TP --host 0.0.0.0 --port 8888

# vLLM multi-node
vllm serve $MODEL --tensor-parallel-size $TP --distributed-executor-backend ray --host 0.0.0.0 --port 8000
```

TP 值应等于所有节点的 GPU 总数（例如 2 节点 × 8 GPU = TP=16）。

## NFS 共享约束

OOB CLI 输出和 RayJob patching 必须共享同一个 NFS：
- OOB CLI 将优化后的内核写入 `<output-dir>/tasks/<user>/<task_id>/workspace/`
- RayJob 的 patch 步骤从同一个 NFS 路径读取
- 确保 RayJob 的 volume mount 包含 OOB 输出路径

## 安全进程管理

**绝不要使用 `pkill -f sglang`**，它会杀掉 Ray worker。只能使用：

```bash
# SGLang:
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
ps aux | grep 'sglang.launch_server' | grep -v grep | grep -v 'ray::' | awk '{print $2}' | xargs -r kill -9 2>/dev/null

# vLLM:
kill $(pgrep -f 'python.*-m vllm.entrypoints') 2>/dev/null
```

kill 后始终验证 Ray cluster 仍存活：
`curl -s http://<HEAD_IP>:8265/api/cluster_status`
