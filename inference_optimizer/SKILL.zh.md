---
name: inference_optimizer
description: |
  启动并监控 Hyperloom 多智能体推理优化器,用于 AMD GPU 上的 LLM 服务推理优化。
  当用户要求优化推理模型、运行 Magpie 基准测试/性能分析、恢复 inference_optimizer
  会话、调优 SGLang/vLLM 服务参数、运行 TraceLens/kernel-agent,或在新推理环境中
  验证端到端吞吐提升时使用本技能。
globs:
  - "**/inference*optim*"
  - "**/inference_optimizer*"
---

# 推理优化器技能(Inference Optimizer Skill)

你是启动器与监控者。优化器本体是本仓库下的 Python `inference_optimizer` 运行时。
除非用于调试,否则不要在对话中手工进行优化;应当启动 CLI、轮询持久化状态、并
客观地汇报进度。

## 本技能运行的内容

CLI 会启动一个 Python Coordinator(协调器),它统筹协调以下角色:

- Orchestration(编排):决定下一步动作(`baseline`、`profile`、`backends`、
  `params`、`sweep`、Kernel 请求、`report`)。
- Kernel(核):响应 `select_kernels`、`run_optimization`、`integrate` 的路径。
- Critic(评审):提案审查(默认:`--critic-agent` —— 驱动 `critic-agent/` 技能
  运行时,带 KB 先验/会话记忆/由 `review_constraints` 控制的判定)。`--critic-mock`
  用于离线/冒烟测试;`--critic-codex-bare` 用于在没有运行时层的情况下调试 LLM 层。
- Robustness(健壮性):本分支中为 mock 健壮性监控器。

状态保存在**一个固定的会话目录**中 —— 默认是 `/workspace/hyperloom`。v0.6.1
将之前的 `<root>/<session_id>/` 布局压扁为一个平铺目录,因为每个沙箱都是一次性使用
的,路径中不再包含 session_id。仅测试场景下可通过
`$INFERENCE_OPTIMIZER_SESSION_DIR` 覆盖。

```text
/workspace/hyperloom/                     # session_dir (fixed)
├── manifest.json                         # Python-written session resume tag
├── state.json                            # SharedState (Coordinator-owned)
├── storage/coordinator.db                # SQLite WAL
├── agents/{orchestration,kernel,critic,robustness}/
│   ├── inbox.jsonl  outbox.jsonl
│   ├── persona.md
│   └── system_prompt.snapshot.md
├── personas/  checkpoints/  findings/  kb/
├── runs/                                 # data-plane (executor outputs)
│   ├── baseline/<task_id>/
│   ├── profile/<task_id>/
│   ├── backends/<task_id>/{variant_NN_*/, result.json}
│   ├── params/<task_id>/{variant_NN_*/, combo/, result.json}
│   ├── sweep/<task_id>/
│   ├── integrate/<task_id>/
│   └── kernel_opt/<kernel_id>/<task_id>/
├── kernel-agent-workspace/<kernel_id>/   # cross-task GEAK/OOB artefacts
├── patches/<kernel_id>/                  # KEEP'd patches + backup
├── reports/                              # `report` action output
└── logs/                                 # cli + reactor + auth-proxy logs
```

始终优先以 `manifest.json` / `state.json` / `coordinator.db` 为准,不要从终端日志
中猜测。

会话目录解析顺序(`inference_optimizer/paths.py`):
1. `$INFERENCE_OPTIMIZER_SESSION_DIR` 环境变量 → 原样使用。
2. 默认 `/workspace/hyperloom`。

## 环境准备(Setup)

本技能**只有两条命令**。不要在对话中重复 setup 步骤 —— 两条命令均为幂等,会自动
探测,重复执行也安全。

### 凭据(env > .env,env 始终优先)

`SAFE_API_KEY` 与 `OPENAI_BASE_URL` 是本技能唯一需要的凭据。`install.sh` 与 CLI
的 `_preflight()` 使用同一套解析顺序:

1. 如果两个变量都已在 env 中 → 直接使用,不动 `.env`。
2. 否则,仅对**缺失**的键从 `$REPO_ROOT/.env` 加载;env 中已存在的键受保护,绝
   不会被 `.env` 覆盖。
3. 如果 env 与 `.env` 都没有提供 → 快速失败。

调用方唯一需要做的:要么 `export REPO_ROOT=<hyperloom_repo_root>`,要么从仓库根
目录调用(让 `$(pwd)` 作为 fallback)。**不要**在对话中手工 `source .env` ——
`install.sh` 和 CLI 都会按"env 优先"的语义自动做。

### Step 1 — 安装(每个 pod / venv 重建一次)

```bash
export REPO_ROOT="$(pwd)"   # repo root containing kernel-agent/ + inference_optimizer/ + .env
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
. "${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}"   # pod-local runtime env
```

`inference_optimizer/scripts/install.sh` 是完整 inference optimization 的唯一安装入口。
它会先安装 optimizer / Magpie / InferenceX,再链式调用
`kernel-agent/scripts/install.sh` 安装内核优化环境。`kernel-agent/scripts/install.sh`
仍可用于单独调试 kernel-agent,但不要作为完整 inference optimizer session 的主入口。

安装阶段始终一次性初始化完整 Hyperloom runtime,即使用户后续运行时指定
`--no-kernel`,也仍会安装 kernel-agent / TraceLens / GEAK / OOB / auth-proxy;`--no-kernel`
只表示本次 `optimize` 不执行 kernel optimization phase,不改变环境初始化。

`kernel-agent/scripts/install.sh` 会安装以下内容(没有需要记忆的 `--with-*` 标志):

- `ray==2.44.1` + `click<8.3.0`
- TraceLens internal(perf-report CLI)
- GEAK CLI + `/workspace/hyperloom/runtime/geak-config/local.yaml`
- OOB CLI + claude/codex npm CLIs + `~/.claude/config.json` + `~/.codex/auth.json`
- **`127.0.0.1:4002` 上的 OOB auth-proxy**(将 `x-api-key` 重写为
  `Authorization: Bearer` 以适配 AMD primus-safe 网关;没有它 Claude SDK 会
  返回 401)

**小贴士 —— 在执行 `install.sh` 之前先把安装源 rsync 到会话目录:**上面的 pip
安装会引用共享路径上的源代码树(例如 `TRACELENS_ROOT=/wekafs/hyperloom/TraceLens-internal`、
`OOB_SRC`、`GEAK_REPO`、`WORKSPACE_ROOT/Magpie`、`INFERENCEX_PATH`)。如果这些路径
在会话运行的节点之间可能发生位移、消失或不一致,请 `rsync -a` 到会话本地镜像
(例如 `$SESSION_DIR/vendor/{TraceLens-internal,OOB,GEAK,Magpie,InferenceX}/`),
并在调用 `install.sh` **之前**覆盖对应的 env 变量指向镜像。这样可以把安装绑定到
会话,避免中途出现诸如 "TraceLens root not found" / "OOB source not found" 的
失败。

对于 `$TRACELENS_ROOT` 只是位于只读 WekaFS 挂载上的常见单节点场景(生产环境的
默认情形),不需要手动 rsync:`kernel-agent/scripts/install.sh:ensure_tracelens`
会检测只读源,然后 `cp -r` 到 `${HYPERLOOM_ROOT}/TraceLens-internal`(与
`ensure_oob` 将 `${HYPERLOOM_BUNDLE}/OOB` 镜像到 `${HYPERLOOM_ROOT}/OOB/oob_cli`
的方式一致),再由 `write_env_file` 重新导出 `TRACELENS_ROOT`,使后续的
`inference_optimizer optimize` / `select_kernels` 子进程继承可写镜像。上面"rsync
到会话目录"的提示仍是多节点 / 上游快速变动场景的逃生通道。

`${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}` 由 `install.sh`
重新生成,包含经过 proxy 重写的 URL、auth 别名、GEAK config 路径与 InferenceX 路径。
请 source 它(不要手工推导)。生成态 env/config 都写在 pod 本地 runtime 目录,不会写回
共享 WekaFS 上的源码目录。

### Step 2 — 启动

```bash
inference_optimizer optimize \
  --model "$MODEL_PATH" \
  --framework vllm \           # or sglang (default)
  --gpu-type MI300X \          # or omit for rocm-smi auto-detect
  --max-hours 2
```

使用随仓库附带的安装器来安装或校验 optimizer + 下游栈。该安装器幂等,且会链式
调用 `kernel-agent/scripts/install.sh`,因此一次调用就覆盖了:inference_optimizer
+ `claude_agent_sdk` 扩展、Magpie、InferenceX 探测、Ray(并启动一个活跃的 ray
head)、TraceLens CLI、GEAK + OOB CLI、`:4002` 上的 OOB auth-proxy,以及
pod-local `kernel-agent.env.sh`。用户提出"优化某个模型"的请求即视为允许在新节点上运行该
安装,不要再额外停下来确认:

```bash
export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/kernel-agent"
export KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"
export WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
export TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"

# Prefer the launcher Python's bin dir, then standard system paths. Do NOT
# hardcode /opt/venv/bin: in bare images that path may not exist.
PYTHON_BIN_DIR="$(dirname "$PYTHON")"
export PATH="${PYTHON_BIN_DIR}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
. "${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}"
"$PYTHON" -m inference_optimizer.cli --help
```

当 `set -u` 生效时,不要把有依赖关系的 export 合并成一条命令。Bash 会在赋值左
值之前先展开所有右值,所以
`export HYPERLOOM_KERNEL_AGENT_ROOT=... KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"`
会在干净环境下报 `unbound variable`。请像上面示例那样把有依赖的变量分行赋值并
导出。

安装器会留下一个活跃运行的 Ray head;`ray status` 应该成功。`select_kernels`
和下游的 kernel agent 都依赖它 —— 它们以 `num_gpus>=1` 提交 Ray 任务。如果你要
手动重启 Ray,**不要**传 `--num-gpus=0`;那样即便 ROCm 看到 GPU 空闲,kernel 优化
也会永远 pending。

CLI 每次启动还会运行 `_preflight()` 作为上述安装步骤的安全网,它会:

1. 从 `SAFE_API_KEY` 重新导出 auth 别名(`ANTHROPIC_AUTH_TOKEN`、`OPENAI_API_KEY`、
   ...)。`OOB_BASE_URL` / `GEAK_BASE_URL` / `LLM_API_BASE` 从 `OPENAI_BASE_URL`
   上游继承(这些客户端原生使用 Bearer,不需要走 proxy)。
2. **自动安装缺失的 Python SDK** 到当前运行的解释器(`sys.executable`):
   `claude-agent-sdk>=0.1.65`、`openai>=1.50`、`httpx>=0.27`。这些都在
   `pyproject.toml` 声明,但只拉了源码树而没解析依赖的沙箱会到这里才缺,导致
   baseline 已经烧掉时钟时间后的第一个 reactor tick 才报
   `BackendError: claude-agent-sdk not installed`。
3. **从 `$OOB_SRC` 引导 `auth_proxy.py` 源文件**(或
   `/wekafs/fully-local/OOB` / `/wekafs/fully-local/inference_optimization/OOB`)
   到 `${HYPERLOOM_ROOT:-/opt/hyperloom}/OOB/oob_cli/`(如缺失)。这就是
   `ensure_auth_proxy.sh` 实际执行的文件;没有它,supervisor 会告警、返回 1,
   `:4002` 也起不来。
4. 重新运行 `ensure_auth_proxy.sh`,**改写 `~/.claude/config.json`** 使其
   `customApiUrl` 指向 `127.0.0.1:4002`,**并在当前运行进程中强制覆盖
   `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` 为 proxy URL**(任何来自 shell rc、
   `.env`、k8s secret 或容器 env 的预设值都会被替换;覆盖动作会打到 stdout)。
   如果 auth-proxy 在一次重试后仍起不来,原始 env 值会被还原,保证这两个变量
   保持一致,并打印一条 WARNING —— 此时 Claude/Codex CLI 直接打网关可能 401。
5. **ROCm 环境清洁**(仅 WARN):当 `ROCR_VISIBLE_DEVICES` 同时被设置时,清掉
   `HIP_VISIBLE_DEVICES`(两者混用会让 Magpie 子进程内 `torch.cuda.is_available()`
   返回 false);通过 `rocm-smi --showid` 校验可见 GPU 数 vs `$TP`;检查
   `/dev/shm` 空闲空间 >= 16 GiB。
6. 在 pod 重建场景下自动补装缺失的 `ray` / `Magpie` / `InferenceX`。
7. 如果未传 `--gpu-type`,则自动探测。
8. 对 `node` / `claude` / `codex` CLI 做仅 WARN 的存在性检查。
9. 输出唯一的规范化 **`Preflight diagnostics:`** 区块,内含 `asset_root`、
   `session_dir`(以及解析它所用的 env 变量)、`magpie_python`、
   `INFERENCEX_PATH`、aiter jit cache 状态(WARM/COLD + `.so` 数量 + 路径)、
   冷启/热启超时上限、当前生效的 proxy URL。启动方应当将该区块**原样**粘贴到
   状态报告里,而不是去源代码里 grep env 名。

`_preflight()` 返回之后、Coordinator 启动之**前**,CLI 还会执行:

10. **`--claude-model` 的硬性模型门禁**。该参数必须等于 `claude-opus-4-7`
    (优先)或 `claude-opus-4-6`(回退)。其他取值一律 `sys.exit(2)`,因为
    在 opus-4-5 / haiku 上的编排漂移曾静默劣化历史 run。随后会探测网关 catalog
    (`GET <OPENAI_BASE_URL>/models`,Bearer,`verify=False`,3 次指数退避重试
    1s/3s/5s);若选中的模型缺失但存在 `claude-opus-4-6`,则重写参数并打印
    WARNING。若所有重试后都不可达,或两个允许模型都不在 catalog 中,则拒绝启动。
11. **Codex 冒烟测试**(仅 WARN)。当 codex 实际会被用到时(`--critic-agent` /
    `--critic-codex-bare`,或启用 kernel 的 `--kernel-codex`),`--codex-model`
    会对同一个 catalog 做校验。
12. **Critic-agent 运行时探测**(仅当 `--critic-agent` 生效时 —— 它是默认值)。
    解析 `critic_agent_root`:env `CRITIC_AGENT_ROOT` > 同级目录
    `$REPO_ROOT/critic-agent/` > 中止。然后必须以 exit code 0 完成
    `python -m runtime.cli --help`(5s 超时,`cwd=root`);否则 optimizer 以
    rc=2 中止并给出指向 `--critic-mock` / `--critic-codex-bare` 的恢复建议。
    默认设置 `WORKSPACE_PATH=$REPO_ROOT`、
    `CRITIC_SESSION_MEMORY_DIR=$SESSION_DIR/critic-session-memory`、
    `CRITIC_KB_CLIENT_MODE=inmemory`;`live` 模式还要求导出 `KB_BASE_URL`。

基于 install.sh 的拉起是规范入口;`_preflight()` 只是在运行中途捕获漂移。
`kernel-agent/SKILL.md` 的 `Installation`、`TraceLens Requirements`、
`Backend Selection` 章节是链式安装器覆盖范围的权威信息源;调试 kernel-agent
层时去读那篇。

你**不需要**手工 `pip install claude-agent-sdk`、从 bundle 路径拷贝
`auth_proxy.py`、`export ANTHROPIC_*`、手工启动 ray、手工 pip install Magpie、
手工 `source .env`、手工编辑 `~/.claude/config.json`,或者手工
`curl /v1/models` 来挑选模型名。任何手工操作正是导致以下故障的根因:
"Claude SDK exit code 1" / HTTP 401 / "claude-sonnet-4 not in catalog" /
"customApiUrl points to a local proxy that isn't running"。

### 恢复(Recovery)

如果 CLI 以 `Claude SDK exit code 1` 或 `Primus.00009 token not present` 退出,
说明 auth-proxy 死了。重新运行 supervisor 并重试 —— 两者都是幂等的:

```bash
bash "$REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh"   # noop if healthy
inference_optimizer optimize ... # rerun
```

如果 `_preflight()` 本身失败,以 `--check-only` 模式运行 install 查看缺哪一块,
然后再跑完整 install:

```bash
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh" --check-only
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
```

在 `/workspace/hyperloom` 不可写的沙箱中,用单个 env 变量覆盖会话位置:

```bash
export INFERENCE_OPTIMIZER_SESSION_DIR="$RUN_ROOT/optimizer-session"
mkdir -p "$INFERENCE_OPTIMIZER_SESSION_DIR"
```

CLI 启动时会调用一次 `make_session_dir()`,就地创建完整的子目录骨架(幂等 ——
重复运行安全)。

## 可移植预检(Portable Preflight)

每次启动新模型 run 之前,校验模型路径、GPU 可见性以及是否有重复进程。永远不要
打印 token。

```bash
export MODEL_PATH=/path/to/model
test -d "$MODEL_PATH"

"$PYTHON" - <<'PY'
import os
try:
    import torch
    print("torch_cuda_available=", torch.cuda.is_available())
    print("torch_cuda_device_count=", torch.cuda.device_count())
except Exception as exc:
    print("torch_check_error=", type(exc).__name__, str(exc)[:300])

patterns = ("inference_optimizer.cli", "Magpie", "sglang.launch_server")
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read()
    except Exception:
        continue
    text = cmd.replace(b"\0", b" ").decode("utf-8", "ignore")
    if text and any(p in text for p in patterns):
        print(f"existing_process {pid}: {text[:300]}")
PY
```

## 基准测试配置(Benchmark Config)

默认配置文件位于:

```bash
inference_optimizer/scripts/configs/baseline_sglang.yaml
inference_optimizer/scripts/configs/baseline_vllm.yaml
inference_optimizer/scripts/configs/profile_sglang.yaml
inference_optimizer/scripts/configs/profile_vllm.yaml
```

每个 YAML 中有两个字段**仅作 fallback** —— 优化器会在运行时覆盖它们:

- `benchmark.model` <- `--model` / `$MODEL_PATH`
- `benchmark.runner_type` <- `--gpu-type` / `$GPU_TYPE` / rocm-smi 自动探测

`benchmark.benchmark_script` 在随仓库发布的 YAML 中**有意**不设置。由于
`runner_type` 在运行时注入,Magpie 会自行选取 `{framework}_{runner_type}.sh`
(例如 `sglang_mi300x.sh` / `sglang_mi355x.sh`)。每个 YAML 在 `framework:`
下方都有一段注释的 `# benchmark_script: ...` 模板,供手工调试覆盖使用。

新模型 run 前,确认以下字段与环境一致:

- `benchmark.model`:模型路径。
- `benchmark.envs.TP`:张量并行规模。
- `benchmark.envs.CONC`、`ISL`、`OSL`:工作负载。
- `benchmark.envs.ROCR_VISIBLE_DEVICES`:GPU pinning。
- `benchmark.envs.PATH`:必须以启动方 Python 的 bin 目录开头
  (`$(dirname "$PYTHON")` —— 在 hyperloom 容器中通常是 `/opt/venv/bin`,
  在裸镜像上 fallback 到 `$(dirname $(which python3))`)。

### 工作负载契约复用(baseline → params/backends/sweep)

`baseline` 执行器会用 operator 的进程 env(`CONC` / `ISL` / `OSL` / `TP` /
`MAX_MODEL_LEN` / `PRECISION` / `RUN_EVAL` / `ROCR_VISIBLE_DEVICES`,以及自适应
的 `NUM_PROMPTS` / `NUM_WARMUPS`)一次性物化它的 YAML,并写到 baseline 工作区
旁边的 `baseline_config.with_envs.yaml`。Coordinator 将该路径存到
`SharedState.baseline_config_path`,后续每个 `params`、`backends`、`sweep` 任务
都会通过 `task.params["config_path"]` 沿用此路径。

这样下游 variant 测的就是**和 baseline 同一个工作负载**。否则,grid runner
会用随仓库 YAML 的冒烟默认值(`TP=1` / `CONC=8` / `ISL=256` / `OSL=256`)渲染
variant,产生比 baseline 低约 10× 的吞吐 —— 即历史上的 "baseline 4367 tok/s
vs variants ~360 tok/s" 基准公平性 bug。

`params` / `backends` / `sweep` 也会基于它们收到的 `config_path` 重新执行
materialization,因此即便 operator 在 `baseline` 之前直接调用其中之一,该契约
仍然成立。Sweep variant 显式提供的 `CONC` / `ISL` / `OSL` 覆盖依然生效,因为
`_grid_runner._build_variant_yaml` 在最后才应用每个 variant 的 `extra_envs`。

## Critic 后端选择(Critic Backend Selection)

Critic 角色有三种后端模式,通过互斥 CLI 标志选择。默认是 `--critic-agent`
(无需显式传)。

| 标志 | 后端类 | 行为 |
|---|---|---|
| (无) / `--critic-agent` | `CriticAgentBackend` | 通过 `python -m runtime.cli prepare-review` → Codex chat completion → `python -m runtime.cli commit-review` 驱动独立的 `critic-agent/` 技能运行时。增加了 KB 先验查询(带 circuit-breaker 处理不可达服务)、按会话的记忆 + 幂等 `reviewed_msg_ids`(避免重复 verdict)、注入到 LLM 提示中的 `judge_bundle.review_constraints`,以及上下文缺失时的 `needs_review` / `critic_unavailable` 来源。 |
| `--critic-mock` | `MockCriticBackend` | 永远通过的适配器。用于 Codex 凭据不可用时的离线 / 冒烟测试。 |
| `--critic-codex-bare` | `CodexBackend` | 旧的直连 chat-completion 路径,无 KB / 会话记忆 / `review_constraints`。用于隔离地调试 LLM 层。(`--critic-real` 是隐藏的向后兼容别名。) |

默认值可按 pod 通过 `INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND` 覆盖(取
`mock` / `agent` / `codex_bare` 之一)。

### 必需 env(仅在 `--critic-agent` 生效时)

| 变量 | 用途 | 默认 |
|---|---|---|
| `CRITIC_AGENT_ROOT` | 包含 `runtime/cli.py` 的目录。 | 同级目录 `$REPO_ROOT/critic-agent/` |
| `CRITIC_KB_CLIENT_MODE` | `inmemory` 使 KB 读写不出网。`live` 需要 `KB_BASE_URL`。 | `inmemory` |
| `KB_BASE_URL` | `CRITIC_KB_CLIENT_MODE=live` 时的 KB 服务 URL。 | 未设置(若 live 模式启动时缺失则中止) |
| `KB_TIMEOUT_MS` / `KB_RETRY_MAX` / `KB_DEAD_LETTER_DIR` | 透传给运行时;详见 `critic-agent/AGENTS.md`。 | 运行时默认值 |
| `CRITIC_SESSION_MEMORY_DIR` | 运行时持久化按会话的决策 / reviewed_msg_ids 的位置。 | `$SESSION_DIR/critic-session-memory`(由优化器自动设置,与 Coordinator 会话同目录,清理时一并清掉)。 |
| `WORKSPACE_PATH` | critic-agent 运行时解析提示资产时使用的技能根。 | `$REPO_ROOT`(自动设置)。 |

`_preflight()` 校验 `CRITIC_AGENT_ROOT` 解析到一个含 `runtime/cli.py` 的真实
目录,然后在 Coordinator 启动前执行 `python -m runtime.cli --help`(5s 超时)。
若运行时缺失或损坏,run 会以明确的错误信息中止,并提示使用 `--critic-mock` /
`--critic-codex-bare` 旁路。

### 每轮产物(审计轨迹)

每次 Critic turn 会写入:

```text
$SESSION_DIR/critic-workdir/<turn_idx 6-digit>/
├── request.json         # raw_prompt + session_id passed to runtime.cli
├── judge_bundle.json    # output of prepare-review (proposals, KB priors,
│                          review_constraints, kb_read_skipped_reason)
├── review.json          # LLM's verdicts (extracted JSON envelope)
└── emit.json            # output of commit-review (intent_envelope +
                           kb_writes); the Coordinator consumes
                           intent_envelope verbatim.

$SESSION_DIR/critic-session-memory/<session_id>/
├── context.json          decisions.jsonl   events.jsonl
└── kb_priors_cache.json  reviewed_msg_ids.json
```

后端每次 tick 会修剪保留最新 50 个 turn workdir 之外的所有内容,避免无限增长。


## 框架选择(Framework Selection)

一个会话只能运行单一框架。通过 `--framework` 或 `$FRAMEWORK` 选择 `sglang`
(默认)或 `vllm`:

```bash
inference_optimizer optimize --framework vllm --model "$MODEL_PATH" --max-hours 2
FRAMEWORK=vllm inference_optimizer optimize --model "$MODEL_PATH" --max-hours 2
```

解析顺序:`--framework` > `$FRAMEWORK` > `sglang`(默认)。

它控制:
- 执行器默认选用的 Magpie YAML
  (`baseline_sglang.yaml` / `baseline_vllm.yaml`、
  `profile_sglang.yaml` / `profile_vllm.yaml`)
- `params` 动作运行的 params grid(`DEFAULT_VLLM_PARAMS_GRID` vs
  `DEFAULT_PARAMS_GRID`)
- `_grid_runner` 写入的 extra-args env 名(`EXTRA_VLLM_ARGS` vs
  `EXTRA_SGLANG_ARGS`)
- 编排从 Marathon KB 读取的 partition

不支持在同一会话中混用 sglang 与 vllm;CLI 会在整个 run 中锁定 `$FRAMEWORK`。
Resume 会从 shell 重新读取 `$FRAMEWORK` —— resume 一个 vLLM 会话时记得设它。

## GPU 运行类型(GPU Runner Type)

通过 `--gpu-type` 或 `$GPU_TYPE` 显式选择 GPU;两者都不指定时,优化器会通过
`rocm-smi --showproductname` 自动探测(回退到
`torch.cuda.get_device_properties(0).gcnArchName`)。

```bash
inference_optimizer optimize --gpu-type mi355x --model "$MODEL_PATH" --max-hours 2
GPU_TYPE=mi300x inference_optimizer optimize --model "$MODEL_PATH" --max-hours 2
```

接受值:`mi300x`、`mi325x`、`mi355x`。**`mi325x` 会被映射到 `mi300x`**(带
警告),因为这两块 GPU 共享同一架构,Magpie 也尚未发布
`sglang_mi325x.sh` / `vllm_mi325x.sh`。若确实需要 MI325X 专用脚本,请在对应
YAML 中取消注释 `benchmark_script:` 模板并指向你在 `InferenceX/benchmarks/...`
下的脚本。

在已知的 ROCm 栈上,除非用户明确要求,否则不要设置 `HIP_VISIBLE_DEVICES`;它会
让 `torch.cuda.is_available()` 返回 false。GPU pinning 应使用
`ROCR_VISIBLE_DEVICES`。

## SGLang 参数搜索

本项目应当先验证 SGLang 上的改进,等 SGLang 路径稳定后再加入 vLLM。`params`
通过 `EXTRA_SGLANG_ARGS` 和 `benchmark.envs` 写入候选;除非 A/B 结果在目标
工作负载上始终保持优势,否则不要把任何 flag 硬编码为默认。

默认的 SGLang 搜索已涵盖 cuda graph batch 上限、连续 decode 步数、内存占比、
调度保守度、chunked prefill 与 max prefill tokens。还应当测试来自 InferenceX
的候选项:

- Cache/scheduler:`--disable-radix-cache`、`--max-running-requests 128/256`。
- Tokenization/streaming:`--tokenizer-worker-num 8/16`、`--stream-interval 30/50`。
- ROCm/TileLang envs:`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1`、
  `SGLANG_HACK_FLASHMLA_BACKEND=tilelang`、
  `SGLANG_OPT_USE_TILELANG_INDEXER=true`。

将投机解码视为模型相关,在验证之前不要打开。对于 MTP/EAGLE,只有在模型具备
所需 draft path 或 MTP 支持时,才使用带 `SGLANG_ENABLE_SPEC_V2=1` 和相应
`--speculative-*` 标志的自定义 grid。请用聊天格式的提示进行基准测试
(DeepSeek-V4 风格的 run 使用 `--dsv4`),因为随机原始提示会让接受率结果有
误导性。

判断 SGLang 候选时,至少比较 `1k/1k` 与 `8k/1k`;若模型放得下,同时包含低并发
和高并发。只在吞吐有提升且 TTFT/E2E 或正确性未出现不可接受回归时保留参数。
Coordinator 管理的长 run 默认以 `max_candidates_per_round=5` 增量测试参数;
直接调用 runner 时可传 `0` 跑完整 grid。

### 单次 run 资产覆盖(进阶)

要在不修改随仓库 YAML 的前提下,以自定义工作负载 env 运行一个模型,物化一个
按 run 维度的资产根并通过 `--asset-root` 传入:

```bash
export ASSET_ROOT="$REPO_ROOT/optimizer_runs/assets_$(basename "$MODEL_PATH")_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ASSET_ROOT/scripts/configs"
for d in actions kernel_opt orchestrator; do
  ln -sfn "$REPO_ROOT/inference_optimizer/$d" "$ASSET_ROOT/$d"
done
ln -sfn "$REPO_ROOT/inference_optimizer/scripts/ab_torch_compile_magpie.py"  "$ASSET_ROOT/scripts/"
ln -sfn "$REPO_ROOT/inference_optimizer/scripts/ab_torch_compile_kernels.py" "$ASSET_ROOT/scripts/"
# Copy + edit baseline_*.yaml and profile_*.yaml under "$ASSET_ROOT/scripts/configs/" for
# this run's TP/CONC/ISL/OSL/MAX_MODEL_LEN/ROCR_VISIBLE_DEVICES. The
# `_workload_envs.materialize_config_with_envs` helper applies most of these
# from process env automatically; you only need a custom asset root for
# fields it does not touch (e.g. profiler.torch_profiler.enabled per yaml).
inference_optimizer optimize --asset-root "$ASSET_ROOT" --model "$MODEL_PATH" ...
```

大多数情况下,随仓库 YAML 加 `--model` / `--gpu-type` 覆盖已经够用;只有在
默认值不适配工作负载时才使用 `--asset-root`。

## 启动一次新优化(Launch a New Optimization)

单条命令 —— 前提是本 pod 已经执行过 Step 1(安装)。没有 `--session-name`;
会话固定位于 `/workspace/hyperloom`(可通过 `$INFERENCE_OPTIMIZER_SESSION_DIR`
覆盖):

```bash
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
. "${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}"
export PATH="$(dirname "$PYTHON"):/usr/local/bin:$PATH"
export RUN_TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"
export RUN_LOG="$REPO_ROOT/optimizer_runs/run_${RUN_TAG}.log"
export PID_FILE="$REPO_ROOT/optimizer_runs/run_${RUN_TAG}.pid"
mkdir -p "$REPO_ROOT/optimizer_runs"

setsid nohup inference_optimizer --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "${FRAMEWORK:-sglang}" \
  --target-gain "${TARGET_GAIN:-10}" \
  --max-hours "${MAX_HOURS:-5}" \
  --tick-interval-sec 30 \
  --kernel-claude \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

任何超过 5 分钟的 run 都必须用 `setsid nohup ... &`。仅靠 Cursor 的后台 shell
不够用 —— SSH 断开可能让进程死掉。

Critic 现在默认是 `--critic-agent`(真正的 critic-agent 运行时 —— KB 先验 /
会话记忆 / 由 `review_constraints` 控制的判定)。离线 / 冒烟 run 可传
`--critic-mock` 回退到永远通过的适配器;调试 LLM 时可传 `--critic-codex-bare`
跑无运行时层的旧版 Codex 直连路径。详见 [Critic 后端选择](#critic-后端选择critic-backend-selection)。

启动后做一次简短健康检查:

```bash
sleep 30
pid="$(cat "$PID_FILE")"
test -d "/proc/$pid" && echo "optimizer_alive=true pid=$pid"
session_dir="${INFERENCE_OPTIMIZER_SESSION_DIR:-/workspace/hyperloom}"
test -f "$session_dir/manifest.json" && echo "manifest_present=true"
test -f "$session_dir/state.json" && echo "state_exists=true" \
  && python3 -c "import json; print(json.load(open('$session_dir/state.json')).get('stop_reason'))"
```

健康 = 优化器进程存活 + `manifest.json` 存在 + `state.json` 存在 + 无过早的
`stop_reason`。

## 恢复已有会话(Resume)

`--resume` 是一个不带参数的标志;它会接管当前规范化 session_dir 中的内容。
如果 `manifest.json` 或 `state.json` 缺失,CLI 会拒绝启动。

```bash
export RUN_TAG="resume-$(date +%Y%m%d_%H%M%S)"
export RUN_LOG="$REPO_ROOT/optimizer_runs/resume_${RUN_TAG}.log"
export PID_FILE="$REPO_ROOT/optimizer_runs/run_${RUN_TAG}.pid"

setsid nohup inference_optimizer --verbose optimize \
  --resume \
  --target-gain "${TARGET_GAIN:-10}" \
  --max-hours "${MAX_HOURS:-5}" \
  --tick-interval-sec 30 \
  --kernel-claude \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

Resume 会保留 baseline、当前最优、params 搜索状态、事件历史、以及 kernel-agent
产物。CLI 会在重试前清掉过期的 `stop_reason` 与 `crash_count`。

## 长 run 的健壮性监控(Robustness Monitor)

任何超过 5 分钟的 run,都要在独立的 `setsid nohup` 进程中启动健壮性监控。
轮询间隔不得短于每 5 分钟一次;会话进入终止 `stop_reason` 时停止;优化器异常
退出时自动 resume 会话。

```bash
export ROBUSTNESS_MONITOR_SCRIPT="$REPO_ROOT/optimizer_runs/robustness_monitor.sh"
export ROBUSTNESS_MONITOR_LOG="$REPO_ROOT/optimizer_runs/robustness_monitor_$(date +%Y%m%d_%H%M%S).log"
export ROBUSTNESS_MONITOR_PID_FILE="$REPO_ROOT/optimizer_runs/robustness_monitor.pid"

cat > "$ROBUSTNESS_MONITOR_SCRIPT" <<'SH'
#!/usr/bin/env bash
set -u
session_dir="${INFERENCE_OPTIMIZER_SESSION_DIR:-/workspace/hyperloom}"
deadline=$(( $(date +%s) + (${MAX_HOURS:-5} + 1) * 3600 ))
read_stop_reason() {
  python3 -c "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); print((json.loads(p.read_text()).get('stop_reason') or '').strip() if p.exists() else '')" "$session_dir/state.json"
}
while [ "$(date +%s)" -lt "$deadline" ]; do
  pid=""
  [ -f "$PID_FILE" ] && read -r pid < "$PID_FILE" || true
  stop_reason="$(read_stop_reason)"
  case "$stop_reason" in
    target_reached|no_more_leverage|time_exhausted|max_ticks)
      echo "[robustness] terminal stop_reason=$stop_reason $(date -Is)"
      exit 0 ;;
  esac
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    echo "[robustness] alive pid=$pid stop_reason=${stop_reason:-none} $(date -Is)"
    sleep 300; continue
  fi
  echo "[robustness] optimizer stopped; resuming $(date -Is)"
  resume_log="$REPO_ROOT/optimizer_runs/resume_$(date +%Y%m%d_%H%M%S).log"
  setsid nohup inference_optimizer --verbose optimize \
    --resume \
    --target-gain "${TARGET_GAIN:-10}" --max-hours "${MAX_HOURS:-5}" \
    --tick-interval-sec 30 --kernel-claude \
    > "$resume_log" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  sleep 300
done
echo "[robustness] deadline reached $(date -Is)"
SH

chmod +x "$ROBUSTNESS_MONITOR_SCRIPT"
setsid nohup bash "$ROBUSTNESS_MONITOR_SCRIPT" > "$ROBUSTNESS_MONITOR_LOG" 2>&1 < /dev/null &
echo $! > "$ROBUSTNESS_MONITOR_PID_FILE"
```

## 监控(Monitoring)

除非在调试启动失败,否则轮询间隔不要短于 5 分钟。

```bash
export SESSION="${INFERENCE_OPTIMIZER_SESSION_DIR:-/workspace/hyperloom}"
python3 - <<'PY'
import json, os, pathlib
s = json.loads((pathlib.Path(os.environ["SESSION"]) / "state.json").read_text())
for k in ("stop_reason", "baseline_tput", "cumulative_gain", "current_best",
          "last_kernel_opt", "last_select_kernels", "last_sweep"):
    print(f"{k}: {s.get(k)}")
print("params_search_last_round:", s.get("params_search", {}).get("last_round"))
PY
```

从 SQLite 读取最近的 action 统计:

```bash
python3 - <<'PY'
import json, os, pathlib, sqlite3
from collections import Counter
db = pathlib.Path(os.environ["SESSION"]) / "storage" / "coordinator.db"
con = sqlite3.connect(db)
c = Counter()
for fa, ta, topic, payload in con.execute(
    "select from_agent,to_agent,topic,payload from events order by seq desc limit 500"
):
    try:
        p = json.loads(payload)
    except Exception:
        continue
    if topic == "proposal":
        c["proposal:" + str(p.get("action_name"))] += 1
    if topic == "delegated_result":
        c["delegated:" + str(p.get("kind")) + ":" + str(p.get("state"))] += 1
    if topic == "request" and ta == "kernel":
        c["kernel_request:" + str(p.get("kind"))] += 1
    if topic == "response" and fa == "kernel":
        c["kernel_response:" + str(p.get("kind")) + ":" + str(p.get("status"))] += 1
print(dict(c))
PY
```

## 预期流程(Expected Flow)

优化器应当:

1. 建立或复用 `baseline_tput`。
2. 仅在当前 server args 与 `last_profile_args` 不同时运行 `profile`;否则
   复用 `last_profile_trace`。
3. 每个 trace/config 只跑一次 `select_kernels`,并缓存到 `last_select_kernels`。
4. 只为 `run_optimization` 选取 `reusable_native_kernel_ids`。
5. KEEP 之前必须有编译 + 正确性 + 微基准/E2E 证据。
6. 使用 `params_search` 增量测试参数,并在 resume 之间记住被拒绝的候选项。
7. 使用 `optimization_stack`,确保 backend + params + kernel 改动互不覆盖。
8. 使用 `sweep` 了解超出冒烟工作负载的、与具体工作负载相关的结果。

## 缓存拓扑(Cache Topology)

为什么这事重要:ROCm 上的 SGLang/vLLM 会通过 `aiter` 路由热点融合 kernel
(RMSNorm、attention、fused MoE、a8w8 blockscale GEMM、RoPE 等)。`aiter`
在第一次见到某 shape 时会 JIT 编译该 shape 的变体,并把产物 `.so` 落盘缓存。
对于 671B FP8 MoE 量级的工作负载(例如 DeepSeek-R1-0528),针对一个新组合
(model、dtype、TP、`max_model_len`、`max_num_seqs`、`gpu_memory_utilization`)
的首次 `vllm serve` / `sglang launch_server` 可能在 `hipcc` 里花掉 30+ 分钟。
后续启动则秒级复用 `.so` 缓存。优化器需要知道这些缓存位于何处,才能(a)正确
解读漫长的冷启动而不是误判为 hang;(b)在缓存为空时自动延长 baseline 超时。

### aiter —— JIT 缓存(主要冷启动成本)

```text
Source:     /sgl-workspace/aiter/aiter/
Git repo:   /sgl-workspace/aiter/

JIT cache:  /sgl-workspace/aiter/aiter/jit/build/
            (also: /usr/local/lib/python3.{10,12}/site-packages/aiter/jit/build/
                   /opt/venv/lib/python3.{10,12}/site-packages/aiter/jit/build/)
            Each kernel has its own build/<kernel_name>/build/<kernel_name>.so

Tuned configs:  aiter/configs/a8w8_blockscale_tuned_gemm.csv
                aiter/configs/tuned_fmoe.csv
RoPE source:    aiter/rotary_embedding.py
GEMM dispatch:  aiter/ops/gemm.py
MoE dispatch:   aiter/fused_moe.py

Clear (specific kernel): rm -rf /sgl-workspace/aiter/aiter/jit/build/<kernel>/
Clear (all):             rm -rf /sgl-workspace/aiter/aiter/jit/build/
```

### Triton 缓存

```text
Path:   ~/.triton/cache/    (resolves via $HOME, NOT $TRITON_CACHE_DIR
                             unless explicitly exported)
Clear: rm -rf ~/.triton/cache
```

### torch.compile / Inductor 缓存

```text
Path:   /tmp/torchinductor_<user>/    (default; override via
                                        $TORCHINDUCTOR_CACHE_DIR)
Clear: rm -rf /tmp/torchinductor_root
```

### sgl_kernel 预编译 `.so`(非冷启动,仅 build 期)

```text
Location: /opt/venv/lib/python3.{10,12}/site-packages/sgl_kernel/
Compiled: common_ops.cpython-3{10,12}-*-linux-gnu.so  (built with image)
Source:   /sgl-workspace/sglang/sgl-kernel/
Build:    cd /sgl-workspace/sglang/sgl-kernel && python setup_rocm.py install
```

这一项仅作信息提示。优化器在 baseline / params / sweep 期间从不重建
`sgl_kernel`;只有 `kernel_opt` / `integrate` 通过 kernel-agent 路径才会动它。

## 冷启动纪律(Cold-start Discipline)

冷启动触发条件(任一条件命中,下一次 baseline 即为冷启动):

- 本 pod 上首次启动某个 (model, dtype, TP) 组合。
- 相对上次活跃的 server,任何 `--max-model-len`、`--max-num-seqs`、
  `--gpu-memory-utilization`、`--cuda-graph-max-bs` 或 `--quantization`
  的变更(改变 aiter 哈希的 shape 签名)。
- `--enable-torch-compile` 开关切换。
- 容器 / pod 重建,清掉了 `aiter/jit/build/`。
- 手工 `rm -rf` 上述任一缓存树。
- aiter 源码级补丁(kernel_opt / integrate 刚落在 `/sgl-workspace/aiter/aiter/`
  下某个 kernel 上)。

`BaselineExecutor` 通过统计 `aiter/jit/build/` 下的 `.so` 数量自动检测冷启动。
阈值:**`< 20` 个文件 = COLD**,否则 WARM。探测路径列表(见
`baseline.py:AITER_JIT_PROBE_PATHS`)中**首个存在的路径**胜出。要覆盖路径,
导出 `INFERENCE_OPTIMIZER_AITER_JIT_DIR=/abs/path/to/jit/build`。

同一探测也会在启动时由 `_emit_preflight_diagnostics()` 跑一次,使解析出的
缓存状态出现在规范化的 preflight 区块中:

```
Preflight diagnostics:
  ...
  aiter jit cache     = 98 .so / 887 MB (WARM) at /sgl-workspace/aiter/aiter/jit/build
  cold_start_timeout  = 3600s
  warm_timeout        = 1500s
  proxy URLs          = http://127.0.0.1:4002/api/v1/llm-proxy (auth-proxy alive)
```

启动方应当读取该区块,而不是在 `cli.py` / `baseline.py` 里 grep env 变量名。
`cold_start_timeout` 行会反映 `INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC` 的
任何活跃覆盖。

超时选择(每次 baseline `__call__` 一次性决定):

| 条件 | 最终 `subprocess.run(timeout=...)` |
| --- | --- |
| `task.params['timeout_sec']` 已设置 | 任务给定值(永远优先) |
| 缓存探测 `found` 且 `kernel_count < 20` | `BASELINE_COLD_START_TIMEOUT_SEC`(默认 3600s;通过 `INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC=N` 覆盖) |
| 缓存探测 `found` 且 `kernel_count >= 20` | `BASELINE_DEFAULT_TIMEOUT_SEC`(1500s) |
| 缓存探测 `not_found` / `error` | `BASELINE_DEFAULT_TIMEOUT_SEC`(1500s)+ WARN 日志 |

每次 baseline 启动都会打印这些标记之一 —— grep `optimizer_runs/run_*.log`
就能验证走的哪条路径:

- `baseline_executor: COLD_START detected — aiter jit/build/ at <path> has N .so (< 20 threshold), M MB. Bumping timeout 1500s -> 3600s. ...`
- `baseline_executor: WARM start — aiter jit/build/ at <path> has N .so, M MB. Using default timeout=1500s.`
- `baseline_executor: timeout=Ns (explicit task param)`
- `baseline_executor: aiter jit cache not located (probe_status=...). Using default timeout=1500s. Cold-start auto-bump disabled for this run.`

若在多次 baseline 重试中反复看到 COLD_START 标记,说明 JIT 大概率在前一次超时
中被 `hipcc` 中途 kill(留下半写的 `.so`)—— 此时应通过
`INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC=5400` 进一步延长冷启动上限,而不是
重新启动。`ProfileExecutor` 继承同样的逻辑(它继承自 `BaselineExecutor`),
所以 `profile` 动作也享受同样的自动延长。

## Kernel 应用安全(Kernel Apply Safety)

Kernel 优化可能改动 `/sgl-workspace/aiter`、`/sgl-workspace/sglang` 或编译产物。
应用补丁之前:

- 备份源文件。
- 在可获取时备份编译后的 `.so` / `.co` 产物。
- REVERT 时先恢复编译产物,再恢复源文件,然后重启 server。在原始编译产物已备份
  的情况下,REVERT 期间避免重建。
- 仅在正确性与 E2E 都可接受时 KEEP。

如果用户没有显式批准环境变更,在真正 apply/rebuild 之前停下来询问。Dry-run
与分析操作是安全的。

## Kernel E2E 重试纪律(Kernel E2E Retry Discipline)

微基准加速还不够。`run_optimization` 返回一个候选 kernel 补丁后,`integrate`
必须用 E2E Magpie 吞吐验证补丁,并把每次尝试记录到 `state.json`。

对于同一个 `kernel_id + patch_path + EXTRA_SGLANG_ARGS`:

- `KEEP`:仅在 E2E 增益越过配置阈值时接受。
- `REVERT`:立即拒绝该补丁,不要再跑。
- `NEEDS_REVIEW`:最多允许 3 次 E2E 尝试。如果没有一次越过 KEEP 阈值,拒绝该
  补丁,把剩余预算用于 params 搜索或换一个 reusable native kernel。

不要因为微基准很强就反复 integrate 同一个补丁。如果 E2E 结果在零增益附近抖动,
正确的做法是把该补丁标记为已拒绝、保留产物供人工审查,把剩余预算花到未测过的
params/backend 候选项或下一个 kernel 上。

## 故障处理(Failure Handling)

- `ERROR: --claude-model=... is not allowed`:静态门禁拒绝了所选模型。编排必须
  使用 `claude-opus-4-7`(优先)或 `claude-opus-4-6`(回退)。删除或修改
  `--claude-model` / `$CLAUDE_MODEL` 后重跑。这是**故意为之** —— opus-4-5 /
  haiku 曾静默劣化历史 run,operator 锁死了允许列表。
- `ERROR: gateway catalog unreachable after retries`:`GET <base_url>/models`
  探测在所有 4 次尝试(初始 + 3 次 1s/3s/5s 指数退避)都失败。手工复现(命令
  在 `terminals/6.txt` 中):
  ```bash
  curl -k -H "Authorization: Bearer $SAFE_API_KEY" \
       "$OPENAI_BASE_URL/models" | jq '.data[].id' | sort
  ```
  若网关有响应,则 proxy / SSL 路径有问题;若无响应,则网关本身宕机。我们故意
  在此 fail-fast,而不是在 baseline 跑 5 分钟后才 401。
- `ERROR: neither claude-opus-4-7 nor claude-opus-4-6 present in gateway catalog`:
  catalog 可达但两个允许模型都不在列表中。要么是网关下线了它们(升级到 operator
  处理),要么这是错误的端点。**不要绕过**门禁;请改 catalog,或在有新模型被
  批准时更新 `cli.py` 中的允许列表常量 `_CLAUDE_ALLOWED_MODELS`。
- `WARNING — claude-opus-4-7 not in gateway catalog; falling back to claude-opus-4-6`:
  在 4-7 被轮换出网关时是预期行为。run 会继续使用 4-6;性能特征几乎相同。
- `Claude SDK exit code 1` / `Primus.00009 token not present`:auth-proxy 死了。
  执行 `bash $REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh` 并重试 CLI。
  **不要**手工改写 `~/.claude/config.json` —— 它由 `_preflight()` 负责。如果
  supervisor 报 `auth_proxy.py` 缺失,把 `OOB_SRC` 指向一个包含它的目录(或
  落到 `/wekafs/fully-local/OOB`、`/wekafs/fully-local/inference_optimization/OOB`
  之一),这样下次 run 时 `_ensure_oob_proxy_source()` 能引导它。
- `ERROR: --critic-agent selected but critic-agent runtime not found`:解析顺序
  是 `$CRITIC_AGENT_ROOT` env > 同级目录 `$REPO_ROOT/critic-agent/`。修复方案
  二选一:
  ```bash
  export CRITIC_AGENT_ROOT=/path/to/critic-agent
  # or:
  test -f "$REPO_ROOT/critic-agent/runtime/cli.py" || \
    git -C "$REPO_ROOT" submodule update --init critic-agent
  ```
  如果一时修不了,用 `--critic-mock`(离线 / 冒烟)或 `--critic-codex-bare`
  (旧版 Codex 直连路径)绕过。
- 每个 critic verdict 都返回 `('needs_review', 'critic_unavailable')`,且
  `kb_skipped=missing_critical_context`、`required_context=['model', 'framework', ...]`
  (见 `critic_agent_backend turn=...` 日志行与
  `BackendTurnResult.metadata['required_context']`):在旧版本里这是一个真实
  bug —— `CriticAgentBackend` 无条件发送 `request.context={}`,导致运行时的
  `CRITICAL_CONTEXT_KEYS=("model","framework")` 门禁对每个提案都触发,
  orchestrator 永远拿不到 `approve`。修复后,后端在 `__post_init__` 中读一次
  `manifest.json`(`_load_static_context_from_manifest()`),把
  model / framework / gpu_type / model_path / tp / workload / precision 注入到
  每次 `prepare-review` 请求。如果再次见到该症状,检查 `manifest.json` 的
  `model_name` 与 `framework` 是否非空(`build_manifest()` 是写入方),并 grep
  `logs/cli.log` 中的 `critic_agent_backend static_context source=... keys=[...]`
  —— keys 列表反映实际加载了什么。调试期间可用 `--critic-mock` 旁路(永远通过,
  无审查安全网),或者在以编程方式调用后端时显式传 `static_context=`。
- `BackendError: critic-agent runtime.cli prepare-review/commit-review exited rc=2`:
  critic-agent 运行时因适配器 bug 中止 —— 按 `critic-agent/AGENTS.md` §Exit codes,
  rc=2 表示运行时内部的 schema 或 validation 失败。检查
  `$SESSION_DIR/critic-workdir/<latest>/{request,judge_bundle,review,emit}.json`
  中的问题载荷,然后要么修上游问题,要么用 `--critic-mock` 重试以让 run 继续
  推进,同时让该运行时 bug 进入调试流程。
- `BackendError: critic-agent runtime.cli ... timed out after 30s`:
  prepare-review / commit-review 通常 <1s 返回。超时意味着 KB 调用卡住或 KB
  写入扇出过重。若 `CRITIC_KB_CLIENT_MODE=live`,本次 run 剩余部分改用
  `inmemory`(不需要 kill switch;下一个进程会继承较低模式)。如果在
  `inmemory` 模式下仍能复现超时,采集运行时日志并提交 bug —— 该路径不应在
  I/O 上阻塞。
- `BackendError: claude-agent-sdk not installed`:`_ensure_python_sdks()` 落地
  之后理论上不会出现。如果还出现(pip 冻结、无网络),手工安装:
  `python -m pip install claude-agent-sdk>=0.1.65 openai>=1.50 httpx>=0.27`。
- `ANTHROPIC_AUTH_TOKEN not set`:重新 source `${KERNEL_AGENT_ENV:-/workspace/hyperloom/runtime/kernel-agent.env.sh}`。
- `Fatal error in message reader`:重试 / resume;瞬时的 Claude CLI 故障会被
  容忍,直到达到 Coordinator 的紧急阈值。
- `No accelerator`:确保 Magpie 子进程 `PATH` 以启动方 Python 的 bin 目录开头
  (`$(dirname "$PYTHON")`,或把 `MAGPIE_PYTHON` 设为正确的解释器),并使用
  `ROCR_VISIBLE_DEVICES` 而不是 `HIP_VISIBLE_DEVICES`。
- 重复出现 `select_kernels`:检查 `last_select_kernels`;如果 trace/config 没有
  变化,这就是 bug。复用缓存候选并运行 optimization。
- `correctness_passed=false`:不要 integrate。审阅 kernel-agent 报告;报告必须
  包含显式的正确性证据。
- `no_more_leverage`:停止 run 并汇报结果;除非用户更改工作负载、搜索空间、
  模型或策略,否则不要 resume 同一个会话。
- `time_exhausted`:resume 同一个 session id;不要从头开始。

## 向用户汇报(Report Back To User)

简洁汇报状态:

- session id(来自 `manifest.json`)和日志路径
- `cumulative_gain` 与 `current_best`
- params 接受 / 拒绝的概览
- 最近优化的 kernel、正确性、微加速、E2E 增益、决策
- 进程是否仍在运行;若已停止,原因是什么

## 会话布局速查(Session Layout cheat sheet)

CLI 将所有内容平铺到一个固定目录 `/workspace/hyperloom`(覆盖:
`$INFERENCE_OPTIMIZER_SESSION_DIR`)。每个 mkdir 由 Python 负责;agents 通过
注入的 `SESSION_DIR` token 引用路径。PolicyGate 拒绝那些逃出 session_dir 的
路径字段(对 `source_file` 而言,则限定在框架允许列表
`/sgl-workspace/{aiter,sglang,vllm}/` 内)。

```text
$SESSION_DIR/                            # /workspace/hyperloom by default
├── manifest.json                        # written first; v1 schema; resume tag
├── state.json                           # SharedState — Coordinator-owned
├── storage/coordinator.db               # SQLite WAL (events/leases/cursors/tasks)
├── agents/<role>/                       # orchestration / kernel / critic / robustness
│   ├── inbox.jsonl  outbox.jsonl
│   ├── persona.md
│   └── system_prompt.snapshot.md        # snapshot of the prompt at boot
├── personas/  checkpoints/  findings/  kb/
├── runs/                                # data-plane (executor outputs)
│   ├── baseline/<task_id>/              # Magpie workspace + materialized YAML
│   ├── profile/<task_id>/               # baseline + torch_trace/
│   ├── backends/<task_id>/{variant_NN_*/, result.json}
│   ├── params/<task_id>/{variant_NN_*/, combo/, result.json}
│   ├── sweep/<task_id>/
│   ├── integrate/<task_id>/             # patch → re-baseline workspace
│   └── kernel_opt/<kernel_id>/<task_id>/
├── kernel-agent-workspace/<kernel_id>/  # GEAK / OOB cross-task artefacts
├── patches/<kernel_id>/                 # KEEP-promoted patches + backup/
├── reports/                             # `report` action output (final.{md,json})
└── logs/                                # cli.log / coordinator.log / <role>.log
```

路径解析助手(使用这些,不要字符串拼接):

| 助手 | 返回 |
|---|---|
| `paths.session_dir()` | `/workspace/hyperloom`(或 env 覆盖值) |
| `paths.make_session_dir()` | session 目录 + 完整骨架,幂等 |
| `paths.db_path_for(sd)` | `<sd>/storage/coordinator.db` |
| `session_paths.runs_dir(sd, kind, task_id)` | `<sd>/runs/<kind>/<task_id>/` |
| `session_paths.kernel_workspace(sd, kernel_id)` | `<sd>/kernel-agent-workspace/<kernel_id>/` |
| `session_paths.patches_dir(sd, kernel_id)` | `<sd>/patches/<kernel_id>/` |
| `session_paths.agent_log(sd, role)` | `<sd>/logs/<role>.log` |
| `session_paths.agent_prompt_snapshot(sd, role)` | `<sd>/agents/<role>/system_prompt.snapshot.md` |
| `manifest.write_manifest(sd, args)` / `load_manifest(sd)` | manifest.json 读 / 写 |
