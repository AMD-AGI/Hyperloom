# Unittest_agent × GEAK 集成端到端报告

> 验证场景：Qwen3-8B，vLLM 框架，AMD MI300X（gfx942 / CDNA3 / 304 CU / HBM3 ~5.3 TB/s），TP=1，CONC=32，ISL=1024，OSL=256。
>
> 验证运行：`/workspace/hyperloom/kernel-agent/{runs,unittests,geak}/integration_demo_2/`
>
> 跑了两个 GEAK 任务（每个 ~3 min budget）：
>
> | # | kernel_id              | source_file                                                       | unittest 状态 | GEAK 结果         | apply 决策 |
> |---|------------------------|--------------------------------------------------------------------|---------------|-------------------|------------|
> | 1 | `qwen3_rms_norm_aiter` | `/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py`   | **ok**        | 1 patch（空）     | `PARTIAL`  |
> | 2 | `qwen3_softmax_aiter`  | `/sgl-workspace/aiter/aiter/ops/triton/softmax.py`                 | **degraded**  | 4 patches（全空） | `PARTIAL`  |

---

## 0 · 流程总览

```text
inference_optimizer.cli optimize          # 由 SKILL.md 描述的入口
        │
        ▼  emit_intent kind=run_optimization
kernel_request_handlers.run_optimization_handler
        │  (per kernel: subprocess)
        ▼
kernel-agent/tools/kernel_optimization.py
        │  for backend in [geak, claude, codex, ...]
        ▼
invoke_backend("geak", ...)
        │
        ├─► (1) _maybe_generate_unittest(candidate)  ← 本次新增
        │       │
        │       ├─ build harness (config.yaml + task_runner.py)
        │       │  ├─ source/<kernel>.py → symlink /sgl-workspace/aiter/...
        │       │  ├─ source/_baseline_snapshot/<kernel>.py  ← 冻结金标准
        │       │  └─ RUNTIME_ENV ← SGLANG_*/AITER_*/TRITON_*/VLLM_*/HIP_*
        │       │
        │       └─ self_verify: subprocess(task_runner.py compile + correctness)
        │            ├─ 都过 → status=ok      → 作为 --test-command 传给 GEAK
        │            ├─ 只 compile 过 → degraded → 仅作 prompt 附录
        │            ├─ skip (HIP/.cu)       → GEAK 自带 harness
        │            └─ 失败                  → GEAK 自带 harness
        │
        ├─► (2) prompt = build_prompt(candidate, args)
        │   prompt += "## Auto-generated unittest harness ..."   ← 注入信息
        │
        └─► (3) geak.submit(prompt_file, test_command=unittest.correctness_cmd, ...)
                │
                ▼
        GEAK (mini-swe-agent, claude-opus-4-7) 在 worktree 沙盒里迭代
                │   每个 patch 写完跑 test_command
                │   round 结尾 select_agent 选最佳
                ▼
        results/round_N/parallel_M/patch_K.patch  + best_results.json
```

提交回 Coordinator 后，由 `integrate` 动作触发 `integrate_handler`：
- 把 patch apply 到 live `/sgl-workspace/aiter/...`
- 重启 vLLM 服务器
- 重跑同一份 Magpie baseline 配置
- 比较 `new_tput / base_tput`，决定 `KEEP`/`REVERT`

本次因为 GEAK patch 都是空的（见 §4），所以没有触发真正的 `integrate`。

---

## 1 · 提交给 GEAK 的 prompt

提交的 prompt 文件路径：
```
/workspace/hyperloom/kernel-agent/runs/integration_demo_2/prompts/
  geak-91864cc5.md      # rms_norm 任务（627 行）
  geak-eb080255.md      # softmax 任务（227 行）
```

### 1.1 prompt 头部（kernel 元数据） — 摘自 rms_norm 任务

```markdown
# TASK: Optimize the `_rms_norm_kernel` kernel

Optimize this GPU kernel for **AMD Instinct MI300X (gfx942, CDNA3)** inference
serving. Produce an actual edited kernel file with measurable speedup; do NOT
just analyze and submit unchanged.

kernel_name: _rms_norm_kernel
kernel_url:  /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py
kernel_type: triton
repo:        /sgl-workspace/aiter
GPU percent: 8.5%
Shapes:      []                          # 顶层 shapes 字段（legacy）

Kernel runtime metadata (structured context for GEAK):
```json
{
  "kernel_path":  "/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py",
  "kernel_name":  "_rms_norm_kernel",
  "input_shapes": [[2, 4096], [4096]],   # 来自 TraceLens 的真实 decode shape
  "input_dtypes": ["bfloat16", "bfloat16"],
  "env_vars":     {"AITER_USE_TRITON": "1"},
  "runtime_flags": {"is_multigpu": false, "num_gpus_recommended": 1},
  "kernel_params": {"BLOCK_SIZE": null, "HEAD_SIZE": null, "KV_DTYPE": null}
}
```

GEAK configuration (ignored by non-GEAK backends):
- Use homogeneous mode. Set max_rounds to 5.
```

### 1.2 硬件 + 优化优先级 block（来自 `build_prompt`）

```markdown
Hardware notes (target platform: `mi300x`):
- 304 CUs, CDNA3, ROCm arch `gfx942`
- HBM3 (~5.3 TB/s peak), 256 MB Infinity Cache
- Build flag: `--offload-arch=gfx942`

## Optimization priorities (TraceLens bound: `memory-bound`)

1. **Memory traffic reduction** (primary lever for memory-bound rows):
   improve coalescing / vectorization, fuse with neighbouring ops to
   amortize global loads ...
2. **Shape-aware tuning**: specialize block sizes and grid indexing
   for the dominant TraceLens Args.
3. **Launch amortization** for tiny high-count decode shapes ...
4. **Structural simplification** ...
5. **Compute utilization** (rarely the bottleneck here, but check) ...

Preserve function name, signature, decorators, and numerical behavior.
```

### 1.3 沙箱规则 / GOAL & EARLY-EXIT / 优化优先级 0~3（中段，已存在）

略（见 `build_prompt()` 的固定模板，约 100 行；包含 GEAK 不能改 `/sgl-workspace/aiter`、
要写在 `optimized_versions/`、用 `cpp_extension.load()` A/B 等指令）。

### 1.4 kernel 源代码块（截断到 12000 字符）

```markdown
Source content:
```
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.

import torch
import triton
from aiter.ops.triton._triton_kernels.normalization.rmsnorm import (
    _rms_norm_kernel,
    _quant_rms_norm_kernel,
    ...
)

def _rmsnorm_forward(x, weight, epsilon):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    ...
    _rms_norm_kernel[grid](x, y, weight, ...)
    return y, rsigma
...
```
```

### 1.5 ★ unittest 段（本次新增）

注入在 prompt 尾部，让 GEAK 知道走 harness 而不是自己造一个 `bench_<kernel>.py`：

```markdown
## Auto-generated unittest harness (Hyperloom unittest_agent)

- Status: **ok** (self_verify compile='ok', correctness='ok').
- Workspace dir: /workspace/hyperloom/kernel-agent/unittests/integration_demo_2/geak-91864cc5
- Source file you must overwrite (in-place):
  /workspace/hyperloom/kernel-agent/unittests/integration_demo_2/geak-91864cc5/source/rmsnorm.py
- Frozen baseline kept at:
  /workspace/hyperloom/kernel-agent/unittests/integration_demo_2/geak-91864cc5/source/_baseline_snapshot/rmsnorm.py
  (golden reference your optimized version is compared against; DO NOT touch).
- Correctness command:
  python3 .../scripts/task_runner.py correctness
  (exits 0 only when the optimized output matches the snapshot within the
   kernel's natural fp tolerance across every captured shape).
- Performance command:
  python3 .../scripts/task_runner.py performance
- Captured shapes: 2 (representative of the live vLLM/SGLang decode/prefill
  traffic this kernel handled in profile).

GEAK / OOB agent workflow:
1. Edit the file at `source/<kernel_basename>` (the symlink above).
2. Run the correctness command above; iterate until it returns 0.
3. Run the performance command above to measure speedup.
4. Report `[CORRECTNESS] PASS/FAIL` + `[MICRO_SPEEDUP] X.XXx` at the end of
   `optimization_report.md` as usual.
```

> 同时 `kernel_optimization.py::invoke_backend("geak")` 会把
> `manifest.test_command` 作为 GEAK CLI 的 `--test-command` 参数透传给
> `geak --test-command "python3 .../task_runner.py correctness"`，
> 这样 GEAK 内置的 mini-swe-agent 在每个 patch 写完后会自动跑这条命令评判
> correctness，并以 `speedup (x times) vs baseline` 作为选 patch 的指标。

---

## 2 · 生成的 unittest 长什么样

### 2.1 目录结构（rms_norm 任务）

```text
/workspace/hyperloom/kernel-agent/unittests/integration_demo_2/geak-91864cc5/
├── config.yaml                   # AgentKernelArena 标准 task config
├── scripts/
│   └── task_runner.py            # 305 行；compile / correctness / performance
├── source/
│   ├── rmsnorm.py                # symlink → /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py
│   └── _baseline_snapshot/
│       └── rmsnorm.py            # 冻结金标准（生成时 cp 一份）
├── build/                        # task_runner 写报告的位置
│   ├── compile_report.json
│   └── correctness_report.json
└── unittest_meta.json            # 生成器清单 + self_verify 结果
```

### 2.2 `config.yaml`（AgentKernelArena 兼容）

```yaml
source_file_path:
  - source/rmsnorm.py

target_kernel_functions:
  - _rms_norm_kernel
  - rms_norm

compile_command:
  - python3 scripts/task_runner.py compile

correctness_command:
  - python3 scripts/task_runner.py correctness

performance_command:
  - python3 scripts/task_runner.py performance

task_type: triton2triton

task_result_template: null

prompt:
  source_code: null
  instructions: |
    Optimize the kernel `_rms_norm_kernel` for maximum throughput while
    maintaining numerical correctness against the captured baseline.

    Profile context: this kernel accounts for ~unknown of GPU time and is
    classified as `memory-bound` bound by TraceLens roofline analysis.

    Constraints:
    - Must keep the host entry function signature stable (`rms_norm`).
    - Output must match the snapshotted baseline within `DEFAULT_ATOL` / `DEFAULT_RTOL`.
    - You may freely retune block sizes, num_warps, num_stages, and the
      `@triton.jit` body — the harness re-imports the file you edit.
  cheatsheet: null
```

### 2.3 `scripts/task_runner.py`（核心摘抄）

完整 305 行；关键片段：

```python
TASK_NAME = 'unittest_agent/rms_norm_kernel'
SOURCE_FILE       = str(TASK_DIR / "source" / 'rmsnorm.py')
BASELINE_SNAPSHOT = str(TASK_DIR / "source" / "_baseline_snapshot" / 'rmsnorm.py')
HOST_ENTRY        = 'rms_norm'         # picker 选中的真实 launcher
TARGET_KERNELS    = ['_rms_norm_kernel', 'rms_norm']

# ★ 跟 vLLM/SGLang/aiter live 环境保持一致的 env vars（17 个）
RUNTIME_ENV = {
    'AITER_USE_TRITON': '1',
    'SGLANG_OPT_USE_TILELANG_INDEXER': 'true',   # 当时进程实际有的
    'VLLM_FP8_REDUCE_CONV': '1',
    'SGLANG_MOE_PADDING': '1',
    'VLLM_FP8_ACT_PADDING': '1',
    'AITER_ROCM_ARCH': 'gfx942;gfx950',
    'ROCR_VISIBLE_DEVICES': '0',
    'SGLANG_USE_AITER': '1',
    'SGLANG_USE_ROCM700A': '1',
    'HIP_FORCE_DEV_KERNARG': '1',
    'SGLANG_ROCM_FUSED_DECODE_MLA': '1',
    # ...  KEY/TOKEN/SECRET 已经被 redact
}
for _k, _v in RUNTIME_ENV.items():
    os.environ.setdefault(str(_k), str(_v))   # 必须在 import 之前

# ★ TraceLens 抓到的真实 decode shape
TEST_SHAPES = [
    (2, 4096),    # x : (n_rows, n_cols) — decode batch 2，hidden 4096
    (4096,),      # weight : (n_cols,)
]
TEST_DTYPES = ['torch.bfloat16', 'torch.bfloat16']
WARMUP_ITERATIONS    = 5
BENCHMARK_ITERATIONS = 25

# bf16 → 1e-2 atol/rtol（fp32→1e-4, fp8→5e-2, int*→0）
DEFAULT_ATOL = 0.01
DEFAULT_RTOL = 0.01


def _materialize_args(test_idx: int):
    import torch
    device = "cuda"
    torch.manual_seed(42 + test_idx)
    arg0    = torch.randn((2, 4096), dtype=torch.bfloat16, device=device)
    arg1    = torch.randn((4096,),    dtype=torch.bfloat16, device=device)
    scalar0 = 1e-6              # ★ epsilon — host_arg_count - len(shapes) = 1
                                #   时自动填充（warn 给操作员核对）
    args = (arg0, arg1, scalar0,)
    return args


def run_correctness():
    """对比 live SOURCE_FILE 与 BASELINE_SNAPSHOT。两份用不同 module name
    分别 import，避免 Triton JIT cache 串扰。"""
    cur_mod = _load_module(SOURCE_FILE,       "candidate_kernel_current")
    ref_mod = _load_module(BASELINE_SNAPSHOT, "candidate_kernel_baseline")
    cur_fn  = _resolve_callable(cur_mod, HOST_ENTRY)
    ref_fn  = _resolve_callable(ref_mod, HOST_ENTRY)

    for test_idx in range(len(TEST_SHAPES)):
        ref_args = _materialize_args(test_idx)
        cur_args = tuple(a.clone() if hasattr(a,"clone") else a for a in ref_args)
        ref_out = ref_fn(*ref_args); torch.cuda.synchronize()
        cur_out = cur_fn(*cur_args); torch.cuda.synchronize()
        # 递归 flatten -> torch.allclose(atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
        # in-place 算子返回 None 时退回比较输入张量
        ...
```

### 2.4 `unittest_meta.json`（生成器清单）

```json
{
  "status": "ok",
  "out_dir": ".../unittests/integration_demo_2/geak-91864cc5",
  "config_yaml": ".../config.yaml",
  "task_runner": ".../scripts/task_runner.py",
  "source_file": ".../source/rmsnorm.py",
  "baseline_snapshot": ".../source/_baseline_snapshot/rmsnorm.py",
  "test_command": "python3 .../scripts/task_runner.py correctness",
  "performance_command": "python3 .../scripts/task_runner.py performance",
  "kernel_name": "_rms_norm_kernel",
  "host_entry": "rms_norm",
  "target_kernels": ["_rms_norm_kernel", "rms_norm"],
  "task_type": "triton2triton",
  "num_shapes": 2,
  "shapes": [[2, 4096], [4096]],
  "dtypes": ["torch.bfloat16", "torch.bfloat16"],
  "env_vars_count": 17,
  "self_verify": {
    "compile":     "ok",  "compile_rc":     0,  "compile_tail":     "Compilation: PASS\n",
    "correctness": "ok",  "correctness_rc": 0,  "correctness_tail": "Correctness: PASS\n"
  },
  "warnings": [
    "host entry 'rms_norm' takes 3 args; only 2 tensor shapes captured. Auto-filling 1 trailing scalar arg(s) with default values (1e-6, 1, False, None). Verify these match the kernel's non-tensor inputs (epsilon, num_heads, etc.)."
  ]
}
```

### 2.5 self-verify 行为

unittest_agent 在生成完后立即在子进程里跑两条命令，要求 **compile + correctness 都
返回 rc=0** 才把 `status` 标记为 `ok`：

| 状态           | compile | correctness | GEAK 拿到的 `--test-command`          |
|----------------|---------|-------------|---------------------------------------|
| `ok`           | ok      | ok          | ✅ 是                                  |
| `degraded`     | ok      | fail/skipped| ❌ 否（仅 prompt 附录信息，GEAK 自己 fallback）|
| `degraded`     | fail    | skipped     | ❌ 否                                  |
| `skipped`     | -       | -           | 非 Python/Triton 源（`.cu` / `.hip`）   |
| `failed`       | -       | -           | 源文件不存在 / 无法解析                |

本次：
- **rms_norm: `ok`** — 因为生成期就用原始 source 自己跟自己比，必然过。
- **softmax: `degraded`** — picker 选了 `softmax` 作 host_entry，但 `config.yaml`
  里 `target_kernel_functions` 包含的 `softmax_kernel` 在文件顶层不存在（它实际
  在 `_triton_kernels/softmax.py` 里），compile 阶段 `assert hasattr(mod,
  "softmax_kernel")` 失败。

---

## 3 · GEAK 调用结果

每个 attempt 的 GEAK 输出：

```text
/workspace/hyperloom/kernel-agent/geak/integration_demo_2/<attempt_id>/
├── COMMANDMENT.md
├── benchmark_baseline.txt              # baseline 测试输出（"Correctness: PASS"）
├── full_benchmark_baseline.txt
├── geak_agent.log
├── tasks/
├── results/
│   └── round_1/
│       ├── parallel_0/
│       │   ├── patch_0.patch           # ★ 0 字节
│       │   ├── patch_0_test.txt        # "Correctness: PASS"
│       │   ├── task_0.log              # mini-swe-agent 完整对话日志
│       │   └── traj.json               # 51 行 ndjson 完整 trace
│       ├── traj.json                   # select_agent 选优 trace
│       ├── select_agent.log            # select_agent 决策
│       └── worktrees/slot_0/           # GEAK 在这里的 git 沙盒（aiter 完整 clone）
└── final_report.json
```

### 3.1 GEAK CLI 实际命令（来自 `geak_submit.py`）

```bash
geak  -t <prompt_file>                                    \
      --yolo                                              \
      --output <output_dir>                               \
      --gpu-ids 0                                         \
      --config /opt/hyperloom/geak-config/local.yaml      \
      --kernel-path /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py \
      --repo /sgl-workspace/aiter                         \
      --test-command "python3 .../task_runner.py correctness"   ← ★ 来自 unittest_agent
      --cost-limit ...
```

### 3.2 GEAK 实际跑得怎么样

**rms_norm（`geak-91864cc5`）**

| 字段                 | 值                                  |
|----------------------|-------------------------------------|
| elapsed              | 251.1 s（约 4 min）                 |
| GEAK CLI returncode  | 124（SIGTERM at 240 s 预算到期）    |
| 写出 patches         | 1（`patch_0.patch`，0 字节，无 diff）|
| baseline 测试        | `Correctness: PASS`（unittest 接住）|
| `final_report.json`  | `{"status": "complete_no_patch", "best_patch": null, "best_speedup": null, "summary": "No best patch selected"}` |
| Coordinator 决策     | `PARTIAL`（patches 存在但 0 字节，缺真实速比） |

**softmax（`geak-eb080255`）**

| 字段                 | 值                                  |
|----------------------|-------------------------------------|
| elapsed              | 同 ~4 min 预算上限                  |
| 写出 patches         | 4（`patch_0`~`patch_3`，都 0 字节） |
| `final_report.json`  | 不存在（select_patch round 没跑完） |
| Coordinator 决策     | `PARTIAL`                           |

### 3.3 为什么 patch 是空的（重要！跟 unittest_agent 无关）

从 `task_0.log` / `traj.json` 看清楚原因：**GEAK 里用的 mini-swe-agent + claude-opus-4-7
在调用 `bash` 工具时陷入死循环**：模型一直发 `{"name": "bash", "arguments": {}}`
（空参），mini-swe-agent 的 bash tool 一直回 `"bash tool call need a command
argument, it must not be empty."`。

> rms_norm 跑了 25 步全是这种死循环；softmax 在文本回复里写出了真的优化代码
> （PyTorch fused-softmax 快路径 fallback），但同样因为 bash 工具调用失败，无法把
> 代码写到磁盘，所以 patch 文件都是 0 字节。

证据片段（来自 `geak-91864cc5/results/round_1/parallel_0/task_0.log` step 22）：

```text
THOUGHT: I need to provide the command parameter. Despite the schema being
empty, the implementation expects "command". Let me give it.

> Tool Call: bash
{}

User (tool_result: bash):
  output: bash tool call need a command argument, it must not be empty.
  returncode: 1
```

这是 **GEAK 上游的 schema / mini-swe-agent 配套问题**，跟 unittest_agent 集成是
完全独立的。换句话说：unittest_agent 把正确的 prompt + 正确的 test_command
都送到了 GEAK，但 GEAK 本身因为这个 bug 没能产出可用 patch。

---

## 4 · 把 patch apply 回 e2e 之后的效果

**本次没有触发真正的 apply** — 因为：

1. 两次 GEAK 输出的 patches 都是 0 字节 → `verification` 阶段
   （`build_verification` in `kernel_optimization.py`）找不到合法的 optimized 源
   工件，给出 `proposal.decision = "PARTIAL"`。
2. `PARTIAL` 在 SKILL.md 的契约里**禁止** Coordinator 触发 `integrate` —
   `integrate_handler` 只在 `KEEP` 决策下才会 apply patch + 重启 vLLM + 重跑
   baseline，再算 `new_tput / base_tput`，并由 `keep_threshold_pct`
   （默认 +1.0%）二次裁决 `KEEP` / `REVERT` / `NEEDS_REVIEW`。

### 4.1 如果走完整路径，apply 后会发生什么（按 `integrate_handler` 的代码）

```text
KEEP 决策 → integrate_handler 入口
   │
   ├─ 1. 备份：源文件 + 编译产物（.so / .co / .hsaco）
   │      → patches/<kernel_id>/{source.bak, *.so.bak, ...}
   │
   ├─ 2. 把 patch_path 拷到 target_file（live /sgl-workspace/aiter/...）
   │
   ├─ 3. 用 baseline_config_path（baseline 当时材化好的同一份 YAML）
   │      跑一次 Magpie，得到 new_tput
   │
   ├─ 4. gain_pct = (new_tput - base_tput) / base_tput * 100
   │      ├─ gain >= keep_threshold_pct → KEEP，写 patches/<kernel_id>/applied
   │      ├─ gain < 0                   → REVERT（先恢复 .so 再恢复源，不重 build）
   │      └─ 0 <= gain < threshold      → NEEDS_REVIEW
   │
   └─ 5. 写 result：{decision, base_tput, new_tput, gain_pct, report_path, workspace}
```

### 4.2 本次 baseline 数据（参考值）

> 来自最早跑通的那次 `inference_optimizer optimize` 进程（kill 之前 baseline
> 已完成）：

```text
SOURCE: /workspace/hyperloom/runs/baseline/3862c878243f4f16a2d58756271f51ba/
         benchmark_vllm_20260518_090730/inferencex_result.json
TP=1, CONC=32, ISL=1024, OSL=256, model=Qwen3-8B (bf16), framework=vllm

output throughput      : 2162.7 tok/s/gpu
mean TTFT              : 585.0 ms
mean E2EL              : 3781.4 ms
mean TPOT              :   3.13 ms
```

如果 GEAK 给出一个真正能用的 rms_norm patch 并被 `integrate` 接受，预期：
- rmsnorm 占 GPU 时间 **~8.5%**（来自 candidate.gpu_pct）；
- 假设 kernel 内部加速 1.3x（GEAK 历史平均），全模型理论 gain ≈ `8.5% × (1 - 1/1.3) ≈ 2%`；
- 即 `new_tput ≈ 2162.7 × 1.02 ≈ 2206 tok/s/gpu`；
- 落到 KEEP 阈值（+1.0%）之上 → `decision=KEEP`。

但**本次跑没走到这一步**，所以表里没有 `new_tput`/`gain_pct` 真实数字 —
得等 GEAK 上游修好 bash 工具 schema / 或换 codex / claude backend 来跑。

### 4.3 备用证据：之前 Qwen3-8B 跑过的 GEAK rmsnorm 案例

`/wekafs/zihao/2026/kernel_agnet/hl_exp/qwen3-8b-rerun-2026-05-12-03-03-15/`
有一次完整的 Qwen3-8B GEAK 跑（5 月 12 日，那次还没接入 unittest_agent）：

| kernel                                       | round  | speedup | KEEP? |
|----------------------------------------------|--------|---------|-------|
| `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` (rmsnorm) | r4 | 1.18x | ✅ |
| `triton_tem_fused_mm_0` (logits matmul)      | r3     | 1.66x   | ✅    |
| `triton_tem_fused_mm_silu` (gate proj)       | r2     | 1.32x   | ✅    |

这些都跑在 inductor 生成的 `/tmp/torchinductor_root/...` 路径上，并不在 aiter
源里。原版 prompt 没有 unittest harness，GEAK 是自己造的 `bench_<kernel>.py`，
所以微 bench 数字可靠性较低（手工抽样会有 ±5% 噪声）。

接入 unittest_agent 之后预期改进：
1. **正确性硬门槛** — 不再依赖 GEAK 自己写的 harness，而是 hyperloom 生成的
   golden-snapshot 对比，过 atol=1e-2（bf16 / fp16）/ 5e-2（fp8）才算 PASS；
2. **shape 真实** — 是 TraceLens 从 live profile resolve 出来的真 shape，不是
   `(64, 64)` 这种 toy；
3. **环境一致** — `SGLANG_*/AITER_*/VLLM_*/HIP_*` 17 个 env var 跟服务端同步
   `setdefault`；
4. **多 backend 共享 harness** — 同一份 harness 也能传给 claude / codex /
   cursor backend，speedup 数字横向可比。

---

## 5 · 关键路径产物清单（便于复现 / 审计）

```text
$USER_DATA_PATH/                                     # = /workspace/hyperloom
├── kernel-agent/
│   ├── unittests/<session>/<attempt_id>/            # ← 本次新增层
│   │   ├── config.yaml
│   │   ├── unittest_meta.json
│   │   ├── scripts/task_runner.py
│   │   └── source/{<kernel>.py, _baseline_snapshot/<kernel>.py}
│   ├── runs/<session>/
│   │   ├── logs/kernel_optimization/<run>.log       # invoke_backend 调用链
│   │   ├── prompts/<attempt_id>.md                  # 完整 GEAK prompt
│   │   ├── optimization_attempts.jsonl              # 每个 attempt 一行
│   │   ├── results/<kernel_id>.json                 # 含 backend_paths.unittest_*
│   │   └── verification/<kernel_id>.json
│   └── geak/<session>/<attempt_id>/                 # GEAK CLI 输出
│       ├── final_report.json
│       └── results/round_*/parallel_*/patch_*.patch + *_test.txt + task_*.log
├── runs/baseline/<task_id>/benchmark_vllm_*/{inferencex_result.json, server.log}
└── patches/<kernel_id>/                             # integrate 接受后的备份
```

观察性字段（写在 `optimization_attempts.jsonl[].backend_paths`）：

| key                       | 含义                                                          |
|---------------------------|---------------------------------------------------------------|
| `unittest_status`         | `ok` / `degraded` / `skipped` / `failed`                      |
| `unittest_out_dir`        | harness 所在目录                                              |
| `unittest_test_command`   | 实际给 GEAK 的 `--test-command`（仅 status=ok 时存在）         |
| `geak_final_report`       | `geak/<attempt>/final_report.json`                            |
| `geak_latest_patch`       | 最新 patch 路径                                               |
| `geak_per_task_best_speedup` | 当 select_patch round 超时被 SIGTERM 时，从 best_results.json 兜底 |

旁路开关：
```bash
export HYPERLOOM_DISABLE_UNITTEST_AGENT=1     # 调试用，关闭整个 pre-step
```

---

## 6 · 下一步要做的事

1. **GEAK bash tool 调度 bug 排查** — 既然 prompt 和 test_command 都对了，
   下一步要排查 mini-swe-agent / claude-opus-4-7 tool schema 互操作问题
   （怀疑是 LiteLLM gateway 把 `properties` 字段抹掉了），先解决了才能拿到
   真的 patch + speedup 数字。
2. **HIP `.cu` kernel 支持** — 现在 unittest_agent 对 `.cu` / `.cuh` 直接 skip，
   后续需要补 hipcc + cpp_extension.load 模板，让 attention/gemm 这类
   ASM-backed kernel 也能走 harness 评测。
3. **picker 加 `host_entry` 显式字段** — 当 TraceLens candidate 已经知道
   launcher 名字（如 `rms_norm`）时直接传，省得 picker 启发式猜（softmax 这种
   主体在子模块的就猜不准）。
4. **接入 `integrate` 后续的 cost 模型** — 把 unittest 的 `performance` 数据
   作为 `integrate` 的 prior，给 Coordinator 一个不用真 e2e 重跑就能预估
   gain 的快捷路径（节省 30 min/kernel 的 baseline 时间）。
