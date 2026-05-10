---
name: inference-optimization
description: |
  面向 AMD MI355X GPU 上 LLM 推理服务的自治 DFS 引导推理优化。
  使用启发式打分的深度优先搜索，系统探索优化动作
  （后端切换、服务参数、内核优化、目标对比），在准确率作为硬约束的前提下，
  最大化每 GPU 吞吐量（tok/s/GPU）。
globs:
  - "**/inference*optim*"
  - "**/benchmark*"
  - "**/sglang*"
  - "**/vllm*"
---

# 推理优化 - DFS 编排器

## 概览

这个 skill 用于优化 AMD MI355X GPU 上 LLM 推理服务的吞吐量（tok/s/GPU）。
它会在启发式打分函数的引导下，对优化动作执行**深度优先搜索**。
搜索过程完全自治，不需要人工提示。

**主要目标：** 最大化 `tok/s/GPU`
**硬约束：** 准确率（数值正确性）不得下降
**可选目标：** 如果提供了外部基线（例如 NVIDIA B200），目标差距会作为所有动作分数的紧迫性乘子。

## 执行模式

这个 skill 支持两种执行模式。**开始前必须先阅读对应模式的文档：**

- **本地模式**（Hyperloom 容器，Ray 调度的 GEAK CLI）：见 [`modes/LOCAL.md`](modes/LOCAL.md)
- **远程模式**（SaFE RayJob，`exec_on_gpu`）：见 [`modes/REMOTE.md`](modes/REMOTE.md)

**自动检测：**
- `MODE=local` → 本地模式
- 远程客户端上下文 → 远程模式

## 铁律（不可协商）

这些规则适用于所有模式。违反任意一条都会使本次优化运行无效。

### IR-1：并行提交所有内核候选

kernel-opt 动作必须将 `GEAK_TOP_CANDIDATES`（默认 5 个）候选同时提交给所有启用的后端（`KERNEL_OPT_BACKENDS`）。当存在多个候选时只提交 1 个，或对后端串行提交而不是并行提交，均属于违规。

### IR-2：GEAK 提交前绝不修改内核源码

必须提交**按原样提取**的内核源码。不要移除装饰器、修改 stride、把 `@triton_heuristics` 替换成 `@triton.jit`，也不要做任何“清理”编辑。GEAK 的 agent 会在内部处理内核适配。

### IR-3：集成（Phase 8）是强制步骤

GEAK 返回优化后的内核后，必须执行 integrate 动作（patch → re-baseline → decide）。跳过该步骤意味着 GEAK 结果从未经过端到端验证。re-baseline 使用 `run_baseline.sh`，不存在 `run_benchmark.sh`。详情见 `actions/integrate.md`。

### IR-4：每次启动服务前都要 kill_server + check_gpu_memory

每次服务启动前，都必须先杀掉已有服务进程，并确认 GPU 显存已释放。

### IR-5：安全进程管理

**绝不要使用 `pkill -f sglang`**，这会在远程模式中杀掉 Ray worker。只能使用：

```bash
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
# 或 vLLM 使用：
kill $(pgrep -f 'python.*-m vllm.entrypoints') 2>/dev/null
```

kill 与重新启动之间等待 `SERVER_KILL_WAIT_S` 秒。profiling 结束后必须始终执行 `unset PROFILE SGLANG_TORCH_PROFILER_DIR`。

### IR-6：使用 `patch_inductor.py --target-file` 修补 Inductor

始终使用带 `--target-file` 的 `scripts/patch_inductor.py`。`--cache-dir` 选项已移除。

**关键：** 当 GEAK 修改 block size 或 warp count 时，也必须通过 `--best-config` 传入更新后的 tiling 参数。只 patch 内核 `.py` 而不更新 `.best_config` 会造成数值损坏（输出乱码）。详情见 `actions/integrate.md`。

### IR-7：绝不修改 GEAK 配置

本地模式和远程/RayJob 模式下，GEAK 都只能通过 Ray 调度的 CLI（`geak_ray_submit.py`）使用；不要使用 GEAK service API 或其他客户端。见 [`modes/LOCAL.md`](modes/LOCAL.md) 和 [`modes/REMOTE.md`](modes/REMOTE.md)。

将 GEAK 配置视为**只读基础设施**。skill 运行时不得修改任何 GEAK 配置文件、设置或参数。具体包括：

- **不要**修改 GEAK server config、workspace settings 或 API configuration
- **不要**写入或改动 GEAK config/settings 目录下的任何文件
- **不要**在运行时修改 `KERNEL_OPT_WORKSPACE`、`GEAK_STEP_LIMIT` 或其他常量
  （使用上方常量表中的值，或使用用户覆盖值）
- **不要**修改任何属于 GEAK 的测试数据、结果或配置文件
  （例如 `tests/test_data/`、`server/config.py`、`server/templates/`）

唯一允许的 GEAK 交互方式是通过 `geak_ray_submit.py`，它会在 Ray GPU 隔离下调用 `geak` CLI。

违规（修改模型/后端）= 立即判定本次运行无效。

### IR-7b：编排器不得自己编写内核优化代码

所有内核优化都必须通过配置的 `KERNEL_OPT_BACKENDS`（`geak`、`codex`、`claude`、`llm`）执行。编排器 Agent 的职责是**准备 prompt、向后端提交任务、验证结果并集成**，绝不能直接编写优化后的 Triton/HIP/CUDA 内核。

即使编排器与某个后端使用同一个 LLM 模型（例如 Claude 通过 OOB Claude 编排），也仍然必须使用后端工具链。后端提供隔离工作区、GPU 侧验证、可复现轨迹，以及 Ray 管理的 GPU 调度，这些都是直接在聊天中生成代码所不具备的。

违规 = 立即判定本次运行无效。

**其他模式专属铁律定义在 [`modes/REMOTE.md`](modes/REMOTE.md)（IR-8 到 IR-11）和 [`modes/LOCAL.md`](modes/LOCAL.md)（IR-12 到 IR-16）。**

## 内核优化与工具常量

以下所有值都是**唯一事实来源**。所有动作都按名称引用这些值。

| 常量 | 值 | 说明 |
|----------|-------|-------------|
| `KERNEL_OPT_BACKENDS` | `geak,codex` | 逗号分隔的启用后端。可任意组合：`geak`、`codex`、`claude`、`llm`。用户可在 prompt 中覆盖。 |
| `OOB_ROUND_ITERATIONS` | 3 | 每轮 Codex/Claude 的迭代次数（提交 → 本地 benchmark → 反馈 → 重新提交）。最佳结果获胜。 |
| `KERNEL_OPT_IMAGE` | *由 CI 或用户提供* | 所有 kernel-opt 后端（GEAK + OOB）使用的框架镜像。每次运行一个镜像，由框架（SGLang/vLLM）决定。 |
| `KERNEL_OPT_WORKSPACE` | `control-plane-moe` | kernel-opt 后端（GEAK + OOB）的 SaFE workspace。用户可覆盖。 |
| `GEAK_STEP_LIMIT` | 100 | 每个 GEAK 任务的最大 agent step 数 |
| `GEAK_MAX_RETRIES` | 3 | 每个内核的最大提交重试次数 |
| `GEAK_MAX_SUBMISSIONS` | 15 | 每次运行的 GEAK 总提交预算 |
| `GEAK_TOP_CANDIDATES` | 5 | 要提交的顶部内核候选数量 |
| `GEAK_CONSECUTIVE_DISCARDS` | 5 | 连续丢弃达到该次数后停止 |
| `GEAK_WALL_CLOCK_MIN` | 30 | kernel-opt 动作的最大墙钟时间（分钟） |
| `GEAK_POLL_INTERVAL_S` | 60 | GEAK 任务状态轮询间隔（秒） |
| `GEAK_POLL_TIMEOUT_MIN` | 15 | 单个 GEAK 任务的最大轮询时间（分钟） |
| `MIN_GPU_PCT` | 3 | 将内核视为 GEAK 候选的最小 GPU 时间占比 |
| `SERVER_KILL_WAIT_S` | 10 | kill 服务与重启之间的等待秒数 |
| `FILTERED_TRACE_NAME` | `filtered-TP-0.trace.json.gz` | TraceLens 分析首选 trace 文件 |

**始终将 `KERNEL_OPT_IMAGE` 传给所有 kernel-opt 后端（GEAK + OOB），无论内核类型如何。** 对于源码存在于镜像中的内核（例如 `/sgl-workspace/aiter/`），pod 使用同一个镜像。对于运行时生成的内核（例如 `torch.compile` 生成的 `/tmp/torchinductor_root/`），不要在 prompt 中包含 `kernel_url`/`kernel_repo`；应将文件复制到共享 NFS，或只依赖 `files[].content`。

**远程模式常量位于 [`modes/REMOTE.md`](modes/REMOTE.md)。**

## 架构

```
SKILL.md（本文件）             — DFS 编排器：循环、启发式、调度
actions/*.md                   — 自包含动作模块（11 个动作）
kernel-opt/                    — 每个后端的内核优化参考
  geak.md                      — 通过 Ray 调度 CLI 使用 GEAK
  codex.md                     — 通过 OOB Ray 调度 CLI 使用 Codex
  claude.md                    — 通过 OOB Ray 调度 CLI 使用 Claude Code
  llm.md                       — LLM Proxy（直接 API）
kb/                            — RAG 知识库（JSONL + query/ingest 脚本）
scripts/                       — baseline/profiling/accuracy shell 脚本
modes/                         — 模式专属执行细节（LOCAL.md、REMOTE.md）
KNOWLEDGE-BASE.md              — 旧版 KB（已归档，并作为种子写入 kb/entries.jsonl）
```

## 常见陷阱（来自 CI 日志验证）

这些是生产 CI 运行中反复出现的错误。**执行前必须阅读。**

1. **PATH：始终先执行 `export PATH="/opt/venv/bin:$PATH"`。** 系统 python3
   （`/usr/bin/python3`）没有 sglang/vllm/numpy。每个 bash 命令都必须
   先加入 venv。失败模式：`ModuleNotFoundError: No module named 'sglang'`。

2. **绝不要覆盖用户指定的 TP。** 如果 prompt 写了 TP=8，就使用 TP=8。不要
   自动检测 GPU_COUNT 并覆盖成 TP=1，因为大模型（120B+）无法在单 GPU 上运行。
   失败模式：OOM 或服务崩溃。

3. **vLLM 参数与 SGLang 不同。** 常见错误：`--disable-log-requests` 不是
   合法 vLLM 参数。vLLM 应使用 `--disable-log-stats`。使用不熟悉参数前，
   始终先检查 `vllm serve --help`。失败模式：`unrecognized arguments` → 服务崩溃。

4. **使用 `run_baseline.sh`，不要手动启动服务。** 该脚本会以经过测试的顺序处理
   服务启动、健康检查等待、benchmark 和 profiling。手动启动会跳过健康检查，
   经常遇到 Exit code 144（陈旧进程导致的 SIGTERM）。

5. **绝不修改 GEAK 运行时配置。** 见 IR-7。GEAK 通过渲染后的 CLI 配置和
   `geak_ray_submit.py` 调用。

6. **记录所有外部调用的开始/结束时间戳。** 调用任何外部组件（GEAK、OOB、
   LLM proxy、TraceLens 或未来后端）前，运行
   `python3 $SCRIPTS_DIR/trace_action.py --component <name> --action start`。
   组件完成后运行 `--action end`。这支持按消息归因成本。如果特定后端 skill
   已包含 tracing 步骤，按其步骤执行。否则，将此规则作为 fallback 使用。
   tracing 失败不阻塞执行；如果脚本不可用，则跳过。

## DFS 搜索树

**阶段：** SETUP → CLASSIFY → TARGET ANALYSIS（可选）→ BASELINE（+ GSM8K 准确率）→ PROFILE → HEURISTIC SCORING → DFS LOOP（取最高分动作 → 执行 → 重新打分 → 重复）→ SWEEP → REPORT

**DFS 循环动作：** backends、params、kernel-opt、integrate、sweep。它们由启发式打分，并按最高分优先弹出。每个动作可以推入子动作（例如 PROFILE 推入 GEAK candidates，BACKENDS 推入组合测试）。agent 沿最有希望的分支深度优先探索，并在分数变化时回溯。

agent 不限于预定义动作。如果 profiling 暴露了意外瓶颈，或 KB 提示了新技术，也可以创建临时动作，并用同一启发式进行打分。

这是单 agent 的顺序循环。每个动作运行完成后才会重新打分。并行发生在动作内部（例如 GEAK 并行提交 5 个内核），而不是动作之间。

## 自治规则

**自治执行，不需要人工确认。** 执行以下操作前不要询问用户：
- 在 SaFE 上创建/停止 RayJob（远程模式）
- 通过 Ray（远程模式）或本地运行 baseline/profiling 脚本
- 提交 GEAK 任务
- 杀掉/重启服务（RayJob 内部或本地）
- patch 内核（Inductor cache 或源码文件）
- 回滚失败 patch

**自治意味着不要请求许可，不代表可以跳步。** 编排器循环中的每个编号步骤（1-11）都是**强制的**，包括：
- Step 3：TARGET ANALYSIS（如果提供了目标数据）
- Step 4：KB WARM-UP（始终执行，baseline 前查询 KB）
- Step 11：KNOWLEDGE HOOK（始终执行，报告后摄取发现）

跳过任何强制步骤都会使运行无效。所有步骤完成后，将**最终优化报告**呈现给用户。

## 启发式打分函数

每个候选动作按以下方式打分：

```
score = (expected_tput_gain_per_gpu / cost_minutes)
        × (1 - accuracy_risk)
        × (1 - crash_risk)
        × target_gap_multiplier
```

| 组件 | 来源 | 范围 |
|-----------|--------|-------|
| `expected_tput_gain_per_gpu` | KB 查询 + 模型类别先验 | 0-100+ tok/s/GPU |
| `cost_minutes` | 估算墙钟时间 | 2-120 min |
| `accuracy_risk` | 来自 KB（kernel mods = 0.15，backends = 0.1，params = 0.0） | 0.0-1.0 |
| `crash_risk` | 来自 KB（vendor kernel mods = 0.5，scheduling = 0.05） | 0.0-1.0 |
| `target_gap_multiplier` | `1 + min(target_gap_pct, 100) / 100` | 1.0-2.0 |

### 初始分数先验（按模型类别）

| 动作 | Dense | MoE+MLA | MoE+SWA | MoE+MLA+NSA |
|--------|-------|---------|---------|-------------|
| backends | 3 | **9** | **8** | **10** |
| params | 5 | 6 | 7 | 5 |
| kernel-opt（GEAK） | **8** | 2 | 2 | 2 |
| torch.compile | **7** | 0 | 0 | 0 |
| sweep | 1 | 1 | 1 | 1 |

每个动作完成后，根据测量结果更新分数。

### 分数更新规则

每个动作完成后：

1. **动作成功（gain > 0%）：** 提升相似动作。例如，如果 `backends` 获得 +5%，
   将剩余未测试 backend 提升 1.5 倍。提升 `combined_test` 分数。
2. **动作失败（gain ≤ 0%）：** 将相似动作降低到 0.5 倍。
3. **出现 2 个以上 backend 获胜后：** 推入 `combined_backends_test`，分数 = 各单项分数之和 × 1.5
4. **所有 backend 测试完成后：** 推入 `re-profile`（用于发现新的 GEAK 目标）
5. **kernel opt 保留后：** 推入 `re-profile + next-kernel`，并提升分数
6. **kernel opt 丢弃后：** 将剩余 kernel 分数降低到 0.7 倍
7. **当所有动作分数 < 1.0 时：** 进入 sweep → report

## 状态 Schema

编排器在整个运行过程中维护以下状态：

```python
state = {
    "model_name": "",
    "model_class": "",           # dense / moe_mla / moe_swa / moe_mla_nsa
    "framework": "sglang",
    "tp": 8,
    "gpu_type": "MI355X",

    "baseline_tput_per_gpu": 0.0,
    "current_tput_per_gpu": 0.0,
    "cumulative_gain_pct": 0.0,

    "target_tput_per_gpu": None,  # 来自 target-analysis（如果可用）
    "target_gap_pct": None,

    "torch_compile_status": None,  # success / failed / skipped
    "accuracy_reference": None,    # reference output 路径
    "baseline_accuracy": None,     # baseline eval 得到的 GSM8K exact_match 分数（0.0-1.0）
    "accuracy_threshold": 0.01,    # REVERT 前允许的最大准确率下降（绝对值）

    "action_stack": [],            # (score, action_name, params) 的优先级栈
    "completed_actions": [],       # (action_name, gain_pct, status) 日志
    "kernel_candidates": [],       # 来自 profiling
    "winning_backends": [],        # 来自 backend exploration
    "winning_params": [],          # 来自 param tuning

    "total_wall_minutes": 0,
    "total_geak_submissions": 0,
    "consecutive_discards": 0,
}
```

## 编排器循环

```
过程 optimize():

  1. SETUP
     → 执行 actions/setup.md
     → 设置 MODEL、TP、CONC、FRAMEWORK 和路径

  2. CLASSIFY
     → 执行 actions/classify.md
     → 设置 model_class、torch_compile_viable 和初始分数先验

  3. TARGET ANALYSIS（如果提供了 $TARGET_DIR）
     → 执行 actions/target-analysis.md
     → 设置 target_tput_per_gpu、target_gap_pct 和 target_gap_multiplier

  4. KB WARM-UP
     → 查询该模型的 KB：python3 kb/kb_query.py --model "$MODEL_NAME" --top-k 20
     → 将 KB 信息用于调整分数先验

  5. BASELINE
     → 执行 actions/baseline.md
     → 设置 baseline_tput_per_gpu、torch_compile_status 和 accuracy_reference
     → 运行 GSM8K eval → 设置 baseline_accuracy（强制步骤，这是准确率下限）

  6. PROFILE
     → 执行 actions/profile.md
     → 用 (name, gpu_pct, source) 填充 kernel_candidates

  7. BUILD ACTION STACK
     → 使用启发式为所有候选动作打分
     → 按分数排序后推入 action_stack（最高分优先）

  8. DFS LOOP:
     当 action_stack 非空且 stopping_criteria_met() 为假时：
       a. 弹出最高分动作
       b. 执行动作（调度到 actions/*.md）
       c. ACCURACY GATE：如果动作的 accuracy_risk > 0：
          - 通过 scripts/eval_accuracy.sh 运行 GSM8K eval
          - 将新分数与 state.baseline_accuracy 对比
          - 如果下降 > accuracy_threshold（默认 0.01）：REVERT，标记 FAIL
          - 见下方“准确率门控协议”
       d. 测量结果：new_tput_per_gpu
       e. 更新状态：current_tput_per_gpu、cumulative_gain_pct
       f. 重新为所有剩余动作打分（每次优化后收益都会改变）
       g. 推入执行过程中发现的新子动作
       h. 写入 completed_actions 日志

  9. SWEEP
     → 执行 actions/sweep.md（完整 ISL/OSL/CONC 参数 sweep）

 10. REPORT
     → 执行 actions/report.md（生成优化报告 + KB 贡献）

 11. KNOWLEDGE HOOK
     → .cursor/hooks/knowledge-sink.py hook 自动触发
     → 摄取运行期间发现的新知识
```

## 准确率门控协议

动作由其 `accuracy_risk` 值决定是否需要门控。**Baseline GSM8K 准确率在步骤 5（BASELINE）测量，并存入 `state.baseline_accuracy`。** 后续任何 `accuracy_risk > 0` 的动作都必须通过准确率门控后才能 KEEP。

### 哪些动作触发门控

| accuracy_risk | 动作 | 是否需要门控 |
|:-------------:|---------|:-------------:|
| 0.0 | 服务调度参数（decode-steps、cuda-graph-max-bs、mem-fraction、chunked-prefill） | 否 |
| 0.05-0.15 | 内核修改（GEAK）、GEMM 调优 | **是** |
| 0.1 | 后端切换（aiter、alter、attention backends） | **是** |
| 0.3 | 影响精度的参数（kv-cache-dtype fp8、quantization changes） | **是** |

### 门控流程

对任何 `accuracy_risk > 0` 的动作，在吞吐 benchmark 成功后：

1. **运行 GSM8K eval**，使用 InferenceX 的 lm-evaluation-harness 访问当前运行中的服务：
   ```bash
   EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
     RESULTS_DIR="$RESULT_DIR/eval_gsm8k_${ACTION_NAME}" \
     bash "$SKILL_ROOT/scripts/eval_accuracy.sh"
   ```

2. **从 eval summary 提取分数：**
   ```bash
   new_accuracy=$(python3 -c "
   import json, glob
   f = sorted(glob.glob('$RESULT_DIR/eval_gsm8k_${ACTION_NAME}/eval_summary_gsm8k.json'))[-1]
   d = json.load(open(f))
   scores = list(d['scores'].values())[0]
   print(scores.get('exact_match,strict-match', scores.get('exact_match,none', 0)))
   ")
   ```

3. **与 baseline 对比：**
   ```
   accuracy_drop = baseline_accuracy - new_accuracy
   if accuracy_drop > accuracy_threshold（默认 0.01 = 1 个百分点）:
       立即 REVERT
       写入 KB：该 action+model 的 accuracy_risk=1.0
       将动作标记为 FAIL（准确率下降）
   else:
       KEEP — 准确率在容忍范围内
   ```

### 内核级预检查（可选，仅用于 GEAK/kernel mods）

完整 GSM8K eval 前，可以用快速 micro-benchmark sanity check 捕获明显错误：
```python
assert torch.allclose(original_output, optimized_output, atol=1e-3, rtol=1e-3)
```
这不能替代 GSM8K 门控；它只是一个提前退出优化。

### 跳过门控的动作

**setup、classify、profile、sweep、report** 这些动作是只读的，永远不会修改服务计算路径。纯调度参数（accuracy_risk=0.0）也跳过。

## 停止条件

| 条件 | 动作 |
|-----------|--------|
| 所有动作分数 < 1.0 | 进入 sweep |
| 累计收益 > 25% | 进入 sweep |
| 所有动作连续丢弃 5 次 | 进入 sweep |
| 总墙钟时间 > 180 min | 进入 sweep |
| 已超过目标（gap ≤ 0%） | 进入 sweep |
| 服务崩溃 2 次以上 | 紧急停止，报告部分结果 |

## KB 集成

每个动作前，查询 KB 中的相关知识：

```bash
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME $ACTION_NAME" --top-k 5 --compact
```

每个动作产生新发现后，摄取到 KB：

```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category $CATEGORY --model "$MODEL_NAME" \
    --action "$WHAT_WAS_DONE" --lesson "$KEY_TAKEAWAY" \
    --tags $TAGS --gain $GAIN --status $STATUS
```

KB 提供：
- **先验知识：** 跳过已知在该模型类别上会失败的动作
- **分数校准：** 基于历史结果调整预期收益
- **冲突检测：** 自动标记矛盾信息

## 动作调度

| 动作 | 模块 | 触发时机 |
|--------|--------|------|
| Setup | [`actions/setup.md`](actions/setup.md) | 始终第一步 |
| Classify | [`actions/classify.md`](actions/classify.md) | 始终第二步 |
| Target Analysis | [`actions/target-analysis.md`](actions/target-analysis.md) | 如果提供 `$TARGET_DIR` |
| Baseline | [`actions/baseline.md`](actions/baseline.md) | classify 之后 |
| Profile | [`actions/profile.md`](actions/profile.md) | baseline 之后 |
| Backend Exploration | [`actions/backends.md`](actions/backends.md) | DFS 循环 |
| Server Params | [`actions/params.md`](actions/params.md) | DFS 循环 |
| Kernel Optimization | [`actions/kernel-opt.md`](actions/kernel-opt.md) | DFS 循环 |
| Integration | [`actions/integrate.md`](actions/integrate.md) | 每个内核的子动作 |
| Parameter Sweep | [`actions/sweep.md`](actions/sweep.md) | DFS 循环之后 |
| Report | [`actions/report.md`](actions/report.md) | 始终最后 |

## 参考：关键经验

以下是最重要的已验证经验。完整细节见 KB 和动作模块。

1. **后端切换通常优于参数 sweep。** GLM-5：backends +16.2%，params <1%。
   始终先探索 backend，再 sweep 参数。

2. **组合协同可能超线性。** 两个 +3% backend → 组合后 +16.2%。
   始终一起测试获胜项。

3. **torch.compile 是 GEAK 获得大收益的前提。** 启用 compile 时最高 +14.72%。
   未启用时 ≤1.76%。

4. **Benchmark 公平性至关重要。** Kimi-K2.5 的 “+40.4%” 是无效结果；
   控制 CONC 不匹配后实际为 +0.81%。始终保存并复用 baseline config。

5. **服务参数调优可能占主导。** Kimi vLLM 通过 gpu-mem + max-num-seqs 获得 +84%。
   CUDA graph coverage 配置错误时，修正后可 +35%。

6. **GEAK 无法击败 vendor kernel。** 永远不要将 `Cijk_*` 或 `aiter::*` 提交给 GEAK。

7. **必须 patch STANDALONE 文件，而不是 graph module。** patch graph module = 0%。
   patch standalone = +9%。

8. **使用 Python AST 做源码 patch。** 朴素 regex 会删除模块级变量。

## 参考：进程管理

- **绝不要在脚本中使用 `pkill -f "sglang.launch_server"`**，这会杀掉脚本自身。
- **kill 服务和重启之间等待 `SERVER_KILL_WAIT_S` 秒**（默认 10）。
- profiling 结束后**始终执行 `unset PROFILE SGLANG_TORCH_PROFILER_DIR`**。
- TraceLens **始终使用 filtered traces**（raw：349MB，filtered：5MB）。
- TraceLens 不支持 `rocprofv3` 格式，只支持 PyTorch Kineto。

## 参考：Benchmark 指标

| 指标 | 单位 | 含义 |
|--------|------|---------|
| `output_throughput` | tok/s | 每秒输出 token 数 |
| `tput_per_gpu` | tok/s/GPU | `output_throughput / TP` |
| `mean_tpot_ms` | ms | 每个输出 token 的时间（decode 延迟） |
| `mean_ttft_ms` | ms | 首 token 时间（prefill 延迟） |

## 参考：服务参数表

已验证的参数表（SGLang、vLLM）和模型专属配置见 [`KNOWLEDGE-BASE.md`](KNOWLEDGE-BASE.md)。查询最新 KB：

```bash
python3 $SKILL_ROOT/kb/kb_query.py --category server_params --compact
```

## 参考：vLLM 集成

所有脚本都支持 `FRAMEWORK=vllm`。参数映射：

| SGLang | vLLM | 备注 |
|--------|------|-------|
| `--model-path` | `vllm serve <model>`（位置参数） | - |
| `--mem-fraction-static 0.8` | `--gpu-memory-utilization 0.85` | - |
| `--disable-radix-cache` | `--no-enable-prefix-caching` | 用于随机 benchmark |
| `--enable-torch-compile` | 默认开启（level=3） | 关闭：`--enforce-eager` |
| `SGLANG_TORCH_PROFILER_DIR` | `VLLM_TORCH_PROFILER_DIR` | 服务启动前设置 |
