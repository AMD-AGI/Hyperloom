# Hyperloom Roofline-v2 设计文档

| 字段 | 值 |
|---|---|
| Status | Draft v2.0（**v1 sub-agent 方向作废，回退后重做**） |
| Owner | xiaofei |
| Branch | `feature/xiaofei/roofline-v2` |
| Worktree | `/wekafs/xiaofei/Hyperloom-roofline-v2` |
| Base | `main` @ `550d24f` |
| Last updated | 2026-05-19（v2.0 重写） |
| 强制规则 | **任何方案改动必须先修改本文档再改代码**；本文档是单一事实来源 |
| v1 → v2 关键变更 | (a) roofline 改为**复合 action**（profile + trace_analyze），**不再调 LLM 二次解读**；(b) analysis.md 直接全文注入主 Orchestration prompt（TraceLens 团队明确反馈的正确用法）；(c) `select_kernels` rename → `trace_analyze`；(d) `discovered_flags` 改为**分层渲染**（按 action × tested 状态），让主 LLM 看到真实 flag 名而非统计行；(e) 接入 Anthropic prompt caching（Claude Code SDK 自动 caching 已在工作，本 PR 优化 prompt 结构 + 度量 hit rate） |

---

## 1. TL;DR

Hyperloom 当前的 Orchestration LLM 在 prompt 里看到的关于 trace 与 flag
的信息**严重残缺**：

* `last_select_kernels=...` 只一行字符串（`top=[k001,...] reusable=[...]`），
  TraceLens `analysis.md` 全文从未被注入；
* `discovered_flags=sglang:backend=42/param=58` **只有数字统计**，58 个真实
  flag 名 LLM 完全看不到 —— 当前 LLM "选" flag 完全是靠 prior knowledge
  + `params_search.tested[fp]` 反推，**幻觉率高**；
* `MODEL_CLASS_ACTION_PRIORS` 是静态 model-class 先验，不反映"当前优化栈
  下 comm 已从 50% 降到 15%、应转 compute" 这种**瓶颈漂移**。

结果是 LLM 60 min session 普遍只跑 1 次 profile、盲推荐 flag 名、
`cumulative_gain_validated` ≈ 0。

### Roofline-v2（v2.0 重写后的方案）

**核心思路 — TraceLens 团队明确反馈的"正确用法"**：

> "应该直接把分析文档（analysis.md）给到 orchestrator，**不要二次解读**"

落地为三件事：

1. **`roofline` 是一个复合 action**（macro / pipeline），其 executor 内部
   按顺序编排 `profile → trace_analyze`（rename 自 `select_kernels`），
   atomic 产出一份新 TraceLens snapshot（`last_profile_trace` +
   `last_select_kernels.analysis_md_text` + `roofline_snapshot_id`）。
   **executor 不调任何 LLM**，不写结构化的 RooflineAnalysis dict。
2. **analysis.md 全文直接注入** 主 Orchestration prompt（snapshot 内
   缓存，B2 决策语义）；同时 `_format_discovered_flags` 重写为**分层
   Z 方案**（按 `sglang.backends` / `sglang.params` × `tested 状态`），
   让主 LLM 看到完整 flag 列表 + 命中率/gain。主 LLM 在自己上下文里直接
   完成 "analysis.md → 选 flag → emit PRUNE_BRANCH / PROPOSE_ACTION" 决策。
3. **接入 Anthropic prompt caching**：claude-agent-sdk 内部已 automatic
   caching（OOB `task_manager.py` 已在读 `cache_creation_input_tokens`
   作为证据），本 PR 优化 prompt 结构最大化 cache hit + 暴露 hit rate 给
   audit。

`roofline` action 是 `backends` / `params` / `kernel_opt` /
`comm_optimization` 这 4 个优化 action 的 **sequence_denial 前置依赖**
（没有 fresh snapshot 不允许开始优化）；60 min session 内 `roofline`
被主 LLM 主动 propose 2-3 次（首次 + 每次 `cumulative_gain_validated`
跳 +3% 后 LLM 决定 refresh），让决策始终基于当前优化栈下的最新报告。

**硬指标**：Qwen3-32B + TP=8 + ISL/OSL=1024/1024 + CONC=64 上
`cumulative_gain_validated_pct` **≥ +5%**；Llama-70B 同 workload
不劣化（≥ -1%）证明通用性。

---

## 2. 目标 与 硬指标

| 项 | 内容 |
|---|---|
| 数据源 | **仅** TraceLens 三件套：`kernel_candidates.json` / `analysis.md` / `summary.json`。**PMC 路径彻底放弃**（已知有 bug 且未修），不在任何路径里使用 |
| 主硬指标 | Qwen3-32B + TP=8 + ISL/OSL=1024/1024 + CONC=64 上 `cumulative_gain_validated_pct` ≥ **+5%**（baseline = 当前 main HEAD `550d24f`） |
| 通用性硬指标 | 至少 1 个非 MoE workload（dense Llama-70B 同参数）`delta ≥ -1%`，证明机制不依赖 R1/Qwen MoE 特性 |
| 红线 | 不硬编码任何 model 名 / 家族；不改 `effective_score` 公式；不引入新 SharedState 顶层字段；不动 `closing_phase` / install gate / ledger 数据结构；不写 `identify_gaps.py` 等价物；不引入 PMC 或新测量手段 |
| 单 PR 边界 | 全部 commit 落在 `feature/xiaofei/roofline-v2` 一个分支（不拆 PR），按 C1-C7 commit 拆分内部步骤 |

---

## 3. 现状（已调研确认）

### 3.1 数据已存在，但消费链路断了

- TraceLens 已经为每个 hot kernel 在磁盘 `kernel_candidates.json` 里写
  `bottleneck`、`arithmetic_intensity`、`recommended_actions` 等字段
  —— 但在最新 main 上**这些字段全是 unknown / null / []**，因为它们
  原本由已被删除的 `pmc_roofline` action 通过
  `merge_roofline_into_candidates()` 注入
- 真正的瓶颈信号在 **`analysis.md` 文本里**：Executive Summary 段、
  Top Operations 段（含 efficiency %）、Recommendations 段
- 当前 prompt 完全看不到 `analysis.md` 内容；只渲染
  `last_select_kernels: top=[ids] reusable_native=[ids] warnings=[...]`
  一行

### 3.2 PR #237 引入的"基建"其实跟 roofline 正交

| 名词 | 真实形态 | 对 roofline-v2 的影响 |
|---|---|---|
| "per-variant gain ledger" | 复用已有 `params_search.tested[fp].gain_pct`，PR #237 只增强了 prompt 渲染 | 无需挂载；ledger 自然记录 |
| "IR-2 install gate" | 进程外 `install.sh`，不在 orchestrator gate 链 | 不拦截 roofline 产物 |
| "closing-phase report flush" | 已有 `_closing_phase_denial` | 不影响 roofline；closing 后 roofline 自然不会再跑 |

### 3.3 prune 基础设施完整存在

| 组件 | 文件:行 | 状态 |
|---|---|---|
| `IntentType.PRUNE_BRANCH` | `intent_parser.py:47, 112` | payload schema `(family, reason)` |
| `Coordinator._handle_prune_branch` | `coordinator.py:2181-2191` | 写 pruned_families + cancel tasks + bus event |
| `_handle_delegate` 硬拦截 | `coordinator.py:1789` | 检查 pruned_families |
| `SharedState.pruned_families` 字段 | `shared_state.py:167` | 已渲染入 prompt（line 1614） |
| `select_kernels` handler 暴露 `trace_report_path` | `kernel_request_handlers.py:651-658` | 即 analysis.md 路径 |

**之前缺的只有**：(a) Orchestration 没有 PRUNE_BRANCH 权限；(b) prompt 不渲染 analysis.md；(c) 没有结构化的 roofline 决策中间件。

### 3.4 Qwen3-32B 真实 trace 形态（来自 transcript `6a95150e-1ac0-4ed2-8c02-3ad4cc77a661`）

- Case A/B/C/D（formal_tp8_1024_1024_c64）：Idle 48-60%，analysis.md
  约 180-210 KB
- 短 trace（issue203 / qwen3-30b-a3b）：Idle <20%，analysis.md 约 10-15 KB
- `cumulative_gain_validated_pct` 在 main HEAD 上 ≈ 0
- `hot_kernels[*].bottleneck` 全为 `unknown`

---

## 4. 设计原则

1. **TraceLens 报告作为 LLM-ready artifact 直接消费，零二次解读**：
   `analysis.md` 是 TraceLens 团队为 LLM/工程师写好的人类可读报告，
   任何 sub-agent / parser / heuristic 在它和主 LLM 之间插一层都是
   信息有损。主 Orchestration LLM 直接读全文，靠它自己的推理能力
   把"瓶颈 → flag → action"链条走完。
2. **`roofline` 是复合 action（macro / pipeline），不是 LLM 解读层**：
   它的 executor 只做编排 `profile → trace_analyze` 这两步原子动作，
   产出 cached snapshot；不调任何 LLM、不写结构化 RooflineAnalysis dict、
   不替主 LLM 做任何决策。
3. **数据驱动、零硬编码 model**：所有判断基于 trace 实际瓶颈分布；
   不引入 model-name / family-name 专用 flag grid。
4. **完整 flag 可见性**：分层渲染 `discovered_flags`（按 action 类型 ×
   tested 状态），让主 LLM 在自己上下文里完成"看到 analysis.md →
   挑没试过的、跟瓶颈匹配的 flag → emit propose"决策链，**杜绝幻觉
   flag 名**（当前 v0 主 LLM 只看到统计行的根本缺陷）。
5. **降级安全**：每个新模块在 unknown / 缺失 / 失败时退化为现状；
   `roofline` executor 中任一子步骤失败 → task 失败但不污染 SharedState。
6. **可验证**：每个 commit 都要有 prompt diff / 数字 / fixture 证据，
   每个 PR 都要在真实 GPU 跑出 +5%（不只是"基建可见"）；prompt
   caching hit rate 通过 `ResultMessage.usage` 度量并暴露到 audit。
7. **小步快走 + 文档先行**：每 commit ≤5 主文件、单测 +20-100 行、
   零回归；任何方案改动必须先修本文档再改代码（见 v1 → v2 教训）。

---

## 5. 完整数据流（v2.0 — roofline=复合 action，analysis.md 全文注入）

```
T0  baseline                                                  [必跑，不变]

T1  ★ roofline #1（主 LLM 主动 PROPOSE_ACTION）            ← 复合 action
    └─ RooflineExecutor 顺序执行：
       (a) profile (复用现有 ProfileExecutor)
              ↓ 产出 last_profile_trace
       (b) trace_analyze (rename 自 select_kernels)
              ↓ 调 kernel-agent/tracelens_analysis.py
              ↓ TraceLens 内部跑 trace_split → kernel_candidates →
                 analysis.md → summary.json
              ↓ 缓存 last_select_kernels {
                   analysis_md_path, analysis_md_text (全文，B2),
                   roofline_snapshot_id=N,
                   roofline_baseline_gain_at_snapshot=<当前 gain>%, ...
                 }
       ※ executor 不调任何 LLM；任一步失败 → task 失败，cache 不写

T2  后续每个 tick 的主 LLM prompt（结构按 prompt caching 优化）:

    [SECTION-A: stable prefix — Claude Code automatic-cached]
      <system_prompt: orchestration.md + 静态 SharedState fields>
      <discovered_flags 分层渲染（Z 方案）>:
        sglang.backends (42 flags):
          --enable-two-batch-overlap            [untested]
          --enable-aiter-allreduce-fusion       [tested: +1.2%]
          --moe-a2a-backend                     [tested: -0.5%]
          ...
        sglang.params (58 flags):
          --cuda-graph-max-bs                   [tested 8 vars, best +0.8%]
          --enable-torch-compile                [untested]
          ...

    [SECTION-B: snapshot-stable — Claude Code automatic-cached within snapshot]
      === TraceLens Analysis (snapshot #N, gain at snapshot = X.XX%) ===
      <analysis.md 全文 verbatim — TraceLens 团队反馈的"正确用法"，
       LLM 在自己上下文里直接读 Executive Summary / Top Operations /
       Recommendations 段，零中间解读层>

    [SECTION-C: per-tick — not cached]
      <变化的 SharedState fields: optimization_stack, params_search.tested,
       attempts_history, ...>
      <Roofline-driven decisions guidance (orchestration.md 内静态段)：
        - 看 analysis.md 后如何决策的指南
        - 何时 emit PRUNE_BRANCH（基于报告 + 已试过的 attempts）
        - 何时 propose params/backends 带 analysis.md Recommendations
          提到的具体 flag（cross-check with discovered_flags）
        - 何时 propose roofline 再来一次（cumulative_gain 跳 +3% 等）>

T3  主 LLM 在自己上下文里直接整合：
    analysis.md (SECTION-B) + discovered_flags (SECTION-A) +
    tested 历史 (SECTION-C) → 决策：
    ├─ emit PRUNE_BRANCH(kernel_opt)            ← C3 已开权限
    ├─ emit PRUNE_BRANCH(deep_kernel_analysis)
    └─ emit PROPOSE_ACTION{
           kind=backends, grid=[{name='roofline_advised',
             extra_sglang_args='--enable-two-batch-overlap',
             ...}, ...]
       }

T4  优化循环（backends / params / kernel_opt / comm_optimization 等）
    被 prune 的 family 在 _handle_delegate 现有路径硬拦截
    LLM 集中试 analysis.md 提到的 flag 方向
    cumulative_gain_validated_pct 累积上升 (0% → 1.2% → 2.5% → 3.2%)

T5  ★ Re-Profile 触发：cumulative_gain - snapshot_baseline ≥ +3%
    主 LLM 看到 SECTION-C guidance 段 → emit PROPOSE_ACTION(roofline)
                                            (注意是 roofline 不是 profile，
                                             因为 v2.0 是复合 action)

T6  ★ roofline #2 — 同样的 profile + trace_analyze 编排
    └─ snapshot_id 自增到 2
       SECTION-B 缓存自动失效（analysis.md 内容变了）→ 触发 Claude Code
          重新 cache_creation；SECTION-A 仍 cache hit
       baseline_gain_at_snapshot = 3.2%

T7  主 LLM 看新 SECTION-B（瓶颈可能已转移 e.g. comm → compute）
    → 新一轮 prune + propose

T8  优化循环 2 ... gain → 4.5% → 5.X%

T9  closing phase（_closing_phase_denial 拦截新 action，只跑 report）
```

### 5.1 prompt 结构与 cache 命中

| Section | 内容 | 变化频率 | Cache 命中（Claude Code automatic） |
|---|---|---|---|
| **A** stable prefix | system_prompt + 静态 SharedState + discovered_flags (含 tested 状态) | 跨整个 session（tested 增量更新时变） | 高（数十次/session 命中，每次只算 read price = 10% write） |
| **B** snapshot-stable | analysis.md 全文（snapshot 内不变） | 每次 roofline action 完成时变 (~2-3 次/session) | 高（snapshot 内 ~30-50 ticks 命中） |
| **C** per-tick | 变化的 SharedState (optimization_stack, attempts_history) + guidance 静态段 | 每 tick 变 | 不 cache（按 write price 算，但 size 小 ~5-10 KB） |

**重要**：本 PR 不引入 manual `cache_control` 字段（claude-agent-sdk
0.2.82 不暴露该接口）；依赖 Claude Code CLI 内部的 **automatic prompt
caching**（OOB `task_manager.py:152-153` 证据：`cache_creation_input_tokens`
/ `cache_read_input_tokens` 已在 response 里返回）。本 PR 做的是**优化
prompt 结构最大化 automatic caching 命中率** + **暴露 hit rate 给 audit**。
如果 C7 跑出来 hit rate < 50%，再考虑下个 PR 换 backend 用网关侧的
`x-auto-prompt-caching: true` 方案（Primus-Claw TS `agent-loop.ts:642`
范本）。

### 5.2 频率估算（60 min Qwen3-32B session）

| 阶段 | wall-clock | roofline 调用 | profile/trace_analyze 调用 |
|---|---|---|---|
| baseline | 3-5 min | 0 | 0 |
| **roofline #1**（profile + trace_analyze） | 8-12 min | **1** | profile 1 + trace_analyze 1 |
| 优化循环 1（backends / params / kernel_opt） | 10-15 min | 0 | 0 |
| **roofline #2**（gain ≥ +3% 触发） | 8-12 min | **1** | profile 1 + trace_analyze 1 |
| 优化循环 2 | 10-15 min | 0 | 0 |
| (可选) roofline #3 | 8-12 min | 0-1 | 0-1 |
| closing | ~3 min | 0 | 0 |
| **合计** | **60 min** | **2-3** | **2-3** |

对比 v0（无 roofline action）：profile 通常只跑 1 次，trace_analyze
（select_kernels）也只跑 1 次，且需要 LLM 手动依次 propose 两个 action
（多浪费 1-2 个 tick）。v2.0 复合 action **每次 roofline = 1 个 propose
+ 一次性产出 snapshot**，编排成本压到最低。

---

## 6. 关键架构决策 — v1 sub-agent 失误回顾 + v2 改回直接注入

### 6.1 v1 选 sub-agent 是错的（教训）

v1 文档 §6 推荐 **sub-agent in executor** 方案，理由三条：(a) token 经济
（200KB × 100 tick 太贵）；(b) 结构化 JSON 输出；(c) 跟"roofline 作为
独立 action"语义对齐。基于这个推荐，v1 实施了：

- C2 `RooflineAnalysis` schema（150 行 + 393 行测试）
- C4a/b/c `RooflineExecutor` sub-agent 编排 + Claude backend + JSON 解析
  + sequence_denial gate（~1450 行）
- C5 `_format_roofline_decision` 结构化结论段渲染（~500 行）

**v1 落地后被发现三个根本缺陷**（讨论时间戳 2026-05-19 18:01）：

1. **schema 是凭空设计的，不基于 analysis.md 实际字段**：
   `primary_bottleneck` 6 个枚举 + `bottleneck_distribution` + 启发式
   阈值（>85% saturated / <30% memory_bound）是从 design 目标反推出来的
   猜测，**v1 实施时没打开过任何一份真实 analysis.md**。TraceLens 团队
   早期就反馈过"我们没有正确使用他们的 analysis.md"，v1 的 sub-agent
   schema 不但没纠正这个问题，反而是变本加厉地"我们决定 LLM 应该看到
   analysis.md 的哪些抽象"。
2. **sub-agent prompt 缺关键上下文**：v1 `_compose_analyzer_user_prompt`
   只传 4 段（analysis.md / cumulative_gain / optimization_stack /
   pruned_families），**没传 `discovered_flags`**。结果 sub-agent 推荐
   "try `enable_two_batch_overlap`" 时不知道这个 flag 在 SGLang 当前
   版本是否存在 → **幻觉 flag 名**。同样没传 `params_search.tested` /
   `backends_search.tested` → **推荐已试过的组合**。
3. **token 担忧本身是错判**：v1 拿 "200KB × 100 tick × $3/M token =
   $15/session" 吓退选项 B，**忽略了 Anthropic prompt caching 已经在
   Claude Code SDK 里 automatic 工作**（OOB `task_manager.py:152-153`
   读 `cache_creation_input_tokens` 即证）。真实成本是 1 次 write +
   99 次 read (10% 价) = ~10% × 100 = ~10x write price，**比 sub-agent
   多调一次完整 LLM 的成本反而便宜**。

**最关键 — TraceLens 团队的原话**（用户转述于讨论时间戳 18:06）：

> "他们的反馈就是我们应该**直接**把他们的分析文档给到 orchestrator，
> 咱们之前不也是这样定的，然后加了份缓存吗？"

v1 把 sub-agent 这一层强行插在 TraceLens 报告和主 LLM 之间，**本质上是
在"未授权地代替 LLM 解读 TraceLens"**，违背了 TraceLens 团队对"应该
怎么用 analysis.md"的明确意见。

### 6.2 v2 改回直接注入 — 三个理由

| 理由 | 说明 |
|---|---|
| **TraceLens 团队意见对齐** | analysis.md 是给 LLM/工程师直接读的人类可读报告，任何二次解读都是 fidelity loss。主 Orchestration LLM 直接读全文，靠它自己的推理能力把"瓶颈 → flag → action"链条走完 |
| **Token 担忧是误判** | Claude Code SDK 内部已 automatic caching；snapshot 内 analysis.md cache hit；v2 只需优化 prompt 结构（A/B/C 三段切分）让 cache 命中最大化，不引入新 LLM 调用 |
| **主 LLM 上下文里有完整 framework 知识** | 当 v2 同时落地"分层渲染 discovered_flags"（让 LLM 看到真实 flag 名 + tested 状态），主 LLM 一个上下文里就能完成 "analysis.md → 选 flag → cross-check tested" — sub-agent 因为隔离上下文反而做不到这种 cross-check |

### 6.3 v2 三个候选重审（v1 表格内容已废，重列）

| 维度 | A. sub-agent 解读（v1 已废） | B. 直接注入主 LLM ★v2 采用 | C. heuristic（无 LLM） |
|---|---|---|---|
| 跟 TraceLens 团队意见对齐 | ❌ 违背 | ✅ 完全对齐 | ❌ 违背（解读层换成关键字匹配，仍是二次解读） |
| 主 LLM 看到 framework flag 真名 | ❌ sub-agent 看不到 discovered_flags | ✅ 主 LLM 上下文里直接有 | ❌ heuristic 不用 LLM |
| Token 成本（含 caching） | 中（sub-agent 调用 + 主 LLM 注入结论）| 中（automatic caching 后主 LLM 注入接近免费） | 低（无 LLM） |
| 实施复杂度 | 高（sub-agent 框架 + JSON schema + 错误处理）| 低（prompt 渲染 + 静态指导段） | 中（关键字 lookup + 误分类风险） |
| 与"action 化"语义一致 | 假对齐（action 内部又调 LLM）| ✅ 真对齐（action 是编排，不是 LLM）| ✅ 对齐 |
| 决策可追溯性 | 中（sub-agent 输出 JSON 可 audit） | 高（主 LLM 决策直接对照 analysis.md） | 高（lookup 表确定） |
| 解读质量 | 中（sub-agent 幻觉 flag）| 高（主 LLM 有完整上下文） | 低（关键字匹配漏报） |

### 6.4 RooflineExecutor 的具体形态（v2.0）

| 项 | 设计 |
|---|---|
| **执行体** | 顺序编排 `profile → trace_analyze`（rename 自 `select_kernels`）的纯 Python 协程；不调任何 LLM；不写 RooflineAnalysis dict |
| **profile 子步骤** | 复用现有 `ProfileExecutor`（`profile_executor`），通过 `tasks.create(kind="profile", ...)` 入队并 await 完成 |
| **trace_analyze 子步骤** | 复用现有 `select_kernels` REQUEST handler 的 handler 函数（直接调，不走 SubAgentRunner，避免双层任务），跑完写 last_select_kernels 缓存 |
| **Atomicity** | profile 失败 → roofline task 失败，cache 不写；profile 成功但 trace_analyze 失败 → roofline task 失败，**但 last_profile_trace 保留**（profile 产物本身有价值，下次 roofline 可以跳过 profile 直接 trace_analyze——v2.1 优化，本 PR 不做） |
| **Idempotency** | task idempotency_key = `f"roofline:{baseline_gain_at_propose}"`；主 LLM 在同一 gain 区间重复 propose 直接复用之前的 task |
| **超时** | profile + trace_analyze 各自的现有 timeout；roofline 整体没有额外 timeout |
| **错误处理** | 子步骤失败 → roofline.status=failed + error_class 透传；不写 fallback cache（v1 的 fallback 设计被废） |

### 6.5 sequence_denial 集成（v2.0）

`roofline` 自身没有前置（baseline 完成后即可 propose）；但 `roofline`
是 **4 个优化 action 的前置依赖**：

- `backends` / `params` / `kernel_opt` / `comm_optimization` 必须有
  `last_select_kernels.analysis_md_text` 非空（即 roofline 至少跑过 1
  次），否则 sequence_denial 拒绝 propose。
- `sweep` / `validate_stack` / `report` / `integrate` 不要求 roofline
  前置（它们不依赖瓶颈分析）。

不在 closing_phase；closing 后所有 action 走现有 `_closing_phase_denial`
拦截。

---

## 7. 模块 / Commit 分解（v2.0 实施跟踪表）

### 7.1 v1 commit 处理（revert / 保留）

按 §6.1 教训，v1 sub-agent 方向的所有 commit 必须 `git revert`；
analysis.md 缓存 (C1) 和 Orchestration PRUNE_BRANCH 权限 (C3) 保留。

| v1 Commit | hash | 状态 v2.0 | 处理 |
|---|---|---|---|
| **C1** SharedState 缓存 analysis.md | `23cb52b` | ✅ 保留 | v2 直接复用（这是 TraceLens 团队反馈"加一份缓存"的正确实现） |
| **C2** `RooflineAnalysis` schema | `9683ea0` | ❌ revert | sub-agent 已废，不需要结构化产物 |
| **C3** Orchestration PRUNE_BRANCH 权限 | `15804ce` | ✅ 保留 | 主 LLM 看完 analysis.md 仍要 emit PRUNE_BRANCH |
| **doc v1** | `92a1a5c` | ⚠️ 不 revert | 历史记录；v2 本文档已大改覆盖 |
| **doc v1.1** | `a0bce0f` | ⚠️ 不 revert | 同上 |
| **C4a** action 注册 + stub | `05febe3` | ❌ revert | v2 RooflineExecutor 完全不同 |
| **C4b** sub-agent LLM executor | `96ea5f1` | ❌ revert | sub-agent 方向已废 |
| **C4c** coordinator + sequence_denial | `dcc4ce3` | ❌ revert (部分) | record_roofline_analysis 调用、roofline-specific gate 已废；但 "roofline 加入 sequence_actions" 是对的（v2 复合 action 也需要） |
| **C5** `_format_roofline_decision` 渲染 + orchestration.md 指导段 | `a4cb7ae` | ❌ revert (部分) | _format_roofline_decision 整段废；orchestration.md 的 PRUNE_BRANCH "You CAN" 修正保留 |
| **C6** verify + audit 脚本 | `4dc246e` | ⚠️ 部分保留 | verify_roofline_v2.py 核心逻辑（gain/wall_clock/action_seq 对比）保留；audit_roofline_decisions.py 里 "advice_consumed_count" 等 sub-agent-specific 字段废，改为统计 cache hit rate + LLM 决策对照 analysis.md 关键字 |

revert 顺序：从最新往最旧（C6 → C5 → C4c → C4b → C4a → C2），每个用
`git revert <hash> --no-edit`，commit message 清晰标注是"v2.0 回退
v1 sub-agent 实施"。

### 7.2 v2.0 新增 commit 计划

| Commit | 内容 | 文件数 | 行数 | 状态 |
|---|---|---|---|---|
| **D1** revert sequence | git revert C6/C5/C4c/C4b/C4a/C2（共 6 个 revert commits） | 0 净代码 | -3400 (回吐之前的实施) | 📝 待执行 |
| **N1** `select_kernels → trace_analyze` rename | 跨 inference_optimizer + kernel-agent 两 repo 改名（action_registry / handler / scoring / cli / tests / kernel-agent tool 入口）；不改语义 | ~15-20 文件（机械替换） | ~50 净改动 | 📝 待执行 |
| **N2** RooflineExecutor 复合 action | (a) `actions/_meta/roofline.yaml`（family=analysis, prerequisites=[baseline]）；(b) `actions/roofline.md` playbook（说明复合 action 语义）；(c) `orchestrator/action_executors/roofline.py` 顺序编排 profile + trace_analyze 协程；(d) `cli.py` 注册；(e) scoring prior=7.5；(f) 单测 | 6 新 + 2 改 | ~250 | 📝 待执行 |
| **N3** Coordinator wire + sequence_denial | (a) `_promote_to_shared_state` 不需要新分支（profile + trace_analyze 子任务的 promote 各自走原路径）；(b) `_sequence_denial_for_action` 加 4 个优化 action 对 analysis.md_text 的依赖检查；(c) 单测 | 1 改 + 1 测试 | ~150 | 📝 待执行 |
| **N4** 分层 flags 渲染 (Z 方案) | (a) `shared_state.py` `_format_discovered_flags` 重写按 `<framework>.<action>` 分组 × tested 状态标记；(b) 单测覆盖典型 / 空 / 大量 flag 场景 | 1 改 + 1 测试 | ~200 |  📝 待执行 |
| **N5** analysis.md 全文注入 + orchestration prompt 指导段 | (a) `shared_state.py` `_format_analysis_md_full()` 渲染整段；(b) `to_prompt_summary` 加入；(c) `orchestration.md` 写 "How to consume analysis.md + discovered_flags + tested" 指导段（取代 C5 的 _format_roofline_decision）；(d) 单测 | 2 改 + 1 测试 | ~250 | 📝 待执行 |
| **N6** prompt 三段切分 + cache hit rate 度量 | (a) `claude.py` ClaudeBackend 改造 prompt 组装为 stable / snapshot-stable / per-tick 三段，让 Claude Code automatic caching 命中最大化；(b) 从 `ResultMessage.usage` 提取 `cache_creation_input_tokens` / `cache_read_input_tokens` 写入 SharedState audit 字段；(c) 单测 mock ResultMessage 验证 metric 流程 | 2 改 + 1 测试 | ~200 | 📝 待执行 |
| **N7** verify + audit 脚本更新 | (a) `verify_roofline_v2.py` 加 cache hit rate 列 + roofline action 调用次数（替代 sub-agent advice 统计）；(b) `audit_roofline_decisions.py` 改为 "decision audit"：统计 LLM PRUNE_BRANCH 是否引用了 analysis.md 段、propose 的 flag 是否在 discovered_flags 中、cache hit rate；(c) 测试 fixture 更新 | 2 改 + 1 测试 | ~150 | 📝 待执行 |
| **N8** Qwen3-32B 真实 GPU 跑 + RATIONALE | C7 等价；填充本文档 §13 | 0 代码 | 0 + 文档追加 | 📝 待 GPU |

**N1-N7 累计估算**：~1250 行新代码（含 ~600 行测试 + ~50 行 yaml/md +
~150 行脚本），核心 Python ~450 行；外加 N1 rename 的 ~50 行机械改动。
**比 v1 实施（~3400 行）少 ~60%**。

### 7.3 sub-commit 拆分原则

每个 N* commit 必须 ≤5 主代码文件、单测独立可跑、零回归（pre-existing
`test_action_scoring` 失败除外）。N2 / N5 / N6 可能触及 5 文件边缘，
必要时进一步拆为 a/b 子 commit（按 N4 已实践的模式）。

---

## 8. C4 实施细化（按文档定义代码边界）

### 8.1 新增文件清单

| 路径 | 用途 | 行数 |
|---|---|---|
| `inference_optimizer/actions/_meta/roofline.yaml` | ActionMetadata（family=`analysis`, prerequisites=`[select_kernels]`, prior 7.5, sub_agent backend hint） | ~25 |
| `inference_optimizer/actions/roofline.md` | 主 LLM 看到的 action playbook（说明何时 propose、产物含义） | ~60 |
| `inference_optimizer/orchestrator/action_executors/roofline.py` | `RooflineExecutor`：spawn sub-agent backend → analyze → return result | ~200 |
| `inference_optimizer/orchestrator/system_prompts/roofline_analyzer.md` | sub-agent system prompt：读 analysis.md → 输出 JSON | ~80 |
| `inference_optimizer/tests/test_roofline_action_executor.py` | mock backend fixture → executor → result 形状校验 | ~180 |
| `inference_optimizer/tests/test_roofline_sequence_denial.py` | propose roofline 不满足前置 → Coordinator 拒绝 | ~100 |

### 8.2 修改文件清单

| 路径 | 修改 | 行数 |
|---|---|---|
| `inference_optimizer/cli.py` | `_REAL_EXECUTORS_FULL` 加 `"roofline": roofline_executor` | +1-3 |
| `inference_optimizer/orchestrator/scoring.py` | `MODEL_CLASS_ACTION_PRIORS` 每个 model_class 加 `"roofline": 7.5`（D1 决策） | +N 行（每个 dict 一条） |
| `inference_optimizer/orchestrator/coordinator.py` | (a) `_handle_request_response` 在 kind=roofline 时调 `record_roofline_analysis`；(b) `_sequence_denial_for_action` 加 roofline 前置检查 | +30 |

**主文件数 = 9（6 新 + 3 改）**。按"超过 5 文件要分解"红线，C4 在
§7 实施跟踪表里拆为 **C4a + C4b + C4c** 三个 sub-commit，每个 sub-commit
≤5 文件、独立可测试、独立可回滚：

- **C4a**（4 文件）：meta yaml + markdown + scoring + cli 注册 stub executor
- **C4b**（3 文件）：替换 stub 为完整 sub-agent executor + analyzer sp + 测试
- **C4c**（2 文件）：coordinator task_kind 分支 + sequence_denial + 测试

C4a 完成后 action 已被识别（propose_action 不会被拒绝为 unknown），但
executor 仅返回 `primary="unknown"` 安全 fallback；C4b 提供真正的 LLM
分析；C4c 把执行结果 wire 回 SharedState 并加门禁。三步串联后才形成
完整闭环。

### 8.3 sub-agent prompt 模板（roofline_analyzer.md 草案）

```
# Roofline Analyzer Sub-Agent

You are a roofline analyzer sub-agent for Hyperloom. Your only job is to
read a TraceLens `analysis.md` report and output a structured JSON
decision so the main Orchestration LLM can prune useless action
families and focus on high-ceiling actions.

## Input format

You will receive in the user message:

- `analysis_md`: full text of the TraceLens report (Executive Summary,
  Top Operations, Recommendations, etc.)
- `cumulative_gain_validated_pct`: current gain since baseline
- `optimization_stack`: list of already-promoted variants
- `pruned_families`: action families already pruned (do NOT recommend
  pruning these again; do NOT recommend actions in these families)

## Output format

Respond with a single JSON object exactly matching this schema (no
prose, no markdown fences):

{
  "primary_bottleneck": "comm" | "compute" | "memory" | "latency" | "idle" | "unknown",
  "bottleneck_distribution": {"comm": float, "compute": float,
                              "memory": float, "latency": float,
                              "idle": float},
  "suggested_prunes": [
    {"family": "<action_family_name>",
     "reason": "<short justification grounded in analysis.md>",
     "confidence": "high" | "medium" | "low"}
  ],
  "suggested_next_actions": [
    {"kind": "<action_kind>",
     "rationale": "<short justification>",
     "priority": "high" | "medium" | "low"}
  ],
  "reprofile_recommended": bool,
  "reprofile_reason": "<reason or empty when false>"
}

## Decision guidelines

- Base every recommendation on quotes / numbers from analysis.md
- A family is "saturated" if its dominant kernel's efficiency >85% AND
  there is no reusable_native_kernel in Top Operations
- Prune `kernel_opt` / `deep_kernel_analysis` when compute saturated +
  no reusable native kernel
- Prune `comm_optimization` when comm < 10% AND not in top-3
- Suggest `params` with specific flag categories that match the primary
  bottleneck (comm → overlap/allreduce flags; latency → graph/compile
  flags; memory → cache/fraction flags; idle → scheduling flags)
- Suggest `reprofile_recommended=true` only when there's a hypothesis
  that the bottleneck distribution has shifted (e.g. gain > 3% since
  last roofline + no new optimization succeeded in last 3 attempts)
- When data is insufficient, prefer "unknown" + empty lists over
  hallucinated recommendations
```

### 8.4 Executor 伪代码

```python
class RooflineExecutor:
    def __init__(self, backend_factory: Callable[[], Backend]):
        # backend_factory lets tests inject a mock backend
        self._make_backend = backend_factory
        self._analyzer_sp = (asset_system_prompts_dir() /
                             "roofline_analyzer.md").read_text()

    async def __call__(self, ctx: RunnerContext) -> dict:
        state = ctx.extra.get("shared_state")  # injected by cli wiring
        cached = state.last_select_kernels or {}
        analysis_md = cached.get("analysis_md_text", "")
        if not analysis_md:
            return {"status": "failed",
                    "error": "no analysis.md cached; run select_kernels first",
                    "degraded": True}

        snapshot_id = cached.get("roofline_snapshot_id", 0)
        # Idempotency: skip when we already analyzed this snapshot
        prev = state.last_roofline_analysis or {}
        if prev.get("snapshot_id") == snapshot_id and snapshot_id > 0:
            return {"status": "succeeded",
                    "snapshot_id": snapshot_id,
                    "idempotency_hit": True,
                    "primary_bottleneck": prev.get("primary_bottleneck"),
                    # ... pass-through to make the result self-describing
                    }

        backend = self._make_backend()
        user_prompt = self._compose_user_prompt(
            analysis_md=analysis_md,
            gain=state.cumulative_gain_validated,
            stack=state.optimization_stack,
            pruned=list(state.pruned_families),
        )
        try:
            turn = await asyncio.wait_for(
                backend.run(prompt=user_prompt,
                            system_prompt=self._analyzer_sp,
                            tools=None, max_turns=1),
                timeout=60.0,
            )
        except (BackendError, asyncio.TimeoutError) as exc:
            return self._fallback_result(snapshot_id, error=repr(exc))

        parsed = self._parse_json_safely(turn.raw_text)
        if parsed is None:
            return self._fallback_result(
                snapshot_id, error="json_parse_failed",
                raw=turn.raw_text)

        parsed["snapshot_id"] = snapshot_id
        parsed["analyzed_at_iso"] = _now_iso()
        parsed["analyzed_at_gain_pct"] = state.cumulative_gain_validated
        parsed["based_on_analysis_md"] = cached.get("analysis_md_path", "")
        parsed["raw_llm_response"] = turn.raw_text
        parsed["status"] = "succeeded"
        return parsed
```

### 8.5 Coordinator 集成

```python
# coordinator.py _handle_request_response (附近 line 2040)
elif kind == "roofline" and status == "ok":
    self.shared_state.record_roofline_analysis(result)
    self.shared_state.save(self.session_dir)
```

注：roofline 是 `PROPOSE_ACTION` 走 SubAgentRunner 路径，不是 REQUEST，
所以集成点可能不是 `_handle_request_response` 而是 task completion
hook。具体集成点在 C4 实现时根据真实 task lifecycle 决定，但语义不变：
**executor 返回 result → SharedState.record_roofline_analysis(result)**。

### 8.6 测试策略

| 测试 | 验证内容 |
|---|---|
| `test_roofline_executor_happy_path` | mock backend 返回 well-formed JSON → result 字段完整、snapshot_id 正确 |
| `test_roofline_executor_idempotency` | 同一 snapshot 第二次调用 → 跳过 LLM call、返回 idempotency_hit |
| `test_roofline_executor_no_analysis_md` | 没有 last_select_kernels → status=failed + degraded |
| `test_roofline_executor_backend_timeout` | mock backend.run 抛 timeout → fallback dict 写入 |
| `test_roofline_executor_malformed_json` | mock 返回非 JSON → fallback dict、raw_llm_response 保留 |
| `test_roofline_executor_schema_validation` | mock 返回 partial JSON → C2 的 record_roofline_analysis 处理缺失字段 |
| `test_roofline_sequence_denial_no_select_kernels` | 没有 select_kernels → policy 拒绝 propose |
| `test_roofline_sequence_denial_closing_phase` | closing 时 → policy 拒绝 |

---

## 9. C5 prompt 渲染设计（提前文档化便于 C4 实施时校对接口）

`shared_state.py` 新增 `_format_roofline_decision(self) -> str`，在
`to_prompt_summary()` 末尾追加（位置紧跟 `_format_last_select_kernels`
之后）。条件渲染：

- `last_roofline_analysis == {}` → 返回 `""`（什么都不加）
- 否则渲染上述 §5 T4 段格式

`prompt_builder.py` 在主 Orchestration system prompt 增加两段静态文本：
"Roofline-driven Pruning Rules" + "Re-Profile Guidance"（§5 T4 已展示），
仅在 `last_roofline_analysis` 非空时显示。

---

## 10. 验证（C7）

### 10.1 对照实验

```bash
# Baseline = main HEAD
cd /wekafs/xiaofei/Hyperloom && git checkout main
nohup hyperloom_opt run \
  --model Qwen3-32B --framework sglang \
  --workload-tp 8 --isl 1024 --osl 1024 --conc 64 \
  --max-minutes 60 \
  --session-dir /tmp/roofline-v2/qwen3-baseline \
  > /tmp/roofline-v2/qwen3-baseline.log 2>&1 &

# Experiment = feature/xiaofei/roofline-v2
cd /wekafs/xiaofei/Hyperloom-roofline-v2
nohup hyperloom_opt run \
  --model Qwen3-32B --framework sglang --workload-tp 8 \
  --isl 1024 --osl 1024 --conc 64 --max-minutes 60 \
  --session-dir /tmp/roofline-v2/qwen3-exp \
  > /tmp/roofline-v2/qwen3-exp.log 2>&1 &

# 验证（C6 脚本）
python scripts/verify_roofline_v2.py \
  --baseline /tmp/roofline-v2/qwen3-baseline \
  --exp /tmp/roofline-v2/qwen3-exp
```

### 10.2 成功标准

| 指标 | 阈值 | 来源 |
|---|---|---|
| `delta cumulative_gain_validated_pct` | **≥ +5%** | 主硬指标 |
| roofline action 实际跑过 ≥ 1 次 | True | session metadata |
| 至少 1 次 PRUNE_BRANCH（source=orchestration） | True | bus event log |
| 至少 1 次 re-profile（gain trigger 之后） | True | session metadata |

### 10.3 通用性验证

```bash
# 同样的对比，model 换 Llama-70B
# 成功标准：delta >= -1% (不劣化)
```

### 10.4 C6 audit 脚本输出表

| 指标 | baseline | exp | delta |
|---|---|---|---|
| `cumulative_gain_validated_pct` | 0.0 | 5.X | +5.X |
| `wall_clock_min` | 60 | 60 | 0 |
| `profile_count` | 1 | 2-3 | +1-2 |
| `roofline_action_count` | 0 | 2-3 | +2-3 |
| `prune_branch_count(source=orchestration)` | 0 | 2-5 | +2-5 |
| `action_seq` | [params, params, kernel_opt, ...] | [params(comm), comm_opt, params(compute), ...] | (路径变化可视化) |

---

## 11. 风险 与 回退

| 风险 | 触发 | 回退（增量加码，不推翻） |
|---|---|---|
| sub-agent JSON 格式不稳定 | C4 跑后 ≥30% fallback rate | (a) prompt 强化 few-shot 示例；(b) 加 JSON repair lib（如 json5）；(c) 极端情况 fallback 到 §6.1 选项 C 的 heuristic（已设计可降级） |
| 主 LLM 看到结论后行为没变 | C7 跑出 delta < +2% | (a) C5 强化 Pruning Rules 语言；(b) `score_mult` 钩子（已有，不改公式）；(c) 极端 fallback：roofline action 直接 enqueue 推荐的 params task（仿 `_maybe_enqueue_pmc_roofline`） |
| sub-agent token 成本超预算 | 实测每 session > $5 | (a) 截断 analysis.md 到 50KB（保留 Executive Summary + Top Ops + Recommendations 三段）；(b) 用更便宜的 LLM（如 Haiku） |
| Qwen3-32B 拿不到 +5% 但 Llama-70B 拿到 | C7 主指标失败但通用性成立 | 反推 Qwen3 trace 是否被 idle gate 清空；考虑 PR #226 idle gate 阈值（已是 80% 默认） |
| re-profile 节奏不对（太频繁/太少） | C7 audit 看到 profile_count = 1 或 = 6 | C5 Re-Profile Guidance 段微调阈值（3% → 5% 或加 wall-clock guard） |
| sub-agent 误剪关键 family | audit 看到某次 prune 之后 LLM 想用该 family 但被拦截 | 当前 PR 不修；下个 PR 引入 UNPRUNE_BRANCH（已知 trade-off） |

---

## 12. 红线 — 不做的事

- ❌ 不写 1306 行 `identify_gaps.py` 等价物（sub-agent + ~200 行 executor + ~80 行 sp 共 ~280 行）
- ❌ 不硬编码任何 model 名 / family
- ❌ 不动 `effective_score` 公式
- ❌ 不引入 PMC 或任何新测量手段
- ❌ 不引入新 SharedState 顶层字段（C2 已遵守：`last_roofline_analysis` 是顶层字段，但**不算新概念**——跟 `last_select_kernels` / `last_kernel_opt` 同级，是"per-action snapshot dict"的标准 pattern）
- ❌ 不动 `closing_phase` / install gate / ledger 数据结构
- ❌ 不引入 UNPRUNE（已知 trade-off，下个 PR）
- ❌ 不让 Coordinator 强制 re-profile（仍由主 LLM 触发，prompt 引导）
- ❌ 不写其他 design 文档（本文档是唯一事实来源；C7 完成后追加一段 §13 数字记录就足够，不另起文件）

---

## 13. C7 实测结果（待 GPU 跑完追加）

_C7 完成后在此追加 baseline / exp 数字、prompt diff 截图引用、audit 输出表。本节占位。_

---

## 14. 变更日志

| 日期 | 改动 | 作者 |
|---|---|---|
| 2026-05-19 | v1 初稿，C1-C3 已实施，C4-C7 待写 | xiaofei + 助手 |
| 2026-05-19 | v1.1 拆 C4 为 C4a/C4b/C4c（每 sub-commit ≤5 文件），调研 ClaudeBackend / scoring / cli 集成点后微调 §8.2 | xiaofei + 助手 |
| 2026-05-19 | **v2.0** 彻底回退 v1 sub-agent 方向（根本原因：违背 TraceLens 团队"直接给 orchestrator"的反馈、sub-agent 缺 discovered_flags 导致幻觉 flag、token 担忧实为误判）。roofline 改复合 action（profile + trace_analyze），analysis.md 全文直接注入，分层 flags 渲染，接入 prompt caching。`select_kernels` rename 为 `trace_analyze`。详见 §6.1 教训、§7.1 commit revert 表、§15 回退执行清单。本次 commit 仅改 §1/§4/§5/§6/§7；§8/§9/§10/§11/§12 + 新增 §15 由后续 doc-v2-C/doc-v2-D commits 完成。 | xiaofei + 助手 |

---

## 15. v2.0 回退执行清单

按 §7.1 表格，本节给出 git revert 的具体顺序 + 校验 + 新 commit 推进
的精确 checklist。**所有 revert 必须先于任何 N* 新代码开始**，确保
worktree 在新代码落地前回到干净的"v1 sub-agent 残骸已移除"状态。

### 15.1 Revert 顺序（从最新 commit 向最旧）

```bash
cd /wekafs/xiaofei/Hyperloom-roofline-v2
git status                  # 确认 worktree clean (设计 doc 改动应已 commit)
git log --oneline -15       # 最新 commit 应是 doc-v2-C，其下是 doc-v2-B/A

# Revert v1 sub-agent 实施 (新→旧)
git revert 4dc246e --no-edit    # C6 verify + audit 脚本 (部分保留，N7 重写)
git revert a4cb7ae --no-edit    # C5 _format_roofline_decision + orchestration.md 段
git revert dcc4ce3 --no-edit    # C4c coordinator 集成 + sequence_denial
git revert 96ea5f1 --no-edit    # C4b sub-agent LLM executor
git revert 05febe3 --no-edit    # C4a action 注册 + stub
git revert 9683ea0 --no-edit    # C2 RooflineAnalysis schema

# 不 revert:
# - 23cb52b (C1 缓存 analysis.md) — v2 复用
# - 15804ce (C3 Orchestration PRUNE_BRANCH) — v2 复用
# - 92a1a5c / a0bce0f (v1 / v1.1 doc commits) — 历史保留
# - 135a1aa / 1fb1cd7 (doc-v2-A/B 当前重写) — v2 在用
```

### 15.2 Revert 后预期状态

| 项 | 验证命令 | 期望 |
|---|---|---|
| revert 总数 | `git log --oneline \| head -15 \| grep -c "Revert"` | 6 |
| 是否还有 sub-agent 痕迹 | `grep -r "sub_agent\|RooflineAnalysis\|roofline_analyzer" inference_optimizer/orchestrator inference_optimizer/actions inference_optimizer/tests` | 0 匹配 |
| C1 / C3 是否仍在 | `git log --oneline 23cb52b 15804ce` | 两 commit 存在 |
| 测试 | `python -m pytest inference_optimizer/tests/test_record_select_kernels_analysis_md.py inference_optimizer/tests/test_orchestration_prune_branch_permission.py -q` | 全过 |
| 整体测试不破 | `python -m pytest inference_optimizer/tests/ -q` | 仅 pre-existing test_action_scoring 失败 |

### 15.3 Revert 后开始 N* 新 commits

按 §7.2 顺序串行执行：N1 (rename) → N2 (RooflineExecutor) →
N3 (Coordinator wire) → N4 (分层 flags) → N5 (analysis.md 注入 +
prompt 指导段) → N6 (prompt 三段切分 + cache metric) → N7 (verify/audit
更新) → N8 (Qwen3-32B GPU + 填本文档 §13)。

每个 N* commit 完成后，commit message 必须引用本文档对应章节
（如 `feat(roofline-v2): RooflineExecutor 复合 action [N2] (design §8.x)`）。

### 15.4 失败回退路径

若 N* 中任一 commit 测试失败或推进困难：

1. 不要往前推 — 当前 N* commit `git reset --hard HEAD~1` 回退
2. 修本文档 §7/§8 对应小节，说明遇到的问题 + 方案微调
3. 重新尝试 N* 实施
4. 若整体方向出问题（如 prompt caching 实测 hit rate < 30%）→ 在本
   §15 加 v2.1 子节，记录方向调整，再继续。**不要在没改本文档前
   动代码**（v1 → v2 的最痛教训）。

### 15.5 不在本 PR 范围（明确划红线）

- `roofline` 复合 action 内部的 profile 失败但 trace_analyze 仍可跑的
  "subset retry" 优化 → v2.1
- UNPRUNE_BRANCH intent（让 LLM 撤销之前的 prune）→ 单独 PR
- 直接调 `anthropic` SDK 绕过 Claude Code（如 cache hit rate < 50%
  才考虑）→ 单独 PR
- 跨 framework （vLLM / TGI）的 flag 分层 yaml taxonomy → 单独 PR
