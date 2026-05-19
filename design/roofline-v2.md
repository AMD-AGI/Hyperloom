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
| **N1** `select_kernels → trace_analyze` rename (跨 inference_optimizer + kernel-agent 同 worktree) | 34 文件机械替换：(a) `actions/_meta/select_kernels.yaml` → `trace_analyze.yaml` rename + 内容更新；(b) `select_kernels_handler` → `trace_analyze_handler`；(c) `record_select_kernels` → `record_trace_analyze`；(d) `last_select_kernels` → `last_trace_analyze`（旧 state.json 多余字段会被现有 load 路径默默忽略，等于 cache miss → 下次 trace_analyze 重做，可接受降级）；(e) request dispatcher key；(f) kernel-agent/tools/tracelens_analysis.py 入口字符串；(g) 所有测试 rename | 34 文件（30 inference_optimizer + 4 kernel-agent） | ~256 净 occurrence | 📝 待执行 |
| **N2** RooflineExecutor 复合 action | (a) `actions/_meta/roofline.yaml`（family=analysis, prerequisites=[baseline]）；(b) `actions/roofline.md` playbook（说明复合 action 语义）；(c) `orchestrator/action_executors/roofline.py` 顺序编排 profile + trace_analyze 协程；(d) `cli.py` 注册；(e) scoring prior=7.5；(f) 单测 | 6 新 + 2 改 | ~250 | 📝 待执行 |
| **N3** Coordinator wire + sequence_denial | (a) `_promote_to_shared_state` 不需要新分支（profile + trace_analyze 子任务的 promote 各自走原路径）；(b) `_sequence_denial_for_action` 加 4 个优化 action 对 analysis.md_text 的依赖检查；(c) 单测 | 1 改 + 1 测试 | ~150 | 📝 待执行 |
| **N4** 分层 flags 渲染 (Z 方案) | (a) `shared_state.py` `_format_discovered_flags` 重写按 `<framework>.<action>` 分组 × tested 状态标记；(b) 单测覆盖典型 / 空 / 大量 flag 场景 | 1 改 + 1 测试 | ~200 |  📝 待执行 |
| **N5** analysis.md 全文注入 + orchestration prompt 指导段 | (a) `shared_state.py` `_format_analysis_md_full()` 渲染整段；(b) `to_prompt_summary` 加入；(c) `orchestration.md` 写 "How to consume analysis.md + discovered_flags + tested" 指导段（取代 C5 的 _format_roofline_decision）；(d) 单测 | 2 改 + 1 测试 | ~250 | 📝 待执行 |
| **N6** cache hit rate 度量（务实简化）| 经 Anthropic prompt-caching 文档复核：automatic caching 已基于 system_prompt + tools + messages prefix 哈希工作，无需应用层"三段切分"。N6 范围缩到：(a) `claude.py` `_invoke_and_collect` 返回 `usage` dict，`run()` 把 `cache_creation_input_tokens` / `cache_read_input_tokens` / `input_tokens` / `output_tokens` 写入 `backend.calls` + `BackendTurnResult.metadata`；(b) 单测覆盖 metric 提取（含 usage 缺失 / 非 dict / SDK 没传 usage 各种降级路径）；(c) 三段切分推到 N6b/下个 PR（实测 cache_hit_rate 若 < 50% 才做） | 1 改 + 1 测试 | ~120 | 📝 待执行 |
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

## 8. N1-N7 实施细化（按文档定义代码边界）

> **v1 §8 全段废弃** — sub-agent 实施细化（roofline_analyzer.md prompt
> 模板、RooflineExecutor sub-agent 伪代码、record_roofline_analysis
> 集成）已 §6.1 整体否决；具体代码会在 §15.1 的 6 个 revert commits
> 里移除。下面是 v2.0 的 N1-N7 实施细化。

### 8.1 N1 — `select_kernels → trace_analyze` rename (跨子目录，同 worktree)

**重要事实**：`kernel-agent/` 跟 `inference_optimizer/` 在同一 git
worktree 下（不是独立 repo），所以 N1 是同一 PR 内的单一 mass rename
commit，**不需要跨 repo 协调**。kernel-agent 侧仅 4 文件 / ~6
occurrence，与 inference_optimizer 一起改。

**目标**：`select_kernels` 这个名字暗示"选 kernel"，但它实际是"调
TraceLens 跑 trace_split + kernel_candidates + 写 analysis.md +
summary.json"。改名为 `trace_analyze` 更准确反映语义。

**实测影响范围**（用 grep 量化，2026-05-19）：

inference_optimizer 内 **30 个文件 / ~250 个 occurrence**：

| 类型 | 文件数 | 关键改动 |
|---|---|---|
| **核心代码（必须语义改）** | 4 | `kernel_request_handlers.py`（handler 函数名 + dispatcher dict key）；`coordinator.py`（`_sequence_denial_for_request` 分支 + `_handle_request_response` kind 比较）；`shared_state.py`（`record_select_kernels` 函数 + `last_select_kernels` dataclass 字段）；`system_prompts/prompt_builder.py`（grid hint 文本提到的 kind） |
| **system prompt（LLM 可见层）** | 1 | `system_prompts/orchestration.md`（提到 `select_kernels` 的描述句） |
| **action meta YAML** | 1 | `actions/_meta/select_kernels.yaml` → `trace_analyze.yaml` rename + name 字段更新 |
| **backend mock / 示例** | 2 | `backends/kernel_mock.py`（mock 路由）；`examples/p*_demo.py`（demo 调用） |
| **breakdown / collectors（旁路）** | 5 | `breakdown/schema.py`、`breakdown/collectors.py`、`breakdown/reporters/_renderers/kernel_lifecycle.py`、`breakdown/SKILL.md`（report 时引用 kind 名） |
| **测试 / fixture** | 17 | `tests/test_p*` 全套测试中提到 select_kernels 的所有断言、fixture、注释 |

**实施方式**：单一 commit (N1a)，机械 grep + replace 所有
`select_kernels` 字面出现 → `trace_analyze`；`SelectKernels` → `TraceAnalyze`
（CamelCase）；测试函数名 `test_select_kernels_*` → `test_trace_analyze_*`；
**不改任何语义**。

**SharedState 字段迁移策略**：`last_select_kernels` → `last_trace_analyze`，
**不保留 alias**。理由：shared_state.py 现有 `load_or_init` 路径已经
通过 dataclass field 过滤未知字段（shared_state.py:356 注释明确说明），
旧 state.json 里的 `last_select_kernels` 会被默默忽略 → 下次
trace_analyze 重做 → 等价 cache miss，可接受降级。这避免引入命名空间
污染，未来不会有人疑惑为什么 SharedState 有两个相似字段。

**已知 pre-existing failures**：N1a 后跑全测试套件需 deselect 2 个
pre-existing failures（`test_action_scoring::test_seed_action_scores_uses_model_class_priors_when_available`、
`test_p1_2_full_action_catalogue::test_every_action_has_non_empty_description`），
这两个都在 D1 之前就存在，与 rename 无关。

**kernel-agent 侧文件清单**（grep 实测）：

| 文件 | occurrence |
|---|---|
| `kernel-agent/tools/tracelens_analysis.py` | 2 |
| `kernel-agent/tools/test_tracelens_csv.py` | 1 |
| `kernel-agent/scripts/install.sh` | 1 |
| `kernel-agent/SKILL.md` | 2 |

### 8.2 N2 — RooflineExecutor 复合 action 新增文件

| 路径 | 用途 | 行数 |
|---|---|---|
| `inference_optimizer/actions/_meta/roofline.yaml` | ActionMetadata：family=`analysis`, prerequisites=`[baseline]`（注意不是 `[select_kernels]`，因为 roofline 自己内部跑 trace_analyze），prior 7.5，cost_minutes_p50=8, p75=15（含 profile + trace_analyze 两步时长） | ~25 |
| `inference_optimizer/actions/roofline.md` | 主 LLM 看到的 action playbook：(a) 复合 action 语义说明（编排 profile + trace_analyze）；(b) 何时 propose（baseline 后 + 每次 cumulative_gain 跳 +3% 后）；(c) 产物（last_profile_trace + last_select_kernels.analysis_md_text + snapshot_id 自增）；(d) 失败语义（任一子步骤失败 → 整体失败，无 fallback） | ~80 |
| `inference_optimizer/orchestrator/action_executors/roofline.py` | `RooflineExecutor` 顺序编排 profile + trace_analyze 的纯 Python 协程；不调任何 LLM；不写 RooflineAnalysis dict | ~180 |
| `inference_optimizer/tests/test_roofline_executor_v2.py` | 测试 fixture mock profile + trace_analyze 子任务的成功/失败/中间状态 | ~180 |

### 8.3 N2 — RooflineExecutor 修改文件清单

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

| 路径 | 修改 | 行数 |
|---|---|---|
| `inference_optimizer/cli.py` | `_REAL_EXECUTORS_FULL` 增加 `"roofline": RooflineExecutor(...)`；新 executor 需要 access 到 coordinator.shared_state（参考 v1 N4 的 closure 模式）+ access 到 SubAgentRunner（用于内部 enqueue profile / trace_analyze 子任务） | +20 |
| `inference_optimizer/orchestrator/scoring.py` | `MODEL_CLASS_ACTION_PRIORS` 四个 model_class 各加一行 `"roofline": 7.5`（D1 决策保留） | +4 |
| `inference_optimizer/actions/_meta/select_kernels.yaml` → `trace_analyze.yaml` | N1 rename 的一部分 | 0 |

### 8.4 RooflineExecutor 伪代码（v2.0）

```python
class RooflineExecutor:
    def __init__(self, *, shared_state, tasks, sub_runner):
        # No backend, no LLM, no JSON parsing — pure orchestration.
        self.shared_state = shared_state
        self.tasks = tasks
        self.sub_runner = sub_runner

    async def __call__(self, ctx: RunnerContext) -> dict:
        # Step 1: profile (reuse existing ProfileExecutor)
        profile_task = await self.tasks.create(
            kind="profile",
            params=self._derive_profile_params(),
            idempotency_key=f"roofline_profile:{ctx.task.task_id}",
        )
        profile_result = await self.sub_runner.run_task(profile_task)
        if profile_result.state != "succeeded":
            return {
                "status": "failed",
                "error_class": "profile_failed",
                "error": profile_result.error or "profile sub-step failed",
                "phase": "profile",
            }

        # Step 2: trace_analyze (reuse select_kernels handler directly)
        # Note: call the handler function, NOT via SubAgentRunner (avoid
        # double-task accounting). trace_analyze writes
        # last_select_kernels cache as side effect.
        from ..kernel_request_handlers import trace_analyze_handler
        ta_result = await trace_analyze_handler(
            payload={"trace_input": self.shared_state.last_profile_trace},
            session_dir=ctx.extra.get("session_dir"),
        )
        if ta_result.get("status") != "ok":
            return {
                "status": "failed",
                "error_class": "trace_analyze_failed",
                "error": ta_result.get("error") or "trace_analyze sub-step failed",
                "phase": "trace_analyze",
            }

        # All success — return summary. The actual data went into
        # last_profile_trace and last_select_kernels via the sub-step
        # promote paths; this result is just status / audit.
        return {
            "status": "succeeded",
            "profile_task_id": profile_task.task_id,
            "snapshot_id": self.shared_state.last_select_kernels.get(
                "roofline_snapshot_id"
            ),
        }
```

### 8.5 N3 — Coordinator integration + sequence_denial

**`_promote_to_shared_state`**：

```python
# Roofline-v2 N3:
# roofline 本身的 task completion 不需要新 promote 分支 —— profile +
# trace_analyze 子步骤已经分别走自己的 promote 路径 (record_profile /
# record_select_kernels)。Roofline 自身的 result 只携带 status +
# snapshot_id，audit-trail (record_action_attempt) 走 _AUDIT_ACTIONS
# 标准流程即可。
elif task_kind == "roofline":
    # No state mutation — sub-steps already wrote last_profile_trace +
    # last_select_kernels. record_action_attempt fires via the standard
    # post-completion path for audit / scoring.
    changed = False
```

**`_sequence_denial_for_action`**：

```python
# Roofline-v2 N3: 4 个优化 action 必须有 fresh roofline snapshot
_ROOFLINE_REQUIRED = {"backends", "params", "kernel_opt", "comm_optimization"}
if action in _ROOFLINE_REQUIRED:
    cached = self.shared_state.last_select_kernels or {}
    if not cached.get("analysis_md_text"):
        return PolicyDenied(
            f"action={action!r} denied: roofline must run first "
            "(no cached TraceLens analysis.md)",
            rule="execution_order",
            hint="propose/delegate `roofline` (a composite action that "
                 "internally runs profile + trace_analyze; do NOT call "
                 "profile / trace_analyze directly)",
        )
```

**`sequence_actions` set 增加 `"roofline"`** —— 走标准 sequence-action
路径（target_analysis / baseline / closing_phase 等 gate 仍然按既有
顺序生效）。

### 8.6 N4 — 分层 `_format_discovered_flags`（Z 方案）

```python
def _format_discovered_flags(self) -> str:
    if not self.discovered_flags:
        return "(none — first backends/params round will populate)"
    out_lines: list[str] = []
    for fw, entry in sorted(self.discovered_flags.items()):
        if not isinstance(entry, dict):
            continue
        for action_kind in ("backends", "params"):
            flags = entry.get(f"{action_kind[:-1]}_flags") or []
            if not flags:
                continue
            out_lines.append(f"  {fw}.{action_kind} ({len(flags)} flags):")
            for flag in sorted(flags):
                # cross-ref params_search.tested / backends_search.tested
                # to mark "tested: +X% / -X% / N variants best +X%"
                tag = self._tested_tag_for_flag(flag, action_kind, fw)
                out_lines.append(f"    {flag:42s} {tag}")
    return "\n" + "\n".join(out_lines)

def _tested_tag_for_flag(self, flag: str, action_kind: str, fw: str) -> str:
    search = self.params_search if action_kind == "params" else self.backends_search
    tested = search.get("tested") or {}
    # Find all fingerprints whose extra_sglang_args contains this flag
    matched_gains: list[float] = []
    for fp_key, snap in tested.items():
        if not isinstance(snap, dict):
            continue
        if flag in str(snap.get("extra_sglang_args") or ""):
            gain = snap.get("gain_pct")
            if isinstance(gain, (int, float)):
                matched_gains.append(float(gain))
    if not matched_gains:
        return "[untested]"
    n = len(matched_gains)
    best = max(matched_gains)
    if n == 1:
        return f"[tested: {best:+.1f}%]"
    return f"[tested {n} vars, best {best:+.1f}%]"
```

主 LLM 看到的 prompt 输出（示例）：

```
discovered_flags=
  sglang.backends (42 flags):
    --enable-two-batch-overlap                 [untested]
    --enable-aiter-allreduce-fusion            [tested: +1.2%]
    --moe-a2a-backend                          [tested: -0.5%]
    ...
  sglang.params (58 flags):
    --cuda-graph-max-bs                        [tested 8 vars, best +0.8%]
    --enable-torch-compile                     [untested]
    --mem-fraction-static                      [tested 3 vars, best +0.1%]
    ...
```

### 8.7 N5 — analysis.md 全文注入 + orchestration prompt 指导段

**shared_state.py 新增**：

```python
def _format_analysis_md_full(self) -> str:
    """Inject TraceLens analysis.md verbatim per TraceLens team
    instruction (v2 §6.2 reason 1). Snapshot-stable; cached by Claude
    Code automatic prompt caching within a snapshot."""
    cached = self.last_select_kernels or {}
    md = cached.get("analysis_md_text") or ""
    if not md:
        return "(no TraceLens snapshot yet — propose `roofline` to produce one)"
    snap = cached.get("roofline_snapshot_id", "?")
    gain = cached.get("roofline_baseline_gain_at_snapshot", 0.0)
    return (
        f"\n=== TraceLens Analysis (snapshot #{snap}, "
        f"gain at snapshot = {gain:.2f}%) ===\n"
        f"{md}\n"
        f"=== End TraceLens Analysis ===\n"
    )
```

`to_prompt_summary` 加入 `f"analysis_md={self._format_analysis_md_full()}"`
（注意：这一段会很大，依赖 Claude Code automatic caching 命中）。

**orchestration.md 新增指导段**（取代 v1 `_format_roofline_decision`）：

```
### How to consume the TraceLens Analysis above

You will see a full `analysis.md` report under the
`analysis_md=...` block on every tick after roofline runs. Read it as
you would read a human-written perf analysis: Executive Summary tells
you the dominant bottleneck, Top Operations gives per-kernel efficiency
numbers, Recommendations explicitly lists what to try next.

When deciding the next action:

1. **PRUNE_BRANCH a family** only when the report directly supports it
   AND you've already tried that family at this snapshot. Example:
   "compute saturated 92%, no reusable_native_kernel in Top Operations"
   → emit `prune_branch{family='kernel_opt', reason='analysis.md says
   compute saturated 92%, no reusable_native'}` ONLY if at least one
   `kernel_opt` attempt already failed since the snapshot was taken.

2. **PROPOSE backends / params with specific flags** by cross-checking
   `discovered_flags` (rendered above as
   `sglang.backends (N flags): --flag-1 [untested], --flag-2 [tested: +X%], ...`):
   - Pick a flag whose name matches the report's bottleneck (comm
     bottleneck → `--enable-two-batch-overlap` / `--enable-aiter-allreduce-fusion`;
     latency → `--cuda-graph-max-bs`; compute → `--enable-torch-compile`).
   - PREFER `[untested]` flags over previously tried ones.
   - Construct `params.grid=[{name, extra_sglang_args, ...}]` explicitly;
     do NOT rely on the executor's default grid alone — it covers only
     ~30% of the flag namespace.

3. **PROPOSE roofline again** when ANY of:
   - cumulative_gain_validated_pct has increased ≥ 3% since the
     snapshot was taken (bottleneck distribution likely shifted)
   - all non-pruned families have been tried at this snapshot with no
     new gain in the last 3 attempts
   - the report itself notes "data may be stale" / similar
```

### 8.8 N6 — prompt 三段切分 + cache hit metric

**目标**：让 Claude Code automatic prompt caching 把"稳定 prefix"做大，
让 cache hit rate 接近上限；同时把 cache metric 从 ResultMessage.usage
提取出来暴露给 audit。

**ClaudeBackend `_invoke_and_collect` 改动**：

```python
async def _invoke_and_collect(
    self, prompt: str, options: Any
) -> tuple[list[Intent], str, int, dict]:  # 返回值加 usage dict
    intents: list[Intent] = []
    text_chunks: list[str] = []
    tool_block_count = 0
    last_usage: dict[str, Any] = {}
    async for message in self.sdk_query_factory(prompt=prompt, options=options):
        for block in self._iter_blocks(message):
            # ... existing logic ...
        # NEW: extract usage from ResultMessage
        msg_usage = getattr(message, "usage", None)
        if isinstance(msg_usage, dict):
            last_usage = msg_usage
    return intents, "".join(text_chunks), tool_block_count, last_usage
```

`BackendTurnResult.metadata` 加 `cache_creation_input_tokens` /
`cache_read_input_tokens` 字段；Coordinator 在 tick post-processing
里累加到 SharedState 新字段 `tick_cache_metrics` 供 audit 脚本统计。

**prompt 结构改动**（在 Coordinator `_compose_prompt` 里）：

```python
def _compose_prompt(self, role: str) -> str:
    # SECTION-A (cache-target: stable prefix)
    section_a = "\n".join([
        self._load_system_prompt(role),
        self.shared_state.format_static_section(),  # baseline_tput / model /
                                                      # baseline_acc / discovered_flags
                                                      # (含 tested 状态)
    ])
    # SECTION-B (cache-target: snapshot-stable)
    section_b = self.shared_state._format_analysis_md_full()
    # SECTION-C (per-tick)
    section_c = self.shared_state.format_dynamic_section()  # optimization_stack /
                                                              # attempts_history /
                                                              # last_action_failures / ...
    return f"{section_a}\n\n{section_b}\n\n{section_c}"
```

注：Claude Code automatic caching 根据 prefix 哈希命中，所以 A 段必须
在 B 段之前、B 必须在 C 之前；这样 A 命中（session 级别）→ B 也命中
（snapshot 级别）→ 只有 C 需要重新计算。

### 8.9 N7 — verify / audit 脚本更新

**`verify_roofline_v2.py`**（在 v1 C6 基础上增加列）：

| 指标 | 来源 |
|---|---|
| `cumulative_gain_validated_pct` baseline vs exp | state.json |
| `wall_clock_min` baseline vs exp | session metadata |
| `roofline_action_count` baseline vs exp | task_attempts (`roofline` kind) |
| **`cache_hit_rate`** baseline vs exp | sum(cache_read_tokens) / sum(cache_read_tokens + cache_creation_tokens) per session |
| `prune_branch_count(source=orchestration)` baseline vs exp | bus event log |
| `tested_flag_count` exp | `params_search.tested` + `backends_search.tested` |
| `analysis_md_referenced_count` exp | 主 LLM emit 的 PRUNE_BRANCH reason 字段或 propose notes 字段含 analysis.md 关键短语（如 "saturated"/"comm-bound"/"efficiency"）的次数 |

**`audit_roofline_decisions.py`**（改为 "decision audit"）：

| 维度 | 内容 |
|---|---|
| roofline 调用次数与时机 | 每次的 task_id / 时间戳 / 触发时 cumulative_gain |
| Cache hit rate 趋势 | 每个 tick 的 cache_creation vs cache_read 比例 |
| LLM 决策对照 | PRUNE_BRANCH 的 reason 是否引用 analysis.md 实际句段；propose 的 flag 是否在 discovered_flags 列表里 + 是否标记为 untested |
| 失败模式 | 主 LLM propose 了 discovered_flags 之外的 flag 名（幻觉） / propose 了已 tested 失败的组合（不读 tested 状态） / 没在 +3% 后 re-propose roofline（不读 guidance） |

### 8.10 整体文件清单（v2.0 实施合计）

| Phase | 新增文件 | 修改文件 | 测试新增 |
|---|---|---|---|
| **D1** (revert) | 0 | 0 (revert 操作) | 0 |
| **N1** (rename) | 0 (.yaml rename) | ~15 (跨 repo 机械替换) | 0 (现有测试也 rename) |
| **N2** (RooflineExecutor) | 4 (yaml + md + .py + 测试) | 0 | 1 |
| **N3** (Coordinator) | 0 | 1 (coordinator.py) | 1 |
| **N4** (分层 flags) | 0 | 1 (shared_state.py) | 1 |
| **N5** (analysis.md 注入 + 指导段) | 0 | 2 (shared_state.py + orchestration.md) | 1 |
| **N6** (cache metric) | 0 | 2 (claude.py + coordinator.py) | 1 |
| **N7** (verify/audit) | 0 | 2 (scripts) | 1 |
| **N8** (GPU + 文档 §13) | 0 | 1 (本文档) | 0 |
| **合计** | **4 新** | **~24 改**（N1 占大头） | **6 测试** |

---

## 9. prompt 渲染设计（v2.0 — 直接注入 analysis.md + 分层 flags）

v1 §9 全段废弃。v2.0 prompt 渲染的具体接口已在 §8.6 / §8.7 / §8.8
详细列出（分层 flags / analysis.md 全文注入 / 三段切分 + cache metric）。
本节只补充三段切分的**位置约束**和**渲染顺序**：

### 9.1 渲染顺序（cache 友好）

```
[SECTION-A — stable prefix, session 级 cache target]
  <orchestration.md system_prompt 全文>
  <SharedState 静态段>:
    session_id / model / model_class / baseline_tput / baseline_acc /
    framework / gpu_type / target_gap_pct / ...
  <discovered_flags 分层渲染 (Z 方案，§8.6)>

[SECTION-B — snapshot 级 cache target]
  <analysis_md full content (§8.7 _format_analysis_md_full)>

[SECTION-C — per-tick, not cached]
  <SharedState 动态段>:
    current_best / optimization_stack / cumulative_gain* /
    params_search / backends_search / attempts_history /
    last_action_failures / pruned_families / tick / closing_phase / ...
  <Roofline-driven decisions guidance (orchestration.md 内静态段
   但 logically per-tick 因为引用 SECTION-A/B 内容)>
```

### 9.2 渲染顺序约束的硬要求

* SECTION-A 必须**完全独立于** roofline / TraceLens 状态（否则
  roofline #1 完成后会破坏 A 段 cache，整 session 命中率崩盘）
* SECTION-B 内 analysis.md 是**逐字符**注入（不 reshape / 不截断
  / 不加 markdown wrapper），以保证同 snapshot 内文本完全一致 →
  cache 命中
* SECTION-C 内的所有 SharedState 渲染必须**不影响** A / B 段；
  特别是 `discovered_flags` 留在 A 段（虽然 tested 状态会变，但
  整体 framework flag 列表稳定，cache miss 频率可接受）

### 9.3 渲染失败降级

* `analysis_md_text == ""` → SECTION-B 渲染 `(no TraceLens snapshot
  yet — propose roofline to produce one)`，提示主 LLM 下一步该
  propose roofline
* discovered_flags 字段缺失 → SECTION-A 渲染 `(none — first
  backends/params round will populate)`，retain v0 兼容文案

---

## 10. 验证（N8）

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

# 验证（N7 脚本）
python -m inference_optimizer.scripts.verify_roofline_v2 \
  --baseline /tmp/roofline-v2/qwen3-baseline \
  --exp /tmp/roofline-v2/qwen3-exp

# 决策审计（N7 脚本）
python -m inference_optimizer.scripts.audit_roofline_decisions \
  --session /tmp/roofline-v2/qwen3-exp
```

### 10.2 主成功标准

| 指标 | 阈值 | 来源 |
|---|---|---|
| `delta cumulative_gain_validated_pct` | **≥ +5%** | 主硬指标 |
| roofline action 实际跑过 ≥ 1 次 | True | task_attempts (kind=roofline) |
| 至少 1 次 PRUNE_BRANCH（source=orchestration） | True | bus event log |
| 至少 1 次 roofline re-propose（gain trigger 之后） | True | task_attempts 时间序列 |

### 10.3 v2 新增 cache 与决策质量标准

| 指标 | 阈值 | 解释 |
|---|---|---|
| `cache_hit_rate` (exp session) | **≥ 50%** | sum(cache_read) / (sum(cache_read) + sum(cache_creation))；低于阈值说明 prompt 结构没生效，需进一步优化或考虑下个 PR 换 backend |
| `analysis_md_referenced_count` (exp) | **≥ 3** | LLM 的 PRUNE_BRANCH reason / propose notes 引用 analysis.md 关键短语次数；为 0 说明 LLM 在忽略 analysis.md |
| `hallucinated_flag_count` (exp) | **= 0** | propose 的 flag 不在 discovered_flags 里的次数；> 0 说明分层 flags 渲染没生效 |
| `retested_flag_count` (exp) | **≤ 2** | propose 已 tested 且失败的 flag 组合的次数；表示 LLM 不读 tested 状态 |

### 10.4 通用性验证

```bash
# 同样的对比，model 换 Llama-70B
for model in DeepSeek-R1-0528 Llama-3.1-70B-Instruct; do
  for branch in main feature/xiaofei/roofline-v2; do
    # ... 同 §10.1 的命令模板
  done
done
# 成功标准：每个 model 上 delta >= -1% (不劣化)；
# 至少有 1 个 model 上 delta >= +5% (v2 不要求每个 model 都到 +5%)
```

### 10.5 verify / audit 输出表（N7 渲染）

| 指标 | baseline | exp | delta |
|---|---|---|---|
| `cumulative_gain_validated_pct` | 0.0 | 5.X | +5.X |
| `wall_clock_min` | 60 | 60 | 0 |
| `roofline_action_count` | 0 | 2-3 | +2-3 |
| `cache_hit_rate` | 30%-50% (Claude Code 默认) | ≥ 50% (v2 优化结构) | +X% |
| `prune_branch_count(source=orchestration)` | 0 | 2-5 | +2-5 |
| `analysis_md_referenced_count` | 0 | 3-10 | +3-10 |
| `hallucinated_flag_count` | unknown (v0 没度量) | 0 (v2 目标) | -X |
| `action_seq` | [params, params, kernel_opt, ...] | [roofline, backends(comm-overlap), comm_opt, roofline, params(compute), ...] | (路径变化可视化) |

---

## 11. 风险 与 回退（v2.0 更新）

| 风险 | 触发 | 回退（增量加码，不推翻 v2 方向） |
|---|---|---|
| automatic prompt caching hit rate < 50% | N8 audit 看到 cache_hit_rate 偏低 | (a) 检查 prompt 结构是否真的 A→B→C 顺序、是否有不稳定字段污染 SECTION-A；(b) 微调 §8.6 _format_discovered_flags 的 tested 状态更新频率（如改为只在 roofline 后刷新而非每 tick）；(c) 仍不行 → 下个 PR 用网关侧 `x-auto-prompt-caching: true` 方案 (Primus-Claw TS agent-loop.ts:642 范本) |
| 主 LLM 看到 analysis.md 后行为没变 | N8 跑出 delta < +2% AND analysis_md_referenced_count < 3 | (a) §8.7 orchestration.md 指导段强化 few-shot 示例（"如果 analysis.md 说 X，emit PRUNE_BRANCH{reason='...analysis.md X段...'}"）；(b) `score_mult` 钩子按 bottleneck 临时提升（已有钩子，不改公式形状）；(c) 极端 fallback：roofline executor 在 trace_analyze 后**自动 enqueue 1 个 params task**带 analysis.md 推荐的 flag (仿现有 `_maybe_enqueue_pmc_roofline` 模式) |
| 主 LLM 幻觉 flag 名 | N8 audit 看到 hallucinated_flag_count > 0 | (a) 检查分层 flags 渲染是否真的进了 prompt、SECTION-A 是否被截断；(b) orchestration.md 指导段加 hard rule "ONLY use flags listed under discovered_flags above; rejected with policy_denied otherwise"；(c) 在 BackendsExecutor / ParamsExecutor 加 strict validation：grid 中含未 discovered 的 flag → 失败并 surface 给 LLM |
| Qwen3-32B 拿不到 +5% 但 Llama-70B 拿到 | N8 主指标失败但通用性成立 | 反推 Qwen3 trace 是否被 idle gate 清空 (PR #226 idle gate 阈值已是 80% 默认)；如确认是 idle 主导 → 在 §8.6 / §8.7 加 idle-specific 引导 (建议 propose `cuda_graph_*` 类 flag) |
| re-profile 节奏不对 | N8 audit 看到 roofline_action_count = 1 或 = 6 | §8.7 orchestration.md re-profile guidance 段微调阈值（3% → 5% 或加 wall-clock guard） |
| roofline 复合 action 子步骤失败率高 | N8 audit 看到 roofline.status=failed 多次 | (a) 区分是 profile 失败还是 trace_analyze 失败，定位 root cause；(b) v2.1 加 subset retry（profile 已成功时直接重跑 trace_analyze） |
| sub-step 同时跑导致 lane 冲突 | N8 跑挂或 resource lock timeout | RooflineExecutor 内部串行（profile 完成才启动 trace_analyze），不并行；profile 已 require `profile_lane`，trace_analyze 不 require lane，理论上无冲突 |

---

## 12. 红线 — 不做的事（v2.0）

* ❌ 不引入任何 **analysis.md 二次解读层**（sub-agent / parser /
  heuristic / classifier 都不行）— 这是 §6.1 v1 失误的根本教训，
  TraceLens 团队明确反对
* ❌ 不写 1306 行 `identify_gaps.py` 等价物
* ❌ 不硬编码任何 model 名 / family
* ❌ 不动 `effective_score` 公式（只用已有 `score_mult` 钩子兜底）
* ❌ 不引入 PMC 或任何新测量手段
* ❌ 不引入新 SharedState 顶层字段（保留 v0 已有的 `last_select_kernels`
  dict 内添加；`last_roofline_analysis` 字段在 D1 revert 后会消失）
* ❌ 不动 `closing_phase` / install gate / ledger 数据结构
* ❌ 不引入 UNPRUNE_BRANCH intent（已知 trade-off，下个 PR）
* ❌ 不让 Coordinator 强制 re-profile / 强制 propose roofline（仍由
  主 LLM 主动 emit，orchestration.md 指导段引导）
* ❌ 不引入 manual `cache_control` 字段（claude-agent-sdk 0.2.82 不暴露，
  依赖 Claude Code automatic caching；如 N8 hit rate < 50% 才考虑下个
  PR 换 backend）
* ❌ 不在本 PR 做 trace_analyze rename 之外的 cross-repo 改名
* ❌ 不写其他 design 文档（本文档是唯一事实来源；N8 完成后追加
  §13 数字记录就足够）

---

## 13. N8 实测结果（待 GPU 跑完追加）

_N8 完成后在此追加 baseline / exp 数字、prompt diff 截图引用、verify
+ audit 输出表（含 §10.3 cache hit rate 与决策质量四项）。本节占位。_

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
