# Hyperloom Roofline-v2 设计文档

| 字段 | 值 |
|---|---|
| Status | Draft v1（实施进行中，C1/C2/C3 已合入） |
| Owner | xiaofei |
| Branch | `feature/xiaofei/roofline-v2` |
| Worktree | `/wekafs/xiaofei/Hyperloom-roofline-v2` |
| Base | `main` @ `550d24f` |
| Last updated | 2026-05-19 |
| 强制规则 | **任何方案改动必须先修改本文档再改代码**；本文档是单一事实来源 |

---

## 1. TL;DR

Hyperloom 当前的 Orchestration LLM 只在 prompt 里看到一行
`last_select_kernels: top=[k001,...] reusable=[...]`，看不到 TraceLens
已经写好的逐 kernel 瓶颈 / efficiency / 推荐建议；同时
`MODEL_CLASS_ACTION_PRIORS` 是 model_class-keyed 的静态先验，无法反映
"当前 trace 已经把 comm 从 50% 降到 15% 了，应该转去做 compute" 这种
**优化栈推进过程中的瓶颈漂移**。结果是 LLM 在 ~60 个 framework flag 里
盲扫，60 min session 普遍只跑 1 次 profile，`cumulative_gain_validated`
≈ 0。

**Roofline-v2** 把 `roofline` 做成一个**独立 action**，由主 LLM 主动
`PROPOSE_ACTION`，executor 内部 spawn 一个 **sub-agent LLM**
（Claude，独立 system prompt）专门读 TraceLens `analysis.md` 全文，
输出结构化决策 dict（`primary_bottleneck` / `suggested_prunes` /
`suggested_next_actions` / `reprofile_recommended`）回写
`SharedState.last_roofline_analysis`；主 LLM 在后续 tick 的 prompt 里
看到 ~800 字符的结构化结论段，据此 emit `PRUNE_BRANCH` + `PROPOSE_ACTION`，
PolicyGate / Coordinator 现有路径硬剪枝并集中尝试推荐 flag。
60 min session 内 `roofline` action 跑 2-3 次，每次跟随一次 re-profile
（由 `cumulative_gain` 跳 +3% 触发），让 LLM 始终基于当前优化栈下的
最新 roofline 报告决策。

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

1. **roofline 作为独立 action**（用户决策）：有 `MODEL_CLASS_ACTION_PRIORS`
   先验、有 sequence_denial 依赖、可被 PRUNE_BRANCH 硬剪枝、可被 audit
2. **sub-agent 推理**：roofline action 内部 spawn 一个轻量 sub-agent LLM
   读 `analysis.md` 全文产出结构化 JSON，不把 200 KB 报告塞主 LLM 上下文
3. **roofline 只建议，不执行**：sub-agent 输出 `suggested_prunes` /
   `suggested_next_actions`；**主 LLM 仍然要自己 emit** PRUNE_BRANCH /
   PROPOSE_ACTION。roofline 不直接写 `pruned_families`、不直接 enqueue task
4. **数据驱动、零硬编码 model**：所有判断基于 trace 实际瓶颈分布
5. **降级安全**：每个新模块在 unknown / 缺失 / 失败时退化为现状
6. **可验证**：每个 commit 都要有 prompt diff / 数字 / fixture 证据，
   每个 PR 都要在真实 GPU 跑出 +5%（不只是 "基建可见"）
7. **小步快走**：每 commit ≤5 主文件、单测 +20-100 行、零回归

---

## 5. 完整数据流（含 Re-Profile 触发）

```
T0  baseline                                                        [必跑，不变]
T1  profile #1                                                      [必跑]
T2  select_kernels #1
    ├─ TraceLens 写 analysis.md #1 (~15-200 KB)
    └─ ★ C1: SharedState.last_select_kernels 缓存 {
          analysis_md_path, analysis_md_text (全文，A3 决策),
          roofline_snapshot_id=1,
          roofline_baseline_gain_at_snapshot=0%,
          ...                                                        }

T3  ★ roofline action #1（C4: 主 LLM 主动 PROPOSE_ACTION）
    ├─ sequence_denial: 必须有 last_select_kernels.analysis_md_text 否则拒绝
    ├─ idempotency: 同一 snapshot_id 内 1 次（D2 决策）
    ├─ executor 内 spawn sub-agent backend (Claude, 独立 sp)
    │     ↓
    │  sub-agent prompt = roofline_analyzer.md + analysis.md 全文 +
    │     当前 cumulative_gain_validated + 当前 pruned_families
    │     ↓
    │  sub-agent 输出 JSON（含 schema 校验，C2 文档化的 schema）
    │     ↓
    │  解析为 RooflineAnalysis dict（失败 → 安全 fallback）
    └─ 回调 SharedState.record_roofline_analysis(result)            ← C2 已就绪

T4  后续每个 tick 的主 LLM prompt（C5）:
    === Roofline Decision (snapshot #N, analyzed at gain=X%) ===
    Primary bottleneck: comm (45%) > compute (30%) > memory (15%) > idle (10%)

    Suggested action families to prune:
    - kernel_opt (HIGH): compute saturated 92%, no reusable_native in top-5
    - deep_kernel_analysis (MED): comm >40%

    Suggested next actions:
    - HIGH params: try enable_two_batch_overlap / aiter_allreduce_fusion
    - HIGH comm_optimization: rccl Allreduce dominates
    - MED  backends: try moe_a2a_backend=deepep

    Full analysis.md: <path>
    Re-profile suggested: no (no gain delta to compare yet)

    === Roofline-driven Pruning Rules ===                            (静态文本)
    Based on the above, you SHOULD emit PRUNE_BRANCH when:
    - the listed family has confidence=HIGH AND
    - no action in that family has succeeded since the snapshot was taken
    Do NOT prune families you haven't tried yet at this snapshot.

    === Re-Profile Guidance ===                                       (静态文本)
    Re-profile (PROPOSE_ACTION profile) when ANY of:
    - cumulative_gain_validated_pct increased ≥ 3% since snapshot #N
    - All non-pruned families tried since snapshot #N with no new gain
    - You suspect the bottleneck has shifted (explain why)

T5  主 LLM 按 prompt 引导决策：
    ├─ emit PRUNE_BRANCH(kernel_opt)                                 ← C3 已开权限
    ├─ emit PRUNE_BRANCH(deep_kernel_analysis)
    └─ emit PROPOSE_ACTION(params, with enable_two_batch_overlap, ...)

T6  优化循环（params/backends/comm_opt 等）
    被 prune 的 family 在 _handle_delegate 现有路径硬拦截
    LLM 集中试 suggested_next_actions 列出的方向
    cumulative_gain_validated_pct 累积上升 (0% → 1.2% → 2.5% → 3.2%)

T7  ★ Re-Profile 触发：cumulative_gain - snapshot_baseline ≥ +3%
    主 LLM 看到 Re-Profile Guidance 段 → emit PROPOSE_ACTION(profile)

T8  profile #2
T9  select_kernels #2
    └─ C1 缓存覆盖 (snapshot_id=2, baseline_gain=3.2%)

T10 ★ roofline action #2（主 LLM 主动 propose；snapshot_id 变了，
    idempotency key 不同 → 允许新跑）
    └─ sub-agent 看到新的 analysis.md（瓶颈可能已转移）
       产出新的 RooflineAnalysis：例如 primary=compute (50%)
       suggested_prunes=[comm_optimization]（反过来了）
       suggested_next=[params w/ enable_torch_compile, operator_tuning, ...]

T11 主 LLM 看新 prompt 段 → 新一轮 prune + propose
    注：现有 pruned 没过期机制，之前 prune 的 kernel_opt 仍被拦截；
    这是已知 trade-off（unprune 留给下个 PR）

T12 优化循环 2 ... gain → 4.5% → 5.X%

T13 closing phase（_closing_phase_denial 拦截新 action，只跑 report）
```

### 5.1 频率估算（60 min Qwen3-32B session）

| 阶段 | wall-clock | profile/TraceLens | roofline action |
|---|---|---|---|
| baseline | 3-5 min | 0 | 0 |
| profile #1 + select_kernels #1 | 8-12 min | 1 | 0 |
| **roofline action #1** | 30-60 s | 0 | **1**（消耗 sub-agent LLM token） |
| 优化循环 1 | 10-15 min | 0 | 0 |
| **roofline-triggered profile #2 + select_kernels #2** | 8-12 min | 1 | 0 |
| **roofline action #2** | 30-60 s | 0 | **1** |
| 优化循环 2 | 10-15 min | 0 | 0 |
| (可选) profile #3 + roofline action #3 | 8-12 min | 0-1 | 0-1 |
| closing | ~3 min | 0 | 0 |
| **合计** | **60 min** | **2-3** | **2-3** |

---

## 6. C4 关键架构决策：sub-agent vs 直接注入

用户提出两个候选：(a) orchestrator 启动 sub-agent 读分析；(b) 把整个文档给 orchestrator。下面对比并定档。

### 6.1 三个候选对比

| 维度 | A. sub-agent in executor ★推荐 | B. 直接给主 orchestrator 上下文 | C. heuristic only (无 LLM) |
|---|---|---|---|
| 主 LLM 上下文负担 | 仅 ~800 字符结论段 | +15-200 KB analysis.md / tick | 仅 ~800 字符 |
| 解读质量 | 高（专精 sub-agent） | 高（同一 LLM 完整推理） | 中（关键字+lookup） |
| Token 成本 / session | 每次 roofline action 1 次 LLM call (~15KB↑ + ~1KB↓) × 2-3 次 | 每 tick 100+ 次注入大 prompt (×100 ticks = 1.5MB-20MB) | 0 新 token |
| 主 LLM 决策可解释性 | 高（看到结构化建议） | 中（要自己消化大段文本） | 高 |
| sub-agent 实施复杂度 | 中（需 spawn backend + new sp + JSON 校验） | 低（只改 prompt 渲染） | 低 |
| 与 D2 决策(每 snapshot 1次)的契合度 | 完美（task idempotency 天然成立） | 不契合（每 tick 都注入） | N/A |
| 测试可控性 | 高（mock backend.run） | 高（mock prompt input） | 高（pure func） |
| 与"action 化"语义一致 | 完全一致 | 不一致（退回隐式注入） | 一致 |

### 6.2 推荐 A：sub-agent in executor

**决策**：选 A。理由三条：

1. **Token 经济性**：B 每 tick 注入 200KB × 100 tick = 20MB，按 Claude
   Sonnet 计算 session token 费用 ~$30；A 仅 2-3 次 ×15KB ≈ 45KB，约 $0.5
2. **解读结构化**：A 的输出是 schema-validated JSON，主 LLM 看到的是
   `Primary: comm (45%); suggested_prunes: kernel_opt (HIGH);
   ...` 这种已经被推理压缩的结论，不需要再消化原始文本；B 让主 LLM
   每 tick 重读全文，结果可能不一致
3. **与 action 化语义对齐**：用户已选定 roofline 做成独立 action，A 是
   "action 内部委托 sub-agent 完成" 的标准模式；B 退化为隐式注入，让
   action 失去意义

C 之所以被否决：sub-agent 推理能力是 LLM 最大的复用收益，砍掉它去做
关键字 lookup 会让 RooflineAnalysis 质量大幅下降，C7 验证 +5% 概率显著
降低；如果只想"先验证基建"，C 是过渡方案而不是终态。

### 6.3 sub-agent 的具体形态

| 项 | 设计 |
|---|---|
| **Backend** | 直接构造 `ClaudeBackend`（同主 Orchestration LLM，按 D3 决策），独立的 client 实例（不复用主 LLM session） |
| **System prompt** | 新增 `inference_optimizer/orchestrator/system_prompts/roofline_analyzer.md`（~80 行），单一职责：读 analysis.md → 输出 schema JSON |
| **User prompt 组成** | (a) analysis.md 全文；(b) 当前 `cumulative_gain_validated_pct`；(c) `pruned_families`（避免建议已 prune 的）；(d) `optimization_stack` 长度（让 sub-agent 知道走过多少步） |
| **Tools** | 不给任何 tool；要求 sub-agent 仅输出 JSON（用 `tools=None, max_turns=1`） |
| **输出格式** | 严格 JSON，符合 C2 文档化的 schema；JSON 解析失败 → 写一个 `primary="unknown"` 的安全 fallback 进 SharedState，标记 `raw_llm_response` 含错误，executor 返回 status=succeeded 但 result.degraded=true |
| **Idempotency** | task idempotency_key = `f"roofline:{snapshot_id}"`，同一 snapshot 多次 propose 直接复用上次结果 |
| **超时** | 60s（sub-agent 看 200KB analysis.md + 1 turn 输出 JSON 充足） |
| **错误处理** | (1) backend 调用失败 → executor 返回 status=failed 但写入 fallback RooflineAnalysis 含 error 字段；(2) JSON schema 失败 → 同上 |

### 6.4 sequence_denial 集成

`roofline` action 必须满足：

- `last_select_kernels.analysis_md_text` 非空（否则没数据可分析）
- 当前 snapshot_id 还没有对应的 last_roofline_analysis（idempotency）
- 不在 closing_phase

不满足 → Coordinator 拒绝 propose（参考现有 `_sequence_denial_for_action`
对 `select_kernels` 的拒绝模式）。

---

## 7. 模块 / Commit 分解（实施跟踪表）

| Commit | 内容 | 主文件数 | 行数（含测试） | 状态 |
|---|---|---|---|---|
| **C1** | SharedState 缓存 analysis.md 全文 + snapshot 元数据 | 1 + 1 测试 | +283 | ✅ `23cb52b` |
| **C2** | RooflineAnalysis schema + `record_roofline_analysis` | 1 + 1 测试 | +393 | ✅ `9683ea0` |
| **C3** | Orchestration 获 PRUNE_BRANCH intent 权限 | 2 + 2 测试 | +180 | ✅ `15804ce` |
| **C4** | `roofline` action：meta yaml + markdown + sub-agent executor + scoring prior + cli 注册 + sequence_denial + coordinator 集成 + 单测（含 mock backend fixture） | 6-7 + 2-3 测试 | ~400 | 📝 待写 |
| **C5** | prompt 渲染 `_format_roofline_decision` 结论段 + "Pruning Rules" 段 + "Re-Profile Guidance" 段 + 单测 | 2 + 1 测试 | ~150 | 📝 待写 |
| **C6** | `scripts/verify_roofline_v2.py` + `scripts/audit_roofline_decisions.py` | 2 | ~250 | 📝 待写 |
| **C7** | Qwen3-32B baseline vs exp 真实 GPU 跑（用 nohup 后台，每 5 min 进度汇报）+ 数字记录追加进本文档 §11 | 0 代码 / 本文档追加 | 0 | 📝 待 GPU |

**累计估算**：~1450 行（含 ~800 行测试 + ~200 行 yaml/md + ~200 行脚本），核心 Python ~250 行；4 GPU·hour 验证。

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

**主文件数 = 9（6 新 + 3 改）**，超出 5 文件红线。按用户"单 PR 单 worktree"
指示允许，但本 commit 内需要严控行数。

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
