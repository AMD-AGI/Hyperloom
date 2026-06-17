# 从 Trace 提取决策时间线（Specialist / Orchestrator / Critic）

本文面向需要从一次 Hyperloom run 的产物里，还原出
**“specialist 的提议历史、orchestrator 的决策历史、critic 的决策历史”**
三条时间线的同学。基于当前 `feature/haiskong/trace-v3` 版本编写，并在最后单独
说明 **PR #561（commit `f7d67afb`）历史数据** 能提取什么、怎么提取。

> 约定：下文 `<sd>` 指一次 run 的 session 目录（`USER_DATA_PATH/<session_id>/`）。
> 标注 `(v3)` 的字段/文件是本版本（trace-v3）新增的；**PR561 时期的历史数据没有这些**，
> 见 [§5](#5-基于-pr561-的历史数据)。

---

## 1. Trace 数据源总览

一次 run 的可提取产物分布在三类位置：`reports/trace/*.jsonl`（逐调用流水）、
`session_breakdown.json`（聚合视图，**推荐入口**）、以及若干运行期状态/工作目录。

| 路径 | 内容 | 关键字段 |
|------|------|----------|
| `<sd>/session_breakdown.json` | **聚合总视图**（导出时生成，组合下列大部分） | `decision_trace` / `token_usage` / `specialist_runs` / `critic_robustness` / `action_timeline` / `attribution` / `kb_provenance` |
| `<sd>/reports/trace/decision_trace.jsonl` | 决策↔成本 join 时间线（collector 产物，每次导出整文件重写） | `phase, tick, ts, decision{...}, tokens{...}` |
| `<sd>/reports/trace/llm_calls.jsonl` | 每次 LLM 调用的 token 账（单写者：父进程） | `component, role, task_id, dyn_id, tick, phase, turn, model, input/output/cache_* tokens, latency_ms(v3), reviewed_msg_ids(v3)` |
| `<sd>/reports/trace/conversations.jsonl` | 每次 LLM 调用的完整（脱敏）prompt/response | `component, role, task_id, tick, phase, turn, model, prompt, response` |
| `<sd>/reports/optimization_journal.json` | KEEP/REVERT 决策权威源 | `entries[]{phase, iter, kind, change, outcome, gain_pct, task_id, variant_name, provenance, scope, fingerprint, metrics, tick, predicted_gain_pct(v3)}` |
| `<sd>/state.json` | 运行期状态 | `specialist_rounds[]`、`explore_search`、`phase_history` |
| `<sd>/reports/trace/specialist_intel.jsonl` (v3) | specialist 工具/情报读取（WebSearch/WebFetch/pr_monitor/cortex_kb/Read/Grep…） | `task_id, turn, tool, query, ts` |
| `<sd>/reports/trace/proposal_task_map.jsonl` (v3) | `proposal_msg_id → task_id` 映射（approve 物化时写） | `proposal_msg_id, task_id, ts` |
| `<sd>/reports/trace/forge_steps.jsonl` (v3) | Kernel-Forge 自治循环关键步骤 | `kind(iteration/summary), kernel_id, iteration, decision, wall_ms, snr_db, rationale, termination_reason` |
| `<sd>/critic-workdir/<turn>/` | critic 每轮原始产物 | `request.json` / `judge_bundle.json` / `review.json` / `emit.json` |
| `<sd>/agents/orchestration/dynamic_actions/<dyn_id>/dispatch_history.jsonl` | dynamic_action 决策流 | `event, verdict, delta_pct, integrate_status, terminal_state` |
| `<sd>/runtime/recipe_snapshot/.audit.jsonl`、`<sd>/runtime/cortex/.kb_audit.jsonl` | KB 远端读取审计 | `method, remote, resolution, hit` |
| SQLite `events` 表（message bus） | 原始消息总线（`proposal` / `review_verdict` / `decision` topic） | `msg_id, from_agent, to_agent, topic, payload, ts` |
| Langfuse（若开启 live push） | 镜像 trace：phase→agent→generation span + decision Score | span `optimization_step:<operation_kind>`、`intel:<tool>`(v3)、`forge:*`(v3)；Score `gain_pct/predicted_gain_pct(v3)/proposal_score(v3)` |

**最快入口**：`session_breakdown.json` 已组合了三条线的绝大部分；
`reports/trace/decision_trace.jsonl` 是按时间排好的决策+成本流。

---

## 2. Specialist 提议历史

**“某个 specialist 在某轮针对某 gap 提了哪些 variant、各自评分多少、读了什么、花了多少 token”**

### 数据源
- 提议本体 + 评分：`state.json → specialist_rounds[]`（或 `session_breakdown.json → specialist_runs[]`）
  - `round_id`（默认 = `task_id`）、`task_id`、`domain`、`gap_canonical_id`、`completed_at`、`confidence`、`summary`
  - `proposal_set[]`：每个 variant 的 `name` / `extra_args`(或 `extra_server_args`) / `extra_envs` / `reason` / `kb_evidence`
  - `ensemble_scores.models.<model_slug>.<proposal_name>.{score, reason}`：proposal_scorer 多模型评分
- 推理全文：`conversations.jsonl` 中 `component == "specialist"`（用 `task_id` 关联，`turn` 区分多轮）
- 成本/时延：`llm_calls.jsonl` 中 `component == "specialist"`（`task_id`、`tick`、`phase`、`turn`、tokens、`latency_ms`(v3)）
- 读了什么 (v3)：`specialist_intel.jsonl`（`task_id`、`tool`、`query`）

### 时间线排序
按 `specialist_rounds[].completed_at`（或各源的 `ts`）升序即得每轮提议的时间线。

### 提取示例
```bash
# 每轮：时间 / 域 / gap / 提了几个 variant / 每个 variant 的评分
jq -r '.specialist_rounds[]
  | "\(.completed_at)  domain=\(.domain)  gap=\(.gap_canonical_id)  n=\(.proposals_total)"
  ' <sd>/session_breakdown.json

# 展开某轮的 variant + 多模型评分
jq '.specialist_rounds[] | select(.task_id=="<task_id>")
  | {round_id, domain,
     proposals: [.proposal_set[].name],
     scores: .ensemble_scores.models}' <sd>/session_breakdown.json

# 该 specialist 这轮读了哪些情报 (v3)
jq -c 'select(.task_id=="<task_id>") | {tool, query}' \
  <sd>/reports/trace/specialist_intel.jsonl
```

---

## 3. Orchestrator 决策历史

**“每一步保留/回退了什么改动、增益多少、是谁提的、属于哪类操作”**

### 数据源
- 权威源：`optimization_journal.json → entries[]`
  - `outcome` ∈ `{"KEEP", "REVERT", "no_promote"}`、`gain_pct`（实测）、`change`、`kind`
  - `task_id` / `variant_name` / `fingerprint`、`provenance`、`scope`、`metrics`、`tick`
  - `predicted_gain_pct`(v3)（提议方自报的预测增益）
- dynamic action 决策：`agents/orchestration/dynamic_actions/<dyn_id>/dispatch_history.jsonl`
- **合成时间线（推荐）**：`reports/trace/decision_trace.jsonl`，每行 =
  ```jsonc
  {
    "phase": "EXPLORE", "tick": 7, "ts": "...Z",
    "decision": {
      "component": "specialist:serving_specialist",   // 解析后的 proposer
      "operation_kind": "backend",                    // 操作类型过滤标签
      "change": "--attention-backend AITER",
      "outcome": "KEEP", "gain_pct": 4.2,
      "task_id": "...", "variant_name": "v01",
      "provenance": "specialist:serving_specialist", "scope": "domain",
      "fingerprint": "fp1", "metrics": {...},
      "proposal_scores": [{"rater": "...", "score": 8.5, "reason": "..."}],
      "predicted_gain_pct": 9.0                        // (v3)
    },
    "tokens": {"by_component": {...}, "total_in": ..., "total_out": ..., "calls": ...}
  }
  ```
- 推理全文：`conversations.jsonl` 中 `component == "orchestration"`

### 提取示例
```bash
# 决策时间线：时间 / 阶段 / 谁提的 / 操作类型 / 结果 / 增益 / 成本
jq -r '"\(.ts)  \(.phase)  \(.decision.component)  \(.decision.operation_kind)  "
       + "\(.decision.outcome)  gain=\(.decision.gain_pct)  tok=\(.tokens.calls)"' \
  <sd>/reports/trace/decision_trace.jsonl

# 只看 KEEP，并按 proposer 聚合总增益
jq -s '[.[] | select(.decision.outcome=="KEEP")]
       | group_by(.decision.component)
       | map({proposer: .[0].decision.component,
              keeps: length,
              gain_sum: (map(.decision.gain_pct // 0) | add)})' \
  <sd>/reports/trace/decision_trace.jsonl
```

---

## 4. Critic 决策历史

**“critic 第几轮评审了哪些 proposal、裁决（approve/reject/…）是什么、依据什么、花了多少 token”**

### 数据源
- 聚合：`session_breakdown.json → critic_robustness.critic_iterations[]`
  - `iter`、`ts`、`verdict`、`summary`、`request_path` / `judge_bundle_path` / `review_path` / `emit_path`
  - `kb_assess` / `kb_priors`（该轮 KB 使用证据）
- 原始产物：`critic-workdir/<turn>/`
  - `judge_bundle.json`：被评审的 `proposals[]`（每个含 `msg_id`）
  - `review.json` / `emit.json`：裁决（`review_verdicts[]{target_proposal_msg_id, verdict, reasoning}`）
- 裁决→提议→决策 关联：
  `review_verdict.target_proposal_msg_id`
  → `proposal_task_map.jsonl`(v3) 得到 `task_id`
  → 回到 `decision_trace.jsonl` 找该 `task_id` 的最终 KEEP/REVERT
- 成本/时延：`llm_calls.jsonl` 中 `component == "critic"`
  （v3 起带 `tick` / `phase` / `latency_ms` / `reviewed_msg_ids`）
- 预测 vs 实测：`decision_trace` 的 `proposal_scores`（评分器）与 `gain_pct`（实测）；
  v3 另有 `predicted_gain_pct`（提议方预测）
- 原始裁决事件（进阶）：message bus 的 SQLite `events` 表 `topic == "review_verdict"`

### 提取示例
```bash
# critic 每轮裁决时间线
jq -r '.critic_robustness.critic_iterations[]
  | "\(.ts)  iter=\(.iter)  verdict=\(.verdict)  \(.summary[0:80])"' \
  <sd>/session_breakdown.json

# 某轮评审了哪些 proposal（含 msg_id）+ 裁决
jq '{proposals: [.proposals[].msg_id]}' <sd>/critic-workdir/000003/judge_bundle.json
jq '.review_verdicts' <sd>/critic-workdir/000003/review.json

# 该 verdict 命中的 proposal 最终物化成了哪个 task (v3)
jq -c 'select(.proposal_msg_id=="<msg_id>")' <sd>/reports/trace/proposal_task_map.jsonl
```

---

## 5. 三条线如何对齐成一条端到端链路

共同连接键：

- `task_id`：specialist round（`task_id=spec-X`）→ 物化的执行任务 → journal/decision_trace 的 `task_id`
- `proposal_msg_id`：critic 评审的 proposal ↔ `proposal_task_map`(v3) ↔ 物化任务 `task_id`
- `ts` / `tick`：全局时间轴；`phase` 可用 `state.json.phase_history` 的窗口回填
- `decision_trace.jsonl` 本身已是 **orchestrator 决策 + critic 评分** 的 join；specialist 经 `task_id` 串入

一条典型链路：
```
specialist round (task=spec-7, proposal_set[v01,v02], ensemble_scores)
   └─(进 inbox 成 proposal, 各有 msg_id)
critic review (judge_bundle.proposals[msg_id], review_verdicts[approve v01])
   └─ approve → 物化任务 (proposal_task_map: msg_id → task_id=approved-…)
explore 跑 variant → journal KEEP/REVERT (task_id, variant_name=v01, gain_pct)
   └─ decision_trace 行 (component=specialist:<domain>, operation_kind, gain_pct, proposal_scores, tokens)
```

---

## 6. 基于 PR #561 的历史数据

**PR #561**（merge `1f9b2725`，commit `f7d67afb`，分支 `session-breakdown-augment`）的主题是
**“proposer attribution + operation_kind 贯穿 timeline & trace”**。它把“谁提的 / 哪类操作 / 怎么测的”
从 explore executor 一路串到 journal、breakdown 时间线和 Langfuse trace。

### 6.1 PR561 历史数据能提取什么
对 **PR561 合入之后产生** 的 session，可从其产物提取：

1. **proposer 归因**：每个 KEEP/REVERT 是谁提的——
   `proposer/component` ∈ `specialist:<domain>` / `grid` / `orchestration`，
   原始标签 `provenance`（`llm_direct` / `default_grid` / `specialist:<domain>`）+ `scope` + `fingerprint`。
2. **operation_kind**：改动类型过滤标签（`backend` / `param` / `env` / `kernel_opt` / `kernel_integrate` / …）。
3. **proposal_scorer 评分 join**：`decision_trace[].decision.proposal_scores`
   与 `specialist_rounds[].ensemble_scores`。
4. **per-variant metrics**：`runtime_sec` / `wall_clock_ratio_vs_baseline` /
   `stack_rebench_tput` / `estimated_output_throughput`（journal 与 decision_trace 的 `metrics`）。
5. **action_timeline 过滤标签**：`session_breakdown.json → action_timeline[]` 的 `proposer` / `operation_kind` /
   `provenance` / `scope` / `fingerprint`（PR561 在 `ActionTimelineEntry.extras` 里补的）。
6. **Langfuse**：每个决策一个 `optimization_step:<operation_kind>` span，metadata 带 `operation_kind` / `proposer` / effect，可按步骤类型过滤。

这些已足以还原 §2/§3/§4 的三条时间线的**主干**（提议→评分→KEEP/REVERT→归因）。

### 6.2 如何提取（PR561 数据）
入口同上，但只用 PR561 时期就存在的文件：
```bash
# 按 proposer × operation_kind 聚合 KEEP 数与总增益（PR561 数据即可）
jq -s '[.[] | select(.decision.outcome=="KEEP")]
       | group_by(.decision.component + "|" + (.decision.operation_kind // "?"))
       | map({key: .[0].decision.component + "|" + (.[0].decision.operation_kind // "?"),
              keeps: length, gain_sum: (map(.decision.gain_pct // 0) | add)})' \
  <sd>/reports/trace/decision_trace.jsonl

# journal 直接带 provenance/scope/metrics（PR561 加的）
jq '.entries[] | {ts, outcome, change, provenance, scope, fingerprint,
                  gain_pct, metrics}' <sd>/reports/optimization_journal.json

# breakdown 时间线的 proposer/operation_kind 标签
jq '.action_timeline[] | {ts, action, change, decision,
                          proposer: .extras.proposer,
                          operation_kind: .extras.operation_kind}' \
  <sd>/session_breakdown.json
```

### 6.3 PR561 历史数据**没有**的（属于 trace-v3，提取时会缺）
以下字段/文件是本版本（trace-v3）才有的，PR561 数据里不存在，提取脚本要做存在性判断：

- `latency_ms`：无调用时延（旧 generation 在 Langfuse 里是零宽点）。
- `overhead` vs `unattributed` 细分：旧数据只有 `unattributed_tokens`，未区分编排/critic/robustness 的“合理跨决策开销”。
- **critic 成本归因**：旧数据 critic token 计入 `unattributed`，无法落到单个 proposal/决策（`reviewed_msg_ids` / `proposal_task_map` 均缺）。
- `predicted_gain_pct` 校准分：旧数据只有 `proposal_scores`（评分器）vs `gain_pct`（实测），没有“提议方自报预测增益 vs 实测”。
- `specialist_intel.jsonl`：无 specialist 工具/情报读取明细。
- robustness / forge 的 token 与 `forge_steps`：旧数据无 robustness RCA token、无 forge token、无 forge 关键步骤 span。
- specialist subprocess 的 per-turn token 粒度：旧数据子进程整轮塌缩为单行。

> 实操建议：写提取脚本时对上述字段一律 `// empty` / `getattr(..., None)` 兜底，
> 这样同一脚本能同时跑 PR561 历史数据与 trace-v3 新数据。

---

## 7. 附：一条命令快速定位

```bash
# 这次 run 到底产了哪些 trace 文件
ls -1 <sd>/reports/trace/ ; ls -1 <sd>/critic-workdir/ 2>/dev/null
# 聚合总视图里有哪些段
jq 'keys' <sd>/session_breakdown.json
```
