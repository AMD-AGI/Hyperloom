# 从 Trace 提取 “KB 使用情况” 指南

本文面向需要从 Langfuse trace / `session_breakdown.json` 中评估 **知识库（KB）是否被使用、命中了什么、是否影响了决策** 的同事。

适用代码版本：`feature/haiskong/trace-kb-provenance`（在 `main` 之上，新增了 recipe-KB 两条通路 gbrain / cortex 的分别归因）。文末单列一节说明 **PR #561 历史数据** 能提取什么、与本版本的差异。

---

## 1. KB 在 trace 里一共有几条记录线

一次优化会话里，KB 的读取来自两个 agent，对应三类 span，外加 `session_breakdown` 里的一个汇总块：

| 来源 | trace span | 说明 |
|---|---|---|
| 编排侧 warm-start | `agent:recipe_kb` → `kb:recipe_snapshot:get_recipe` | T0 暖启动 / CLOSE 读取历史最优 recipe |
| critic（每轮 review） | `agent:critic` → `kb_priors:iter_N` | 历史先验（pitfalls/lessons/同类决策）查询 |
| critic（每轮 review） | `agent:critic` → `kb_assess:iter_N` | substrate 对每个 proposal 的合理性评估 |
| 汇总 | `session_breakdown.kb_provenance` | 整会话 KB 审计聚合（durable artifact） |

> 注意：**specialist / orchestrator 不直接读 recipe-KB**，所以它们名下没有 KB span。recipe-KB 的读取统一挂在专用的 `recipe_kb` agent 下。

这些 span 由 `orchestrator/trace/langfuse_emitter.py::record_kb_span` 落库；recipe span 来自 `runtime/recipe_snapshot/.audit.jsonl` 的会话末回填。

---

## 2. 每类 span 的字段（判读要点）

### 2.1 `kb:recipe_snapshot:get_recipe`（编排侧 warm-start）

`output` 结构：

```json
{
  "method": "get_recipe",
  "remote": "composite",            // none / cortex / gbrain / composite
  "resolution": "remote",           // remote / remote_miss / remote_error / local
  "hit": true,
  "candidates": 2,                  // 合并后候选数
  "request": {
    "canonical_id": "inference:qwen-qwen3-32b:mi300x:sglang:0.5.11:fp8",
    "prefer_keys": ["conc","ep","framework_version","isl","max_model_len","osl","tp"],
    "label_match": null
  },
  "result": {
    "canonical_id": "...",
    "exact": true,
    "best_throughput": 5637.79,
    "best_config_nonempty": true,   // 关键：是否带可执行配置
    // —— 以下为 composite 命中时新增的两条通路归因 ——
    "sources": ["gbrain", "cortex"],          // 本行由哪些路贡献
    "best_config_source": "gbrain",           // 可执行配置由哪条路提供
    "field_sources": {"best_config":"gbrain","best_throughput":"cortex","lessons":["cortex"]},
    "source_candidates": {"gbrain": 1, "cortex": 1}  // 每条路各返回多少候选
  },
  "ts": "..."
}
```

判读：
- `hit && best_config_nonempty == true` → 拿到了**可用**的历史最优配置（真正能暖启动）。
- `hit && best_config_nonempty == false` → 命中了条目但配置为空（冷启动占位），会被降级 `seed_only`，不注入实际配置。
- `result.best_config_source` / `result.sources` → **gbrain 还是 cortex 提供了配置**（评价两条路贡献的核心字段）。
- `source_candidates` → 每条路覆盖度。

### 2.2 `kb_priors:iter_N`（critic 历史先验）

```json
{
  "configured": true,
  "mode": "per_proposal",       // 或 per_decision
  "client_mode": "inmemory",    // 关键：inmemory=未接持久化KB；live=已接HTTP KB
  "scope_filter": {"org":"hyperloom","framework":"sglang","model":"...","precision":"fp8","workload":"..."},
  "limit": 5,
  "requests": [{"msg_id":"...","topic":"target_analysis","cache":"miss","count":0}],
  "skipped_reason": null,
  "prior_count": 0,             // 关键：返回了几条先验
  "referenced_in_verdict": false // 关键：最终 verdict 是否引用了 KB
}
```

判读：
- `client_mode == "inmemory"` → critic 的 priors 接的是**进程内内存桩**，永远返回 0，等于这条通路空转（需设 `CRITIC_KB_CLIENT_MODE=live` + `KB_BASE_URL` 才会真正查库）。
- `prior_count > 0` → 实际取到了历史先验。
- `referenced_in_verdict == true` → KB 证据进入了最终决策。

### 2.3 `kb_assess:iter_N`（critic 合理性评估）

```json
{
  "configured": true,
  "skipped_reason": null,        // 或 no_proposals / not_configured / unknown_model
  "focus": {"model":"...","hardware":"mi300x","framework":"sglang","precision":"fp8"},
  "requests": [
    {"msg_id":"...","skipped":"no_levers"},                 // 提案无可调lever，跳过
    {"msg_id":"...","params_keys":["..."],"responded":false} // 调了但substrate没返回
  ],
  "verdict_count": 0,            // 关键：拿到几个verdict
  "injected": false,            // 关键：verdict是否喂进了LLM prompt
  "mode": "dry_run",            // dry_run=影子模式（不喂LLM）；inject=喂LLM
  "referenced_in_verdict": false
}
```

判读：
- `mode == "dry_run"` / `injected == false` → assess 只采集不喂 LLM（开 `CORTEX_KB_ASSESS_INJECT=1` 才喂）。
- `verdict_count > 0 && injected == true && referenced_in_verdict == true` → assess 真正参与了决策。
- 每个 request 的 `skipped` / `responded` 解释了为什么没产出 verdict。

### 2.4 `session_breakdown.kb_provenance`（durable 汇总，推荐离线分析用）

```json
{
  "cortex_session_id": "...",
  "warm_start_recipe_seen": true,
  "warm_start_recipe_tier": "exact",      // exact / relative / seed_only / miss
  "warm_start_recipe_source": "gbrain",   // 最终应用的recipe来自哪条路（本版本新增）
  "warm_history_injected": true,
  "recipe_snapshot_reads": {
    "count": 2, "hits": 2,
    "by_resolution": {"remote": 2},
    "by_remote": {"composite": 2},
    "by_source": {"gbrain": 2, "cortex": 1},          // 两条路各命中几次（本版本新增）
    "best_config_by_source": {"gbrain": 1},           // 配置由哪条路供给（本版本新增）
    "tail": [ /* 最近10条recipe审计行，含上面2.1的result */ ]
  },
  ...
}
```

`session_breakdown` 里另有 `critic_robustness.critic_iterations[].kb_assess / kb_priors`，是每轮 critic 的 KB 审计（与 2.2/2.3 的 span 同源）。

---

## 3. 三种提取方式

### 方式 A：Langfuse UI（单 trace 人工排查）
1. 打开 trace，按 span 名筛 `kb_assess` / `kb_priors` / `kb:recipe_snapshot`。
2. 看每个 span 的 Output（即上面的 JSON），按第 2 节判读要点判断。

### 方式 B：读 `session_breakdown.json`（推荐，离线、可批量）
session 结束后产物里有 `*_session_breakdown.json`。直接取 `kb_provenance` 与 `critic_robustness.critic_iterations[]`：

```bash
jq '.kb_provenance.recipe_snapshot_reads | {hits, by_source, best_config_by_source}' breakdown.json
jq '.kb_provenance.warm_start_recipe_source' breakdown.json
jq '[.critic_robustness.critic_iterations[]
     | {iter, priors: .kb_priors.prior_count, verdicts: .kb_assess.verdict_count,
        ref: (.kb_priors.referenced_in_verdict or .kb_assess.referenced_in_verdict)}]' breakdown.json
```

### 方式 C：ClickHouse 直查（后端批量统计，需 langfuse DB 访问）
Langfuse v3 的 observation 落在 ClickHouse `observations` 表，`output` 为 JSON 字符串。

```sql
-- 单 trace 的所有 KB span
SELECT start_time, name, output
FROM observations
WHERE trace_id = '<trace_id>'
  AND (name LIKE 'kb_priors%' OR name LIKE 'kb_assess%' OR name LIKE 'kb:recipe%')
ORDER BY start_time;

-- recipe 两条路的命中归因（per trace）
SELECT
  JSONExtractString(output, 'remote')                                   AS remote,
  JSONExtractBool(JSONExtractRaw(output,'result'),'best_config_nonempty') AS actionable,
  JSONExtractString(JSONExtractRaw(output,'result'),'best_config_source') AS best_config_source,
  JSONExtractString(JSONExtractRaw(output,'result'),'sources')           AS sources
FROM observations
WHERE name LIKE 'kb:recipe%' AND trace_id = '<trace_id>';

-- critic 命中率（注意 observations 是 ReplacingMergeTree，需按 id 去重）
SELECT
  countIf(JSONExtractInt(output,'prior_count') > 0)   AS priors_hit,
  countIf(JSONExtractInt(output,'verdict_count') > 0) AS assess_hit,
  max(JSONExtractBool(output,'referenced_in_verdict')) AS kb_referenced
FROM (SELECT any(output) AS output FROM observations
      WHERE trace_id='<trace_id>' AND (name LIKE 'kb_priors%' OR name LIKE 'kb_assess%')
      GROUP BY id);
```

> 性能提示：`observations.output` 里 `session_breakdown` 行很大（~130KB），批量扫描时用 `--max_block_size` 调小、按 `trace_id` 分批，避免内存超限。

---

## 4. 评价 “KB 是否起作用” 的判定清单

| 维度 | 起作用的信号 | 空转/无效的信号 |
|---|---|---|
| recipe warm-start | `hit && best_config_nonempty=true`，`warm_start_recipe_tier=exact/relative` | `best_config_nonempty=false` 或 `tier=seed_only/miss` |
| 哪条路贡献 | `best_config_source` / `warm_start_recipe_source` = gbrain 或 cortex | `by_source` 为空（非 composite） |
| critic priors | `prior_count>0` 且 `client_mode=live` | `client_mode=inmemory` 或 `prior_count=0` |
| critic assess | `verdict_count>0` 且 `injected=true` | `mode=dry_run` / `verdict_count=0` |
| 是否进决策 | 任一 KB span `referenced_in_verdict=true` | 全为 false |

---

## 5. 基于 PR #561 提交下的历史数据

**PR #561**（merge `1f9b2725`，单 commit `f7d67afb`，分支 `feature/haiskong/session-breakdown-augment`）引入：
- `action_timeline[]` 每条 action 带 `extras.operation_kind` 与 `extras.proposer / provenance`；
- 每个决策的 `optimization_step:<operation_kind>` span（metadata 含 `operation_kind` / `proposer` / `effect`）。

### 5.1 PR561 历史数据**能**提取什么
凡是用 PR561（含）之后代码跑出来的 trace / breakdown，均可提取：
1. **KB 使用情况（composite 层）**：recipe_snapshot 的 `hit / resolution / candidates / best_config_nonempty`，critic 的 `prior_count / verdict_count / mode / client_mode / referenced_in_verdict`，以及 `kb_provenance.recipe_snapshot_reads.{by_remote,by_resolution,hits}`。
   - 提取方式同第 3 节（方式 B/C 完全适用）。
2. **决策归因**：每个 action 的 `proposer`（谁提出的）+ `operation_kind`（env/param/backend/kernel/specialist…）+ `decision`（KEEP/REVERT/no_promote…）+ `key_metric`。
   - 提取：`jq '.action_timeline[] | {action, decision, proposer: .extras.proposer, kind: .extras.operation_kind, key_metric}' breakdown.json`
   - 或 ClickHouse 查 `optimization_step:%` span 的 metadata。
3. 把 1 与 2 交叉，即可回答 “KB 命中后，由哪个 proposer 采用、最终 KEEP 还是 REVERT”。

### 5.2 PR561 历史数据**不能**提取什么（与本版本的差异）
PR561 时代的 recipe 审计 **只记录到 `remote: "composite"` 层**，不区分 gbrain / cortex：
- `result` 里**没有** `sources / best_config_source / field_sources / source_candidates`；
- `kb_provenance.recipe_snapshot_reads` **没有** `by_source / best_config_by_source`；
- `kb_provenance` **没有** `warm_start_recipe_source`（PR561 时 warm 源是按 remote 类型猜的，composite 下一律误标 `cortex-kb`）。

因此对 PR561 历史数据，**无法按 gbrain vs cortex 拆分贡献**——只能看到 composite 整体命中。要做两条路的分别归因，需用 `feature/haiskong/trace-kb-provenance`（含）之后的代码重新产出 trace。

### 5.3 对 PR561 历史数据的近似提取
若必须从 PR561 历史数据近似区分来源，只能间接推断：
- 看 `recipe_snapshot_reads.by_remote`：若 `remote != "composite"`（即单 `gbrain` 或单 `cortex`），则该次读取的来源即 `remote` 本身；
- composite 的读取无法回溯拆分（per-source 信息在合并时已丢弃，未落审计）。

---

## 6. 相关代码位置

- recipe 审计与两条路归因：`recipe_kb/dispatcher.py::_read_audit_event`、`recipe_kb/composite_remote.py::_merged_search`
- warm-start 来源归属：`orchestrator/cortex_t0.py::_warm_recipe_source`
- critic KB 审计：`orchestrator/backends/critic_agent.py::_build_kb_assess_trace / _build_kb_priors_trace`，`critic-agent/runtime/decision_reviewer.py::_inject_kb_assess`
- span 落库：`orchestrator/trace/langfuse_emitter.py::record_kb_span`
- 汇总：`breakdown/collectors.py::collect_kb_provenance`，schema 见 `breakdown/schema.py::KBProvenance`
