# Gap-08 — T3 verify 仍是 per-task, 未升级到 per-variant

> 严重度: **P1 主要** (与 Gap-07 配对)
> 主轴影响: **主轴 B (知识外接 Cortex)**
> 体检报告: `../KB_design_gaps.MD` §5 Gap-8

## 1. 问题描述

KB_design §3.13 M5 §5 step 7: "for each variant: KEEP → ingest_attempt
PASS + verify edge=kb_edge_ids[variant] outcome=confirmed promote=EXPERIENTIAL;
REVERT → ingest_attempt FAIL + verify outcome=refuted".

实际:
- dispatcher 每 task 跑完调一次 `_cortex_t3_hook` (~3133-3137)
- `_cortex_t3_hook` (~3177-3268) 用 task-level `promoted` (来自
  `_is_promotable_result`) 决定整批 confirmed/refuted
- explore executor 内部已经 per-variant KEEP/REVERT (~10-21), 但**不上送
  Cortex**

后果:
- KB negation edge 永远基于"整批 variant 是否有 ≥ 1 KEEP", 无法学到
  "variant X refuted 但 variant Y confirmed"
- 跨 session 复用知识时, Cortex 把整个 round 标 confirmed/refuted,
  无法精确过滤个别失败的 variant

## 2. 现状代码 trace

### 2.1 dispatcher per-task

`coordinator.py:3133-3137`:

```text
async def _on_task_completed(self, task: Task, result: dict) -> None:
    ...
    promoted = self._is_promotable_result(task, result)
    if self.cortex_kb:
        await self._cortex_t3_hook(task, result, promoted=promoted)
```

只调一次 t3 per task.

### 2.2 _is_promotable_result

`coordinator.py:2929-2936`:

```text
def _is_promotable_result(self, task, result) -> bool:
    status = (result or {}).get("status", "")
    if task.kind == "explore":
        return status != "failed"   # 整批结果非 failed 即 "promoted"
    ...
```

整批粒度, 不分 variant.

### 2.3 _cortex_t3_hook

`coordinator.py:3177-3268`:

```text
async def _cortex_t3_hook(self, task, result, *, promoted: bool) -> None:
    ...
    edge_id = self._get_kb_edge_id_for_task(task)  # 单 ID
    outcome = "confirmed" if promoted else "refuted"
    await self.cortex_kb.ingest_attempt(...)
    await self.cortex_kb.verify(edge_id=edge_id, outcome=outcome, ...)
```

单 edge_id + 单 outcome.

### 2.4 explore.py 已 per-variant KEEP/REVERT

`explore.py:10-21`:

```text
class ExploreExecutor:
    """Run a grid of variants. Each variant gets its own KEEP/REVERT
    decision based on the inlined stack rebench."""
```

executor 内部维护 per-variant 状态, 但只把 *aggregate* (status='succeeded'
+ accepted list + rejected list) 上送, 不暴露 per-variant outcome.

## 3. 设计意图

M5 §5 step 7:
- KEEP variant → ingest_attempt PASS + verify confirmed
  → KB edge 标 EXPERIENTIAL → 下次 session 检索可见
- REVERT variant → ingest_attempt FAIL + verify refuted
  → KB negation edge → 下次 session 检索 *过滤* 此 variant

设计目的:
- 跨 session "fail-fast" — Cortex 见过的失败 variant 不再 propose
- specialist 在 KB subgraph 中明确看到"这个 variant 上次失败了"

## 4. 根本原因

与 Gap-07 一对: M5 设计了 per-variant 流, 但 M3 (explore 合并) PR 只
接 *executor 入口*, 没接 *outcome 出口*. 三个工件链断在 explore.py 和
coordinator t3 hook 之间.

## 5. 修复路径

### PR 5.1 — explore.py 上送 per-variant outcomes

`action_executors/explore.py` 在 result dict 中暴露:

```text
result = {
    "status": "succeeded",
    "accepted": [...],
    "rejected": [...],
    # NEW (v0.8 M5 §5 step 7):
    "per_variant_outcomes": [
        {
            "variant_name": "vA",
            "outcome": "KEEP",   # or "REVERT", "SKIPPED_DEDUP"
            "metrics": {"tput": ..., "accuracy": ...},
            "reason": "",        # for REVERT
            "kb_edge_id": variant.get("kb_edge_id", ""),
        },
        ...
    ],
}
```

### PR 5.2 — `_cortex_t3_hook` 升级 per-variant 路径

```text
async def _cortex_t3_hook(self, task, result, *, promoted: bool) -> None:
    if not self.cortex_kb:
        return

    # Per-variant path (explore action with detailed outcomes)
    per_variant = result.get("per_variant_outcomes")
    if task.kind == "explore" and isinstance(per_variant, list):
        await self._cortex_t3_per_variant(task, per_variant)
        return

    # Legacy: task-level promote (kernel_opt / integrate / baseline / ...)
    await self._cortex_t3_per_task(task, result, promoted=promoted)
```

### PR 5.3 — `_cortex_t3_per_variant` 实现

```text
async def _cortex_t3_per_variant(
    self, task: Task, outcomes: list[dict],
) -> None:
    pending = self._find_pending_proposal_for_task(task)
    if not pending:
        log.warning("T3 per-variant: no pending proposal for task %s", task.task_id)
        return
    sid = self.shared_state.cortex_session_id

    for vo in outcomes:
        variant_name = vo.get("variant_name", "")
        outcome = vo.get("outcome", "")
        edge_id = pending.kb_edge_ids.get(variant_name) or vo.get("kb_edge_id", "")
        if not edge_id:
            continue

        # ingest_attempt
        if outcome == "KEEP":
            attempt_outcome = "PASS"
            verify_outcome = "confirmed"
            promote = "EXPERIENTIAL"
        elif outcome == "REVERT":
            attempt_outcome = "FAIL"
            verify_outcome = "refuted"
            promote = None
        else:  # SKIPPED_DEDUP / etc.
            continue

        try:
            await self.cortex_kb.ingest_attempt(
                session_id=sid,
                iter=task.task_id,
                outcome=attempt_outcome,
                metrics=vo.get("metrics") or {},
                attrs={"variant_name": variant_name, "task_kind": "explore"},
            )
            await self.cortex_kb.verify(
                edge_id=edge_id,
                outcome=verify_outcome,
                promote=promote,
                attrs={"variant_name": variant_name},
            )
        except Exception as e:
            log.warning("T3 verify failed for variant %s: %s", variant_name, e)
```

### PR 5.4 — `_cortex_t3_per_task` 沿用旧逻辑

非 explore 任务走旧 per-task 路径, 改动是把现有 `_cortex_t3_hook` 主
体移到 `_cortex_t3_per_task`.

### PR 5.5 — 测试

`tests/test_v08_t3_per_variant.py`:

- mock cortex_kb verify + ingest_attempt
- 构造 explore task with 3 variants: 1 KEEP, 1 REVERT, 1 SKIPPED_DEDUP
- 验证 verify 调 2 次 (KEEP + REVERT), ingest_attempt 调 2 次
- 验证 KEEP edge outcome='confirmed', REVERT outcome='refuted'
- 验证 SKIPPED_DEDUP 完全跳过

## 6. 验收口径

- [ ] explore round with K variants (K ≥ 2, 含 mixed KEEP/REVERT) →
      Cortex verify 被调 K 次 (减去 skipped)
- [ ] breakdown.kb_provenance.edges_promoted / edges_negated 长度匹配
      per-variant outcomes
- [ ] kernel_opt / integrate / baseline 等非 explore action 仍走 per-task
      路径 (back-compat)
- [ ] explore.py per_variant_outcomes 字段在 v0.6 reader 读 result 时不
      崩 (新字段, JSON 兼容)

## 7. 风险 / 回退

- **per-variant verify 失败**: 单个 variant verify 失败不阻塞其他
  variants; NDJSON 兜底 (R-01).
- **outcome 不一致风险**: explore 内部判 KEEP, 但 stack rebench 后判
  REVERT → outcome="REVERT" 上送, KB 标 refuted. 这是正确语义.
- **回退**: explore.py 不上送 per_variant_outcomes → `_cortex_t3_hook`
  回退到 per-task 路径. Gap-07 不动情况下也可正常工作.

## 8. 关联 gap

- **配对**: Gap-07 (T2 per-variant). 单独修 Gap-08 没用 (没有 per-variant
  edge_id 可 verify).
- **依赖**: Gap-03 (specialist_done 路由) PR 5.4 在 _handle_specialist_done
  内消费 per-variant 路径
- **关联**: Gap-11 (Critic per-variant verdict) — Critic 应当能给单个
  variant verdict, KB 才能记录"variant X 被 Critic 拒, 因此 refuted"
