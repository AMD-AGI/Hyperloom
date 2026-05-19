# Gap-09 — `SharedState.gaps[]` 字段不存在; prompt 用 proxy

> 严重度: **P1 主要** (LLM 决策依据降级)
> 主轴影响: **主轴 A (流程固化) + 主轴 C (specialist 派发)**
> 体检报告: `../KB_design_gaps.MD` §5 Gap-9

## 1. 问题描述

KB_design §3.3 / §3.5 / §3.9 §6 把 `gaps[]` 列为 Orchestration prompt
的核心决策原料字段, 也是 specialist domain selection 的主要 anchor:

```
gaps[] — 当前未解决的瓶颈
  字段: canonical_id + symptom + layer + 已尝试历史
```

实际:
- `shared_state.py` 全文**无 `gaps:` dataclass 字段**
- `prompt_builder.py:500-504` 注释明说: 暂用 `last_action_failures` +
  `explore_search.winners_history` 作 proxy
- M2 milestone 仍把 gaps 留作 future work

后果:
- LLM 决策依据降级到字面 failure 历史 (字符串列表), 失去结构化锚点
- specialist domain selection (§3.5 §11) 失去"按 gap layer 派 domain"
  的能力
- KB negation 边反向作用 (把已 refuted 的 gap 从视野滤掉) 失效
- Cortex `traverse(issue_node)` 拉回的 sub-graph 没地方放

## 2. 现状代码 trace

### 2.1 字段不存在

```
$ rg "^    gaps:|gaps: list\[" inference_optimizer/orchestrator/shared_state.py
(no matches)
```

### 2.2 prompt_builder proxy

`prompt_builder.py:500-504`:

```text
# v0.8 §3.9 — 'gaps[]' field is planned but not yet on SharedState.
# Until then, use the proxies:
#   - last_action_failures (recent REVERT events)
#   - explore_search.winners_history (cumulative KEEP velocity)
# When §3.6 fills gaps[], replace this block with structured gap list.
```

### 2.3 specialist prompt 也用 proxy

`specialist_runner.py:323-327`:

```text
gap_id = params.get("gap")  # canonical_id, from delegate.params
```

specialist 拿到 `gap_id` (字符串 canonical_id), 但 SharedState 里没有
对应的 `gaps[]` 项可以反查 symptom / layer / 尝试历史.

## 3. 设计意图

§3.3 §4.2 / §3.5 §11:

```
gaps: list[Gap] where Gap = {
    canonical_id: str,           # issue_node canonical from Cortex
    symptom: str,                # human-readable: "MoE comm overhead"
    layer: enum,                 # comm / kernel / framework / param
    attempts: list[Attempt],     # 历史尝试 (action + outcome)
    severity: enum,              # high / medium / low
    domain_hint: str,            # which specialist domain best fits
}
```

设计目的:
- specialist 派发: "framework_specialist 负责 layer=framework 的 gap"
- KB negation: 已 refuted 的 gap 在下次 traverse 时过滤
- Critic review: judge_bundle 包含每个 gap 的 attempts 历史

## 4. 根本原因

§3.6 (KnowledgePlane) 章节预设 `gaps[]` 由 KnowledgePlane.cortex_traverse
填充, 但 M4 PR 只接了 KnowledgePlane 类, 没接到 Coordinator. M2 PR
focus 在 phase 字段, 没人负责 gaps[].

设计文档跨 §3.3 / §3.5 / §3.6 / §3.9 都引用 gaps[], 但没有一份文档明
确写 "gaps[] 的写者是谁, 在什么时机刷新". 结果是 *谁都用, 谁都不写*.

## 5. 修复路径

### PR 5.1 — SharedState 加字段

`shared_state.py`:

```text
@dataclass
class SharedState:
    ...
    # v0.8 §3.3 / §3.5 — current unresolved gaps. Coordinator-only
    # writes; LLM agents read via prompt. Refreshed by:
    #   - baseline completion (initial symptom extraction)
    #   - explore round KEEP/REVERT (gap.attempts append)
    #   - Cortex traverse on issue_node (cross-session priors)
    gaps: list[dict[str, Any]] = field(default_factory=list)
```

### PR 5.2 — PolicyGate lock

`policy.py CORE_STATE_FIELDS` 加 `"gaps"`.

### PR 5.3 — Coordinator `_refresh_gaps` (3 入口)

```text
async def _refresh_gaps(self, *, reason: str) -> None:
    """KB_design §3.3 — refresh gaps[] from observable signals.

    Called at:
    1. baseline completion (initial extraction)
    2. EXPLORE round KEEP/REVERT (update attempts)
    3. Cortex traverse periodic refresh (cross-session priors)
    """
    state = self.shared_state
    new_gaps: list[dict] = []

    # 1. From baseline metrics (initial)
    if reason == "baseline_done":
        new_gaps.extend(self._extract_gaps_from_baseline())

    # 2. From recent KEEP/REVERT events
    new_gaps.extend(self._extract_gaps_from_attempts())

    # 3. From Cortex KB
    if self.knowledge_plane:
        try:
            for gap in await self.knowledge_plane.cortex_traverse_issues(
                model_class=state.model_class,
                gpu_type=state.gpu_type,
            ):
                new_gaps.append(self._merge_gap(gap))
        except Exception as e:
            log.warning("gaps refresh from Cortex failed: %s", e)

    # Deduplicate by canonical_id
    state.gaps = self._dedupe_gaps(new_gaps)
    log.info("gaps refreshed (reason=%s): %d gaps", reason, len(state.gaps))
```

### PR 5.4 — Coordinator 调 `_refresh_gaps`

3 个入口:

1. baseline 完成: `_promote_to_shared_state[task_kind=='baseline']`
   末尾.
2. EXPLORE phase 内每轮 explore 结果: `_handle_specialist_done` 或
   explore promote 后.
3. EXPLORE phase entry hook (Gap-04 框架内): pr_feed 后一并刷新 gaps.

### PR 5.5 — prompt_builder 渲染 gaps[]

`prompt_builder.py:500-504` 删除 proxy 注释, 改为渲染 `state.gaps`:

```text
def _render_gaps_section(state) -> str:
    if not state.gaps:
        return "(no gaps surfaced yet — fresh session)"
    lines = ["=== Current gaps (newest first) ==="]
    for gap in state.gaps[-10:]:  # cap at 10 for prompt size
        lines.append(
            f"  - {gap['canonical_id']} [{gap['layer']}/{gap['severity']}]: "
            f"{gap['symptom']!r} (attempts={len(gap.get('attempts', []))})"
        )
    return "\n".join(lines)
```

### PR 5.6 — specialist runner 消费

specialist task params 已经接收 `gap_canonical_id`. Coordinator 派发
specialist 前 (Gap-01 PR 5.4 hook) 同时从 `state.gaps` 取对应 gap 的
symptom / layer, 写到 task.params:

```text
params.setdefault("gap_symptom", gap.get("symptom"))
params.setdefault("gap_layer", gap.get("layer"))
params.setdefault("gap_attempts", gap.get("attempts", [])[-5:])
```

### PR 5.7 — 测试

`tests/test_v08_gaps_field.py`:

- 验证字段存在 + 默认 []
- 验证 baseline 完成后 gaps 非空
- 验证 explore REVERT 后对应 gap.attempts 增长
- 验证 dedupe by canonical_id
- PolicyGate update_state{changes={gaps: ...}} → denied (lock 工作)

## 6. 验收口径

- [ ] fresh session baseline 完成后, state.gaps 至少 1 项
- [ ] EXPLORE round REVERT 后, 对应 gap.attempts 列表增长
- [ ] Orchestration prompt 含 `=== Current gaps ===` 段
- [ ] specialist task params 含 gap_symptom / gap_layer / gap_attempts
- [ ] PolicyGate 拒绝 LLM update_state{gaps: ...}
- [ ] breakdown 中 attribution.phase_breakdown.explore.by_gap_layer
      非空 (Inv-12.2)

## 7. 风险 / 回退

- **gaps[] 长度爆炸**: 长 session 内 KEEP/REVERT 多 → gap list 长. 缓
  解: `_dedupe_gaps` 用 canonical_id 去重 + 最多保留 50 项 + 长 prompt
  渲染时 cap 10.
- **Cortex traverse 失败**: PR 5.3 的 step 3 已 try/except, 失败时仅
  step 1+2 工作 (本 session 内的 gaps).
- **gap_canonical_id 不稳定**: 同一 symptom 在不同 session 取的
  canonical 可能不同. 这是 Cortex 一侧的 schema 问题 (R-13), 不在本 gap
  范围.
- **回退**: 删除 _refresh_gaps 调用, prompt 退回 proxy.

## 8. 关联 gap

- **解锁**: Gap-01/03 (specialist 派发用 gaps[] 路由 domain)
- **依赖**: Gap-02 (KnowledgePlane bootstrap, traverse 来源)
- **关联**: Gap-07 `_resolve_issue_canonical` 优先从 gaps[] 取 anchor
