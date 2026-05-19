# Gap-07 — T2 hypothesize 仍是 per-proposal, 未升级到 per-variant

> 严重度: **P1 主要** (KB 视图层粒度不足)
> 主轴影响: **主轴 B (知识外接 Cortex)**
> 体检报告: `../KB_design_gaps.MD` §5 Gap-7

## 1. 问题描述

KB_design §3.13 M5 §5 step 6: "for each variant in grid: propose-point
optimization_node ... hypothesize hypothetical edge issue_node→opt_node
attrs={variant: ...} ... PendingProposal.kb_edge_ids[variant.name] =
tentative_edge_id".

实际:
- `coordinator._cortex_t2_hook` (~2050-2124) 每 PendingProposal 写一条
  propose_point + 一条 hypothesize 边, 用 `proposal_msg_id` 作 anchor.
- `PendingProposal.kb_edge_id: str` (~218) — **单 edge_id, 非 per-variant
  map**.

后果:
- KB 跨 session 检索时无法定位"具体哪个 variant confirmed/refuted"
- specialist confidence 评估缺粒度
- Critic per-variant verdict (Gap-11) 也无法对应到 KB 边

## 2. 现状代码 trace

### 2.1 PendingProposal 单 edge_id

`coordinator.py:204-222`:

```text
@dataclass
class PendingProposal:
    msg_id: str
    proposal_intent: Intent
    action_name: str
    ...
    kb_edge_id: str = ""   # ← 单 ID, 非 dict
    ...
```

### 2.2 T2 hook per-proposal

`coordinator.py:2050-2124`:

```text
async def _cortex_t2_hook(self, proposal: PendingProposal) -> None:
    ...
    # Build canonical_id from proposal_msg_id (not per-variant)
    canonical = f"opt.session-{sid}.proposal-{proposal.msg_id}"
    outcome = self.cortex_kb.propose_point(
        canonical_id=canonical,
        kind="optimization_node",
        attrs={...},
    )
    edge_outcome = self.cortex_kb.hypothesize(
        from_node=issue_canonical,
        to_node=canonical,
        attrs={...},
    )
    proposal.kb_edge_id = edge_outcome.edge_id   # 单 ID
```

无论 grid 有几个 variant, 只写一条边.

### 2.3 explore executor 已经支持 per-variant kb_edge_id

`explore.py:129, 157`:

```text
for variant in grid:
    kb_edge_id = variant.get("kb_edge_id", "")  # 读取但 t2 hook 不填
    ...
```

支持读 per-variant `kb_edge_id`, 但因 T2 hook 不填, 永远是空.

## 3. 设计意图

§3.5 §10 / M5 §5 step 6:
- 每个 variant 独立 hypothetical edge → 跨 session 知识可定位"variant X
  refuted, variant Y confirmed"
- KB negation 边可以精确指向 *一个* 失败的 variant, 而不是"整组都失败"
- Critic per-variant verdict (Gap-11) 依赖此粒度

## 4. 根本原因

M3 (explore 合并) 落地时, T2 hook 沿用 M1 per-proposal 设计, 没升级.
M5 设计文档里写了"per-variant 升级", 但 M5 实施时焦点在 SpecialistRunner
(Gap-01), 没人接 T2 hook 改造.

附加: explore.py 在 PR 阶段已经接收 per-variant kb_edge_id, 这暗示当
时设计者**预期** T2 hook 在另一 PR 改, 但那个 PR 没合.

## 5. 修复路径

### PR 5.1 — PendingProposal 数据结构升级

`coordinator.py` 改:

```text
@dataclass
class PendingProposal:
    ...
    # v0.8 M5 — per-variant KB edge IDs (KB_design §3.13 M5 §5 step 6).
    # Keyed by variant_name; empty dict for non-explore proposals.
    kb_edge_ids: dict[str, str] = field(default_factory=dict)

    # Backwards-compat: single kb_edge_id for non-grid proposals.
    # Kept as @property reading from kb_edge_ids when len==1.
    @property
    def kb_edge_id(self) -> str:
        if len(self.kb_edge_ids) == 1:
            return next(iter(self.kb_edge_ids.values()))
        return ""
```

### PR 5.2 — T2 hook 遍历 variants

```text
async def _cortex_t2_hook(self, proposal: PendingProposal) -> None:
    if not self.cortex_kb:
        return
    sid = self.shared_state.cortex_session_id
    if not sid:
        return

    action = proposal.action_name
    payload = proposal.proposal_intent.payload or {}
    grid = payload.get("grid") or []

    if action != "explore" or not grid:
        # Non-grid proposal (kernel_opt / integrate / etc.) — single edge
        await self._t2_hypothesize_single(proposal)
        return

    # M5 per-variant: one optimization_node + one hypothesize edge per variant
    issue_canonical = self._resolve_issue_canonical(proposal)
    for variant in grid:
        variant_name = variant.get("name", "")
        if not variant_name:
            continue
        canonical = (
            f"opt.session-{sid}.proposal-{proposal.msg_id}.variant-{variant_name}"
        )
        try:
            self.cortex_kb.propose_point(
                canonical_id=canonical,
                kind="optimization_node",
                attrs={
                    "action": action,
                    "variant_name": variant_name,
                    "extra_sglang_args": variant.get("extra_sglang_args"),
                    "extra_envs": variant.get("extra_envs"),
                    "domain": variant.get("domain", ""),  # from specialist
                    "round": proposal.round_id,
                },
            )
            edge = self.cortex_kb.hypothesize(
                from_node=issue_canonical,
                to_node=canonical,
                attrs={
                    "phase": self.shared_state.phase,
                    "domain": variant.get("domain", ""),
                    "round": proposal.round_id,
                    "specialist_task_id": variant.get("specialist_task_id", ""),
                },
            )
            proposal.kb_edge_ids[variant_name] = edge.edge_id
        except Exception as e:
            log.warning("T2 hypothesize failed for variant %s: %s", variant_name, e)
            # NDJSON 兜底已经在 cortex_kb_client 内部处理
```

### PR 5.3 — `_resolve_issue_canonical` helper

```text
def _resolve_issue_canonical(self, proposal: PendingProposal) -> str:
    """Find the issue_node canonical_id this proposal addresses.

    Sources, in order:
    1. proposal_intent.payload['gap_canonical_id'] (explicit)
    2. SharedState.gaps[i].canonical_id matching domain (Gap-09)
    3. Fallback: session-level synthetic issue
    """
    payload = proposal.proposal_intent.payload or {}
    explicit = payload.get("gap_canonical_id")
    if explicit:
        return str(explicit)
    # ... gaps[] fallback once Gap-09 lands
    return f"issue.session-{self.shared_state.cortex_session_id}.synthetic"
```

### PR 5.4 — explore.py 读 per-variant kb_edge_id

无需改动 (已经支持). 只需确保 grid 中携带 `kb_edge_id`:

PR 5.5 (下面) `_promote_to_shared_state` 在 PartialResult 内回填.

### PR 5.5 — 测试

`tests/test_v08_t2_per_variant.py`:

- mock cortex_kb.propose_point + hypothesize
- 构造 grid 4 variants, 派 propose_action='explore'
- 验证 propose_point 被调 4 次 + hypothesize 4 次
- 验证 PendingProposal.kb_edge_ids 长度 4
- 验证每个 variant 的 canonical_id 包含 variant_name

## 6. 验收口径

- [ ] fresh session 派 1 个 4-variant explore round 后, Cortex 中有 4
      条 hypothetical edges (而非 1 条)
- [ ] kb_provenance.points_created 长度 = 4 × #rounds
- [ ] 每条边 attrs.variant 字段非空
- [ ] PendingProposal.kb_edge_ids[variant_name] 在 promote 时可用

## 7. 风险 / 回退

- **Cortex 写入压力**: 一次 4-variant round 从 1 edge 变 4 edges, 量级
  4×. 对小规模 session 无影响; 对极端长 session (1000+ rounds) 累积
  edges 可能压垮 Cortex 短期容量. 缓解: §3.14 R-01 NDJSON 兜底已经覆
  盖.
- **canonical_id 长度**: variant_name 加入 canonical 后 ID 变长.
  Cortex 限制是 256 chars, variant_name 不太可能超 100 chars, 安全.
- **回退**: 改回单 `kb_edge_id`, PendingProposal.kb_edge_ids 退到 dict
  长度 ≤ 1.

## 8. 关联 gap

- **同期**: Gap-08 (T3 per-variant verify) — 必须一起改, 否则单 edge
  vs 多 edges 不对称
- **依赖**: Gap-03 (specialist_done 路由) 在 PR 5.4 内消费 kb_edge_ids
  写回 proposal_set
- **依赖**: Gap-09 (`gaps[]` 字段) — `_resolve_issue_canonical` 优先从
  gaps[] 取
