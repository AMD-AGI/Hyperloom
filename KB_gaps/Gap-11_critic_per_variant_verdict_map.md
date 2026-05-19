# Gap-11 — Critic Review 仍 per-proposal verdict, 未升 per-variant `verdict_map`

> 严重度: **P1 主要** (协作面大改动)
> 主轴影响: **主轴 A (流程固化, partial KEEP 场景)**
> 体检报告: `../KB_design_gaps.MD` §5 Gap-11

## 1. 问题描述

KB_design §3.5 §5 / M5 §5 step 5: "Critic 收到一组 K variant +
judge_bundle (含 KB priors 各 variant); 返回 verdict map {variant_name
→ verdict}".

实际:
- `intent_parser.py:115` `IntentType.REVIEW_VERDICT` payload 字段:
  `("target_proposal_msg_id", "verdict")` — `verdict` 单数, 单值
- Critic agent prompt 模板基于 v0.6 review-one-proposal 模式
- 一组 K variant 收到一个 verdict, 整批 KEEP 或整批 REVERT

后果:
- explore round partial-success (3 KEEP + 2 REVERT) 场景下无法精确传
  verdict
- 退化为"一刀切批准/驳回", 大部分 partial-success 被 critic 误判
- KB 边 (Gap-07/08) 也无法精确记录"variant X 被 critic 拒"

## 2. 现状代码 trace

### 2.1 IntentType payload schema

`intent_parser.py:107-130`:

```text
INTENT_REQUIRED_FIELDS: dict[IntentType, tuple[str, ...]] = {
    ...
    IntentType.REVIEW_VERDICT: ("target_proposal_msg_id", "verdict"),
    ...
}
```

`verdict` 是单一字段 (string: "KEEP" / "REVERT" / "NEEDS_INFO").

### 2.2 Critic prompt

`critic.md` + critic_prompt_builder:

```text
"返回 verdict ∈ {KEEP, REVERT, NEEDS_INFO}"
```

只要求单 verdict.

### 2.3 Coordinator `_handle_review_verdict`

```text
async def _handle_review_verdict(self, source, intent) -> None:
    verdict = intent.payload.get("verdict")  # single value
    ...
    if verdict == "KEEP":
        # dispatch entire proposal
        ...
    elif verdict == "REVERT":
        # skip entire proposal
        ...
```

整批二选一.

## 3. 设计意图

§3.5 §5 / M5 §5 step 5:

```
[Critic Review]
  ├ 收到一组 K variant + judge_bundle (含 KB priors 各 variant)
  ├ 返回 verdict map {variant_name → verdict}
  └ Coordinator 处理 verdict → 把 approved 的 variant 通过 delegate
    → ExploreExecutor
```

设计目的:
- specialist 提议的 K variants 可能质量参差; Critic 应当能 *选择性*
  批准
- 与 KB negation 联动: critic 拒绝的 variant 触发 KB refuted edge
- Reduce explore round 内的浪费 bench (REVERT variant 不跑)

## 4. 根本原因

REVIEW_VERDICT 协议是 v0.6 设计 (DESIGN §18.2, 单 proposal 单 verdict).
v0.8 §3.5 升级到 batch + verdict_map, 但 intent_parser 的 schema 没改,
critic 提示也没改.

M5 PR 链中, 升级 critic 协作面是 §PR4 (新 SpecialistRunner) 的下游,
本应有 PR 8 (critic 协作升级), 实际没合.

## 5. 修复路径

### PR 5.1 — IntentType.REVIEW_VERDICT payload schema 扩展

`intent_parser.py`:

```text
INTENT_REQUIRED_FIELDS = {
    ...
    IntentType.REVIEW_VERDICT: ("target_proposal_msg_id",),  # verdict 改可选
}

INTENT_OPTIONAL_FIELDS = {
    ...
    IntentType.REVIEW_VERDICT: ("verdict", "verdict_map", "rationale"),
}

# 校验逻辑: verdict 或 verdict_map 必须有一个 (mutually exclusive)
def _validate_review_verdict_payload(payload):
    has_single = "verdict" in payload
    has_map = "verdict_map" in payload and isinstance(payload["verdict_map"], dict)
    if not has_single and not has_map:
        raise SchemaError("REVIEW_VERDICT must include 'verdict' or 'verdict_map'")
    if has_single and has_map:
        raise SchemaError("REVIEW_VERDICT: 'verdict' and 'verdict_map' are mutually exclusive")
```

### PR 5.2 — Critic prompt 升级

`critic_prompt_builder.py` 在 verdict 段加:

```text
"""
## OUTPUT PROTOCOL

If the proposal you reviewed contains a `grid` of K variants (explore
action), return per-variant verdicts via:

  emit_intent{type='review_verdict', payload={
    target_proposal_msg_id: '<id>',
    verdict_map: {
      'variant_name_A': {verdict: 'KEEP', rationale: '...'},
      'variant_name_B': {verdict: 'REVERT', rationale: 'KB shows similar tried 3x, all failed'},
      ...
    },
  }}

For single-action proposals (kernel_opt / integrate / ...), keep using
the legacy single-verdict form:

  emit_intent{type='review_verdict', payload={
    target_proposal_msg_id: '<id>',
    verdict: 'KEEP',
    rationale: '...',
  }}
"""
```

### PR 5.3 — `_handle_review_verdict` per-variant 分支

```text
async def _handle_review_verdict(self, source, intent) -> None:
    payload = intent.payload
    proposal = self.pending_proposals.get(payload["target_proposal_msg_id"])
    if not proposal:
        return

    verdict_map = payload.get("verdict_map")
    if verdict_map and proposal.action_name == "explore":
        await self._handle_verdict_map(proposal, verdict_map)
        return

    # Legacy: single verdict path (non-explore proposals)
    await self._handle_single_verdict(proposal, payload.get("verdict"))


async def _handle_verdict_map(
    self, proposal: PendingProposal, verdict_map: dict,
) -> None:
    """Per-variant verdict handling (KB_design §3.5 §5 / M5 §5 step 5)."""
    original_grid = proposal.proposal_intent.payload.get("grid", [])
    approved: list[dict] = []
    rejected: list[dict] = []

    for variant in original_grid:
        vname = variant.get("name", "")
        v = verdict_map.get(vname, {})
        verdict = v.get("verdict", "NEEDS_INFO")
        if verdict == "KEEP":
            approved.append(variant)
        elif verdict == "REVERT":
            rejected.append({
                **variant,
                "critic_rationale": v.get("rationale", ""),
            })
            # Trigger KB refuted edge for this variant (Gap-08)
            await self._cortex_t3_critic_rejected(proposal, variant)

    if not approved:
        log.info("verdict_map: all %d variants rejected by critic",
                 len(original_grid))
        await self._mark_proposal_rejected(proposal, reason="critic_rejected_all")
        return

    # Dispatch only approved variants
    new_payload = dict(proposal.proposal_intent.payload)
    new_payload["grid"] = approved
    new_payload["critic_filtered_count"] = len(rejected)
    await self._dispatch_proposal(proposal, payload_override=new_payload)
```

### PR 5.4 — `_cortex_t3_critic_rejected`

Critic 拒绝的 variant 也应当走 T3 refuted (而非等 explore 跑后才知道):

```text
async def _cortex_t3_critic_rejected(
    self, proposal: PendingProposal, variant: dict,
) -> None:
    """Variant rejected at critic stage → KB refuted edge."""
    edge_id = proposal.kb_edge_ids.get(variant["name"])  # Gap-07 dependency
    if not edge_id or not self.cortex_kb:
        return
    await self.cortex_kb.verify(
        edge_id=edge_id,
        outcome="refuted",
        promote=None,
        attrs={
            "variant_name": variant["name"],
            "rejection_stage": "critic",
            "rationale": variant.get("critic_rationale", ""),
        },
    )
```

### PR 5.5 — Critic test fixtures

`tests/test_v08_critic_verdict_map.py`:

- 构造 grid 4 variants
- Critic emit verdict_map: 2 KEEP, 1 REVERT, 1 NEEDS_INFO
- 验证 dispatch 后 grid 只含 2 variants (approved)
- 验证 Cortex verify 被调 1 次 (REVERT, outcome=refuted)
- 验证 explore round 完成后 NEEDS_INFO variant 不再 dispatch
- 验证 single-verdict path 仍工作 (kernel_opt 等非 grid)

## 6. 验收口径

- [ ] Critic 对 4-variant explore round emit `verdict_map` → 仅 approved
      variants 进入 explore executor
- [ ] REVERT variant 触发 KB verify outcome='refuted' (无需等 explore
      run)
- [ ] kernel_opt / integrate 等非 grid action 仍走单 verdict 路径
- [ ] PolicyGate 拒绝 verdict_map 中 variant_name 不在原 grid 内的项
      (schema 校验)
- [ ] breakdown.critic_robustness.kb_writes_summary 按 verdict 分组准确

## 7. 风险 / 回退

- **协作面大改动**: Critic agent 是另一个 LLM 角色, 提示升级后 critic
  端可能输出旧格式. 缓解: schema validator 同时接受新旧两种 format,
  逐步过渡.
- **Critic 性能瓶颈** (§3.14 R-07): per-variant 评审 token 量上升. 缓
  解: judge_bundle 已经按 variant 切分, prompt 只多 K 行 verdict 输出,
  不显著.
- **回退**: critic_prompt_builder 改回单 verdict 提示; coordinator 走
  `_handle_single_verdict` (verdict_map 始终走旧路径). Schema validator
  仍接受新格式但不强制.

## 8. 关联 gap

- **依赖**: Gap-07 (T2 per-variant) — verdict_map 用 variant_name 索引
  kb_edge_ids
- **同期**: Gap-08 (T3 per-variant verify) — critic rejected 也走 T3
  refuted
- **关联**: Gap-10 (legacy backends/params 关闭) — critic 协作面整体
  升级时一并删除 legacy action 提及
