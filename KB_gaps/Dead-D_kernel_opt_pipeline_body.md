# Dead-D — `_KERNEL_OPT_PIPELINE_BODY` 仍把 scoreboard 当真

> 风险等级: **HIGH-misleading** (kernel 阶段每次注入)
> 体检报告: `../KB_design_gaps.MD` §12.5
> 关联: Dead-B (scoreboard 残骸)

## 1. 问题描述

KB_design §3.9 / M2 删除 scoreboard. 但 `prompt_builder.py` 中的
`_KERNEL_OPT_PIPELINE_BODY` 常量 (kernel 阶段开启时注入到 system
prompt 的大段文本) 仍写:

```
"... the scoreboard surfaces `kernel_opt` ..."
"... the scoreboard decides ..."
```

这段文本每个 KERNEL phase 都注入到 orchestration / kernel 角色的 prompt
中. LLM 读到会以为 scoreboard 仍在驱动 kernel_opt.

## 2. 详细位置

`prompt_builder.py:697, 741-744`:

```text
_KERNEL_OPT_PIPELINE_BODY = """\
... lots of text ...

Action selection: the scoreboard surfaces `kernel_opt` when:
  - last KEEP was > 5% gain
  - profile was within the last 3 ticks
  - no kernel_opt for this kernel_id has reached PARTIAL ≤ 2

The scoreboard decides which kernel_id to propose; you should:
  ...
"""
```

`prompt_builder.py:846-847` 注入:

```text
if kernel_enabled and PHASE_KERNEL in current_phases:
    sections.append(_KERNEL_OPT_PIPELINE_BODY)
```

## 3. 矛盾点

同一 prompt 中:
- `prompt_builder.py:493-495` 说 "v0.8 retired the v0.6 Action scores"
- `prompt_builder.py:697-744` 说 "scoreboard surfaces / decides"

LLM 看到内部矛盾, 会以为"scoreboard 被 retire 是指 Action scores top-12
块, 但 kernel 段的 scoreboard 是另一个东西". 实际两者是同一个,
v0.8 都不存在.

## 4. 设计意图

§3.9 §6 描述 v0.8 决策原料替换:

| v0.6 字段 | v0.8 字段 |
|---|---|
| Action scores top-12 | phase + phase_allowed_actions + phase_budget_remaining_pct |
| ... | gaps[] + kb_subgraph_per_gap (Gap-09) |
| ... | last_action_failures + winners_history (事实层) |
| ... | specialist_round_summary |

KERNEL 阶段决策应当基于 phase + gaps + last_kernel_opt + KEEP/REVERT
历史, 不是 scoreboard.

## 5. 修复路径

### PR 5.1 — 重写 `_KERNEL_OPT_PIPELINE_BODY`

```text
_KERNEL_OPT_PIPELINE_BODY = """\
KERNEL phase decision framework (KB_design §3.2 §5.3, §3.9 §6):

Phase contract:
- You are in KERNEL phase, allowed actions: profile (entry only),
  kernel_opt, integrate, deep_kernel_analysis, operator_tuning,
  vendor_kernel_config, recover.
- `last_profile_trace` is your anchor for `select_kernels` (already
  cached at KERNEL entry — Gap-04 auto-profile).

Action selection (no scoreboard — KB_design §3.9 Inv-9.1):
- Read `state.gaps[]` (Gap-09) for kernel-layer gaps.
- Read `last_action_failures` for recent kernel_opt failures.
- Same-kernel_id PARTIAL retry cap = 2 (rejected_kernel_partial_overflow).
- Prefer kernel_opt for unattempted kernel_ids; integrate after KEEP'd
  kernel_opt.

Plateau:
- `plateau_kernel` (3 consecutive REVERTs across distinct kernel_ids)
  → Coordinator transitions to SWEEP.
"""
```

### PR 5.2 — 测试

`tests/test_v08_drop_scoreboard.py` 加:

```text
def test_kernel_opt_body_no_scoreboard_vocab():
    from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
        _KERNEL_OPT_PIPELINE_BODY,
    )
    forbidden = {"scoreboard", "score_mult", "Action scores top-12",
                 "MARATHON_PRIORS", "effective_score"}
    for w in forbidden:
        assert w.lower() not in _KERNEL_OPT_PIPELINE_BODY.lower(), \
            f"forbidden token {w!r} in kernel pipeline body"

def test_kernel_opt_body_references_v08_signals():
    body = _KERNEL_OPT_PIPELINE_BODY
    assert "gaps" in body
    assert "last_action_failures" in body
    assert "plateau_kernel" in body or "KB_design" in body
```

## 6. 验收口径

- [ ] `_KERNEL_OPT_PIPELINE_BODY` 不含 scoreboard / score_mult / Action
      scores 等词汇
- [ ] 含 phase / gaps / last_action_failures / plateau_kernel 等 v0.8
      vocab
- [ ] `test_v08_drop_scoreboard.py` 新增测试全绿

## 7. 风险 / 回退

- 纯 prompt 文本改动, 无功能影响
- **回退**: revert 单 PR

## 8. 关联

- `Dead-B` — 同源 scoreboard 残骸
- Gap-09 — 新 prompt 文本引用 gaps[], 需 Gap-09 字段存在才完整
