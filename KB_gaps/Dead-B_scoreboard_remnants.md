# Dead-B — scoreboard 残骸 (stub 方法 + prompt 残句 + denial hint)

> 风险等级: **HIGH-misleading** (LLM 误以为评分系统还在)
> 体检报告: `../KB_design_gaps.MD` §12.3
> 关联: §3.9 (砍 scoreboard) 落地完成, 但残骸未清

## 1. 问题描述

KB_design §3.9 / M2 明确删除 "action_scores / scoring.py / MARATHON_PRIORS
/ Action scores prompt 块". 模块层面已删 (M2 commit `7917257`), 但残骸:

1. Coordinator 保留一组 `_score_action_*` no-op stub 方法, **仍被 5 处生
   产代码调用**, 形成"看着像在 scoring, 实际无效"的迷惑
2. `_ensure_action_scores_seeded` 在 `__init__` 中被调 (no-op stub)
3. `all_top_actions_policy_locked` 永远返回 False 的死分支
4. **Live policy denial hint 仍说 "pick another action from Action scores"** ← 这是 LLM 每次 deny 都看见的字符串
5. `_KERNEL_OPT_PIPELINE_BODY` 残留 "scoreboard surfaces / decides" 句
   子 (Dead-D 详述)
6. prompt_builder 提到 `backends_search` / `params_search` (Dead-A 关联)

对 LLM 误导:
- 阅读 coordinator 看到大量 `_score_action_keep` / `_apply_action_score_update`
  调用 → 以为 scoring 仍在
- 阅读 denial hint → 找不到 "Action scores" 数据源, 困惑

## 2. 详细位置清单

### B.1 stub 方法

`coordinator.py:601-641`:

```text
def _score_action_keep(self, action_name: str) -> None:
    return None  # v0.8 §3.9 stub

def _score_action_failure(self, action_name: str) -> None:
    return None

def _score_action_no_promote(self, action_name: str) -> None:
    return None

def _score_action_lock(self, action_name: str, reason: str) -> None:
    return None

def _apply_action_score_update(self, ...) -> None:
    return None
```

调用方 (5 处):

```text
coordinator.py:2556  self._score_action_keep("kernel_opt")
coordinator.py:2591  self._score_action_no_promote("integrate")
coordinator.py:3667  self._apply_action_score_update(task_kind, result, ...)
coordinator.py:3814  self._apply_action_score_update(task_kind, result, ...)
coordinator.py:3865  self._apply_action_score_update(task_kind, result, ...)
```

### B.2 `_ensure_action_scores_seeded` 在 `__init__`

`coordinator.py:421-426`:

```text
# Resume detection runs BEFORE we seed action_scores so a fresh
# session is not misdetected as a resume (seeding writes state.json
# which the resume probe treats as evidence of an existing session).
self._resumed_from = self._detect_resume_state()
# Seed per-action scoring now that resume status is locked in.
self._ensure_action_scores_seeded()
```

stub no-op, 但调用点 + 注释暗示 scoring 仍在工作.

### B.3 死分支

`shared_state.py:1357-1365`:

```text
def all_top_actions_policy_locked(self, registry=None) -> bool:
    return False  # v0.8 §3.9 stub
```

`coordinator.py:1013-1018` `_has_no_more_leverage` 的一个分支:

```text
if self.shared_state.all_top_actions_policy_locked(self.action_registry):
    return True
```

永远不进入. 但函数 `_has_no_more_leverage` 整体仍用 `params_search` /
`backends_search` 做 plateau 判定 (M3 explore_search 双轨).

### B.4 Live denial hint

`coordinator.py:2292-2295`:

```text
return PolicyDeniedSummary(
    rule="phase_incompatible",
    hint=f"you can't propose {action!r} in phase={phase!r}; "
         f"pick another action from Action scores top-12",  # ← 死提示
)
```

LLM 每次 R1 deny 都看见此字符串.

### B.5 `_KERNEL_OPT_PIPELINE_BODY` (详见 Dead-D)

### B.6 prompt_builder ledger 名

`prompt_builder.py:600-614` 描述 ledger 段时用 `backends_search` /
`params_search`, v0.8 canonical 是 `explore_search`.

## 3. 根本原因

§3.9 / M2 PR 删 scoring.py 时, *保留 stub 方法* 是一种 backward-compat
策略, 避免引入大量 import 错误. 但**调用方应当一起删**, 没人跟进.

denial hint (B.4) 是历史遗留字符串, M2 PR 没注意到.

## 4. 修复路径

### PR 4.1 — 删除 stub 方法

`coordinator.py:601-641` 5 个 stub 方法物理删除.

### PR 4.2 — 删除调用方

5 处调用 (~2556, 2591, 3667, 3814, 3865) 物理删除. 这些都是 promote/
result 处理后的 "scoreboard 更新" 调用, 删除后无副作用.

### PR 4.3 — 删除 `_ensure_action_scores_seeded`

`coordinator.py:421-426` 删除调用 + stub 方法.

### PR 4.4 — 修复 denial hint

`coordinator.py:2292-2295`:

```text
hint=(
    f"action {action!r} is not allowed in phase={phase!r}; "
    f"the allowed set is {allowed_actions!r}. See KB_design §3.2."
)
```

### PR 4.5 — 修复 `all_top_actions_policy_locked` 死分支

`shared_state.py:1357-1365` 删除函数; `coordinator.py:1013-1018` 删除
调用分支.

### PR 4.6 — `_has_no_more_leverage` 用 explore_search

`coordinator.py:1002-1074` 内部 plateau 判定改用 `state.explore_search.winners_history`
取代 `backends_search` / `params_search`. 与 §3.8 / M7 真 plateau 函数
一致.

### PR 4.7 — prompt_builder ledger 名

`prompt_builder.py:600-614` 把 `backends_search` / `params_search` 改
为 `explore_search`.

### PR 4.8 — 测试

`tests/test_v08_drop_scoreboard.py` 已有大量回归测试 (M2 PR 落). 加:

```text
def test_score_stub_methods_removed():
    """v0.8 §3.9 — no scoring stub methods on Coordinator."""
    from inference_optimizer.orchestrator.coordinator import Coordinator
    for name in ("_score_action_keep", "_score_action_failure",
                 "_score_action_no_promote", "_score_action_lock",
                 "_apply_action_score_update",
                 "_ensure_action_scores_seeded"):
        assert not hasattr(Coordinator, name), f"{name} should be deleted"

def test_denial_hint_no_scoreboard_vocab():
    # ... emit a phase_incompatible deny, check hint string doesn't contain
    # "Action scores", "scoreboard", "score", etc.
```

## 5. 验收口径

- [ ] `coordinator.py` 全文 `grep -n "_score_action\|_apply_action_score\|_ensure_action_scores_seeded"` 0 命中
- [ ] `coordinator.py` 全文 `grep -ni "action scores\|scoreboard"` 仅历
      史注释 (引用 §3.9 退场) 命中
- [ ] denial hint 不含 "Action scores"
- [ ] `all_top_actions_policy_locked` 不存在 (函数被删)
- [ ] `_has_no_more_leverage` 内部用 `explore_search`, 不用
      `backends_search` / `params_search`

## 6. 风险 / 回退

- 全部死代码清理, 无功能影响
- **回退**: revert 单 PR 即可 (各 PR 独立, 互不依赖)

## 7. 关联

- `Dead-D` (KERNEL_OPT_PIPELINE_BODY scoreboard 残句) — 同源
- `Dead-A` (legacy ledgers in prompt_builder) — 同源
- Gap-15 (plateau proxy 双轨) — PR 4.6 涉及
