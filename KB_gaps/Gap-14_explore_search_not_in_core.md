# Gap-14 — `explore_search` 未加入 CORE_STATE_FIELDS

> 严重度: **P2 次要** (防御性 gap, 实际未观察到攻击)
> 主轴影响: **Inv-10.2 Coordinator 单写者**
> 体检报告: `../KB_design_gaps.MD` §6 Gap-14

## 1. 问题描述

KB_design §3.10 §6.2 把 `explore_search` 列为 Coordinator-only 字段.
PolicyGate `CORE_STATE_FIELDS` 应当包含它, 拒绝 LLM `update_state`
直接改 ledger.

实际: `policy.py CORE_STATE_FIELDS` (~300-363) 不含 `explore_search`.

后果:
- 理论上 LLM 可 emit `update_state{changes: {explore_search: {...}}}`
  绕过 Coordinator 直接改 ledger
- Inv-10.2 防御薄一环
- 现实中 LLM prompt 没引导这么做, 攻击面狭小, 但属于"防御不完整"问题

## 2. 现状代码 trace

```
$ rg "explore_search" inference_optimizer/orchestrator/policy.py
(no matches)
```

CORE_STATE_FIELDS 含 phase / cortex / specialist / lane / 等, 但唯独
缺 explore_search.

## 3. 设计意图

§3.10 §6.2:

```
SharedState 写者权限表:
  explore_search / specialist_rounds | Coordinator only
```

`explore_search` 与 `specialist_rounds` 同一行, 后者已在
CORE_STATE_FIELDS 内, 前者漏了.

## 4. 根本原因

§3.10 PR 实施时, CORE_STATE_FIELDS 扩展是手动列字段, **遗漏**
explore_search. 没有人 cross-check §3.10 §6.2 字段表 vs CORE_STATE_FIELDS
实际内容.

## 5. 修复路径

### PR 5.1 — policy.py 加字段

`policy.py CORE_STATE_FIELDS`:

```text
CORE_STATE_FIELDS: frozenset[str] = frozenset({
    ...
    # v0.8 §3.10 §6.2 — explore ledger (KEEP/REVERT history)
    "explore_search",
    # Legacy ledgers (kept for v0.6 resume compat)
    "backends_search",
    "params_search",
    ...
})
```

注: 同时把 `backends_search` / `params_search` 加入 (它们是 Coordinator-
written ledgers, 同样应当 lock; 设计漏掉是因为 v0.6 时代 CORE 概念尚
不完善).

### PR 5.2 — 测试

`tests/test_v08_shared_state_evolution.py` 加:

```text
def test_explore_search_in_core_state_fields():
    from inference_optimizer.orchestrator.policy import CORE_STATE_FIELDS
    assert "explore_search" in CORE_STATE_FIELDS

def test_policy_blocks_llm_explore_search_write():
    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"explore_search": {"tested": {}}}},
    )
    with pytest.raises(PolicyDenied) as e:
        gate.validate_intent("orchestration", intent)
    assert e.value.rule == "state_field"
```

## 6. 验收口径

- [ ] `CORE_STATE_FIELDS` 含 `explore_search`, `backends_search`,
      `params_search`
- [ ] LLM `update_state{changes: {explore_search: ...}}` → policy_denied
- [ ] 现有 `test_v08_shared_state_evolution.py` + `test_v08_drop_scoreboard.py`
      仍全绿

## 7. 风险 / 回退

- 极低风险, 单字段加入
- **回退**: 从 frozenset 移除 explore_search

## 8. 关联 gap

- 与 Gap-09 (gaps[] 字段) 同样应该加入 CORE_STATE_FIELDS
- 独立 PR, 无依赖
