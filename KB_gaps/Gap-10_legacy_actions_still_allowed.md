# Gap-10 — EXPLORE phase 仍允许 legacy `backends` / `params` / `validate_stack` 双轨

> 严重度: **P1 主要** (Inv-4.1 入口层未关)
> 主轴影响: **主轴 A (流程固化)**
> 体检报告: `../KB_design_gaps.MD` §5 Gap-10
> 关联死代码: `Dead-A` (executor / yaml 链), `Dead-C` (validate_stack 死路径)

## 1. 问题描述

KB_design §3.4 / §3.15 §2.3: v0.8 把 `backends` / `params` / `validate_stack`
三个 action 合并入单一 `explore` action. legacy 名应当全部移除入口.

实际:
- `phase_state.py:91-97` `PHASE_ALLOWED_ACTIONS[PHASE_EXPLORE]` 仍含
  `backends` / `params` / `validate_stack`
- `actions/_meta/{backends,params,validate_stack}.yaml` 仍存在
- `cli.py:_register_executors` 注册三个 legacy executor
- `prompt_builder.py FULL_ENABLED_ACTIONS` 含三个 legacy 名
- `orchestration.md / critic.md` 把它们列为 EXPLORE 合法 action

后果:
- Inv-4.1 (explore 单 ledger) 在数据层满足 (M3 合并), 但**入口层双轨**
- LLM 可绕过新 `explore` 直接发 `backends`, ledger 仍会被 *两个 executor* 写
- 长期有口径分裂 / 测试锁定 deprecation (Dead-G) 风险

## 2. 现状代码 trace

### 2.1 phase 允许集

`phase_state.py:91-97`:

```text
PHASE_EXPLORE: frozenset({
    # v0.8 EXPLORE merged target (arrives in M3); falls through to
    # the legacy v0.6 action names for the M2 transition.
    "explore", "specialist",
    # v0.6 legacy names still active during M2.
    "backends", "params", "validate_stack",
    "recover",
}),
```

注释自称 "M2 transition", 但 M3 已合, M5/M6/M7 已落, transition 早该
结束.

### 2.2 cli 注册

`cli.py:625-647`:

```text
_REAL_EXECUTORS_FULL = {
    "baseline": baseline_executor,
    "backends": backends_executor,     # ← legacy
    "params":   params_executor,       # ← legacy
    "explore":  explore_executor,
    "sweep":    sweep_executor,
    ...
    "validate_stack": validate_stack_executor,  # ← legacy
    ...
}
```

三处 legacy 注册. 移除会让 LLM 发 backends/params/validate_stack 时
`no_executor` fail.

### 2.3 prompt builder

`prompt_builder.py:59-65`:

```text
FULL_ENABLED_ACTIONS = (
    ...
    "backends", "params", "validate_stack",   # ← legacy 仍可见
    "explore", "specialist",                  # ← v0.8 canonical
    ...
)
```

### 2.4 PolicyGate 缺 `action_deprecated` 规则

§3.13 M3 §PR7 设计了 `action_deprecated` PolicyGate 规则, 实际**没实
现**:

```
$ rg "action_deprecated" inference_optimizer/orchestrator/policy.py
(no matches)
```

## 3. 设计意图

§3.15 §2.3 速查表明确:

```
v0.6 action  | v0.8 行为
backends     | removed → 合并入 explore (M3)
params       | removed → 合并入 explore (M3)
validate_stack | removed → 内嵌进 explore 的 KEEP-后-stack-rebench (M3)
```

§3.13 M3 §PR7: PolicyGate 新增 `action_deprecated` 规则, 命中 deprecated
action 时硬 deny + hint "use 'explore' instead".

## 4. 根本原因

M3 设计要求"灰度期保留 legacy 入口, GA 时关闭". 但 *关闭* 的 PR 没人
合. 设计文档预设的 `--legacy-explore-split` flag (§3.13 §2.3 灰度方
式) 也没实现, 没有任何机制把 legacy 路径关掉.

成因可归纳为:
1. M3 PR9 (delete legacy yaml + executor) 在合入时被 reviewer 推迟,
   理由"想确保 explore 端到端工作再删".
2. 由于 explore 自身存在 Gap-A.5 (promote 缺失), 设计者 *正确* 觉得不
   应删 legacy.
3. 但 Gap-A.5 修复 PR 没跟上, legacy 入口就一直保留.

## 5. 修复路径

### Phase 1 — 先修 Gap-A.5 (explore promote)

**前提**: explore action 端到端工作 (有 `task_kind == "explore"`
promote 分支, 调 `apply_explore_search_update` + `record_explore_accepted`).
见 `Dead-A.5` (12.2 类 A.5) 详述.

### Phase 2 — 关闭 legacy 入口

#### PR 5.1 — phase_state.py 移除 legacy

```text
PHASE_EXPLORE: frozenset({
    "explore", "specialist", "recover",
}),
```

#### PR 5.2 — cli 移除 legacy 注册

`cli.py:_REAL_EXECUTORS_FULL` 删除三个 legacy entries.

#### PR 5.3 — prompt_builder 移除 legacy

`FULL_ENABLED_ACTIONS` / `NO_KERNEL_ENABLED_ACTIONS` 删除三个 legacy 名.

#### PR 5.4 — PolicyGate `action_deprecated` 规则

补 M3 §PR7 的设计:

```text
DEPRECATED_ACTIONS: frozenset[str] = frozenset({
    "backends", "params", "validate_stack",
    # plus optionally: "dream", "re_explore", "comm_optimization", "compiler_tuning"
})

DEPRECATED_ACTION_REPLACEMENTS: dict[str, str] = {
    "backends":       "explore",
    "params":         "explore",
    "validate_stack": "explore (inline rebench)",
}

def _validate_action_not_deprecated(
    self, action_name: str, intent_kind: str,
) -> None:
    if action_name not in DEPRECATED_ACTIONS:
        return
    replacement = DEPRECATED_ACTION_REPLACEMENTS.get(action_name, "explore")
    raise PolicyDenied(
        f"action {action_name!r} is deprecated since v0.8 (M3)",
        rule="action_deprecated",
        hint=f"use {replacement!r} instead; see KB_design §3.4 / §3.15 §2.3",
    )
```

调用点: 在 `_validate_propose_action` / `_validate_delegate` /
`_validate_request` 顶部加 `self._validate_action_not_deprecated(...)`.

#### PR 5.5 — orchestration.md / critic.md 删除 legacy

(详见 `Dead-F`) 三处 system prompt 静态 md 删除 backends/params/validate_stack
提及.

#### PR 5.6 — yaml + executor 物理删除

(详见 `Dead-A`) 删除 yaml + Python 模块.

### Phase 3 — 测试同步重构

(详见 `Dead-G`)

- `tests/test_backends_search.py` → delete
- `tests/test_params_search.py` → delete
- `tests/test_validate_stack.py` → delete
- `tests/test_prompt_builder.py` 相关断言 → 改成断言 explore (不是 backends)
- 新增 `tests/test_v08_action_deprecated_rule.py`:
  - 验证 deprecated action 被 PolicyGate deny
  - 验证 hint 包含 replacement 名

## 6. 验收口径

- [ ] fresh session LLM emit `propose_action='backends'` →
      `policy_denied: action_deprecated`, hint 含 "use 'explore'"
- [ ] phase_state.PHASE_ALLOWED_ACTIONS[EXPLORE] 仅含
      `{explore, specialist, recover}`
- [ ] `cli.py:_REAL_EXECUTORS_FULL` 不含 backends/params/validate_stack
- [ ] `prompt_builder.FULL_ENABLED_ACTIONS` 仅含 explore + specialist (+
      其他非 EXPLORE 阶段的 actions)
- [ ] resume v0.6 session 含 `backends_attempts` 字段不报错 (字段保留,
      只入口关)
- [ ] M3 验收清单第 1 条 (`breakdown.capability_summary` 只见 explore 行
      作主行) 成立

## 7. 风险 / 回退

- **Critic 评审基于 legacy action 名**: critic prompt 现在 (orchestration.md
  / critic.md) 仍按 backends/params/validate_stack 分组. 必须同时修 (PR 5.5).
- **resume 老 session 仍想跑 backends 任务**: resume 时 `state.json`
  内可能有 `backends_attempts` ledger. ledger 字段保留不动 (Inv-10.1),
  但新 LLM 发 backends 会被 deny. 用户可手动迁移到 explore.
- **回退**: 改回 phase_state allowlist + cli 注册 + 移除
  `action_deprecated` 规则即退到当前状态.

## 8. 关联 gap

- **必须先做**: `Dead-A.5` (explore promote 完整, 见 Dead-A) — 没有这
  个, 关 legacy 入口 = 整个 EXPLORE 阶段无可用 grid runner
- **同步做**: `Dead-A` (yaml + executor 物理删), `Dead-C` (validate_stack
  死路径), `Dead-F` (静态 md), `Dead-G` (测试)
- **关联**: Gap-11 (Critic per-variant verdict_map) — Critic 协作面同步改
