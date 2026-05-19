# Gap-15 — Phase budget proxy 与真 plateau 口径不一致 (R-09 双轨)

> 严重度: **P2 次要** (设计明知, 灰度期问题)
> 主轴影响: **主轴 A (plateau 判定)**
> 体检报告: `../KB_design_gaps.MD` §6 Gap-15
> 关联风险: `../KB_design/3.14_risks/README.md` R-09

## 1. 问题描述

KB_design §3.8 §6 / M7 已实装真 plateau 函数 (基于 explore_search +
specialist_rounds 的统计). 但 legacy resume 路径仍依赖
`params_no_promote_streak` proxy.

灰度期短暂双轨, 操作员 confusion. M7 完成后真 plateau 在 fresh
session 有效, 但 v0.6 resume 走 proxy.

## 2. 现状代码 trace

`phase_state.py:exit_normal_explore` 已升级到真 plateau:

```text
def exit_normal_explore(state, ...) -> bool:
    # M7 true plateau (KB_design §3.8 §6)
    if compute_plateau_explore(state, ...).should_exit:
        return True
    # Legacy fallback for resumed v0.6 sessions
    params_streak = int(getattr(state, "params_no_promote_streak", 0) or 0)
    if params_streak >= 5:
        return True
    return False
```

两条出口口径并存:
- M7 真 plateau: 基于 explore_search winners_history + specialist 空 streak
- legacy proxy: 仅基于 params_no_promote_streak

resume v0.6 session 时, `explore_search` 是从 backends_search +
params_search 合流的, winners_history 长度可能很短 (因为 v0.6 没区分),
真 plateau 难触发 → 走 proxy.

## 3. 设计意图

§3.14 R-09 已识别此 gap, 缓解:

> M2 文档明确该 proxy 是临时方案, M7 替换; 灰度文档说明 R-09.

设计目的: 给 v0.6 → v0.8 灰度期一个安全 fallback, 避免真 plateau 在
迁移期触发不可预知的退出.

## 4. 根本原因

R-09 是 *主动选择* 的双轨, 不是漏掉. 但**灰度期结束**应该删 proxy.
"灰度期结束"判定:
- 所有 v0.6 session resume 已经迁移完
- 至少跑过 N 个 fresh v0.8 session 验证真 plateau 触发正常

设计文档没明确"灰度期长度". 实际可能永久双轨.

## 5. 修复路径

### 选项 A — 加 deprecation 时钟

`phase_state.py` 加 env flag:

```text
def exit_normal_explore(state, ...) -> bool:
    if compute_plateau_explore(state, ...).should_exit:
        return True
    # Legacy fallback — deprecated since v0.8.M7 GA.
    if os.environ.get("INFERENCE_OPTIMIZER_DISABLE_PLATEAU_PROXY") == "1":
        return False
    params_streak = ...
    if params_streak >= 5:
        warnings.warn(
            "v0.8 §3.14 R-09 — falling back to params_no_promote_streak "
            "proxy; this path will be removed in v0.9. Set "
            "INFERENCE_OPTIMIZER_DISABLE_PLATEAU_PROXY=1 to forbid.",
            DeprecationWarning,
        )
        return True
    return False
```

### 选项 B — 直接删 proxy (M7 GA 之后)

确认 fresh session + resume 都用真 plateau 路径无误后, 直接删 fallback:

```text
def exit_normal_explore(state, ...) -> bool:
    return compute_plateau_explore(state, ...).should_exit
```

resume v0.6 session 在 winners_history 短时, 真 plateau 可能不触发 →
依赖 explore_phase_budget_exhausted 兜底, 也是 v0.8 设计内.

### 推荐

- 当前: 选项 A (加 deprecation warning + env flag) 上线
- 1-2 个月观察期后: 选项 B 删除

## 6. 验收口径

- [ ] fresh session plateau_explore 触发由真 plateau 函数驱动 (不走 proxy)
- [ ] resume v0.6 session: breakdown.warnings 含 `plateau_proxy_provisional`
      (R-09 探测信号)
- [ ] `INFERENCE_OPTIMIZER_DISABLE_PLATEAU_PROXY=1` 启动时, proxy 路径
      不触发

## 7. 风险 / 回退

- **R-09 自身**: 灰度期 phase 退出时机不一致. 选项 A 保留 fallback 减
  小风险.
- **回退**: 删除 env flag check, fallback 永远启用 = 当前状态.

## 8. 关联 gap

- 关联 §3.14 R-09 (设计已识别)
- 独立, 与其他 gap 无依赖
