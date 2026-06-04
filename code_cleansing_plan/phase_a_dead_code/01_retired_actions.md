# Phase A · 步骤 01 — 退役 action 名清除

## 对象

已退役、不应作为活动指令出现的 action 名:
`setup`、`classify`、`backends`、`params`、`validate_stack`、`select_kernels`

它们的逻辑早已合并进 `explore` / kernel `trace_analyze` / 由 CLI flag 取代。残留的是**注释、兼容分支、proxy、文档**。

## 已知残留位置(逐个核实并清)

| 残留 | 位置 |
|---|---|
| 注释引用 | `orchestrator/coordinator.py`(~5231 / 5755 / 9635 附近) |
| 模块头注释"backends.py/params.py stay tiny" | `orchestrator/action_executors/_grid_runner.py` |
| 注释"after every validate_stack KEEP" | `orchestrator/action_executors/session_breakdown.py`(6–7) |
| 注释"looping on validate_stack" | `orchestrator/action_executors/recover.py`(9) |
| "setup / classify" 注释 | `orchestrator/sub_agent_runner.py` |
| 列了 `params` 的注释 | `orchestrator/action_executors/_server_patcher.py`(~45) |
| `select_kernels → trace_analyze` 重命名注释 | `orchestrator/kernel_request_handlers.py`(~2348–2354)、`coordinator.py`(~9569) |
| `last_select_kernels` 字段 | `orchestrator/shared_state.py`(~559–563,resume 丢弃逻辑) |
| `params_no_promote_streak` plateau proxy | `coordinator.py`(~8445)、`phase_state.py`(~1056/1185–1207)、`shared_state.py` |

## 操作

1. 全仓定位:

```bash
rg -n -e 'select_kernels' -e 'validate_stack' -e 'last_select_kernels' \
  -e 'params_no_promote_streak' \
  inference_optimizer kernel-agent critic-agent robustness-agent framework-agent
```
> `backends`/`params`/`setup`/`classify` 因是常用词,改用更精确的模式定位(如 `"validate_stack"`、`action.*params`、`'backends'`),人工区分动作名 vs 普通用法。

2. 区分三类残留并分别处理:
   - **纯注释/文档**:直接删该注释或改成当前语义的一行描述。
   - **兼容分支/字段**(如 resume 丢弃 `last_select_kernels`、`params_no_promote_streak` proxy):整段删除(本次允许破坏内部行为)。
   - **PolicyGate 里对退役 action 的 R1 拒绝规则**:确认 PolicyGate 仍会因"不在 PHASE_ALLOWED_ACTIONS"而拒绝它们后,可删除**专门**针对退役名的硬编码规则;若某拒绝测试依赖它,Phase E 处理对应测试。

3. 同步清理 `actions/_meta/` 下是否有退役 action 的 yaml(若有,删)。

## 验收

- [ ] `rg` 退役名引用计数归零(迁移/拒绝测试除外)。
- [ ] 护栏全绿。
- [ ] commit:`Remove retired action references (setup/classify/backends/params/validate_stack/select_kernels)`(可拆多个)。

## ⚠️ 注意

- `phase_state.py` 的 `disable_legacy_proxy` / `INFERENCE_OPTIMIZER_DISABLE_PLATEAU_PROXY`:删 `params_no_promote_streak` proxy 时,连同其开关与 env 一并删。
- 删除前确认这些退役名**不再**是任何外部消费者(SKILL/launcher/CLI flag)的输入。它们不是 CLI flag,安全。
