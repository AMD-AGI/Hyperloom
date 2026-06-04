# Phase B · 步骤 04 — 跨子系统 / 模块内重复去重

## 决策:默认"接受复制",只统一退役项

各 agent 是**独立可安装包**,抽公共库会增加跨包引用复杂度,违背主旨。
因此**默认不抽公共库**;仅在"同一包内"或"纯数据/常量"层面去重。

| 重复 | 位置 | 处理 |
|---|---|---|
| `_payload_aliases.py` 两份 | `inference_optimizer/compat/` + `kernel-agent/tools/` | Phase A 已删 `extra_sglang_args` shim;若整体退役则两份都删,不抽库 |
| `envelope.py` 镜像 `intent_parser.py` | `robustness-agent/src/.../role/envelope.py` ↔ `inference_optimizer/orchestrator/intent_parser.py` | **保留复制**(独立包);只确保两边退役的 IntentType(OBJECTION/VOTE 等)同步删除,平价测试守住 |
| 各 agent backend `run()` 子进程调用样板 | `orchestrator/backends/critic_agent.py` / `robustness_agent.py` | 同包内(都在 inference_optimizer):可抽 `_run_agent_subprocess(...)` 公共 helper |
| executor 里 Magpie 结果解析/leak salvage | `action_executors/*` | 已有 `benchmark_result.py`;把散落的重复解析收敛到它 |

## 模块内重复(优先,低风险高收益)

- `grid` 运行 / fingerprint:`explore.py` 与 `sweep.py` / `conc_sweep.py` 是否重复实现网格循环 → 收敛到 `_grid_runner.py`。
- `_workload_envs.materialize_config_with_envs`:确认 baseline/profile/explore/sweep 都走它,无各自复制。

## 操作

```bash
# 找重复块(相同长字符串/函数名)
rg -n 'def _run_with_session_kill|materialize_config_with_envs|harvest_leaked_artifacts|variant_fingerprint' \
  inference_optimizer/orchestrator
```
对每处重复:确认语义一致 → 选单一归属 → 改引用 → 删副本 → 护栏。

## 验收

- [ ] 同包内重复收敛(grid/workload-env/result-parse 单一实现)。
- [ ] 跨包退役项两边同步删除,平价测试绿。
- [ ] 未引入新的跨包公共库(除非净删更多)。
- [ ] commit:`Dedupe grid/workload-env logic` 等。

## ⚠️ 注意

- 跨包"看似重复"的 envelope/常量是**有意复制**以保持包独立 —— 不要为了 DRY 抽库,那会增加引用复杂度。
- 合并前确认两处语义**完全一致**;有细微差异(超时、默认值)的别强合,会引入隐性行为变化。
