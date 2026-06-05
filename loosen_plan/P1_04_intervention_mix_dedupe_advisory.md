# P1_04 — intervention-mix 去重 + 删 ESCALATION 指令 + 删 prose MUST

- **Phase**: P1 · **风险**: 低 · **依赖**: 无 · **后继**: P1_09

## 目标

消除"config vs code_patch"策略的**双写漂移**与**重复注入**。当前:
- `to_intervention_mix_summary`(`shared_state.py` 3196–3252)在 `consecutive_config_only_rounds >= 2` 时输出命令式 `ESCALATION:`(必须改派 serving_specialist 写 source patch)。
- 该块在一个 orchestration tick 内被**注入两次**(`coordinator.py` 4617–4623 与 4695–4701)。
- 同一策略又在 `orchestration.md` 181–188 写成硬 **MUST**(NEXT EXPLORE dispatch MUST be specialist code patch)。

三处表达同一策略,既冗余又互相强制。

## 不变量 vs 策略判定

- "不要长期 config-only,应升级到 code patch" = **STRATEGY**(优化判断,做错只是效率低)。
- intervention ledger 计数本身可作为**中性事实**(telemetry)保留;但"ESCALATION 必须做 X"的指令性 = STRATEGY,删除。

## 改动清单(删除优先)

### 1. 去重注入(`coordinator.py`)
- 删除两处注入中的一处。保留 4695–4701 的 `=== Intervention mix (config vs code_patch) ===`(语义更完整),删除 4616–4623 的 `=== Intervention mix (advisory) ===` 重复块。

### 2. 删 ESCALATION 指令文本(`shared_state.py`)
- `to_intervention_mix_summary`(3196–3252):删除 3239–3251 的 `ESCALATION:` 命令式分支。保留**中性计数**(config keeps / code_patch keeps / consecutive counter 的客观数字)。
- 标题改为中性,如 `=== Intervention mix (telemetry) ===`。

### 3. 删 prose MUST(`orchestration.md`)
- 删除 181–188 "Do not settle for config-only … NEXT EXPLORE dispatch MUST be …" 的硬指令段。如需保留引导,改为一句**事实**:"intervention ledger 显示连续 config-only 轮次时,可考虑 source patch 路线"(非 MUST)。

### 4. 计数器去留(可选,偏删除)
- `consecutive_config_only_rounds`(`shared_state.py` 859)、`record_intervention`(2953–2985)、`_record_intervention_for_task`(`coordinator.py` 9593–9664):
  - **保守**:保留计数作 telemetry。
  - **删除优先**:若 telemetry 无消费者(去掉 ESCALATION 后仅剩数字展示),可整体删除计数器与 hook,进一步瘦身。建议先保留 telemetry 一版,确认 LLM 不依赖后再删。

## 连带测试

| 文件 | 函数 | 动作 |
|---|---|---|
| `test_intervention_mix_escalation.py` | `test_two_consecutive_config_only_rounds_escalate`(36)、`test_config_heavy_zero_patch_escalates_*`(113)、`test_code_patch_keep_resets_*`(49) | **删除/改写**:不再有 ESCALATION 文本;改测"输出中性计数,无 ESCALATION 指令" |
| `test_intervention_mix_escalation.py` | `test_empty_ledger_renders_nothing`(19)、`test_single_config_keep_shows_counts_*`(26) | 保留(中性计数行为) |
| `test_robustness_storm_and_mix.py` | `test_intervention_mix_summary_fires_on_*`(272–289) | 改写为中性计数断言 |
| `test_robustness_storm_and_mix.py` | `test_*_hook_records_*`(109–156)、`test_consecutive_config_only_*`(61–70) | 若保留计数则保留;若删计数则删 |

## 验证
- orchestration prompt 一个 tick 内只出现**一次** intervention-mix 段,且无命令式 ESCALATION。
- `orchestration.md` 无 config-only MUST。
- 烟测:连续 config-only 多轮时,run 不会因缺少 ESCALATION 而异常(本就 advisory)。

## 回退
- 恢复第二处注入、ESCALATION 分支、prose MUST 及相关测试。

## 残留风险
- 低。这是纯去重 + 去指令化。`_record_intervention_for_task` 的 hook(任务完成时记账)若保留,不影响行为。
