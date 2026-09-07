# 聚合量逐条审计表

commit `da54496de`。共 197 条。

## 判据

**已定：取消一切读时推导，全部走产生式录快照。** 因此判据不再是「推导还是快照」，而是「在哪一刻落快照」。作用域决定产生点：

| 记号 | 作用域 | 快照时刻 | 机制 |
|------|--------|----------|------|
| **E** | 事件级 | `finish_event` | **已存在**：`assemble_{baseline,roofline,kernel}_ext` 在事件结束时汇总该事件所有行 |
| **P** | 决策级 | 决策发生那一刻 | 在决策点直接写 fragment |
| **C** | session 级 | close 序列 | close 时读齐 fragment、算一次、冻结 |
| **X** | — | 不记录 | 当前实现是猜的，或字段恒空 |
| **DUP** | — | 不记录 | V6 timeline 已有同语义内容，见 `SBD_V6_DEDUP.md` |

### 作用域按分区的默认归属

多数条目的作用域由它所属的分区机械决定，不需要逐条判断：

| 分区 | 默认作用域 | 说明 |
|------|-----------|------|
| §3 kernel_lifecycle、§4 conc_sweep、§6 roofline、§18 baseline 部分 | **E** | 归入对应 timeline 事件的 `finish_event` |
| §7–§11 optimizations 的 per-attempt 与 gain ledger | **P** | 每次 adoption / promote 决策时钉死 |
| §21 framework plateau / streak 一族 | **P** | 这是运行时用来决定切 arm 的输入，必须录当时的值 |
| §2 capability_summary、§12–§14 telemetry、§16 token、§9 summaries、§10 validation 汇总 | **C** | 跨多个事件，只有 close 时才齐 |
| §1 phase_segments、§19 enablement、§22 outcome/close、§23 KB | 逐条见表 | 混合 |

**需要逐条看的只有三类**：标 `!` 的（当前实现含 fallback 或猜测）、标 **X** 的（建议不记录）、标 **DUP** 的（V6 已覆盖）。其余按上表默认归属即可。

原表的 `建议` 列（D/S 判定）已被本方案取代：原判 **D** 的条目按其分区归入 E / P / C，原判 **S** 的条目落在 P 或 C。列内容保留是为了看出**哪些条目原本就无法推导**——那些是即使在推导模型下也必须快照的，风险最高。

`决策` 列留空，请逐条填。

---

## 1. phase_segments / phase_timeline

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A1 | `phase_timeline[]` 去重 | dedup-union | `timeline.py:265` | 是（`phase_transitions`） | **D** | 录制后只有一个源，去重逻辑本身可删 | |
| A2 | `phase_segments[*].elapsed_seconds` | rollup | `timeline.py:767` | 是 | **D** | transition 两两配对 | |
| A3 ! | `phase_segments[*].actions` | grouping | `timeline.py:826` | 部分 | **D** | 现在靠时间窗归属；录制时让 action 直接带 phase 即可，猜测消失 | |
| A4 | `phase_segments[*].events` | grouping | `timeline.py:809` | 是 | **D** | | |

---

## 2. capability_summary

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A5 !! | `capability_summary.geak.attempts` | count | `timeline.py:442,505` | 部分 | **D** | 现在 jsonl 缺失时用 `max(list_len, 1)` 猜；录制 attempt 行后猜测消失 | |
| A6 | `capability_summary.forge.attempts` | count | `timeline.py:442` | 是 | **D** | | |
| A7 !! | `capability_summary.{geak,forge}.keeps` | dedup-union | `timeline.py:445,534` | 部分 | **D** | jsonl 与 integrate ledger 不同步导致误判；统一录制后消失 | |
| A8 | `capability_summary.{geak,forge}.reverts` | count | `timeline.py:446` | 是 | **D** | | |
| A9 !! | `capability_summary.geak.pending_integrate` | count | `timeline.py:447,497` | 部分 | **D** | 同 A7 | |
| A10 !! | `capability_summary.geak.micro_only_keeps` | count | `timeline.py:448,499` | 部分 | **D** | 同 A7 | |
| A11 | `capability_summary.geak.e2e_gain_pct` | max | `timeline.py:449` | 是 | **D** | | |
| A12 !! | `capability_summary.{geak,forge}.status` | status-inference | `timeline.py:487,534` | 部分 | **D** | GEAK 分支现在有 route-evidence fallback，是整表最脏的一处 | |
| A13 | `capability_summary.explore.attempts` | count | `timeline.py:294` | 是 | **D** | | |
| A14 ! | `capability_summary.explore.keeps` | count | `timeline.py:296,310` | 是 | **D** | `_fold_search_ledger_keeps` 会抬高计数；录制 keep 行后取消折叠 | |
| A15 ! | `capability_summary.explore.status` | status-inference | `timeline.py:302` | 是 | **D** | | |
| A16 | `capability_summary.explore.tested` | count | `timeline.py:548` | 是 | **D** | | |
| A17 | `capability_summary.explore.best_gain_pct` | max | `timeline.py:551` | 是 | **D** | | |
| A18 | `capability_summary.explore.keep_unstable_count` | count | `timeline.py:558` | 是 | **D** | | |
| A19 | `capability_summary.explore.winners_history` | count | `timeline.py:565` | 是 | **D** | | |
| A20 | `capability_summary.specialist.attempts` | count | `timeline.py:611` | 是 | **D** | | |
| A21 | `capability_summary.specialist.tested` | sum | `timeline.py:612` | 是 | **D** | | |
| A22 | `capability_summary.specialist.keeps` | sum | `timeline.py:613` | 是 | **D** | | |
| A23 | `capability_summary.specialist.status` | status-inference | `timeline.py:655` | 是 | **D** | | |
| A24 !! | `capability_summary.specialist.by_specialist.{domain}.{attempts,keeps,tested}` | sum / **均分猜测** | `timeline.py:631,640` | 否 | **D**（依赖 R5 补实现） | 现在没有 `domain_breakdown` 时按域数**平均分摊**，纯编造。你已决定补实现 proposal 明细，届时可真推导 | |
| A25 | `capability_summary.specialist.by_specialist.{domain}.status` | status-inference | `timeline.py:666` | 否 | **D** | 同上 | |
| A26 | `capability_summary.*.not_attempted_reason` | status-inference | 新增 | 否 | **D** | 产生点在 `policy/gate.py:2064`、`kernel.py:3980`、`_grid_runner.py:390`，录 denial 行后推导 | |

---

## 3. kernel_lifecycle

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A27 | `kernel_lifecycle.funnel.*` 五段计数 | funnel | `reporters/cross_section.py` | 是（kernel 事件行） | **D** | 已经是读时 `len()`，保持 | |
| A28 | `detected[*].{geak,forge}.attempts` | count | `kernels.py:441` | 是 | **D** | | |
| A29 | `detected[*].{geak,forge}.best_speedup` | max | `kernels.py:442` | 是 | **D** | | |
| A30 ! | `detected[*].{geak,forge}.decision` | status-inference | `kernels.py:447,143` | 部分 | **D** | 现在靠 verification 轮转启发式盖章；录制时直接写 decision | |
| A31 | `detected[*].selected_for_optimization` | status-inference | `kernels.py:582` | 是 | **D** | | |
| A32 ! | `detected[*].adopted_by` | status-inference | `kernels.py:617` | 部分 | **D** | | |
| A33 ! | `detected[*].final_decision` | status-inference | `kernels.py:631` | 部分 | **S** | resume 后 rejected 与 adopted 有竞态，事后无法判定先后 | |
| A34 ! | `detected[*]` 静态字段补全 | imputation | `kernels.py:513` | 部分 | **X** | 从 `benchmark_report.kernel_summary` 尾部猜；discovery 时录全即可 | |
| A35 ! | `optimized[*].total_attempts` | count + `max()` | `kernels.py:730,770` | 是 | **D** | 现在与 ledger 取 max，掩盖 retry 语义；录制 attempt 行后直接数 | |
| A36 | `optimized[*].successful_attempts` | count | `kernels.py:737` | 是 | **D** | | |
| A37 | `optimized[*].best_micro_speedup` | max | `kernels.py:731` | 是 | **D** | | |
| A38 ! | `optimized[*].last_decision` | status-inference | `kernels.py:739,771` | 部分 | **D** | | |
| A39 | `optimized[*].attempts_summary` | grouping | `kernels.py:740` | 是 | **D** | | |
| A40 | `rejected[*]` 合成行 | dedup-union | `kernels.py:871` | 是 | **D** | | |
| A41 ! | `detected[*].kernel_id` 从目录名推断 | imputation | `kernels.py:322` | 部分 | **X** | 录制时 kernel_id 是已知的，不需要从文件名反推 | |
| A42 ! | 候选来自 `hot_kernels_top15` | imputation | `kernels.py:408` | 部分 | **X** | 同上 | |

---

## 4. conc_sweep

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A43 !! | `conc_sweep_summary.summary.successful_pairs` | count | `kernels.py:1083` | 否（导出时 recovery） | **D** | recovery 会覆盖 stale report，是 bug 源；改为 sweep 时逐点录制 | |
| A44 !! | `conc_sweep_summary.comparison[*]` | rollup | `kernels.py:1083` | 否 | **D** | 同上 | |
| A45 !! | `conc_sweep_summary.status` | status-inference | `kernels.py:1087` | 否 | **D** | 同上 | |
| A46 | `timeline[conc_sweep].status` | status-inference | `v6_stages.py:230` | 部分 | **D** | budget_exhausted → degraded，规则明确 | |
| A47 | `timeline[conc_sweep].{start,end}_time` | min/max | `v6_stages.py:258` | 部分 | **D** | 事件 open/finish 自带 | |
| A48 | `timeline[conc_sweep].ext.result.best_conc` | rollup | `v6_stages.py:298` | 是 | **D** | | |
| A49 | `timeline[conc_sweep].ext.result.best_speedup` | max | `v6_stages.py:299` | 是 | **D** | | |
| A50 | `timeline[conc_sweep].ext.comparison[*].error` | status-inference | `v6_stages.py:172` | 否 | **D** | 配对失败原因在 sweep 时已知，录下即可 | |
| A51 | `timeline[conc_sweep].ext.comparison[*].speedup` | ratio | passthrough | 是 | **D** | | |

---

## 5. param_search / explore_search

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A52 | `param_search.{ledger}.tested_count` | count | `explore.py:107` | 是 | **D** | ledger ∈ {explore, params, backends} | |
| A53 | `param_search.{ledger}.top_by_gain[*]` | top-N | `explore.py:100` | 是 | **D** | 排序 + 截 20，纯读时逻辑 | |
| A54 | `param_search.explore.no_promote_streak` | streak | `explore.py:174` | 是 | **D** | 从 no-promote 行计数；state 保留供循环决策 | |
| A55 | `param_search.discovered_flags` | map | `explore.py:186` | 否 | **待补实现** | 你已决定补 AST 扫描（R5） | |

`params_search` / `backends_search` 两个 ledger 在 orchestrator 侧无 writer，恒空。补实现或删，随 A55 一起定。

---

## 6. roofline

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A56 | `roofline[0].mode` | status-inference | `roofline_snapshot.py:517` | 是 | **D** | | |
| A57 | `roofline[0].delta.compute_pct` | rollup | `roofline_snapshot.py:530` | 是 | **D** | | |
| A58 | `roofline[0].delta.idle_pct` | rollup | `roofline_snapshot.py:534` | 是 | **D** | | |
| A59 | `roofline[0].delta.comm_pct` | rollup | `roofline_snapshot.py:538` | 是 | **D** | | |
| A60 | `roofline[0].delta.top_kernel_efficiency_pct` | rollup | `roofline_snapshot.py:542` | 是 | **D** | | |
| A61 | `roofline[0].delta.within_roofline_pct` | rollup | `roofline_snapshot.py:546` | 是 | **D** | ceiling 变动时置 null，规则明确 | |
| A62 | `roofline[0].delta.gap_to_roofline_pct` | rollup | `roofline_snapshot.py:554` | 是 | **D** | | |
| A63 ! | `roofline_progress.trajectory[*].gain_pct` | ratio | `roofline.py:135` | 部分 | **D** | resume 中途缺 stack 行会算错；录 promote 行后可靠 | |
| A64 | `roofline_progress.trajectory` 排序 | trajectory | `roofline.py:127` | 部分 | **D** | | |
| A65 !! | `roofline_progress.current_best_tput` | max/last | `roofline.py:163` | 部分 | **S+D** | 已知会与 `state.current_best.tput` 分歧（代码里有 warning）。快照记权威值，推导用于校验 | |
| A66 | `roofline_progress.cumulative_gain_pct` | fact | `roofline.py:164` | 是 | **D** | | |
| A67 | `roofline_progress.ceiling_tok_per_sec` | max/last | `roofline.py:157` | 是 | **D** | | |
| A68 | `roofline_progress.target_tok_per_sec` | ratio | `roofline.py:160` | 是 | **D** | ceiling × 0.70，常量 | |
| A69 | `roofline_progress.current_best_pct_of_ceiling` | ratio | `roofline.py:165` | 是 | **D** | 随 A65 | |
| A70 | `roofline_progress.current_best_pct_of_target` | ratio | `roofline.py:168` | 是 | **D** | 随 A65 | |
| A71 | `roofline_progress.current_best_pct_of_latency_ceiling` | ratio | `roofline.py:192` | 是 | **D** | | |
| A72 | `roofline_progress.ceiling_kind` | status-inference | `roofline.py:178` | 是 | **D** | | |
| A73 | `roofline_progress.ceiling_available` | status-inference | `roofline.py:159` | 是 | **D** | | |
| A74 | `roofline_progress.roofline_failure_streak` | streak | `roofline.py:211` | 是 | **D** | 从失败行计数 | |

---

## 7. optimizations — per-attempt

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A75 !! | `attempts[*].agent` | status-inference | `optimizations.py:186` | 部分 | **D** | 三级 fallback（recorded → kind map → patch_author）；产生时录 agent 即可 | |
| A76 !! | `attempts[*].agent_method` | status-inference | `optimizations.py:453` | 部分 | **X** | 这个字段只是在自报"上面那个值是猜的还是真的"。真录制后无意义 | |
| A77 !! | `attempts[*].decision` | status-inference | `optimizations.py:424` | 是 | **D** | 四级 fallback chain，录制后取消 | |
| A78 ! | `attempts[*].adopted` | status-inference | `optimizations.py:432` | 是 | **D** | | |
| A79 !! | `attempts[*].local_gain_pct` | ratio chain | `optimizations.py:442` | 是 | **D** | alias 冲突是当前主要错源 | |
| A80 !! | `attempts[*].throughput_{before,after}` | max/first | `optimizations.py:397` | 是 | **S** | `adoption_pinned_stale` 说明 pin 的测量会失效。before/after 应在 adoption 时刻钉死为快照 | |
| A81 | `attempts[*].duration_sec` | rollup | `optimizations.py:471` | 是 | **D** | | |
| A82 | `attempts[*].measurements[*].occurrence` | count | `optimizations.py:548` | 是 | **D** | | |
| A83 | `attempts[*].measurements[*].occurrences_of_name` | count | `optimizations.py:372` | 是 | **D** | | |
| A84 !! | `attempts[*].measurement_source` | status-inference | `optimizations.py:329` | 是 | **X** | 同 A76，是"我用了哪条 fallback"的自述 | |
| A85 !! | `attempts[*].alias_conflicts` | status-inference | `optimizations.py:355` | 是 | **D** | 冲突检测本身有价值，保留为读时校验 | |
| A86 | `attempts[*].backend_attempts[*].duration_sec` | rollup | `optimizations.py:279` | 是 | **D** | | |
| A87 | `backend_attempts[*].sequence` | count | `optimizations.py:146` | 是 | **D** | | |

---

## 8. optimizations — gain ledger

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A88 !! | `entries[*].gain_pct` | ratio | `optimizations.py:944` | 是 | **S** | 吞吐对缺失时投影 local_gain。gain 是 promote 决策的依据，应在 promote 时刻钉死 | |
| A89 !! | `entries[*].cumulative_gain_pct` | sum | `optimizations.py:927` | 部分 | **S+D** | 累计值有 drift 项，推导与快照必然不等；分歧本身要暴露 | |
| A90 !! | `entries[*].gain_method` | status-inference | `optimizations.py:951` | 部分 | **X** | 同 A76，自述 fallback 路径 | |
| A91 !! | `entries[*].chain_continuous` | status-inference | `optimizations.py:943` | 部分 | **D** | 链断裂检测保留为读时校验 | |
| A92 | `entries[*].stack_index` | count | `optimizations.py:983` | 是 | **D** | | |

---

## 9. optimizations — summaries

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A93 | `summary_by_agent.{agent}.attempts` | count | `optimizations.py:668` | 是 | **D** | | |
| A94 | `summary_by_agent.{agent}.{keeps,reverts}` | count | `optimizations.py:668` | 是 | **D** | | |
| A95 | `summary_by_agent.{agent}.attributable_gain_pct` | sum | `optimizations.py:668` | 是 | **D** | 随 A88 | |
| A96 | `summary_by_agent.{agent}.non_attributable_keeps` | count | `optimizations.py:668` | 是 | **D** | | |
| A97 | `summary_by_agent.{agent}.by_kind.{kind}.*` | grouping | `optimizations.py:668` | 是 | **D** | | |
| A98 | `summary_by_source.{src}.keeps` | count | `optimizations.py:1040` | 是 | **D** | | |
| A99 | `summary_by_source.{src}.total_gain_pct` | sum | `optimizations.py:1040` | 是 | **D** | | |
| A100 | `summary_by_source.{src}.by_backend.{b}.*` | grouping | `optimizations.py:1040` | 是 | **D** | b ∈ {geak, forge, unattributed} | |
| A101 | `summary_by_kind.{kind}.{keeps,total_gain_pct}` | count/sum | `optimizations.py:1062` | 是 | **D** | | |

---

## 10. optimizations — validation

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A102 ! | `validation.method` | status-inference | `optimizations.py:1161` | 部分 | **X** | 自述用了哪个源 | |
| A103 !! | `validation.validated_total_gain_pct` | max/prefer | `optimizations.py:1170` | 部分 | **S** | 这是"验证过的总增益"，是 session 结论。必须在 validate 动作时刻录快照 | |
| A104 !! | `validation.ledger_total_gain_pct` | sum | `optimizations.py:1178` | 部分 | **D** | 与 A103 对照用 | |
| A105 !! | `validation.reconciliation_gap_pct` | ratio | `optimizations.py:1183` | 部分 | **D** | = A103 − A104，读时算 | |
| A106 !! | `validation.attributed_total_gain_pct` | sum | `optimizations.py:935` | 部分 | **D** | | |
| A107 !! | `validation.unattributed_gain_pct` | sum | `optimizations.py:934` | 部分 | **D** | 步间 drift | |
| A108 !! | `validation.attribution_gap_pct` | ratio | `optimizations.py:1186` | 部分 | **D** | | |
| A109 | `validation.attempt_count` | count | `optimizations.py:1190` | 是 | **D** | | |
| A110 | `validation.keep_count` | count | `optimizations.py:1191` | 是 | **D** | | |
| A111 | `validation.non_attributable_keep_count` | count | `optimizations.py:1192` | 是 | **D** | | |
| A112 ! | `validation.unmeasured_keep_count` | count | `optimizations.py:1194` | 部分 | **D** | | |
| A113 ! | `validation.projected_keep_count` | count | `optimizations.py:1198` | 部分 | **X** | 随 A88/A90，真录制后不存在"投影的 keep" | |
| A114 !! | `validation.stale_evidence_count` | count | `optimizations.py:1200` | 部分 | **D** | 随 A80 快照化后应恒为 0，保留做守卫 | |
| A115 !! | `validation.unclaimed_integration_count` | count | `optimizations.py:1204` | 部分 | **D** | 保留做守卫 | |
| A116 ! | `validation.unscored_keep_count` | count | `optimizations.py:1128` | 部分 | **D** | | |
| A117 | `validation.validated_at_stack_len` | max/last | `optimizations.py:1162` | 部分 | **S** | 随 A103，同一时刻钉死 | |

---

## 11. optimizations — gemm_tuning

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A118 | `gemm_tuning_runs[*].adopted` | status-inference | `optimizations.py:769` | 是 | **D** | | |
| A119 ! | `gemm_tuning_runs[*].gain_pct` | fallback | `optimizations.py:774` | 是 | **D** | 随 A88 | |
| A120 | `gemm_tuning_runs[*].duration_sec` | rollup | `optimizations.py:762` | 是 | **D** | | |
| A121 | `gemm_tuning_runs[*].candidates` | grouping | `optimizations.py:776` | 是 | **D** | | |

---

## 12. telemetry — GPU

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A122 | `gpu_monitor_aggregate.samples` | count | `telemetry.py:176` | 是（子进程 report） | **D** | 在 `extract_benchmark_measurement` 录采样行后推导 | |
| A123 ! | `gpu_monitor_aggregate.avg_power_w` | mean | `telemetry.py:237` | 是 | **D** | 现在有 `power`/`power_w` 字段名 fallback，录制时统一 | |
| A124 | `gpu_monitor_aggregate.max_power_w` | max | `telemetry.py:238` | 是 | **D** | | |
| A125 ! | `gpu_monitor_aggregate.avg_temp_c` | mean | `telemetry.py:239` | 是 | **D** | 同 A123 | |
| A126 | `gpu_monitor_aggregate.max_temp_c` | max | `telemetry.py:240` | 是 | **D** | | |
| A127 ! | `gpu_monitor_aggregate.avg_clock_mhz` | mean | `telemetry.py:241` | 是 | **D** | `clock_mhz`/`sclk_mhz` fallback | |

采样量可能很大。如果每 sample 一行 fragment 太重，可考虑每次 benchmark 录一条"该次 benchmark 的采样小结"，session 级再推导——这是 A122–A127 唯一需要你定的点。

---

## 13. telemetry — lane

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A128 !! | `lane_timeline[].live_holders` | count | `telemetry.py:287` | 否 | **S** | **查的是导出那一刻未过期的 lease**，重跑导出值就变，不是 session 事实。应改为录制 acquire/release 行，输出改成"峰值占用"语义 | |
| A129 | `lane_timeline[].lease_expired_count` | count | `telemetry.py:303` | 部分 | **D** | expire 已发事件，录行后推导 | |
| A130 ! | `lane_timeline[].capacity` | 配置 | `telemetry.py:281` | 部分 | **D** | 缺 `lane_capacity` 表时用 schema 默认值；capacity 在 coordinator 启动时已知，录一次 | |
| A131 !! | `lane_timeline[__total__].live_holders` | sum | `telemetry.py:341` | 否 | **S** | 随 A128 | |
| A132 | `lane_timeline[__total__].lease_expired_count` | count | `telemetry.py:343` | 部分 | **D** | 注意：现在是全局计数，不等于各 lane 之和 | |
| A133 | `lane_timeline[__total__].capacity` | sum | `telemetry.py:337` | 部分 | **D** | | |

---

## 14. telemetry — orchestration context

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A134 | `orchestration_context.seed_prompts` | rollup | `telemetry.py:417` | 是 | **D** | 录每 turn 模式行 | |
| A135 | `orchestration_context.delta_prompts` | rollup | `telemetry.py:418` | 是 | **D** | | |
| A136 | `orchestration_context.compactions` | count | `telemetry.py:420` | 部分 | **D** | | |
| A137 | `orchestration_context.degenerate_compactions` | count | `telemetry.py:421` | 部分 | **D** | | |
| A138 | `orchestration_context.tick_count` | rollup | `telemetry.py:422` | 是 | **D** | | |
| A139 !! | `orchestration_context.compactions_per_tick` | ratio | `telemetry.py:423` | 部分 | **D** | resume 改变分母。若要 per-leg 语义则需 **S**，取决于你要"整个 session"还是"本次运行" | |
| A140 | `orchestration_context.delta_ratio` | ratio | `telemetry.py:424` | 部分 | **D** | | |
| A141 | `orchestration_context.context_tokens_at_compaction.min` | min | `telemetry.py:411` | 部分 | **D** | | |
| A142 | `orchestration_context.context_tokens_at_compaction.median` | median | `telemetry.py:413` | 部分 | **D** | | |
| A143 | `orchestration_context.context_tokens_at_compaction.max` | max | `telemetry.py:414` | 部分 | **D** | | |

---

## 15. critic_robustness / specialist_runs

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A144 | `critic_robustness.kb_writes_summary.total` | count | `telemetry.py:101` | 部分 | **D** | `critic_iterations` 已是 section，补 verdict 字段即可 | |
| A145 | `critic_robustness.kb_writes_summary.by_verdict.{V}` | grouping | `telemetry.py:113` | 部分 | **D** | | |
| A146 | `specialist_runs[].domain_breakdown.{d}.dispatched` | rollup | `telemetry.py:598` | 是 | **D** | | |
| A147 | `specialist_runs[].domain_breakdown.{d}.proposals_total` | rollup | `telemetry.py:598` | 部分 | **D** | 依赖 R5 补实现 | |
| A148 | `specialist_runs[].domain_breakdown.{d}.proposals_kept` | rollup | `telemetry.py:598` | 部分 | **D** | 同上 | |
| A149 | `specialist_runs[].domain_breakdown.{d}.proposals_rejected` | rollup | `telemetry.py:598` | 部分 | **D** | 同上 | |
| A150 ! | `specialist_runs[].transcripts[].domain` | status-inference | `telemetry.py:626` | 部分 | **D** | 老 round 无 `task_domains` map 时靠单 tag 猜；dispatch 时录 domain | |

---

## 16. decision_trace / token_usage

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A151 | `token_rollup.session_total.total_in` | sum | `decision.py:820` | 是 | **D** | | |
| A152 | `token_rollup.session_total.total_out` | sum | `decision.py:820` | 是 | **D** | | |
| A153 | `token_rollup.session_total.total_cache_creation` | sum | `decision.py:820` | 是 | **D** | | |
| A154 | `token_rollup.session_total.total_cache_read` | sum | `decision.py:820` | 是 | **D** | | |
| A155 | `token_rollup.session_total.total_reasoning_out` | sum | `decision.py:820` | 是 | **D** | | |
| A156 | `token_rollup.session_total.calls` | count | `decision.py:820` | 是 | **D** | | |
| A157 | `token_rollup.by_component.{c}.*` | grouping | `decision.py:809` | 是 | **D** | | |
| A158 !! | `token_rollup.by_phase.{p}.*` | grouping | `decision.py:814` | 部分 | **D** | call 无 phase 时按时间窗回填。`append_llm_call` 时 phase 是已知的，直接写进 call 行 | |
| A159 | `unattributed_tokens.*` | sum | `decision.py:793` | 是 | **D** | | |
| A160 | `overhead_tokens.*` | sum | `decision.py:794` | 是 | **D** | | |
| A161 !! | `decision_trace[].tokens.*` | sum | `decision.py:765` | 是 | **D** | 现在同 key 多次 decision 时**只有首次拿到 token**（防重复计数）。retry 场景归因偏首次，需要在录制时给 decision 加序号 | |
| A162 !! | `decision_trace[].tokens.by_component.{c}.*` | grouping | `decision.py:765` | 是 | **D** | 同 A161 | |
| A163 | `token_usage.session_total.total_in_out` | sum | `decision.py:478` | 是 | **D** | | |
| A164 | `token_usage.session_total.grand_total` | sum | `decision.py:382` | 是 | **D** | | |
| A165 | `token_usage.session_total.cache_hit_rate` | ratio | `decision.py:384` | 是 | **D** | | |
| A166 | `token_usage.by_component.{c}.*` | sum/ratio | `decision.py:479` | 是 | **D** | | |
| A167 | `token_usage.by_phase.{p}.*` | sum/ratio | `decision.py:480` | 部分 | **D** | 随 A158 | |
| A168 | `token_usage.attribution.attributed_to_decisions.*` | sum | `decision.py:435` | 是 | **D** | | |
| A169 | `token_usage.attribution.attributed_calls_pct` | ratio | `decision.py:441` | 是 | **D** | | |
| A170 | `token_usage.attribution.overhead_calls_pct` | ratio | `decision.py:443` | 是 | **D** | | |
| A171 ! | `token_usage.timeline[].tokens` | join | `decision.py:447` | 部分 | **D** | 无 LLM 开销的 action → null，语义正常 | |

---

## 17. langfuse

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A172 !! | `langfuse.counts.*` | rollup | `decision.py:534` | 部分 | **S** | 首次 build 在 flush 之前，拿到的是 pre-flush 计数（`counts_final=False`）。flush 完成时录快照，配合已有的 `patch_breakdown_langfuse` | |
| A173 !! | `langfuse.counts.*`（live 分支） | rollup | `decision.py:545` | 否 | **X** | 读内存 emitter singleton，进程被 kill 就没了 | |
| A174 | `langfuse.receipt_source` | status-inference | `decision.py:541` | 派生 | **X** | 自述用了哪个 tier | |
| A175 | `langfuse.enabled` / `config.*` | status-inference | `decision.py:552` | 否 | **D** | 配置探测，session 开始时录一次 | |
| A176 | `metadata.langfuse.trace_url` | status-inference | `v6.py:90` | 部分 | **D** | host + trace_id 拼接 | |

---

## 18. session / baseline / final

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A177 !! | `session.elapsed_minutes` | rollup | `sessions.py:627`, `exporter.py:123` | 部分 | **S** | 未结束的 session 每次导出都变长。应在 close 时录一次最终值 | |
| A178 | `session.recovery.recovered` | status-inference | `sessions.py:706` | 是 | **D** | | |
| A179 ! | `session.stop_reason` | status-inference | `sessions.py:756` | 是 | **D** | CLOSE phase 会覆盖 `time_exhausted` | |
| A180 !! | `session_meta.session_duration_seconds` | rollup | `sessions.py:796` | 派生 | **S** | 随 A177 | |
| A181 ! | `baseline.ttft_mean_ms` | mean | `sessions.py:969` | 是 | **D** | 现在 workspace 失败会走"最新 mtime report"猜；改为 measurement 提取时录 | |
| A182 ! | `baseline.e2el_mean_ms` | mean | `sessions.py:969` | 是 | **D** | 同上 | |
| A183 ! | `baseline.ttft_e2el_source` | status-inference | `sessions.py:973` | 派生 | **X** | 自述走了哪条路 | |
| A184 !! | `baseline.attempts_history[]` | trajectory | `sessions.py:1027,1111` | 否 | **D** | state 空时从 `runs/baseline/**` 合成历史。改为每次 attempt 录行 | |
| A185 | `baseline.failure_streak` | streak | `sessions.py:1102` | 是 | **D** | 从失败行数；state 保留供决策 | |
| A186 | `baseline.total_failures` | count | `sessions.py:1104` | 是 | **D** | | |
| A187 | `final.cumulative_gain_pct_validated` | rollup | `sessions.py:1284` | 是 | **D** | 随 A103 | |
| A188 | `final.stack_changed_after_validation` | status-inference | `sessions.py:1292` | 是 | **D** | 随 A117 | |
| A189 | `final.action_path[]` | trajectory | `sessions.py:1218` | 是 | **D** | | |
| A190 !! | `final.{ttft,e2el}_mean_ms` | mean | `sessions.py:1230` | 部分 | **D** | 同 A181 | |

---

## 19. enablement

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A191 !! | `enablement.engaged` | status-inference | `sessions.py:1573` | 部分 | **D** | 四个信号或运算（attempts / dispatched / kept_patches / eval origin）。录 enablement 事件后从事件存在性直接判定 | |
| A192 ! | `enablement.origin` | status-inference | `sessions.py:1572` | 是 | **D** | 注意成功后 `origin` 被清空但 `baseline_eval_kind` 保留，录事件时把 origin 钉在事件上 | |
| A193 | `enablement.trigger_kind` | status-inference | `sessions.py:1652` | 是 | **D** | | |
| A194 | `enablement.human_review_count` | count | `sessions.py:1636` | 是 | **D** | | |
| A195 | `enablement.build_attempt_count` | count | `sessions.py:1684` | 是 | **D** | | |
| A196 | `enablement.attempt_runtimes[].promoted` | status-inference | `sessions.py:1669` | 是 | **D** | | |
| A197 | 整段 `{}` 隐藏门控 | status-inference | `sessions.py:1574` | 部分 | **D** | 随 A191 | |

---

## 20. geak

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A198 ! | `geak.gain_pct` | ratio | `geak.py:963` | 是 | **S** | 末 cycle 会覆盖前面的；应每 cycle 录快照 | |
| A199 !! | `geak.kernels_optimized` | count | `geak.py:943` | 部分 | **D** | 随 A200 | |
| A200 !! | `geak.accepted_kernels[]` | dedup-union | `geak.py:446,532` | 部分 | **D** | 三条重建路径（result → journey → integrate_result）。GEAK 结束时主进程读一次并录制 | |
| A201 ! | `geak.accepted_kernels_kind_sources.{s}` | grouping | `geak.py:128` | 部分 | **D** | | |
| A202 !! | `geak.stages_reached[]` | funnel | `geak.py:659` | 否 | **D** | 现在用**目录是否存在**判断阶段完成，非常脏。每阶段结束录一行 | |
| A203 !! | `geak.likely_cause` | status-inference | `geak.py:839` | 派生 | **X** | 纯启发式猜失败原因。录制真实失败原因后删除 | |
| A204 ! | `geak.last_artifact_ts` | max | `geak.py:776` | 派生 | **X** | 取文件 mtime 最大值，事件有真实时间戳 | |
| A205 !! | `geak.recovered_from_disk` / `status` | status-inference | `geak.py:898` | 派生 | **X** | `recovered_from_disk` 这个字段的存在本身就是投影模型的产物 | |
| A206 | `geak.engaged` | status-inference | `geak.py:894` | 是 | **D** | | |

---

## 21. framework_agent event ext

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A207 | `timeline[framework_agent].status` | status-inference | `v6.py:1900` | 派生 | **D** | 事件 finish 时定 | |
| A208 | `timeline[framework_agent].{start,end}_time` | min/max | `v6.py:1920` | 部分 | **D** | 事件 open/finish 自带 | |
| A209 !! | `ext.config_arm.plateau.recent_keep_gain_pct` | mean | `v6.py:950` | 部分 | **S** | **lookback 窗口在导出时重算**。plateau 判定是运行时决策，判定用的值必须是当时那个值 | |
| A210 !! | `ext.config_arm.plateau.empty_streak` | streak | `v6.py:957` | 派生 | **S** | 同 A209 | |
| A211 ! | `ext.config_arm.plateau.tested_this_cycle` | count | `v6.py:964` | 是 | **D** | | |
| A212 !! | `ext.config_arm.plateau.triggered` | status-inference | `v6.py:967` | 派生 | **S** | 这是真实发生过的决策，必须录当时的结论 | |
| A213 ! | `ext.config_arm.specialist_runs[].status` | status-inference | `v6.py:620` | 部分 | **D** | | |
| A214 | `ext.config_arm.rounds[].status` | status-inference | `v6.py:790` | 是 | **D** | | |
| A215 !! | `ext.source_arm.plateau.consecutive_no_keep` | streak | `v6.py:1693` | 部分 | **S** | 反向扫描 progress 重算，同 A210 | |
| A216 ! | `ext.source_arm.plateau.candidates_exhausted` | status-inference | `v6.py:1706` | 部分 | **S** | 同上 | |
| A217 !! | `ext.source_arm.plateau.triggered` | status-inference | `v6.py:1713` | 派生 | **S** | 同 A212 | |
| A218 ! | `ext.source_arm.candidate_discovery_runs[].status` | status-inference | `v6.py:1071` | 部分 | **D** | | |
| A219 ! | `ext.source_arm.authoring_runs[].status` | status-inference | `v6.py:1231` | 部分 | **D** | | |
| A220 | `ext.source_arm.attempts[].status` | status-inference | `v6.py:1273` | 是 | **D** | | |
| A221 | `ext.exit.switch_bottleneck` | status-inference | `v6.py:1757` | 派生 | **S** | 随 plateau，是当时的判定 | |
| A222 | `ext.exit.{reason,trigger}` | mapping | `v6.py:1743` | 部分 | **D** | phase exit 时已知 | |
| A223 ! | `ext.failure.*` | status-inference | `v6.py:1775` | 派生 | **D** | 失败原因产生时已知 | |
| A224 ! | `ext.critic_reviews[]` | dedup-union | `v6.py:1595` | 部分 | **D** | | |
| A225 | `ext.policy.*` | rollup (last-wins) | `v6.py:481` | 是 | **D** | 取最新 operation 值 | |

plateau / streak 一族（A209、A210、A212、A215、A216、A217、A221）是整表里最需要快照的一组：它们是**运行时用来决定要不要切换 arm 的输入**，导出时按当前窗口重算出来的值可能和当时的判定相反。

---

## 22. outcome / metadata / close

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A226 ! | `outcome.status` | status-inference | `v6.py:2113` | 部分 | **D** | | |
| A227 !! | `outcome.stage_reached` | funnel | `v6.py:2034` | 部分 | **S** | PRELUDE 内有多层键探测 fallback。改为每进入一个 stage 录一行，`stage_reached` 取最后一行 | |
| A228 ! | `outcome.validation.attribution.by_source.{s}.total_gain_pct` | sum | `v6.py:2127` | 是 | **D** | | |
| A229 | `outcome.validation.attribution.by_source.{s}.keep_count` | sum | `v6.py:2135` | 是 | **D** | | |
| A230 | `outcome.validation.attribution.by_source.kernel.by_backend.geak.non_attributable_keep_count` | sum | `v6.py:2137` | 是 | **D** | | |
| A231 | `outcome.final.gain_pct` | rollup | `v6.py:2173` | 是 | **D** | 随 A103 | |
| A232 | `outcome.validation.attributed_gain_pct` 等 | rollup | `v6.py:2179` | 是 | **D** | | |
| A233 | `metadata.task_config.architecture.model_class` | status-inference | `v6.py:75` | 部分 | **D** | | |
| A234 | `metadata.warnings[]` | rollup | `exporter.py:641` | 派生 | **D** | | |
| A235 !! | `close.status` | status-inference | `v6_close.py:195` | 部分 | **S** | **首次 build 必为 degraded**（breakdown 自己就是 close step 之一）。close 序列完成时录快照，走 `patch_breakdown_close` | |
| A236 | `close.start_time` | min | `v6_close.py:215` | 是 | **D** | | |
| A237 | `close.end_time` | max | `v6_close.py:216` | 是 | **D** | | |
| A238 | `close.robustness.escalated` | status-inference | `v6_close.py:226` | 是 | **D** | | |
| A239 | `close.artifacts.artifact_package_path` | status-inference | `v6_close.py:140` | 是 | **D** | | |

---

## 23. KB 三事件

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A240 ! | `timeline[warm_start].status` | status-inference | `kb_timeline.py:184` | 是 | **D** | 有 recipe+tier fallback 推断 | |
| A241 | `timeline[warm_start].ext.matched.experience.lessons_count` | count | `kb_timeline.py:220` | 是 | **D** | | |
| A242 | `timeline[warm_start].ext.matched.experience.pitfalls_count` | count | `kb_timeline.py:221` | 是 | **D** | | |
| A243 | `timeline[warm_start].ext.reads.count` | count | `kb_timeline.py:158` | 是 | **D** | 注意现在只取最后 50 行 | |
| A244 | `timeline[warm_start].ext.reads.hits` | count | `kb_timeline.py:148` | 是 | **D** | | |
| A245 | `timeline[warm_start].ext.reads.by_resolution.{r}` | grouping | `kb_timeline.py:144` | 是 | **D** | | |
| A246 | `timeline[warm_start].ext.reads.by_remote.{r}` | grouping | `kb_timeline.py:146` | 是 | **D** | | |
| A247 | `timeline[warm_start].ext.reads.by_source.{s}` | grouping | `kb_timeline.py:151` | 是 | **D** | | |
| A248 | `timeline[warm_start].ext.reads.best_config_by_source.{s}` | grouping | `kb_timeline.py:153` | 是 | **D** | | |
| A249 | `timeline[warm_replay].status` | status-inference | `kb_timeline.py:295` | 是 | **D** | | |
| A250 !! | `timeline[warm_replay].ext.before_tput` | ratio | `kb_timeline.py:330` | 部分 | **S** | 缺失时**从 after 和 gain 反解**。re-baseline 之后除法锚点是错的。replay 开始时录真值 | |
| A251 ! | `timeline[warm_replay].ext.result_type` | status-inference | `kb_timeline.py:304` | 部分 | **D** | reason 子串分桶 | |
| A252 !! | `timeline[warm_replay].ext.accuracy.passed` | status-inference | `kb_timeline.py:417` | 部分 | **S** | 从 status / eval_ran **猜**是否通过精度校验。这是结论性字段，必须录真值 | |
| A253 | `timeline[warm_replay].ext.applied.kernel.{total,kept,reverted}` | count | `kb_timeline.py:375` | 是 | **D** | | |
| A254 ! | `timeline[kb_write_back].status` | status-inference | `kb_timeline.py:532` | 是 | **S** | publish 中途被 kill 时 pending→failed 是猜的 | |
| A255 | `timeline[kb_write_back].ext.result_type` | status-inference | `kb_timeline.py:539` | 是 | **D** | | |
| A256 | `timeline[kb_write_back].ext.queue.pending_lines` | count | `kb_timeline.py:502` | 是 | **D** | | |
| A257 | `timeline[kb_write_back].ext.queue.flushed_bookmarks` | count | `kb_timeline.py:502` | 是 | **D** | | |
| A258 | `timeline[kb_write_back].ext.queue.dead_letter_lines` | count | `kb_timeline.py:502` | 是 | **D** | | |

---

## 24. exporter 内一致性

| ID | 字段 | 类型 | 计算位置 | 事实已录制? | 建议 | 理由 | 决策 |
|----|------|------|----------|-------------|------|------|------|
| A259 | `phase_timeline[]` merge 去重 | dedup-union | `exporter.py:35` | 部分 | **X** | 单一源后不需要 merge | |
| A260 | `warnings[]` | rollup | `exporter.py:245` | 派生 | **D** | | |
| A261 | `warnings[]` geak 一致性检查 | status-inference | `exporter.py:430` | 派生 | **D** | 保留做守卫 | |
| A262 !! | `kernel_journey.kernels[].discovery.*` backfill | imputation | `exporter.py:155` | 部分 | **X** | 用 roofline 表按 kernel_id/name join 补空值。discovery 时录全 | |
| A263 !! | `kernel_journey.kernels[].bound_type` backfill | imputation | `exporter.py:203` | 部分 | **X** | 同上 | |
| A264 | `optimizations.available==false` tripwire | status-inference | `exporter.py:746` | 派生 | **D** | 保留做守卫，正是它发现了 fragment 缺失 | |
| A265 | `metadata.warnings` 二次去重 | dedup-union | `exporter.py:1009` | 派生 | **D** | | |

---

## 25. attribution（当前未导出）

`collect_attribution` 的 816 行不在 `exporter.build()` 里，输出不出现在 `session_breakdown.json`。V6 的归因走 `outcome.validation.attribution`（A228–A230），源是 `optimizations.summary_by_source`。

建议整文件删（方案文档 §6.1 已列）。如果你想保留更细的归因维度（按 phase / domain / scope / PR / kernel_id / lever 分解），需要单独说——它现在的实现里 `phase_breakdown.geak.by_kernel_id` 有整行增益重复计入多个 kernel 的问题。

---

## 汇总

按全快照模型的作用域分布：

| 作用域 | 条数 | 落点 |
|--------|------|------|
| **E** 事件级 | 约 96 | `finish_event`，复用已有的 `assemble_*_ext` |
| **P** 决策级 | 约 42 | adoption / promote / plateau 判定时刻 |
| **C** session 级 | 约 78 | close 序列 |
| **X** 不记录 | 17 | 见下 |
| **DUP** V6 已覆盖 | 约 18 | 见 `SBD_V6_DEDUP.md` |

**E 占最大头，且不需要新机制**——kernel / roofline / baseline 三类事件的 `finish_event` 早就在做这件事，缺的只是把还没录的行补进去。

### 原本就无法推导的 26 条（风险最高）

即使在推导模型下这些也必须快照，说明它们的信息在事后是真的丢了。全快照模型下它们仍应优先处理：

1. **运行时决策的输入**（7 条，全部 **P**）：framework plateau / streak 一族 A209、A210、A212、A215、A216、A217、A221。导出时按当前窗口重算可能与当时判定相反。
2. **导出时刻依赖**（5 条，**C**）：A128、A131 lane 占用查的是导出那一刻未过期的 lease；A177、A180 未结束 session 的时长每次导出都变长；A172 首次 build 早于 langfuse flush。
3. **结论性数值必须钉死**（7 条，全部 **P**）：A80 adoption 的 before/after、A88 单步增益、A103 已验证总增益、A117 验证时栈长、A198 GEAK 每 cycle 增益、A250 replay 基线、A252 精度是否通过。
4. **状态机不可事后判定**（5 条）：A33 kernel 竞态（**E**）、A227 stage_reached（**P**，改为每进 stage 录一行）、A235 close 首次必为 degraded（**C**）、A254 publish 中途被 kill（**P**）。

### X 类 17 条

其中 8 条（A76、A84、A90、A102、A174、A183、A205 及 `gain_method` 一族）是「我这个值是猜的还是真的」的自述字段。它们的存在本身是投影模型的产物——产生时录制后没有意义。

另 9 条是纯启发式或恒空：A34、A41、A42 从文件名和 report 尾部反推 kernel 身份，A173 读内存 emitter，A203 猜 GEAK 失败原因，A204 取文件 mtime，A259、A262、A263 是 exporter 的 merge 与 backfill。

### 与去重的交叉

约 18 条被 `SBD_V6_DEDUP.md` 判定为 V6 timeline 已覆盖，主要集中在 §20 geak（A198–A206 大部分）和 §10 validation 头部。

注意 A198 `geak.gain_pct` 属于**假重复**：V6 把它拆成了 `ext.geak.claim.self_reported_gain_pct`（GEAK 自报，未验证）和 `ext.geak.rebench.attempts[].delta_pct`（orchestrator 复测，采纳依据）两个来源明确的字段。应采用 V6 的拆分，废弃 legacy 的模糊字段，而不是当作重复保留其一。
