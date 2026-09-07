# V6 timeline 已覆盖内容与去重

commit `da54496de`。

前置决定：**取消一切读时推导，全部走产生式录快照。**

---

## 0. 快照落在哪：按作用域定产生点

「产生式录快照」对单点事实是明确的，对聚合量需要指定"哪一刻算产生"。答案由聚合的作用域决定，三种，都已有现成机制或明确时刻：

| 作用域 | 快照时刻 | 机制 | 例子 |
|--------|----------|------|------|
| **E 事件级** | `finish_event` | **已存在**：`assemble_baseline_ext` / `assemble_roofline_ext` / `assemble_kernel_ext` 在事件结束时把该事件所有行汇总写进 timeline | kernel 漏斗计数、roofline delta、baseline action 采纳值 |
| **P 决策级** | 决策发生那一刻 | 在决策点直接写 fragment | plateau 判定、adoption 前后吞吐、验证总增益 |
| **C session 级** | close 序列 | close 时读齐 fragment、算一次、写快照 | token 总量、GPU 采样聚合、capability_summary |

**E 是关键**：它说明这套做法不是新增机制，而是把 `finish_event` 已有的模式推广到全部。kernel / roofline / baseline 三类事件早就是这么干的。

关于 **C** 需要说清一件事：close 时算一次仍然是在做算术，只是把它从"每次导出算一遍"挪到了"一辈子算一次并冻结"。收益是结果不再随导出时刻变化、可复现、re-export 幂等。如果你要的是连这次算术都不要，那唯一替代是用 `record_upsert_singleton`（`recorder.py:322`）在每条事实到达时增量累加——但它只有进程内锁，不跨进程安全，而且 resume 后累加起点难保证。**建议 C 类走 close 时算一次**。

---

## 1. 结论速览

| 项 | 结果 |
|----|------|
| 可整段删除且零信息损失的 legacy 段 | **0 个** |
| 移除已定的 X 类字段后可整段删除的 | **1 个**（`geak`） |
| 真重复（同语义同测量点） | 约 70 个字段 |
| **假重复**（同名不同测量点，必须保留区分） | 约 12 组 |
| V6 内部自我重复 | 3 处 |

---

## 2. 真重复：V6 已完整覆盖，legacy 不再记录

### 2.1 `geak` → `timeline[kernel].ext.geak`

这是重叠度最高的一段。以下 legacy 字段与 V6 同语义同测量点，**只记 V6 一份**：

| legacy `geak.*` | V6 路径 |
|-----------------|---------|
| `engaged` | `ext.geak.engaged` |
| `status` | `ext.geak.delegation.runner_status` |
| `error_class` / `error` / `returncode` | `ext.geak.delegation.{error_class,error,returncode}` |
| `exp_root` / `eval_dir` / `report_path` | `ext.geak.delegation.{exp_root,eval_dir,report_path}` |
| `stages_reached` | `ext.geak.delegation.stages_reached` |
| `runner_timeout_s` / `kill_timeout_s` | `ext.geak.delegation.{runner_timeout_sec,kill_timeout_sec}` |
| `handoff.{model_path,framework,gpu_type,tp,workload,raw_baseline_tput,bench_client}` | `ext.geak.handoff.*` 同名 |
| `handoff.accepted_flags` | `ext.geak.handoff.baseline_flags`（改名） |
| `kernels_optimized` | `ext.geak.claim.kernels_optimized` |
| `validated_regimes` | `ext.geak.claim.validated_regimes` |
| `accepted_kernels[].{kernel_id,name,op_kind,e2e_delta_pct}` | `ext.geak.claim.authored_kernels[].{kernel_id,short_name,op_kind,e2e_delta_pct}` |
| `accepted_heads[]` | `ext.geak.claim.env_selections[]` |
| `accepted_config` | `ext.geak.product.accepted_config` |
| `final_launch_script` / `bench_script` / `final_patch` | `ext.geak.product.*` 同名 |

legacy `geak` 的 UNIQUE 字段只剩六个：`opbench_results`、`runner_log_tails`、`likely_cause`、`flushed_result_status`、`last_artifact_ts`、`recovered_from_disk`。**这六个全部是磁盘重建路径的产物**，已在审计表标为 X（A203–A205）。

> **`geak` 段可整段删除**，前提是 GEAK runner 结束时主进程把 `ext.geak` 各块录全（方案 §3.3）。这是唯一一个零损失整段可删的 legacy 段。

### 2.2 `kernel_journey` → `timeline[kernel].ext.geak.attempts`（仅 GEAK 路线）

| legacy | V6 |
|--------|-----|
| `discovery_runs[].{source,status,hot_kernel_count,scan}` | `ext.geak.attempts.discovery_runs[]` 同名 |
| `kernels[].{kernel_id,dispatched,backends,skip_reason,task_group}` | `ext.geak.attempts.kernels[]` 同名 |
| `kernels[].backend_result.*` | `ext.geak.attempts.kernels[].backend_result.*` |
| `kernels[].e2e.*` | `ext.geak.attempts.kernels[].e2e.*` |
| `kernels[].micro_speedup` | `ext.geak.attempts.kernels[].backend_result.speedup` |

仍需单独记录的：`discovery_runs[].{ts,duration_sec,error}`（V6 discovery 行没有时间戳）、forge 路线的多 attempt 列表（V6 `ext.forge.lanes.kernel_rewrites[]` 只保留 adopted 形态）、`kernels[].roofline`（见 §4.2）。

### 2.3 `critic_robustness.robustness_signals` → `close.robustness.signals`

完全一致，原样透传。只记一份。

`critic_iterations[].framework_reviews[]` 同样已在 `framework_agent.ext.critic_reviews[]`。

### 2.4 `optimizations.validation` 头部 → `outcome.validation`

`attributed_total_gain_pct`、`unattributed_gain_pct`、`reconciliation_gap_pct`、`notes` 四项与 `outcome.validation.*` 同值。`summary_by_source` 的三源桶也已在 `outcome.validation.attribution.by_source.*`。

只记 `outcome` 一份。`optimizations` 的 attempt 级 ledger（`attempts[]` / `entries[]` / `backend_attempts[]`）是 UNIQUE，保留。

### 2.5 `final` 头部 → `outcome.final`

`throughput_tok_s_per_gpu`、`cumulative_gain_pct_validated`（→ `gain_pct`）、`action_path`、`extra_envs`、`extra_server_args` 五项重复。只记 `outcome`。

### 2.6 `baseline` 四个测量值 → `outcome.baseline`

`throughput_tok_s_per_gpu`、`accuracy`、`ttft_mean_ms`、`e2el_mean_ms` 与 `outcome.baseline.*` 同值，且同时又是 `timeline[baseline].ext.actions[].measurement.*`。三处同值。

处理：**事实只在 `timeline[baseline].ext.actions[].measurement` 记一次**，`outcome.baseline` 作为 close 时的 headline 快照保留（见 §5）。legacy `baseline` 段的这四个字段删除。

---

## 3. V6 内部自我重复

| 重复 | 处理 |
|------|------|
| `metadata.exported_at_utc` = 顶层 `exported_at_utc` | 删 `metadata` 里那份 |
| `metadata.versions.schema_version` = 顶层 `schema_version` | 删 `metadata` 里那份 |
| `metadata.versions.hyperloom` = `metadata.session.code_revision` | 保留 `session.code_revision`，删 `versions.hyperloom` |

---

## 4. 假重复：同名但不同测量点，必须保留区分

**这一节是本文档最重要的部分。** 以下字段名字相同或相近，但记录的是不同时刻、不同主体的测量，合并会直接产生错误数据。

### 4.1 GEAK 的三个增益

| 字段 | 是什么 | 谁测的 |
|------|--------|--------|
| `ext.geak.claim.self_reported_gain_pct` | GEAK 自己报的增益 | GEAK runner，**未经验证**（旁边有 `claim.verified: false`） |
| `ext.geak.rebench.attempts[].delta_pct` | orchestrator 复测的增益 | orchestrator，这才是采纳依据 |
| `ext.geak.handoff.raw_baseline_tput` | 交给 GEAK 时的基线 | orchestrator，交接时刻 |
| `ext.outcome.session_baseline_tput` | session 基线 | orchestrator，baseline 阶段 |

legacy 的 `geak.gain_pct` / `geak.final_throughput_tok_s` / `geak.throughput_speedup` 没有区分这些。**V6 的拆分是改进，不是冗余**——直接采用 V6 命名，废弃 legacy 那三个模糊字段。

### 4.2 roofline 的两层

`timeline[roofline]` 的 ext 里**没有** per-kernel 的 `bound_type`、`efficiency_percent`、`ceiling`。它只有 `outcome.snapshot_id`、`outcome.kernel_roofline_path` 这个路径，以及一个不含 bound 类型的 `hot_kernels` 摘要（只有 `gpu_time_us` / `gpu_pct`）。

所以 legacy `kernel_roofline.kernels[].{bound_type, arithmetic_intensity, rocprof_roofline, ...}` 是**真 UNIQUE**，必须新增录制，不能当重复删掉。

### 4.3 baseline 的 action 级与 round 级

`ext.actions[].measurement` 是**采纳值**；`ext.actions[].runs[].rounds[].measurement` 是**每一轮的原始测量，含被丢弃的 cold warmup 轮**。两者经常不等，且这个差异本身是诊断信息（`warmup_round_tput`、`cold_anchor` 就是为此存在的）。不合并。

### 4.4 `phase_timeline` 与 `timeline[]`

粒度不同：`timeline[]` 是 stage 级（一个 baseline 事件、一个 kernel 事件），`phase_timeline` 是 action 级（每次 dispatch 的尝试、决策、key_metric）。两者都需要。V6 事件里没有 action 流。

### 4.5 `capability_summary.geak.keeps` 与 `ext.outcome.by_source.*`

计数口径不同：`capability_summary` 用 integrate 裁决计 keep，`ext.outcome.by_source` 用 lane outcome 计。当前两者会不一致，正是因为口径没对齐。迁移时要**先定一个口径**，再只记一份——不要因为数字不同就都留着。

---

## 5. `outcome` / `metadata` 的定位

这两个是**刻意的摘要**，不是意外重复。`outcome.baseline` / `outcome.final` / `outcome.validation` 都能从 timeline 事件找到源。

在全快照模型下它们的定位应该明确为：**close 时刻的结论快照**，值从事件流取一次并冻结，同时记下它来自哪个 event id。这样：

- 数据只有一个产生点（事件），`outcome` 是引用而非二次测量
- 冻结后 re-export 幂等
- 出现分歧时能追到源事件

`metadata` 同理，但它多数字段是 session 开始时就固定的配置，直接在 bootstrap 时快照即可。

需要注意 `metadata` 现在的投影是**裁剪过的**：`model_info` 的 19 个字段只投影了 5 个，`versions` 丢了每个工具的 `root_dir` 和 `commit`，`langfuse` 丢了 7 个字段。这些不是重复，是丢失（方案 §3.1 已列）。

---

## 6. `outcome.stage_reached` 不是重复

我原以为它能从 timeline 推出来，实际不能：

- `profile` 和 `enablement` 会出现在 `stage_reached`，但 timeline 里**没有对应事件类型**
- `close` 出现在 `stage_reached`，但按 schema 注释**故意不进 timeline**
- PRELUDE 内部的子阶段（roofline / profile / warm_replay / enablement / baseline / warm_start）靠探测 20 多个 state 键决定，不等价于"最后一条 timeline 事件的类型"

它是独立的深度标签。按全快照模型，应改为**每进入一个 stage 就录一行**，`stage_reached` 取最后一行（审计表 A227 已标 S）。这样也顺带干掉那 20 多个键的探测 fallback。

---

## 7. 落到删除清单的净变化

在方案文档 §6 的基础上，去重带来的额外删除：

| 内容 | 依据 |
|------|------|
| `geak` 整段（`collectors/geak.py` 1053 行全删） | §2.1，唯一零损失整段可删 |
| `kernel_journey` 的 GEAK 路线部分 | §2.2 |
| `critic_robustness.robustness_signals` | §2.3 |
| `optimizations.validation` 的 4 个头部字段 + `summary_by_source` | §2.4 |
| `final` 的 5 个头部字段 | §2.5 |
| `baseline` 的 4 个测量字段 | §2.6 |
| `metadata.exported_at_utc` / `metadata.versions.schema_version` / `metadata.versions.hyperloom` | §3 |

**不要**因为看起来重复而删的（§4 全部）：GEAK 三个增益的拆分、`kernel_roofline` 的 per-kernel bound 数据、baseline 的 round 级测量、`phase_timeline`、`outcome.stage_reached`。
