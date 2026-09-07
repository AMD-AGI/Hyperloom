# V6 最终保留字段

commit `da54496de`。基于全快照（产生式录制）模型。

标记：`[争]` = 有争议或待决；`[缺]` = V6 当前没有、必须新增录制；`[新]` = 新增结构。

---

## 0. 最终顶层结构

```
session_breakdown.json
├── schema_version          常量
├── exported_at_utc         导出时间
├── exporter_version        导出器版本
├── metadata                会话身份 + 任务配置（含 metadata.session）
├── outcome                 会话结论
├── timeline[]              全部事件
├── close                   收尾序列
├── enablement          已降为 timeline 事件（`enablement:0:enablement`），顶层段待删，见 §5
├── telemetry           [争] 现状基本不可用，见 SBD_V6_TELEMETRY.md
├── warnings                导出警告
└── source_files            产物路径清单
```

删除的 26 个 legacy 顶层键见 §7。

---

## 1. `metadata`

保留 `metadata.session`，删除顶层 `session` / `session_meta`。

### 1.1 `metadata.session`

| 字段 | 含义 | 业务来源 |
|------|------|----------|
| `session_id` | Hyperloom 会话 ID | bootstrap |
| `claw_session_id` | SaFE/Claw 会话 ID | bootstrap |
| `sandbox_user_id` | 沙箱用户 | bootstrap |
| `created_at_utc` | 首次创建时间 | bootstrap |
| `start_ts` | 预算锚点 | bootstrap；resume 后仍是预算锚点，不再兼任 `elapsed_minutes` 的起点 |
| `ended_at_utc` | 结束时间 | close |
| `host` | 主机名 | bootstrap |
| `session_dir` | 会话目录 | bootstrap |
| `user_data_path` | 数据根 | bootstrap |
| `code_revision` | 代码 revision | bootstrap |
| `pid` | 进程 ID | bootstrap |
| `max_minutes` | 时间预算 | 启动参数 |
| `elapsed_minutes` | 本次运行腿的实际时长 | 每次 save 快照，最后一次即 close 值；起点是 `resumed_ts or start_ts` |
| `total_elapsed_minutes` `[新]` | 跨 resume 累计实际时长 | 已结束各腿在 resume 边界 banked（`SharedState.prior_legs_elapsed_s`）+ 本腿 |
| `tick_count` | tick 数 | 循环计数，close 快照 |
| `image` | 容器镜像 | `session/manifest.py:179` `_detect_image`，manifest 落章时录 |
| `image_id` | 镜像 tag 后缀 | 同上 |
| `recovery.recovered` | 是否经历 crash/resume | crash 记录 |
| `recovery.crash_count` | crash 次数 | crash 记录 |
| `recovery.crash_timestamps` | 各次 crash 时刻 | state 存 epoch，录制时转 ISO |
| `recovery.degraded_mode` | 是否降级 | state |
| `recovery.resume_pending_revalidation` | resume 后待复验 | state |
| `recovery.last_tick_exception` | 最后一次 tick 异常 | state，录紧凑头部、丢 traceback |

### 1.2 `metadata.task_config`

| 字段 | 含义 | 业务来源 |
|------|------|----------|
| `model_name` / `model_path` | 模型 | 启动参数 |
| `framework_name` / `framework_version` | 框架 | 启动参数 + 探测 |
| `gpu_type` | GPU SKU | 探测 |
| `tp` / `conc` / `isl` / `osl` | 并行与负载形状 | 启动参数 |
| `precision` | 精度 | 启动参数 |
| `max_model_len` | 最大上下文 | 模型配置 |
| `objective` | 优化目标 | 启动参数 |
| `launch_env` | 启动额外 env | operator 输入 |
| `launch_server_args` | 启动 server args | operator 输入 |

**`architecture` 子块——19 个字段已全部补回**（外加 `has_shared_expert`、`num_shared_experts`）：

`model_class`、`model_family`、`model_type`、`architectures`、`attention_type`、`num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、`head_dim`、`hidden_size`、`intermediate_size`、`max_position_embeddings`、`vocab_size`、`torch_dtype`、`kv_cache_dtype`、`quantization`、`is_moe`、`num_experts`、`num_experts_per_tok`

来源都是模型 config 解析，`model_class` 为派生（缺省按 dense/moe 分），其余整体搬运不再摘要。

### 1.3 `metadata.versions`

| 字段 | 含义 | 业务来源 |
|------|------|----------|
| `framework` / `framework_version` | 框架版本 | 探测 |
| `tools.{tool}.version` | 工具版本 | `instrument.record_tool_version` |
| `tools.{tool}.commit` | 工具 commit | 同上，整行保留不再压扁 |
| `tools.{tool}.root_dir` | 工具路径 | 同上 |

工具集：geak、tracelens、forge、claude、codex、inferencex、kernel_agent。

`versions.schema_version`（与顶层重复）与 `versions.hyperloom`（与 `session.code_revision` 重复）已删除。

`task_config.framework_version` 的产生点补在两处：manifest 落章时取 `manifest.framework_version`，之后每次 save 取 `state.framework_version`——**仅在非空时写**，因为该 state 字段可能一直为空，而 singleton 的叶子合并没有"空值不覆盖"的概念。

### 1.4 `metadata.langfuse`

按你的决定，**在 recorder 处直接写入**，不再事后读 receipt。已落地。

| 字段 | 含义 | 业务来源 |
|------|------|----------|
| `enabled` | 是否启用 | 配置，bootstrap 录一次 |
| `disabled_reason` | 未启用原因 | 配置判定处 |
| `trace_id` / `session_id` | trace 标识 | emitter 建 trace 时 |
| `trace_url` | trace 链接 | host + trace_id |
| `counts` | 各类 span 计数 | **flush 完成时录快照**，走已有的 `patch_breakdown_langfuse` |

`receipt_source`（自述取自哪个 tier）与 `counts_final`（快照后恒 true）已删除。

---

**§1 状态：已完成。** 唯一遗留是 `framework_version` 与 `versions.framework*` 仍有 collector 投影兜底，需等 `collect_workload` 随 P3 退役；其余字段均已 author-time 录制。

---

## 2. `outcome`

定位：**close 时刻的结论快照**，值从事件流取一次并冻结，记下来源 event id。

| 字段 | 含义 | 业务来源 |
|------|------|----------|
| `stop_reason` | 原始停止原因 | 循环退出点 |
| `status` | completed / failed / aborted | stop_reason + install/model_gate 失败 |
| `stage_reached` `[缺]` | 到达的最深阶段 | **改为每进入一个 stage 录一行**，取最后一行。当前靠探测 20+ 个 state 键；`enablement` 已有事件类型（改读 timeline，旧探测只留给老会话），`profile` / `close` 仍无 |
| `baseline.throughput_tok_s_per_gpu` | 基线吞吐 | 引用 `timeline[baseline].ext.actions[].measurement` |
| `baseline.accuracy` | 基线精度 | 同上 |
| `baseline.ttft_mean_ms` / `e2el_mean_ms` | 基线延迟 | 同上 |
| `final.throughput_tok_s_per_gpu` | 最终吞吐 | 最后一次采纳后的测量 |
| `final.gain_pct` | 验证累计增益 | 见 `validation` |
| `final.action_path` | 动作栈路径 | 优化栈 |
| `final.extra_envs` / `extra_server_args` | 最终配置 | 优化栈 |
| `final.ttft_mean_ms` / `e2el_mean_ms` `[缺]` | 最终延迟 | 当前 outcome 没有，只有 baseline 有 |
| `validation.attributed_gain_pct` | 各步贡献之和（同一分母） | `timeline[stack].ext`，见 §2.1 |
| `validation.unattributed_gain_pct` | 采纳之外的锚点移动 | 同上，等于 `guards.chain_breaks` 之和 |
| `validation.chain_total_gain_pct` | 末次采纳对 baseline 的读数 | 同上，一次测量而非求和 |
| `validation.validated_total_gain_pct` | 整栈独立复测值 | 同上 |
| `validation.reconciliation_gap_pct` | 整栈实测与链总量之差 | 同上，**这才是该告警的数** |
| `validation.attribution.by_source.*` | 分来源归因 | 同上，含 `kernel.by_backend` |
| `validation.guards` | `unmeasured` / `chain_breaks` | 同上 |
| `validation.notes` | 对账发现项 | 由录制行推出，为空才是好结果 |

### 2.1 validation 的数据来源：`stack` 事件（已落地）

`outcome.validation` 原先**直接读 `optimizations.validation` 和 `optimizations.summary_by_source`**（`collectors/v6.py:505-582`），而 `optimizations` 自己又是从 v4 的 `operations` / `measurements` / `adoptions` 三条流重建出来的。三个问题，其实是同一个错误：**在某一刻是事实的数字被丢掉了，事后从残留里重建**。

**一、每次采纳打败的那个吞吐没被录下来。** `entries[]` 只有 `throughput_after`，没有 `throughput_before`；chain 上的 before 是靠遍历 attempt 列表配对推出来的。而那个数字就是 `GradedComparison.reference`——lift 拒绝任何打不过它的 winner，所以它在 `_lift_to_current_best` 里精确存在，并且被紧随其后的 `current_best` 写入覆盖掉。**那一刻是唯一能录它的时机。**

**二、各步增益是在不同分母上求和的。** 每步增益按上一步的吞吐算，然后相加——比率不是这么复合的。10/100 + 10/110 + 10/120 = 26.5%，而链实际涨了 30%，差出来的 3.5% 被发布成 `unattributed_gain_pct`，好像是"谁都认不下的增益"，其实是算术产物。现在每步的贡献都对**会话 baseline** 算（三步共享的那个分母），10+10+10=30 正好等于链总量，`unattributed_gain_pct` 于是只承载它本该承载的那一件事：**两次采纳测量之间锚点自己动了多少**，没有任何采纳认领。这是恒等式，不是估计——它等于 `chain_breaks` 之和。

**三、8 个守卫计数是为了发现不一致而存在的。** `stale_evidence_count`、`unclaimed_integration_count`、`unscored_keep_count` 这些，每一个都是在检查那三条 v4 流是否互相对得上。**在事实成真那一刻录一次的数据，没有可以互相矛盾的对象**，所以多数守卫在新结构里没有指代。只留下两个和"测量本身"有关而不是和"记账"有关的：`unmeasured`（贡献无法测量的采纳数）和 `chain_breaks`（锚点在采纳之外移动的次数）。

**落地形态**：`timeline[stack]`，事件 id `stack:0:stack`。phase 段是字面量而不是 `state.phase`——栈是一个对象，PRELUDE warm replay / EXPLORE / FRAMEWORK_AGENT / KERNEL_AGENT 的采纳构成**一条有序链**，按 phase 切事件会正好在链的 before/after 需要对上的地方（phase 边界）把它切断。

三个收口：

| 收口 | 录什么 |
|---|---|
| `writeback._lift_to_current_best` 的 `optimization_stack.append` | 一行 `stack_adoption`，键为 `stack_index`；before/after/baseline 三个吞吐全在手 |
| `writeback._update_cumulative_gain_validated` | 一行 `stack_validation`，键为 `stack_len`；同一长度上的复测覆盖前值（复测是对前值的修正，不是第二个事实） |
| `close._close_stack_ledger` | 关事件、算对账。没走到 close 的 run 留 `interrupted`——对账确实没跑过 |

录在 append 而不是 lift 返回处：lift 对已在栈里的配置会**跳过 append** 直接重锚，录在返回处会把同一个优化算两遍——正是对账要抓的那类重复计数。测试 `test_an_already_stacked_config_is_not_recorded_twice` 钉住这条。

`by_source` 五桶（`warm_replay` / `explore` / `framework_agent` / `kernel` / `unattributed`）在 assembler 里从行上的 `action` 分桶，`kernel` 再按 `backend` 分 geak / forge / unattributed。空桶也出现且置零，这样"这个子系统什么都没挣到"和"这个子系统没被统计"读起来不一样。backend 不在 lift 作用域内的（integrate 路径只有 kernel_id）折进 `unattributed`，保证 backend 分项之和等于 kernel 桶——分项不会悄悄少报。

`collect_v6_outcome` **已不再接收 `optimizations` 参数**。阻塞 `optimizations` 删除的耦合是从结构上断的，不是靠约定。

---

## 3. `timeline[]`

### 3.0 事件类型总表

| 类型 | 现状 | 说明 |
|------|------|------|
| `install` | 已实时录制 | preflight |
| `model_gate` | 已实时录制 | 模型准入 |
| `baseline` | 已实时录制 | 基线测量 |
| `roofline` | 已实时录制，但**缺三大块** | 见 §3.4 |
| `kernel` | 已实时录制，但**缺 lifecycle 与 profiling** | 见 §3.5 |
| `framework_agent` | 已实时录制（五子流），投影已删 | 见 §3.6 |
| `conc_sweep` | 仍是投影，**缺一大批** | 见 §3.7 |
| `warm_replay` | 已实时录制（门控逐条落地），投影待删 | 见 §3.8 |
| `warm_start` / `kb_write_back` | 仍是投影 | 见 §3.8 |
| `critic` `[新]` | 不存在（但 `critic_iterations` 已按轮录制，缺的是 per-proposal 粒度） | 见 §3.9 |
| `enablement` | 已实时录制（触发/每轮/构建/复验各一行），顶层投影待删 | 见 §5 |
| `phase` `[新]` | 已实时录制（每段一事件，dispatch 行挂在段上） | 见 §3.1 |
| `stack` `[新]` | 已实时录制（每次采纳一行，一个 session 一个事件） | 见 §2.1 |
| `action` | **不设独立事件**（与 stage 事件的 per-dispatch 行重复） | 见 §3.1 |

### 3.1 `phase` 事件（吸收 phase_timeline / phase_segments）

**本节结论已改**。原方案是"新开一条 action 级事件流，吸收 `phase_timeline` / `phase_segments`"。做完 §3.3–§3.9 与 §5 之后这个方案不成立了，理由和替代方案如下。

#### 为什么不做平行的 action 事件流

原方案那一行（`ts` / `action` / `task_id` / `status` / `decision` / `key_metric` / `phase` / `macro_cycle` / `workspace` / `error_class` / `extras`）是各 stage 事件**已经录的** per-dispatch 行的真子集，而且更贫：

| stage 事件 | 已有的 per-dispatch 行 | 比原方案多出来的 |
|------------|----------------------|------------------|
| `baseline` | `ext.actions[]` | `runs[]` / `runs[].rounds[]` 两层、`attempt_reason`、原始测量含被丢弃的 cold warmup |
| `roofline` | `ext.actions[]` | `profile.runs[]`（trace validate / trace_health）、`analysis.runs[]`、25 字段 snapshot |
| `conc_sweep` | 扁平 ext + `superseded_sweeps` | 每 `(conc, batch)` 格点 |
| `framework_agent` | `ext.attempts[]` | `arm`、`provenance`、`critic`、`blocked_by` |
| `enablement` | `ext.attempts.rows[]` | `stack_action`、`localization_manifest`、`stall_streak_after` |

再录一条扁平流，等于同一件事在两处各写一遍，正是这轮重构要消灭的漂移源。所以**不新增 action 事件类型**。

#### 原方案真正指对的两件事（都跟 action 行无关）

1. **phase 归属现在是猜的。** `record_action_attempt` 明明把 `phase` / `macro_cycle` 传给了 `instrument.record_phase_event`（`shared_state.py:2918`），后者却只把它们喂给 v4 mirror，**payload 里丢掉了**（`instrument.py:849-859`）。于是 export 时 `collect_phase_segments` 只能拿 `[entered_ts, exit_ts)` 时间窗去套（`collectors/timeline.py:826-855`）。事实就在手边却不写，这是本文档整个论点的教科书案例。
2. **phase 段落在 timeline 上没有落点。** `phase_transitions` fragment 已经在录（`machine_state.py:3228`），但没有任何消费方把它发布出去；`phase_segments` 是 export 期从 `phase_history` 两两配对推的。timeline 因此没法回答"这个 baseline 事件是在哪个 phase 里跑的"。

#### 定案：`phase` 成为 timeline 事件

`phase_segments` 的行本来就是一个事件——`phase` + `entered_ts` + `exit_ts` + `exit_reason` + `evidence` + `events[]` + `actions[]` + `elapsed_seconds`，这是一段有始有终的时间跨度。按事件录，`phase_segments` 的推导和 `phase_timeline` 的时间窗猜测同时消失。

**事件 id**：`{phase}:{macro_cycle}:phase`，进入时开、离开时关。同一 phase 在不同 macro_cycle 各自成事件。

两个段：

**`phase_event`**（每段一行，`ext` 本体）

| 字段 | 含义 | 录制时刻 |
|------|------|----------|
| `phase` / `from_phase` / `macro_cycle` | 身份 | 进入（`record_phase_transition`） |
| `entered_at` / `entered_reason` / `entered_evidence` | 入场 | 进入 |
| `exited_at` / `exit_reason` / `exit_evidence` / `to_phase` | 离场 | 离开（下一次 transition） |
| `duration_sec` | 段长 | 离开时算，不再从 `phase_history` 配对推 |
| `markers[]` | 非 transition 的 `phase_history` 子事件 | 产生时 |

**`phase_action`**（每次 dispatch 一行）

| 字段 | 含义 | 录制时刻 |
|------|------|----------|
| `action` / `task_id` | 身份 | **dispatch**（`run_task_registered`） |
| `phase` / `macro_cycle` / `tick` | 归属 | **dispatch**——决定跑它的那个 phase 拥有它，不是它落地时的 phase |
| `dispatched_at` | 起点 | dispatch |
| `status` / `decision` | 结果与裁决 | settle（`_reap_dispatched_task`） |
| `settled_at` | 终点 | settle |
| `duration_sec` | 跨度 | 装配时按本行两端算 |
| `error_class` / `workspace` | 诊断 | settle |

细节不复制：`baseline` / `roofline` / `conc_sweep` / `integrate_patch` / `targeted_build` 的 per-dispatch 详情归各自 stage 事件，`phase_action` 只留身份 + 归属 + 裁决，**靠 `task_id` join 过去**——stage 事件的行本来就带 `task_id`。

一度打算加个 `event_ref` 字段直接指向 stage 事件，撤了：它需要一张 kind→component 的 author-time 静态表，而 `enablement` 的事件 id 还不是 phase 派生的（`enablement:0:enablement`），表里得开特例。那正是 `_AUDIT_ACTIONS` 覆盖 4/15 之后再没人维护的同一种脆弱。join key 不需要维护。

**只有没有 stage 事件的 action**（`report`、`recover`、`session_breakdown`、`target_analysis`）在这里是唯一记录，join 什么也找不到——这本身就是答案。

`count` 与 `settled` 的差是在飞或被中途杀掉的 dispatch。旧的 `phase_timeline` 表达不了这个：它从已 settle 的 audit 行生成，所以关机时被取消的 dispatch 一行都不留，读起来跟从没派过一样。

#### 收口位置

覆盖面不靠枚举 15 个 action，靠代码里本来就唯一的三个收口：

| 收口 | 位置 | 录什么 |
|------|------|--------|
| dispatch | `dispatcher.run_task_registered`（docstring 自称 "The only way an action runs"，`sub.run_task` 全库仅此一个调用点） | 开 `phase_action` 行 |
| settle | `dispatcher._reap_dispatched_task` 算 `kept` 的分支（promote 与 unpromotable 两路都过这里） | 合并裁决到同一行 |
| phase 切换 | `machine_state.record_phase_transition` | 关上一段、开下一段 |

`_AUDIT_ACTIONS` 那 4 个动作的白名单因此**不需要扩**：它是 `{action}_attempts` 这些 legacy state 列表的门控，与 timeline 覆盖面脱钩。

事件 id 是 `{phase}:{macro_cycle}:phase`。一个 macro_cycle 内重入同一 phase **不另开事件**——id 的三段必须全部可从持久化状态重算，没有第四段能区分两次进入——所以每次进入是一行 `phase_segment`，事件覆盖该 cycle 内这个 phase 的全部时间，`duration_sec` 按各段求和而不是首进到末出，否则会把中途跑在别处的时间算给它。

离场时的事件 id 靠**查回未关闭的段**定位，不信调用方传的 cycle：loopback 在离开 phase 的路上就把 `macro_cycle` 加了 1，照它算会去关一个从没打开过的事件，真正那个则永远悬着。

#### 附带修掉的两处

1. `instrument.record_phase_event` 现在把 `phase` / `macro_cycle` 写进 payload 了。legacy `phase_timeline` 要到第 9 项才删，在它还活着的这段时间里，归属不再是时间窗猜的。
2. 一处死代码：`writeback.py` 里 `conc_sweep` 的两处 `record_action_attempt`（unpromotable 分支与 `_promote_conc_sweep`）已删。`conc_sweep` 不在 `_AUDIT_ACTIONS` 里，该方法首行就 `return None`——那两个精心拼的 extras dict 每次 sweep 都拼一遍、扔一遍。这些事实（skip_reason / budget_exhausted / summary）在 conc_sweep 事件的 `result` 块里有真的。

### 3.2 `install` / `model_gate`

已完整。`install.ext` 有 `run_kind`、`hard_fail_step_id`、`runtime_snapshot`、`steps[]`（每步 `step_id` / `category` / `status` / `skip_reason` / `detail`）。`model_gate.ext` 有 `workload`、`checks[]`（三道 gate 各自 `verdict` + `detail`）、`degraded`、`failure`。

### 3.3 `baseline`

已完整，三层结构且**三层不是重复**：

- `ext.actions[]` — 一次 dispatched baseline 任务，`measurement` 是**采纳值**
- `ext.actions[].runs[]` — 一次 executor pass，带 `attempt_reason`
- `ext.actions[].runs[].rounds[]` — 一次 Magpie subprocess，`measurement` 是**原始测量，含被丢弃的 cold warmup 轮**

`measurement` 字段：`throughput_tok_s_per_gpu`、`ttft_mean_ms`、`e2el_mean_ms`、`tpot_mean_ms`、`accuracy`、`accuracy_task`、`accuracy_metric`、`accuracy_source`、`benchmark_report_path`、`workspace`。

补充需要的 `[缺]`：`ttft_e2el_source`（哪里读到的延迟）、`invocation.framework_args` 与 `framework_args_source`（现在是导出时从 server.log 和 yaml 里 parse，应改为启动时上报）、`failure_streak` / `total_failures`。

### 3.4 `roofline` — 三大块缺失

现有的是 action 级执行记录：`profile.runs[]`（含 trace validate、trace_health）、`analysis.runs[]` 与 `effective_run`、`outcome`（snapshot_id 与各 artifact 路径）。

**缺失一：per-kernel roofline 表** `[缺]`
`timeline[roofline]` 里**没有** `bound_type`、`efficiency_percent`、`arithmetic_intensity`，只有一个 `kernel_roofline_path` 路径和一个不含 bound 类型的 hot_kernels 摘要（只有 `gpu_time_us` / `gpu_pct` / `category`）。

要补的每 kernel 字段：`kernel_id`、`name`、`source_file`、`kernel_category`、`bound_type`、`arithmetic_intensity`、`flops_per_byte`、`efficiency_percent`、`gpu_pct`、`call_count`、`duration_us`、`reusable_native_kernel`、`rocprof_roofline`、`bottleneck`、`suggestion`、`compute_utilization_pct`、`bandwidth_utilization_pct`、`recommended_actions`、`roofline_source`。

产生点：TraceLens 路由在 `agents/kernel/tools/tracelens_analysis.py` 末尾写 `reports/kernel_roofline.json`；bypass 路由走 `_bypass_report.build_kernel_roofline`。

**缺失二：snapshot 全字段** `[缺]`
`ext.actions[].outcome` 只有 `snapshot_id`。完整 snapshot 有约 25 个字段：`theoretical_peak_tok_per_sec`、`roofline_mem_ceiling_tok_per_sec`、`roofline_cmp_ceiling_tok_per_sec`、`roofline_bound_kind`、`achieved_tok_per_sec`、`within_roofline_pct`、`gap_to_roofline_pct`、`roofline_ceiling_exceeded`、`compute_pct` / `idle_pct` / `comm_pct`、`top_bottleneck`、`top_kernel.*`、`e2e_mean_ms` / `roofline_ideal_ms`、`roofline_provenance.*`、`perfmodel_breakdown.ops[]`。

产生点：`SharedState._append_roofline_snapshot_history`，由 `record_trace_analyze` 在每次成功分析后调用——**时间上就在 `finish_event` 之前**，recorder 完全够得着，只是现在没读。

**缺失三：progress 轨迹** `[缺]`
`ceiling_tok_per_sec`、`target_tok_per_sec`（ceiling × 0.70）、`trajectory[]`（baseline 加每次 KEEP 的步进曲线）、`current_best_pct_of_ceiling`、延迟域的那套、`roofline_failure_streak`。这是跨事件的 session 级量，close 时快照。

### 3.5 `kernel` — GEAK 完整，forge 侧缺口大

**`ext.geak` 基本完整**，可以吸收整个 legacy `geak` 段。子块：`handoff`（交接参数与基线）、`delegation`（runner 结局、stages_reached、versions）、`attempts.discovery_runs[]` / `attempts.kernels[]` / `attempts.counts`、`claim`（GEAK 自报，带 `verified: false`）、`product`（产物路径与配置）、`rebench`（orchestrator 复测裁决）。

你问的几条，答案是**都已覆盖**：

| 审计项 | V6 路径 |
|--------|---------|
| A5 `geak.attempts` | `ext.geak.attempts.counts.dispatched` |
| A7 `geak.keeps` | `ext.geak.attempts.counts.integrated` + `ext.outcome.adopted[]` |
| A9 `pending_integrate` | `ext.outcome.pending_review[]`（带 `why`） |
| A10 `micro_only_keeps` | `counts.backend_ok` 减 `counts.integrated` |
| A12 `geak.status` | `ext.geak.delegation.runner_status` + 事件 status |

删 `geak` 段要**补回**的（这些不是磁盘重建产物，是真丢失）`[缺]`：`ttft_mean_ms`、`tpot_mean_ms`、`output_parity`、`metric_basis`、`accepted_kernels[]` 的 `gpu_pct` / `micro_speedup` / `validated` / `decision` / `backend` / `target_file` / `aliases`。

**`ext.forge` 缺口大**，详见 `SBD_V6_KERNEL_LIFECYCLE_MERGE.md`。要点：

- `record_kernel_rewrite()` **生产代码根本没调用**，只有测试在用
- 现有行是 per-run 的且设计上只保留 adopted 形态，**多 attempt 历史会丢**
- `detected[]` 的 profiling 富字段（`duration_us`、`call_count`、`bandwidth_util_pct`、`compute_util_pct`、`recommended_actions`、`recommended_backends`）在 kernel 和 roofline 事件里**都不存在**

`ext.outcome`：`route`、`verdict`、`exit_reason`、`tput_before` / `after`、`net_gain_pct`、`session_baseline_tput`、`cumulative_gain_validated_out`、`stack_depth_out`、`adopted[]`、`pending_review[]`、`by_source`、`stack_delta`（当前未写值）。

### 3.6 `framework_agent` — 已实时录制（五子流）

**已完成。** 投影（v6.py 约 1500 行）已删除，事件由 phase 自己在产生时录制。

形状从双 arm 改为**五个子流，arm 降为行上的字段**。原来按 arm 切顶层的分法有两个表达不了的路径：orchestration agent 自己提的 config 提案、以及 seed grid 未经提案的变体，都没有 specialist 可挂，在 arm 树里无处落地。

`ext` 结构：

| 子流 | 内容 |
|------|------|
| `macro_cycle` | 构造 recorder 时盖上，跨 cycle 隔离是结构性的，不再靠时间窗匹配 |
| `policy.*` | `keep_threshold_pct`、`variant_timeout_sec`、`overtime_kill_ratio`、`force_exit_budget_pct`、`config.{lookback, keep_gain_threshold_pct, empty_streak_threshold}`、`source.{authoring_enabled, no_keep_streak_threshold, discovery_retry_limit}`。由各字段的 owner 在 open 时解析后录入 |
| `proposals[]` | `proposal_id`、`arm`、`producer` / `producer_ref`（归因）、`lever_kind`、`scope`、`run_ref`、`repo` / `title` / `source_ref` / `verdict`、`attempt_refs[]`、`lifecycle[]`（`step` / `ts` / `outcome` / `reason` / `run_ref`）、`terminal.{disposition, reason, settled_at}`、`critic_review.{verdict, reason, reviewer, reviewed_at}` |
| `runs[]` | `run_id`、`role`（discovery / authoring）、`arm`、`status`、`domain`、`reason`、`dispatched_at` / `completed_at`、`produced_ids[]` |
| `attempts[]` | 两个 arm 统一一种行：`attempt_id`、`arm`、`round_id`、`task_id`、`proposal_ref`、`provenance`、`outcome` / `decision` / `reason`、`adopted`、`attribution_eligible`、`measurement.{before_tput, after_tput, gain_pct, runtime_sec, estimated_output_throughput}`、`measured_against`（当时的栈）、`accuracy.{required, reference, value, passed}`、`gates[]`（`gate` / `passed` / `observed` / `threshold` / `ts`）、`failure`、`artifacts`；config 侧另有 `fingerprint` / `variant_name` / `config_delta` / `accepted_kernels`，source 侧另有 `candidate_id` / `source_ref` / `route` / `patch_source` / `patch_path` / `patches_applied[]` / `target_files[]` |
| `plateau[]` | 每次判定追加一行：`arm`、`path`（advisory / exit）、`evaluated_at`、`triggered`、`inputs`、`thresholds` |
| `exit.*` / `failure.*` / `duration_sec` | 同前 |

两个原 `[争]` 项的处置：

1. `by_source.framework_agent` 合并 `explore` 桶 —— **已定：保持不变**（见 §8 #7）
2. plateau 导出时重算 —— **已解决**。`record_plateau` 在判定那一刻把 `inputs` 与 `thresholds` 连同 `triggered` 一起快照，不再事后按 lookback 窗口重算（见 §8 #8）

老形状（`config_arm` / `source_arm` / `critic_reviews`）到新形状的对应关系：`config_arm.rounds[].variants[]` 与 `source_arm.attempts[]` → `attempts[]`（按 `arm` 区分）；`specialist_runs[]` / `candidate_discovery_runs[]` / `authoring_runs[]` → `runs[]`（按 `role` 区分）；`critic_reviews[]` → `proposals[].critic_review`；`*_arm.plateau` → `plateau[]`；`variants[].source.specialist_round_id` → `attempts[].proposal_ref` → `proposals[].run_ref` 链。

会话级字段不再逐行重复：`framework` 是 `metadata.task_config.framework_name`，`workload_signature` 是 `metadata.task_config.workload_signature`。`decision_tput` 与 `after_tput` 在写入方同值，已去掉。

### 3.7 `conc_sweep` — 缺一大批

现有投影覆盖：`plan.concs_requested`、`runtime.{workspace, elapsed_sec, budget_remaining_sec}`、`arms.{baseline,optimized}.points[]` 的部分字段、`comparison[]` 部分、`result.{status, best_conc, best_speedup, metric, budget_exhausted}`、`artifacts.*`。

**缺失** `[缺]`：

- 整块 `roofline_ceiling`（每 CONC 的理论峰值、bound_kind、baseline/optimized 的 MBU%）
- agentx 轴指标：`total_token_throughput`、`intvty_p90`、`tpot_p90_ms`、`request_throughput`、`input_throughput`
- summary 统计：`successful_pairs`、`failed_pairs`、`median_speedup`、`mean_speedup`
- comparison 的 `delta_pct`、`baseline_status`、`optimized_status`
- 会话上下文：`session_id`、`isl`、`osl`、`tp`、`benchmark_mode`
- point 级：`duration_seconds`、`completed_requests`、`killed_overtime`、`estimated_output_throughput`、`workspace`
- 两个 arm 的 `extra_envs`

另外 sweep 过程中已知但**从未落盘**的：boot-retry-descend 的每次尝试与失败 CONC、per-variant 的 NUM_PROMPTS、每 rung 的 budget cap 快照、单 server 与 Option B 的执行策略分支、lifecycle 资格判定、arm 顺序、per-variant 起止时间戳。

### 3.8 KB 三事件

`warm_start`：`matched.*`（命中的 recipe 与经验）、`reads.{count, hits, by_resolution, by_remote, by_source, best_config_by_source}`。

`warm_replay`：**已实时录制**，投影（`collect_warm_replay_event`）留到 §7 统一删。

事件形态：`request`（身份与阈值）/ `measurement`（测量）/ `gates[]`（逐门裁定）/ `applied`（实际跑的配置）/ `promotion` / `rollback` / `skip` / `verdict`。

| 子块 | 字段 |
|------|------|
| `request` | `task_id`、`baseline_action_ref`、`tier`、`config_source`、`config_donor_tier`、`donor.{canonical_id, model, session_id, family_tags, gain_pct, breakdown_link}`、`expected_gain_pct`、`confidence`、`min_reproduce_pct`、`session_baseline_tput`、`kernel_count`、`recipe_suppressed` |
| `measurement` | `before_tput`、`after_tput`、`gain_pct`、`hot_tput`、`cold_tput`、`accuracy`、`baseline_accuracy`、`eval_ran` |
| `gates[]` | `gate`、`passed`、`reason`、`observed`、`threshold`、`ts`；门有 `tput_valid` / `quality` / `accuracy` / `keep_threshold` / `promotion` / `params_present` |
| `blocked_by` | 结束这条弧的门；成功的弧为 `null` |
| `applied` | `extra_server_args`、`extra_envs`、`kernel.{status, total, kept, reverted, validation}` |
| `promotion` | `promoted_checkout`、`replayed_patch_refs`、`stack_entry` |
| `rollback` | `ok`、`errors` |
| `skip` | `code`、`reason`、`details` |
| `verdict` | `outcome_status`、`reason`、`error_class`、`keep_threshold_pct`、`below_historical_reproduce_pct`、`historical_reproduce_bar_pct`、`settled_at` |

投影消掉的四处失真：

1. `before_tput` 原先从 after 与 gain 反解，re-baseline 后锚点是错的；现在在用它的那一刻录下来。
2. `accuracy.passed` 原先从终态状态猜——精度拒绝与画质拒绝都表现为"未复现"，评估跑了但无分与根本没跑也分不开。现在每个门自己写裁定，`passed=None` 表示"跑了但无法裁定"，没跑到的门**不写行**,于是"没通过"和"不适用"可区分。
3. 跳过原因原先靠**子串匹配散文 reason** 归到 13 个码里（`"legacy native records do not satisfy"` 这类），改词就会静默改分类；现在在拒绝的那一处直接录稳定码。
4. `drift`（测了但没过保留阈值）原先在 recorder 的状态表里缺失会被读成 `failed`；它是**已裁定的拒绝**，不是失败。

另外新增了投影里完全没有的:`applied`（拒绝的重放也说明是什么输了，原先只能从晋升推的 stack entry 里恢复）、`baseline_accuracy`、`rollback`、`duration_sec`、门的 `observed`/`threshold`/`ts`。

历史复现门刻意**不记为门行**：它读的是实测增益对已声明增益的比例，但从不否决——过了保留阈值的重放照样晋升。记成 `passed=False` 会让 `blocked_by` 把一条成功的弧标成被它拦住，所以这个判定只进 `verdict`。

`kb_write_back`：`result_type`、`queue.{pending_lines, flushed_bookmarks, dead_letter_lines}`。

### 3.9 `[新]` `critic` 事件

按你要的统一，详见 `SBD_V6_CRITIC_UNIFY.md`。核心结论：**主粒度应该是 per-proposal verdict，而不是 per-turn**。现在 `critic_iterations` 是 turn 粒度、`framework_agent.ext.critic_reviews[]` 是 proposal 粒度，两者并存。

建议事件形状：

| 字段 | 含义 |
|------|------|
| `review_batch_id` | 同一次 critic turn 的分组键 |
| `subject.{proposal_msg_id, candidate_id, variant_name, action_name, decision_id}` | 被评审对象 |
| `verdict` / `effective_verdict` / `source` | 裁决 |
| `reasoning` / `confidence` / `risks` / `required_evidence` / `failure_reason_code` | 依据 |
| `advice_text` / `alternative_action` / `followup_task_ids` | 后续 |
| `kb.{priors_trace, writes[], referenced_in_verdict}` | KB 侧效应 |
| `artifacts.{request, judge_bundle, review, emit}` | 工件路径 |
| `downstream.{materialized, framework_denied, patch_verdict_key}` | 下游影响 |
| `phase` / `macro_cycle` | 归属，可为空 |

critic 每 tick 都跑，不绑定单一 phase，所以它**不应该只嵌在 framework_agent 里**——KERNEL 等 phase 的 review 现在产生了 `critic_iterations` 但不进任何 timeline 投影。

`[争]` 一个实际 bug：`robustness_signals` 期望的 `signal.json` / `action.json` **全代码库没有任何写入方**，robustness backend 只写 `request.json` 和 `emit.json`。所以 `close.robustness.signals` 在真实 session 里基本是空的，或只有 workdir 路径。要么补 writer，要么改读 `agents/robustness/findings/*.jsonl`。

---

## 4. `close`

| 字段 | 含义 | 业务来源 |
|------|------|----------|
| `status` | 收尾完整性 | **close 完成时快照**（首次 build 必为 degraded，因为 breakdown 自己就是 close step 之一） |
| `start_time` / `end_time` | 起止 | close steps |
| `close_sequence_done` | sequencer 是否跑完 | close 序列 |
| `steps[]` | 每步 `step`/`status`/`ts`/`task_id`/`detail` | close 序列，每步录一行 |
| `robustness.escalated` | 是否 robustness 升级停止 | stop_reason |
| `robustness.signals[]` `[争]` | 故障与恢复信号 | 见 §3.9 的 writer 缺失问题 |
| `artifacts.final_json_path` / `final_md_path` | 报告路径 | close step |
| `artifacts.session_breakdown_path` | 本文件名 | 常量 |
| `artifacts.artifact_package_path` | 打包路径 | close step |

---

## 5. `enablement`

**已落地：降为 timeline 事件 `enablement`。** 一个 session 一条 lane 一个事件，事件 id 固定
`enablement:0:enablement` —— phase 段是字面量而不是 `state.phase`，因为 pump 每 tick 跑一次、
与 phase 无关：触发在 PRELUDE 录，裁决在 FRAMEWORK_AGENT 录，按 phase 分段会把一条 lane 切成
两个各缺一半的事件。同理没有 recorder 对象：facts 来自五个模块的不同 tick，每个入口自己
幂等 open 事件。

`ext` 结构（详见 `schema.py` 的 `V6EnablementExt`）：

| 块 | 内容 |
|----|------|
| `mode` / `origin` / `engaged` | 准入模式、boot 还是 eval、是否启用过。三个都是录的，不再从生命周期不同的字段或运算 |
| `trigger` | 开 lane 的那次失败。只保留第一次：后来的同一 gap 属于面对它的那一轮 |
| `attempts.rows[]` | 每轮授权一行，按 specialist task id 归并。派发写"被指向哪个 gap"，rearm 写"落成什么" |
| `builds.rows[]` | 每次 targeted build 一行 |
| `revalidations.rows[]` | eval-origin 的每个复验窗口一行，按 generation 归并 |
| `human_review.rows[]` | 分类不出来、连轮都没派的失败，每个 digest 一行 |
| `result` | 终态：`succeeded` / `stalled` + 落地产物。被 kill 的 lane 没有 `result`，事件是 `interrupted` |

顶层 `collect_enablement` 与 `EnablementBreakdown` 已被事件取代，进第 9 项删除清单。

原始字段与产生点（录制接线的依据）：

| 字段 | 含义 | 产生点 |
|------|------|--------|
| `mode` | 准入模式 | `_accuracy_gate.py:199` |
| `origin` | boot 还是 eval 触发 | 事件存在性直接判定，不再从四个信号或运算 |
| `engaged` | 是否启用过 | 同上 |
| `attempts` / `dispatched` | 尝试与在途 | `enablement/lane.py:27` `_maybe_enqueue_enablement_specialist` |
| `succeeded` / `pending` / `stall_streak` | 结果与停滞 | `enablement/lane.py:225` `_maybe_rearm_enablement` |
| `kept_patches` / `kept_artifacts` | 保留产物 | `integrate_patch.py:3065` + re-arm |
| `localization_manifest` | 本地化清单 | `integrate_patch.py:3083` |
| `setup_commands` / `accepted_config` / `setting_script` | 落地配置 | re-arm |
| `build_attempts[]` / `last_build_failure` | 构建 | `targeted_build_executor.py:101` |
| `trigger_kind` / `observed_accuracy` / `accuracy_floor` | eval 触发 | `baseline.py:3035` 判定 + `writeback.py:856` 落盘 |
| `probe_config_path` / `eval_contract_fingerprint` | eval 契约 | 同上 |
| `human_review_count` | 人工复核次数 | state list |
| `attempt_runtimes[]` | 各次运行时环境 | runtime 记录 |

录制接线点：

| 位置 | 录什么 |
|------|--------|
| `writeback._handle_unpromotable_result` | boot 触发（首次 launch_log） |
| `writeback._persist_eval_failure` | eval 触发 + 复验失败 + 停滞终态 |
| `lane._maybe_enqueue_enablement_specialist` | 派发：attempt / failure_kind / launch_log / candidate_refs |
| `lane._maybe_record_enablement_human_review` | 待人工 |
| `lane._maybe_rearm_enablement` | 每轮裁决 + 成功/停滞终态 |
| `targeted_build_executor._record_result` | 构建 |
| `revalidation._maybe_enqueue_enablement_baseline_revalidation` | 复验窗口开启 |
| `writeback._promote_baseline` | 复验 promote/低于 floor + 成功终态 |

---

## 6. 顶层杂项

| 字段 | 处理 |
|------|------|
| `warnings` | 保留。导出与一致性守卫的告警 |
| `source_files` | 保留。改为各产物在生成时录路径，不再导出时扫盘 |
| `telemetry` | `[争]` 见 `SBD_V6_TELEMETRY.md`，现状基本不可用 |

---

## 7. 删除清单

| legacy 顶层键 | 去向 |
|---------------|------|
| `session` | → `metadata.session`（重复） |
| `session_meta` | → `metadata.session` |
| `workload` | → `metadata.task_config` |
| `model_info` | → `metadata.task_config.architecture` |
| `baseline` | → `timeline[baseline]` + `outcome.baseline` |
| `final` | → `outcome.final` |
| `phase_timeline` | → `timeline[phase].ext.actions[]`（见 §3.1，非独立 action 事件） |
| `phase_segments` | → `timeline[phase]` 事件本体 |
| `capability_summary` | → 各事件内计数 |
| `geak` | → `timeline[kernel].ext.geak`（补 §3.5 列的字段） |
| `kernel_lifecycle` | → `timeline[kernel].ext.forge`（见合并文档） |
| `collective` | → `ext.forge.lanes.collective_runs[]` |
| `param_search` / `explore_search` | → `timeline[framework_agent].ext.attempts[]`（`arm="config"`） |
| `critic_robustness` | → `timeline[critic]` + `close.robustness` |
| `critic_iterations` | → `timeline[critic]` |
| `specialist_runs` | → `ext.runs[]`（按 `role` 区分 discovery / authoring） |
| `optimizations` | `[争]` **阻塞**，见 §2.1 |
| `kernel_roofline` | → `timeline[roofline]`（需新增录制） |
| `kernel_optimization_summary` | → `timeline[kernel]` |
| `conc_sweep_summary` | → `timeline[conc_sweep]`（需补 §3.7） |
| `roofline` / `roofline_progress` | → `timeline[roofline]` + close 快照 |
| `decision_trace` | **删除**，不再记录（数据不准） |
| `token_usage` | **删除**，不再记录（数据不准） |
| `langfuse` | → `metadata.langfuse`，recorder 处写入 |
| `kernel_journey` | → `timeline[kernel]` |
| `versions` | → `metadata.versions`（补 commit / root_dir） |

---

## 8. 争议与阻塞项汇总

| # | 项 | 说明 |
|---|----|----|
| 1 | ~~`optimizations` 不能直接删~~ | **已解决并落地**：新增 `timeline[stack]` 事件承载跨阶段采纳账本，在 `_lift_to_current_best` 的 append 那一刻录一行，`throughput_before` 直接取 `graded.reference`。`collect_v6_outcome` 已不接收 `optimizations` 参数，耦合从结构上断开。8 个守卫计数只留 2 个有指代的。见 §2.1 |
| 2 | `record_kernel_rewrite` 生产未接线 | 只有测试调用；且设计上只留 adopted 形态，多 attempt 历史会丢 |
| 3 | kernel profiling 富字段无处可去 | `duration_us` / `call_count` / 利用率 / `recommended_*` 在 kernel 和 roofline 事件都不存在 |
| 4 | `robustness_signals` 无 writer | `signal.json` / `action.json` 全库没有写入方 |
| 5 | telemetry 基本不可用 | 单节点 GPU 采样架构上就是空的；lane 占用导出时必为 0 |
| 6 | explore 三个 ledger 死了两个 | `params_search` / `backends_search` 无 writer；`synergy_attempted` 无 append 方；`discovered_flags` AST 扫描从未实现 |
| 7 | ~~framework_agent 合并了 explore 桶~~ | **已定：保持不变**。证据层继续产生 `kind="explore"`，输出层继续合并，不新增分开看的能力。新 ledger 的 `source` 沿用现有五值 |
| 8 | ~~plateau 一族必须快照~~ | **已解决**：`record_plateau` 在判定那一刻快照 `inputs` / `thresholds` / `triggered`，每次判定追加一行 |
| 9 | ~~`enablement` 是否降为事件~~ | **已定并落地：降为事件**。一个 session 一条 lane 一个事件（phase 段固定为 `enablement`，因为 pump 本身与 phase 无关），触发/每轮/构建/复验/待人工各占一个 section；顶层 `collect_enablement` 待删 |
| 10 | ~~`elapsed_minutes` 的 resume 语义~~ | **已定并落地**：`elapsed_minutes` = 本腿实际时长，新增 `total_elapsed_minutes` 承载跨腿累计，两者都在 save 时快照 |
| 11 | ~~是否新开 action 级事件~~ | **已定并落地：不开**。原方案那一行是各 stage 事件已有 per-dispatch 行的真子集，再录一遍就是一个语义两处写。改为把 **phase 本身**降为事件（`{phase}:{macro_cycle}:phase`），dispatch 行挂在段上、只留身份+归属+裁决，细节靠 `task_id` join 回 stage 事件。覆盖面来自 `run_task_registered` / `_reap_dispatched_task` / `record_phase_transition` 三个既有收口，不来自 kind 白名单；`phase_timeline` / `phase_segments` 待删。见 §3.1 |
