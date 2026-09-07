# Session Breakdown 全量实时录制迁移方案

目标：

1. 所有字段改为在事实产生的那一刻就地录制（record-at-production）。
2. 删除全部导出期读取 `state.json` / `manifest.json` / `reports/*.json` / `*.jsonl` 重建事实的代码。

本文按「事实的产生点」而非「V6 schema 有没有这个字段」来组织。

---

## 0. 分类轴

对每个 legacy 字段，要问的不是「V6 里有没有」，而是三个问题：这个事实在代码里哪里产生、能不能在那里就地写入、以及它到底是事实还是聚合。由此得到五类：

| 类别 | 含义 | 工作量 |
|------|------|--------|
| **R1** | 产生点已在录制，导出侧仍保留 collector 兜底 | 补齐 fragment 字段 + 删兜底 |
| **R2** | 事实在主进程产生，但没有录制 | 在产生点加 record 调用 |
| **R3** | 事实在子进程或外部工具产生 | 主进程在结果回传时录制 |
| **R4** | 聚合 / 派生量，不是事实 | 不录制，改为从事件流推导 |
| **R5** | 没有产生点，事实压根没被计算过 | 要么实现，要么放弃 |

两个需要先对齐的判断：

**R3 不是妥协，是既定架构。** ContextVar 不跨进程，且 `test_breakdown_recorder_no_subprocess_writers` 明确禁止 agent 模块 import recorder。所以子进程的正确做法是写结论 JSON，由主进程在 writeback 时读一次并录制。关键区别在时刻：**在动作完成时读工具输出是录制，在导出时读同一个文件才是投影。**

**R4 不应该录制。** 录制一个 running counter 会产生第二个真相来源，并且在 resume 后必然漂移。正确做法是录制导致计数变化的那次决策，计数由消费方推导。这个结论在代码里已有先例——`recorder.py:156-163` 的 `DERIVED_SECTIONS` 已经声明 `capability_summary`、`attribution`、`phase_segments`、`source_files` 永不写 fragment。

**一个范围澄清：** `state.json` 不是要删的东西。优化循环自己要读 `failure_streak`、`stall_streak` 这类计数来做决策，它们必须留在 state 里。要删的是**导出侧读 state 重建事实**的代码。

---

## 1. 录制基建现状

### 1.1 写入 API

| API | 语义 | 位置 |
|-----|------|------|
| `get_recorder(producer=...)` | 唯一推荐的生产入口，需要 bound session | `recorder/recorder.py:583` |
| `record_singleton(section, payload)` | 该 producer 对该 section 的唯一 blob，覆盖写 | `recorder/recorder.py:303` |
| `record_item(section, payload, key=...)` | 追加一行；给 key 则幂等，适合 resume/retry | `recorder/recorder.py:352` |
| `record_upsert_item(section, payload, key=...)` | 同 key deep-merge，是 `EventSink` 的底层原语 | `recorder/recorder.py:429` |
| `open_event` / `finish_event` | 事件两段式写入：先写 `status=running` 的壳，结束时用 assembled ext 原地更新 | `recorder/event_timeline.py:155` / `:228` |
| `EventSink.record(section, payload, row_type=, natural_ids=)` | 事件内的行写入，自动注入 `event_id` | `recorder/event_sink.py:108` |

Fragment 落在 `<session>/runtime/breakdown/parts/*.json`，timeline 事件落在 `<session>/reports/sbd_v6/timeline/{seq}-{type}.json`。

### 1.2 已注册的 48 个 section

`recorder.py:72-152` 的 `SECTION_SHAPES` 已经声明了绝大部分 legacy 段，包括 `session`、`workload`、`baseline`、`final`、`kernel_lifecycle`、`explore_search`、`critic_robustness`、`critic_iterations`、`robustness_signals`、`telemetry`、`specialist_runs`、`kernel_roofline`、`kernel_optimization_summary`、`conc_sweep_summary`、`roofline`、`roofline_progress`、`versions`、`geak_invocations`、`forge_invocations`、`phase_timeline`，以及三套事件行 section（kernel / roofline / baseline）。

这意味着 R1 的范围比预想的大：这些段已经有 fragment 生产者，导出侧只是还留着 collector 兜底。

### 1.3 只有 5 种 durable 事件类型

`session/sbd_v6.py:31`：

```
_EVENT_TYPES = ("install", "model_gate", "roofline", "kernel", "baseline")
```

其中 `install` 和 `model_gate` 不走 fragment，直接 `write_timeline_event_at`（preflight / model_gate 在 session 建立前就要写）。`kernel`、`roofline`、`baseline` 走完整的 `open_event` → rows → `finish_event`。

`framework_agent`、`conc_sweep`、`warm_start`、`warm_replay`、`kb_write_back` 这 5 种仍然是导出期投影。

### 1.4 新增一种事件类型要动的地方

以 `baseline_event.py` 为模板：

1. `session/sbd_v6.py:31` 的 `_EVENT_TYPES` 加类型名
2. `recorder/recorder.py:72` 注册各 row section（item）
3. `recorder/assembler.py:471` 加 `XXX_EVENT_SECTIONS` 并并入 `EVENT_SECTIONS`
4. 新建 `recorder/xxx_event.py`：event id 构造、`XxxEventRecorder`、`begin()`、各 record 方法、`assemble_xxx_ext()`、`make_xxx_recorder()` factory
5. `recorder/event_finalize.py:57` 加 finalize spec（否则被 kill 的事件不会被关闭）
6. `recorder/__init__.py` 导出符号
7. 生产侧 wiring：executor 或 phase 里 `make_xxx_recorder` + `_resolve_sink`
8. `breakdown/schema.py` 加 TypedDict
9. 测试：`test_sbd_v6_xxx_timeline.py`

---

## 2. 硬约束与执行顺序

### 2.1 必须先切断 V6 collector 对 legacy collector 的依赖

四个 V6 collector 现在都拿 legacy 段当输入参数：

```617:631:src/hyperloom/inference_optimizer/breakdown/exporter.py
    metadata = _safe_collect(
        "metadata",
        lambda: collectors.collect_v6_metadata(
            exported_at_utc=exported_at,
            session=session_section,
            workload=workload,
            model_info=model_info,
            langfuse=langfuse,
            versions=versions,
            ...
```

- `collect_v6_metadata` ← `session` / `workload` / `model_info` / `langfuse` / `versions`
- `collect_v6_outcome` ← `session` / `baseline` / `final` / `optimizations`
- `collect_v6_timeline` ← `phase_timeline` / `conc_sweep_summary` / `critic_robustness`
- `collect_v6_close` ← `critic_robustness`

所以不能先删 legacy collector。顺序必须是：先让这些段的 fragment 字段完整，再把 V6 collector 的输入切到 fragment，最后删 collector。

### 2.2 分期

| 期 | 内容 | 产出 |
|----|------|------|
| **P0** | 逐段核对 fragment 与 collector 的字段差集，补齐 fragment 写入 | 每段一份「fragment 已覆盖 / 仍缺」清单 |
| **P1** | V6 collector 输入切换到 fragment，删掉 `_pick` 兜底 | 20 个 R1 段的 collector 变成死代码 |
| **P2** | R2 补录制（主进程内，最便宜） | phase_timeline 扩面、phase_segments、lease、context、roofline 快照、policy denial |
| **P3** | R3 补录制（writeback 时刻） | GPU 采样、benchmark invocation、specialist round、build manifest |
| **P4** | 新建 4 类事件 recorder | framework_agent、conc_sweep、KB、enablement |
| **P5** | 删除投影层 + reporters 改读 V6 | 约 1.6 万行 |

---

## 3. 逐域迁移方案

### 3.1 R1：已有 fragment，只需补字段 + 删兜底

导出侧走 `_pick(section, collector_fallback)` 或 `_merge_*` 的段：

`session`、`session_meta`、`workload`、`model_info`、`baseline`、`geak`、`geak_invocations`、`forge_invocations`、`explore_search`、`critic_robustness`、`telemetry`、`specialist_runs`、`kernel_roofline`、`kernel_optimization_summary`、`conc_sweep_summary`、`roofline`、`roofline_progress`、`enablement`、`kernel_journey`、`versions`、`phase_timeline`。

这批的处理方式统一：

1. 用真实 session 跑一次，dump fragment 与 collector 两侧结果做字段级 diff。
2. fragment 缺的字段，回到产生点补写。
3. 删掉 `_pick` 的第二个参数和对应 collector 函数。

**注意两个已知的字段级缺口：**

- `versions`：`_tool_versions()`（`v6.py:58`）把每个工具压成一个字符串，丢掉 `root_dir` 和 `commit`。产生点 `instrument.record_tool_version`（`instrument.py:2919`）本来就有完整的 `{tool, root_dir, commit, version}`，是投影时压扁的。修 `_tool_versions` 即可，不用改录制。
- `model_info`：`_architecture()`（`v6.py:75`）只投影 5 个字段，丢掉 14 个结构字段。同理，产生点有完整数据，是投影裁剪。

这两个说明一个普遍问题：**部分「字段丢失」其实是 V6 投影函数裁剪导致的，不是录制缺失。** P0 的 diff 会把这类和真正的录制缺失区分开。

**不在 `_pick` 里、纯 collector 的段：** `final`、`kernel_lifecycle`、`collective`、`decision_trace`、`token_usage`、`langfuse`、`phase_segments`、`capability_summary`、`source_files`。这些要按下面 R2/R3/R4 分别处理。

### 3.2 R2：主进程内产生，加 record 调用

| 事实 | 产生点 | 录制方式 |
|------|--------|----------|
| phase 转换（含 `exit_reason`、`evidence`） | `phases/machine_state.py:3129` `record_phase_transition`，调用方 `phases/machine.py:353` 已传入 reason 和 evidence | 已有 `phase_transitions` section，在 transition 时刻写行；`phase_segments` 由它推导（R4） |
| phase_timeline 扩面 | `state/shared_state.py:416` 的 `_AUDIT_ACTIONS` 只含 4 个动作 | 把 `validate_stack`、`integrate`、`report`、`eval`、`conc_sweep` 加入 audit set，`record_action_attempt` 自动写 `phase_timeline` |
| lease 获取/释放/过期 | `bus/resource_lock.py:235` `acquire_many`，expire 已发 `lease_expired` 事件 | 在 acquire/release/expire 三处写 item 行；`lane_timeline` 由它推导 |
| 上下文压缩 | `loop/maintenance.py:216` `_maybe_run_orchestration_checkpoint`，payload 已含 `context_tokens`；degenerate 分支在 `:241` | 每次 compaction 写一行观测 |
| prompt 模式计数 | `loop/conversation.py:97` `_count_prompt_mode` | 录制每次 turn 的模式，`seed/delta` 计数由它推导 |
| roofline ceiling | `state/shared_state.py:3114` `record_baseline_roofline_ceiling`，纯 Python 在主进程算 | 写 singleton |
| roofline 快照 | `state/shared_state.py:3486` `record_trace_analyze` append | 每次 append 写一行 |
| 能力未尝试的原因 | `policy/gate.py:2064` `record_policy_denial`；`phases/kernel.py:3980` `_kernel_opt_dispatch_skip_reason`；`_grid_runner.py:390` `unsupported_capability_reason` | 三个决策点都在主进程，写 denial/skip 行；`capability_summary` 由它推导 |
| LLM 调用与 token | `trace/llm_trace.py:388` `append_llm_call` 已经在写 `llm_calls.jsonl` | 这已经是 record-at-production，只是写的是自有 JSONL 而非 fragment。改为同时写 fragment，或让导出侧把它当 durable 源（见 §5 决策点） |
| 容器镜像 | `session/manifest.json` 由 `session/manifest.py:179` `_detect_image` 在 bootstrap 时写 | 已经是产生时写入。`image` / `image_id` 应进 `session` fragment，而不是导出时再探测一次（`sessions.py:449` 的 `_detect_image_for_session` 是重复探测，删） |

### 3.3 R3：子进程产生，writeback 时刻录制

| 事实 | 子进程产出 | 主进程录制点 |
|------|-----------|-------------|
| GPU 采样聚合 | benchmark 子进程写 `benchmark_report.json` 的 `gpu_monitor` 列表；多机路径 `benchmark_result.py:530` `harvest_mn_gpu_metrics` | `benchmark_result.py:745` `extract_benchmark_measurement` 已在解析 report，在此处录制采样或聚合 |
| benchmark invocation | 子进程启动 server，写 `server.log` 与 config yaml | `writeback.py:4776` 已在记 `server_log_path`；`framework_args` / `framework_args_source` 目前是导出期从 log 和 yaml 里 parse（`sessions.py:320` `_extract_framework_args`），应改为在 launch 时由启动方直接上报 |
| ttft / e2el 及其来源 | 子进程 report 的 `latency.ttft/e2el` | `benchmark_result.py:771` 解析处录制，同时录 `ttft_e2el_source` |
| specialist round 汇总 | specialist 子进程写 `transcript.jsonl`，completion 回传 `transcript_path` | `phases/explore.py:1849` `_build_specialist_round_entry` 构造 round 行；`proposals_kept/rejected/domain_breakdown` 目前没填（见 R5） |
| targeted build | build 子进程 | `targeted_build_executor.py:101` `_record_result` 写 manifest，在此录制 per-attempt 行 |
| GEAK 运行结果 | GEAK runner 子进程写 `geak/` 树 | 目前靠 `geak.py:617` `_geak_reconstruct_from_disk`（249 行）在导出期重建。应改为 runner 结束时由主进程读一次 `result.json` / `handoff.json` / opbench / log tail 并录制，然后删掉整个重建函数 |
| kernel attempt 明细 | kernel-agent 子进程写 `optimization_attempts.jsonl` | 目前靠 `kernels.py:301` `collect_kernel_invocations` 在导出期 replay。`instrument.record_kernel_invocations` 已存在，需确认覆盖完整后删 replay |
| conc sweep | 子进程写 `reports/conc_sweep_summary.json` | `kernel/conc_sweep.py:1120` 已 `record_singleton_section`。`kernels.py:1041` 的 `_recover_conc_sweep_summary_from_runs` 是兜底重建，删 |

### 3.4 R4：不录制，改为推导

这些是聚合量，应该由消费方从事件流算，不进 fragment。`DERIVED_SECTIONS` 已有 4 个成员，建议扩充：

| 派生量 | 推导来源 |
|--------|----------|
| `phase_segments` | `phase_transitions` 行两两配对 |
| `capability_summary` | 各能力的 attempt 行 + policy denial 行 + dispatch skip 行 |
| `kernel_lifecycle` 五段漏斗 | kernel 事件的 discovery / dispatch / backend / e2e 行 |
| `kernel_journey.kernels[]` | 同上，按 `kernel_id` 聚合 |
| `optimizations.summary_by_*` | `adoptions` + `operations` 行分组 |
| `token_usage` 全部 rollup | LLM 调用行按 phase / component 分组 |
| `roofline_progress` 的比率与曲线 | ceiling singleton + snapshot 行 + baseline/current_best |
| `telemetry.gpu_monitor_aggregate` | GPU 采样行 |
| `lane_timeline` 计数 | lease 行 |
| `compactions_per_tick` / `delta_ratio` | compaction 行 + tick 数 |
| `conc_sweep.summary` 的 median / mean / pair 计数 | comparison 行 |
| specialist `proposals_*` / `domain_breakdown` | proposal 行（前提是 proposal 逐条录制） |
| `source_files` | 各 artifact 行的路径字段 |
| 各种 streak（`failure_streak`、`stall_streak`、`no_promote_streak`、`roofline_failure_streak`） | 对应的失败行计数。**state 里仍保留供循环决策，只是不再作为导出源** |
| enablement 的 `engaged` / `origin` / `trigger_kind` | 目前已经是 `sessions.py:1571` 的导出派生。改为从 enablement 事件行推导 |

### 3.5 R5：没有产生点

这三项不是迁移问题，需要单独决策：

| 项 | 现状 |
|----|------|
| `discovered_flags` | AST 扫描从未实现。`explore.py:2021` 固定返回 `"discovered_flags_update": None`，全仓库无赋值。`state.discovered_flags` 恒空 |
| `dispatch_history.jsonl` | 只有 reader（`decision.py:223`），没有 writer。读到的恒为空 |
| specialist `proposals_kept` / `proposals_rejected` / `proposals_skipped` / `domain_breakdown` / `confidence_avg` | `_build_specialist_round_entry`（`explore.py:1849`）不填这些字段，导出侧靠 `timeline.py:615` 从 tags 猜 |

---

## 4. 新建的 4 类事件 recorder

### 4.1 `framework_agent`（最大一块）

现状：`collectors/v6.py:397-1932` 约 1500 行投影，读 `state.framework_agent_batches`、`framework_agent_phase_progress`、`specialist_rounds`、`framework_config_exploration_results`、`explore_search`、`plateau_overrides`，外加 `reports/trace/proposal_task_map.jsonl` 和 `critic-workdir/*/*.json`。

迁移：建 `FrameworkAgentEventRecorder`，按 macro cycle 开事件，录制 config arm 的 round 与 variant 行、source arm 的 discovery / authoring / attempt 行、critic review 行、plateau 与 exit 判定。产生点分散在 `phases/explore.py`、`loop/writeback.py`、`loop/proposals.py`、`roles/critic_agent.py`，都在主进程。

### 4.2 `conc_sweep`

现状：`collectors/v6_stages.py:194-319` 投影，输入是 `conc_sweep_summary` 段和 `phase_timeline`。

迁移：`kernel/conc_sweep.py` 已经在写 singleton，改为开一个事件、逐 CONC 点写行、结束时 finish。`benchmark_mode` 和 agentx 指标在这里顺便补上。

### 4.3 KB 三事件（`warm_start` / `warm_replay` / `kb_write_back`）

现状：`breakdown/kb_timeline.py`（595 行）投影，读 recipe audit jsonl、KB queue ndjson 和一堆 `warm_start_*` / `warm_replay_*` state 键。

迁移：产生点在 `orchestrator/knowledge/`，主进程内。KB 查询、replay 尝试、write-back 各写一个事件。

### 4.4 `enablement`

现状：完全没有 V6 事件类型，全靠 `sessions.py:1529` `collect_enablement` 读 `state.enablement` 子树。

迁移：这是唯一有完整动作生命周期却零录制的模块。产生点已定位：
- 入队与 attempts：`enablement/lane.py:27` `_maybe_enqueue_enablement_specialist`
- re-arm、kept_patches、localization、accepted_config：`enablement/lane.py:225` `_maybe_rearm_enablement`
- eval 触发判定：`actions/executors/baseline.py:3035` `_mark_eval_rooted_baseline_failure`
- eval 落盘：`loop/writeback.py:856` `_persist_eval_failure`
- build：`targeted_build_executor.py:101` `_record_result`

除 build 和 eval 判定在子进程外，其余都在主进程。

---

## 5. 已定的三项决策

### 5.1 载体统一到 recorder fragment

已有的自有格式 trace 虽然本身就是产生时写入，但导出侧只认 fragment 一种源。以下产生点改为**同时写 recorder fragment**：

| 现有载体 | 产生点 | 处理 |
|----------|--------|------|
| `reports/trace/llm_calls.jsonl` | `trace/llm_trace.py:388` `append_llm_call` | 每次调用同时写 fragment 行。顺便在 call 行上直接写 `phase`（消除审计表 A158 的时间窗回填） |
| `reports/trace/ext/<component>-<pid>.jsonl` | 子进程 shard | 子进程不能 import recorder，保持写 shard，由主进程回收时录制 |
| `reports/trace/proposal_task_map.jsonl` | `loop/proposals.py:591` `_record_proposal_task_map` | 同时写 fragment |
| `session/manifest.json` | `session/manifest.py:179` `_detect_image` | 镜像身份写进 `session` fragment；删掉导出侧的二次探测 `sessions.py:449` |
| `storage/coordinator.db` `leases` 表 | `bus/resource_lock.py:235` | acquire / release / expire 三处写 fragment 行 |
| `storage/coordinator.db` `events` 表 | `loop/maintenance.py:216` 等 | compaction 等观测写 fragment 行 |

DB 和 JSONL 本身继续存在（运行时要用），只是不再是导出侧的读取对象。

### 5.2 聚合量全部走产生式快照，不做读时推导

已产出 `SBD_V6_AGGREGATE_AUDIT.md`，197 条聚合量逐条列了计算位置、依赖事实和 resume 风险。

判据不是「推导还是快照」，而是「在哪一刻落快照」，由作用域决定：

| 作用域 | 快照时刻 | 机制 | 条数 |
|--------|----------|------|------|
| **E** 事件级 | `finish_event` | **已存在**：`assemble_{baseline,roofline,kernel}_ext` | 约 96 |
| **P** 决策级 | 决策发生那一刻 | 决策点直接写 fragment | 约 42 |
| **C** session 级 | close 序列 | close 时读齐 fragment、算一次、冻结 | 约 78 |
| **X** 不记录 | — | 猜测或恒空字段 | 17 |
| **DUP** | — | V6 已覆盖，见 §5.4 | 约 18 |

**E 占最大头且不需要新机制**——三类事件的 `finish_event` 早就是这个模式，缺的只是把还没录的行补进去。

关于 **C**：close 时算一次仍是在做算术，只是从「每次导出算一遍」变成「一辈子算一次并冻结」，收益是 re-export 幂等、值不随导出时刻变化。若要连这次算术也去掉，唯一替代是 `record_upsert_singleton`（`recorder.py:322`）增量累加，但它只有进程内锁、不跨进程安全，resume 后累加起点也难保证。因此 C 类走 close 时算一次。

### 5.4 去重

已产出 `SBD_V6_DEDUP.md`。要点：

- **唯一零损失整段可删的是 `geak`**（`collectors/geak.py` 1053 行）。它与 `timeline[kernel].ext.geak` 有约 30 个字段 DUP-EXACT，剩余 UNIQUE 字段全是磁盘重建产物（已标 X）。
- 其余 12 个 legacy 段都含 UNIQUE 字段，不能整段删，只能删重复子集。
- **约 12 组是假重复**，必须保留区分。最重要的是 GEAK 的三个增益：legacy `geak.gain_pct` 一个字段，V6 拆成了自报值（`claim.self_reported_gain_pct`，旁边有 `verified: false`）、复测值（`rebench.attempts[].delta_pct`）、基线（`handoff.raw_baseline_tput`）。V6 的拆分是改进，应采用 V6 命名废弃 legacy 字段，而不是当重复删掉其中一份。
- 另外三处**假重复**同样不能合：baseline 的 action 级采纳值 vs round 级原始测量（后者含被丢弃的 warmup 轮）；`phase_timeline` 的 action 粒度 vs `timeline[]` 的 stage 粒度；`kernel_roofline` 的 per-kernel bound 数据（V6 roofline 事件里**没有** bound_type / efficiency，只有一个路径和不含 bound 的 hot_kernels 摘要）。
- V6 内部有 3 处自我重复：`metadata.exported_at_utc`、`metadata.versions.schema_version`、`metadata.versions.hyperloom`。

### 5.3 R5 三项补实现

| 项 | 要做的 |
|----|--------|
| `discovered_flags` | 实现框架源码 AST 扫描。当前 `explore.py:2021` 固定返回 `None`，`explore_state.py:759` 的 `record_discovered_flags` 持久化 API 已就绪，缺的是扫描器本身 |
| `dispatch_history` | 补 writer。当前只有 reader（`decision.py:223`），路径 `agents/orchestration/dynamic_actions/<id>/dispatch_history.jsonl` |
| specialist proposal 明细 | 在 `phases/explore.py:1849` `_build_specialist_round_entry` 填 `proposals_kept` / `proposals_rejected` / `proposals_skipped` / `domain_breakdown` / `confidence_avg`。这会消除审计表 A24 的**按域数平均分摊**猜测——那是整表最不可信的一处 |

`params_search` / `backends_search` 两个 ledger 同样在 orchestrator 侧无 writer、恒空，随 `discovered_flags` 一起定。

---

## 6. 删除清单

以下为投影层，在对应域完成录制后删除。行号基于 commit `da54496de`。

### 6.1 整文件删除

| 文件 | 行数 | 说明 |
|------|------|------|
| `collectors/attribution.py` | 816 | 已未接入 `exporter.build()`，纯死代码，可立即删 |
| `collectors/telemetry.py` | 648 | critic / robustness / telemetry / specialist 全靠扫盘 |
| `collectors/explore.py` | 187 | 纯 state ledger 读取 |
| `collectors/roofline.py` | 361 | state + `reports/kernel_roofline.json` |
| `collectors/decision.py` | 865 | trace JSONL + journal 合并，并且有写副作用（`:837` 写 `decision_trace.jsonl`） |
| `breakdown/kb_timeline.py` | 595 | KB 三事件投影 |
| `collectors/v6_stages.py` | 319 | conc_sweep 投影 |

### 6.2 段删除

| 位置 | 内容 |
|------|------|
| `collectors/sessions.py:932-1668` | baseline / final / enablement 的 state 与磁盘投影 |
| `collectors/sessions.py:1111-1185` | `_reconstruct_baseline_attempts` |
| `collectors/sessions.py:320-412` | `_extract_framework_args` |
| `collectors/sessions.py:449-496` | `_detect_image_for_session`（重复探测） |
| `collectors/geak.py:446-944` | 全部 GEAK 磁盘重建，含 `_geak_reconstruct_from_disk`（617-865） |
| `collectors/kernels.py:301-354` | `collect_kernel_invocations` attempt replay |
| `collectors/kernels.py:1041-1099` | `_recover_conc_sweep_summary_from_runs` |
| `collectors/kernels.py:1525-1570` | `collect_source_files` |
| `collectors/timeline.py:143-270` | journal + state 拼 `phase_timeline` |
| `collectors/timeline.py:387-690` | `capability_summary` state 投影 |
| `collectors/v6.py:397-1932` | `_framework_*` 投影树 |
| `collectors/v6.py:2006-2014` | conc_sweep 与 KB 投影调用 |
| `collectors/_common.py:441-488` | journal 与 profile 扫描 helper |
| `exporter.py:283-637` | 全部 collector 兜底分支 |
| `exporter.py:155-216` | `_attach_kernel_roofline` 内存 join |

### 6.3 重写而非删除

| 位置 | 改法 |
|------|------|
| `collectors/v6.py:104` `collect_v6_metadata` | 输入从 legacy 段改为 fragment；修 `_tool_versions` 和 `_architecture` 的裁剪 |
| `collectors/v6.py:2102` `collect_v6_outcome` | 输入改 fragment；`_stage_reached`（2034-2099）读 20+ 个 state 键，改为读事件流 |
| `collectors/v6_close.py:157` | close steps 改为在 close 序列执行时录制 |
| `collectors/optimizations.py:786` | 这个是 keeper，已经只消费 fragment |
| `reporters/` 整包 3807 行 | 改读 V6，见 §7 |

### 6.4 保留

| 位置 | 原因 |
|------|------|
| `recorder/assembler.py` `assemble_parts` | fragment 组装 |
| `exporter.py:712` `_load_assembled` | 同上 |
| `session/sbd_v6.py:212` `read_timeline_events` | durable timeline 读取 |
| `recorder/event_finalize.py` `finalize_events` | 关闭被 kill 的事件，必须在读 timeline 前跑 |
| `exporter.py:869-956` `_patch_breakdown` / `patch_breakdown_langfuse` | 事后补丁，读的是自己产出的 JSON |

---

## 7. reporters

现状：3807 行，18 个 renderer 加 `cross_section.py`，全部读 legacy 段，没有一个模块读 `metadata` / `outcome` / `timeline` / `close`。

`cross_section.build_global_facts` 的事实包对以下派生量有硬依赖，重写时必须先有对应推导：

- `_headline`：baseline / final 吞吐 + validated gain
- `_capabilities_split`：每能力 status（R4）
- `_kernel_funnel`：五段漏斗计数（R4）
- `_gain_attribution_lines`：`optimizations.summary_by_source` 和 `validation.method`

另外发现一个现存 bug：`geak_invocations` / `forge_invocations` 在 `exporter.py:341-342` 被计算出来，但没有放进最终 breakdown dict，两个 renderer 因此永远拿不到数据——正好是 `compose.py:55-59` 注释警告的情形。重写时需决定这部分明细要不要真的接上。

---

## 8. 优先级建议

1. 立即可做、零风险：删 `collectors/attribution.py`（816 行死代码，未接入 `exporter.build()`）。
2. 最高性价比：P0 的字段级 diff。它会把「投影裁剪导致的假缺口」和「真正的录制缺失」分开，很可能大幅缩小后续工作量。
3. 最大单块：`framework_agent` recorder，替换约 1500 行投影。其中 plateau / streak 一族按 §5.2 走快照。
4. 唯一零覆盖模块：`enablement` 事件。
5. 与迁移并行：§5.3 的三项补实现。specialist proposal 明细优先，因为它直接消除当前的均分猜测。

## 9. 相关文档

- `SBD_V6_AGGREGATE_AUDIT.md` — 197 条聚合量的逐条审计表（E / P / C / X / DUP 作用域判定）
- `SBD_V6_DEDUP.md` — V6 timeline 已覆盖内容、真重复与假重复的区分
