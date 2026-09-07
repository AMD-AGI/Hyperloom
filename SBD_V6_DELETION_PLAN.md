# V6 删除计划

配套 `SBD_V6_FINAL_FIELDS.md`。那份是**字段去向表**（26 个 legacy 顶层键去哪），这份是**代码删除顺序**。

基线 commit `65a1e285a`（`sbd-slim`）。此时 envelope 有 31 个顶层键，V6 目标 11 个，20 个待删。

标记：`[定]` = 已决策；`[阻]` = 决策未定，阻塞下游。

---

## 0. 顺序原则：先补后删

**不能先删代码再补字段。** 两条理由：

1. **collector 是 `[缺]` 字段的唯一参考实现。** `FINAL_FIELDS` §7 的 26 个键里，24 个的去向都附带"需新增录制"或"需补字段"。`collectors/roofline.py` 是 §3.4 三大块（per-kernel 表 19 字段、snapshot 约 25 字段、progress 轨迹）的唯一现存实现；`collectors/geak.py` 是 §3.5 要补回的 `output_parity` / `metric_basis` / `accepted_kernels[]` 子字段的唯一实现。先删则补录制时要从产生点重新逆推。

2. **没有 legacy 输出就无法双跑校验。** 判断新录制是否正确的唯一可靠手段是同一 session 同时产出新旧两套值做 diff。摘掉 envelope 键就失去基准。

唯一例外是 **P0**：与字段规格完全无交集的死代码，删掉不丢任何 `FINAL_FIELDS` 要保留的字段。

**完整顺序**

```
P0  清死代码（不依赖字段表，不依赖任何决策）
 ↓
P1  定 #1 adoption ledger、#8 plateau 快照
 ↓
P2  逐段补 author-time 录制（对照 FINAL_FIELDS）
 ↓
P3  逐段：摘 envelope 键 → 删 collector → 删 renderer
```

P2 与 P3 **按 section 交替**，不是两个大批次。每段的节奏是：补录制 → 双跑 diff 通过 → 摘键 → 删代码。

---

## 1. 决策记录

### 1.1 已定

| # | 项 | 决策 |
|---|----|------|
| 7 | explore 与 framework_agent 桶是否分开 | `[定]` **保持不变**。证据层继续产生 `kind="explore"`（`writeback.py:5705`、`:5779`、`phases/explore.py:1461`），V6 输出层继续合并（`v6.py:2266`）。不新增"分开看"的能力，也不删除证据层的区分 |
| 9 | `enablement` 是否降为 timeline 事件 | `[定]` **独立记录**。保留 `enablement` 顶层段，采集改为 recorder author-time 录制。不降级为事件 |
| 10 | `elapsed_minutes` 的 resume 语义 | `[定]` **表示实际运行时长**；跨 resume 累计**另加字段** |

#### #9 的执行含义

`enablement` 保留为顶层键，所以它**不在** 20 个待删键里，V6 目标顶层结构不变。要做的是把 `collect_enablement` 从"导出时扒 `state.json` 20+ 字段"改成"读 recorder fragment"。

产生点已在 `FINAL_FIELDS` §5 列全：`_accuracy_gate.py:199`（mode）、`enablement/lane.py:27`（attempts/dispatched）、`:225`（succeeded/pending/stall_streak）、`integrate_patch.py:3065`（kept_patches）、`:3083`（localization_manifest）、`targeted_build_executor.py:101`（build_attempts）、`baseline.py:3035` + `writeback.py:856`（eval 触发一族）。

模块本身在 `orchestrator/enablement/`（`lane.py` / `build.py` / `params.py` / `revalidation.py`），生命周期为：入队 → specialist → `integrate_patch` → targeted build → re-arm。

#### #10 的执行含义

现状两个缺陷：

- `start_ts` 在 resume 时**条件重置**（`cli/bootstrap.py:608-611`）：上次因 `time_exhausted` 类 stop_reason 停的重置为 `resumed_ts`，clean stop 则保留。导致 `elapsed_minutes = now - start_ts`（`shared_state.py:3731`）在不同 session 语义不同。
- **每次导出重算**（`exporter.py:123`），session 未结束时该值持续变长，同一 session 导两次得两个值。

改法（已落地）：

| 字段 | 语义 | 时机 |
|------|------|------|
| `metadata.session.elapsed_minutes` | 本次运行腿的实际墙钟，起点 `resumed_ts or start_ts` | 每次 save 快照，最后一次即 close 值 |
| `metadata.session.total_elapsed_minutes` `[新]` | 跨 resume 累计实际运行时长 | `SharedState.prior_legs_elapsed_s` + 本腿 |

不引入 `run_legs[]`：累计走 `prior_legs_elapsed_s` 单调累加，在 resume 边界由 `_bank_previous_leg_elapsed` 结算。代价是崩溃腿没有 `stop_ts` 因而不入账——与 `_bank_previous_leg_phase_segment` 对相位时钟的处理一致，宁少记不虚记。

### 1.2 仍阻塞

| # | 项 | 阻塞什么 |
|---|----|----------|
| 1 | adoption ledger 形状 `[阻]` | `optimizations` 段（`collectors/optimizations.py` 1211 行）、`outcome.validation`、以及 `operations` / `measurements` / `adoptions` / `artifacts` 四条枢纽流能否退役。见 `FINAL_FIELDS` §2.1 |
| 8 | plateau 一族是否快照 `[阻]` | `timeline[framework_agent]` 那约 1500 行投影（`collectors/v6.py`）能否被替换 |

#1 是整个计划最大的一把锁。#7 已定"保持不变"，意味着新 ledger 的 `source` 枚举**可以沿用现有五值**（`optimizations.py:28-34` 的 `warm_replay` / `explore` / `framework_agent` / `kernel_agent` / `unattributed`），输出时按 `v6.py:2266` 的方式合并 explore 与 framework_agent。这一项不再阻塞 #1。

---

## 2. P0：可立即删除（零字段损失，零决策依赖）

### 2.1 孤立的 v4 entity stream

SBD v4 时代的 canonical author-time stream。recorder 一直在写 fragment，**`exporter.py` 与 `assembler.py` 全文无任何读回**（已核 `assembled.get` / `assembled[...]` / `_pick` 全部形式，含多行调用）。

| section | 写入点 | 读取点 |
|---------|--------|--------|
| `run_snapshot` | `instrument.py:1013`；`record_run_snapshot`（`:4356`）**零调用** | 无 |
| `subjects` | `instrument.py:4411`；`record_subject` **零外部调用** | 无 |
| `phase_transitions` | `instrument.py:790`、`:4392` | 无 |
| `trace_events` | `instrument.py:4668` | 无 |
| `sweep` | `instrument.py:409` | 无 |
| `optimization_stack` | `instrument.py:1118` | 无 |

要一并删除：

- `recorder/recorder.py:72-160` `SECTION_SHAPES` 中这 6 项
- `recorder/assembler.py:36`（`phase_transitions`）、`:37`（`subjects`）、`:42`（`trace_events`）的 key 声明
- `instrument.py` 对应写函数与 `__all__` 导出（`:4695` 一带）
- `recorder/__init__.py:105`、`:193` 的 re-export

**外部调用点（仅 3 处）**

| 位置 | 调用 | 处理 |
|------|------|------|
| `phases/machine_state.py:3235` | `instrument.record_phase_transition(...)` | 删这一行 |
| `phases/machine_state.py:3247` | `instrument.record_trace_event(...)` | 删这一行 |
| `knowledge/kb_writeback.py:288` | `instrument.record_trace_event(...)` | 删这一行 |

**⚠ 同名函数陷阱。** 存在两个 `record_phase_transition`：

- `phases/machine_state.py:3170` —— 状态机的，写 `state.phase_history`。**保留**，这是 phase 事实的真来源，`phases/machine.py:63`、`:103`、`:353` 经 `SharedState.record_phase_transition`（`shared_state.py:2038`）调它。
- `recorder/instrument.py:4370` —— v4 stream 的。**删除**。

只删 `machine_state.py:3235` 那一行 instrument 调用，不要动 `machine_state.py:3170` 整个函数，否则 `phase_history` 一起没了。

**不可删**：同批 `SECTION_SHAPES` 里的 `operations` / `measurements` / `adoptions` / `artifacts` 是枢纽流，见 §5。

### 2.2 死生产代码

| 目标 | 行数 | 依据 |
|------|------|------|
| `collectors/attribution.py` 整个模块 | 816 | `collect_attribution`（`:418`）**只有测试调用**，`exporter.build` 从不调；`attribution` 不在 31 个 envelope 键里。`outcome.validation.attribution` 在 `v6.py:2260` 独立构建，数据来自 `optimizations.summary_by_source`，与本模块无关 |
| `recorder/kernel_event.py:694` `record_kernel_rewrite` | — | 全库只有定义，**零生产调用**（`FINAL_FIELDS` §8#2 已记录，已复核） |

删 `attribution.py` 需同步处理 `collectors/__init__.py:126-130`、`:191` 的 re-export，以及 5 个测试文件的引用（`test_phase_state_plateau.py`、`test_forge_lane.py`、`test_geak_acceptance_identity.py`、`test_sbd_optimizations.py`、`test_geak_gain_alignment.py`）。

`record_kernel_rewrite` 的删除与否取决于是否要按 `SBD_V6_KERNEL_LIFECYCLE_MERGE.md` 把它接线。若计划接线则**不要删**，标为待接线即可。

### 2.3 去向即"删除"的两个键

`FINAL_FIELDS` §7 里唯一两个不需要任何补录制的：

| 键 | 理由 | 涉及代码 |
|----|------|----------|
| `decision_trace` | 数据不准，不再记录 | `collectors/decision.py` 的 trace 部分 |
| `token_usage` | 数据不准，不再记录 | `collectors/decision.py` 的 token 部分 |

`collectors/decision.py` 共 865 行，**保留 `collect_langfuse`**（仍服务 `metadata.langfuse` 的 fallback），其余可删。同时删 `exporter.py` envelope 里这两个键与对应 renderer。

### 2.4 无 writer 的空数据路径

`FINAL_FIELDS` §8 里被记为"缺陷"的几项，实际是**可无损删除**的证据——对应 collector 投影的是恒空数据。已逐项复核：

| 项 | 复核结果 | 可删代码 |
|----|----------|----------|
| §8#4 `robustness_signals` | `signal.json` / `action.json` **只有 reader 无 writer**：读点在 `collectors/telemetry.py:80-81` 与 `instrument.py:4142-4143`，全库无写点 | `close.robustness.signals` 的这条读盘路径。**注意**：字段本身在 `FINAL_FIELDS` §4 是保留的，所以这里只删"读不存在的文件"这段，需先按 §3.9 决定是补 writer 还是改读 `agents/robustness/findings/*.jsonl` |
| §8#6 `params_search` / `backends_search` | 两个 ledger **只有读点**：`collectors/explore.py:176`、`:179` 从 `state` 取，全库无写点 | `collectors/explore.py`（187 行）这两段。整模块的去向是 `timeline[framework_agent].ext.config_arm`，属 P3 |

`synergy_attempted` 与 `discovered_flags` 情况不同，**不列入 P0**：`state/_shared_state/explore_state.py:525-536` 有 merge 逻辑，`:759` 有 `record_discovered_flags` setter，需单独核实是否真的无产生方，暂按"存疑"处理。

### 2.5 P0 汇总

| 项 | 量级 | 风险 |
|----|------|------|
| 6 个孤立 v4 stream + 3 处外部调用 | recorder/instrument 若干段 | 低，但需避开同名函数陷阱 |
| `attribution.py` | 816 行 + 5 个测试文件 | 低 |
| `decision.py` 的 trace/token 部分 | 约 600 行（保留 `collect_langfuse`） | 低 |
| `explore.py` 两个死 ledger 段 | 少量 | 低 |

P0 全部完成后 envelope 从 31 键降到 29 键（摘掉 `decision_trace`、`token_usage`）。

---

## 3. P2 / P3 逐段解锁表

每一行的前置是"该段的 author-time 录制补完并双跑 diff 通过"。

| envelope 键 | 前置补录制（FINAL_FIELDS 章节） | 解锁后可删 |
|-------------|--------------------------------|-----------|
| `baseline` | §3.3 补 `ttft_e2el_source`、`invocation.framework_args`、`failure_streak` | `collect_baseline`；`outcome.baseline` 改读 timeline |
| `final` | §2 补 `final.ttft_mean_ms` / `e2el_mean_ms` | `collect_final`（注意：当前**无 `_pick`**，`exporter.py:300` 恒走 collector） |
| `phase_timeline` / `phase_segments` / `capability_summary` | §3.1 新增 action 级事件与 phase 转换行 | `collectors/timeline.py` 大部分（864 行） |
| `geak` | §3.5 补回 `ttft_mean_ms` / `tpot_mean_ms` / `output_parity` / `metric_basis` + `accepted_kernels[]` 7 个子字段 | `collectors/geak.py`（1053 行） |
| `kernel_lifecycle` / `collective` / `kernel_journey` / `kernel_optimization_summary` | §3.5 `ext.forge` 缺口，见 `KERNEL_LIFECYCLE_MERGE.md` | `collectors/kernels.py` 大部分（1570 行，**保留 `collect_source_files`**）；`assembler.py:588` `_compose_kernel_journey`；`exporter.py:155` `_attach_kernel_roofline` |
| `kernel_roofline` / `roofline` / `roofline_progress` | §3.4 三大块：per-kernel 表 19 字段、snapshot 约 25 字段、progress 轨迹 | `collectors/roofline.py`（361 行） |
| `conc_sweep_summary` | §3.7 补 `roofline_ceiling`、agentx 轴指标、summary 统计等 | `collect_conc_sweep_summary`；`collectors/v6_stages.py` 的投影；`exporter.py:309` `_merge_phase_timeline` |
| `critic_robustness` / `critic_iterations` | §3.9 新增 `critic` 事件（per-proposal 粒度）；同时解决 §8#4 | `collectors/telemetry.py` 的 critic 部分；`assembler.py:446` `_compose_critic_robustness` 改为只喂 `close` |
| `param_search` / `specialist_runs` | §3.6 `ext.config_arm` / `source_arm` 补全 + #8 plateau 快照 | `collectors/explore.py` 剩余；`collect_specialist_runs` |
| `optimizations` | `[阻]` #1 adoption ledger | `collectors/optimizations.py`（1211 行）+ 四条枢纽流 |

`enablement`（#9 已定保留）不在此表——它是"改采集"而非"删段"，可与 P2 并行。

`telemetry` 待 `SBD_V6_TELEMETRY.md` 结论，`FINAL_FIELDS` §8#5 记为架构性不可用，可能整段删除，需单独决策。

---

## 4. 下游连带改动

### 4.1 renderer（19 个）

| 类别 | 数量 | 处理 |
|------|------|------|
| 零改动存活 | 3 | `session`、`workload`、`source_files`（已改读 `metadata.*`） |
| 改读 `outcome.*` | 4 | `baseline`、`final`、`optimizations`、`attribution` |
| 直接删除 | 12 | `capability_summary`、`param_search`、`phase_timeline`、`kernel_lifecycle`、`geak_invocations`、`forge_invocations`、`roofline`、`critic_robustness`，以及 4 个 exporter 从未写过键的死 renderer：`decision_journal`、`kernel_profiling`、`kernel_decision_path`、`data_provenance` |

4 个死 renderer 可并入 P0。

`reporters/cross_section.py`（334 行）几乎全指向 legacy 段，随 P3 整体删除。`reporters/compose.py:44-67` 的 `SECTION_GROUPS` 需同步重写。

### 4.2 `breakdown/` 外

| 位置 | 读的 legacy 键 | 处理 |
|------|---------------|------|
| `tools/dump_session_breakdown.py:138-145` | `final`、`optimizations`、`kernel_lifecycle`、`sweep`、`decision_journal`、`kernel_profiling` | 随 P3 改读 `outcome.*` |
| `tools/dump_session_report.py` | 经 compose 间接读 | 同上 |

已对齐 V6、无需改动：`orchestrator/trace/langfuse_emitter.py:932`（已读 `outcome.stop_reason`）、`langfuse_mapping.py`（不读 breakdown）、`breakdown/session_package.py`（只打包路径不解析）。

---

## 5. 不可删清单

以下在 P3 完成前都是 load-bearing，任何阶段都不要删：

| 项 | 位置 | 服务的 V6 键 |
|----|------|-------------|
| `operations` 流 | `exporter.py:389` → `recorded_operations` | `timeline[framework_agent]` + `outcome.validation` |
| `measurements` / `adoptions` / `artifacts` 流 | `exporter.py:396-398` | `outcome.validation` |
| `_compose_versions` | `assembler.py:427` | `metadata.versions.tools` |
| `_normalize_kernel_route_operations` | `assembler.py:207` | 变异 `operations`，喂 timeline 与 optimizations |
| `_drop_event_rows` | `assembler.py:573` | 防止 event 子流泄漏进 envelope |
| `_pick` / `_merge_session` / `_safe_collect` | `exporter.py` | 仍服务 `metadata` / `telemetry` / `enablement` / `baseline` / `versions` 的 fallback |
| `collect_source_files` | `collectors/kernels.py` | `source_files`（当前**无 `_pick`**，`exporter.py:561` 恒走 collector；`FINAL_FIELDS` §6 要求改为产生时录路径） |
| `collect_langfuse` | `collectors/decision.py` | `metadata.langfuse` fallback |
| `collect_session` / `collect_workload` / `collect_model_info` | `collectors/sessions.py` | `metadata.*` fallback；§1 的 `[缺]` 字段（`image`、`crash_timestamps`、architecture 14 字段、tools 的 `commit` / `root_dir`）补完前是唯一参考实现 |

`operations` 四条流的退役完全取决于 #1 adoption ledger，是 P3 的最后一步。

---

## 6. 下一步

P0 无决策依赖，可立即执行。P1 的 #1 与 #8 建议在 P0 期间并行讨论。
