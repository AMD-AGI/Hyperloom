# SBD V6 第 9 项：清理方案

配套文档：`SBD_V6_FINAL_FIELDS.md`（第 1–8 项的字段定稿）。本文只回答"删什么、按什么顺序、什么拦着"。

---

## 0. 一个结论先行

**你问的两个问题是同一条链的两端。**

```
instrument.record_*  →  v4 spool section  →  assembler  →  collector  →  legacy 顶层 key  →  report renderer
   （问题 1 在这头）                                                                      （问题 2 在那头）
```

双写清不掉，是因为 v4 section 还有 reader；v4 section 还有 reader，是因为 legacy key 还要产出；legacy key 还要产出，是因为 renderer 还在读。**所以两个问题只有一个答案：从 renderer 那头往回拆。**

而这条链的终点比预期轻得多：

> **`reporters/` 整个包（3855 行、20 个 renderer）在全库只有一个消费者**：手动运维 CLI `tools/dump_session_report.py:121`。
>
> ```
> $ rg -n "render_session_report" src/hyperloom --glob '!**/tests/**'
> tools/dump_session_report.py:121:    result = render_session_report(breakdown, llm_client=llm)
> ```
>
> orchestrator、CLOSE、`SessionBreakdownExecutor`、CLI finally 路径**都不渲染 markdown 报告**。每个 session 的自动产物只有 `session_breakdown.json`（给 Langfuse / 审计）和 `reports/final.md`（`ReportExecutor` 走完全独立的路径，不碰 `reporters/`）。

**这把"下线 renderer"从风险动作变成了可做的动作。**代价是一个要人手敲命令才会跑的工具，不是生产链路。

补充一个佐证：**19 个报告小节里有 6 个在每一份报告里都是空的**——exporter 从不产出它们的 key。这个工具已经部分坏了很久，没人发现。

---

## 1. 问题 1：双写能否清成 V6 only？

### 1.1 结论

**能，但不是一步。** 分三类：

| 类 | Section 数 | 能否直接删 |
|---|---|---|
| A. 死 section（写了没人读） | 5 | **能，立刻** |
| B. 从未有 writer 的注册项 | 7 | **能，立刻** |
| C. 有真实 reader 的 v4 section | ~16 | 要先拆 renderer 链 |

`SECTION_SHAPES` 共 83 项，其中 45 项是 V6 event 子流（经 `event_sink` 写入，**不属于双写，不动**），剩下约 38 项是 v4 面。

### 1.2 A 类：死 section，可立刻删

写入方在跑，但全库没有任何消费者。

| Section | Writer | 验证 |
|---|---|---|
| `run_snapshot` | `instrument._snapshot_v4_run` | 全库检索 `"run_snapshot"` 除 `recorder.py`/`instrument.py` **零匹配** |
| `phase_transitions` | `machine_state.py:3269` | 仅 `assembler.py:36` 的 id 合并表引用，exporter 不读 |
| `subjects` | instrument 内部 | 仅 `assembler.py:37` |
| `trace_events` | `machine_state.py:3281`, `kb_writeback.py:288` | 仅 `assembler.py:42` |
| `optimization_stack`（spool 段） | `instrument._snapshot_optimization_stack` | 所有 `optimization_stack` 引用都是 `state["optimization_stack"]`，无人读 spool 段 |

`phase_transitions` 值得单独说一句：它记的 phase 迁移已由 `timeline[phase]` 的 `phase_segment` 完整覆盖（第 7 项落地），而它自己**从来就没有 reader**。删它不是"迁移完成后可以删"，是"它一直是纯写入开销"。

**动作**：删这 5 个 writer 调用 → 删 `SECTION_SHAPES` 条目 → 删 `assembler._V4_ENTITY_IDS` 中对应行。

### 1.3 B 类：注册了但从未写入

`workload`、`baseline`、`final`、`kernel_lifecycle`、`sweep`、`telemetry`、`kernel_roofline` —— 这些 `SECTION_SHAPES` 条目没有任何 writer，对应的 collector 都是从 `state.json` 或磁盘读的。纯注册表垃圾。

**动作**：删 `SECTION_SHAPES` 条目。零风险。

### 1.4 C 类：真正的双写，拦在 `optimizations` 上

C 类的**绝大部分重量压在一个地方**：

```
operations / measurements / adoptions / artifacts   （v4 四条实体流）
        ↓  唯一消费者
collect_recorded_optimizations()   （collectors/optimizations.py，1231 行）
        ↓
breakdown["optimizations"]
        ↓
_renderers/optimizations.py + _renderers/attribution.py + cross_section.py + dump_session_breakdown.py
```

四条流的消费者只有一个，就在 `exporter.py:389-398`：

```python
recorded_operations = [row for row in assembled.get("operations") or [] if isinstance(row, dict)]
optimizations = _safe_collect("optimizations", lambda: collectors.collect_recorded_optimizations(
    ..., recorded_operations,
    [... assembled.get("measurements") ...],
    [... assembled.get("adoptions") ...],
    [... assembled.get("artifacts") ...],
    ...))
```

**这意味着链条是干净的**：一旦 `optimizations` 这个 legacy key 退役，四条流立刻失去全部 reader，喂它们的那批 `instrument.record_*`（`record_action_operation`、`record_geak_operation`、`record_gemm_tuning_operation`、`record_collective_promotion`、`record_kernel_*`、`record_session_validation` 等）全部变成死代码。`instrument.py` 4935 行、28 个 `record_*` 函数的大头就在这里。

而 `optimizations` 退役的前置条件已经在第 8 项做掉了一半：**`outcome.validation` 已不再依赖它**（`collect_v6_outcome` 连参数都不收了）。剩下的只有 renderer + dump CLI + exporter 自检三处。

### 1.5 真正的 V6 缺口：五个事实只有 v4 有 —— **已决策**

清到 V6 only 会丢这五样。**落点已定**（2026-09-07）：

| 事实 | v4 位置 | 决策落点 |
|---|---|---|
| **critic agent 逐轮迭代** | `critic_iterations`（`critic_agent.py:900`） | **新增 `critic` 顶级字段**录 session 级迭代。与第 5 项（critic 进子项目内部）不冲突：per-proposal 的裁决继续挂在各子项目里，`critic` 顶级字段承载 session 级 agent 迭代本身（每轮 topic / verdict / summary / artifact 路径） |
| **robustness signals** | `robustness_signals`（`robustness_agent.py:245`） | **新增 `robustness` 顶级字段**，在 `robustness_agent.py:222` 实时录 intents。现有写入方是读空文件的空壳，见 §1.5.1 / §1.5.1a |
| **specialist runs** | `specialist_runs`（`explore_state.py:54`） | **合进 `framework_event` 的 run 行**。前提两条：(a) 不与现有 `framework_run` 重复；(b) 用 V6 方式直接写（`event_sink`），不经 `instrument` |
| **tool version 探测** | `versions` item 流（`kernel.py:2362`） | **`metadata.versions` 直接录**，由 `session_metadata` 承担，删 `record_tool_version` |
| **backend invocation 逐行** | `geak_invocations` / `forge_invocations` | **直接干掉**。新业务记录在 `timeline[kernel].ext` 的 geak / forge 子结构下（事件类型是 `kernel`，phase 是 KERNEL_AGENT）。对应 §2.1 里那两个本来就永远为空的报告小节，一并删 |

**注意 `gemm_tuning` 和 `collective_promotion` 不在这张表里**：它们不是独立 section，是写进 `operations` 的 `kind`。随 `optimizations` 一起走。

#### 1.5.1 `robustness_signals` 的写入方是空壳 —— 补录制前必须先修

决策 #2 要新增 `robustness` 顶级字段，但**现在被录进去的东西是空的**，先把这条链看清：

```
robustness_agent.py:169  allocate_turn_workdir(... "robustness-workdir" ...)
                         → 写入 request.json、emit.json
robustness_agent.py:245  instrument.record_robustness_signal(workdir=workdir)
instrument.py:4370-4371  read_json(wd / "signal.json")     ← 全库无人写
                         read_json(wd / "action.json")     ← 全库无人写
instrument.py:4372-4377  payload = {"ts": "", "signal": "", "action": "", "workdir": "<NNN>"}
```

`signal.json` / `action.json` 在全库只有 **read** 没有 **write**：

```
$ rg -n "signal\.json|action\.json" src/hyperloom --glob '!**/tests/**'
collectors/telemetry.py:80,81       ← 读
recorder/instrument.py:4370,4371    ← 读
（无写入方）
```

agent 实际写的是 `request.json` / `emit.json`，两个名字都不对。所以**每一行 `robustness_signals` 都是三个空字符串加一个目录名**。它还顺带在 `operations` 里铸一行 `status="partial"` / `metadata_completeness="partial"`——那个 "partial" 就是这个空壳的自我描述。

`collect_critic_robustness`（`telemetry.py:75-89`）走磁盘也读同两个文件名，同样永远空。

**含义**：决策 #2 不是"把 v4 的 robustness 记录搬到 V6"，而是**先决定 robustness signal 这个业务事实到底该由谁产出**。

#### 1.5.1a 决策：不读文件，在产出时刻实时录

**已定（2026-09-07）**：signals 全部改成实时 record，`robustness` 建顶级字段。

这比"改读 `emit.json`"更彻底——**一个文件都不用读**。事实在调用点上方四行就已经在内存里了：

```
robustness_agent.py:216  envelope = emit.get("intent_envelope")
robustness_agent.py:222  intents = validate_envelope(envelope)      ← 事实在这一刻成立
robustness_agent.py:226  parse_warnings = list(emit.get("parse_warnings") or [])
robustness_agent.py:228  intent_summary = [(i.type.value, i.payload.get(...)) for i in intents]
robustness_agent.py:245  instrument.record_robustness_signal(workdir=workdir)   ← 却去磁盘捞空文件
```

现有代码把手里已经校验过的 `intents` 丢掉，转身去读两个不存在的文件。这正是 v4 "导出时重建"的老病：事实产出的那一刻不录，事后从磁盘拼。

**录制点**：`robustness_agent.py:222` 之后（`validate_envelope` 成功、workdir 未被裁剪）。此刻同时在作用域里的：

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `turn_idx` / `tick_index` | 参数 / `emit["tick_index"]` | 天然 key |
| `intents[]` | `validate_envelope` 返回 | 每条有 `type.value` + `payload`（`severity` / `topic`） |
| `parse_warnings[]` | `emit["parse_warnings"]` | agent 自述的解析问题 |
| `workdir` | `allocate_turn_workdir` | 保留为溯源指针，不再当数据源 |

失败路径也要录，且比成功路径更值钱——`NoIntentEmitted`（envelope 非法）和 `BackendError`（emit 缺 envelope）现在都只进日志，报告里看不到 robustness agent 哑火过。录 `outcome=invalid_envelope` / `no_envelope`，一并带上 `parse_warnings`。

**落点**：`robustness` 顶级字段。V6 里非 timeline 的顶级内容照 `close` 的成例做——recorder 写 section，collector 读出来装成顶级键（`close_out.py` 写 / `v6_close.py` 读）。robustness 是 session 级的一条独立通道，不挂在任何 phase / macro cycle 上，所以走这个模式而非 timeline 事件。

删掉 `instrument.record_robustness_signal` 和 `collect_critic_robustness` 里那段同样读空文件的磁盘路径（`telemetry.py:75-89`）。

### 1.6 问题 1 的答案

**能清成 V6 only，路径是：先删 A+B 类 12 个 section（零风险，今天就能做）→ 拆 `optimizations` 的 renderer 依赖 → 四条 v4 实体流和 `instrument.py` 大头一起死。**

`optimizations` 顶级字段已定**废弃**，所以 `operations` / `measurements` / `adoptions` / `artifacts` 四条流连同喂它们的那批 `instrument.record_*` 一起删（见批次 3）。§1.5 五个缺口的落点已全部定完，`robustness` 一条改为在 agent 产出 intents 的那一刻实时录，不再走磁盘（§1.5.1a）。

---

## 2. 问题 2：report renderer 能否下线？

你的目标是"新字段里有对应的业务记录 + 下线"。逐节对了一遍，20 个 renderer 分四档。

### 2.1 立刻下线：6 个小节现在就是空的

exporter **从不产出**这些 key，它们在每一份报告里都被 skip：

| 小节 | 读的 key | exporter 产出？ |
|---|---|---|
| `decision_journal` | `decision_journal` | 无 collector、无 key |
| `kernel_profiling` | `kernel_profiling` | 无 |
| `kernel_decision_path` | `kernel_decision_path` | 无 |
| `data_provenance` | `data_provenance` | 无 |
| `geak_invocations` | `geak_invocations` | collector 在 `exporter.py:330` 跑了，**没放进 dict** |
| `forge_invocations` | `forge_invocations` | 同上 |

后两个尤其荒谬：`compose.py:55-57` 有一段注释解释为什么这两节重要（"没有它们，报告会报出某个 lane 采纳了多少 kernel，却不显示背后任何一次尝试"），而这个意图**从未生效过**——数据算出来了，忘了写进文档。

**下线这 6 个不丢任何东西,因为它们从来没产出过东西。**这是纯删除:6 个 renderer 模块 + `SECTION_GROUPS` 里 6 个 id + `cross_section._SECTIONS_WITHOUT_PRODUCER`（这张表本身就是为了压制这几节的"数据缺失"告警而存在的,一起删）。

### 2.2 已经是 V6：2 个小节无需动

`session` 和 `workload` 的 renderer 已经在读 `metadata.session` / `metadata.task_config`（经 `base.py` 的 `session_of()` / `task_config_of()`）。legacy 顶层 `session` / `workload` 删掉它们不受影响。

### 2.3 改读 V6 即可下线 legacy key：5 个小节

业务记录在 V6 里齐全，只是 renderer 指错了地方。

| 小节 | 现在读 | 改读 |
|---|---|---|
| `optimizations` | `optimizations.summary_by_source` / `validation` / `entries[]` | `outcome.validation.attribution.by_source` + `timeline[stack].ext.adoptions.rows[]` |
| `attribution` | `optimizations.validation`（不读 legacy `attribution` key，那个 key 从来没 export 过） | `outcome.validation.*` |
| `phase_timeline` | `phase_timeline[]` 扁平行 | `timeline[phase].ext.actions.rows[]` |
| `source_files` | `source_files` | key 名不变，collector 换实现 |
| `cross_section` 的 6 个函数 | `baseline`/`final`/`optimizations`/`attribution`/`capability_summary`/`kernel_lifecycle` | `outcome.*` + timeline 聚合 |

`cross_section.py` 值得单说：它的 `_gain_attribution_lines` 有一条 fallback 读 legacy `attribution.source_breakdown`——那个 key **exporter 从不产出**，所以这条 fallback 永远走不到。删。

### 2.4 V6 有缺口的小节 —— **已决策**

这几个的业务信息 V6 只覆盖了一部分。**这是你"新字段里有对应的业务记录"这个前提目前不成立的地方**，和 `SBD_V6_FINAL_FIELDS.md` 里标 `[缺]` 的条目一一对应。落点已定（2026-09-07）：

| 小节 | V6 缺什么 | 决策落点 |
|---|---|---|
| `baseline` | attempts 历史表、Invocation 子块（image/command/envs/log）、`ttft_e2el_source` 标注、`failure_streak` | **补进 `timeline[baseline]`**。attempts → `ext.actions[].runs[].rounds[]`；Invocation / 标注 / streak 挂在 action 上 |
| `final` | `revalidation_pending`、`geak_pending`、GEAK drop 叙事 | **进 `timeline[kernel].ext.geak` 自己的子结构**——这三样确实都是 geak 的事实 |
| `final`（续） | `validated_at_stack_len` / `validated_ts` | **不是 geak 的**，见 §2.4.1。已由第 8 项的 `timeline[stack].ext.validations` 覆盖，**无需新录**，只要 renderer 改读 |
| `final`（续） | final 的 TTFT / e2el | **不是 geak 的**，见 §2.4.1。属于测出最终配置的那次测量，落在 `timeline[baseline]` 的 action measurement 上 |
| `roofline` | per-kernel 表、完整 snapshot 字段、progress 曲线 | **进 roofline 事件子结构自动录制**；其它经 tool 调用的路径也一并加这些字段 |
| `kernel_lifecycle` | detected[] 的 profiling 富字段（`duration_us` / `call_count` / 利用率 / `recommended_*`）、forge 多 attempt 历史 | **加到 profile 子结构里** |
| `capability_summary` | 整张 rollup 表无 V6 一等字段 | **在 timeline 里分开即可**，不再建 rollup 顶级字段；报告要的话由 renderer 从各事件聚合 |
| `critic_robustness` | per-proposal 全表 | **待判**，见 §2.4.2 |
| `param_search` | `discovered_flags`、`synergy_attempted` | `synergy_attempted` 已录（改读 attempts）；`discovered_flags` 生产者不存在，**待判**。见 §2.4.3 / §2.4.3a |

#### 2.4.1 `final` 里哪些不是 geak 的

你的原话是"如果也是 geak 的情况也是一样的"，所以先答这个条件：

**`validated_at_stack_len` / `validated_ts` 不是 geak 的。**它们是**会话级验证水印**——`_update_cumulative_gain_validated`（`writeback.py:839-905`）在任何促进路径成功后都会更新它，geak 只是其中一个调用方（另有 integrate、explore、integrate_patch、warm replay、gemm）。这个事实第 8 项已经录了：`timeline[stack].ext.validations.rows[]` 每行带 `stack_len` / `ts` / `validated_gain_pct` / `source` / `measurement_basis`，`settled` 指向最终那次，`at_head` 说明它覆不覆盖最终栈长。**不需要新录制，renderer 改读即可。**

**final 的 TTFT / e2el 也不是 geak 的。**它们是"最终配置的延迟"，由测出最终配置的那次测量产生。`_lift_to_current_best` 已经把 `ttft_mean_ms` / `e2el_mean_ms` / `tpot_mean_ms` 写进 `current_best`（`writeback.py:3300` 附近），但那是 state 不是录制。落点应该跟 baseline 的延迟对称——挂在 `timeline[baseline]` 的 action measurement 上，因为它们是同一种事实（一次 benchmark 的延迟读数），只是测的配置不同。

#### 2.4.2 `critic_robustness` 的 per-proposal 全表：**这张表不存在**

先说结论：审计里说的"per-proposal 全表"是**从代码结构推断出来的，实际从未渲染过**。

数据侧其实很干净，两个来源本来就是分开的：

```
critic_robustness = {
    "critic_iterations":  [...],   ← critic-workdir/<NNN>/{review,emit}.json
    "robustness_signals": [...],   ← robustness-workdir/<NNN>/{signal,action}.json
    "kb_writes_summary":  {...},
}
```

`_compose_critic_robustness`（`assembler.py:536-557`）和 `collect_critic_robustness`（`telemetry.py:30-89`）都产出这个**三键 dict**。

而 renderer 把它当 **list** 用：

```python
cr = breakdown.get("critic_robustness") or []      # critic_robustness.py:33
for c in cr:                                        # 迭代 dict → 拿到的是 KEY 字符串
    if isinstance(c, dict): cr_norm.append(c)
    else: cr_norm.append({"prompt": str(c)})        # → {"prompt": "critic_iterations"} …
```

迭代一个 dict 得到的是它的三个键名。所以这个小节实际渲染出的是三行 prompt-only 条目，然后因为"无 actionable payload"被 skip。**它和 §2.1 那 6 个空小节是同一类**，只是坏的方式更隐蔽。

**所以问题"是 critic 的还是 robustness 的"答案是：两者都有，且数据侧已经分好了，正好对应你 §1.5 的决策 #1（`critic` 顶级字段）和 #2（`robustness` 顶级字段）。**

这条一分为二后：

- `critic_iterations` → `critic` 顶级字段（每轮 iter / ts / topic / verdict / summary + 四个 artifact 路径）——数据源真实可用
- `robustness_signals` → `robustness` 顶级字段，改为实时录 intents（§1.5.1a）；现有磁盘路径读错文件名、内容永远空，一并删
- `critic_robustness` 这个合成字段 + 它的 renderer + `kb_writes_summary` 一起删

#### 2.4.3 `param_search` 的两个字段：**能力未实现，不是冗余字段**

你问的是"没实现还是单纯冗余"。答案是**前者，而且比报告字段严重——这两个都在喂 orchestration prompt**。

**`discovered_flags`：AST 扫描器不存在。**

管道是完整的：state 字段（`shared_state.py:1115`）、写入方（`explore_state.py:759 record_discovered_flags`）、prompt 消费方（`prompt_builder.py:698`："pull untested boolean toggles from `discovered_flags.<framework>.backend_flags`"）、报告列。缺的是**生产者**：

```
writeback.py:4263   disc_update = result.get("discovered_flags_update")
writeback.py:4264   if isinstance(disc_update, dict): → record_discovered_flags(...)
explore.py:2126     "discovered_flags_update": None,      ← 唯一的产出点，硬编码 None
```

`record_discovered_flags` 的 docstring 写的是"Persist the **AST-discovered** flag list"。那个 AST 扫描从未实现，所以 `discovered_flags` 恒为 `{}`，而 prompt 每轮都在告诉 agent 去读它。

**`synergy_attempted`：去重历史永远为空，且它要去重的那个参数也不存在。**

合并逻辑在 `shared_state.py:1647-1676`，注释写"normalize executor-side combos"。但它只从一个源读：

```python
for source in (existing.get("synergy_attempted") or [],):    # 单元素 tuple
```

一个只有一个元素的 tuple 上跑 for 循环——这是它本来打算迭代 **existing + update** 两个源的痕迹，而 update 那个源从没接上。normalizer 同时处理 `["a","b"]` 和 `"a+b"` 两种形态，说明它是为了接收 executor 输出而写的，那个输出从未到达。`explore.py` 只在 128 / 1000 行把它 seed 成 `[]`，2066 行原样传回。**这是一个在空值上的不动点。**

更关键的是它要去重的对象：

```
$ rg -n "synergy" src/hyperloom --glob '!**/tests/**'
prompt_builder.py:692    "   `synergy_mode='auto'` (deduped against `synergy_attempted`)."
shared_state.py:1651     （normalizer docstring）
_renderers/param_search.py   （报告列）
```

**`synergy_mode` 这个参数在全库除了这行 prompt 之外不存在。**prompt 在指挥 agent 使用一个不存在的参数，并去重一个永远为空的历史——也就是说 agent 可以无限次重试同一个 synergy 组合。

**这两条都不是"清理"范围内的事，是 §8 #6（explore 三个 ledger 死了两个）的延伸，且影响的是 agent 行为不是报告外观。**处置选项：

- (a) 实现：写 AST 扫描器 + 接上 synergy 组合的 append，两个 prompt 指令才成立
- (b) 删干净：删两个 state 字段 + writer + normalizer + 报告列 + **prompt 里那两条指令**（第 2 条和第 4 条 idea generation 规则）
- (c) 只删 prompt 指令，保留 state 字段待日后实现

不建议维持现状：现状是 prompt 在对 agent 撒谎。

#### 2.4.3a 能否实时 record？两个字段答案相反

判据只有一条：**存在某一刻这个事实成立吗**。实时录制录的是已发生的事，不是补算没发生的事。

**`synergy_attempted`：不需要新录——事实已经录了两遍。**

一个 synergy 组合就是一个"改了多个 flag 的 variant 被下发并测量"。这件事每次都真实发生，而且每次都已经落在两处：

| 已有落点 | 内容 |
| --- | --- |
| `timeline[framework_agent].ext.attempts[]`（V6） | `config_delta`（`extra_server_args` / `extra_envs` / `remove_args` / `unset_envs` / `args_mode`）+ `fingerprint` + `variant_name` + `round_id` + `outcome` + `decision` + `measurement.gain_pct` |
| `explore_search.winners_history[]` / `.rejected[]`（state） | `extra_args` + `extra_envs` + 同套 controls + `gain_pct` + `round_id` |

也就是说"哪些组合试过"是 `{normalize(row.config_delta) for row in attempts}`，而且比 `synergy_attempted` 强：attempt 行连**结果**一起带着，agent 能区分"试过且赢了"和"试过且亏了"，而一个裸组合列表只能说"试过"。

`synergy_attempted` 从来不是一个缺失的事实，而是**同一事实的第二份手抄本**，抄写的那一步（executor→state 的 append）没接上，于是手抄本永远空白。所以处置是 (b) 删，不是补录制：删字段 + normalizer 那个单元素 tuple，prompt 的去重指令改指向已录的 attempt / winners 台账。

**`discovered_flags`：不能实时 record——没有那一刻。**

消费方签名自己说明了本来打算怎么产出：

```python
record_discovered_flags(framework=..., backend_flags=..., param_flags=..., source_path=...)
```

`source_path` 说明生产者是**对 framework 源码做一次扫描**。那个扫描器没写，所以不存在任何一刻这个事实成立——不是"录制点没接上"，是**能力缺失**。无论录制层怎么改都变不出 flag 列表来。这跟 `synergy_attempted` 有本质区别：后者事实发生了只是没抄，前者事实从未发生。

所以只有两条路：实现扫描器（(a)），或连 prompt 指令一起删（(b)）。**不能靠录制补。**

**#7.2 落点（若日后实现扫描器）**：`metadata`，跟你已定的 `metadata.versions` 并列（如 `metadata.framework_flags`）。理由：flag 集合是 framework / 环境的静态属性，一次探测得出、session 内不变、不挂任何 phase 或 macro cycle——和版本探测是同一类事实，不该进 timeline。

### 2.5 问题 2 的答案

**能下线,而且比你预期的容易——因为 renderer 不在生产链路上。**

- **7 个小节立刻删**,零损失（6 个从来是空的，加上 §2.4.2 查明的 `critic_robustness`——它把 dict 当 list 迭代，同样从未渲染过内容）
- 2 个小节**已经是 V6**
- 5 个小节**改读 V6** 就能放掉 legacy key
- 剩下的**有真实缺口**,落点已在 §2.4 定完，只余 `discovered_flags` 一项待判（§2.4.3a：不是录制问题，是能力缺失）

一个额外选项要摆在桌上:既然 `reporters/` 只服务一个手动 CLI,**整包下线**也是合法选择——`session_breakdown.json` 本身就是给机器读的,markdown 报告是给人看的方便产物。如果这个工具实际没人用（6 个空小节没人报过 bug 是个信号），把 3855 行连同它服务的 12 个 legacy key 一起删,比逐个 repoint 省事得多。这个我不替你决定,但它应该被当成一个选项。

---

## 3. 第 9 项完整清单

按依赖顺序排。每批做完可以独立跑测试。

**执行顺序已定（2026-09-07）**：`reporters/` 的去向暂缓，先做 **批次 0 → 1 → 4 → 3**，把批次 2 和 reporters 决策留到最后。

理由：批次 1（零 reader）和批次 4（补录制）与 reporters 无关，前者纯删死码、后者纯增事实,两批都不受那个决定影响,可以立刻做。批次 2 是"改渲染器读 V6",它的工作量完全取决于 reporters 是否保留,所以必须等决策。

**这个顺序有一个后果要认**：批次 3 退役 v4 写入面之后，`optimizations` 等 legacy 顶级字段不再产出，而 `reporters/` 目前仍在读它们，于是**手动 CLI 报告会出现空洞小节**（不会崩——小节取不到内容就静默跳过，`critic_robustness` 本来就是这个状态）。因为 CLI 不在 production 路径上，这个窗口期可以接受；等 reporters 决策落地时，选 repoint 就补回来，选下线就一起删掉。若不能接受这个窗口，把批次 3 也推到决策之后。

### 批次 0：先修两个活 bug（不是清理）

**0a. `warm_replay` 事件在生产里重复发布两次。**

`conc_sweep` 有 guard 且注释解释了原因（`v6.py:344-349`）：

```python
if not _recorded_types(timeline, "conc_sweep"):
    # A sweep the executor recorded already carries everything the
    # projection could rebuild ... projecting alongside it would publish
    # the same sweep twice -- the second time worse.
```

`collect_kb_events`（`kb_timeline.py:407`）**无条件**投影一个 warm_replay 事件，而 `prelude.py` 已经在录真事件。同一个 warm replay 出现两次。

修法：`v6.py:357` 加同样的 `_recorded_types(timeline, "warm_replay")` guard。

**0b. ~~`geak_invocations` / `forge_invocations` 算了不用~~ —— 原判断有误，已更正。**

原文说"跑了 collector 没写进 dict，算了不用"。**前半句对，后半句错**：这两个列表虽然从未作为 breakdown 顶级 key 输出，但被三个下游 collector 消费——

```
exporter.py:324-331  geak_c, forge_c = collect_kernel_invocations(...)   # _pick 优先取已录的 v4 section
exporter.py:341-347  → collect_capability_summary(state, geak_invocations, ..., forge_invocations)
exporter.py:353-359  → collect_kernel_lifecycle(sd, state, geak_invocations, ..., forge_invocations)
exporter.py:394-400  → collect_recorded_optimizations(..., geak_invocations, forge_invocations)
```

所以这里**没有可修的 bug**，`exporter.py` 那两行是正常的中间变量。真正死的是**两个报告小节**：`_renderers/invocations.py:103/121` 注册的 renderer 去读 `breakdown["geak_invocations"]`，而 exporter 从不输出这个 key（这两个名字只在 `recorder.py:96-97` 作为 section 存在）。这与 §2.1 的结论一致，删除动作已在批次 1 / S2 覆盖，不属于批次 0。

**同时修正 §1.5 决策 #5 的可行性**：v4 的 invocations 两条流不能"直接干掉"，它们目前是 `capability_summary` 和 `kernel_lifecycle` 的输入。删流之前这两个 collector 必须先改读 `timeline[kernel].ext` 的 geak / forge 子结构——即这项删除属于 **S5/S6**，不是独立的早期清理。

结论：批次 0 只剩 0a 一项。

### 批次 1：零 reader，直接删

| 对象 | 位置 | 证据 |
|---|---|---|
| 5 个死 v4 section + writer | §1.2 | 全库检索无消费者 |
| 7 个无 writer 的注册项 | §1.3 | 无 writer |
| 6 个空报告小节 + renderer | §2.1 | exporter 不产出 key |
| `critic_robustness` 小节 + renderer + `kb_writes_summary` | §2.4.2 | renderer 把三键 dict 当 list 迭代，从未渲染过内容 |
| `geak_invocations` / `forge_invocations` 两条 v4 流 + `record_kernel_invocations` | §1.5 决策 #5 | 业务记录已在 `timeline[kernel].ext` 的 geak / forge 子结构 |
| `cross_section._SECTIONS_WITHOUT_PRODUCER` | `cross_section.py:182-188` | 服务对象删完即无意义 |
| `collectors/attribution.py` 整模块 | 816 行 | 不在 `exporter.build`，仅测试调用 |
| `collect_optimization_stack` | `kernels.py:1169` | 仅 `test_sbd_optimizations.py:86` |
| `collect_gemm_tuning` | `kernels.py:1273` | 仅 `test_geak_breakdown_unit.py:703` |
| 7 个零 reader 的 legacy 顶层 key | `phase_segments`、`enablement`、`kernel_journey`、`kernel_optimization_summary`、`token_usage`、`roofline_progress`、`specialist_runs` | 全库检索无 `breakdown.get(...)` |

`enablement` 和 `phase_segments` 是第 6、7 项的直接成果——事件录制上线后投影就没人读了。

### 批次 2：改读 V6，放掉 legacy key

1. `_renderers/optimizations.py` + `_renderers/attribution.py` → `outcome.validation` + `timeline[stack]`
2. `cross_section.py` 6 个函数 → `outcome.*` + timeline（同时删 legacy `attribution` fallback）
3. `tools/dump_session_breakdown.py:139-140` → `outcome.validation`
4. `exporter.py:415-440` geak 一致性自检 → 读 `timeline[stack].ext.adoptions.by_source.kernel.by_backend`
5. **然后删** `optimizations` key + `collect_recorded_optimizations`（1231 行）
6. `_renderers/phase_timeline.py` → `timeline[phase]`，然后删 `phase_timeline` key + `exporter._merge_phase_timeline`

第 5 步是整个第 9 项的枢纽——它一落地，批次 3 就全变成死代码。

### 批次 3：v4 写入面退役

`optimizations` 一死，四条 v4 实体流失去全部 reader：

1. 删 `operations` / `measurements` / `adoptions` / `artifacts` 四个 section
2. 删喂它们的 `instrument.record_*`（`record_action_operation`、`record_geak_operation`、`record_gemm_tuning_operation`、`record_collective_promotion`、`record_kernel_discovery/dispatch/backend_result/e2e`、`record_session_validation`、`record_operation`、`record_measurement`、`record_adoption`、`record_artifact` …）
3. 删 `assembler.py` 的 legacy 块：`_V4_ENTITY_IDS` + 实体合并栈、`_normalize_kernel_route_operations`、`_compose_kernel_journey`、`_compose_critic_robustness`、`_compose_versions`——约占 1033 行的六成
4. 删 `kernel.py:2291-2367` 那段把 GEAK journey 整个 replay 进 instrument 的代码（V6 `kernel_event` 已录）
5. 删 `writeback.py:884` 处 `instrument.record_session_validation` 那一半双写（`stack_event.record_validation` 已在旁边）

### 批次 4：补录制（落点已定，见 §1.5 / §2.4）

这批是**新录制工作**，不是清理。按依赖排：

| # | 工作 | 落点 | 阻塞 |
|---|---|---|---|
| 1 | tool version | `metadata.versions` 由 `session_metadata` 直接录 | 无 |
| 2 | specialist runs | 合进 `framework_event` 的 run 行，走 `event_sink` | 先确认与现有 `framework_run` 不重复 |
| 3 | critic session 级迭代 | 新 `critic` 顶级字段 | 无（`critic-workdir` 数据源真实可用） |
| 4 | baseline attempts / Invocation / 标注 / streak | `timeline[baseline]` 的 action 与 rounds | 无 |
| 5 | final 的 TTFT / e2el | `timeline[baseline]` action measurement（§2.4.1） | 无 |
| 6 | geak 的 `revalidation_pending` / `geak_pending` / drop 叙事 | `timeline[kernel].ext.geak` | 无 |
| 7 | roofline per-kernel 表 / 完整 snapshot / progress | roofline 事件子结构；其它 tool 调用路径同补 | 无 |
| 8 | kernel profiling 富字段 | profile 子结构 | 无 |
| 9 | robustness signals | 新 `robustness` 顶级字段，在 `robustness_agent.py:222` 实时录 intents（§1.5.1a） | 无（不再读磁盘，事实已在内存） |
| 10 | `capability_summary` | 不建顶级字段，timeline 里分开 | 无 |

**无需新录制的三项**：

- `validated_at_stack_len` / `validated_ts` 已由第 8 项的 `timeline[stack].ext.validations` 覆盖（§2.4.1），只要 renderer 改读
- `synergy_attempted` 已由 `timeline[framework_agent].ext.attempts[].config_delta` 覆盖（§2.4.3a），属删除范围而非补录范围
- `discovered_flags` 无法靠录制补（生产者不存在，§2.4.3a），需单独决定实现还是删除

### 不能删的例外

**`collect_decision_trace`。** 它的 breakdown key 是死的（无人读），但它有副作用：写 `reports/trace/decision_trace.jsonl`，而那是 Langfuse 的数据源（`langfuse_emitter.py:1253`、`langfuse_mapping.py:427`）。**key 可以删，collector 不能删。**

---

## 4. 量级估计

| 批次 | 删除量（估） | 风险 |
|---|---|---|
| 0（修 bug） | +10 行 | 无 |
| 1 | ~1500 行 | 无（零 reader 已验证） |
| 2 | ~1800 行（含 `optimizations.py` 1231） | 中（要改 renderer 逻辑） |
| 3 | ~3500 行（`instrument.py` 大头 + `assembler` 六成） | 中（双写去掉后无回退） |
| 4 | 净增 | — |

批次 1–3 合计约 **6800 行**净删除。如果选 §2.5 那个"整包下线 `reporters/`"的激进选项，再加 3855 行，批次 2 的工作量也基本消失。

---

## 5. 决策状态

### 已定（2026-09-07）

- **§1.4 C 类**：`optimizations` 顶级字段废弃，四条 v4 实体流一起删
- **§1.5 五个 v4-only 事实**：落点全定（`critic` / `robustness` 两个新顶级字段、specialist runs 合进 framework run 行、versions 进 metadata、invocations 直接删）
- **§2.4 缺口落点**：baseline / final-geak / roofline / kernel profiling / capability_summary 全定
- **§1.5.1a robustness 数据源**：不读文件，在 `validate_envelope` 之后实时录 intents（含失败路径），落 `robustness` 顶级字段，走 `close` 那套 recorder+collector 成例
- **§2.4.3a `synergy_attempted`**：判定为已录事实的第二份手抄本，删字段 + normalizer，prompt 去重改指向 attempt / winners 台账
- **§2.4.3a `discovered_flags`**：**删干净**。AST 扫描器从未实现且不补，删 state 字段（`shared_state.py:1115`）+ `record_discovered_flags`（`explore_state.py:759`）+ normalizer + `writeback.py:4263-4273` 的消费分支 + `explore.py:2126` 的硬编码 `None` + 报告列 + **`prompt_builder.py:698` 那条 "Mine flags" 指令**。附带删 `discovered_flags_error`。

- **执行顺序**：`reporters/` 去向暂缓，先做批次 0 → 1 → 4 → 3，批次 2 留到最后（见 §3 开头，含"CLI 报告空洞窗口期"的后果说明）

### 待判（一项，已推迟）

1. **`reporters/` 整包下线，还是逐个 repoint？** 已明确它是**呈现层**（投影层下游），3855 行 / 20 个渲染器 + 可选 LLM 叙述层，唯一消费方是手动 CLI `dump_session_report.py:121`，production 路径（orchestrator / CLOSE / `SessionBreakdownExecutor`）不渲染 markdown。实质是"还要不要人读报告能力"。**决定推迟到批次 3 之后**，届时 legacy 字段已停产，正好按实际需要判断。

---

## 6. 执行计划

### 6.0 进度

| 步 | 状态 |
|---|---|
| **S1** | **已完成**（2026-09-07）。`v6.py:358` 加 `warm_replay` guard + 修 docstring；新增 `test_a_recorded_replay_is_not_projected_a_second_time` 并验证其在无 guard 时确实失败（`['warm_replay','warm_replay']`）。批次 0 的 0b 经查为误判，已更正（见 §3 批次 0）。SBD 相关 1235 测试通过。 |
| S2–S7 | 未开始 |

**提交边界的现实约束**：工作树当前有 90 个文件未提交（第 1–8 项的累积成果），所以 §6.2 "每步 1 个提交"要先把历史工作分离出去才成立。S1 的改动本身只涉及 2 个文件（`collectors/v6.py`、`test_sbd_v6_kb_timeline.py`）。

### 6.1 顺序与理由

```
S1 修活 bug  →  S2 删死码  →  S3 补录制  →  S4 退 v4-only 写入面
                                                      ↓
                              S6 改渲染器  ←  S5 退 v4 实体流双写
                                    ↑
                              reporters 决策（此处才需要）
```

顺序不是"从易到难"，是**被一条约束定死的：任何 v4 写入面的删除，必须在同一事实的 V6 录制上线之后**。

这条约束产生一个 §3 没写出来的中间步骤。§3 的批次 3 只覆盖了 `operations` / `measurements` / `adoptions` / `artifacts` 四条**实体流**，而 §1.5 那**五个 v4-only 事实**（critic / robustness / specialist_runs / versions / invocations）的写入面退役不属于任何批次。它必须单列，且必须排在补录制之后——否则会出现一段"v4 已删、V6 未录"的真空，那批事实在窗口期内彻底丢失，而不是像 CLI 报告空洞那样只是不显示。这是 S4。

其余顺序按已定决策（§5）：`reporters/` 去向暂缓，所以依赖它的 S6 排最后；S1–S5 都不受那个决定影响。

### 6.2 步骤表

| 步 | 内容 | 对应 §3 | 验收门 | 提交边界 |
|---|---|---|---|---|
| **S1** | 修 `warm_replay` 重复发布 | 批次 0 | 已录 replay 的 session 里事件恰好 1 个，且留下的是录制版 | 1 个提交 |
| **S2** | 删零 reader 的死码（12 类，含 7 个 legacy 顶层 key） | 批次 1 | 全库检索无残留引用；测试全绿 | 按对象分 3–4 个提交 |
| **S3** | 补 10 项录制（含 `robustness` / `critic` 两个新顶级字段） | 批次 4 | **事实对等检查**，见 §6.3 | 每项 1 个提交 |
| **S4** | 退役四个 v4-only 事实的写入面 | *§3 缺，本节补* | S3 对等检查通过后才动 | 1 个提交 |
| **S5** | 退役四条 v4 实体流 + 双写 | 批次 3 | 测试全绿；接受 CLI 报告空洞窗口 | 按流分 4 个提交 |
| **S6** | 渲染器改读 V6 / 或整包下线 | 批次 2 | 取决于 reporters 决策 | 决策后再排 |
| **S7** | 删 `discovered_flags` + `synergy_attempted` | §2.4.3a | 见 §6.4 | **单独 1 个提交** |

S7 与 S1–S6 无依赖，可任意插入，但**必须单独提交**：它是全清单里唯一改 agent 行为的一步（动 `prompt_builder`），出问题要能独立回滚，不该和删死码混在一个提交里。

### 6.3 S3 的验收门：事实对等检查

删 v4 之前要证明 V6 真的接住了，靠"测试全绿"不够——死字段不会让测试失败。方法是拿**同一条真 session** 跑一次，逐事实比对：

| 事实 | v4 旧位置 | V6 新位置 | 判据 |
|---|---|---|---|
| tool versions | `versions` section | `metadata.versions` | 键集合相同、值相同 |
| specialist runs | `specialist_runs` section | `framework_event` run 行 | 条数相同、无重复（与既有 `framework_run` 去重后） |
| critic 迭代 | `critic_iterations` section | `critic` 顶级字段 | 条数相同、每条 workdir 可溯源 |
| robustness signals | `robustness_signals` section | `robustness` 顶级字段 | **不可比对等**——旧的恒为空壳（§1.5.1），新的应当非空。判据是"新字段有真实 intent 内容"，而非"与旧值相同" |

robustness 那一行是这批里唯一**不能**用对等法验收的：旧值是空的，对等只能对等出"两边都空"。它的验收是正向的——一条跑过 robustness agent 的 session 里，`robustness` 字段应含至少一条带 `type` / `severity` 的 intent，且 agent 哑火的 turn 应留下 `outcome=invalid_envelope` 之类的记录。

### 6.4 S7 的范围（`discovered_flags` + `synergy_attempted`）

两个字段的删除面不同，别混着改：

**`discovered_flags`（能力不存在，全删）**

- state 字段 `shared_state.py:1115` + 两处 `setdefault`（`explore.py:127`、`explore.py:1001`）
- writer `record_discovered_flags`（`explore_state.py:759`）+ `discovered_flags_error`
- 消费分支 `writeback.py:4263-4273`
- 硬编码产出点 `explore.py:2126`
- prompt 指令 `prompt_builder.py:698`（"Mine flags"）
- 报告列（`_renderers/param_search.py`）

**`synergy_attempted`（事实已录，删副本 + 改 prompt 指向）**

- state 字段 + `explore.py:128` / `explore.py:1000` 两处 seed + `2066` 的原样回传
- normalizer `shared_state.py:1647-1676`（含那个单元素 tuple）
- 报告列
- prompt 指令 `prompt_builder.py:692`：**不是删，是改指向**——去重依据从 `synergy_attempted` 换成已录的 attempt / winners 台账，同时删掉 `synergy_mode='auto'`（全库不存在的参数）

改完要确认 prompt 里不再出现任何指向已删字段或不存在参数的指令，这是本步的真正验收点。

### 6.5 通用约束

- **测试同步**：S2 删的对象里有几个只被测试引用（`collect_optimization_stack` ← `test_sbd_optimizations.py:86`、`collect_gemm_tuning` ← `test_geak_breakdown_unit.py:703`）。删实现的同一个提交里删对应测试，不留悬空测试。
- **`collect_decision_trace` 不能删**（§3 例外）：它的 breakdown key 可删，collector 有 Langfuse 副作用。S2 只删 key。
- **每步独立可跑**：每步做完 SBD 相关测试应全绿，不依赖后续步骤。
- **不在本次范围**：`reporters/` 的最终去向（S6，待决策）、AST 扫描器的实现（已定不做）。
