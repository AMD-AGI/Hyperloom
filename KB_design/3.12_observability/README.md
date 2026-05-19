# §3.12 可观测性 — breakdown schema v2 + 新段落

## 1. 设计目标

让 v0.8 新增的所有概念 (phase / specialist / KB writes / lane 占用)
都有**明确的可观测出口**, 集中体现在 `session_breakdown.json` schema
v2 中, 同时保持 v1 reader 的兼容 (老 dashboard 不需要立即升级)。

成功标准:

- 一份 schema v2 spec, 列清新增 / 修改 / 删除段。
- v1 reader 在 v2 文件上**不会崩溃**, 多余字段忽略, 已知字段保留语义。
- 新增段 (specialist_runs / kb_provenance / lane_timeline) 各有明确
  数据来源 + 写入时机。
- §3.1 §6 不变量观测点全部落地。

## 2. 现状回顾

v0.6 的 `session_breakdown.json` 由 `breakdown/exporter.py` 产出, 14
段 + envelope, schema 字符串 `hyperloom.session_breakdown.v1`:

```
session / workload / baseline / final / phase_timeline /
capability_summary / geak_invocations / oob_invocations /
kernel_lifecycle / param_search / sweep / critic_robustness /
telemetry / attribution / warnings / source_files
```

主要消费方: claw-stats-service, hyperloom-results-service, 离线
notebook。

痛点 (v0.6 → v0.8 的 gap):

- `phase_timeline` 是 action 序列 timeline, 不是 phase 边界 timeline。
- `capability_summary` 把 backends / params / sweep / validate_stack
  分行展示, v0.8 合并后不一致。
- 没有"specialist 派发 + 假设 + 假设结果"的可视化数据。
- 没有 KB 写入 (Cortex provenance) 的反查面。
- 没有 lane 占用情况 (operators 想看 "research_lane 峰值用量")。

## 3. 不变量

继承 §3.1 三主轴 + 三不变量。本节核心:

### Inv-12.1 — schema 向后兼容

v2 在 v1 已有段 *只增不删字段*, 新增字段在老 reader 中被忽略, 不影
响其读取。删除段 (例如 `param_search` 改名 `explore_search`) 通过
*写两份* 的方式过渡 (v2 reader 读新名, v1 reader 读老名), 不直接删。

### Inv-12.2 — 段落数据来源唯一

每个段从一处数据源派生 (一份 SharedState 字段 / 一组 events / 一
份外部 service 响应), 不允许"段 A 和段 B 都从同一个东西派生但口径不
一致"。

### Inv-12.3 — 写入时机分级

breakdown 写入分两类:

- **end-of-session** (CLOSE phase 内): 全段 dump, 一次性。
- **on-demand** (LLM 请 `session_breakdown` action 或 dashboard 拉
  取): 全段 dump, 但允许某些段截断 (例如 transcript 太大)。

不允许 *增量更新* breakdown (写一半被读会读到不一致状态)。

## 4. schema v2 段落结构

### 4.1 不动段

下列段在 v2 中字段集**完全不动**, 仅可能新增子字段:

- `session` (新增子字段: `phase_history`, `cortex_session_id`,
  `schema_version=2`)
- `workload` (无变化)
- `baseline` (无变化)
- `final` (无变化, current_best 字段)
- `geak_invocations` / `oob_invocations` / `kernel_lifecycle` (无变化)
- `sweep` (无变化, 仍记 grid + best_overall + pareto_front)
- `critic_robustness` (新增子字段: 见 §4.4)
- `telemetry` (新增 `lane_timeline`)
- `attribution` (新增 `phase_breakdown`)
- `warnings` (新增 v0.8 迁移警告条目)
- `source_files` (新增 cortex 子目录路径)

### 4.2 改动段

`phase_timeline`: 重新定义为**phase 边界 timeline + 内嵌 action 序列**
两层结构 (而非 v0.6 的单层 action 列表)。

外层每条记录 = 一段 phase:

- `phase`: enum
- `entered_at` / `exited_at`: ts
- `entered_reason` / `exited_reason`: enum (来自 §3.2 §6 词表)
- `actions[]`: 该 phase 内发生的 action 序列 (与 v0.6 的 action timeline
  字段相同结构)
- `specialist_rounds[]`: phase 内每轮 specialist 派发摘要 (仅
  EXPLORE 段非空)

兼容: v1 reader 把 v2 的 `phase_timeline` 当作 list 读, 直接拿不到
action 序列; 因此**同时**保留一个顶层 `action_timeline` 字段 (= 所有
phase 的 actions[] 拼接), 给 v1 reader 用。Inv-12.1 保证。

`capability_summary`: 每行展示一个 family 的统计。v0.8 改动:

- 删除 `backends` 行 / `params` 行 / `validate_stack` 行
- 新增 `explore` 行 (合并)
- 新增 `specialist` 行 (展示 specialist 派发 / 提议 / KEEP 数)

为 v1 兼容: 保留旧名行作 alias, 数据从 explore 行复制 (例如 `backends`
行的 `tested = explore.tested.length`, 这是近似但 v1 reader 能读到
合理数字)。

### 4.3 新段 — `specialist_runs`

数据源: `SharedState.specialist_rounds[]` + 每个 specialist task 的
`runs/specialist/<task_id>/specialist_done.json`。

每条记录字段 (概念层):

- `round_id`: int (EXPLORE 内自增)
- `dispatched_at` / `completed_at`: ts
- `domains[]`: 派出的 specialist domain 列表
- `parallelism`: 实际并发数 (= research_lane 占用峰值)
- `proposals_total`: 所有 specialist 提议的 variant 总数
- `proposals_kept`: 这一轮 explore executor 跑完后被 KEEP 的数
- `proposals_rejected`: 被 REVERT 的数
- `proposals_skipped`: 被 fingerprint dedup 跳过的数
- `kb_edge_ids[]`: T2 创建的 hypothetical edge_id 集合
- `confidence_avg`: specialist 自评信心均值
- `domain_breakdown`: 按 domain 拆解 (派几个 / 提几个 / KEEP 几个)
- `transcripts[]`: 每个 specialist 的 transcript path 引用
- `notes`: 兜底自由文本 (例如"全部 stale, 自动合成空 specialist_done")

### 4.4 新段 — `kb_provenance`

数据源: 整个 session 内的 KB 写入记录 (由 KnowledgePlane 写出时同步
落 audit log 到 `<session_dir>/runtime/cortex/.kb_audit.jsonl`)。

字段 (概念层):

- `cortex_session_id`: SharedState 镜像
- `cortex_session_summary`: T4 commit 返回的摘要
- `points_created[]`: 本 session mint 的 point (canonical_id, kind,
  point_id) 列表
- `edges_hypothesized[]`: T2 hypothesize 创建的 edge_id 列表
- `edges_promoted[]`: T3 verify confirmed 提级的 edge_id 列表
- `edges_negated[]`: T3 verify refuted 创建的 negation_edge_id 列表
- `attempts_ingested[]`: ingest-attempt 的 (sid, iter, outcome,
  metrics) 列表
- `kb_writes_from_critic[]`: critic-agent commit-review 产出的
  kb_writes 镜像 (与上面合流, 只是来源标 `critic`)
- `pending_at_close`: NDJSON drain 后是否仍有 pending 行 (理论应为 0)
- `dead_letter_count`: 进 `.kb_dead_letter.ndjson` 的条数

`critic_robustness` 段子字段: 新增 `kb_writes_summary` (引用上面段
的 kb_writes_from_critic.length / verdicts breakdown)。

### 4.5 新段 (放在 telemetry 内) — `lane_timeline`

数据源: `leases` 表的事件序列 (acquire / release ts) + lane_capacity
表。

字段 (概念层):

- `lane`: lane 名
- `capacity`: 容量
- `holders_timeline[]`: list of (ts, holder_count) sample (每秒/每 30 秒
  一个 sample, 持续整个 session)
- `peak_holders`: 峰值
- `total_acquire_count`: 整 session 总申请次数
- `total_lane_full_count`: 因容量满被拒次数 (research_lane 主用)

### 4.6 新段 — `phase_breakdown` (放在 attribution 内)

按 phase 拆解 cumulative_gain 来源:

- `prelude`: 0 (定义上不产生 gain)
- `explore`: explore phase 内 KEEP 的 cumulative_gain delta
- `kernel`: kernel phase 内 KEEP 的 cumulative_gain delta
- `sweep`: 通常为 0 (sweep 不产生新 KEEP, 仅验证 / 标 unstable)

每段内再细分:

- explore 段: 按 specialist domain 拆 (kernel_specialist 贡献 X% /
  framework_specialist Y% / ...)
- kernel 段: 按 kernel_id 拆

## 5. v0.6 → v0.8 段落映射 (兼容表)

| v0.6 段 | v0.8 行为 |
|---|---|
| session | + schema_version=2, + phase_history, + cortex_session_id |
| workload | 无变化 |
| baseline | 无变化 |
| final | 无变化 (current_best) |
| phase_timeline | 重新定义为 phase 边界 timeline; 同时保留 action_timeline 顶层字段供 v1 reader |
| capability_summary | 删 backends/params/validate_stack 行 + 新增 explore/specialist 行 + 旧名行作 alias |
| geak_invocations | 无变化 |
| oob_invocations | 无变化 |
| kernel_lifecycle | 无变化 |
| param_search | rename → `explore_search`; 同时保留 `param_search` alias 字段 (= 同数据) |
| sweep | 无变化 |
| critic_robustness | + kb_writes_summary 子段 |
| telemetry | + lane_timeline 子段 |
| attribution | + phase_breakdown 子段 |
| warnings | + v0.8 迁移警告条目 |
| source_files | + runtime/cortex/* paths |
| **NEW** specialist_runs | 新增 (§4.3) |
| **NEW** kb_provenance | 新增 (§4.4) |

## 6. 接口/契约

`session_breakdown.json` 的 envelope:

```
{
  "schema_version": "hyperloom.session_breakdown.v2",
  "exporter_version": "<v0.8 build hash>",
  "generated_at_utc": "...",
  "session": {...},
  "workload": {...},
  ...
  "specialist_runs": [...],
  "kb_provenance": {...},
  "action_timeline": [...]   // v1 兼容
}
```

读 / 写时机:

- **写**: CLOSE phase 内 + 任何 `session_breakdown` action 触发时。
- **读**: claw-stats-service, hyperloom-results-service, 离线工具,
  v0.8 内部"resume diagnostics" 命令。

## 7. 实施步骤

1. **schema spec 锁定**: 把 §4 的所有段 / 字段写入一份独立 contract
   (建议在 `breakdown/SCHEMA_v2.md`)。
2. **版本号 bump**: schema_version 字符串 v2; exporter 写出时同时
   写入 v1 alias 字段 (方便操作员观察兼容)。
3. **新段 collector**: specialist_runs / kb_provenance / lane_timeline
   各设计独立 collector 概念, 数据从 §4 标注的来源派生。
4. **dashboard 升级路径**: claw-stats-service 在切到 v2 reader 前先
   验证 v1 reader 在 v2 文件上不报错; 切换 reader 是单独 PR。
5. **CLI flag**: `--breakdown-include-transcripts=true|false` 控制
   specialist transcript 是否内嵌 (默认 false, 仅记 path)。

## 8. 边界条件 / 失败模式

| 场景 | 行为 |
|---|---|
| EXPLORE 阶段未派任何 specialist | specialist_runs = [], capability_summary.specialist 行字段全 0 |
| Cortex T4 commit 失败 | kb_provenance.pending_at_close > 0; warnings 段加 entry; breakdown 仍写出 |
| schema_version 字段被 v1 reader 当字符串读 | 不影响 (v1 reader 一般不校验 schema_version) |
| breakdown 文件超大 (transcripts 太多) | 默认不内嵌 transcripts, 仅 path; flag 可强制内嵌 |
| 老 reader 拿到 capability_summary 中没有 backends 行的 file | 通过 alias 字段读到 explore 数据 |

## 9. 验收标准

- [ ] 一次 fresh v0.8 session, breakdown 文件内有
      `schema_version=v2` + 全部 §4 新段。
- [ ] v0.6 的 reader 加载 v0.8 breakdown 不报错; 关键字段 (workload,
      baseline, final, capability_summary 旧行) 读到合理值。
- [ ] specialist_runs / kb_provenance / lane_timeline 与 SharedState /
      cortex audit log / leases 表 cross-check 一致 (Inv-12.2)。
- [ ] phase_breakdown 的 gain 加和 ≈ cumulative_gain (允许 < 0.5% 浮
      点误差)。
- [ ] §3.1 §6 三个观测点 (core_state_writes, kb_provenance.generator,
      lane_timeline) 在 breakdown 中均可见。

## 10. 依赖与影响面

- **上游**: §3.2 (phase_history), §3.4 (explore_search), §3.5
  (specialist_rounds), §3.6 (cortex_session_id, kb audit), §3.7
  (lane 状态), §3.10 (新增 SharedState 字段)。
- **下游**:
  - claw-stats-service / hyperloom-results-service 的 reader 升级
    路径 (操作员 RFC, 不阻塞 v0.8 内部)。
  - §3.13 milestone M1 / M2 / M3 / M5 / M6 各自加自己的 breakdown
    段输出。

## 11. 哲学回引

本节是**主轴 A / B / C** 的可观测落地: phase 边界可见 (主轴 A), KB
写入可反查 (主轴 B), specialist 与 deterministic 执行体可分别统计
(主轴 C)。**Inv-12.1 向后兼容** 保证操作员的现有 dashboard 不需要
立刻升级, 减少推全集成成本。
