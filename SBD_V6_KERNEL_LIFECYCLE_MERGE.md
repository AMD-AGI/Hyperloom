# `kernel_lifecycle` 合并进 kernel 事件

commit `da54496de`。待讨论文档。

问题：legacy `kernel_lifecycle` 能不能并进 `timeline[kernel].ext.forge`？

结论：**能，但不是 drop-in**。需要新增 3 个数组、1 个聚合层，并且改 forge 的录制接线。

---

## 1. 结构错配：per-kernel vs per-run

| 维度 | `kernel_lifecycle` | `ext.forge` |
|------|-------------------|-------------|
| 主键 | `kernel_id` | `run_id`（每次 lane run） |
| 组织 | 5 个漏斗阶段数组，每 kernel 一行携带该 kernel 全部历史摘要 | 4 个 lane 数组 + rebench ledger + trace_analyze runs |
| 时间范围 | **session 级**（导出时聚合全 session） | **单次 KERNEL visit**（一个 macro_cycle） |

好消息：`kernel_rewrites[]` 每行**有 `kernel_id`**（`kernel_event.py:767`），所以 per-kernel 视图可以 GROUP BY 重建。测试里也确认同一 kernel 可以对应多行。

坏消息有两个，见 §2。

---

## 2. 两个阻塞问题

### 2.1 `record_kernel_rewrite()` 生产代码根本没调用

全库 grep 只有 `tests/` 和 `kernel_event.py` 里的定义本身。orchestrator 未接线。

forge 的多 attempt 历史现在写在**另一条路**：`instrument.record_kernel_backend_result`（`instrument.py:3098-3239`）每 attempt 一行写 `kernel_backend_result` fragment。这条路是活的，但**不进 V6 kernel 事件**。

GEAK 的 rebench 已经接进 timeline（`writeback.py:3666-3725`），forge 的 rewrite / rebench 没有同样接线。

### 2.2 设计上只保留 adopted 形态，多 attempt 会丢

`V6KernelRewriteRun` 的行模型带 `adopted_backend` + `backends_tried[]`，语义是**一次 rewrite 战役的汇总**，不是 attempt 枚举。`SBD_V6_DEDUP.md:74` 也明确写了只保留 adopted 形态。

后果：`kernel_lifecycle.optimized[].attempts_summary[]` 那份逐 attempt 历史（`attempt_id` / `backend` / `decision` / `micro_speedup` / `ts`）**无法重建**。

**必须选一个**：

- A. 每个 attempt 调一次 `record_kernel_rewrite`，`run_id = attempt_id`
- B. 保留汇总行，另加 `forge.by_kernel[kid].attempts_summary[]`，在 `finish_event` 时从 `kernel_backend_result` 行聚合

我倾向 A，因为 lane 行的语义本来就该是「一次运行」。

---

## 3. profiling 富字段无处可去

`detected[]` 里最有价值的那批 per-kernel profiling 数据，在 kernel 和 roofline 事件里**都不存在**：

| 字段 | V6 事件里有吗 | 原始产生点 |
|------|--------------|-----------|
| `duration_us` | 否（只有截断摘要里的 `gpu_time_us`，且常缺） | TraceLens → `kernel_candidates.json`；benchmark → `kernel_summary.time_ms` |
| `call_count` | 否 | 同上 |
| `bandwidth_util_pct` | 否 | TraceLens hot kernel → `_build_hot_kernel_summaries` |
| `compute_util_pct` | 否 | 同上 |
| `kernel_category` | 部分（`hot_kernels.top[].category`，只有 top-15） | 同上 |
| `arithmetic_intensity` | 否 | trace_analyze result + `kernel_roofline.json` |
| `bottleneck` | 否（roofline 用 `bound_type`） | candidates / benchmark |
| `recommended_actions` / `recommended_backends` | 否 | `_build_hot_kernel_summaries` |
| `optimization_notes` | 否 | candidates |
| `source_file` | 否 | candidates |

V6 事件里的 `trace_analyze_runs[].analysis.hot_kernels` 只是 `{count, top: [{name, op_name, category, gpu_time_us, gpu_pct, count}]}`，**最多 15 条且没有 kernel_id**，跟 `detected[]` 对不上。

完整数据在三个地方：`kernel_candidates.json`、`state.hot_kernels_top15`、`reports/kernel_roofline.json`。都是 `record_trace_analyze` 时刻可得的。

---

## 4. 逐字段合并表

### `detected[]`

| legacy | 目标 |
|--------|------|
| `kernel_id` / `name` / `gpu_pct` | 新增 `forge.discovered_kernels[]` |
| `duration_us` / `call_count` / `bandwidth_util_pct` / `compute_util_pct` | 新增，同上 |
| `kernel_category` / `bottleneck` / `arithmetic_intensity` | 新增，同上 |
| `reusable_native_kernel` | 新增（现在只能从 `reusable_native_kernel_ids` 成员检查推） |
| `source_file` / `optimization_notes` | 新增 |
| `recommended_actions` / `recommended_backends` | 新增 `forge.recommended_kernels[]`，或作为 discovered 的子集标记 |
| `detected_from_task` / `benchmark_report_path` | 新增 `discovered_kernels[].provenance` |
| `selected_for_optimization` | 新增 `discovered_kernels[].selected` |
| `geak{attempts, best_speedup, decision, last_status}` | `geak.attempts.kernels[]` 按 kernel_id 聚合 |
| `forge{...}` | `forge.lanes.kernel_rewrites[]` 按 kernel_id 聚合 |
| `integrate_gain_pct` | `kernel_rewrites[].e2e.e2e_gain_pct` 或 `rebench_ledger[].delta_pct`（**测量点不同，注意别混**） |
| `adopted_by` | 新增 `forge.by_kernel[kid].adopted_by`，finish 时 join |
| `final_decision` | 新增 `forge.by_kernel[kid].final_decision`，finish 时算 |

### `recommended[]`

整行 → 新增 `forge.recommended_kernels[]`。等价于 `hot_kernels_top15` 的投影，在 trace_analyze 完成时录一次快照。

### `optimized[]`

| legacy | 目标 |
|--------|------|
| `kernel_id` | `kernel_rewrites[].kernel_id` |
| `backend` | `adopted_backend` |
| `best_micro_speedup` / `best_artifact_path` | `speedup` / `artifact_path` |
| `last_decision` | `micro_decision` / `e2e.decision` |
| `total_attempts` / `successful_attempts` | **新增或聚合**，单行读不出来 |
| `attempts_summary[]` | **新增**，见 §2.2 |

### `adopted[]`

| legacy | 目标 |
|--------|------|
| `kernel_id` | `outcome.adopted[].ref`（形状不匹配，adopted 行没有 kernel_id 字段） |
| `patch_path` / `target_file` | `kernel_rewrites[].e2e.*` |
| `e2e_gain_pct` / `validated` | `outcome.adopted[].gain_pct` / `e2e.validated` |
| `extra_server_args` | **新增** `outcome.adopted[].extra_server_args` |
| `last_status` / `adopted_at` / `attempt_count` / `basis` / `alignment_status` | **新增** `forge.integrations[]` |

### `rejected[]`

整行 → **新增** `forge.rejected_kernels[]`。V6 的 outcome 里没有 rejected 列表，只有 `pending_review[]`。

---

## 5. 建议的合并后形状

```
ext:
  entry:               不变
  forge:
    engaged
    reprofile:         不变
    trace_analyze_runs: 不变（hot_kernels 保持摘要，避免膨胀）
    discovered_kernels: 新增  ← detected[] 的 profiling
    recommended_kernels: 新增 ← recommended[]
    lanes:
      kernel_rewrites:  改为每 attempt 一行
      fusion_runs / gemm_tuning_runs / collective_runs: 不变
    rebench_ledger:    不变
    integrations:      新增  ← adopted[] 全字段
    rejected_kernels:  新增  ← rejected[]
    by_kernel:         新增  ← finish_event 时聚合出漏斗终态
  geak:                不变
  outcome:             adopted 行补 extra_server_args
```

`by_kernel[kid]` 的内容：`lane_summaries.{forge, geak}`、`attempts_summary[]`、`integrate.{gain_pct, decision, patch_path}`、`final_decision`、`adopted_by`。

---

## 6. 需要的接线改动

1. trace_analyze 完成时录 `discovered_kernels`（从 candidates 或 `hot_kernels_top15`）
2. 每次 backend attempt 调一次 `record_kernel_rewrite`
3. integrate / reject 时录 `forge.integrations` / `forge.rejected_kernels`
4. `assemble_kernel_ext` 里从 lane 行 + rebench + integrate 行算出 `by_kernel.*` 和 `final_decision`

---

## 7. 一个跨 visit 的问题

`kernel_lifecycle` 是 **session 级**的，一个 kernel 的历史可能跨多个 macro_cycle 的 KERNEL visit。而 kernel 事件是 **visit 级**的。

所以合并后，「这个 kernel 全程经历了什么」这个视图需要跨多个 kernel 事件 fold。两个选择：

- A. 消费方自己跨事件聚合（读的人麻烦，但没有第二份真相）
- B. close 时录一份 session 级的 `by_kernel` 快照（读的人方便，但要保证与事件流一致）

按你定的全快照方向应该是 B，但这块想听你的意见。
