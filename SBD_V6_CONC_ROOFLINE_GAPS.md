# conc_sweep 与 roofline 的 V6 漏记清单

commit `da54496de`。实现用清单。

前提说明：V6 的 `conc_sweep` 事件**今天还是事后投影**（读 `conc_sweep_summary` + `state.last_conc_sweep`），没有专门的实时 recorder。`roofline` 事件**是实时录制的**，但 `finish_event` 只组装 spool 里的 action 行，不读 snapshot 和 per-kernel 表。

---

## A. conc_sweep

### A.1 完全缺失的整块

**`roofline_ceiling`**（每 CONC 的理论峰值与 MBU）——产生点 `orchestrator/kernel/conc_sweep.py:_build_roofline_ceiling`

```
schema_version, source, gpu_type, precision, tp, isl, osl,
model_meta.*,
rows[]: conc, t_mem_tok_s, t_cmp_tok_s, t_peak_tok_s, bound_kind,
        mbu_baseline_pct, mbu_optimized_pct
```

### A.2 逐字段缺失

**顶层**

| 字段 | 含义 |
|------|------|
| `schema_version` | 报告版本（`"1.0"` 或恢复态 `"recovered-v1"`） |
| `was_skipped` | 从未启动 variant 的 decline 标记 |
| `session_id` | 会话 ID |
| `isl` / `osl` / `tp` | 负载形状与并行度 |
| `benchmark_mode` | 绘图轴（synthetic vs agentx） |
| `source` / `original_report_path` | 恢复元数据 |
| `status` 的 `in_progress` 态 | 增量 flush 的中间态，V6 的 `ext.result.status` 没有这个值 |

**arm**

`baseline.extra_envs` 与 `optimized.extra_envs` 都缺（V6 只有 `extra_server_args`）。

**points[]**

| 字段 | 含义 |
|------|------|
| `arm` | baseline / optimized（V6 靠容器隐含，扁平化后丢） |
| `request_throughput` | 请求吞吐 |
| `total_token_throughput` | 总 token 吞吐（agentx 轴的主指标） |
| `input_throughput` | 输入吞吐 |
| `intvty_p90` | agentx 交互性指标 |
| `tpot_p90_ms` | TPOT p90 |
| `duration_seconds` | variant 运行时长 |
| `completed_requests` | 完成请求数 |
| `killed_overtime` | 是否超时被杀 |
| `estimated_output_throughput` | 估计吞吐 |
| `workspace` | variant workspace |
| `raw_result_path` | 恢复路径专用 |

**comparison[]**

| 字段 | 含义 |
|------|------|
| `delta_pct` | 百分比增益（V6 只有 `speedup`） |
| `baseline_status` / `optimized_status` | 两侧点状态（V6 部分合成进了 `error` 字符串） |

注意 V6 把 legacy 的 `baseline_tput` / `optimized_tput` 改名成了 `baseline_throughput` / `optimized_throughput`。

**summary**

| 字段 | 含义 |
|------|------|
| `successful_pairs` / `failed_pairs` | 成功与失败配对数 |
| `median_speedup` / `mean_speedup` | 中位与均值加速比 |

### A.3 过程中已知但从未落盘

这些在 `run_conc_sweep` 执行时都是确定的，但 summary JSON 和 V6 都没写：

| 事实 | 何处可知 | 价值 |
|------|----------|------|
| sweep `task_id`（`conc_sweep_{utc}`） | `:1321` | 关联 `runs/conc_sweep/` workspace |
| `variant_timeout_sec` / `num_prompts_factor` | 函数参数 | 解释 budget 与 timeout 行为 |
| `resolved_model` / `resolved_gpu` | materialize 后 | 复现环境 |
| `base_yaml_path` | materialize 后 | 复现 Magpie 配置 |
| 执行策略分支（single-server vs Option B） | `_sweep_one_arm_single_server` | 解释 boot / reuse 失败 |
| lifecycle 资格（`lc_eligible` / `lc_reason` / `port` / `framework`） | `resolve_lifecycle_params` | 解释为何 fallback |
| **boot-retry-descend 的每次尝试 CONC 与失败列表** | `failed_boots` / `boot_idx` 循环 | 诊断 OOM 与容量 |
| 每 variant 的 `NUM_PROMPTS`（conc × factor） | `_build_arm_grid` | 复现负载 |
| 每 rung 的 `_granted_cap_sec` | budget gate 前 | 解释 budget skip |
| `deadline` epoch / `_session_soft_dl` | budget 计算 | 解释 session_deadline_reserve |
| **每 variant 的起止时间戳** | `run_grid` 返回 | 现在只有整 sweep 的 `elapsed_sec` |
| arm 顺序（optimized → baseline） | `arms_order` | 解释跨 arm budget 消耗 |
| Ray `serving_lease` 是否持有 | `maybe_serving_lease` | GPU 租约诊断 |
| AgentX budget 自动抬高（9000 → raised） | `:1370-1385` | 解释非预期 budget |
| 每次 budget gate 的 `_arm_remaining` / `_reuse_remaining` | 各检查点 | 精确 budget 消耗轨迹 |

### A.4 V6 里的空占位

`ext.trigger.*`、`ext.input_anchor.*`、`ext.plan.grid_source`、`ext.failure.failed_task_id` 都是恒 null 的占位，没有产出方。要么接上，要么删掉。

---

## B. roofline

### B.1 整块缺失一：per-kernel roofline 表

来源：`reports/kernel_roofline.json`

写入方两条路：
- TraceLens 路由——`agents/kernel/tools/tracelens_analysis.py` 末尾 `atomic_write_json`，路径由 `kernel_roofline_path_for_run` 解析为 `{session}/reports/kernel_roofline.json`
- bypass 路由——`_bypass_report.build_kernel_roofline`，`source: "bypass"`

V6 的 roofline 事件**只有一个 `kernel_roofline_path` 路径**，正文不读。

**顶层字段**：`schema_version`、`source`、`trace_input`、`trace_input_type`、`roofline_json_path`（外部 rocprof JSON）

**`kernels[]` 每行**：

| 字段 | V6 现状 |
|------|---------|
| `kernel_id` | **缺**（hot_kernels 摘要里没有 id） |
| `name` | 部分（`hot_kernels.top[].name`，只有 15 条） |
| `source_file` | **缺** |
| `kernel_category` | 部分（V6 叫 `category`） |
| `bound_type` | **缺** |
| `arithmetic_intensity` | **缺** |
| `flops_per_byte` | **缺** |
| `efficiency_percent` | **缺** |
| `gpu_pct` | 部分 |
| `call_count` | 部分（V6 叫 `count`） |
| `duration_us` | 部分（V6 是 `gpu_time_us`，单位不同） |
| `reusable_native_kernel` | **缺**（只有一个 id 集合） |
| `rocprof_roofline` | **缺** |
| `bottleneck` | **缺** |
| `suggestion` | **缺** |
| `roofline_name` | **缺** |
| `compute_utilization_pct` | **缺** |
| `bandwidth_utilization_pct` | **缺** |
| `recommended_actions` | **缺** |
| `roofline_source` | **缺**（analytical vs placeholder） |
| `roofline_attainment_pct` / `roofline_measured` | **缺**（bypass 专有） |

### B.2 整块缺失二：snapshot 全字段

来源：`state.roofline_snapshots`，写入方 `SharedState._append_roofline_snapshot_history`，由 `record_trace_analyze` 在每次成功分析后调用。

**关键：这个 append 发生在 `finish_succeeded` 之前**（`roofline.py:1370-1426`），数据在进程内已经就位，recorder 完全够得着，只是 `assemble_roofline_ext` 没读。

V6 现在只有 `outcome.snapshot_id`。完整 snapshot 行：

| 字段 | 含义 |
|------|------|
| `ts` / `framework` | 时刻与框架 |
| `theoretical_peak_tok_per_sec` | 主天花板 |
| `roofline_mem_ceiling_tok_per_sec` | 内存侧天花板 |
| `roofline_cmp_ceiling_tok_per_sec` | 计算侧天花板 |
| `roofline_bound_kind` | memory / compute / unknown |
| `throughput_unit` | tok/s 或 img/s |
| `achieved_tok_per_sec` | 实测吞吐 |
| `within_roofline_pct` | 饱和度（cap 100） |
| `gap_to_roofline_pct` | 距天花板差距 |
| `within_roofline_pct_uncapped` | 未 cap 比值 |
| `roofline_ceiling_exceeded` | 超天花板标志 |
| `compute_pct` / `idle_pct` / `comm_pct` | Executive Summary 三分 |
| `top_bottleneck` | 顶瓶颈类别 |
| `top_kernel.{name, gpu_pct, efficiency_pct, bound_type}` | 最热 kernel 摘要 |
| `e2e_mean_ms` / `roofline_ideal_ms` | 扩散模型延迟对 |
| `macro_cycle` | 宏周期 |
| `roofline_provenance.*` | dtype / 公式 / peak 来源 |
| `perfmodel_breakdown.ops[].{name, flops, bytes_moved, ai, time_s, bound, pct_time}` | 逐算子分解 |

**snapshot 对比族**（`mode`、`ceilings_comparable`、`baseline` / `latest` / `delta`）也全缺。`delta` 含 `compute_pct`、`idle_pct`、`comm_pct`、`top_kernel_efficiency_pct`、`within_roofline_pct`、`gap_to_roofline_pct` 的变化量。

另外 `state.baseline_roofline_ceiling`（`record_baseline_roofline_ceiling`，baseline promote 后调用）也没进 V6，它是同族结构外加 `ceiling_arm: "baseline"`。

### B.3 整块缺失三：progress 轨迹

来源：`collect_roofline_progress` 从 snapshots + `optimization_stack` 算。这是跨事件的 session 级量，应在 close 时快照。

| 字段 | 含义 |
|------|------|
| `ceiling_kind` | throughput / latency / none |
| `ceiling_tok_per_sec` | 参考天花板线 |
| `target_tok_per_sec` | ceiling × 0.70 |
| `ceiling_ratio_target` | 0.70（`DEFAULT_ROOFLINE_TARGET_RATIO`） |
| `ceiling_available` | 是否有 tok/s 天花板 |
| `latency_ceiling_ms` / `achieved_latency_ms` / `latency_ceiling_available` / `current_best_pct_of_latency_ceiling` | 扩散延迟域 |
| **`trajectory[]`** | baseline 加每次 KEEP 的步进曲线，每点 `{ts, tput, label, action, gain_pct, flags, extra_envs}` |
| `baseline_tput` / `current_best_tput` | 头线数字 |
| `cumulative_gain_pct` | 累计增益 |
| `current_best_pct_of_ceiling` / `current_best_pct_of_target` | 相对天花板与目标 |
| `roofline_failure_streak` | 连续 roofline 失败次数 |
| `snapshots[]` | 归一化 snapshot 历史 |
| `snapshot_top_bottleneck` / `snapshot_within_roofline_pct` / `snapshot_gap_to_roofline_pct` | latest 快照头线 |

---

## C. 实现顺序建议

roofline 的三块难度差别很大：

1. **snapshot 全字段**——最容易。数据在 `finish_event` 时已在 `SharedState` 里，只需在 recorder 的 `finish_succeeded` / `_close` 路径注入。
2. **per-kernel 表**——中等。需要在 trace_analyze 完成时读 `kernel_roofline.json` 正文录成行。跟 `SBD_V6_KERNEL_LIFECYCLE_MERGE.md` §3 的 `discovered_kernels` 是同一批数据，应该一起做，别录两份。
3. **progress 轨迹**——需要 close 级快照，依赖 §2.1 那个跨阶段 adoption 账本先定下来（`trajectory[]` 本质就是采纳序列加吞吐）。

conc_sweep 需要先建实时 recorder，工作量比 roofline 大。
