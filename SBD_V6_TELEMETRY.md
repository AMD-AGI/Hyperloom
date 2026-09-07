# telemetry 现状诊断

commit `da54496de`。待处理文档。

你的印象是对的：telemetry 三个子树在典型单节点 session 里**大多为空或语义失真**。下面是每块的根因。

---

## 结论速览

| 子树 | 状态 | 单节点 session 实际表现 |
|------|------|------------------------|
| artifact 路径扫描 | **正常** | 有 benchmark 就有路径 |
| `gpu_monitor_aggregate` | **架构上就是空的** | 恒为 `{}` |
| `lane_timeline` | **语义错误** | `capacity` 有值，`live_holders` 恒为 0 |
| `orchestration_context` | **条件可用** | 非 conversational 后端全零 |

---

## 1. `gpu_monitor_aggregate` — 单节点永远是空的

### 根因

全库**只有一条**写入 `benchmark_report.json` 的 `gpu_monitor` 键的路径：`harvest_mn_gpu_metrics`。而它第一件事就是：

```
# Multi-node only: single-node uses Magpie's own client-side GPUMonitor,
# so never touch its result path.
from ._multi_node_env import is_multi_node
if not is_multi_node():
    return out
```

`benchmark_result.py:552-558`

注释声称单节点由 Magpie 自己的 client-side GPUMonitor 负责，但：

- Magpie 官方 benchmark config 文档里**没有 `gpu_monitor` 字段**
- 仓库内没有 Magpie 源码可以核实
- `bypass_report.py:107` 反而显式写了 `"gpu_monitor": None`

### 讽刺的地方

baseline 和 grid runner **确实**通过 `GPU_METRICS_CSV` env 把 `gpu_metrics.csv` 写到 workspace（`baseline.py:4829`、`_grid_runner.py:1174`），`harvest_leaked_artifacts` **也确实**把 csv 复制出来了。

但**没有任何代码**把这个 csv 解析进 `benchmark_report.json.gpu_monitor`，collector 也**只读 report 不读 csv**。数据就在磁盘上躺着，没人管。

### 多节点的条件

多节点下 `harvest_mn_gpu_metrics` 会跑，但要凑齐 6 个条件才有样本：

1. `state.nodes >= 2` 或 `$INFERENCE_OPTIMIZER_NODES >= 2`
2. `$HYPERLOOM_MN_SERVER_LOG_DIR` 或默认路径能展开成绝对路径
3. 该目录存在且含 `gpu_metrics_<host>.csv`
4. `destination/benchmark_report.json` 已存在（无 report 就跳过注入）
5. CSV 行时间戳落在 `[subprocess_started - 2s, now + 2s]` 窗口内
6. rocm-smi 的列名能被 `_row_to_gpu_sample` 识别

### git 历史

MN 注入和这条单节点 gate 注释是 Jul 2026 的 `c00f0f27d`（infera 多节点移植）引入的。**从来没有 commit 为单节点实现 csv → report 注入**，V6 的实时录制也没落地。

### 要改什么

最省事的是去掉 `is_multi_node()` gate，让同一套逻辑也读本地 workspace 的 csv——`_row_to_gpu_sample` 已经现成。彻底一点是在 `extract_benchmark_measurement` 里直接录 GPU 采样 fragment。

---

## 2. `lane_timeline` — 导出时刻查询必然为 0

### 根因

`live_holders` 查的是 `leases WHERE expires_at > datetime('now')`。代码注释自己也承认了这是 point-in-time 快照（`telemetry.py:252-253`）。

问题在于：breakdown 是在 **CLOSE 阶段**导出的。那时所有 task 已结束、lease 已全部 release 或过期，所以 `live_holders` **必然全为 0**，看起来像「从未占用过任何 lane」。

### `lease_expired_count` 也低估

- 正常 `release()` 只 DELETE，**不 emit 事件**（`resource_lock.py:368-374`）
- 只有 TTL 强制过期才 emit（`:235-257` 和 `:376-409`）

所以这个计数只统计超时过期，大量正常周转不计入。

另外 `db_maintenance.prune_events` 只保留最近 5000 条事件，超长 session 的早期 lease_expired 可能已经被 prune 掉。

### 测试为什么没发现

`test_resource_lanes.py:636-671` 是在**持有 lease 期间**调 collector 的，断言 `live_holders == 3`。合成场景，完全不模拟 CLOSE 后导出。

### 要改什么

录 acquire / release / expire 三种事件行，导出时从事件流算 peak occupancy，而不是查 `leases` 表现状。

---

## 3. `orchestration_context` — 只在 conversational 后端下有值

### `seed_prompts` / `delta_prompts`

`_count_prompt_mode` 是接线了的，但被 gate：

```
if agent_name == "orchestration":
    if self._orchestration_conversational():
        self._count_prompt_mode("seed" if push_full else "delta")
```

`conversation.py:470-483`

非 conversational 后端**从不调用**，seed/delta 永远是 0。需要确认：production 用的是 conversational 后端吗？如果不是，全零是预期还是 bug。

另外 `_count_prompt_mode` 自己不调 `save()`，靠 tick 结束的常规 save 路径落盘。

### compaction 一族

checkpoint 事件要求四个条件同时满足：`_checkpoint_enabled`（非 `INFERENCE_OPTIMIZER_DISABLE_ORCH_CHECKPOINT=1`）、`_orchestration_conversational()`、backend.conversational、`_orchestration_seeded`（至少完成一次 SEED turn）。

历史上 #1095（`4c2d17771`）修了「每 tick compaction storm」的 token 单位 bug，同时才加了 `_collect_orchestration_context`。修复前 compaction 数虚高，修复后应该反映真实节奏。

### `tick_count`

这个是可靠的，coordinator 每 tick 递增并 save。

---

## 4. 测试覆盖为什么全都没发现

| 子树 | 测试 | 为什么发现不了 |
|------|------|---------------|
| `gpu_monitor_aggregate` | **无 collector 测试** | — |
| MN harvest 注入 | `test_benchmark_result_branches_unit.py:362-488` | 合成 csv + monkeypatch `is_multi_node`，只测 harvest 函数不测 breakdown |
| `lane_timeline` | `test_resource_lanes.py:636-711` | 合成 DB，live lease 期间查询，不覆盖 CLOSE 后 |
| `orchestration_context` | `test_breakdown_exporter_unit.py:423-446` | 手写 DB events + 手填 state |
| prompt mode 接线 | `test_codex_conversation_continuity.py:162-177` | 只断言内存 census，不断言 state.json 持久化 |
| reporter | `test_reporters_smoke.py:86-89` | hardcoded fixture dict |

**没有任何测试**从真实 session 目录断言 `gpu_monitor_aggregate.samples > 0`，或者验证 CLOSE 后 `live_holders` 的语义。

---

## 5. 处理建议

三块的性质不一样，建议分开处理：

**GPU 采样**——这是真正有价值的数据，csv 已经在磁盘上了，只差解析注入。优先级最高，改动最小。

**lane 占用**——现在的字段是误导性的，应该改成事件流录制。如果短期不改，至少该把 `live_holders` 从 breakdown 里去掉，避免读的人以为「lane 从未被占用」。

**orchestration context**——先确认 production 后端类型，再决定是补非 conversational 分支的 census，还是在文档里明确只支持 conversational 模式。

对应到 V6 的字段决定：telemetry 这一段在补录之前**不应该按现有形状迁移**，否则是把三个坏字段搬进新结构。
