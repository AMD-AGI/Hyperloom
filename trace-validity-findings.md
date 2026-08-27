# Trace 有效性判定调研

> 目标:界定「KERNEL 阶段无法优化」的责任归属。回答"什么样的 trace 文件才算可用、无异常",并给出可落地的判据。
>
> 调研日期:2026-08-27 · 代码基线:`Hyperloom@30a46bde9`

---

## 0. 摘要

1. **现有的文件级检查(存在 / 可读 / 有事件)在真实故障面前 0 命中。** 实测的 8 个故障 rank 全部是"文件完好、CPU 侧完美、有 GPU kernel 事件"的。所有故障都是**语义级**的。
2. 实测到 **5 类**故障,其中两类(图内 kernel 记录丢失、profiler 开窗偏斜 barrier)覆盖了绝大多数"KERNEL 阶段归零"的案例。
3. **6/6 次 TraceLens 运行都带 barrier;3/3 个 session 的第一次 roofline 都被 `low_gpu_compute_pct` 门槛毙掉,第二次都通过。** 差别只在 barrier 大小恰好落在 10% 门槛哪一边。
4. 判定 trace 是否可用的核心是三组指标:**GPU 记录完整性**、**时间轴独占性**、**跨 rank 一致性**。
5. 现有代码里已有的 `_graph_under_recorded` 检查**语义用反了**——它触发后的动作是"放行",而正确语义是"这份 trace 不可用"。

---

## 1. 数据基础

| 来源 | 内容 |
|---|---|
| 7 个 session 的全部 kernel-agent run | GLM-5.2-MXFP4 / Qwen3.8-2.4T-A95B-MXFP4 ×2 / Kimi-K3 ×2 / DeepSeek-V4-Pro / MiniMax-M3-MXFP4 |
| 完整流式解析 | Qwen3.8 事故的全部 8 个 rank trace(`TOP-Model-Analysis/0826/trace-bak/`,19–25 MB/rank) |
| 事件类别普查 | GLM(健康组)、MiniMax(失败组)的 rank trace |
| 已有复盘 | `TOP-Model-Analysis/0826/kimi-k3-kernel-phase.md`、`qwen3.8-kernel-phase.md` |

环境:sglang + TP8 + MI355X (gfx950) + ROCm 7.2.0。**vLLM / 单卡 / xDiT 无样本。**

---

## 2. 否定结论:文件级检查无效

Qwen3.8 run1 的 8 个 rank(其中 3 个严重损坏)在文件层面的表现:

| 检查项 | 8 个 rank 实测 | 能否区分好坏 |
|---|---|---|
| 文件存在 / 可解压 | 全部正常 | ✗ |
| `stream_errors`(截断 / 损坏) | **0 / 8** | ✗ |
| 事件总数 | 均为数十万~百万量级 | ✗ |
| `cpu_op` 数 | 81,664 / 81,920 / 81,920 …几乎完全一致 | ✗ |
| `user_annotation` 数 | 1,024,全 rank 一致 | ✗ |
| 有无 GPU kernel 事件 | 全部 > 0(最少 3,594) | ✗ |

**结论:「存在 / 可读 / 有内容」这三件事在真实故障中一件都不会亮。** 判据必须建立在 GPU 侧记录密度与时间结构上。

---

## 3. 实测到的五类故障

### F1 图内 kernel 记录丢失(graph replay under-recording)

Qwen3.8 run1,同一次 profile 的 8 个 rank:

| rank | kernel 数 | `graph_launch` | **有 kernel 的 replay** | busy% | 文件大小 |
|---|---:|---:|---:|---:|---:|
| TP-0 | 3,599 | 128 | **1** | 0.62% | 19.2 MB |
| TP-1 | 315,008 | 128 | 128 | 95.11% | 25.3 MB |
| TP-2 | 3,604 | 128 | **1** | 0.72% | 18.9 MB |
| TP-3 | 315,008 | 128 | 128 | 93.77% | 25.3 MB |
| TP-4 | 315,008 | 128 | 128 | 95.87% | 25.3 MB |
| TP-5 | 315,008 | 128 | 128 | 97.48% | 25.2 MB |
| TP-6 | 3,594 | 128 | **1** | 0.19% | 18.9 MB |
| TP-7 | 315,008 | 128 | 128 | 95.90% | 25.3 MB |

判据 `有 kernel 记录的 graph_launch / graph_launch` 的**区分度是完美的:0.008 vs 1.000,无灰区**。kernel 数差 87 倍。

同现象在 GLM 健康 session 的 run2 复现:TP-0 / TP-4 为 16.7–16.8 MB(饿死),另外 6 个 23.2 MB。

> **同一次 profile 内部 rank 好坏混杂是常态,不是例外。**

Kimi-K3 是同类故障的极端形态(`HYPERLOOM_TRACELENS_PATCH_STATUS: unavailable`,128 次 replay 只插桩 1 次),详见 `kimi-k3-kernel-phase.md`。

### F2 profiler 开窗偏斜产生的假 collective(barrier)

同一批 rank 的另一半问题:

| rank | 最大单 kernel | 时长 | **占窗口** |
|---|---|---:|---:|
| TP-5 | `cross_device_reduce_2stage` | 30,884.7 ms | **85.7%** |
| TP-7 | 同 | 19,643.8 ms | **78.9%** |
| TP-1 | 同 | 19,945.1 ms | **78.5%** |
| TP-4 | 同 | 16,843.0 ms | **76.6%** |
| TP-3 | 同 | 15,747.0 ms | **73.9%** |

- 各 rank **起始**时间戳散布 **30.88 s**,**结束**时间戳全部落在 **0.5 s** 内。
- TP-5 的 barrier 30,884.7 ms ≈ 它比最晚那个 rank 早开窗的时长。**barrier 时长 = 开窗偏斜,零通信含量。**

**这不是 Qwen 独有——6/6 次 tracelens 运行全部有 barrier**,包括被当作健康对照的 GLM:

| session | run | 最大单 collective 占窗口 | Compute% | 结果 |
|---|---|---:|---:|---|
| GLM | #1 | **97.21%** | 2.55% | **suppressed,0 候选** |
| GLM | #2 | 80.46% | 17.89% | 20 hot kernels |
| Qwen 0825 | #1 | 93.7% | 5.75% | **suppressed** |
| Qwen 0825 | #2 | 76.2% | 23.70% | 12 hot kernels |
| Qwen 0826 | #1 | — | 5.41% | **suppressed** |
| Qwen 0826 | #2 | 72.6% | 25.10% | 13 hot kernels |

> **3/3 个 session 的第一次 roofline 都被 10% compute 门槛毙掉,第二次都通过。** 唯一差别是 barrier 大小恰好落在门槛哪一边——是抽签。

**TraceLens 自己在散文里写对了**,只是没进结构化字段:

- GLM #1:*"This is a peer-arrival/rank-skew barrier at capture start, not collective bandwidth cost."*
- Qwen 0826 #2:*"that split is dominated by a single measurement artifact."*
- GLM #2:*"Excluding that stall, the remaining 821.67 ms of real decode work is **91.4% compute**."*

门槛读的是被污染的 17.89%,不是干净的 91.4%。

### F3 稳态切片零 GPU 事件

MiniMax-M3,`steady_state_chunk_empty`,连续两次。切出的 chunk 内容:

```
python_function  23,488   |  cpu_op      1,472  |  cuda_runtime  640
user_annotation      32   |  kernel          0
```

32 个 step 注解、640 次 launch、**0 个 kernel**。源 trace 全局也只有 2,745 kernel 对 983,398 个 `python_function`(**98.2%**)——`with_stack=true` 压垮 roctracer 的形态。

### F4 只有 capture sidecar

DeepSeek-V4-Pro:`torch_trace/` 下**只有** `capture_traces/`,288 个 `bs_*_rank*.json.gz`,顶层零个 workload trace。

这是**唯一被现有检查干净拦住**的一类(`trace_input_capture_only`)。但注意 `profile.py` 的 `capture_only_fallback` 先把它当作合法结果放行了,由下游 `tracelens_analysis.py` 才拒绝——责任边界模糊。

### F5 分析器崩溃 / 未产出 analysis.md

MiniMax 第三次运行,跑 22 分钟后 `TraceLens analysis.md was not produced`。

---

## 4. 「可用、无异常」的定义

每一条判据均从上述实测数据反推,非设计假设。

| # | 判据 | 坏值(实测) | 好值(实测) | 现状 |
|---|---|---|---|---|
| **A1** | 目录下有非 capture-sidecar 的 workload trace | DeepSeek 0/288 | — | ✅ 已有 |
| **A2** | 至少一个 rank 有 GPU kernel 事件 | — | 全部通过 | ✅ 已有(无区分力) |
| **B1** | **有 kernel 记录的 `graph_launch` / `graph_launch` ≈ 1.0** | **1/128 = 0.008** | **128/128 = 1.0** | ⚠️ 有实现,语义用反(见 §5.1) |
| **B2** | 带 GPU 记录的 step 注解 / step 注解 ≈ 1.0 | K3: **0/128** | — | ❌ 无 |
| **B3** | `kernel` 数 / `cuda_runtime` launch 数 不接近 0 | MiniMax chunk **0/640** | GLM ≈ 43× | ❌ 无 |
| **C1** | **单个 kernel 时长 / 窗口跨度 < ~30%** | **73.9–97.2%** | 干净窗口 < 1% | ❌ 无 |
| **C2** | 剔除 C1 异常事件后重算的 compute% 才用于门槛 | 污染后 2.55% | 干净后 **91.4%** | ❌ 无(门槛读污染值) |
| **D1** | 跨 rank kernel 数极差 < ~2× | **87×**(3,594 vs 315,008) | — | ❌ 无 |
| **D2** | 跨 rank 起始时间戳散布 ≪ 窗口跨度 | 起始散布 **30.88 s** / 窗口 21 s | 结束散布 0.5 s | ❌ 无 |
| **E1** | `python_function` 占比不压倒性 | MiniMax **98.2%** | — | ❌ 无 |
| **E2** | 选中的稳态 chunk 有非零 GPU 事件 | 0 | — | ✅ 已有 |

### 一句话定义

> 一份可用的 trace,是 **GPU 侧记录密度与 CPU 侧声明的工作量相匹配**(B 组)、**时间轴上没有单个事件独占窗口**(C 组)、**且各 rank 之间自洽**(D 组)的 trace。
>
> 文件层面的完整性(能打开、能解析、有事件)是必要条件,但在真实故障中从不失败。

---

## 5. 三个可直接改变结论的发现

### 5.1 B1 的检查已存在,但用反了

`src/hyperloom/agents/kernel/tools/_bypass_trace_reader.py:93` 的 `_graph_under_recorded()` 计算的正是 `graph_launches_with_kernels / graph_launch_count`,阈值 `_GRAPH_RECORDED_LAUNCH_COVERAGE_MAX = 0.5`。K3 的 0.008 会触发。

但触发后的动作(`tracelens_analysis.py:_evaluate_idle_gate_with_graph_guard`)是:**跳过 idle / compute 闸门,保留 `hot_kernels[]` 继续跑**——语义是"idle% 不可信,别拦"。

正确语义应为:**"这份 trace 丢了 99% 的 GPU 记录,不可用"**。当前实现在帮倒忙。

### 5.2 rank 选择策略决定成败,两条路由选反了

| 路由 | 实现 | 策略 |
|---|---|---|
| TraceLens | `tracelens_analysis.py:discover_trace_inputs` | **按文件大小降序** |
| bypass | `_bypass_trace_reader.py:resolve_trace_file` | **按最小 rank 索引** |

- GLM run2 → TraceLens 选中 TP-2(23.2 MB,记录完整)→ **成功,20 hot kernels**
- TP-0 在 GLM run2 和 Qwen run1 中**都是饿死的那个**;K3 的 bypass 分析的正是 TP-0

> **按文件大小降序是歪打正着的正确策略**(记录完整的 rank 文件更大:25.3 MB vs 18.9 MB),**按 rank 号选是系统性踩雷。**

### 5.3 C2 是投入产出比最高的一条

6/6 次运行都有 barrier,3 次因此归零。TraceLens 已把干净分母算出来放在散文里(91.4% / 91.5%),只是没进结构化字段。让门槛读"剔除单个独占事件后的 compute%",这三次归零全都不会发生。

---

## 6. 最小判据集(按性价比)

若只做三条:

1. **C1 单事件独占率** —— 一次流式扫描顺带算出,命中 100% 的 suppressed 案例
2. **B1 graph replay 记录率** —— 代码已有,只需改判定语义 + 让 TraceLens 路由也用
3. **D1 / D2 跨 rank 一致性** —— 只需读各 rank 的 `ts_min` / `ts_max` + kernel 计数,可抽样而非全量解析

B2 / B3 / E1 属同一族(GPU 记录密度),可合并为一个指标一起计算。

---

## 7. 现有实现清单与冗余

只算「trace 文件本身」相关的逻辑,散在 4 个模块:

| 层 | 位置 | 规模 |
|---|---|---|
| ① 分类(是否 capture sidecar) | `tools/_capture_shapes.py` | 109 行,**已统一**,4 个调用方共用 |
| ② 发现 + 选择 | `profile.py:404-531`、`tracelens_analysis.py:983-1193`、`_bypass_trace_reader.py:115-232`、`request_handlers.py:3211-3226` | **4 份独立实现** |
| ③ 可读性 + kernel 计数 | `profile.py:71-160`、`tracelens_analysis.py:938-1107`、`_bypass_trace_reader.py:234-475` | **3 份独立实现** |
| ④ 结构检查 | `profile.py:162-401` `_validate_trace_structure` | 1 份,240 行,**仅单机分支调用** |

### 7.1 三份「发现 + 选择」语义不一致

| | `profile.py` | `tracelens_analysis.py` | `_bypass_trace_reader.py` |
|---|---|---|---|
| 扫描模式 | 只 `*.trace.json.gz` | 5 种(含 `*.json`) | 5 种 `_TRACE_EXTS` |
| capture sidecar | 排除 | 降到 bucket 3(保留) | 排除,全是则回退全量 |
| split 产物 | 仅靠目录名排除 | 目录名 + 文件名正则 + `trace_annotation_iteration_N` | **不认识** |
| `TP-*-DECODE` | 不认识 | bucket 2 | 不认识 |
| 排序 | `(-size, path)` | `(bucket, -size, name)` | merged > **最小 rank** > 最大 size |
| 空输入 | 空列表 | `raise FileNotFoundError` | `None` |

### 7.2 三份「有无 GPU kernel」也不一致

| | `profile.py` | `tracelens_analysis.py` | `_bypass_trace_reader.py` |
|---|---|---|---|
| 方法 | 子串计数 `'"cat": "kernel"'` | `json.load` 全文件 + `is_kernel_event` | 流式扫描 |
| 内存 | 抽样 2 MB + 流式确认 ≤64 MB | **整个文件进内存** | 恒定 |
| 截断文件 | 读到多少算多少 | 抛异常 → unreadable | 保留前缀 + `stream_errors` |
| 零 kernel 的反应 | 记 `zero_ops`,**只警告** | 直接 `raise` | 发 `bypass_no_gpu_kernels` |

### 7.3 合并的约束

`_capture_shapes.py`(109 行)与 `_idle_gate.py`(218 行)已确立可行形态:

- 必须放在 `agents/kernel/tools/` 下
- **只依赖 stdlib**——支持双导入:standalone 工具走 `from _capture_shapes import ...`(`sys.path` hack),orchestrator 走 `from hyperloom.agents.kernel.tools._capture_shapes import ...`
- 不能碰 TraceLens

测试面约 11,700 行(`test_tracelens_csv.py` 单文件 5,440 行),合并须保留旧函数名作为薄别名,分两步走。

### 7.4 其他盲区

1. **多机路径完全不做结构检查**:`_validate_trace_structure` 唯一调用点 `profile.py:1220` 在 `elif workspace_str:` 单机分支内。
2. **`trace_health` 只活在 profile 返回值里**:被 `instrument.py:620` 记入 breakdown metadata,但单独调用 `trace_analyze` 时拿不到,只有 `RooflineExecutor` 同时持有两半。
3. `request_handlers.py:3211` 第四份 glob(block-FP8 shape 采集)只排除 capture,不排除 split 产物。

---

## 8. 局限与待确认

- **阈值样本量小**:7 个 session / 3 个模型 / 纯 sglang + TP8 + MI355X。C1 的 30% 与 D1 的 2× 是"能干净分开这批数据"的取值,**无理论依据**。建议上线时先只记录不拦截,攒够分布再定阈值。
- **vLLM / 单卡 / xDiT 无样本**,判据在这些路径上的表现未知。
- **C2(干净分母)严格说不属于"trace 有效性判定"**,是门槛口径修正。但它是唯一能把 3 次归零直接变成 3 次有候选的改动,需决定是否一并纳入。
- F1 的根因(`HYPERLOOM_TRACELENS_PATCH_STATUS: unavailable` 与 graph 插桩的因果)来自三方相关性推断,未读 patch 源码确认。单变量验证方式见 `kimi-k3-kernel-phase.md` §8.1。

---

## 附:关键文件索引

| 用途 | 路径 |
|---|---|
| 采集执行器 | `src/hyperloom/orchestrator/actions/executors/profile.py` |
| 结构检查 | 同上 `_validate_trace_structure:162` |
| roofline 复合动作 | `src/hyperloom/orchestrator/actions/executors/roofline.py` |
| trace_analyze 分发 | `src/hyperloom/orchestrator/kernel/request_handlers.py:5361` |
| TraceLens 路由主流程 | `src/hyperloom/agents/kernel/tools/tracelens_analysis.py:main` |
| bypass 路由 | `src/hyperloom/agents/kernel/tools/bypass_trace_analysis.py` |
| Kineto 流式解析器 | `src/hyperloom/agents/kernel/tools/_bypass_trace_reader.py` |
| graph 少记检测 | 同上 `_graph_under_recorded:93` |
| idle / compute 闸门 | `src/hyperloom/agents/kernel/tools/_idle_gate.py` |
| capture 分类(已统一) | `src/hyperloom/agents/kernel/tools/_capture_shapes.py` |
| 事故复盘 | `/shared_nfs/jqliu/TOP-Model-Analysis/0826/{kimi-k3,qwen3.8}-kernel-phase.md` |
| session 索引 | `/shared_nfs/jqliu/TOP-Model-Analysis/{0826,0827}/session-log-paths.md` |
