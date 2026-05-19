# M6 — Specialist 多 domain + 并发 dispatcher

## 1. 设计目标

把 M5 的"单 domain 单实例" specialist 升级为 **6 类 domain 并发**, 并
把 Coordinator dispatcher 从串行 for loop 升级为按 lane 容量并发的
asyncio.gather 模型。

落地后用户应当看到: EXPLORE 内一轮派出多个 specialist (例如 kernel
+ framework + comm + pr_intel 4 个并发), 所有 specialist 跑完后
Orchestration 汇总 proposal_set 选 top-M, 进入 explore executor 串行
跑变。

## 2. 范围

**包含**:

- research_lane capacity 提到默认 6 (CLI flag 可调)。
- 6 类 specialist domain 全部接入: kernel / framework / comm / compiler
  / system / pr_intel。
- domain → 仓库 yaml 全表 (M4 已部分写, 此处补齐)。
- domain 选择策略 (Orchestration 一侧, prompt 中明确 6 类 domain 的
  选用条件)。
- Coordinator `_pump_dispatcher_once` 改并发 (asyncio.gather +
  per-lane try_acquire 节流)。
- specialist_done 汇总逻辑 (Orchestration tick 中等 N 个 specialist
  全部 done 才进 propose_action; 或部分 done 即推进, 由 prompt 决定 —
  本里程碑给保守"全 done 才推进"作默认, M7 灰度部分推进)。
- breakdown.specialist_runs 字段 parallelism 实际 > 1 时验证。
- breakdown.telemetry.lane_timeline 段 (lane 占用峰值与 timeline)。

**不包含**:

- plateau 阈值参数化 (M7)。
- specialist domain 之间的 *依赖关系*建模 (例如 pr_intel 给 kernel
  提引用); 如有, 由 LLM 自己在两轮派发间协调, 不在系统层建模。

## 3. 与 M3/M4/M5 的关系

- M3 提供 explore action; 多 domain 提议汇总后仍走 explore。
- M4 提供 PR Monitor 工具集 + domain 仓库映射; M6 全用上。
- M5 提供 SpecialistRunner; M6 仅扩 capacity + dispatcher 并发, 不动
  specialist 内部生命周期。

## 4. 概念交付物

| 交付物 | 说明 |
|---|---|
| research_lane capacity 提到 6 | CLI flag, manifest 锁定 |
| 5 类新 domain prompt 模板 | kernel / comm / compiler / system / pr_intel (framework 已在 M5 落地) |
| domain 选择策略 prompt 段 | Orchestration prompt 中明确"什么 gap 派什么 domain"指引 |
| 并发 dispatcher | asyncio.gather + per-lane 节流; 容量满时任务保持 queued |
| specialist 汇总语义 | 默认"全 done 才推进 explore"; 灰度 flag 允许"K 个 done 即推进" (M7 调) |
| breakdown.specialist_runs.parallelism | 真实记录每轮的并发数 |
| breakdown.telemetry.lane_timeline | 见 §3.12 §4.5 |

## 5. domain 选择策略 (Orchestration prompt 段)

新增 prompt 段 "DOMAIN SELECTION GUIDE":

```
You are entering an EXPLORE round. Based on the gaps below, decide
which specialist domains to dispatch in parallel:

  - kernel_specialist:   gap mentions attention / MoE / GEMM / fused
                         attention / aiter / triton kernels
  - framework_specialist: gap mentions scheduler / cuda_graph /
                         kv_cache / batching / chunked prefill /
                         max-num-seqs
  - comm_specialist:     gap mentions allreduce / NCCL / RCCL /
                         QuickReduce / collective / topology
  - compiler_specialist: gap mentions torch.compile / inductor /
                         triton codegen / AMDGCN / register pressure
  - system_specialist:   gap mentions dispatch overhead / launch
                         latency / memory fragmentation / driver /
                         KFD
  - pr_intel_specialist: dispatch ONE every K rounds (default K=3)
                         to bring fresh PR references for next round

For each gap, choose 1-3 most relevant domains. Multiple gaps can
share a specialist (the prompt will let the specialist see all
relevant gaps).

Total parallelism ≤ research_lane capacity (default 6).
```

domain 选择由 LLM 决定, 系统不干预 (符合主轴 A "决策由 LLM")。

## 6. 并发 dispatcher 概念

```
async def _pump_dispatcher_once():
    queued = TaskRegistry.queued()
    spawned = []
    for task in queued:
        try:
            lease = locks.try_acquire_many(task.requires_lanes,
                                           holder=task.task_id)
        except (LaneBusy, LaneFull):
            continue   # 留在队列
        spawned.append(asyncio.create_task(
            sub.run_task_with_lease(task, lease)
        ))
    # 不 await 全部完成, 让本 tick 继续推进; 下个 tick 收割已完成任务
    for t in spawned:
        if t.done():
            ...   # 处理 result, 写 delegated_result event
```

要点:

- *try_acquire* 是非阻塞的, lane 满 / 冲突即跳过该任务。
- spawned 任务不在本 tick 等; 异步跑, Coordinator 主 loop 继续。
- 每 tick 末尾扫已完成的 spawned, 处理 result。

具体实现 (asyncio.gather 全 await vs TaskGroup vs background tasks
register) 留实施稿; 概念上是"per-lane 节流的 fan-out"。

## 7. specialist 汇总策略

Orchestration 在 tick 中等 specialist 完成。M6 阶段两种模式 (CLI flag
切换):

| 模式 | 描述 |
|---|---|
| `wait_all` (默认) | 本轮派的 N 个 specialist 全 done 才进 propose_action; 任一 stale 也合成 empty done 算 done |
| `partial_k` | 收到 ≥ K 个 done 即推进 (K=ceil(N/2) 默认); 慢 specialist 的结果**进下一轮**而非本轮 |

`wait_all` 是保守正确; `partial_k` 是优化, M7 灰度调。M6 阶段默认
`wait_all` 防止"上一轮慢 specialist 结果错位汇总到下一轮"的复杂语义。

## 8. 实施步骤 (PR 拆分)

| PR | 内容 |
|---|---|
| 1 | research_lane capacity 配置升级 (manifest + CLI flag) |
| 2 | leases 表 schema 升级 (PK 改 (lane, holder_id), lane_capacity 元表) |
| 3 | acquire_many 容量检查 (LaneFull 异常类) |
| 4 | LANE_CONFLICTS 表更新 (research_lane 行/列) |
| 5 | _pump_dispatcher_once 改并发 |
| 6 | 5 类新 domain prompt 模板 |
| 7 | Orchestration prompt 中 DOMAIN SELECTION GUIDE 段 |
| 8 | breakdown.specialist_runs.parallelism + telemetry.lane_timeline 段 |
| 9 | specialist 汇总策略 (wait_all 默认 + partial_k flag) |

## 9. 验收清单

- [ ] EXPLORE 内一轮可见 ≥ 2 个并发 specialist 同时跑 (transcript ts
      重叠), parallelism 字段 ≥ 2。
- [ ] benchmark_lane.holders 在任一时刻 ≤ 1 (Inv-7.1 守住)。
- [ ] research_lane.peak_holders 可达 capacity (默认 6)。
- [ ] LaneFull 异常在 capacity 满时被 dispatcher 接住, 任务保持 queued,
      下 tick 自动启动新任务。
- [ ] 6 类 specialist 全部能被派出 (在合适的 gap 描述下)。
- [ ] resume v0.5/M5 老 session, lane 表 schema 升级无损。
- [ ] `wait_all` 模式下任一 stale 触发 empty done 合成, round 推进
      不卡。

## 10. 风险与回退

主风险:

- **并发后 SharedState 写竞争**: Coordinator 主 loop 仍是单写者, 写
  state.json 仍串行; specialist 不直接写 state, 只 emit intent. 这个
  风险通过 Inv-1 + Inv-5.2 守住.
- **LLM quota 撞墙**: 6 个并发 specialist 同时调 LLM, 短时间内打高
  QPS. 缓解: per-LLM-backend rate limiter (在 backend 层做, 不在 lane
  层); 必要时 capacity 默认调 3.
- **PR Monitor 跨集群带宽**: 6 个 specialist 同时调 MCP, PR Monitor
  压力. 缓解: 每 specialist 默认 max_turns=8, 单 turn 工具调用上限,
  实际 PR Monitor 调用 ≤ 6 × 8 × 5 = 240 次, 一个 EXPLORE round.

回退:

- `--research-lane-capacity 1` 退到 M5 单 specialist 模式。
- `--research-lane-capacity 0` 完全 degrade 到 M3。
- LANE_CONFLICTS 改动是 schema 改动, 不可热回滚, 但 capacity=1 后行为
  等价 v0.6。

## 11. 哲学回引

本里程碑落地 §3.7 资源 lane 重设计; **Inv-7.1 / Inv-7.2 / Inv-7.3** 三
条 lane 不变量在 leases 表 schema 升级 + LANE_CONFLICTS 表更新 + dispatcher
改动中协同保证。**主轴 C (执行体一分为二)** 在并发 specialist 同时跑
+ benchmark_lane 严格串行的对比中得到最强体现。
