# §3.7 资源 lane 重设计 — 引入 research_lane + 并发 dispatcher

## 1. 设计目标

让 specialist sub-agent 能够并发, 而 deterministic executor (跑 E2E
bench / 应用 patch / 重启 server) 仍然严格串行。核心手段:

- 把 v0.6 的"每 lane 单 holder"模型升级为 `(lane_name, capacity)` 模型。
- 新增第五条 lane `research_lane`, capacity > 1, 与 benchmark / profile /
  server_lifecycle 全部不冲突。
- Coordinator 的 dispatcher 从串行 `for task in queued: await run`
  改为 *按 lane 容量并发*。

成功标准:

- N 个 specialist (N ≤ research_lane.capacity) 在同一时刻运行, 不阻塞
  benchmark_lane。
- benchmark_lane 仍然在任意时刻最多 1 个 holder (跑 E2E)。
- 资源拒绝路径清晰: lane 容量满时新任务进入 queued 等候, 不报错也不
  丢任务。

## 2. 现状回顾

v0.6 (`orchestrator/resource_lock.py`):

- 4 条 lane: `server_lifecycle`, `workspace_mutation`, `benchmark_lane`,
  `profile_lane`。
- 实现: SQLite `leases` 表, lane 是 PRIMARY KEY, *天然单 holder*。
- 冲突: `LANE_CONFLICTS` 表给出 lane 间的互斥关系 (例如 benchmark 与
  profile 互斥)。
- 调度: `Coordinator._pump_dispatcher_once` 串行 `for task in queued:
  result = await self.sub.run_task(task)`。

痛点:

- 想让 N 个 specialist 同时跑, 没办法 — lane 表 PK 在 lane 列。
- 即使能并发申请 lane, dispatcher 是一个 for 循环, 不会真的并发。
- 没有"per-lane semaphore"的概念。

## 3. 不变量

继承 §3.1 三主轴 + 三不变量。本节核心不变量:

### Inv-7.1 — serving GPU 单租户 (重申)

任何时刻 `benchmark_lane.holders ≤ 1` 且 `profile_lane.holders ≤ 1`,
且 `server_lifecycle.holders ≤ 1`。这条来自 §3.1 Inv-3, 是 v0.8 容
量化设计的 *底线*。

### Inv-7.2 — research_lane 不污染 serving lane

research_lane 与 server_lifecycle / benchmark_lane / profile_lane 的
LANE_CONFLICTS 关系为 *不冲突*。即 research 任务申请 lane 时, 不阻塞
benchmark 任务申请, 反之亦然。

### Inv-7.3 — lane 申请仍然全或无

一个任务申请多条 lane (`requires_lanes`), 必须**全部成功获取**才能跑;
任何一条失败, 整批回滚。这条来自 v0.6 已有的 `acquire_many` 事务
语义, v0.8 不放松。

## 4. 核心机制

### 4.1 lane 概念升级: 引入 capacity

每条 lane 用 `(name, capacity)` 描述:

| lane | v0.6 capacity | v0.8 capacity (默认) | 备注 |
|---|---|---|---|
| `server_lifecycle` | 1 (隐式) | 1 | 重启 / 应用 patch 时占用; 整个 session 内保持单租户 |
| `workspace_mutation` | 1 (隐式) | 1 | kernel patch apply 时占用 |
| `benchmark_lane` | 1 (隐式) | 1 | 真 E2E bench, 严格串行 (Inv-7.1) |
| `profile_lane` | 1 (隐式) | 1 | KERNEL phase 入口的 1 次 profile |
| `research_lane` (新) | — | 6 (CLI 可调, 0–32) | specialist 在此并发; 与 serving lane 不冲突 |

实施层: `leases` 表的 PK 不再是 `lane`, 而是 `(lane, holder_id)`,
并配套一个 `lane_capacity` 元表。`acquire_many` 时检查 capacity:

```
for lane in requested_lanes:
    used = count(holders where lane = ?)
    if used >= capacity[lane]: rollback (LaneFull)
```

剩余的 `LANE_CONFLICTS` 互斥逻辑保留 (v0.6 中 `_expand_lanes` 做的事:
申请 benchmark_lane 隐式也要把 profile_lane / server_lifecycle 都申到)。

### 4.2 LANE_CONFLICTS 矩阵 (v0.8)

```
                  server_  workspace_  benchmark_  profile_  research_
                  lifecycle mutation    lane        lane     lane
   server_       │  ×       —          C            C         —
   workspace     │  —       ×          —            —         —
   benchmark     │  C       —          ×            C         —
   profile       │  C       —          C            ×         —
   research      │  —       —          —            —         (cap N)

   × = 自身互斥 (容量 1)
   C = cross-lane 互斥 (传递扩展, 申一占两)
   — = 无冲突
   (cap N) = 容量 N, 自身可并发
```

要点:

- research_lane 自身可并发 (capacity > 1), 不与任何其它 lane 冲突。
- benchmark / profile / server 之间保持 v0.6 现有冲突: 一个 bench 跑
  起来时, 没有 profile 或 server restart 能同时进行。
- workspace_mutation 仍然单容量, 但**研究阶段不应当抢它**:
  specialist 只读源码, 不动 workspace; 真正动 workspace 的是 kernel
  patch apply 阶段, 这条不冲突 research 的设计是有意的 (specialist
  在 patch apply 期间仍能读源码, 只要源码挂载与 patch 工作目录分离)。

### 4.3 dispatcher 并发化

Coordinator 主 loop 中 `_pump_dispatcher_once` 改为 *按 lane 容量
并发*:

```
queued = TaskRegistry.queued()
for task in queued:
    if can_acquire(task.requires_lanes):  # 即时 try, 不阻塞
        spawn task                          # 异步 task, 不 await
    else:
        leave in queue
await asyncio.gather(spawned, return_exceptions=True)
```

每 tick 收割已完成的任务, 启动队列里能拿到 lane 的新任务。具体异步
原语 (asyncio.gather / TaskGroup / per-lane semaphore) 留实施稿。

并发节奏的关键点:

- specialist 任务只占 research_lane → 多个 specialist 可在同 tick 启动。
- explore 任务占 server + benchmark → 同一时刻最多 1 个 explore。
- profile 任务占 profile + server → 与 explore 互斥, 但与 research 不
  冲突, 所以 profile 跑的同时仍可派 specialist (实际上 profile 在
  KERNEL phase 入口跑, EXPLORE 不跑, 所以这条罕见)。
- kernel apply (workspace_mutation) 与 specialist 不冲突, 但与 server
  restart 串行。

### 4.4 capacity 配置

CLI flag:

- `--research-lane-capacity N` (默认 6, 范围 [0, 32])
- `N = 0` 等价于关闭 specialist 派发 (degrade 路径; EXPLORE 退化为
  v0.6 风格的 LLM 自填 grid)

环境变量:

- `INFERENCE_OPTIMIZER_RESEARCH_LANE_CAPACITY` 同义。

运行时调整:

- session 启动时锁定 capacity, 不允许中途改 (避免 lease 状态机抖动)。

## 5. 失败模式与回退

### 5.1 LaneFull (新增异常类)

`acquire_many` 当某条 lane 已达 capacity, 抛 `LaneFull`, 与 v0.6
`LaneBusy` (互斥冲突) 区分语义。dispatcher 收到 LaneFull 时**不**报
错, 把任务保留在 queued 队列, 下一 tick 再试。

### 5.2 Lease 续约失败

specialist 跑得久 (例如 8 分钟), lease TTL 默认 1800 秒应当够; 但若
lease 过期, robustness 走现有 stale lease 路径 (Lease 自动释放, holder_id
不再持有)。specialist 不需要额外续约逻辑, 只要在 max_specialist_turns
内跑完。

### 5.3 specialist 占满 research_lane 后 explore 派不出

EXPLORE 阶段是先派 specialist 后 propose explore, 不会同时抢; 所以
不会出现 "explore 等 research_lane" 的情况 (explore 不抢 research_lane)。

如果 *并发派 explore + specialist* 的 race condition 被引入 (例如
robustness 抢着重启 server 时 explore 入队), 由 dispatcher 串行
benchmark_lane 自动序列化。

### 5.4 capacity 0 模式 (degrade)

- specialist 不派, 直接由 LLM 用 default_grid 提 explore variant。
- breakdown 中 specialist_runs 段为空, capability_summary.specialist
  行不出现。
- 该模式作为**应急回退**: Cortex 不可达 / LLM quota 紧张时, 让 EXPLORE
  仍能跑通 v0.6 风格。

## 6. 实施步骤

1. **leases 表 schema 升级**: PK 改成 `(lane, holder_id)`, 加表
   `lane_capacity(lane TEXT PK, capacity INT)` 在启动时填入默认值,
   CLI flag 覆盖。
2. **acquire_many 容量检查**: 概念上是"先 SELECT count, 再 INSERT 行,
   单事务"。具体实现 (BEGIN IMMEDIATE 内做 count + insert) 留实施稿。
3. **LANE_CONFLICTS 表更新**: 加 research_lane 这一行 / 列, 全 `—`
   (与已有 lane 不冲突, 自身用 capacity 控制)。
4. **dispatcher 并发化**: `_pump_dispatcher_once` 收 try-acquire +
   asyncio.gather; 已完成任务在每 tick 收割。
5. **CLI flag**: 新增 `--research-lane-capacity`, 默认 6; manifest
   写入此值便于 resume 一致。
6. **resume 兼容**: 老 v0.6 leases 表 schema 升级时做一次性 migration
   (PRAGMA + ALTER TABLE 概念); lane_capacity 表如果不存在就用 default
   值填; 不破坏老 session 的 in-flight lease。
7. **breakdown 段**: 在 `phase_timeline` 中加 lane 占用 timeline (每
   段 phase 哪条 lane 何时被占, 容量峰值多少); 由 §3.12 统一写入。

## 7. 边界条件

| 场景 | 行为 |
|---|---|
| 用户把 capacity 设到 32 | 上限不变, 但实际 LLM quota / Cortex 容量会先扛不住; CLI 给 warn 但不拒绝 |
| capacity = 0 | EXPLORE 退化为无 specialist 模式; 不视为错误 |
| research_lane 全占满 + Orchestration 还想再派 specialist | 任务进 queued; 一旦有 specialist 退出, 下 tick 自动启动新任务 |
| 多个 specialist 同时引用同一份本地源码文件 | 不冲突, 全是只读 |
| specialist 跑期间用户手动重启 server (绕过 Coordinator) | server_lifecycle lane 不感知 (绕过 lease 是用户责任); 后续 explore 跑会发现 server 状态异常, 由现有 baseline_failure_streak 路径处理 |
| migration 时 v0.6 的 lease 行 lock 没释放 | 沿用 v0.6 stale lease 自动回收逻辑 (TTL 过期 → robustness 清理) |

## 8. 验收标准

- [ ] `leases` 表升级后, 同一 lane 可有 N 个 holder (N ≤ capacity)。
- [ ] specialist 派发 6 个时, 6 个并发跑;第 7 个进 queued, 等。
- [ ] 任意时刻可观测 `benchmark_lane.holders ≤ 1`。
- [ ] explore 跑期间, specialist 仍可在 research_lane 上跑 (跨 phase
      不会, 但同 EXPLORE 内可能, 比如有 specialist 还没收尾 explore
      已经在跑的情况; 这种情况由 phase 节奏避免, 此条仅作能力验证)。
- [ ] capacity 0 模式下, EXPLORE 走 v0.6 风格 default_grid, 不挂。
- [ ] resume 时 capacity 配置由 manifest.json 还原, 不允许中途调。
- [ ] LaneFull 与 LaneBusy 在 breakdown 中可区分 (前者纯容量, 后者冲突)。

## 9. 依赖与影响面

- **上游**: §3.1 (Inv-3 serving GPU 单租户), §3.5 (specialist sub-agent
  需要并发 lane)。
- **下游**:
  - §3.4 EXPLORE 派发 N specialist, N 由本节 capacity 决定。
  - §3.5 specialist runner 自动占 research_lane。
  - §3.10 SharedState 不直接持 lane 状态 (lease 在 SQLite), 但 manifest
    新增 `research_lane_capacity` 字段。
  - §3.11 PolicyGate 不直接关心 lane (lane 是资源, 不是权限), 但
    `delegate{action='specialist'}` 的合法性校验保证只 Orchestration 派。
  - §3.12 breakdown 中 `lane_timeline` 段。
  - §3.13 M5 / M6 实施 capacity 与并发。

## 10. 哲学回引

本节是**Inv-3 (serving GPU 单租户)** 的具体落地: serving lane 容量
保持 1, 而把 specialist 的并发承载放到独立的 research_lane;
**主轴 C** (执行体一分为二) 由 lane 划分得到强制分隔, 防止 specialist
误闯 serving lane。
