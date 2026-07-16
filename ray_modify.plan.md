# Ray-managed GPU Execution 统一 GPU 执行层计划(只做值得的部分)

## 0. 目标(一句话)

核心思想明确为 **Ray-managed GPU execution**:让所有会实际占用 GPU 的 serving /
benchmark / profile / kernel-bench / GPU-specialist 进程都运行在 Ray task/actor
持有的 GPU resource lease 内,由 Ray 负责 GPU 排队、可见设备隔离和跨节点放置。

把 **GPU + serving 类执行**(baseline / profile / explore / sweep / conc_sweep /
replay_warm_recipe / rebench / kernel-bench / needs_gpu specialist)统一到
**Ray(`ray.remote` task/actor + placement group)** 之上,让 **单节点 = 1 节点 Ray
集群**,从而:①折叠现在分裂的 `is_multi_node()` 两条执行路径;②用 Ray 的
`num_gpus` 调度取代自研的 SQLite `SpecialistGpuPool` 物理 GPU-id 分配;③把 lane
互斥不变量映射成 Ray 自定义资源,保证 GPU specialist 与 serving 不再抢卡,而是在
Ray 资源队列里串行/排队。**不改** Coordinator 状态机、SQLite journal/artifact
契约、轻量 CPU/LLM 步骤与 host-local 健康探针。

注意:这里说的不是把所有步骤都包成 Ray Dashboard Jobs REST 的 `submit_job`。
Dashboard Jobs 只保留跨 sandbox→Ray 集群边界的启动/提交用途;真正的统一执行层是
Ray-managed task/actor,并且 GPU 进程的生命周期必须被 Ray actor/task 持有。

---

## 1. 范围锁定(明确"值得"与"不做")

### 1.1 做(worthwhile)

| # | 内容 | 现状痛点 |
|---|---|---|
| A | 统一 GPU 执行后端 `RayExecutionBackend`,单节点也走 Ray-managed GPU execution | 单/多节点两条路径分叉,`is_multi_node()` 遍布执行器 |
| B | 折叠 executors 里的 `is_multi_node()` 分支 | `baseline.py`/`_grid_runner.py`/`profile.py` 各有一套远程/本地逻辑 |
| C | GPU specialist 整体进入 Ray actor,用 Ray `num_gpus` 取代 `SpecialistGpuPool` 物理 GPU-id 分配 | 手写 SQLite GPU 池 + ROCR 掩码注入,易错;specialist 直接 Bash 触 GPU 时 Ray 现在不可见 |
| D | lane 互斥 → Ray 自定义资源(信号量) | serving⊥profile⊥benchmark⊥gpu_research 目前靠 SQLite 协同获取 |

### 1.2 不做(out of scope,明确排除)

- **Coordinator 状态机本身**:仍是本地 Python 状态机,不进 Ray。
- **CPU/LLM 轻量步骤**:orchestration 决策、critic、target_analysis、report、
  session_breakdown、trace_analyze —— 留在本地(进程内或本地子进程),不套 Ray。
- **SQLite lane/lease + task_registry + message_bus + resume**:作为 **Coordinator
  侧调度视图**保留;Ray 只负责**物理放置与执行**(见 §4.5 的双层设计)。
- **host-local 健康探针**:`robustness/sources/local_probe.py` 的 `rocm-smi`/`ps`/
  日志尾仍在执行发生的宿主上跑,不改其形态。
- **Dashboard Jobs REST 作为万能传输**:仅保留跨 sandbox→Ray 集群边界的启动/提交
  用途;不把 REST `submit_job` 推广成所有动作的执行 API。统一执行层用
  `ray.remote` task/actor,不是 job_logs/heredoc/base64 那套控制面。

---

## 2. 现状(为什么值得改)

### 2.1 执行路径分叉(核心痛点)

同一个 benchmark 有两条实现:

- **单节点**:`BaselineExecutor._run_single`(`actions/executors/baseline.py`)、
  `run_grid`(`_grid_runner.py`)本地 `run_with_session_kill([magpie_python, "-m",
  "Magpie", "benchmark", ...])` 起 Magpie,Magpie 再本地拉起 sglang/vllm,GPU 由
  `ROCR_VISIBLE_DEVICES` 直接占用。
- **多节点**:先 `restart_server_for_round`(`_multi_node_server_lifecycle.py`)经
  Ray Dashboard 在 RayJob pod 重启服务,再 `magpie_remote_env()` 注入
  `MAGPIE_RUN_PHASE=client` + `BENCHMARK_BASE_URL`,Magpie 只当 client 打 head pod。

后果:`baseline.py` / `profile.py` / `_grid_runner.py` / `sweep.py` /
`kernel/request_handlers.py` 到处是 `is_multi_node()` 分支 + `magpie_remote_env()` +
每轮 restart,复杂度与 bug 面都集中在这条缝上。

### 2.2 三套资源/GPU 管理并存

| 机制 | 位置 | 负责 |
|---|---|---|
| SQLite lane/lease | `bus/resource_lock.py`(`KNOWN_LANES` / `LANE_CONFLICTS`) | serving/benchmark/profile/gpu_research 的互斥与容量 |
| SQLite GPU 池 | `bus/gpu_pool.py`(`SpecialistGpuPool.try_acquire` → `GpuLease.gpu_ids`) | GPU specialist 的物理 GPU-id 分配 + TTL |
| Ray `num_gpus` | `agents/kernel/tools/backends/ray_runtime.py` + `geak_submit.py` | GEAK 的 GPU 任务调度(已在用 Ray) |

GEAK 已经证明 Ray 能在本机管好 GPU(`ensure_ray_cluster` 处理了 #433 fd 上限、#432
版本不匹配,并把 dashboard 绑 loopback 规避未认证 Jobs API RCE)。C/D 就是把 GPU 池
与 lane 互斥收敛到这套已验证的 Ray 能力上。

### 2.3 执行器注册点(改造入口)

`cli/executors.py::_register_executors` 把每个 executor 注册到
`coordinator.sub.register_executor(kind, fn)`;executor 通过
`sub_agent_runner.py` 的 `RunnerContext.extra`(含 dispatcher 注入的
`extra_context["gpu_ids"]`)拿到上下文。**这一层保持不变** —— 我们只替换 executor
"如何执行子进程",不动注册/调度契约。

---

## 3. 目标架构

```
                     Coordinator(本地状态机,不变)
                     │  依旧: SQLite lane 门 + task_registry + resume
                     ▼
        SubAgentRunner.register_executor(kind, fn)   ← 契约不变
                     │
        ┌────────────┴─────────────┐
        │                          │
   CPU/LLM 执行器               GPU/serving 执行器
   (本地,不变)                 经 Ray-managed GPU execution
   target_analysis/report/      ├─ ensure_ray(1 节点或多节点)
   trace_analyze/critic         ├─ ServingActor / ServingGroupManager 持有服务生命周期
                                ├─ GpuSpecialistActor 持有 needs_gpu specialist 生命周期
                                ├─ ray.remote(num_gpus=..., resources={...})
                                │     内部仍 run_with_session_kill(Magpie/bench/helper ...)
                                └─ 产物落"统一寻址"共享目录(§4.6)
                     │
                     ▼
        Ray 调度(单节点=1节点集群,多节点=同一 API 跨节点)
        · num_gpus 取代 SpecialistGpuPool
        · custom resource 取代 lane 物理互斥
        · GPU 进程不得脱离其 Ray actor/task 存活
```

关键:**executor 外部接口(`async def __call__(ctx) -> dict`)零改动**,只把"本地
subprocess"这一步换成"Ray actor/task 里跑同样的 subprocess"。单节点自动获得 Ray
的 GPU 调度;`is_multi_node()` 分支收敛为"Ray 集群是 1 节点还是 N 节点"。

硬不变量:任何占 GPU 的 server / benchmark / profile / specialist / kernel-bench
进程都必须在仍存活的 Ray actor/task 内运行。**不允许** Ray task 提交后 `nohup` /
detach 一个 GPU 进程然后退出,否则 Ray 会提前释放 GPU resource,无法保证 serving 与
specialist 不抢卡。

---

## 4. 详细设计

### 4.1 A —— `RayExecutionBackend`(新,薄封装)

新增 `orchestrator/actions/executors/_ray_backend.py`:

```python
class RayExecutionBackend:
    def ensure(self) -> None: ...          # 复用 kernel ray_runtime.ensure_ray_cluster
    async def run_subprocess(
        self, cmd, *, env, cwd, num_gpus, resources, timeout_s, result_dir,
    ) -> SubprocessResult: ...             # ray.remote(num_gpus, resources) 内跑
                                            # run_with_session_kill(cmd)
```

- **单节点**:进程启动时 `ensure_ray_cluster(num_gpus=detect_gpu_count())`
  (直接复用 `agents/kernel/tools/backends/ray_runtime.py`,已处理 fd/版本/loopback)。
- **执行**:把现有 `run_with_session_kill(Magpie cmd, env, cwd)` 原样搬进
  `@ray.remote(num_gpus=..., resources=...)` 的 worker;worker 内 **Ray 已设好**
  `ROCR/HIP/CUDA_VISIBLE_DEVICES`,删掉执行器里手工 ROCR 掩码逻辑(`_grid_server_args.py`
  的 ROCR 注入、`geak_submit.py` 式的 logical-id 重映射保留在 GPU-worker 内即可)。
- **失败语义不变**:worker 返回 `(rc, stdout, stderr)` → executor 现有解析
  (`extract_benchmark_measurement` / `benchmark_report.json`)完全复用。

开关:`INFERENCE_OPTIMIZER_RAY_EXEC=1`(默认关,灰度);关时回落现有本地 subprocess,
保证可逐执行器切换、可回滚。

### 4.2 核心不变量 —— GPU 进程必须由 Ray lease 持有

为了真正达成"Ray 管 GPU 排队",所有占 GPU 的进程必须满足:

1. **进程生命周期 ≤ Ray actor/task 生命周期**:actor/task 退出前必须 kill 并 reap
   它启动的 server / benchmark / specialist 子进程。
2. **禁止 detached GPU 进程逃逸**:不允许 `nohup` / daemonized / background server
   在 Ray actor/task 结束后继续占 GPU。现有多节点 `restart-server` 的 detached
   server 模式必须在收敛阶段替换成 long-lived actor 模式。
3. **Ray 是物理 GPU 真相源**:GPU 可见设备由 Ray `num_gpus` 设置;Coordinator 不再
   分配物理 GPU id,只保留 lane/TTL/task 状态视图。

新增两类 long-lived actor:

- `ServingActor`(单节点) / `ServingGroupManager`(多节点):持有 serving 所需的
  `num_gpus` / placement group / `serving_slot`,内部启动 sglang/vLLM/Magpie server,
  提供 health / stop / log / trace 回收。server 活多久,actor 就活多久。
- `GpuSpecialistActor`:持有 `needs_gpu` specialist 的 `num_gpus`,在 actor 内启动
  specialist subprocess。specialist 的 Bash 权限仍保留,但任何 GPU 命令都发生在
  Ray 已隔离的可见设备内。

这条不变量是本计划的成败线:如果还有 GPU 进程绕过 Ray actor/task 存活,Ray 就无法
保证 serving 与 specialist 不抢 GPU。

### 4.3 B —— 折叠 `is_multi_node()` 分支

- `baseline.py` / `_grid_runner.py` / `profile.py`:把"本地起服务 vs client 打
  远端"两分支,统一为"向 `RayExecutionBackend` 提交一个带 `num_gpus=tp` 的服务+bench
  任务"。
  - 单节点:Ray worker 就在本机,行为等价现在的本地路径。
  - 多节点:同一提交落到跨节点 **placement group**;`restart_server_for_round` 的
    detached server 重启模式收敛为 `ServingGroupManager` 生命周期(actors 起停),
    `magpie_remote_env()` 仅在"bench client 与 serving actors 分离"时保留为一种放置
    策略,不再是独立代码路径。
- 里程碑内保持 `is_multi_node()` 可用(灰度期两条路径并存),切换完成后再删分支。

### 4.4 C —— Ray `num_gpus` 取代 `SpecialistGpuPool`

- `dispatcher.py::_spawn_fitting_queued` 里 `needs_gpu` 分支目前调用
  `gpu_pool.try_acquire(count=gpu_count, ...)` 拿 `gpu_ids` 并注入
  `extra_context["gpu_ids"]`。改为:**不再分配物理 id**,而是把 `gpu_count` 作为
  `GpuSpecialistActor` 的 `num_gpus` 传给 `RayExecutionBackend`,由 Ray 调度 + 设
  可见设备。
- `SpecialistGpuPool` / `gpu_pool.py` 降级为**仅容量记账**(用于 Coordinator 的
  wall-budget/TTL 视图与 prompt 展示),不再是物理分配的真相源;或在灰度完成后删除。
- bench-capable specialist(`mode=patch & bench=true`)的 `gpu_count` floor 到
  serving TP 的逻辑(`specialists/profile.py::uses_whole_machine_gpu_lane`)保留,
  只是落点从"SQLite 选 id"变成"Ray 请 N 卡"。

### 4.5 D —— lane 互斥 → Ray 自定义资源(双层,低风险)

**不删 SQLite lane**(避免动 resume/gating/大量测试)。采用双层:

- **上层(不变)**:Coordinator 仍用 `resource_lock.py` 的 lane 门做**调度决策**
  (谁能被 dispatch),这层便宜、已测试、支撑 resume。
- **下层(新)**:把物理互斥交给 Ray **自定义资源当信号量**。给每台 GPU 节点声明
  `resources={"serving_slot": 1}`(单节点 `ray start --resources`,KubeRay/RayJob
  worker pod 启动参数同样声明;非 GPU 节点不声明);`benchmark`/`profile`/
  `server_lifecycle`/`gpu_research` 类任务在 Ray 侧 `resources={"serving_slot": 1}`
  独占该节点的服务槽,天然实现
  `benchmark_lane ⊥ profile_lane ⊥ server_lifecycle ⊥ gpu_research_lane`。
  - 映射表:

  | SQLite lane(上层门) | Ray 资源(下层物理) |
  |---|---|
  | `server_lifecycle` / `benchmark_lane` / `profile_lane` | `serving_slot=1`(整机服务槽) |
  | `gpu_research_lane`(cap-1,与 serving 互斥) | `serving_slot=1` + `num_gpus=whole` |
  | `gpu_specialist_pool`(serving-disjoint) | `num_gpus=n`(不占 `serving_slot`) |
  | `research_lane`(纯 CPU LLM) | 不进 Ray(留本地) |

- 好处:上层门与下层信号量互为冗余,任一层生效都不会让 serving 与 GPU specialist
  物理共卡;迁移期可先只上下层其一,风险可控。

### 4.6 前置条件 —— 产物"统一寻址"(最大风险,先解决)

单节点现在白拿"一块本地盘";Ray 化后 worker 可能在别的节点。必须先统一:

- 所有 per-task 产物(`benchmark_report.json`、trace、`server.log`、`RESULT_DIR`)
  写到**会话级共享根**(单节点=本地盘;多节点=已有的 wekafs 共享根,复用
  `HYPERLOOM_MN_PROFILE_TRACE_DIR` / `mn_profile_trace_root()` 的命名约定)。
- 复用 #523 的本地镜像思路(`baseline.py::_ensure_local_inferencex`):对 relative-path
  dump 敏感的 server cwd 仍就近落本地盘,产物再回收到共享根。
- `session_breakdown.json` 的 collector 路径解析统一走"共享根相对路径",保证下游契约
  不变。
- **验收前置**:单节点开 `INFERENCE_OPTIMIZER_RAY_EXEC=1` 跑通一次 baseline,产物
  路径与关路径 byte 级一致(或差异可解释)。

---

## 5. 涉及文件清单

| 文件 | 改动 |
|---|---|
| `orchestrator/actions/executors/_ray_backend.py`(**新**) | `RayExecutionBackend`:`ensure` + `run_subprocess`;`ServingActor`/`GpuSpecialistActor` 的公共提交/取消/日志接口 |
| `orchestrator/actions/executors/_ray_serving.py`(**新**) | `ServingActor`(单节点) + `ServingGroupManager`(多节点 placement group/rank actors);禁止 detached server 逃逸 |
| `orchestrator/actions/executors/_grid_runner.py` | `run_grid` 单次 Magpie 提交改走 backend;删本地 ROCR 手工注入(移入 worker) |
| `orchestrator/actions/executors/baseline.py` | `_run_single` 经 backend;`is_multi_node()` 分支收敛;产物落共享根 |
| `orchestrator/actions/executors/profile.py` | 同上;torch trace 目录统一寻址 |
| `orchestrator/actions/executors/sweep.py` / `conc_sweep`(在 executors 内) | 经 `run_grid` 自动继承,无需单独改 |
| `orchestrator/actions/executors/_multi_node_server_lifecycle.py` | `restart_server_for_round` 收敛为 `ServingGroupManager` 生命周期(灰度期保留旧路径) |
| `orchestrator/actions/executors/_multi_node_env.py` | `magpie_remote_env()` 降级为放置策略之一;`is_multi_node()` 收口 |
| `orchestrator/loop/dispatcher.py` | `needs_gpu` 分支:`gpu_pool.try_acquire` → 提交 `GpuSpecialistActor(num_gpus=...)`;移除物理 id 注入 |
| `orchestrator/bus/gpu_pool.py` | 降级为容量记账(或灰度后删除);`try_acquire`/`release` 保留桩以过测试 |
| `orchestrator/bus/resource_lock.py` | 不动语义;新增 lane→Ray 资源映射常量供 backend 读取;声明 `serving_slot` 互斥意图 |
| `orchestrator/kernel/request_handlers.py` | kernel-bench / integrate re-bench 经同一 backend(多节点 `cmd_kernel_bench` 收敛) |
| `agents/kernel/tools/backends/ray_runtime.py` | 复用其 `ensure_ray_cluster`/`quiet_ray_init`;抽出可被 orchestrator import 的稳定入口 |
| `inference_optimizer/cli/__init__.py` | 启动时按 `INFERENCE_OPTIMIZER_RAY_EXEC` 调 `backend.ensure()`;`--nodes` 语义统一为"Ray 集群规模" |
| `inference_optimizer/cli/executors.py` | 不改注册契约;可选把 backend 注入 executor 构造 |

**净收敛**:两条 benchmark 路径 → 一条;SQLite GPU 池物理分配 → Ray `num_gpus`;
手工 ROCR 掩码 → Ray 托管。预期执行器分支代码显著下降。

---

## 6. 落地顺序(阶段化,可逐步回滚)

| 阶段 | 内容 | 门槛/验收 |
|---|---|---|
| **P0(硬不变量 vertical slice)** | §4.6 产物统一寻址 + `RayExecutionBackend` 骨架 + 单节点 `ServingActor` 跑一次 baseline + 同时提交一个 GPU-specialist mock | serving actor 存活时 Ray GPU resource 被占用;specialist mock 必须 pending;actor 退出后 server 子进程全灭 |
| **P1(单节点 serving actor 化)** | baseline/profile/explore/sweep 单节点走 `ServingActor`;旧本地路径灰度并存 | 单节点全 pipeline 与关路径结果一致;`RAY_EXEC=0` 可回滚;无 detached GPU 进程 |
| **P2(GPU specialist actor 化)** | `needs_gpu=true` specialist 整个 subprocess 进入 `GpuSpecialistActor(num_gpus=...)`;`SpecialistGpuPool` 降级为记账 | specialist 的任意 Bash GPU 命令都在 Ray 可见设备内;TTL/cancel 能 kill 子进程;dispatch/并发不退化 |
| **P3(lane→Ray resource 双层互斥)** | 注册 `serving_slot` 自定义资源;SQLite lane 门保留,Ray resource 做物理排队 | 并发压力下 serving/specialist 不共卡;`ray status` 可见 pending 资源需求;lane 门冗余生效 |
| **P4(多节点 ServingGroupManager)** | 多节点 serving 用 placement group + rank actors 替换 detached `restart_server_for_round`;`magpie_remote_env()` 降级为放置策略 | 多节点全 pipeline 通过;rank actors 持有 GPU 直到 stop;health/log/trace 回收一致 |
| **P5(删旧路径,兑现代码减少)** | 删除旧 `is_multi_node()`/本地 GPU 池物理分配/冗余 restart 分支 | 回归绿后统计净删代码量;`RAY_EXEC` 可翻默认 |

每阶段以 env 开关灰度,任一阶段可停在"并存"状态回滚。

---

## 7. 不退化保障(硬约束)

- **Coordinator 契约**:`register_executor(kind, fn)` 与 `async __call__(ctx)->dict`
  接口不变;resume / task_registry / journal / `session_breakdown.json` schema 不变。
- **单节点默认行为**:`RAY_EXEC=0` 时严格回落现有本地 subprocess 路径(逐执行器可切)。
- **GPU 隔离不变量**:serving 与 GPU specialist 仍互斥(上层 lane 门 + 下层 Ray 资源
  双保险);production 服务端口 8888 由 specialist 独占禁用规则保留
  (`specialists/rebench.py::PRODUCTION_SERVING_PORT`)。
- **Ray lease 生命周期不变量**:任何 GPU server / benchmark / specialist 子进程都不得
  在其 Ray actor/task 结束后继续存活;cancel/timeout 必须先 kill 进程树再释放资源。
- **安全姿态**:单节点 Ray head 仍 `--dashboard-host=127.0.0.1`(沿用 ray_runtime.py),
  不暴露未认证 Jobs API。
- **host-local 监控**:robustness 探针仍在"执行实际发生的宿主"上采样,不因 Ray 化失明。

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 产物文件系统假设(最大) | P0 先做统一寻址 + #523 本地镜像思路;未过 P0 不进 P1 |
| Ray 变强依赖 / raylet 稳定性(#433/#432) | 直接复用 kernel 已加固的 `ensure_ray_cluster`;`RAY_EXEC` 开关兜底 |
| 短任务开销 | CPU/LLM 步骤明确不进 Ray;只 GPU/serving 走 Ray |
| detached server 绕过 Ray resource | P0/P1 强制 `ServingActor` 持有 server 生命周期;测试中扫描 orphan `sglang/vllm/Magpie` GPU 进程 |
| specialist 直接 Bash 触 GPU 绕过 Ray | P2 把整个 `needs_gpu` specialist subprocess 放进 `GpuSpecialistActor`,而不是只包装 rebench helper |
| lane 互斥在 Ray 里表达不全 | 双层设计:SQLite 门保留 + Ray 自定义资源冗余,不一步到位删门 |
| 自定义资源没注册导致 pending | P3 明确 `ray start --resources` / KubeRay pod 参数;非 GPU 节点不声明 `serving_slot` |
| Ray-in-Ray 嵌套(GEAK 已知痛点) | GPU specialist / kernel 任务在 Ray worker 内不再二次 `ray.init`;GEAK 保留现有单层 Ray 或在 actor 内走 direct worker 路径 |
| 爆炸半径 | 每阶段 env 灰度 + 逐执行器切换 + 保留旧路径,直到 P5 才删分支 |

---

## 9. 测试 / 验证

- **单元**:`_ray_backend` 用 Ray local_mode / 假 `ray.remote` 跑
  `run_subprocess` 契约;`num_gpus` 透传;`resources` 信号量映射。
- **生命周期不变量**:`ServingActor` 存活时 `ray status` 显示 GPU/`serving_slot`
  被占用;actor cancel/timeout 后 server 子进程树全部退出,无 orphan GPU 进程。
- **排队验证**:同时提交 serving benchmark 与 `GpuSpecialistActor` mock,后者必须 pending
  到 serving 释放 `serving_slot`/GPU 后才运行;验证不会共卡。
- **执行器回归**:`test_*baseline*` / `test_roofline_executor*` / `test_sweep_*` 在
  `RAY_EXEC=0/1` 双跑,断言结果与产物路径一致。
- **dispatcher**:`test_gpu_research_lane_dispatch` / `test_specialist_concurrent_dispatch`
  在 C/D 后仍绿(GPU specialist 并发/互斥不退化)。
- **多节点(mock)**:沿用现有 `test_multi_node_scripts.py` 风格,mock Ray 集群断言
  `ServingGroupManager` 使用 placement group/rank actors,不再依赖 detached server。
- **端到端 smoke**:单节点 1 卡 baseline+explore+roofline 全链路(`RAY_EXEC=1`)与关
  路径对照;覆盖门槛沿用 CI 90% line。

---

## 10. 待 owner 拍板的决策(附建议默认)

1. **lane 迁移深度**:双层(SQLite 门 + Ray 资源冗余)**[建议默认]** vs 一步迁到纯
   Ray 资源(删 SQLite lane,风险高)。
2. **单节点是否强制 Ray**:默认 `RAY_EXEC=0` 灰度、稳定后翻默认 **[建议默认]** vs 直接
   强制。
3. **`SpecialistGpuPool` 去留**:P2 后降级为记账、P5 后删 **[建议默认]** vs 立即删。
4. **产物共享根**:单节点用本地盘 + 多节点用现有 wekafs 命名 **[建议默认]** vs 统一
   引入 object store 暂存(更大改动)。
5. **多节点传输**:P4 用 `ServingGroupManager` + placement group 取代 detached
   `restart-server` 路径 **[建议默认]** vs 保留 REST 边界仅收敛单节点。
6. **Ray 术语**:对外统一称 **Ray-managed GPU execution** **[建议默认]**,避免把本计划
   误解为"所有步骤都用 Dashboard Jobs REST submit_job"。

---

## 10.1 决策已定(owner 拍板,覆盖上面的建议默认)

| # | 决策 | 影响 |
|---|---|---|
| 1 | **一步迁到纯 Ray 资源**;SQLite lane/lease **仅作 logs/记账**,不再是权威互斥门 | 权威互斥下沉到 Ray 自定义资源;`resource_lock.py` 从"门"降为"日志/记账视图" |
| 2 | **单节点强制 Ray**(不灰度) | `ray_exec_enabled()` 在单节点默认 **ON**;多节点默认 OFF(见 #4) |
| 3 | `SpecialistGpuPool` **先降级为记账、后删除**;记账统一用 SQLite | Ray `num_gpus` 是物理真相源;SQLite 仅留容量/TTL 视图 |
| 4 | **只做单节点**;多节点代码尽量不改,交由多节点维护者镜像本次改动 | 不动 `_multi_node_*` / `is_multi_node()` 多节点分支;单节点强制,多节点保持现状 |
| 5 | **先收敛单节点**;多节点 `ServingGroupManager`/placement group 延后 | P4 暂缓,单节点闭环优先 |
| 6 | 统一称 **Ray-managed GPU execution** | 术语一致 |

**据此对落地顺序(§6)的修订**:去掉"逐执行器灰度回滚"这一保护网(决策 2 强制),
但保留 `INFERENCE_OPTIMIZER_RAY_EXEC` 作为**紧急逃生阀**(默认按单/多节点自动取值);
删除双层 lane(决策 1),单节点直接用 Ray 资源;多节点路径原样保留(决策 4/5)。

---

## 11. 实现进度(Implementation Progress)

### ✅ P0 —— Ray 执行后端 + 硬不变量(已完成,已在真机验证)

- 新增 `orchestrator/actions/executors/_ray_backend.py`:
  `RayExecutionBackend.ensure()`(复用 kernel `ensure_ray_cluster`/`quiet_ray_init`)、
  `run_subprocess(num_gpus, resources)`(在 `ray.remote` worker 内跑现有
  `run_with_session_kill`;worker 保留 Ray 的 `*_VISIBLE_DEVICES`,caller env 只叠加非
  device 变量)、`resolve_shared_artifact_root()`、进程级单例。
- 新增 `orchestrator/actions/executors/_ray_serving.py`:
  `ManagedServerProcess`(新 POSIX 会话 + `PR_SET_PDEATHSIG` + 整树 reap)、
  `ServingActor`/`GpuSpecialistActor`(薄 `ray.remote` 封装,持有 `num_gpus`)。
- 单测 `tests/test_ray_backend_unit.py`(21 项):开关门、可见设备合并不变量、
  `run_subprocess` 契约(内联 fake ray)、真实子进程 reap 不变量。ruff/mypy 通过。
- 真机验收 `scripts/ray_exec_p0_smoke.py`(4 卡 Ray 集群):serving actor 占满 GPU →
  第二个 GPU actor **PENDING**(不共卡)→ kill serving actor 后子进程被 reap(无逃逸)→
  pending actor 随后调度。**全部不变量成立**。

### ✅ 决策落地(本次)

- `ray_exec_enabled()` 改为**单节点强制**语义(决策 2+4):env 未显式设置时,单节点
  返回 True、多节点返回 False;`INFERENCE_OPTIMIZER_RAY_EXEC=0/1` 仍可强制覆盖。
- 新增执行侧门 `_should_use_ray_backend()`:pytest 下默认走本地(除非显式
  `RAY_EXEC=1`),保证测试套件不依赖 Ray 集群;生产单节点强制。
- 落地正确的构建块:`RayExecutionBackend.run_subprocess_sync()`(供同步调用点)、
  `_grid_runner._num_gpus_for_config()`(从物化 YAML 读 TP 作 num_gpus)。

### ⚠ 关键设计发现(实现 T1 时确认,已据此修正 §12)

**按"每次 `_run_magpie` 调用"粒度包一个 Ray task 是错的**,只对**真正一次性**
benchmark 成立。任何 `server_lifecycle` / `auto_warmup` 路径(conc_sweep 单 server、
baseline 双跑、explore/sweep 预热轮)会让 Magpie **detached** 启动 server(独立
session + pid 文件),该 server 必须**跨多个 `run_grid` 调用**存活;而 per-call Ray
task 结束时 Ray 会**提前释放 GPU lease**,detached server 仍占卡 → 违反 §4.2 核心不
变量并造成 Ray 重复下发。**正确切法**:一个 Ray lease 必须覆盖"共享同一 server 的所有
轮次",即由 **`ServingActor` 持有 server 生命周期**(P0 已实现该 actor,尚未接线)。

同理 **ROCR 收敛(原 T2)不能在 materialize 期按 flag 静态剥离**:若该次执行最终回落
本地(lifecycle 情况),剥离后本地 server 就丢了 ROCR。ROCR 的剥离必须与"是否真的走
Ray 执行"在**执行期一致**(在 Ray 执行分支内改写 config,而非物化期全局剥离)。

> 因此本次**回退**了 per-call `_run_magpie` 路由与 materialize 期 ROCR 剥离,只保留
> 正确的构建块 + 单节点强制开关 + 本设计发现。下一步 T1 按 ServingActor 重做。

### ✅ P1 —— 单节点 serving-lease 化(T1+T2 完成,T3 待真机验收)

按 §11「关键设计发现」重做:**一个 Ray lease 覆盖"共享同一 server 的所有轮次"**,由
调用方(executor)持有并在 `finally` 中关闭,`run_grid`/`_run_magpie` 只是"用"它。

- **T1 机制**(commit `feat(ray-exec): P1 serving-lease mechanism …`):
  - `_ray_serving.py`:新增 `ServingActor.run_blocking()`(在 actor 的 lease 内跑一轮
    benchmark,返回 `(rc, out, err)`;硬超时用哨兵 returncode `_ACTOR_TIMEOUT_RC`
    回传,避免 Ray 无法跨 worker 边界重建 `TimeoutExpired`);`ServingLease`(持
    `num_gpus` 的 actor,`run_session_kill()` 对上层重抛 `TimeoutExpired`、把 worker
    侧 Ray 故障降级为 benchmark 失败而非崩会话,`close()` 幂等);`maybe_serving_lease()`
    ——调用方唯一入口,仅在**单节点 Ray 执行**时返回 lease,否则 `None`(本地路径)。
  - `_grid_runner.py`:`_run_magpie(serving_lease=…)` 有 lease 时经 actor 跑(用剥离后
    config),`run_grid(serving_lease=…)` 透传给每一轮(warmup/mn_warmup/measure)。
- **T1 接线**(commit `feat(ray-exec): route single-node serving executors …`):
  - `conc_sweep`:每 arm 一个 lease,覆盖 boot + 所有 CONC reuse 轮(及 Option B 每
    变体重启),与 lifecycle teardown 一起在 `finally` 关闭。
  - `baseline`(含 `profile` 子类):一个 lease 覆盖单轮或 double-run 的 warmup+measure。
  - `sweep`:一个 lease 覆盖整个 grid(每个变体 + 其 auto-warmup)。
  - `explore`:每变体一个 lease,覆盖 warmup + decision + stack-rebench(共享热 server);
    `_stack_rebench.measure_stack_rebench` 透传 lease 到 `run_grid`。
- **T2 ROCR 执行期收敛**:`_ray_backend.strip_visible_devices_from_config()` 只在**确实
  走 Ray 执行**的分支把 config `benchmark.envs` 里的 `ROCR/HIP/CUDA_VISIBLE_DEVICES`
  剥掉(Ray 已在 worker 设好),本地回落分支 config 原样。
- **硬不变量**:每个 lease 在其 `finally` 里"先 `teardown_lifecycle_server` 杀 server、
  再 `close()` 释放 lease",保证没有 GPU 进程活过它的 Ray lease(§4.2)。
- **不退化**:`serving_lease` 各处默认 `None` → 本地 `run_with_session_kill` 路径逐字
  不变;pytest 下 seam 默认关闭(除非 `INFERENCE_OPTIMIZER_RAY_EXEC=1`)。
- **测试**:`test_ray_backend_unit.py` 扩到 36 项(strip 幂等/剥离、`ServingLease`
  success/timeout-reraise/worker-error-degrade/close、`maybe_serving_lease` 门、
  `_run_magpie` 路由 + 本地路径对照,均用内联 fake ray);本地跑通 grid/baseline/
  conc_sweep/sweep/explore/roofline/recover/specialist 相关套件(共 ~900 项)全绿。
  本地无 `ruff`/`mypy`,交由 CI(T9)。

### ✅ P2 —— GPU specialist actor 化(T4+T5 完成)

复用 P1 的"executor 持 lease"范式:`needs_gpu` specialist 的**整个 subprocess**(agent
运行时 + 它 spawn 的任意 Bash/GPU 命令)进入 Ray `GpuSpecialistActor(num_gpus)`,Ray 设
可见设备,specialist 再也够不到 lease 外的卡(§4.2/§8)。specialist 子进程本就把 stdout 写
**文件**(`process.log`),reaper 靠 done.json/heartbeat/log mtime + 进程存活轮询(非 live
pipe),因此天然适配 P0 的 `ServingActor.start/is_alive/stop` 模型。

- **T4 机制**(commit `feat(ray-exec): P2 GpuSpecialistLease …`):
  - `_ray_serving.py`:新增 `GpuSpecialistLease`(封装 `make_gpu_specialist_actor`,暴露
    `start`/`pid`/`is_alive`/`exit_code`/`stop`/`close` 这套"长活+被轮询"接口,区别于
    `ServingLease` 的单次阻塞轮);`maybe_gpu_specialist_lease()`——dispatcher 唯一入口,
    单节点 Ray 执行且 `num_gpus>0` 才返回 lease,否则 `None`(本地路径)。
    `ManagedServerProcess` + actor 新增 `exit_code()`。
- **T4 接线**(commit `feat(ray-exec): route needs_gpu specialists through a Ray lease …`):
  - `dispatcher._spawn_fitting_queued`:SQLite `try_acquire`(现降级为容量/TTL 门)之后,
    Ray 路径再取 `GpuSpecialistLease(num_gpus=gpu_count)`,把 specialist 见到的**逻辑**
    `gpu_ids=range(N)`(Ray mask 下的 0..N-1)与 lease 一并放进 `extra_context`;lease 在
    `_run_dispatched_with_gpu_release` 的 `finally` 里 `close()`(绑定 task 生命周期,完成/
    异常/取消都释放)。`gpu_count` 解析(whole-machine vs serving-disjoint、bench TP floor)
    不变。
  - `subprocess_.run`:有 `gpu_lease` 时剥离 env 的 `*_VISIBLE_DEVICES`(Ray 在 worker 设)
    并经 `actor.start` 启动;reaper 轮询 Popen 形态的 `_RayLeaseProcess`(`poll` 走 actor
    `is_alive`/`exit_code`),`_kill` 经 actor reap 进程树。done.json/heartbeat/log 信号仍
    文件驱动、本地可读(单节点)。本地路径不变;`runner` 透传 `ctx.extra['gpu_specialist_lease']`。
- **T5 记账降级**:`gpu_pool.py` 文档化 `SpecialistGpuPool` 的新角色——Ray 下仅容量/TTL
  记账;`try_acquire`/`release` 保留(并发门 + wall-budget/TTL 视图 + 现有 dispatch 测试仍
  工作),但其 `gpu_ids` 不再是物理 pin(dispatcher 用逻辑 `range(N)` 覆盖)。
- **不退化**:Ray 路径外(多节点 / `RAY_EXEC` off / pytest)`gpu_lease=None` → SQLite
  gpu-id 设备路径逐字不变;`try_acquire` 仍被调用(`test_specialist_concurrent_dispatch`
  的 spy 断言、并发限流、`gpu_research_lane` 互斥均保留)。
- **测试**:`test_ray_backend_unit.py` +8(`exit_code` 锁存、`GpuSpecialistLease` 生命周期/
  未启动、`maybe_gpu_specialist_lease` 四门);`test_specialist_subprocess.py` +2(`run`
  经 lease 启动 + 剥离设备 env、`_kill` 对 `_RayLeaseProcess` 走 actor reap)。本地跑通
  specialist/dispatch/pool/coordinator 相关套件(concurrent_dispatch、gpu_research_lane、
  dispatch_params、auto_retry、resource_lanes、gpu_pool、framework_whole_machine、
  longrun_phase0、rebench 等,共 ~350 项)全绿;9796 项 collect 无 import 错。ruff/mypy 交 CI。

### ✅ P3 —— lane 互斥下沉到 Ray 自定义资源(T6+T7 完成,决策 1)

把**权威**物理 GPU 互斥从 SQLite lane 下沉到 Ray 自定义资源:整机 `serving_slot`(serving
族)+ `num_gpus`(GPU specialist),Ray 从物理上阻止共卡;SQLite lane 降为冗余的调度/可观测
视图(保留,不删)。

- **T6 注册 + 持槽**(commit `feat(ray-exec): P3 serving_slot custom resource …`):
  - `ray_runtime.py`:`ensure_ray_cluster` + `force_restart_local_cluster` 在**单节点本地 head**
    的 `ray start` 上声明 `--resources '{"serving_slot": 1}'`(经 `_resources_start_args()`)。
    放在**共享**启动器里,谁(kernel-agent / orchestrator)先起 head 谁声明,规避"资源未声明→
    actor 永久 PENDING"(§8);多节点连外部集群、不走此路径;该资源对 GEAK 无害。
  - `_ray_serving.py`:`ServingLease`/`maybe_serving_lease` 默认 `serving_slot=True`(每个 serving
    族调用者整机互斥,第二个 serving lease 在 `serving_slot` 上 PENDING 直到前者释放);
    `make_gpu_specialist_actor`/`GpuSpecialistLease`/`maybe_gpu_specialist_lease` 新增 `serving_slot`
    开关——serving-disjoint 池 `False`(仅 `num_gpus`,可跑在与 serving 不相交的卡上),
    whole-machine/bench-capable(`gpu_research_lane`)`True`(与 serving 互斥)。
- **T6 接线 + T7 降级**(commit `feat(ray-exec): dispatch gpu_research specialists onto serving_slot …`):
  - `dispatcher`:whole-machine/bench specialist(`uses_whole_machine_gpu_lane`)向
    `maybe_gpu_specialist_lease` 传 `serving_slot=True`,Ray 令其与 serving 在 `serving_slot`
    上互斥;serving-disjoint 池保持 `num_gpus` only。
  - **T7**:`resource_lock.py` + `dispatcher` 文档化——单节点 Ray 下**权威**互斥是 Ray 自定义
    资源(`serving_slot` + `num_gpus`),SQLite lane 不再是 GPU 互斥真相源;lane 门**保留**(不删)
    为便宜、支持 resume 的调度/可观测层(acquire/release/expiry 事件仍喂 lane timeline),两层冗余。
- **不退化**:Ray 路径外(多节点 / `RAY_EXEC` off / pytest)一切照旧——SQLite lane 仍是唯一门;
  `serving_slot` 仅在真机 Ray 集群声明,对现有 lane/dispatch 测试无影响。
- **测试**:`test_ray_fd_limit.py` +2(`ensure`/`restart` 的 `ray start` argv 含 serving_slot);
  `test_ray_backend_unit.py` +4(`_resources_start_args`、`maybe_serving_lease` 默认持槽、
  `maybe_gpu_specialist_lease` serving_slot 透传、`GpuSpecialistLease.start` 透传 serving_slot);
  本地跑通 ray_fd_limit / version_mismatch / specialist / dispatch / lane / framework_whole_machine /
  longrun / coordinator 套件(共 ~360 项)全绿;9802 项 collect 无 import 错。ruff/mypy 交 CI。
- **验收(§6 P3)**:并发压力下 serving/specialist 不共卡(serving_slot + num_gpus)、`ray status`
  可见 pending 资源需求、lane 门冗余生效——三项均满足;真机 `ray status` 对照并入 T3 类真机跑。

### 🚧 P4 —— 多节点 ServingGroupManager(**骨架已交付,默认关闭;决策 4/5 本轮不落地**)

决策 4/5 明确 P4(多节点)**不在本轮范围**("只做单节点;多节点代码尽量不改" / "P4 暂缓")。
因此本轮只交付**骨架构建块**(类比 P0 交付单节点 `ServingActor` 骨架),**默认关闭、未接线**,
活的多节点 serving 路径(detached `restart_server_for_round`,SSH / RayJob Dashboard)**逐字不变**。

- **骨架**(commit `feat(ray-exec): P4 ServingGroupManager skeleton …`):
  - `_ray_serving.py`:新增 `ServingGroupManager`(`start`/`pids`/`ranks_alive`/`is_alive`/
    `stop`/`close`,接口与 `ServingLease` 对齐),底层 `_make_serving_placement_group`
    (STRICT_SPREAD,每节点一个 `{GPU, [serving_slot]}` bundle)+ `_make_rank_actor`
    (`PlacementGroupSchedulingStrategy` 按 bundle 钉住);每 rank 一个 `ServingActor` 经
    `ManagedServerProcess` 跑 server-rank 子进程 → rank actor 持 GPU 至 stop、rank server
    随 actor 亡(无 detached 逃逸,§4.2)。`maybe_serving_group_manager()` 是唯一 opt-in 入口:
    仅当**多节点 AND** `INFERENCE_OPTIMIZER_RAY_MN_SERVING` 显式开启才返回 manager,否则 `None`
    ——默认什么都不跑。
- **明确留给多节点维护者**(§12 footer):分布式 sglang/vLLM 的 rank bootstrap(rendezvous /
  head-vs-worker / KV transport)、接线进 `restart_server_for_round`、`magpie_remote_env()`
  降级为放置策略——骨架只从调用方接收 rank 命令,不发明分布式启动。
- **不退化 / 不可验证**:纯新增、默认关;无 GPU 更无多节点 RayJob/KubeRay 集群,真机全 pipeline
  验收(§6 P4 门槛)无法在本轮执行,交由多节点维护者镜像 T1–T3 后一并跑。
- **测试**:`test_ray_backend_unit.py` +6(`ServingGroupManager` 生命周期:start 建 PG + 每
  bundle 一 rank、stop reap、close 杀 actor + 删 PG;start arity 校验;
  `maybe_serving_group_manager` 四门,均用 fake ray + 注入 PG/rank 工厂);9808 项 collect 无
  import 错。ruff/mypy 交 CI。

### ⏸ P5 —— 删旧路径(**本轮暂缓;owner 确认 defer:前置门槛未达 + 旧路径非死代码**)

结论:P5/T8 的破坏性删除**本轮不执行**——不是"漏做",而是**前置条件未满足 + 删除目标非死代码**,
现在删会破坏测试套件与逃生阀。owner 已确认暂缓,本段记录阻塞状态与解除条件。

- **门槛未达**:T8 明写前置"单节点闭环回归绿",即 T3 真机 `RAY_EXEC=1` baseline+conc_sweep
  验收。本轮全程无 GPU,T3 未跑(§11 P1 段 T3 仍"待真机")。Ray 路径未经真机端到端验证前删
  本地回落属冒险。
- **"旧本地路径"是负重代码,非死代码**:
  - pytest 下 `_should_use_ray_backend()` 返 False(`PYTEST_CURRENT_TEST` 门)、`maybe_*_lease`
    返 None → **全部 ~9808 项测试都走本地 subprocess 路径**;删掉即 CI 全红(CI 无 Ray 集群)。
  - `RAY_EXEC=0` 是**保留的紧急逃生阀**(决策 2 修订)→ 本地路径。
  - SQLite `SpecialistGpuPool` 物理 id 分配 + 手工 ROCR 掩码(`_grid_server_args` /
    `_workload_envs` / `subprocess_.py`)仍被本地路径 + 多节点回落(决策 4)使用。
  - P1–P4 刻意保留全部本地回落(逃生阀 + hermetic 测试),因此当前**没有实质的单节点死代码**
    可安全删除;`SpecialistGpuPool` 已在 T5 降级为记账(彻底删亦待此)。
- **解除条件(Unblock)**:①T3 真机 `RAY_EXEC=1` 全 pipeline 回归绿;②另行决策**退役
  `RAY_EXEC=0` 逃生阀**(现由决策 2 修订保留)并把测试套件改造成可在 Ray 下运行(mock Ray 或
  要求集群)。二者满足后方可删单节点本地 GPU 分配/ROCR 掩码 + `SpecialistGpuPool`,兑现 §5 净
  收敛(多节点分支保留)。
- 本轮 P5 交付:仅**记录**上述阻塞状态与解除条件(本段 + §12 T8),不动代码。

### ⏭ 进行中 / 待办 —— 见 §12

---

## 12. Todos(修订后,单节点优先、纯 Ray、强制)

- [x] **T1 (P1, 已修正,已完成)** 单节点 serving/benchmark 走 **`ServingLease` 持 lease
  跨轮**:一个 Ray lease(`num_gpus=tp`)覆盖"共享同一 server 的所有 `run_grid`/warmup/
  measure 轮次";lease 由 executor 持有,`finally` 里先 teardown server 再 `close()`
  释放,禁止 detached GPU 进程活过其 lease。`run_grid` 的 warmup+measure、conc_sweep
  单 server 每 arm、baseline double-run、sweep 整 grid、explore warmup+decision+rebench
  均映射到"一个 lease"。仅 `not is_multi_node()`(`maybe_serving_lease` 内门控)。
- [x] **T2 (P1, 已修正,已完成)** ROCR 执行期收敛:`strip_visible_devices_from_config()`
  仅在**确实走 Ray 执行**的分支内,把交给 Magpie 的 config 的 `ROCR/HIP/CUDA_VISIBLE_DEVICES`
  去掉(Ray 已在 worker 设好),本地回落分支保持 config 不变。真机 server 只见 Ray 卡的
  校验并入 T3。
- [ ] **T3 (P1 验收/§4.6,待真机)** 单节点 `RAY_EXEC=1` 跑通一次 baseline+conc_sweep,
  产物路径与关路径 byte 级一致(或差异可解释),且 `ray status` 显示 server 存活期 GPU
  被占、结束后释放,无 orphan server 进程。复用 `scripts/test_conc_sweep_flow.py` 加
  `RAY_EXEC=1` 对照。**代码已就绪**;需在有 GPU 的机器上执行(本轮无 GPU,未跑)。
- [x] **T4 (P2, 已完成)** `needs_gpu` specialist 整体进 `GpuSpecialistActor(num_gpus=...)`:
  `dispatcher._spawn_fitting_queued` 的 `needs_gpu` 分支在 Ray 路径提交 `GpuSpecialistLease`
  (整个 subprocess 经 `actor.start` 跑在 lease 内,Ray 设可见设备),`try_acquire` 保留为
  容量/TTL 门(不再是物理分配真相);`extra_context["gpu_ids"]` 收敛为 specialist 在 Ray
  mask 下实际所见的逻辑 `range(N)`;lease 在 task `finally` 里 `close()`。仅单节点 Ray。
  真机验证(specialist 的 Bash GPU 命令落在 Ray 卡内、TTL/cancel kill 子进程)并入 T3 类
  真机跑。
- [x] **T5 (P2, 已完成)** `SpecialistGpuPool` 降级为 SQLite 记账(容量/TTL 视图 + 并发门 +
  prompt 展示),`try_acquire`/`release` 保留(现有 dispatch 测试仍绿);物理分配交给 Ray
  `num_gpus`,其返回的 `gpu_ids` 不再是设备 pin(Ray 路径由 dispatcher 用逻辑 `range(N)`
  覆盖)。彻底删除留待 P5。
- [x] **T6 (P3, 决策 1,已完成)** 注册 `serving_slot` 自定义资源:`ensure_ray_cluster` /
  `force_restart_local_cluster` 在单节点本地 head 的 `ray start` 上声明 `--resources
  '{"serving_slot":1}'`;serving/benchmark/profile/gpu_research 类 lease 持
  `serving_slot=1`(`ServingLease`/`maybe_serving_lease` 默认 True、whole-machine specialist
  传 True),serving-disjoint GPU specialist 用 `num_gpus` 不占 slot。仅单节点本地 head 受影响
  (多节点连外部集群、不走此路径)。真机 `ray status` 对照并入 T3 类真机跑。
- [x] **T7 (P3, 决策 1,已完成)** `resource_lock.py` + `dispatcher` 从权威门降为 logs/记账
  (文档化):单节点 Ray 下权威互斥是 Ray 自定义资源;dispatcher 不再以 SQLite lane 作为 GPU
  互斥真相源。lane 门**保留**(便宜、支持 resume、可观测),acquire/release/expiry 事件仍写入
  喂 lane timeline;两层冗余。彻底删 lane 门待 P5(避免动 resume/大量测试)。
- [ ] **T8 (P5, 暂缓 — owner 确认 defer;门槛未达)** 前置"单节点闭环回归绿"(T3 真机
  `RAY_EXEC=1`)**未跑**;且"单节点旧本地 GPU 分配/手工 ROCR 掩码"当前是**逃生阀
  (`RAY_EXEC=0`)+ hermetic 测试(pytest 走本地)+ 多节点回落**的负重路径,**非死代码**——现在删
  即 CI 全红 + 失去逃生阀。**解除条件**:①T3 真机全 pipeline 回归绿;②另行决策退役 `RAY_EXEC=0`
  逃生阀 + 测试套件可在 Ray 下运行。二者达成后再删单节点本地 GPU 分配/ROCR 掩码 + `SpecialistGpuPool`,
  兑现 §5 净收敛(多节点分支保留)。详见 §11 P5 段。
- [ ] **T9** 每步:ruff + mypy + 相关单测;涉及 GPU 的用真机 smoke/flow 脚本验收。

> 多节点(原 P4/§4.3 收敛、`ServingGroupManager`、`magpie_remote_env` 降级)按决策 4/5
> **不在本轮范围**;保留现状,交由多节点维护者镜像 T1–T3 的单节点改动。
>
> **P4 更新**:已交付 `ServingGroupManager` **骨架**(`_ray_serving.py`,默认关闭,
> `INFERENCE_OPTIMIZER_RAY_MN_SERVING` opt-in;详见 §11 P4 段)作为多节点接线的基座——
> placement group + per-node rank actors 的生命周期已就位,活的 detached
> `restart_server_for_round` 路径逐字不变。剩余接线(分布式 rank bootstrap、替换
> `restart_server_for_round`、`magpie_remote_env` 降级)+ 多节点真机验收仍待多节点维护者完成。
