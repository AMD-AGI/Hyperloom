<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Specialist 控制面审查（第二轮）：GPU pool 重建、端口/kill 门、inprocess 回退、patch provenance、proposal 上限

> 承接 [specialist.discuss2.md](specialist.discuss2.md)。本轮回答 6 个问题，目标同前：把确定性 gate 中属于"口味"和"死代码"的部分交还给模型智能，只保留模型结构上无法承担的部分。
>
> 方法：工作流 `specialist-freedom-audit-2`，4 个分片源码审查 agent + 4 个对抗式复核 agent，8/8 完成，0 失败，814 次工具调用，约 127 万 token，约 54 分钟。**agent 之间在若干关键点互相反驳，所有决定性事实由我本人重新读码裁定**，下文标注了裁定依据。
>
> 代码基线：分支 `feat/zgong/explore-opt-11`，2026-07-29。

---

## 0. 本轮总结论

| # | 问题 | 结论 |
|---|---|---|
| 1 | carve-off 在交错派发下是否无用 | **半对，且错的那半是承重的**。互斥确实由 lane 表提供，但 lane 是**租约作用域而非进程作用域**；且默认配置下 carve-off 计算结果是**空池**，等于意外关停了非 bench GPU specialist |
| 2 | 两个端口门是否重复 | **不重复，但重复的判断依据错了**。真正的分界是 GPU specialist（被 lane 互斥保护）vs **CPU specialist（`research_lane` 不与任何 lane 冲突，可与在跑的 benchmark 并发）** |
| 3 | Global kill prohibition 是否生效 | **零执行力，但应当保留**。它守的是 CPU specialist 并发路径，且只值约 40 token |
| 4 | inprocess 回退何时触发 | **正常流程确实不触发，但触发条件真实存在且它是危险的静默降级** |
| 5 | patch provenance 为何需要 / 范围差异 | **需要，且 enablement 与 explore 走同一条路径**；真正的差异在 grounding base 选错树 |
| 6 | proposal 上限是否生效 | **完全没有生效，纯 prompt 文本**。可以删，改成 prompt 里写 6 |

**贯穿本轮的一个判据（与上一轮相同）**：失败发生在模型被调度的窗口内还是窗口外。但本轮发现该判据本身需要修正——见 §1.6。

---

## 1. Serving-GPU carve-off 与 lane 互斥是否重复

### 1.1 完整 lane 冲突矩阵

源：`resource_lock.py:56-79`，容量 `bus/storage/schema.py:29-38`。

| Lane | 默认容量 | 声明的冲突 | `_expand_lanes([lane])` 实际共同获取 |
|---|---|---|---|
| `server_lifecycle` | 1 | `benchmark_lane, profile_lane, gpu_research_lane` | 服务四元组 |
| `benchmark_lane` | 1 | `profile_lane, server_lifecycle, gpu_research_lane` | 同 |
| `profile_lane` | 1 | `benchmark_lane, server_lifecycle, gpu_research_lane` | 同 |
| `gpu_research_lane` | 1 | `benchmark_lane, profile_lane, server_lifecycle` | 同 |
| `research_lane` | 1 → 启动时改写为 `2×GPU` | **`frozenset()`** | `{research_lane}` |
| `workspace_mutation` | 1 | `frozenset()` | `{workspace_mutation}` |
| `build_lane` | 1 | `frozenset()` | `{build_lane}` |

**关键语义**：`_expand_lanes`（`resource_lock.py:85-104`）不是"检查这些空闲"，而是 `acquire_many` **实际插入行**的集合。所以四条服务系 lane 是**一个不可分的 capacity-1 令牌**。

推论：`profile.yaml` / `roofline.yaml` / `deep_kernel_analysis.yaml` 只声明 `profile_lane`，实际静默取走 `server_lifecycle + benchmark_lane + gpu_research_lane`。**YAML 里的 lane 声明差异是装饰性的，运行时完全等价。**

### 1.2 各 action 的 lane 需求

| 声明 | actions | 展开后 |
|---|---|---|
| `server_lifecycle, benchmark_lane` | `baseline`, `explore`, `sweep`, `conc_sweep`, `replay_warm_recipe` | 服务四元组 |
| `+ workspace_mutation` | `integrate`, `integrate_patch`, `kernel_opt`, `gemm_tuning`, `operator_tuning`, `vendor_kernel_config`, `framework_agent` | 服务四元组 + `workspace_mutation` |
| `profile_lane` | `profile`, `roofline`, `deep_kernel_analysis` | 服务四元组 |
| `research_lane` | **`specialist`** | 仅 `{research_lane}` |
| `[]` | `report`, `session_breakdown`, `target_analysis` | 不取租约 |

**没有任何 YAML 声明 `gpu_research_lane`**，它只在运行时注入三处：`intent_router.py:449`、`framework.py:1052`、`explore.py:768`。

### 1.3 用户论点的对与错

**对的部分**：在单次派发代际内，服务四元组塌缩成一个全局 capacity-1 互斥，GPU specialist 与持 lane 的服务动作确实互斥，carve-off 在**这里**不增加任何东西。

**错的部分（承重）**：互斥是**租约作用域，不是进程作用域**。

- **没有任何地方续租**：`SqliteLeaseBackend.heartbeat`（`resource_lock.py:312-332`）与 `ResourceLockManager.heartbeat`（`:583-593`）**零生产调用者**（仅两个测试）。TTL 在 acquire 时固定。
- TTL 与实际运行时长对不上：`explore.yaml` TTL 7200s，但单个 variant 的硬上限是 `DEFAULT_EXPLORE_TIMEOUT_CEILING_SEC = 14400`，一轮跑 N 个 variant；`baseline.yaml` 4200s，而光是 boot 的 `SERVER_READY_TIMEOUT_SEC` 就是 2700s × 两轮；`conc_sweep.yaml` 9000s。
- `reap_dead_holders` 救不了：holder PID 是 **coordinator 自己的**（`resource_lock.py:113` `os.getpid()`），永远活着。

**存在两条结构性路径，服务进程活着但不持有任何 lane：**

1. **多节点，按设计如此**。`_multi_node_server_lifecycle.py:174-176` 明确禁用 `server_lifecycle` 复用（"multi-node is local-only"）；服务器在 pod 上跨任务持续存活，`restart_server_for_round` 有 resume 快路径（配置匹配且 pod 存活则完全跳过 kill+launch）。
2. **内联 kernel handler**。`kernel_opt`/`integrate`/`gemm_tuning` 等**从不获取它们声明的 lane**——它们作为 REQUEST intent 在 `intent_router.py:600-690` 内联处理，完全在 task/lease 路径之外。而 `integrate_handler` 会构造 `Task(kind="baseline")` 并以 `RunnerContext(lease=None)` 直接调 `BaselineExecutor`（`request_handlers.py:6020-6038`）——**一个不持任何 lane 的真实服务进程**。

### 1.4 一处需要裁定的 agent 分歧

复核 agent 提出 C1/C2：pump 排干到空才返回，所以 `reclaim_expired_running` 的三个调用点（`dispatcher.py:154` pump 入口、`maintenance.py:63` pump 之后、`explore.py:389` phase hook）**都不会在协程存活时运行**，因此"task 行被标 failed 而执行器还在跑"不可达。

**我核实后的裁定：C2 对一半。**

- 对 `reclaim_expired_running` 而言，C2 成立——我核了 `dispatcher.py:130-172`，三个 reclaim 确实都在 `inflight = []`（`:167`）和 `while True`（`:172`）之前，不是每次 poll 都跑。
- **但 lane 过期不走这条路。** `acquire_many` 内部有**机会性过期清理**（`resource_lock.py:232-256`：先收集 `expired`，再 `DELETE FROM leases` 并发 `lease_expired` 事件），而 `_spawn_fitting_queued` 在 `while True` 循环里**每一轮都调用**。所以租约行可以在 pump 中途、执行器仍在运行时被删除，`lane_holders()` 过滤 `expires_at > now`（`:481-485`）随即报告空闲。

C2 的"不可达"只覆盖 task 行状态机，不覆盖 lane 表本身。**TTL 洞是真的。**

### 1.5 默认路径上 carve-off 实际计算出什么

```python
# gpu_pool.py:145-153
if explicit: return explicit[:cap]              # 算子池，不做 carve
serving = max(0, int(serving_tp or 0))
mask_ids, mask_present = _visible_device_mask()
if mask_present: return mask_ids[serving:][:cap]
return list(range(cap))[serving:]
```

默认：`capacity = detect_gpu_count() = 8`，`serving_tp = shared_state.tp = --tp = 8`。无 mask → `list(range(8))[8:]` == **`[]`**。有 mask 也是 `[]`。

**所以在 TP == GPU 数这个规范场景下，serving-disjoint 池是空的**，`SpecialistGpuPool.capacity == 0`，PolicyGate 以 `specialist_gpu_request_exceeds_capacity` 预先拒绝。**非 bench 的 GPU specialist 在默认路径上被无条件禁用。** `specialists/profile.py:151-155` 的 docstring 公开承认了这一点并把 bench/framework 路由绕开。

**这不是安全控制，是意外的功能关停。**

而且 carve-off 在默认路径上**对准入完全不起作用**：`_should_use_ray_backend()` 单机返回 True，走 `try_acquire_ray_observation`（`gpu_pool.py:295-367`），该函数**完全忽略 `self.gpu_ids` 和 capacity**，发放 `[100000, 100000+pending_limit)` 的合成 slot；`extra_context["gpu_ids"]` 随后又被 `dispatcher.py:531-533` 覆写为 `list(range(gpu_count))`。

**净结果：PolicyGate 用一个数字拒绝，dispatcher 用另一套逻辑准入。两者不一致。**

carve-off 本身还有两个缺陷：
1. **证据是配置值而非事实**。`_resolve_serving_tp`（`dispatcher.py:656-675`）读 `shared_state.tp`，回退 `$TP`，默认 0。无 PID 扫描、无 `rocm-smi`、无 pidfile。它假设服务占据**前** `serving_tp` 个 id——纯约定。
2. **一次算好，永不重建**。池在 `Coordinator.__init__`（`coordinator.py:605-611`）构造，TP 中途变化（framework agent、不同 `--tp` 的 explore variant）后 carve 就是陈旧的。

### 1.6 判据修正

上一轮的判据是"失败是否发生在模型不被调度的窗口"。本轮发现该判据需要补一条：

**默认单机路径上，物理互斥的真正持有者是 Ray（`num_gpus` + `serving_slot`），不是 SQLite lane 表。** Ray actor 的生命周期绑定执行器（`_ray_serving.py:390-400`，`finally` 中 `close()`），**是进程作用域而非时钟作用域**。所以：

- **Ray 开启（默认单机）**：TTL 洞导致的是**排队**，不是共卡——lane 空了但 Ray 调度不出 GPU。
- **Ray 关闭**（`INFERENCE_OPTIMIZER_RAY_EXEC=0`、多节点、pytest）：SQLite lane 是唯一互斥，TTL 洞变成**真正的双花**。

多节点下唯一救场的是：`gpu_specialist_ceiling` 来自 launch host 的 `detect_gpu_count()`，无 GPU 的控制节点得到 capacity 0，于是 PolicyGate 拒绝。**这是部署拓扑的巧合，不是不变量。**

### 1.7 重建 GPU pool 时必须存活的最小不变量集

| # | 不变量 | 当前机制 | 缺失后果 |
|---|---|---|---|
| **I1** | 服务系互斥是**单个** capacity-1 令牌，不是四条耦合 lane | `LANE_CONFLICTS` + `_expand_lanes` | 不会新坏，但保留四个名字会诱使人相信 `profile_lane` 是独立的。应塌缩成一个 `gpu_exclusive` 令牌，其余降为纯观测标签 |
| **I2** | 互斥令牌的寿命必须绑定 **GPU 进程**，不是 TTL | **缺失**（`heartbeat` 零调用者） | Ray 关闭时是真双花；Ray 开启时是排队。要么执行器续租，要么把令牌 key 在 server pidfile 上 |
| **I3** | 释放是幂等且崩溃安全的，且 liveness 主体可证明已死 | `reap_dead_holders` 按 `pid`——但 pid 是 coordinator 的，永远活着 | 没有正确的 liveness 主体，I2 就没有安全兜底 |
| **I4** | **每条**碰 GPU 的路径都获取令牌，**包括内联 kernel handler** | 违反：`request_handlers.py:6020-6038` 以 `lease=None` 启动 baseline server | 今天只靠 tick 顺序（pump 排干语义）保护，Ray 关闭时无保护 |
| **I5** | 卡级不相交要有**事实**证据，不是配置猜测 | `_resolve_serving_tp` 读 `shared_state.tp` / `$TP` | TP 陈旧就静默把服务卡发给 specialist。应从真实 mask/pidfile 推导，且**每次准入重算**而非 `__init__` 一次 |
| **I6** | **一个**准入决策，不是两个互相矛盾的 | PolicyGate 用 carve 后的池大小，dispatcher 走 Ray 忽略它 | 拒绝与准入不匹配；carve 看起来承重实则不然 |
| **I7** | 多节点持久服务器纳入同一套核算 | 结构性在外 | 今天只靠无 GPU 控制节点把 `detect_gpu_count()` 归零。任何带 GPU 的控制节点即破 |
| **I8** | `research_lane` 多持有者语义（cap>1，`PRIMARY KEY (lane, holder_id)`） | `schema.py:44-53` | specialist 扇出退化为串行。**这是当前设计里唯一无争议正确的部分** |
| **I9** | 从持久化行做 resume 对账 | `leases` + `gpu_leases` 表，boot 时 `reap_orphaned_servers` | resumed session 无法判断卡是否被占 |
| **I10** | `PR_SET_PDEATHSIG` + `start_new_session` 在每个 GPU 子进程上 | `_ray_serving.py:51-63`、`_subprocess_kill.py:34-43` | 唯一阻止 detached server 活过租约、把 VRAM 带进下一轮的机制 |

**重建结论**：SQLite lane 表**可以整个去掉**——`resource_lock.py:15-24` 自己就写着 Ray 的 `num_gpus`/`serving_slot` 在默认路径上已提供物理互斥。**不能去掉的是 I2 + I3 + I4**：一个寿命绑定服务进程而非时钟的互斥令牌，由可证明已死的主体释放，且被**每一条** GPU 路径获取（含内联 kernel handler）。

carve-off（I5）只有在**每次准入按真实证据重算**时才值得保留；作为 `__init__` 时冻结的 `range(cap)[tp:]`，它要么是 no-op，要么是把非 bench GPU lane 整个关停，**两者都不是 docstring 声称的东西**。

---

## 2. 两个端口门是否重复

### 2.1 rebench helper 的真实地位

`specialists/rebench.py` 是一个**具名 python 函数，不是 MCP 工具**。specialist 要用它必须 `python -c "from hyperloom.orchestrator.specialists.rebench import ..."`。它**没有任何生产调用者**。

其中的端口逻辑：
- `PRODUCTION_SERVING_PORT = 8888`，`_resolve_port` 拒绝显式 8888 并改选。
- `_pick_free_port` 的重试循环和 `18888` 兜底是**死代码**——`/proc/sys/net/ipv4/ip_local_port_range` 是 `32768 60999`，第一次就不可能撞上 8888。

**所以门 2（rebench 端口拒绝）只在"specialist 显式给一个零生产调用者的 helper 传 `--port 8888`"时触发。它可以删。**

### 2.2 但删之前必须先修 rebench 本身（阻断项）

复核发现一个比端口严重得多的问题：`rebench.py:171-183` 调 `run_grid` 时 `preclean_before_run` 默认 `True` → `_grid_runner.py:905` 执行 `_kill_stale_servers()`。而它：

- SIGKILL 匹配到的服务进程
- **`unlink /dev/shm/{vllm,nccl,cuda,torch,atom}*`**（`_grid_runner.py:827-840`）——这会破坏**正在运行的**服务器的 NCCL/shm 段，不只是陈旧的
- `my_pgid` 排除（`:760-763`）对 specialist 无效，因为 specialist 是 `start_new_session=True`（`subprocess_.py:553`），pgid 不同

**删掉端口守卫而留着这个，严格劣于什么都不做。**

### 2.3 端口 8888 是否真的在用（一处 agent 内部反驳，我采信反驳方）

第一份报告称"`_server_lifecycle.py:197` 之后单机端口是临时分配的，8888 已经过时"。复核方用代码推翻：

`resolve_lifecycle_params` 在到达 `_assign_free_port` **之前**就返回的情形有四种：多节点（`:171-175`）、scriptable 框架（`:161-168`）、非内建 `benchmark_script`（`:177-180`）、**`torch_profiler` 启用**（`:182-185`）。这些情形下 `info["port"]` 保持 `REUSE_PORT_DEFAULT = 8888`，且 `envs["PORT"]` 根本不写入，benchmark 脚本走 `PORT=${PORT:-8888}`。

**所以对每一次 `profile` / `roofline`（profiler 恒不 eligible）、每一次非内建脚本运行，服务端口就是字面的 8888。** 我采信这一裁定。

### 2.4 真正的分界不是"两个门重复"，而是 GPU vs CPU specialist

这是本题最重要的发现，两份报告都在第一轮漏掉了：

- **GPU specialist**：`needs_gpu` 触发 `intent_router.py:449` 注入 `gpu_research_lane`，与服务四元组互斥。**它在时间上不可能与 benchmark 重叠，端口条款对它是冗余的。**
- **CPU specialist**：`_meta/specialist.yaml` 只要 `research_lane`，而 **`LANE_CONFLICTS["research_lane"] = frozenset()`**（`resource_lock.py:74`）——不与任何 lane 冲突。**一个 `needs_gpu=false` 的 specialist 与正在跑的 benchmark 完全并发**，拥有不受限的 `Bash`，正是能绑 8888 或 `pkill -f sglang` 打断测量的那个角色。

**结论**：

| 控制 | 判定 |
|---|---|
| rebench 端口拒绝（`rebench.py:41-83`） | **删**（但先修 §2.2 的 preclean） |
| Iron Rule 1 端口条款 **GPU 分支**（`specialist_prompt_builder.py:1890-1897`） | **删**——与 `gpu_research_lane` 互斥冗余，前提是重建时保留该互斥 |
| Iron Rule 1 端口条款 **无 GPU 分支**（`:1900-1906`） | **保留**——`research_lane` 不冲突任何 lane，这是该路径上唯一的控制，且失败（污染在飞测量）对模型不可见 |

---

## 3. Global kill prohibition 是否生效

**执行力：零。** `BASH_KILL_SAFETY_PREAMBLE`（`specialist_prompt_builder.py:44-49`）+ `leaf.py:33` + Iron Rule 8 全是 prompt 文本。无 PreToolUse hook、无 bash wrapper、无 seccomp、无 PID namespace 隔离（grep 确认）。

**但应当保留。** 理由不是"防御纵深"这种空话，而是：

- 它守的是 §2.4 那条 CPU specialist 并发路径——一个并发 specialist 的 `pkill` 会破坏一次它**自己观察不到、也无法修复**的测量。
- 失败属于类别 (b)：发生在另一个 agent 的测量窗口内，受害者不在场。
- 成本约 40 token。

**这是本次审查里少数几个"无执行力但仍应保留"的条目**——判据是它守的失败对受害方不可见，而不是它有多强。

---

## 4. Subprocess-to-inprocess fallback 的实际触发场景

### 4.1 判定条件

唯一决策点，`cli/executors.py:138-146`：

```python
claude_bin = shutil.which("claude") or ""
use_subprocess = dispatch_mode != "inprocess" and bool(claude_bin)
if dispatch_mode == "subprocess" and not claude_bin:  # -> log.warning
```

默认 `dispatch_mode = "subprocess"`（`parser.py:1028-1038`）。所以**回退只在 `which("claude")` 为空时触发**。

### 4.2 正常流程确实不触发，但触发条件真实存在

`assets/install.sh` 自己不装 CLI，它链到 `agents/kernel/scripts/install.sh` 的 `ensure_forge_claude_cli`。**那里每一个失败分支都是 `warn` 而非 `die`**：无 npm 且无 apt-get → `warn; return 0`；NodeSource 失败 → `warn; return 0`；nodejs apt 安装失败 → `warn`。而且 `chain_kernel_agent` 在脚本缺失时返回 0，整条链可用 `--skip-kernel-agent` 跳过。

**所以 IR-2「install.sh 成功」并不蕴含 `claude` 在 PATH 上。** 真实场景：无 npm 的气隙 pod、`--skip-kernel-agent`、只在 head 装了 CLI 的多节点 RayJob worker、未 source `runtime/kernel-agent.env.sh`。

### 4.3 能力差异（两份报告有三处冲突，我逐条裁定）

**in-process 丢失的：**

1. **git worktree 隔离**——`_maybe_setup_worktree` 在 `subprocess_config is None` 时直接返回 `(None, None, "")`（`runner.py:1372-1373`）。
2. **文件系统作用域**——无 `--add-dir`，无 `cwd=worktree`；SDK 继承**编排器进程的 cwd**。
3. **env 清洗**——`_build_specialist_env`（允许列表 + `scrub_child_process_env` 去除 `LD_PRELOAD` 类 hook）是 subprocess 专属。
4. **GPU 掩码**——这是最能污染测量的一条。`subprocess_.py:474-477` 设置 `HIP/CUDA/ROCR_VISIBLE_DEVICES`，且 `else` 分支为 CPU specialist **弹出**它们（"CPU specialists must not inherit serving GPU visibility"）。in-process 下 `gpu_ids` 只进 prompt 和 done 载荷。**复核方指出更尖锐的形式：受害者是 CPU specialist——它继承编排器的完整设备可见性，可与在跑的服务进程共卡，正是 carve-off 存在的理由。我采信。**
5. **进程组 kill / 孤儿回收**——无 `start_new_session` + `killpg`，specialist 的 Bash 派生的服务进程会泄漏。
6. **心跳看门狗 + 硬墙钟 + done 文件宽限期**——`_reap_loop` 全部缺席。
7. **patch grounding**——`base_checkout = prep.worktree_base or prep.worktree`（`runner.py:1304`）两者皆 `None` → `ground_patch_text` 短路为 `GROUND_UNCHECKED`，**`git apply --check` 与 missing-target 幻觉守卫完全跳过**。

**三处需要裁定的冲突：**

| 争点 | 第一份报告 | 复核 | 我的裁定 |
|---|---|---|---|
| `--allowedTools` 是否丢失 | 丢失，SDK 的 `allowed_tools` 只是自动批准列表 | **不丢失**——SDK transport 在 `subprocess_cli.py:493-494` 发出**同一个** `--allowedTools` CLI flag | **采信复核**。两条路径发相同的 flag |
| `--permission-mode` 变化的方向 | 变成 `default` 交互模式，无人应答会挂 | **in-process 更严格**——无 `can_use_tool` 回调时 SDK 在 `query.py:432-433` 抛错并回复确定性拒绝，不挂 | **采信复核**。方向是反的：in-process 拒绝了 subprocess 会 bypass 的东西 |
| CLI 版本 pin 被绕过 | 是 fallback 特有的讽刺 | **是每一次 `ClaudeBackend` 调用的固有属性**——`_find_cli` 无条件优先 bundled binary（`subprocess_cli.py:151-155`），包括 orchestration 角色 | **采信复核**，且这是个独立问题，与 fallback 无关 |

**in-process 获得的**：`emit_intent` 与真实 `Intent` 对象、逐 turn 的 tool-violation 检测、真实 `turns_used`、SDK 级重试/退避。

### 4.4 最严重的一条：in-process 没有墙钟

`grep wall_budget_sec runner.py` → 只有 622（进 prompt 文本）、1042/1043/1058（subprocess 侧）。**`_run_via_backend` 的 turn 循环（`runner.py:869-985`）没有任何 `time.monotonic()` deadline、没有 `asyncio.wait_for`。**

`DEFAULT_SPECIALIST_MAX_TURNS = 1000`，且每次 `backend.run(max_turns=1)` 被 `claude.py:295` 静默抬到 `max(1, 8)` → **最多 1000 × 8 = 8000 个 agentic turn**。而 `ClaudeBackend` 以 `conversational=False` 构造，每轮从头重发完整 prompt，模型跨 turn 无记忆——**结构上倾向于永不发出 `specialist_done` 而一路循环到 1000**。

复核指出了比 token 浪费更尖锐的后果：`kill ≤ gpu_lease TTL ≤ gpu_research_lane TTL` 这条铁律在这条路径上**结构性不可满足**，因为左项不存在。TTL = `wall_budget × 1.1`，到点后 maintenance reaper 删掉租约行，而协程还在跑 GPU 工作，下一个 `try_acquire` 就把卡发给第二个 specialist。**Ray 关闭时这是真双花。**

### 4.5 一条被送给模型的假话

`_prepare` 是两条路径共享的，所以墙钟 prompt 段在 in-process 下照样渲染，包括字面这句：*"The Coordinator hard-kills your subprocess when this budget is exceeded"*（`specialist_prompt_builder.py:1105`）。**在 in-process 路径上这句是假的，且根本没有 subprocess。** 而 dispatch mode 在 prompt、breakdown、manifest、state.json 里**一处都没渲染**（grep 零命中）。

### 4.6 判定：危险的静默降级，不是安全网

它在三个意义上都不 fail safe：

1. **在使用点静默降级**——只有构造时一条 `log.warning`，发生在任何 specialist 运行之前。manifest、`state.json`、specialist workspace、`session_breakdown.json`、`SpecialistRunResult.notes` 全都没有记录。对比之下 `worktree_setup_failed` **是**被记进 notes 的（`runner.py:562-563`）。
2. **丢掉的正是 subprocess 路径存在的全部理由**——同时失去 worktree、`--add-dir` 围栏、cwd 约束、env 清洗、GPU 掩码、进程组 kill、心跳看门狗、墙钟、patch grounding。这不是"同一件事少点遥测"，是完全不同的信任边界。
3. **无界 turn 循环是定时炸弹**，且触发条件（缺 `claude`）恰好与配置退化的 pod 相关——正是算子最不可能盯着的时候。

**建议**：把 `dispatch_mode == "subprocess"` 且缺 `claude` 改成 `_preflight` 硬错误，`inprocess` 保留为显式 opt-in 测试 flag（测试已经显式传了）。若必须保留回退，至少 (a) 在 `runner.py:869` 加 `wall_budget_sec` deadline，(b) 往 notes 里加一条永久 `dispatch_mode:inprocess_no_isolation`。

---

## 5. Output/patch provenance 与 path 的适用范围

### 5.1 各项检查的分类

`vet_patches()`（`patch_safety.py:439-471`），由 `_finalize`（`runner.py:1308-1311`）调用一次：

| 检查 | 拒绝什么 | 效果 | 类别 |
|---|---|---|---|
| 文件可读 | `OSError` | 丢弃 | (b) 否则会带进 `integrate_patch` 再失败 |
| `is_unified_diff` | 无 `@@` 头的文本 | **丢弃** | (a) 散文冒充 patch 会烧掉一个 integrate 槽 |
| `patch_escapes_tree` | 绝对路径或含 `..` 的 diff 头 | **丢弃** | (a) **真安全检查**：`integrate_patch` 会在多个 strip level 上 `git apply -p<N>`，`-p0` 配 `/etc/...` 会写到框架根之外 |
| `patch_targets_missing` | 在 `base_checkout` 下任何 strip level 都不存在的修改/删除目标 | **丢弃** | (a)+(b) 幻觉出的框架布局 |
| `git apply --check` | 什么都不拒 | 仅记录 verdict | (c)/(b) **刻意只做建议**——stale 也保留，因为 integrate 的 `-p` 自动探测/三方合并可能仍能救 |
| `scan_quantitative_claims` | 什么都不拒 | 仅记录 | (b)，且**很弱**，见 §5.3 |

另有两个在 runner 而非 patch_safety 的路径守卫：`_patch_path_within_bases`（`runner.py:158-183`，patch **文件自身位置**必须在 worktree/workspace 内，类别 (a)）和悬空声明对账（`runner.py:1269-1296`）。

### 5.2 路径限制与 `--add-dir` 的不一致

**`patch_safety` 里没有根允许列表。** 对 diff 头的约束只有：非绝对、无 `..`、目标须存在于单一 `base_checkout`。允许列表在 apply 时才进入（`integrate_patch._resolve_framework_root`）。

**不一致在工具授权，不在 patch_safety**：`subprocess_.py:687-696` 发出 `--add-dir <每个 framework_source_root>`，配 `bypassPermissions` 和 `Edit/Write/MultiEdit/Bash`。而配置 docstring 声称"其余是只读 `--add-dir` 条目（写入仍需 worktree）"——**没有任何东西强制这一点**。只读性只在 prompt 文本里断言（`specialist_prompt_builder.py:1700-1703` 提到 "PolicyGate R4"，而 R4 根本看不到 subprocess 的工具调用，因为 gate 只校验 intent 而 subprocess 一个 intent 都不发）。

**净结果：patch 头路径受约束、patch 文件位置受沙箱约束，而直接 `Write`/`Edit`/`Bash` 写进活的框架树完全不受限。** 一个就地编辑 `/sgl-workspace/sglang/...` 而不是产出 diff 的 specialist，绕过了 benchmark/accuracy 门，也没有回滚记录。（这与 discuss2 §2.4 是同一个问题的两面。）

### 5.3 `FORBIDDEN_PROPOSAL_FIELDS`：检测而非剥离

字段：`{expected_gain, expected_gain_pct, bench_evidence, confidence, score, rank, force_provenance}`。理由（模块注释）："吞吐/增益数字是 Coordinator 测出的真相，永不是 worker 的自报"。

- `expected_gain*` / `bench_evidence` → 会让伪造的先验冒充 KB 推导值（`expected_gain` 是 KB 拥有的字段，本来就渲染进 specialist 自己的 prompt）
- `confidence`/`score`/`rank` → 会与唯一被认可的评分通道 `ProposalScorer`（明确标注"Advisory only"）冲突。注意 `confidence` 作为**顶层** `specialist_done` 字段是合法的，禁的是 per-proposal
- `force_provenance` → provenance 由运行时盖章；自设它等于伪造自己的 KB 回写资格

**关键缺口：这是检测，不是剥离。** 违规 key **仍留在 `done_payload`**，写进 `specialist_done.json`，逐字拷进 `specialist_rounds[].proposal_set`，再进 orchestration prompt。实际执行被委托给 Critic（`prompts/critic.md:135-144`："若这些到达你这里，上游层已经回退了"）。**注释与现实相反：上游层从来没有移除过它们。**

### 5.4 enablement 与 explore 的范围差异

**同一条代码路径。** FRAMEWORK authoring、local-explore authoring、apply-retry authoring、enablement、EXPLORE 的 freeform/domain specialist——全部以 `kind="specialist"` 派发，全部经过 `_finalize` → `vet_patches`。**没有 framework 专属旁路，也没有 framework 专属放宽。**

真正的差异有四处：

1. **交付物形态与路由**。explore 的 `proposal_set` 是纯配置，由 orchestration LLM 读来构建下一个 grid（多节点则由 `_maybe_materialize_mn_explore` 自动物化，上限 6）。FRAMEWORK 把 config-lever 的 `proposal_set` 当作一等 patch 等价物，经 `_maybe_autosubmit_framework_config` 路由进 `integrate_patch.config_changes`（上限 8）。
2. **patch 被丢弃只在 FRAMEWORK 有状态机后果**。`_record_framework_agent_authoring_empty_outcome` 存在的原因正是 `vet_patches` 会清空 `patches_written`——否则候选没有终态行，FRAMEWORK pump 永远重派（gap-5 livelock）。（注：该处注释称 `forbidden_fields` 能清空 `patches_written`，**这是错的**，只有前四项会丢弃 patch。）
3. **grounding base 选树是框架盲的——这是最值得修的一条**。`base_checkout = prep.worktree_base or prep.worktree`，而 `worktree_base = _pick_worktree_base(framework_source_roots)` 取**第一个带 `.git` 标记的条目**，`_DEFAULT_SOURCE_ROOTS` 顺序是 `aiter, sglang, vllm, atom, xDiT`。**所以在 aiter 是 git checkout 的机器上，每一个 flow 里的每一个 specialist 都被拿去对 aiter 树做 grounding**，一个合法的 sglang/vllm patch 会在全部九个 strip level 上触发 `GROUND_MISSING_TARGET` 而被静默丢弃。`integrate_patch._resolve_framework_root` 早就用 target-aware 选根解决了这个问题（注释明写"`vllm/...` 的 patch 必须在 vllm 根下 apply，而不是第一个允许列表条目 aiter"），`patch_safety` / `_maybe_setup_worktree` 从未拿到这个修复。**FRAMEWORK 受害最深**，因为那里的丢弃会转成一条终态 `author_empty` 进度行——一个被当成真实结果记录的假阴性。
4. **enablement** 即使交付物为空也照样走 `integrate_patch`，以便 stall 计数生效。

### 5.5 判定

| 项 | 判定 |
|---|---|
| `is_unified_diff` + `patch_escapes_tree` | **保留**——真安全检查，且失败在模型不在场时发生 |
| `patch_targets_missing` | **保留，但先修 §5.4.3 的选树 bug**——当前它在多框架机器上是假阴性生成器 |
| `git apply --check` grounding | **保留**——已经是纯建议，不误杀 |
| `FORBIDDEN_PROPOSAL_FIELDS` | **保留检测，但应改成真剥离**——当前"检测但不删"是两头不落好：既没保护，又让 Critic 背锅 |
| `_patch_path_within_bases` | **保留**——类别 (a) |
| `--add-dir` 写权限与只读 docstring 的矛盾 | **修**（与 discuss2 §2.4 同一条） |

---

## 6. Proposal-count target 是否生效

### 6.1 全部读取点

`DEFAULT_SPECIALIST_MAX_PROPOSALS: int = 12` 定义在 `policy/gate.py:285`（奇怪的位置——`gate.py` 自己从不用它）。

| 站点 | 做什么 | 是执行吗 |
|---|---|---|
| `explore.py:1007-1012` | `params.setdefault("max_proposals", DEFAULT_...)` | 否 |
| `runner.py:625` | 进 `SpecialistPromptInputs`；行内注释："shapes the prompt, not a hard cap" | 否 |
| `specialist_prompt_builder.py:823-824` | Section 1 文本 "cap your final proposal_set at the top-{N}" | prompt 文本 |
| `specialist_prompt_builder.py:1824-1829` | Section 8 文本 "MUST contain AT MOST {N}" | prompt 文本 |
| `runner.py:1236` | 注释："a prompt-side target, not a hard cap" | 无 |

**没有任何截断。** `_finalize`（`runner.py:1234-1238`）只把 `proposal_set` 默认成 `[]` 并推导 `empty`；`:1173` 的 docstring 仍声称它会"sanitise and **truncate**"——**陈旧且与代码矛盾**。`_validate_specialist_done_payload` 检查列表性、`empty` 一致性、逐条 dict 性与非空 `name`、summary ≤4096、confidence ∈ [0,1]——**无数量检查、无名称唯一性检查**。而且 subprocess 路径根本到不了那个校验器。

`_build_specialist_round_entry` 读 `done_payload.get("proposals_truncated_from")`（`explore.py:1728`）——**一个已不存在的截断留下的化石，全仓没有任何地方写这个 key**。

**specialist 返回 50 条会怎样：50 条全部接受**，写进 `specialist_done.json`，以 `proposals_total=50` 记录，全量进入下游 prompt。

### 6.2 下游消费与成本

| 消费者 | 随 N 的成本 | 自带上限 |
|---|---|---|
| `ProposalScorer`（多模型 LLM 集成） | prompt ∝ N × 模型数 | **有**，`_MAX_PROPOSALS_SCORED = 16` |
| 多节点自动 explore grid（每个 variant 是整服重启+benchmark） | **GPU 小时 ∝ N** | **有**，`_MN_AUTO_EXPLORE_GRID_CAP = 6` |
| FRAMEWORK config grid | **GPU 小时 ∝ N** | **有**，`_FRAMEWORK_CONFIG_GRID_CAP = 8` |
| orchestration prompt（`_recent_proposed_variants`） | token ∝ N | 无 |
| orchestration prompt（`_research_scout_seed_block`，`json.dumps` 每条未测提案 × 所有轮次） | token ∝ N × 轮数 | 无（仅指纹去重） |
| 单机 explore grid | GPU 小时 ∝ grid 大小 | **无上限**——只靠 LLM 自律 |
| Critic 评审 | LLM 成本 ∝ N | 无 |

**两条把 N 直接换成 GPU 小时的路径已经独立截断在 6 和 8——都低于 12。** 未设限的暴露是 prompt token、Critic 成本和 state 体积。

### 6.3 判定：可以删，改成 prompt 里写 6

**可以删，几乎不损失任何东西。**

- 它**本来就只是 prompt 文本**，零执行点；整条 param 管道存在的唯一目的是插值两个 f-string。
- **没有任何调用方传过非默认值**——`params.setdefault` 是全仓唯一写入者，per-task override 通道是死重。
- **12 已经是管道里最松的数**：自动物化的消费者截断到 6 和 8，评分器到 16。把 prompt 目标降到 6 反而让指令与多节点/FRAMEWORK 实际会 bench 的数量**一致**，而不是让 specialist 写 12 条再静默丢掉 6 条。

**具体会损失什么：**
1. per-task override 钩子（当前未使用，要用时重加很便宜）。
2. **单机 explore 是唯一一处 >6 的 `proposal_set` 不被结构性截断的地方**——LLM 直接据此构建 grid。降到 6 会收窄单机每轮的探索宽度。这是否是损失取决于你认为"12 条精选"是否优于"6 条精选 + 再来一轮"；prompt 自己已经写着 "Fewer is better than padding"。
3. 三个测试会挂（`test_critic_verdict_map.py:710/723/733`）。

**建议配套的两处修复**（常量本身是这里较小的问题）：
- 给 `_validate_specialist_done_payload` 加真实数量检查，或在 `_finalize` 里做截断并**真的写** `proposals_truncated_from`，让 `explore.py:1728` 那个已存在的读取方不再是死代码。今天一轮 50 条提案是被完全接受的。
- 修 `runner.py:1173` 声称会截断的陈旧 docstring。

---

## 7. 本轮的可交付清单

### 第一档：直接删（零风险）

| # | 内容 | 位置 |
|---|---|---|
| 1 | `DEFAULT_SPECIALIST_MAX_PROPOSALS` 常量 + `max_proposals` param 管道 | `gate.py:285`、`explore.py:1012`、`runner.py:625` → prompt 里直接写 6 |
| 2 | rebench 端口拒绝逻辑（`PRODUCTION_SERVING_PORT`、`_resolve_port`、`_pick_free_port` 重试与 `18888` 兜底） | `rebench.py:41-83`（**先修 §2.2 的 preclean**） |
| 3 | Iron Rule 1 端口条款的 **GPU 分支** | `specialist_prompt_builder.py:1890-1897` |
| 4 | `proposals_truncated_from` 死读取（或反过来让它活起来） | `explore.py:1728, 1757-1758` |
| 5 | 陈旧 docstring：`runner.py:1173` 声称截断、`subprocess_.py:129-132` 声称 `--add-dir` 只读 | 两处 |

### 第二档：结构性重建

**GPU pool**：可以整个重建，SQLite lane 表可去。必须保留 §1.7 的 I2 + I3 + I4（进程作用域令牌、可证明已死的释放主体、每条 GPU 路径都获取）。carve-off 只在改成"每次准入按真实证据重算"时保留，否则删掉——它当前要么是 no-op，要么是把非 bench GPU specialist 整个关停。

**inprocess fallback**：改成硬错误，`inprocess` 保留为显式测试 flag。

### 第三档：保留

| 控制 | 理由 |
|---|---|
| Iron Rule 1 端口条款 **无 GPU 分支** | `research_lane` 不冲突任何 lane，CPU specialist 与 benchmark 并发，这是唯一控制 |
| `BASH_KILL_SAFETY_PREAMBLE` | 同上；40 token；守的失败对受害方不可见 |
| `is_unified_diff` / `patch_escapes_tree` / `_patch_path_within_bases` | 真安全检查 |
| `patch_targets_missing` | 保留但**先修选树 bug** |
| `research_lane` 多持有者语义 | 当前设计里唯一无争议正确的部分 |

### 必须先修的 bug（本轮新增）

1. **`rebench.run_grid` 的 `preclean_before_run=True`** 会 SIGKILL 活的服务器并 `rm` 活的 `/dev/shm/nccl*`，而 `my_pgid` 排除对 `start_new_session` 的 specialist 无效。**删端口守卫前必须先修这个。**
2. **grounding base 框架盲选树**（§5.4.3）——在多框架机器上把合法 patch 静默丢弃，且在 FRAMEWORK 里被记成真实的 `author_empty` 结果。
3. **in-process 路径无墙钟**（§4.4）——最多 8000 turn，且使 `kill ≤ gpu_lease TTL` 铁律结构性不可满足。
4. **`FORBIDDEN_PROPOSAL_FIELDS` 检测但不剥离**（§5.3）——违规字段一路流进 orchestration prompt，执行被推给 Critic。
5. **送给模型的假话**（§4.5）——in-process 下 prompt 仍称"Coordinator 会硬杀你的 subprocess"，而根本没有 subprocess。

（承接 discuss2 的三条：`kill_task` 静默吞结果、`reclaim_expired_running` 强杀活任务、`--add-dir` 写权限。）

---

## 8. 与上一轮判据的关系

discuss2 的判据是"失败是否发生在模型不被调度的窗口"。本轮需要给它加一条限定：

**默认单机路径上，物理互斥的真正持有者是 Ray（进程作用域），不是 SQLite lane 表（时钟作用域）。** 因此同一个 TTL 洞：

- Ray 开启 → 排队（可容忍）
- Ray 关闭（`INFERENCE_OPTIMIZER_RAY_EXEC=0` / 多节点 / pytest）→ 真双花

这意味着**不能只按"模型在不在场"判定，还要按"哪一层真正持有物理互斥"判定**。很多看起来承重的 SQLite 层控制，在默认路径上其实是 Ray 在兜底；而在 Ray 关闭的配置上，它们又确实是唯一防线。重建 GPU pool 时这是最需要想清楚的一点。

---

## 附：审查方法与分歧裁定

工作流 `specialist-freedom-audit-2`：4 个分片审查 + 4 个对抗复核，8/8 完成，0 失败，814 次工具调用，约 127 万 token，约 54 分钟。

**agent 之间的实质分歧与我的裁定：**

| 争点 | 裁定 | 依据 |
|---|---|---|
| pump 排干是否使 `reclaim_expired_running` 不可达 | **对一半**——对 task 状态机成立，但 **lane 过期不走这条路**：`acquire_many` 内的机会性清理（`resource_lock.py:232-256`）在 `while True` 循环里每轮 `_spawn_fitting_queued` 都执行 | 我核了 `dispatcher.py:130-172` 与 `resource_lock.py:232-256` |
| TTL 洞是"最大的真洞"还是"只在 Ray 关闭时" | **后者**——Ray actor 生命周期绑执行器 `finally`，是进程作用域 | `_ray_serving.py:390-400` |
| 单机端口是否已临时化、8888 是否过时 | **未过时**——profiler/scriptable/非内建脚本四种情形在 `_assign_free_port` 之前就返回，端口保持字面 8888 | `_server_lifecycle.py:161-185` |
| in-process 是否丢失 `--allowedTools` | **不丢失**——SDK transport 发同一个 CLI flag | `subprocess_cli.py:493-494` |
| in-process 的 `--permission-mode` 是更松还是更严 | **更严**——无回调时 SDK 确定性拒绝，不挂 | `query.py:432-433` |
| CLI 版本 pin 绕过是否 fallback 特有 | **不是**——`_find_cli` 无条件优先 bundled binary，每次 `ClaudeBackend` 调用皆然 | `subprocess_cli.py:151-155` |
| 端口门的真正分界 | **不是"两门重复"，而是 GPU specialist（被 lane 保护）vs CPU specialist（`research_lane` 零冲突，可与 benchmark 并发）** | `resource_lock.py:74` |

**本人另行读码核实的关键事实**：lane 冲突表与容量（`resource_lock.py:56-79`、`schema.py:29-38`）、20 个 action 的 `requires_lanes`、`gpu_research_lane` 的三处运行时注入、`serving_slot_busy` 探测的存在本身（`_ray_backend.py:115-137`——**如果 lane 表真能保证互斥，这个探测就不必存在**）、carve-off 实现（`gpu_pool.py:99-153`）、`max_proposals` 全部读取点与 `runner.py:1236` 的自认注释。
