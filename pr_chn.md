# 删掉从不触发的闸门，让规划器看见它此前的盲区

15 个 commit，分两半。前一半删除生产路径根本到不了、或者拒绝了运行时本来就能优雅降级的派发闸门。后一半补上删除动作暴露出来的可观测性缺口：编排模型派发的工作它看不见，而且一个被 SIGKILL 的 specialist 会被报告成执行成功。

## 结果

| | 新增 | 删除 | 净变化 |
|---|---:|---:|---:|
| 源码 | +802 | −530 | **+272** |
| 测试 | +357 | −448 | **−91** |
| **合计** | **+1,159** | **−978** | **+181** |

42 个文件。仅 `policy/gate.py` 一个文件就是 +72/−277。源码净增是因为后一半新增了一个拉取工具、一个 prompt 区块、一个 intent 和一条双向文件通道；前一半是纯删除。

**回归：零。** 失败集合与同一机器上同一棵树的 stash 基线逐条一致——17 项失败（`inference_optimizer/tests` 15 项，`agents/robustness/tests` 2 项），全部来自 shell 环境变量泄漏（`ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` / 一个网关 key）以及一个 venv 探测测试。每一批改动都单独与基线做过 diff 才把失败归因，CI 会独立复核这一点。

## 这些闸门为什么该删

下面每一条都在源码层面确认过不可达或不承重，不是靠 grep 推断的。

**freeform 红线扫描**匹配 `params.task_description` 里的破坏性 shell。它只扫这一个字段，而 `notes`、`research_hints`、`arch_notes`、`gap_symptom` 会原样进入同一份 specialist system prompt 且完全不扫——同一段文字换个字段就通过了。派发之后也没有任何东西执行它：子进程跑在 `bypassPermissions` 下，带着 `Bash`、`Write`、`Edit`，且不存在 PreToolUse hook。它自己的注释就写着这不是安全边界。它唯一可靠的效果，是让规划器为一段**描述**破坏性命令（而非执行它）的文字损失一个 tick。

**`specialist_done` 校验器（R3）**从未见过真实载荷。生产环境的 specialist 写 `specialist_done.json`，dispatcher 读取后直接调用 `_record_specialist_result`，绕过 `_handle_intent`，因而也绕过 `validate_intent`。它同时也是冗余的：`runner._finalize` 会用派发参数重新盖章 `gap_canonical_id` 和 `domain`，并为 `proposal_set`、`empty`、`summary` 填默认值——它检查的每个字段，都已经由 specialist 无法影响的代码保证了。

**五条 scope/tag 拒绝降级为观测。** `resolve_specialist_profile` 的文档明确写着永不抛异常——它会按 tag 数量重新推断 scope——而 `SpecialistRunner` 会为无法解析的锚点合成一个结构完整的空结果。拒绝这些情况，等于把优雅降级变成了白白损失一个 tick。

**同批删除的死代码与自限逻辑**：wave 形状检查（非 list、非 dict 元素、空描述）在 `_fan_out_specialist_wave` 和 `intent_router` 里都已重新检查过；负数 `max_turns` 产生空的 turn range；`gpu_count` 类型检查会被 dispatcher 用同样的默认值重新解析。

**任务注册表中没有对应方的残留**：`failed -> running`（自动重试建的是带 `-autoretryN` key 的**新行**）、`needs_manual_review`（无生产写入方）、`Task.attempts`（无生产读取方——重试上限由 `params["_auto_retry_attempt"]` 驱动）。

### 刻意保留的部分

- **`max_turns` 上界。** in-process backend 的 turn 循环没有任何 wall-clock 检查，这是那条路径上唯一的界。
- **wave 数量上限（16）。** `research_lane` 容量限制的是并发度，不是总开销。一个 turn 可以授权 N 个 `claude` 子进程，而它们的生命周期长于那个 turn 本身。
- **`params` 必须是 dict。** 否则 `AttributeError` 会在 `validate_intent` **内部**抛出，逃过只捕 `PolicyDenied` 的处理器，中断该 turn 剩余的 intent，同时递增紧急停止的崩溃计数。
- **`gpu_count <= 0`。** 在默认单机 Ray 路径上这不会 livelock——`try_acquire_ray_observation` 会成功，dispatcher 随后把 `ROCR_VISIBLE_DEVICES=100000` 写进 specialist，于是它测出垃圾数据并报告成功。

## 由此暴露的盲区

一个在 wall-clock 上限被 SIGKILL 的 specialist，渲染出来**与完全成功的那个逐字节相同**。三件事叠加导致了这一点：executor 从不抛异常，失败信息藏在结果信封里而 `SubAgentResult.error` 是 `None`；信封报告的是 `runner_status`，而它不在渲染器的状态 key 列表中；reaper 的错误文本被折叠成了裸常量。规划器读到的是 `kind='specialist' state='succeeded'`，别的什么都没有——而且即使主动拉取也拿不到原因，因为 `get_recent_outcomes` 复用同一个格式化函数。

修复方式：把 `runner_status` 加入状态 key，回退读取嵌套的 error，surface 审计 notes（一个 patch 全部因未 ground 而被丢弃的运行，不再读起来像干净成功），并在分类器 token 之后保留 reaper 的原始文本，使耗时与阈值数字得以留存。基础设施失败后抢救出的 checkpoint 现在报告 `partial` 而非 `succeeded`，并被归类为不可重试，避免重试丢弃已抢救的成果。

另外补上三个缺口：

- **静默放弃。** 自动重试的两个 bail-out 分支都直接返回、不发任何消息，于是规划器看得到第 1..N−1 次重试，却看不到放弃这一刻。现在会广播 `specialist_auto_retry_exhausted`。
- **看不到在跑的工作。** `get_recent_outcomes` 只查终态事件，且没有任何 prompt 区块携带任务行，所以一个已派发任务在 `task_queued` 与 `delegated_result` 之间是隐形的——对 GPU specialist 而言那是数小时。`get_running_tasks` 把 running 行与 lease、gpu_lease 表 join：已运行秒数、domain 与 gap、lease TTL 与剩余时间、持有的 lane、租用的 GPU id，以及取自 reap 循环所轮询的同一批文件的 heartbeat 年龄。
- **无法判断 GPU 请求是否可调度。** prompt 要求规划器基于 serving TP 推理 `gpu_count`，却从不渲染 TP，也不渲染任一 pool 的大小。`=== Resource pools ===` 区块报告 PolicyGate 实际据以准入的那些数字，包括：当 serving 占满所有卡时，serving-disjoint pool 是空的。

## 新增能力

**orchestration 可以 `kill_task`。** 此前限定给 Robustness 是角色划分，不是安全论证。但这需要先修一个 bug：杀掉一个运行中的任务会摧毁它的结果——cancel 使该行进入终态，而 executor 仍在运行，于是它的收尾迁移抛出 `IllegalTransition`，逃出 `run_task`，被 reap 循环丢弃：没有 `delegated_result`，没有记账，最多可损失数小时的 GPU 工作。`scope` 仍然只允许 task。

**`extend_lease`。** 此前没有任何地方续租：`heartbeat` 存在、受 CAS 保护、零调用方，所以 `lease_ttl_sec` 在入队时就固定了，一个合理地超出注册表默认值的任务会被从活跃工作底下判定失败。该 intent 会同时刷新任务 TTL、它的 lane 行和它的 GPU 行，保持 `kill <= gpu_lease TTL <= gpu_research_lane TTL`。

**specialist 双向通道。** 此前是发射后不管：父进程写 `prompt.md`、spawn、等进程死后读 `specialist_done.json`。一个在一小时后发现自己 mandate 有误的 specialist，只能把剩余预算烧完。

- *上行*：reap 循环本来就每 5 秒 stat 一次工作区，但只在进程死后把 `specialist_done.partial.json` 当兜底来读。现在它在 specialist 存活期间解析每一次重写，并作为 `specialist_progress` 观测重新发布。
- *下行*：发往 `specialist:<task_id>` 的 `send_message` 会追加到该 specialist 工作区的 `inbox.json`，prompt 告诉它在每步之间读取。reaper 忽略这个文件，所以一条消息能引导一次活跃运行而不是终止它——这正是缺失的另一半，因为 `residual_questions` 此前根本没有回传路径。

## 自审中发现并修复的 bug

最后两个 commit 是审查前 13 个 commit 的产物。它们值得作为一个整体来读，因为六条里有四条是这次仪表化工作自己引入的缺陷。

### 触发这轮审查的指令

这条指令是通用的，任何一批改动收尾时都应当照此自审一遍（可以用工作流和 sub agent 并行跑多个视角）：

> 从 `<base-commit>` 开始的这些 commit，我需要你反思这些问题：
> 1. 改动是否完全，docstring 是否也已经修改好了；
> 2. 改动是否冗余，不要过度提交兜底代码，我只需要逻辑代码；
> 3. 注释是否太长？只允许做功能注释。
>
> 保证整体修改简洁、精确。

三个问题对应三类不同的缺陷，实测都命中了：

- **"是否完全"** 抓出了两个硬故障——`EXTEND_LEASE` 缺 `_PAYLOAD_REQUIRED` 条目，以及一个已经在失败的契约测试。二者都因为 `agents/robustness/tests` 是独立测试树，而此前只跑了 `inference_optimizer/tests`。这一问的价值在于：**行为改了，描述它的每一处 docstring、注释、prompt 文本、镜像枚举都要跟着改**，漏一处就是下一个 bug 的入口。
- **"是否冗余"** 删掉了 6 处不可能触发的 `try/except`、一个没人读的 `-> bool` 返回、一个手工复制的 `needs_gpu` 转换（仓库已有 `coerce_needs_gpu`），以及一个只因被调方恰好只读两个字段才能工作的伪造 `Lease` 对象。
- **"注释是否太长"** 删掉约 20 行讲历史与理由的叙述（"这里原本…"、"我们改成…"）。这类内容属于 commit message，不属于代码。

一条经验：**自审要用对抗式复核，不能只跑一遍。** 本轮 8 个 agent 中，审查方与复核方在若干结论上互相反驳，最终以代码为准裁定；被推翻的结论里既有"该删却不能删"的，也有"说没问题其实有问题"的。

1. **`EXTEND_LEASE` 没有 `_PAYLOAD_REQUIRED` 条目。** `validate_envelope` 用裸下标，因此抛的是 `KeyError` 而非 `IntentValidationError`。各 backend 只捕后者，所以模型发出的第一个 `extend_lease` 会被当作 SDK 流失败吞掉，**丢弃该轮已收集的全部 intent。** 同时镜像进了 robustness 的 envelope——它的契约测试会 diff 两份枚举，此前已经在失败。
2. **`_transition_resilient` 在所有调用点吞 `IllegalTransition`**，包括 `queued -> running`，而那里的拒绝**正是**双派发守卫。现在改为按调用点显式开启，只有三个终态迁移容忍它。
3. **`extend_lease` 重置了 `updated_at`**，这在 `extra_sec` 之外额外赦免了已耗用时间，并且清零了 health 区块与 `get_running_tasks` 都依赖它推导的已运行秒数。
4. **`extend_lease` 没有刷新 `gpu_leases`**，于是 GPU reaper 仍可能在活跃 specialist 底下释放显卡——这恰恰是该 intent 要防止的失败。
5. **specialist inbox 被写进了 workspace**，而 prompt 通告的是 worktree。有 worktree 的 specialist——也就是生产场景——永远收不到引导消息。测试之所以通过，只是因为它的环境没有 worktree。
6. **`get_running_tasks` 对多 lane 任务报告了任意一条 lane 的到期时间**；现在报告最早的那个，因为回收从它开始。

## 同一轮清理掉的防御性脚手架

这次审查还剥掉了不可能触发、或调用方已经提供的守卫：一个套在已被包了两层的读取之上的第三层 `_rows` 闭包；对一个非 Optional 且从不重新赋值的字段做的 `session_dir is None` 检查；对字面量 action 与 uuid task id 而言不可达的 `runs_dir` 守卫；包住三行算术的 `try/except`；以及一个紧挨着更大的无守卫渲染器的 resource-pools 守卫。`_handle_extend_lease` 里的裸 `except Exception` 被收窄为一个坏 `task_id` 实际会产生的两个注册表异常——此前它把基础设施故障当成规划器自己的错误报告回去。一个仅仅因为 `heartbeat` 恰好只读它两个字段才能工作的手工构造 `Lease`，被替换为 `heartbeat_by_task`。

由此还带出一个正确性修复：`Resource pools` 区块从 `shared_state.tp` 读 `serving_tp`，而紧挨着的 pool 大小来自 `_serving_tp_for_policy`，后者还会读 `TP` 环境变量——该区块可能一边报 `serving_tp=0`，一边报被切走四张卡。

## 审阅建议

**按可达性论证读删除部分，而不是按 diff 大小。** 上文为每一处被删的闸门给出了具体的不可达路径；真正值得问的是：在**你的**配置下那条路径是否真的不可达。Ray 与非 Ray 的分野在这里很关键：默认单机 Ray 路径上，物理 GPU 互斥由 Ray 的 `num_gpus` / `serving_slot` 持有，是进程作用域；而在 `INFERENCE_OPTIMIZER_RAY_EXEC=0`、多节点或 pytest 下，SQLite lane 是唯一互斥，且是时钟作用域。

**对新增的 intent，请检查接线是否齐全，而不是检查逻辑。** 一个 intent 必须落到 `IntentType`、`_PAYLOAD_REQUIRED`、角色 intent 集合、PolicyGate 分派、IntentRouter 分派表、`emit_intent` 工具描述、`claude.py` 枚举，以及 robustness envelope 镜像。漏掉其中一个，就是上面第 1 条得以出厂的原因。

**双向通道是风险最高的新增。** 它的下行会写入一个 specialist 正在并发读取的目录；tmp+rename 的原子替换是承重的。另外注意：该路径必须与 `worktree or workspace` 一致才能工作，而这一点没有任何测试覆盖到。

**值得写进 release note 的行为变化：** orchestration 现在可以取消运行中的任务，且 specialist prompt 新增了 `inbox.json` 约定。两者都没有改变已有接口，但都会改变一个运维人员观察 session 时看到的内容。
