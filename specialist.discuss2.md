<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Specialist / Orchestration 控制面审查：哪些可以交给模型智能

> 目标：给 specialist 和 orchestration（与当前最新模型同等能力）最大的对话与行动自由，判断哪些确定性 gate 是必要的、哪些只是把聪明模型困在规则里。
>
> 方法：4 个分片源码审查 agent + 4 个对抗式复核 agent（工作流 `specialist-freedom-audit`，8 agent 全部完成，0 失败，约 120 万 token）。agent 之间在若干关键结论上互相反驳，**所有决定性事实由我本人重新 grep / 读码核实后以代码为准**，下文标注了哪些是纠正后的结论。
>
> 代码基线：分支 `feat/zgong/explore-opt-11`，2026-07-29。

---

## 0. 总结论

**方向判断是对的，但被限制得最死的不是 specialist，而是 orchestration —— 而且不是被 gate 限制的，是被"什么都看不见"限制的。**

三个核心事实：

1. **specialist 侧几乎没有真正的约束。** 运行在 `bypassPermissions` + `Bash` + `Write/Edit/MultiEdit` + `Task` 下，无 PreToolUse hook、无 `--disallowedTools`、无 Bash 过滤器。所谓的 gate 拦的是"orchestration 在任务描述里写了什么"，不是"specialist 做了什么"。
2. **orchestration 对在飞 specialist 完全失明。** 派发时一条 `task_queued`，终止时一条 `delegated_result`，中间零信息。而一个 GPU specialist 可以运行 4 小时。
3. **失败信息在渲染层被吃掉。** 一个被 SIGKILL 的 specialist，在 prompt 里与完全成功的 specialist **逐字节相同**。

> 删 gate 只能拿回几个 tick；把眼睛打开才能拿回判断力。

---

## 1. 派发种类区分 / dispatch payload gate 能否放宽

### 1.1 20 条拒绝条件的分类

| 类别 | 条数 | 判定 |
|---|---:|---|
| 真结构性（不校验就崩/污染） | 2 | 必须留 |
| 资源准入（GPU 池） | 3 | 必须留，但理由与文档所写不同 |
| 纯口味（scope/tag/长度/波形状） | 11 | 可降级为观测 |
| 安全剧场（红线正则） | 1 | 应当删除 |
| 死代码（下游已防御） | 3 | 可删 |

### 1.2 必须保留的两条真结构性检查

**`params` 必须是 dict**（`gate.py:1637`）

删除后 `AttributeError` 在 `validate_intent` **内部**抛出，而 `intent_router.py:83-86` 的 `try` 只捕 `PolicyDenied`。异常逃逸到 `coordinator.py:1602` 的 tick catch → `_record_coordinator_exception` → `increment_crash_count()`。后果：**中断该轮剩余全部 intent，并计入紧急停机阈值（24h 内 25 次）**。

**`gpu_count <= 0` 且 `needs_gpu`**（`gate.py:1802`）

此处一个 agent 主张的 "livelock" 论证被另一个 agent 用 `_ray_backend.py:44-59` 推翻，我已核实：`_should_use_ray_backend()` 在 `INFERENCE_OPTIMIZER_RAY_EXEC` 未设且非 pytest 时返回 `not is_multi_node()`，即**单机生产默认走 Ray 路径**。该路径上 `try_acquire_ray_observation`（`gpu_pool.py:295-366`）忽略 count 与 capacity，发放合成 slot `100000`，随后 `subprocess_.py:468-471` 的 `elif gpu_ids:` 分支把 `HIP/CUDA/ROCR_VISIBLE_DEVICES=100000` 写进 specialist 环境。

**结果不是 stall 而是静默错误答案**：specialist 在垃圾设备掩码上跑出垃圾数字并报告成功。比 livelock 危险得多。

（livelock 分析只在非 Ray 路径成立：多节点、`INFERENCE_OPTIMIZER_RAY_EXEC=0`、pytest。）

### 1.3 红线正则应当删除

`_FREEFORM_REDLINE_PATTERNS`（`gate.py:305-316`）+ `_check_freeform_task_description`（`gate.py:1985-2022`）：

- 自身注释即写明 "NOT a security boundary"（`gate.py:302-304`）。
- **只扫 `task_description` 一个字段。** `notes`、`research_hints`、`arch_notes`、`gap_symptom`、`target_gap_notes`、`kb_subgraph`、`static_recon_checklist` 全部不扫，**且逐字渲染进 specialist 的 system prompt**（`specialist_prompt_builder.py:2093-2099`、`:1184-1192`、`:1068-1073`）。同一段文字从 `task_description` 挪到 `notes` 即绕过，无需任何混淆。
- **派发之后零执行**：specialist 跑在 `--permission-mode bypassPermissions`（`subprocess_.py:115`）+ `Bash`（`runner.py:109`）+ `Write/Edit/MultiEdit`（`runner.py:105-107`）下。`runner.py:108` 注释 "runners may further filter via a callback" 所描述的回调 **在代码中不存在**（grep 确认）。

实际效果：拦不到任何危险行为，只让 orchestration 损失一整个 tick。

### 1.4 anchored vs freeform 在代码层面几乎不存在区别

同一 executor、同一 prompt builder，仅字段要求不同。`resolve_specialist_profile`（`profile.py:222-224`）文档明确写"永不抛异常"，scope 不合法即静默从 tag 数量重新推断。因此 scope 三条（#5/#6/#7）拦截的情况下游本来就优雅降级。

### 1.5 一条与"放开"方向相反的保留：波数上限 16

`gate.py:1963` 的 `len(tasks) > 16` **应当保留**，此处需纠正一份 agent 报告的宽松判定。

`research_lane` 的 capacity 限制并发（`schema.py:152-156`、`coordinator.py:625-628`，默认 `2 × GPU`），**不限制总量**。一轮 turn 发 `tasks=[...200]` 会创建 200 个 `claude` CLI 子进程，跨小时排干；**授权它们的模型在燃烧期间不运行，且无法召回**（orchestration 无 `kill_task`，无批量取消 intent）。典型 between-turns 双花。

---

## 2. Emitter confinement 管什么？tool / T+1 agent 接线了吗？

### 2.1 Emitter confinement 在生产路径上是死代码

`gate.py:2026-2069` 规定 `specialist:<task_id>` 只能发 `specialist_done` / `send_message` / `alert`。但：

- `specialist:<task_id>` 身份字符串由**父进程在 `dispatcher.py:781` 合成**，specialist 从未产生过它。
- 生产路径：specialist 用 `Write` 写 `specialist_done.json` → 父进程 `_read_done()`（`subprocess_.py:876-925`）→ `dispatcher.py:778` **直接调 `_record_specialist_result`，绕过 `_handle_intent`，因此绕过整个 PolicyGate**。
- `_validate_specialist_done_payload`（R3，`gate.py:2072-2170`）**生产中一次都不执行**。
- 且其检查的每个字段（`gap_canonical_id` / `domain` / `proposal_set` / `summary`）**已被 `runner._finalize`（`runner.py:1222-1245`）从 dispatch params 重新盖章**，specialist 无法影响。

**R3 既是死代码又是冗余。**

补充：`_validate_payload_paths` 对 specialist 载荷同样是 no-op —— `PolicyGate.strict_paths` 默认 `False`（`gate.py:662`），仅 `INFERENCE_OPTIMIZER_STRICT_PATHS=1` 才启用。

### 2.2 tool 接线

- `allowedTools` 是**允许列表**：`Read, Grep, Glob, Edit, Write, MultiEdit, Bash, TodoWrite, Task` + `WebSearch/WebFetch` + MCP（`runner.py:98-119`）。
- **`emit_intent` 被显式剥离**（`subprocess_.py:677`）——子进程没有 in-process MCP server。它留在 `DEFAULT_SPECIALIST_TOOLS`（`runner.py:100`）纯粹是为了 in-process 回退路径。
- 唯一 denylist：`SPECIALIST_TOOL_DENYLIST = {mcp__cortex_kb__propose_point}`（`runner.py:123`）。它在**每任务 `allowed_tools` 替换默认值之后**再做减法（`runner.py:443-457`），所以不是纯重言式。

### 2.3 T+1 agent：接了，但只靠"不给 Task 工具"

- `LEAF_AGENT_TOOLS = ("Bash", "Read", "Grep", "Glob")`（`leaf.py:20`），无 `Task`。
- **没有任何深度计数器**（grep `depth` 只有注释）。一层限制完全依赖 claude CLI 自己的 `--agents` 工具作用域。
- **`Bash` 在两层都有**，`PATH` 在 env 允许列表（`subprocess_.py:52`），`ANTHROPIC_API_KEY` 等默认继承（`HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV` 默认 true，`subprocess_.py:80-83`）。因此 `Bash("claude --print ... --dangerously-skip-permissions")` 完全畅通。
- **containment 比"纯 prompt"强**：`start_new_session=True`（`subprocess_.py:553`）+ `_kill` 的 `killpg`（`subprocess_.py:843-855`）意味着孙子进程**会**被回收，除非它自己 `setsid`/`nohup`。
- **真正的泄漏是账本**：`parse_claude_stream_json_usage` 只读父进程 `process.log`，Bash 派生的 `claude` 写自己的 stdout，token 花销结构性不可见。

### 2.4 一个所有文档都写反了的问题：`--add-dir` 授予写权限

`subprocess_.py:687-694` 把 `cfg.framework_source_roots` 全部加进 `--add-dir`，而 **`--add-dir` 授予读写，没有只读变体**。

`framework_source_roots` = `resolve_source_file_allowlist()`（`cli/executors.py:137` → `framework/paths.py:21-43,287-305`），包括 `/sgl-workspace/{sglang,vllm,aiter}/`、`/app/ATOM/atom/`、全部 site/dist-packages、**`/opt/rocm/`**。

配合 `Write`/`Edit` + `bypassPermissions`，**specialist 可以就地改动正在服务的框架树**。这直接反驳：

- `subprocess_.py:129-132` docstring：*"the rest are read-only `--add-dir` entries (writes still need the worktree)"*
- Iron Rule 2（`specialist_prompt_builder.py:1920-1928`）

**这不是自由度问题**：污染后，之后每一次 baseline/explore 的测量基准都被静默改变，而只有后来的 agent 能观察到结果、且不可能知道发生过改动。

---

## 3. Persistent task state machine 在限制什么

状态（`task_registry.py:28-35`，同时是 SQL CHECK 约束）：
`queued → running → {succeeded, failed, cancelled, needs_manual_review}`，`failed → running` 可重试，终态无后继。

### 3.1 真正 load-bearing 的只有两条

- **`queued → running` 在 `BEGIN IMMEDIATE` 内做 CAS**（`task_registry.py:303-329`）。唯一阻止 pump 与 `run_action_now` 内联路径（`dispatcher.py:1251-1269`）同时执行同一行的机制。
- **`idempotency_key` UNIQUE**（`schema.py:99`）。因为 `create_or_return_existing` 的 SELECT（`:173`）与 INSERT（`:179`）之间**释放了 `_async_lock`**，UNIQUE 约束本身才是串行化器。

### 3.2 残留物

- `failed → running` 与 `succeeded → running` 的不对称是**残留**：生产中无任何代码重派 `failed` 行，auto-retry 建的是带 `-autoretryN` 的**新 task**（`explore.py:782-788`）。
- `needs_manual_review` 是**死状态**，无任何生产写入者（仅 `close.py:553` 列举）。
- `Task.attempts` 列**零生产读者**（grep 确认），纯遥测；重试上限实际由 `params["_auto_retry_attempt"]` 驱动。

### 3.3 威胁模型需要缩小（纠正）

"另一个 coordinator 进程写同一 session dir" **不是活威胁**：`cli/__init__.py:374-399` 在任何 state/lease 变更前取排他 flock（`session/lock.py:139-162`），否则以 `SESSION_BUSY_EXIT_CODE=3` 退出；robustness 以 `mode=ro` 打开 DB。**每 session 恰好一个写连接**，由 `SqliteConnection._async_lock` 串行化。

竞态仍然真实存在，但威胁模型是**单事件循环上的协作式交错**（每个 `await` 都是让出点），不是多进程。

### 3.4 两个真 bug，比"限制太多"严重

**Bug A — `kill_task` 静默吞掉结果**

`_handle_kill_task`（`intent_router.py:813`）把 `running → cancelled`，`cancelled` 是终态。**但子进程不停**（无任何消费者订阅 kill topic，grep 确认）。等它跑完：

1. `_transition_resilient(task_id, "succeeded")` 抛 `IllegalTransition`
2. 该函数**只吞 `TaskNotFound`**（`sub_agent_runner.py:169-182`，已核实）
3. 异常逃出 `run_task`，在 `dispatcher.py:726-732` 被 `continue` 丢弃
4. **无 `delegated_result`，无 `_record_specialist_result`，proposal_set 全部蒸发**

lease 会被两个 `finally` 正确释放。所以 robustness 对运行中任务唯一的杠杆，今天的效果是把最多 4 小时的 GPU 工作变成沉默。

**Bug B — `reclaim_expired_running` 对活任务做同样的事**

**没有任何地方续租**：`resource_lock.heartbeat()`（`:312-332`）零生产调用者（grep 确认）。`updated_at` 只在 `transition()` 时写（`:325-328`），因此 TTL 测的是"进入 running 起的时间"，不是"距上次活着的时间"。

`explore`/`sweep`/`kernel_opt` 的 `lease_ttl_sec = 7200`。一个合法超过 2 小时的 grid 在下个 pump 顶被强制 `failed`；完成时 `failed → succeeded` 非法，结果同样被丢弃。GPU specialist 免疫，仅因其 TTL 被重新取自 `wall_budget × 1.1`（`dispatcher.py:471`）。

**在放开任何权限之前，这两个必须先修**，否则等于把一把会毁掉工作而不是停止工作的枪交给模型。

---

## 4. Lease release / watchdog 为什么不能交给模型

### 4.1 硬事实：模型在需要它们的整个窗口里不被调用

```
coordinator.py:1560-1563   reactor passes      ← LLM 在这里运行
coordinator.py:1573        _pump_dispatcher_once()   ← 排干全部在飞任务才返回
```

`_pump_dispatcher_once`（`dispatcher.py:172-180`）的 `while True` **只在 `inflight` 为空时 return**（已核实）。GPU specialist 的 `wall_budget_sec = min(60min × (macro_cycle+1), 240min)`（`dispatcher.py:632-654`）。

**因此：一个 GPU specialist 运行期间，orchestration 模型可以连续 4 小时一次都不被调用。** 期间不跑 reclaim、不跑 TTL sweep、不跑 maintenance（三个 reclaim 都在 `dispatcher.py:138/148/154`，即 pump 顶部，maintenance 在 pump 之后且每 50 tick）。唯一活着的是 `_reap_loop` 的 5 秒 mtime 轮询和 `asyncio.wait` 的 10 秒 poll。

"让模型注意到 specialist 结束了再释放 lease" —— 在需要释放的那一刻，模型不在执行，能执行的只有 `finally`。`gpu_research_lane` 是 capacity-1 且与所有 serving lane 互斥（`resource_lock.py:76`），泄漏一次锁死到 TTL（最长 15840s），唯一兜底 sweep 是每 50 tick 的 maintenance。

**推论**：TTL reclaim 实际是**重启兜底**，不是在飞看门狗。

### 4.2 但 orchestration 确实被限制得太死 —— 死因是失明

核对 `_compose_prompt`（`conversation.py:257-568`）全部 section 后，orchestration 关于在飞 specialist 收到的信息量是：

1. 派发时一条 `task_queued`（`intent_router.py:501-508`）
2. 终止时一条 `delegated_result`（`dispatcher.py:748-761`）

**中间什么都没有。** 无 elapsed、无心跳年龄、无 partial 输出、无 turn 计数。

拉取工具也填不上：`CONTEXT_TOOL_SPECS`（`mcp_context_tools.py`）**没有 `get_running_tasks`**；`get_recent_outcomes` 只查 `topic IN ('delegated_result','review_verdict')` 即终态事件。

**而 `=== Specialist health ===` 块已经存在、已经渲染好了**（`conversation.py:496-521`，已核实），只是被 `if agent_name == "robustness":` 挡住。`_scan_stale_specialists`（`explore.py:597-629`）已算出 `{task_id, kind, running_seconds}`。**纯读操作，零并发含义。**

同样，`KILL_TASK_SOURCE_ALLOWLIST = frozenset({"robustness"})`（`gate.py:434`）是**角色划分选择，不是安全论证**。orchestration 才是拥有战略上下文判断"这个 specialist 在追死胡同"的角色。

另外两处不一致：
- `prune_branch` 的 `cancel_family` 过滤 `WHERE state='queued'`（`task_registry.py:517`），**对 running 无效**，而 `_handle_prune_branch` docstring（`intent_router.py:840`）声称"cancels its in-flight tasks"。文档 bug，非设计不变量。
- `_reap_loop` 把 `process.log` mtime 当作活着的证据（`subprocess_.py:769-781`）。一个对着降级网关疯狂重试的 specialist 会持续写日志，**永不触发 300s stale 阈值**，只能等最长 4 小时的硬 cap，期间一直占着 `gpu_research_lane`。

---

## 5. Single-exit failure synthesis 能正确传给 orchestration 吗

### 5.1 不能 —— 本次审查最严重的发现

追一个被 SIGKILL 在 wall-clock 上限的 specialist：

| 步骤 | 位置 | 内容 |
|---|---|---|
| 1 | `subprocess_.py:797` | `"specialist subprocess exceeded 1800s wall-clock cap"` |
| 2 | `runner.py:1137-1146` | 折叠成常量 `"subprocess_timeout"`（1800 这个数字丢失） |
| 3 | `cli/executors.py:236-249` | 放进 `{"runner_status": ..., "error": ...}`；**executor 永不抛异常**（docstring: *"Always returns a dict (even on failure)"*） |
| 4 | `sub_agent_runner.py:299-303` | 因此走成功分支：`SubAgentResult(state="succeeded", error=None)` |
| 5 | `dispatcher.py:760` | 发布 `"error": result.error` → **`None`** |
| 6 | `coordinator.py:350` | `_first_present(result, ("status","verdict","outcome"))` —— **executor 发的是 `runner_status`，不在元组内** → `None` |

**模型实际读到的整行：**

```
seq=4127 msg_id=m-8f3a from=coordinator topic=delegated_result kind='specialist' state='succeeded'
```

**一个被 SIGKILL 的 specialist，在 prompt 里与完全成功的 specialist 逐字节相同。** 不是"看到 0 proposals"，是"看到成功"。

失败原因确实写进了 `shared_state.last_specialist`（`writeback.py:1618-1631`）和 `specialist_rounds[].reason`（`explore.py:1747`），但 **grep 确认无任何 prompt renderer 读这两个字段**；`to_prompt_summary`（`render.py:491-557`）没有 specialist 行。`get_shared_state` / `get_recent_outcomes` / `why_denied` 也读不到（后者复用同一个坏掉的 `_format_inbox_event`）。**该信息对模型不可达，即使主动去问。**

唯一的邻近信号是 `specialist_done_recorded` 观测（`writeback.py:1650-1661`），载荷含 `proposals_total: 0, empty: True` —— **告诉模型"空"，从不告诉"为什么"**。

### 5.2 为什么还要合成 —— 理由比文档所写小得多

两个 agent 在此打架，我核实后：**合成应当保留，但它自称的两个理由都是假的。**

- **R3 论证是假的**：R3 生产不可达（§2.1）。
- **FRAMEWORK livelock 论证是假的**：`dispatcher.py:777` 读 `result_dict.get("specialist_done") or {}`，`{}` 通过 `isinstance(..., dict)`，`_record_framework_agent_authoring_empty_outcome` 照常触发；它需要的一切都来自 `task.params` 而非 done 载荷（`framework.py:4724-4795`）。终态行被盖章，pump 前进。
- **plateau 论证也是假的**：`_build_specialist_round_entry`（`explore.py:1735-1758`）每个字段都有 `or ""` / `or []` 默认，`round_id` 回退到 `task.task_id`。`{}` 仍产生 `proposals_total=0` 的行，`_round_is_empty` 正确计入空streak。

**真实损失只有两个字段**：`domain` 空 → `note_specialist_dispatched("")` → `_anchor_for("")` 返回 `""` → 计数器不重置；`gap_canonical_id` 空 → `append_gap_attempt` 跳过（`writeback.py:1730-1733`）。**而这两个本来就在 `task.params` 里，dispatcher 只是没读。**

所以合成是对的（它在无模型运行时执行，是字段兜底而非闸门），**它的缺陷是 `_finalize` 是"干净空结果"与"硬件失败"的同一出口，而区分二者的 `runner_status` 在渲染层被丢掉了。**

### 5.3 Partial-result recovery 把硬 kill 呈现成成功

`_finalize`（`runner.py:1222+`）在"有 payload"分支上**根本不读 `backend_error`**，硬编码 `status = "succeeded"`，`SpecialistRunResult.error` 保持 `""` 默认。于是：

- 被 SIGKILL 的 timeout **带 partial** ⇒ `runner_status='succeeded'`、无 error
- **不触发 auto-retry**（`classify_specialist_failure` 对 `succeeded` 返回 `(NONE, False)`，`runner.py:312-313`）—— **partial recovery 抑制了 §5.4 的重试**
- 半成品 proposals 进入 `specialist_rounds` 参与 plateau 数学
- 多节点下 `_maybe_materialize_mn_explore`（`writeback.py:1665-1677`）会把它们自动物化成真实 benchmark 任务

**且该 bug 也吞掉干净情形**：exit 0 但只有 partial 的子进程同样被标成 `succeeded`。

`_recovered_from_partial: True` 这个诚实的标记：`_build_specialist_round_entry` 不拷贝、`update_last_specialist` 不拷贝、`specialist_done_recorded` 观测不携带、`_format_inbox_event` 够不到。**除两个写它的地方外无任何读者**（grep 确认），也无任何分支依赖它。

### 5.4 auto-retry 的放弃是静默的

- 触发：`dispatcher.py:738-746`，在发布 `delegated_result` **之前**；调度了重试就 `continue`，**失败那次的 `delegated_result` 根本不发布**。
- 资格：仅 `runner_status == "stale"`，即 `{TIMEOUT, STALE_HEARTBEAT, CRASH}`。语义空结果（`NO_OUTPUT`/`CONFIG`）与 `tool_violation` 不重试 —— **这个排除是正确的**。
- 上限：`SPECIALIST_AUTO_RETRY_MAX = 2`（`coordinator.py:25`）。
- 可见性：重试 1..N-1 有 `specialist_auto_retry` 观测（`explore.py:792-804`）；**`attempt >= cap` 时在 `explore.py:743-744` 裸 return，不发任何观测**；`explore.py:789-791` 槽位已占时同样静默。

**模型能看到重试，看不到放弃。** 因此无法区分"3 个 domain 没话说"与"3 个 domain 各超时两次后被放弃" —— 不是推理失败，**是这些比特不在 prompt 里**。

### 5.5 orchestration 能否自己重试

能。没有任何按 gap 去重的机制。唯一刹车是 idempotency key：`_handle_delegate`（`intent_router.py:434-499`）依次尝试 `raw_key`、`raw_key-retry1..-retry5`；仍在运行的重复以 `rule='duplicate_idempotency_key_running'` 拒绝，5 次滚动后 `rule='duplicate_idempotency_key'`。

**若模型不给 `idempotency_key`，自动 key 是 `f"{source}:{action}:t{tick}:{sha1(params)[:10]}"`（`intent_router.py:419-424`）—— tick 作用域**，所以后续 tick 的重新派发永不被去重。这一条其实已经给了想要的自由。

---

## 6-7. Partial recovery / bounded retry 在防范什么

- **Partial recovery** 防的是 budget kill 浪费已完成的工作。目的正当，**当前实现净负**：抹掉失败类别、抑制本该发生的重试、把半成品当完整数据喂给 plateau 数学、多节点下自动物化成 benchmark。
- **Bounded retry** 防的是基础设施抖动（timeout / stale heartbeat / crash）浪费一整个 domain 的探索机会。机制正确（正确排除语义空、cap=2、`-autoretryN` key 互斥），**叙事坏掉**（放弃静默）。

---

## 8. 可交给模型智能的清单

### 第一档：直接删 / 降级为观测（约 −800 行，零风险）

| # | 内容 | 位置 | 理由 |
|---|---|---|---|
| 1 | 红线正则整个删 | `gate.py:302-316`、`:1985-2022` | 自认非安全边界；只扫一个字段；同文可挪到 `notes` 绕过；派发后零执行 |
| 2 | R3 全部删 | `gate.py:2026-2170`、`_SpecialistPseudoRole` `:394-409`、`intent_router._handle_specialist_done` `:510-542` | 生产不可达，且字段已被 `_finalize` 重新盖章 |
| 3 | scope/tag 一致性 5 条降级为观测 | `gate.py:1659-1701`（#3–#7） | `resolve_specialist_profile` 文档写明永不抛异常，下游全部优雅降级 |
| 4 | 波形状 3 条删 | `gate.py:1953-1962` 等（#15/#17/#18） | `intent_router.py:404-411` 与 `explore.py:648-655` 已防御，纯死代码 |
| 5 | `max_turns` 下界删、`gpu_count` 类型检查删 | `gate.py:1727`、`:1795-1801` | 前者自限（空 range → 空结果）；后者 `dispatcher.py:445-447` 自带 try/except |
| 6 | `failed→running` 不对称、`needs_manual_review`、`Task.attempts` 删 | `task_registry.py:40-41`、`:28-35`、`:322-324` | 无写者 / 无读者的残留 |
| 7 | 解锁 `=== Specialist health ===` 给 orchestration | `conversation.py:496` 去掉 `if agent_name == "robustness"` | 纯读，已渲染好，零并发含义 |

**`max_turns` 上界需保留**（`gate.py:1727-1738`）：`_run_via_backend`（`runner.py:869`）的 turn 循环**没有任何 wall-clock 检查**（grep 确认，`runner.py` 里 `wall_budget_sec` 只出现在 subprocess 路径的 622/1042/1058 行）。在 `--specialist-dispatch-mode inprocess` 或 PATH 缺 `claude` 时，`max_turns` 是唯一的界。

### 第二档：补仪表后交给模型

**这不是"放松限制"，是"把模型的眼睛打开"。当前的不自由约 90% 来自失明，不是来自 gate。**

| 要交出去的判断 | 缺的仪表 | 位置 |
|---|---|---|
| 失败该不该重试 | `runner_status` + 内层 `error` 进入渲染 | `coordinator.py:269` 加 `"runner_status"`；`:343-364` 分支在 `payload["error"]` 为 None 时回退读 `result["error"]` |
| 放弃是否合理 | `specialist_auto_retry_exhausted` 观测 | `explore.py:743-744`、`:789-791` 两处静默 return |
| 是否该杀在飞 specialist | `get_running_tasks` 工具 + 放开 `kill_task` | `mcp_context_tools.py` `CONTEXT_TOOL_SPECS`；`gate.py:434` |
| 是否该延长 lease | `EXTEND_LEASE` intent | `resource_lock.heartbeat()`（`:312-332`）已存在、CAS 保护、零调用者 |
| specialist 中途方向错了 | 双向通道（见下） | — |
| patch 被全部丢弃时的应对 | `notes`（`tool_violations` / `patch_safety_dropped` / `patches_claimed_but_missing`）进入渲染 | `runner.py:1288-1332` → `coordinator.py:343-366` |
| GPU 请求是否可调度 | `=== Resource pools ===` 块 | `render.py:491-557`（pool size / serving_tp / lane capacity 目前一个都不渲染） |
| 失败原因是否可判断 | 保留 `outcome["error"]` 原文而非折叠成常量 | `runner.py:1137-1146` |

**`get_running_tasks` 应返回**：`task_id, kind, params.domain/gap_canonical_id, idempotency_key, running_seconds`（`explore.py:611-628` 已算）、`wall_budget_sec`（`dispatcher.py:632-654`，目前只进 specialist 的 prompt）、`lease_ttl_sec / lease_expires_in_sec`、`lanes_held`、`gpu_ids`、`heartbeat_age_sec`（`subprocess_.py:769-781` 的真实信号，目前只在 `_reap_loop` 内部）。

**关于双向通道 —— 这是"自由对话"最核心的缺失，成本比想象低：**

- **child → parent 已有物理层**：`specialist_done.partial.json` 已存在、已被要求原子重写（`specialist_prompt_builder.py:1749-1760`）、`_reap_loop` 每 5 秒就在旁边 stat 文件。**它只是从不在 specialist 活着时读它**（`subprocess_.py:592-604` 只在进程死后作为兜底读）。让 `_reap_loop` 解析 partial 并 append 一条 bus observation，就有了实时上行。
- **parent → child 完全不存在**：`Popen` 没传 `stdin`（`subprocess_.py:547-554`），`prompt.md` 写一次不再重写（`subprocess_.py:436-441`），grep `inbox`/`guidance.json`/`nudge`/`specialist_inbox` 全仓无结果。需新增一个 reaper **不**当作退出信号的 `inbox.json`。

今天的代价：specialist 发现 mandate 错了，只能烧完全部 60–240 分钟预算，无法问一句。

### 第三档：不能动

**判据只有一条：失败是否发生在模型不被调度的窗口里。**

| 控制 | 位置 |
|---|---|
| 两个 `finally` lease 释放 | `dispatcher.py:614-630`、`sub_agent_runner.py:304-307` |
| `queued→running` 的 `BEGIN IMMEDIATE` CAS | `task_registry.py:303-329` |
| `idempotency_key` UNIQUE | `schema.py:99` |
| `acquire_many` 多 lane 全或无 + 冲突集展开 | `resource_lock.py:164-310` |
| `try_acquire` / `try_acquire_ray_observation` 单事务 | `gpu_pool.py:231-366` |
| 死 PID 回收 / TTL 回收 / `_reap_loop` 5 秒轮询 | `task_registry.py:437-500`、`resource_lock.py:420-478`、`subprocess_.py:700-803` |
| 终态不可变（但应改为显式 already-terminal 检查而非靠抛异常） | `task_registry.py:41-43` |
| `_transition_resilient` 吞 `TaskNotFound` | `sub_agent_runner.py:169-182`（retention 会剪掉终态行） |
| **波数上限 16** | `gate.py:1963`（唯一逆"放开"方向保留的） |
| `params` 必须是 dict | `gate.py:1637` |
| `gpu_count <= 0` | `gate.py:1802` |
| `SPECIALIST_TOOL_DENYLIST`（KB 写） | `runner.py:123` —— 唯一跨 session 的可变共享状态，写坏污染未来 session 的 warm-start，而当时的模型早已不在 |
| `_reconcile_cancelled_policy_denied_integrate_tasks` | `dispatcher.py:205-282`（resume 顺序修复，运行在任何模型 turn 之前） |

### 必须先修的 bug（在放开权限之前）

1. **`kill_task` 静默吞结果**（§3.4 Bug A）。先修再放开 `KILL_TASK_SOURCE_ALLOWLIST`。
2. **`reclaim_expired_running` 强杀活任务**（§3.4 Bug B）。要么续租，要么只作用于没有活 asyncio task 的行。
3. **`--add-dir` 对 `/sgl-workspace/*` 与 `/opt/rocm` 授予写权限**（§2.4）。与 docstring 和 Iron Rule 2 直接矛盾；污染基准且不可观测。

### 两个顺带发现的渲染 bug

1. `coordinator.py:366` 测 `payload.get("kind") == "policy_denial"`，而 `writeback.py:176` 写的是 `"policy_denied"`。专用分支对所有 gate 拒绝都是死的。**当前它反而有利**（回退分支 `coordinator.py:392-394` 会 dump 完整 payload），所以修的时候应让分支显式渲染 `rule`/`hint`/`reason`，而不是只改字符串匹配 —— 否则模型看到的信息会**变少**。
2. `explore.py:684` 的 `_fan_out_specialist_wave` 直接调 `_handle_delegate` 而非 `_handle_intent`，因此 wave 子任务**永不产生 `policy_denied` 观测**；拒绝只在 `sub_agent_runner.py:214-231` 表现为 `reason="policy_denied"` 的 cancelled 任务，而无任何 prompt 路径读它。

---

## 9. 一句话总结

"不要把聪明的模型用规则限制起来"的判断是对的，但这个系统限制模型的方式不是规则 —— **是它让 orchestration 盯着一个 4 小时的黑盒，然后在 specialist 被 SIGKILL 时告诉它 `state='succeeded'`**。

删 gate 只能拿回几个 tick；把眼睛打开才能拿回判断力。

---

## 附：审查方法与可信度

- 工作流 `specialist-freedom-audit`：4 个分片源码审查 agent（dispatch-gate / emitter-and-tools / lifecycle-state / failure-path）+ 4 个对抗式复核 agent。8/8 完成，0 失败，744 次工具调用，约 120 万 token，约 49 分钟。
- **agent 之间存在实质分歧**，以下结论是复核后以代码为准的裁定：
  - GPU 池 livelock vs Ray 路径静默垃圾掩码 → **后者**（`_ray_backend.py:44-59` 决定单机默认走 Ray）
  - 合成的必要性 → **保留，但 R3 / FRAMEWORK livelock / plateau 三个理由全假**，真实理由只有 `domain` + `gap_canonical_id` 两个字段
  - 波数上限 16 → **保留**（一份报告判为可删，理由是 lane capacity；但 capacity 限并发不限总量）
  - 多进程并发威胁 → **降级为单事件循环协作式交错**（session flock 已排除多写者）
- 本人另行 grep / 读码核实的关键事实：`emit_intent` 剥离（`subprocess_.py:677`）、`KILL_TASK_SOURCE_ALLOWLIST`（`gate.py:434`）、`_OUTCOME_STATUS_KEYS` 与 `runner_status` 不匹配（`coordinator.py:269` vs `executors.py:237`）、`Specialist health` 的 robustness 门（`conversation.py:496`）、pump 排干语义（`dispatcher.py:172-180` + `coordinator.py:1560-1573`）、`_transition_resilient` 只吞 `TaskNotFound`（`sub_agent_runner.py:169`）、reap 的异常丢弃（`dispatcher.py:726-732`）。
