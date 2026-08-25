# RELEASE-BLOCKERS — Phase 1 重建版

> 本文件根据 2026-08-11 至 2026-08-12 对 `code-clean-ref/` 的二次源码复核记录重建。
> 原始 `RELEASE-BLOCKERS.md` 已遗失，因此本文**不是逐字备份**，也不沿用原先无法复现的
> “419 条原始 findings”计数。本文恢复已回到当前源码确认的 P0/P1，并按二次审核结果
> 将原 P2/P3 去重、纠错后重组为可独立验收的根因。
>
> 复核基线：当时工作区 HEAD `b7f85e3ba`。后续源码若已变化，应重新确认锚点。

# Iron rule

这个文件永远不要进git tree. 

## 分级

- **P0**：错误 benchmark/正确性结果可能被当成成功，或秘密可能自动落盘/外送。
- **P1**：会话、补丁、任务、租约、远端状态或报告可能被错误选择、破坏、遗漏或伪报成功。
- **P2**：已确认的契约、错误边界、数据形状或可观测性缺陷，进入首个补丁窗口。
- **P3**：不直接改变正确结果，但会持续制造漂移、重复、误导或测试维护成本的清理项。

## Phase 1 计数

- **P0：10 条**
- **P1：44 条**
- **P2：50 条**
- **P3：24 条**
- **合计：128 条**

---

## P0 — 发布阻塞（10 条）

### P0-01 · Langfuse 外送完整环境变量

`record_session_start()` 把整个 `os.environ` 交给 Langfuse payload；
`redact_env()` 只按有限的变量名片段屏蔽。自定义 gateway key、鉴权 header、
含 userinfo 的 URL 或名称不符合既有模式的秘密可能离开本机。

- 锚点：`src/hyperloom/orchestrator/trace/langfuse_emitter.py:773-797`
- 锚点：`src/hyperloom/orchestrator/trace/langfuse_mapping.py:362-412`
- 建议：改为显式安全字段 allowlist；再按 secret 名和值做第二层递归脱敏。

### ~~P0-02 · Agent command/tool transcript 未统一脱敏~~ ✓ 已关闭（删除方式）

**处置**：sink 已整条删除——`agent_transcript.jsonl` 两条写入路径
（Claude 侧 `_run_tracelens_skill_claude` 与 Codex 侧 `_run_tracelens_skill_codex`）
及其序列化层（`_write_codex_transcript`、`_serialize_sdk_message`、`_serialize_sdk_block`、
`_json_safe`、`_cap_str`）均已从 `tracelens_skill_runner.py` 中移除；
Codex 的 `$ command` 日志循环（`describe_codex_item`）与 item 归一化链
（`normalize_codex_items`、`codex_item_type`、`codex_file_changes`、`_item_dict`）
已从 `codex_session.py` 中移除；`CodexSessionResult.items` 字段已删除；
`coordinator.py` 的 `_LIFECYCLE_PATH_KEYS` 中 `tracelens_agent_transcript` 条目已删除。
凭据不再有任何落盘路径。

### ~~P0-03 · Multi-node snapshot 可明文持久化 `--extra-env` 凭据~~ ✓ 磁盘副本已删除

**处置（本次）**：`_dump_mn_input_params` 函数及其在 `cli/__init__.py` 的 re-export 与调用点已删除；
`$USER_DATA_PATH/optimizer_runs/mn_input_params_*.json` 不再产生。
同批次还删除了 `_gc_old_profile_traces`（生产零调用点的孤立 GC 函数）、
`orchestration_turns.jsonl` 写入逻辑（`OrchestrationTurnRecord` / `append_orchestration_turn`，
coordinator 方法收窄为 `_trace_mcp_setup` 保留 MCP setup 落盘）、
`_persist_audit` 冗余副本（`agents/framework/audit.py`，stdout 与 session 耐久副本已覆盖）、
以及孤儿工具 `apply_and_bench.py`（全仓零调用者，文件与测试一并退役）。

**出境通道仍存（有意豁免）**：`INFERENCE_OPTIMIZER_EXTRA_ENV` 与
`INFERENCE_OPTIMIZER_SERVER_ARGS` 这两个容器型键仍经 Langfuse `session_start`
明文出境——`redact_env`（`langfuse_mapping.py:362-412`）只按键名片段匹配，
对容器型键名无效，已实测确认（`{"HF_TOKEN": "hf_..."}` 原样通过）。
本次有意豁免：Langfuse 属自控环境，优先级低于磁盘副本。

**三种脱敏残缺形态（P0-01/P0-02/P0-03 的共同根因）**：

1. **只看键、不看值**：`multi_node._redact`（4 个子串）、`langfuse_mapping.redact_env`（15 个 marker）。
   键名沉入嵌套结构即失效——`INFERENCE_OPTIMIZER_EXTRA_ENV` 的外层键名不含 marker，
   内部 `HF_TOKEN` 明文落盘。
2. **只看值、不看键**：`orchestration_trace._safe_value`。递归形状正确但不检查键名，
   实测 `{"HF_TOKEN": "hf_..."}` 原样输出；有已知前缀（`sk-*`）时才能拦下。
3. **序列化后跑正则**：`env_safety.redact_secret_values` 与 `conversation_trace.redact_secrets`。
   字符类 `[^\s,;'\"]+` 排除双引号，JSON 形态 `{"HF_TOKEN": "hf_..."}` 整条失配。

正确修法是在序列化**之前**对 mapping 递归脱敏（键名判定 + 值形状判定），而非序列化后补正则。
仓库里已有最接近正确的实现：`remote_recipe/sanitize.sanitize_publish_env_mapping`（allowlist + 值级脱敏）。

**附加 finding（本次审计新发现，有意豁免）**：
enablement setup 命令字符串可含 `KEY=VALUE` 凭据（`_is_allowlisted_setup_command` 明确放行前导赋值）
并原样进入 `enablement_setup.log` 与运行日志两个 sink（`integrate_patch.py:156-162,269,284,292`）。
修法：对 `cmd` 与 stdout/stderr 过 `redact_secret_values`（裸 `KEY=value` 形态已实测可拦）。

**profile-trace 保留期**：`_gc_old_profile_traces` 随本次删除，
profile trace 保留期仍属未实现（接线等于新增"自动删用户数据"行为，建议单独立项）。

**复核结论**：`experiments/ab_torch_compile.py` 无 write-only 产物（config.yaml 被子进程立即消费，
report 是 `--out-json` 契约）；`apply_and_bench.py` 已确认孤儿并随本次退役。

### P0-04 · 通用 harness 在 FP32 路径使用默认 FP16 容差

`check_correctness_val()` 的默认 dtype 是 FP16；`mode_correctness()` 调用时未传
当前 config dtype。BF16/FP16 当前阈值相同，但 FP32 会被套用更宽的 FP16 容差，
错误 kernel 可能通过 correctness gate。

- 锚点：`src/hyperloom/agents/kernel/tools/harness_generator.py:895-903,932-943`
- 建议：依据 `ref.dtype/out.dtype` 或 config dtype 选择容差，并为 FP32/FP16/BF16/FP8 增加正反例。

### P0-05 · Baseline 可读取旧 report 并把本轮失败标成 succeeded

复用 output dir 时只清理部分文件，随后从目录中的 `benchmark_*` 选择结果。
即使本轮 subprocess 非零，只要旧 report 有正吞吐，仍可能返回
`status="succeeded"`。

- 锚点：`src/hyperloom/orchestrator/actions/executors/baseline.py:2760-2775,2892-3034`
- 建议：每轮使用唯一 result dir；结果必须带 run/task identity、开始时间和本轮 returncode。

### P0-06 · 全局 `/workspace` rescue 可串用另一会话的 benchmark

benchmark rescue 只按 mtime 从全局位置找报告，没有校验 session、task、模型或 subprocess
启动时间。并发 session 可读取彼此结果并构造“有效” measurement。

- 锚点：`src/hyperloom/orchestrator/actions/executors/benchmark_result.py:80-172`
- 建议：禁止全局无身份 rescue；至少要求 session/task fingerprint 与时间窗口全部匹配。

### P0-07 · A/B 实验可复用旧 arm 结果并返回成功

`ab_torch_compile.py` 不清理 arm 目录；运行后从全部旧 `benchmark_*` 中取最后一个。
Magpie arm 又忽略本次 returncode，本轮失败可读取旧成功报告并最终返回 0。

- 锚点：`src/hyperloom/inference_optimizer/experiments/ab_torch_compile.py:184-201,526-538`
- 建议：每次运行创建不可复用的 arm 目录；结果必须绑定本轮 PID/start timestamp。

### P0-08 · GEAK wrapper 可把旧 `result.json` 当成本轮结果

输出目录复用时没有先删除旧 `result.json`。新 runner 失败或没有写结果时，
wrapper 仍可加载旧成功记录；`exit_code` 又未参与最终 `ok` 判定。

- 锚点：`src/hyperloom/agents/kernel/tools/gemm_tuning.py:242-294`
- 锚点：`src/hyperloom/agents/kernel/tools/backends/geak_runner.py:79-85,130-151`
- 建议：启动前删除/隔离旧结果；校验 result mtime、run id 和本轮 returncode。

### P0-09 · Bypass benchmark 没有执行被声明验证的 env variant

variant YAML 中的大部分 per-variant env 没有传给 server；候选可能实际运行 baseline，
却因为噪声被 KEEP。`unset_envs` 还可删除 pinned workload 字段，LLM 生成 env 也未经过完整 blocked-env 过滤。

- 锚点：`src/hyperloom/orchestrator/actions/executors/bypass_runner.py:835-852`
- 锚点：`src/hyperloom/orchestrator/actions/executors/bypass_report.py:35-39`
- 建议：server/eval 必须使用同一经过安全过滤的 materialized env；将实际 env fingerprint 写入结果并在 KEEP 前比对。

### P0-10 · Parallel E2E 全部 attempt 失败仍顶层 succeeded

每个 attempt 的异常会压成行级 failed，但聚合层不检查是否存在成功 attempt，
无条件输出顶层 `"status": "succeeded"` 并返回 0。

- 锚点：`src/hyperloom/agents/kernel/tools/parallel_e2e_runner.py:215-229,481-511`
- 建议：零成功时返回 failed/non-zero；partial success 需要单独状态和成功计数。

---

## P1 — 发布前修复或书面接受风险（44 条）

### Session / SharedState

#### P1-01 · 裸 `--resume` 跨模型选择全局 latest

共享 `$USER_DATA_PATH` 下会扫描所有模型目录并选择时间最新的 session，
随后 pin 给全部子进程。并发运行时可恢复到另一模型或另一操作者的会话。

- 锚点：`src/hyperloom/inference_optimizer/session/paths.py:151-188`
- 锚点：`src/hyperloom/inference_optimizer/cli/__init__.py:1608-1634`
- 建议：要求 `--resume-from`/launch-info；不唯一时 fail closed。

#### P1-02 · fresh session 只有秒级目录名，且 workspace root 可为相对路径

同模型同秒启动会命中同一目录；相对 `USER_DATA_PATH` 被写入 session env 后，
子进程 cwd 不同会解析到另一位置。

- 锚点：`src/hyperloom/inference_optimizer/session/paths.py:100-102,205-221`
- 建议：使用微秒+随机/UUID；创建后 pin 绝对 resolved path。

#### P1-03 · 空 task id 统一落到 `unknown` 工作目录 —— 已修复，实际定级 P3

**已修复**（`refactor(session): reject blank path-id components instead of a shared
placeholder`）。**实际定级 P3 而非 P1**：原描述的危害不可达。

同一份回退服务三个落点 —— `runs/<action>/unknown/`、`patches/unknown/`、
`kernel-agent/runs/unknown/`（原条目只写了第一个）。但空 ID 打不到该分支：注册表
task id 是 `uuid.uuid4().hex`（`state/task_registry.py:175`），其余 25 个调用点各自
或替换占位符、或提前返回、或结构上不可能为空；占位符唯一的消费者是 3 条测试断言，
无生产读者也无 docs/prompt 引用，属反向腐坏而非可达缺陷。

真正的缺陷是为保留该回退而在 containment helper 里加的
`if v == "unknown": return v` —— 一条**跳过 traversal 检查**的提前返回。

- 锚点（修正）：`session_paths.py:99-110`；原锚点 `137-169` 指向 docstring。
- 处置：空 ID 与 path-like 合并为单条拒绝谓词，旁路分支随之消失；
  `integrate_patch.py` 的三个同类死占位符一并删除。
- 陷阱：朴素删除回退会让空 ID 返回 `""`，pathlib 吞掉空段后落到 action **根**目录，
  比原缺陷更糟 —— 必须替换为显式拒绝，不能直接删。
- 关联：同族占位符 `anon`（`kernel/request_handlers.py:1053`）与 `warm_kernel`
  （`phases/prelude.py:439`）**可达且承重**，落在 REVERT 备份上，另记 `pr.issue.md`。

#### P1-04 · `failed` 不在真正的 terminal state 集合

TaskRegistry 文档与迁移逻辑都把 failed 当终态，但 `TERMINAL_STATES` 漏掉它。
失败的 revalidation、auto-roofline 或 framework task 会被后续逻辑视为仍运行，
pending gate 无法清理。

- 锚点：`src/hyperloom/orchestrator/state/task_registry.py:6-13,39-47`
- 锚点：`src/hyperloom/orchestrator/loop/writeback.py:4202-4237`
- 建议：把 failed 加入 terminal；保留独立的 pruning/retention 集。

#### P1-05 · Robustness runtime 重建 SharedState 时丢 12 个字段

fallback reconstruction 只复制 7/19 个 snapshot 字段，可能丢失 stop reason、
budget、validated gain、kernel attempts 等，导致本轮 diagnosis 与 host 状态不一致。

- 锚点：`src/hyperloom/agents/robustness/runtime/cli.py:202-220`
- 建议：对原 snapshot 使用 `dataclasses.replace()`，不要手写字段子集。

#### P1-06 · Resume 配置重建忽略 `unset_envs`，异常 event scan 可误回滚 KEEP

resume materialization 不应用 env 删除，空 replace args 又会保留旧值；
pending-integrate event scan 异常被吞后仍按“没有 KEEP”执行 reverse rollback。

- 锚点：`src/hyperloom/orchestrator/loop/writeback.py:3719-3749,4105-4130`
- 建议：使用唯一 config materializer；无法读取权威事件时禁止破坏性回滚。

#### P1-07 · Objective 可用未验证的累计增益触发 `target_reached`

`TargetGainObjective` 直接消费 cumulative gain；Coordinator 据此终止。
如果各轮百分比不可组合或含未验证值，会把未达到目标的会话标成完成。

- 锚点：`src/hyperloom/orchestrator/state/objective.py:148-158`
- 锚点：`src/hyperloom/orchestrator/loop/coordinator.py:1699-1708`
- 建议：只消费 validated、同一 baseline 上可复算的 throughput 增益。

#### P1-08 · phase budget 规范化会“复活”禁用 phase，3 小时门还会直接终止短 run

预算重分配用 0 表示禁用，但 normalize 会丢掉 0 并恢复默认份额；
默认 3 小时门作用于首次 EXPLORE，常见 `--max-hours 2` 可能一进入 EXPLORE 就退出。

- 锚点：`src/hyperloom/orchestrator/phases/machine_state.py:744-810,1089-1113,1272-1339,1979-1990`
- 建议：显式区分 disabled 与 unspecified；3 小时门只约束 reloop。

### Coordinator / Policy / Specialist

#### P1-09 · Intent handler 写回的 response 被 cursor 立即越过

handler 给 source 写 response 后，router 把 cursor 推进到当前最新消息；
下一轮模型可能永远看不到本次请求结果。

- 锚点：`src/hyperloom/orchestrator/loop/intent_router.py:99-111,717-732`
- 锚点：`src/hyperloom/orchestrator/loop/dispatcher.py:85-94`
- 建议：cursor 只推进到已消费的输入 seq，不越过新产生的输出。

#### P1-10 · `kill_task` 后任务仍继续运行并可被 promote

DB 行被标 cancelled，但 runner 不会停止真实 executor；后续 succeeded transition
可被 terminal 容错吞掉，dispatcher 仍可能提升结果。

- 锚点：`src/hyperloom/orchestrator/loop/intent_router.py:813-859`
- 锚点：`src/hyperloom/orchestrator/loop/sub_agent_runner.py:300-310`
- 锚点：`src/hyperloom/orchestrator/loop/dispatcher.py:1000-1009`
- 建议：把取消 token 传入 executor；任何 cancelled task 的结果都不得 promote。

#### P1-11 · Lease 获取后缺少统一异常/取消清理

queued→running、policy-denied transition、GPU/Ray acquire 等部分路径位于统一
`try/finally` 之外；`CancelledError` 也不被 `except Exception` 捕获。
任务可永久停在 running，lane/GPU lease 留到 TTL。

- 锚点：`src/hyperloom/orchestrator/loop/sub_agent_runner.py:210-315`
- 锚点：`src/hyperloom/orchestrator/loop/dispatcher.py:383-550`
- 建议：一旦接管任何 lease，立即进入单一 finally；取消必须写 terminal 状态。

#### P1-12 · Targeted-build spawn 后状态写失败会遗留进程和 lease

subprocess 已启动，但 task 转 running 或 state save 失败时，handle 尚未进入可 reap 集合；
终态 transition 失败也会留下 running row 与 stale sentinel。

- 锚点：`src/hyperloom/orchestrator/loop/build_lifecycle.py:103-156`
- 建议：先持久化 spawn intent，再启动；所有中间失败都 kill/release/写终态。

#### P1-13 · 生产 PolicyGate 没有注入 ActionRegistry

Coordinator 构造 gate 时 registry 尚未加载，后续也没有回填。
未知动作检查和派发前纵深防御处于 permissive 模式。

- 锚点：`src/hyperloom/orchestrator/loop/coordinator.py:711-715,875-880`
- 建议：先加载 registry，再构造 gate；启动时断言 gate 持有同一 registry。

#### P1-14 · Robustness delegate 权限未按角色真正窄化

角色元数据与注释宣称 Robustness 只能做有限 UPDATE/DELEGATE，
但 PolicyGate 除 `recover` 特例外，没有限制其派发其它当前 phase 可执行动作。

- 锚点：`src/hyperloom/orchestrator/policy/gate.py:900-976`
- 锚点：`src/hyperloom/orchestrator/roles/agent_role.py:88-96`
- 建议：PolicyGate 必须消费角色 allowed-intents/actions，而不是只依赖 prompt。

#### P1-15 · readonly/research specialist 仍拥有 Edit/Write/Bash，Codex 还忽略 tool allowlist

`readonly=True` 只改变 profile/是否建 worktree，不机械剥离写工具；
Claude 默认可写，Codex launch 完全不消费 `allowed_tools`。

- 锚点：`src/hyperloom/orchestrator/specialists/profile.py:225-231`
- 锚点：`src/hyperloom/orchestrator/specialists/runner.py:683-685,1473-1477`
- 锚点：`src/hyperloom/orchestrator/specialists/subprocess_.py:811-826,912-919,1263-1272`
- 建议：runtime 层实施 deny-by-default tool policy；readonly 任务使用文件系统隔离。

#### P1-16 · Specialist 默认继承控制面秘密

subprocess 默认继承多类 API key/header，再结合 Bash、外部 KB/PR prompt，
形成 prompt-injection 到凭据访问的直接通道。

- 锚点：`src/hyperloom/orchestrator/specialists/subprocess_.py:166-180,429-445`
- 建议：子进程环境从空 allowlist 构造；仅注入任务明确需要的短期凭据。

#### P1-17 · 不可信任务数据被提升到 system prompt

Claude subprocess 把 system 与包含 KB/PR/notes 的 user prompt 合并后整体传入
`--system-prompt-file`，不可信内容获得 system 层优先级。

- 锚点：`src/hyperloom/orchestrator/specialists/subprocess_.py:923-925,1398-1402`
- 建议：system/user 分通道传输；外部材料加明确 data delimiter，不进入 system prompt。

#### P1-18 · Codex/Claude intent 与工具契约已漂移

Codex orchestration 输出说明只列少量 intent，漏 delegate/request/kill/lease/prune 等；
Claude specialist 后缀漏 `specialist_done`；Codex 的 tools/max_turns 只写 metadata，不控制 SDK。

- 锚点：`src/hyperloom/orchestrator/roles/codex.py:54-85`
- 锚点：`src/hyperloom/orchestrator/roles/claude.py:99-104`
- 锚点：`src/hyperloom/orchestrator/roles/codex_agent.py`
- 建议：从同一 protocol schema 生成 prompt、MCP schema 与 runtime validator。

### Kernel / Framework / Patch

#### P1-19 · Kernel 非 KEEP 路径不检查 revert 是否成功，仍返回 `status="ok"`

补丁可能留在 live tree，但状态宣称已回滚；paired A/B re-apply 失败时，
decision 还可能保留 KEEP 并 finalize 一个已回滚 manifest。

- 锚点：`src/hyperloom/orchestrator/kernel/request_handlers.py:6674-6780`
- 建议：revert/re-apply 失败必须 terminal failed，禁止 finalize/KEEP。

#### P1-20 · GEAK handoff 使用错误 framework/TP 来源，rebench 又不在 KERNEL inflight allowlist

handoff 忽略 SharedState，回落到 env 或硬编码 `sglang/1`；
rebench 使用 `explore` task，但 KERNEL allowlist/inflight 不含它，idle guard 可提前退出。

- 锚点：`src/hyperloom/orchestrator/phases/kernel.py:614-664,753-782`
- 锚点：`src/hyperloom/orchestrator/phases/machine_state.py:90-102,2063-2092`
- 建议：所有 handoff 参数来自持久化 state；为 rebench 定义合法内部 action。

#### P1-21 · GEMM promotion 整体覆盖 `current_best`

promotion 没有合并此前 KEEP 的 server args/envs/patch metadata，
会丢失已经验证的配置并让后续 benchmark 使用不完整 stack。

- 锚点：`src/hyperloom/orchestrator/phases/kernel.py:2043-2056,2310-2323`
- 建议：以 stack delta 合并，禁止整对象覆盖；promotion 后立即重算 fingerprint。

#### P1-22 · Kernel source-root containment 使用字符串包含关系

只专门拒绝 `..`，任意包含白名单片段的外部路径可能通过。

- 锚点：`src/hyperloom/orchestrator/kernel/request_handlers.py:263-283,877-881`
- 建议：对 resolved path 使用 `relative_to()`；拒绝 symlink escape。

#### P1-23 · Forge GEMM KEEP 修改全局 aiter 配置

KEEP 路径直接覆盖全局安装的 aiter 配置，跨 session/并发运行相互污染。

- 锚点：`src/hyperloom/orchestrator/kernel/request_handlers.py:3443-3468`
- 建议：改为 session-local overlay/patch artifact；使用进程级锁并保存原值。

#### P1-24 · Targeted-build freshness 结果未参与成功判定，dotted symbol 校验错误

recipe 调用 freshness verifier 后不检查 `verified`，仍返回 `ok=True`；
`verify_symbols("aiter.ops.fp4_moe")` 又使用单次 `getattr`，不会遍历 dotted path。

- 锚点：`src/hyperloom/orchestrator/framework/targeted_build.py:519-520,726-727,1008-1009`
- 锚点：`src/hyperloom/orchestrator/framework/build_utils.py:353-358`
- 建议：freshness/symbol verification 都必须是成功 gate。

#### P1-25 · Framework patch executor 的回滚、提交与下载边界不安全

首个 no-git patch 部分失败时 `applied=[]`，回滚直接返回；
KEEP 使用 `git add -A` 会提交并发改动；任意 `diff_url` 交给 `curl -L`，无 scheme/host allowlist。

- 锚点：`src/hyperloom/orchestrator/actions/executors/framework_agent.py:161-179,742-760,1101-1106`
- 建议：按 touched paths 事务化；下载只允许受信 host；任何 partial apply 都恢复备份。

#### P1-26 · Benchmark workload env 只校验变量名形状，危险变量可进入子进程

来自 LLM/KB 的 env 会放行 `LD_PRELOAD`、`BASH_ENV`、`PYTHONPATH`、`PATH` 等；
最终过滤主要针对凭据，而不是 loader/启动 hook。

- 锚点：`src/hyperloom/orchestrator/actions/executors/_workload_envs.py`
- 锚点：`src/hyperloom/common/env_safety.py`
- 建议：不可信 env 使用 `BLOCKED_UNTRUSTED_ENV_NAMES` 和 allowlist 双重过滤。

### Multi-node

#### P1-27 · Pod Python override 进入未引用的远端 shell

`HYPERLOOM_MN_POD_PYTHON` 作为 interpreter 直接拼接进远端 shell，
可包含 shell metacharacters。

- 锚点：`src/hyperloom/inference_optimizer/multi_node/commands/infera.py:768,774-779`
- 锚点：`src/hyperloom/inference_optimizer/multi_node/_internal/ssh_client.py:252-255`
- 建议：interpreter 必须是受控 argv/绝对路径，禁止任意 shell fragment。

#### P1-28 · known_hosts 使用未经认证的 TOFU，mismatch 后自动信任新 key

首次连接直接 `ssh-keyscan`；host-key mismatch 后又自动刷新并重试，
等价于信任变化后的 key。

- 锚点：`src/hyperloom/inference_optimizer/multi_node/_internal/ssh_known_hosts.py:67-93`
- 锚点：`src/hyperloom/inference_optimizer/multi_node/cli.py:428-436,478-486`
- 建议：要求预置 fingerprint/CA；mismatch 必须人工确认，不能自动替换。

#### P1-29 · 跨 pod PID 被当成本地 PID 回滚，vLLM worker 还返回 PID 0

driver 对远端 pod PID 调本地 `os.killpg`；PID namespace 不同会杀不到目标，
甚至杀本地同号进程。PID 0 会指向 driver 当前进程组。

- 锚点：`src/hyperloom/inference_optimizer/multi_node/scripts/launch_multinode.py:608-616,1160-1174`
- 建议：远端生命周期全部经 pod RPC；本地不得解释远端 PID。

#### P1-30 · Multi-node kill/reaper 范围过宽且失败仍成功

launcher 按全机 `/proc` cmdline 模式 TERM/KILL，只排自身/PID1；
`kill_multinode.py` 即使 still_alive、ports_busy、gpu_busy 或 actor timeout 也可返回 0。

- 锚点：`src/hyperloom/inference_optimizer/multi_node/scripts/launch_infera_node.py:329-383,474-532`
- 锚点：`src/hyperloom/inference_optimizer/multi_node/scripts/kill_multinode.py:597-637`
- 建议：只回收 session 记录的 PID+start time/cgroup；任何残留返回非零。

#### P1-31 · 终态失败与配置错误的退出码语义互相颠倒

多个 terminal launch/router failure 最终落到 transient 1；
某些单节点 PD 配置错误却返回 terminal 2；另有基于错误文案子串的分类。

- 锚点：`src/hyperloom/inference_optimizer/multi_node/cli.py`
- 建议：使用 typed exception 和统一常量，删除文案分类。

#### P1-32 · PD 参数未完整转发，resume fast path 也不比较拓扑

RayJob builder 漏 per-role PD 参数，单节点 builder 漏 EP；
resume 只比较部分字段，可复用错误 nodes/TP/transfer/backend 拓扑。

- 锚点：`src/hyperloom/inference_optimizer/multi_node/cli.py`
- 锚点：`src/hyperloom/inference_optimizer/multi_node/commands/infera.py`
- 建议：构建完整 topology fingerprint；任何字段不同都禁止 fast resume。

### Breakdown / Report

#### P1-33 · Producer 与 TypedDict/wire schema 大面积漂移

producer 输出 `session_meta`、final/capability/kernel/roofline/KB 等多项未声明字段；
同一字段在不同路径还有不同形状。消费者无法依赖静态 schema。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/schema.py`
- 锚点：`src/hyperloom/inference_optimizer/breakdown/collectors/`
- 建议：以运行时 schema 生成 TypedDict 或反向生成 producer validator；CI 做 producer→schema 对拍。

#### P1-34 · Session package manifest 可宣称包含实际写入失败的文件

单文件 `zf.write()`/loose copy 失败后继续，manifest 仍记录文件；
允许路径中的 symlink 也可跟随到 session 外。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/session_package.py:142-178,400-407`
- 建议：只在成功写入后加入 manifest；resolved path 必须位于 session 下。

#### P1-35 · Verbose report 可读取 session 外任意文件

`session_root / rel_path` 接受绝对路径和 `..`，可把 session 外日志嵌入报告。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/`
- 建议：所有 artifact path resolve 后执行 `relative_to(session_root)`。

#### P1-36 · Breakdown/export 的“never raises”与隔离边界不成立

多个 minimal-final、collector、renderer 在 stat、类型转换、写盘时仍可抛错；
renderer 列表推导没有逐 section 隔离，一条坏数据可终止整份报告。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/exporter.py`
- 锚点：`src/hyperloom/inference_optimizer/breakdown/reporters/compose.py:138`
- 建议：参数求值也放入隔离 wrapper；每个 section 独立 warning/failure。

### Knowledge / Robustness / Recovery

#### P1-37 · Recipe canonical ID 可形成路径越界

canonical ID 校验没有拒绝 `/`、`.`、`..` 等路径成分；直接传入的 ID
可路由到 store root 之外。

- 锚点：`src/hyperloom/orchestrator/knowledge/recipe_kb/canonical_id.py:97-119`
- 锚点：`src/hyperloom/orchestrator/knowledge/recipe_kb/local_store.py`
- 建议：每个 segment 使用封闭字符集；最终路径再做 containment。

#### P1-38 · Remote recipe/KG 写入可误报成功或覆盖已有页面

Recipe store 忽略 `put_page` 的 in-band error；KG `_node_exists` 把 transport failure
当“不存在”，随后写最小 stub，可能覆盖真实远端页面。

- 锚点：`src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_store.py:339-360`
- 锚点：`src/hyperloom/orchestrator/knowledge/recipe_kb/kg_client.py:542-586`
- 建议：区分 not-found 与 transport failure；所有写入检查 envelope 并返回真实 success。

#### P1-39 · T0 冷启动 cascade 被刚写入的 exact seed 短路

流程先写 exact seed，再执行 cascade；L1 立即命中新 seed 并提前返回，
冷启动时 L2/L3 跨模型检索实际上不可达。

- 锚点：`src/hyperloom/orchestrator/knowledge/recipe_kb_t0.py:908-915,1140-1203`
- 建议：先执行已有知识 cascade，再决定是否 seed；seed 不参与本次查询。

#### P1-40 · Robustness stale-lease 查询使用不存在的列

探针查 `holder_pid`，真实 `leases` 表只有 `pid`；SQLite 错误被吞成空列表，
I3 stale-lease 在生产上恒拿不到数据，测试因自建错误 schema 而假绿。

- 锚点：`src/hyperloom/agents/robustness/sources/local_probe.py:1715-1749`
- 锚点：`src/hyperloom/orchestrator/bus/storage/schema.py:44-54`
- 建议：查询 `pid AS holder_pid`；测试必须使用真实 `ensure_schema()`。

#### ~~P1-41 · External mount timeout 没有真正的 timeout~~ ✓ 已关闭（删除方式）

**处置**：`external_mount_stat_timeout_s`（`LocalProbeConfig`、`Config`、`factory.py` plumbing 行）与
`_probe_external_mounts` 的 `timeout_s` 形参和 `"timeout_ms"` 输出字段已整体删除；
`_probe_external_deps` 对应实参已同步移除。mount 延迟预算收敛到唯一有执行体的一份：
`ExternalDepsConfig.mount_latency_warn_ms / mount_latency_critical_ms`（`signals/external_deps.py:35-36`）。

`test_probe_external_mounts_records_latency` 新增 key-set 断言，锁死消费者可见的字段契约
`{env_name, path, ok, error, latency_ms}`，阻止死字段回流。

**原条目更正**：

1. **锚点漂移**：原锚点 `local_probe.py:2014-2059` 已漂移；实际缺陷位置为 `:2146-2193`（`_probe_external_mounts`）与 `:2028-2064`（`_probe_external_deps`）。

2. **影响被高估**：原描述「整个 robustness tick 可无限阻塞」在生产路径不成立。生产路径经
   `roles/robustness_agent.py` → `_runtime_bridge.py:invoke_runtime_cli`，以
   `subprocess.run(timeout=30)` 驱动每个 tick，挂死的 tick 以 `BackendError` 上报。
   无界的只有 `main.py` 的 standalone 循环，而其 docstring 自述「Production hosts drive
   the same reactor via `runtime.cli` in a subprocess instead」，并已列入
   `pyproject.toml` coverage omit。

3. **原建议不可实现**：「放入可取消线程」在 CPython 中无效——阻塞在系统调用的线程不可取消；
   且 `runtime/cli.py:217` 的 `asyncio.run()` 在退出时会等待泄漏的 worker 线程，
   `emit_json` 无法写出，tick 照样被 Coordinator 杀掉。子进程方案虽可行但成本过高，
   此缺口由外层边界兜底已足够。

4. **文档漂移**：已删除的 `_probe_external_mounts` docstring 曾声明「echoed back as
   `timeout_ms` so the signal layer can flag slow mounts」——signal 层实际读的是
   `ExternalDepsConfig`，与声明不符。

**残留缺口（有意接受）**：mount 永久挂死（D 态进程不返回）时，`latency_ms` 写不出来，
不触发 `wekafs_degraded`，只由 Coordinator 的 30 s 边界以 `BackendError` 报出。

#### P1-42 · Coordinator events 方向相反且双源不去重

LocalProbe 返回 `seq DESC`，多个 detector 按时间正序消费；
inbox 与 DB 又可能是同一消息的两个视图，normalize 后丢失可去重 ID，
失败 streak 可能被掩盖或提前翻倍。

- 锚点：`src/hyperloom/agents/robustness/sources/local_probe.py:320-370`
- 锚点：`src/hyperloom/agents/robustness/signals/critic_health.py:168-195`
- 锚点：`src/hyperloom/agents/robustness/signals/repeated_payload.py`
- 建议：保留 msg_id/seq，先去重再按统一单调序号升序处理。

#### P1-43 · Postmortem 部分失败仍写完成 marker

两份输出只成功一份或全部失败时仍可能写 marker；下次 finalize 会跳过，
失败的文件永远不补写。marker 本身也不是原子写。

- 锚点：`src/hyperloom/agents/robustness/role/postmortem.py:169-179,508-533`
- 建议：全部必需输出成功后才原子写 marker；部分成功允许幂等重试。

#### P1-44 · IR-1 preflight 与 recover 完成判据均可假成功

`preflight_optimizer` 只打印显存信息，不解析 `<=500 MiB`，命令失败也不置失败；
recover 的 `looks_complete` 又忽略 breakdown/counts，缺最终产物时仍可返回 0。

- 锚点：`src/hyperloom/inference_optimizer/tools/preflight_optimizer.py:64-105,130-141`
- 锚点：`src/hyperloom/inference_optimizer/cli/recover.py:49-63,97-99`
- 建议：把 IR-1 与恢复产物定义做成可执行 validator，任何未知状态 fail closed。

#### P1-45 · MoE shape 不对齐时 aiter args 防护已写好但从未接线

`strip_aiter_args_for_unaligned_moe`（`orchestrator/actions/executors/_grid_server_args.py:1052`）
实现了"MoE shape 不对齐时剥除 aiter server args"的安全约束，函数内部日志写着
"aiter's asm MoE path faults on this shape"，但该函数从未被串进 `compose_server_args`
（`:86-187`）的生产路径。后者的五个生产调用点全部走不含保护的路径，导致该形状的 server
启动可能崩溃。接线需给 `compose_server_args` 增加 model/tp 形参并改五个调用方，且会真的
开始剥 args，属于行为变更。

- 锚点：`src/hyperloom/orchestrator/actions/executors/_grid_server_args.py:1052`
- 锚点：`src/hyperloom/orchestrator/actions/executors/_grid_server_args.py:171`（`compose_server_args`）
- 定级：P1（任务可能因服务器启动崩溃被破坏）

#### P1-46 · Ray head 进程退出后不会被停止

`stop_ray_if_owned`（`src/hyperloom/agents/kernel/tools/backends/ray_runtime.py:307`）设计为
配合 `ensure_ray_cluster` 使用的所有权感知停止函数，但零生产调用。`ensure_ray_cluster` 的
唯一生产调用点 `orchestrator/actions/executors/_ray_backend.py:256` 丢弃了返回值（即所有权
标志），导致任何后续停止调用都无法知道本次是否启动了 head。进程退出后 Ray head 持续占用
GPU/内存资源。修法：在调用点接住返回值，在 executor 退出时调 `stop_ray_if_owned`。

- 锚点：`src/hyperloom/agents/kernel/tools/backends/ray_runtime.py:307`（`stop_ray_if_owned`）
- 锚点：`src/hyperloom/orchestrator/actions/executors/_ray_backend.py:256`（唯一生产调用，丢弃返回值）
- 定级：P3（资源泄漏，不直接破坏任务正确性）

---

## P2 — 首个补丁窗口（50 条）

### Common / shared contracts

#### P2-01 · 模型 identity 丢失 org，跨组织同名模型被判相同

`model_identity_candidates()` 对普通 `org/repo` 字符串走 `Path(raw).name`，org 在候选构造时即丢失；
`a/Llama-8B` 与 `b/Llama-8B` 实测 `match=True`，而 `model_identities_match` 的 docstring 明确承诺 `False`。
防护逻辑（`model_paths.py:85-86` 的 org 双侧存在性检查）自引入 commit `f0931199e` 起即因候选集无 org 而从未触发。
全仓零测试覆盖该属性。下游影响：`cli/model_gate.py:400` 的陈旧性闸门 fail-open，但下游仅是 prompts-only advisory arch profile。

- 锚点（修正）：`src/hyperloom/common/model_paths.py:44-63`（`model_identity_candidates` 的 `Path(raw).name` 路径）
- 已修复：`f0931199e` 引入，本批 Stage 7 修正。

#### P2-02 · `jsonio` fenced JSON 抽取与 tolerant read 的类型契约不闭合

**原描述机制错误**：非贪婪 regex 对嵌套对象实测正常（`{"a":{"b":1}}` 解析成功）。
真实缺陷是反向的：fence 内在合法对象之后存在含 `}` 的散文，或 fence 内有两个 JSON 对象时，`extract_first_json_with_key` 返回 `None`——一个本来完好的回复被丢掉。
另一缺陷：`coerce_dict` 的 `path.is_file()` 在超长路径上抛出 `OSError` errno 36，穿越 tolerant 边界（同一路径下 `read_json` 正常返回默认值）。

- 锚点（修正）：`src/hyperloom/common/jsonio.py:14,147-200`（fence regex + bare_re 收缩循环）；`jsonio.py:141-144`（`coerce_dict`）
- 建议：用 `json.JSONDecoder().raw_decode` 扫描代替正则+收缩；`coerce_dict` 包裹 `is_file()` OSError。

#### P2-03 · 公共数值 helper 接受 bool/Inf 或在混合类型排序时崩溃

`gain_math` 会把 bool 当吞吐/并发，正无穷也可进入统计；混合类型 conc 在排序时报错。

- 锚点：`src/hyperloom/common/gain_math.py:11-15,55-68`
- 建议：统一拒绝 bool 与非 finite 数值，并在入口规范化 conc。

#### P2-04 · profile args 对坏引号 fail-open，并可吞掉下一个合法 flag

未闭合引号会原样降级；需要值的被删除 flag 若缺值，会把后续合法 option 当成值一起吞掉。

- 锚点：`src/hyperloom/common/profile_args.py`
- 建议：parse error 显式失败；只在确认 token 不是 option 时消费 value。

#### P2-05 · Kernel source document validator 没有验证完整 identity/schema

重复或空 `kernel_id`、字符串 `source_line`、非 finite `gpu_pct` 等可通过；
root 为 `/` 时 source containment 也可退化（`rstrip(os.sep)` 产生 `""`, `"".startswith("")` 对所有字符串为 True）。

**已部分修复**：`confidence` 字段的 bool 排除 + `isfinite` + `[0,1]` 值域校验已在 `e4852a96c` 中完成，本条目未反映。

- 锚点（修正）：`src/hyperloom/common/kernel_source_contract.py:156-210`（validator）；`kernel_source_contract.py:262-267`（containment rstrip bug）
- 建议：kernel_id 唯一性 + 非空、gpu_pct finite、source_line int-or-None；containment 改用 `Path.relative_to`；同时修 `make_entry` 生产者。

#### P2-06 · LLM gateway URL、响应 shape 与 token usage 缺少统一校验

DeepSeek anthropic URL 可派生到错误路径；合法 JSON list 被当空成功响应；
token counter 接受 bool/负数。

- 锚点：`src/hyperloom/common/llm_config.py:331-360,839-844,955-968`
- 建议：按 provider endpoint family 显式映射；响应必须为 object；usage 非负且拒绝 bool。

### Agents

#### P2-07 · Framework unified diff 对删除文件和路径 containment 处理错误

标准删除 diff 的 `+++ /dev/null` 可覆盖真实路径；绝对路径和 `..` 也可能逃出 root。

- 锚点：`src/hyperloom/agents/framework/_audit_common.py:60-89,115-120`
- 建议：分别保存 old/new path；所有目标 path resolve 后校验 root。

#### P2-08 · PR-KB backend 忽略 `pr_states`，relevance 也未形成稳定排序

默认 open-only 请求仍可返回其它状态；缺失/错误 relevance 绕过 floor，
结果按 PR 号而非 relevance 排序。

- 锚点：`src/hyperloom/agents/framework/sources/pr_kb.py:48-140`
- 建议：统一 backend filter contract，并保留 relevance 到 Candidate score。

#### P2-09 · Framework 指标接受 NaN/Inf，异常候选可成为 winner

`_metric_float` 没有 finite 校验；NaN throughput 实测产出 `winner=True`，reason 字段写 "throughput and accuracy gates passed"，违反验收第 6 条。

**锚点错误**：条目锚定的 `evaluate_candidate_outcome`（`tools_api.py:187-202`）没有生产调用者，为对外 API。
生产孪生是 `decision.py:161-191` 的 `winner_decision`，两处共用同一个 `_metric_float`，修一处两处同时修。

**定级应为 P1**（错误 benchmark 结果被当成成功，违反验收第 6 条）。本轮维持 P2 处置（manual-only promotion 作为书面理由），但本批 Stage 3 已修复。

- 锚点（修正）：`src/hyperloom/agents/framework/runtime/tools_api.py:187-202`（`_metric_float`，已删除）；`src/hyperloom/agents/framework/decision.py:183-191`（生产 winner gate）；`src/hyperloom/agents/framework/explorer.py:579-580`（`_evaluate_candidate`）

#### ~~P2-10~~ → P3 · Framework shell template 进行多轮替换，值会被二次解释

前一个变量值中包含 `{later_key}` 时会在后续轮次再次替换，破坏单变量单次引用语义。

**定级降为 P3**：实测证明 `shlex.quote=True`（生产路径唯一调用形态）未被击穿——注入 payload 仍保留在单个 shell word 内，bash 不执行额外命令。生产变量值（`PR:1234`、路径字符串）不含 `{}`，实际路径无触发。本批 Stage 10 已作为正确性修复处理（单遍渲染）。

- 锚点：`src/hyperloom/agents/framework/shell.py:69-101`（已修复）

#### ~~P2-11~~ → P1 · Critic 多 verdict 提交不是事务，且 target 未绑定本轮 proposal（已修复）

**重定级 P1**：实测第一条 verdict `mark_reviewed` 落盘后第二条抛 `ReviewValidationError`，
intent_envelope 为空、下一轮 `filter_unreviewed` 过滤掉已标记 msg_id——判决永久丢失，属"报告被遗漏"。

**处置（收敛）**：拆成两趟：第一趟只做校验+构建 intent，全部通过后第二趟统一落盘副作用。
同批内重复 `target_proposal_msg_id` 被拒绝。

- 锚点（修正）：`src/hyperloom/agents/critic/runtime/decision_reviewer.py:889-938`（原 `964-1011` 指向另一函数 `_commit_decision_request`）

#### ~~P2-12~~ → P3 · Critic prior cache key 漏 `kind/limit`（已修复）

**降 P3**：生产路径两个 `list_priors` 调用点显式传 `kind=None`、`limit=prior_limit`（同一常量），
缓存冲突只在 `CRITIC_KB_CLIENT_MODE=live` 且 operator 用 `hyperloom-critic list-priors --kind` 命令时可达。

**处置（删除死形参 + 收敛）**：删掉 `metadata_filter`（零调用方传递），把 `kind` 与 `limit` 折进
`scope_cache_key`（`scope_builder.py:183-199`），`list_priors` 不再传 `metadata_filter` 给 KB 客户端。

- 锚点（修正）：`src/hyperloom/agents/critic/runtime/scope_builder.py:183-199`；`src/hyperloom/agents/critic/runtime/kb_writer.py:200-208`

#### P2-13 · SessionMemory 合法非 object JSON 会破坏调用方（已修复）

**补充**：条目说"`null`/list"，实测零字节文件是最常见触发（`_read_json` 透传 `empty_value=None`，
随后 `merge_context` 对 `None` 迭代抛 `TypeError`）。五个落点三种不同待遇。

**处置（删除重复 helper + 接线）**：删除本地 `_read_json`，改用
`jsonio.read_json(path, default={}, require_dict=True, strict=True)`，
捕获 `ValueError` 包成 `SessionMemoryError`。删掉 `mark_reviewed` / `filter_unreviewed` 的冗余 `isinstance` 守卫。

- 锚点（修正）：`src/hyperloom/agents/critic/runtime/session_memory.py:471-489`（原 `506-524` 超出文件末尾）+ 落点 `244,375,407`

#### ~~P2-14~~ → P3 · Dead-letter replay 对合法非 object JSON 整批崩溃（已修复）

**降 P3**：DLQ 文件只由 `DeadLetter.append()` 写入（永远是 object 行），触发需手工编辑队列文件。
`CRITIC_KB_CLIENT_MODE=live` 默认不启用，DLQ 实际生产中很少填满。

**处置（接线）**：把 object 判定移进 try、非 object 行走既有的"失败行保留"路径，与畸形 JSON 一致。

- 锚点：`src/hyperloom/agents/critic/runtime/dead_letter.py:168-186`（不变）

#### ~~P2-15~~ · TraceLens composite route 丢失路由元数据（STALE）

**STALE**：`OpResolution.leaf_resolutions()` 用 `replace(self, target_index=i)` 整体复制，
原描述的"子集复制"形态不存在。而且 `fanout`/`kernel_kinds`/`prebuilt_binaries`/`runtime_backends`
四个字段从未被生产路径填充（两个构造点都是 `kind="single"` + 空 list），已整体删除。

- 处置：STALE，随 `remove(kernel): drop inert OpResolution per-source metadata` 删除死字段。

#### ~~P2-16~~ → P3 · TraceLens `--top-k` flag 与 CLI help 语义不符（已修复，降 P3）

**降 P3**：`--top-k` flag 从未被 `request_handlers.py` 转发（转发条件 `payload["top_k"] is not None`，
而全仓无任何编排代码写该键）。生产唯一入口是 `HYPERLOOM_KERNEL_CANDIDATES_TOP_K` 环境变量，
其 `_default_top_k()` 实现已正确映射 `0/负数 → 1_000_000`。

**处置（删除）**：删掉 `--top-k` argparse 定义和 `request_handlers.py:5382-5384` 的转发分支；
四个内部使用点改为直接调用 `_default_top_k()`。

#### ~~P2-17~~ · TraceLens 目录把任意 JSON 当 trace，且只探测首文件（STALE）

**STALE**：已由两处改动覆盖：（1）`discover_trace_inputs` 按 size 降级 splitter 碎片和 annotation
sidecar，使真实 capture 排首位；（2）`_KERNEL_PROBE_LIMIT = 8` 候选探测循环选取首个真带 GPU kernel
的文件。定向测试 `test_a_non_trace_sidecar_does_not_lead_discovery`（前失败后通过）已在仓库中。

- 修正锚点：`discover_trace_inputs:1211-1247`；`probe loop:7557-7592`

#### P2-18 · Forge backend 修改 git config/info-exclude 且不恢复（已修复）

**撤回子 claim**："staged/unstaged 区分丢失"是刻意取舍——
`test_forge_long_horizon_cli.py:608-616` 注释明确写着 deliberately collapsed。

**实际缺陷**：`user.name/email` 永久写进活仓库 `.git/config`；`info/exclude` 恢复只在
`if inplace:` 分支调用，linked worktree 路径从不恢复，而 `--git-common-dir` 指向主仓库。

**处置（消除写入）**：给 forge-loop 子进程 env 注入
`GIT_AUTHOR_NAME`/`GIT_COMMITTER_NAME`（已被 `forge_collective.py:87-90` 使用的现有机制）；
Hyperloom 侧唯一提交改用 `git -c user.name=forge-bot -c user.email=forge-bot@local commit`；
`_restore_generated_driver_exclude` 调用移到 `_finalize_forge_workspace` 的 `if inplace:` 之外。

- 锚点（修正）：`forge_submit.py:589-590`（已删）、`:1101-1102`（已改）、`:3609`（恢复调用提前）

#### P2-19 · GEAK/Ray backend 对 JSON、端口和恢复状态缺少值域校验（已修复）

**补充实测**：`GEAK_FLUSH_GRACE_S` 无人设置；负值让 `inner_timeout` 反超外层硬杀，
且 `communicate(timeout=负数)` 立即超时，result.json 来不及刷写。
非 dict `result.json`（合法 JSON list/string）让 `result.update()` 抛 `AttributeError`（实测）。

**处置**：
- `geak_runner.py:91`：`GEAK_FLUSH_GRACE_S` 解析失败或非正时回落默认 180。
- `geak_runner.py:130-135`：非 dict `result.json` 走既有的 `status=error` 分支。
- `ray_runtime.py:98`：`HL_RAY_HEAD_PORT` 加 1-65535 值域判定，越界回落 `_free_tcp_port()`。
- `ray_runtime.py:278-282`：`ray start` 返回 0 后补 `ray_status_ok()` 探活。
- `ray_runtime.py:460-465`：版本不匹配重试改为 `address="auto"` 连本地 head，不复用 stale `RAY_ADDRESS`。

- 锚点（修正）：`geak_runner.py:91,130-135`；`ray_runtime.py:97-100,278-282,460-465`

#### P2-20 · Quantization driver 对 manifest、eval 和工具列表的边界不完整（已修复）

**补充发现**：`_has_glob` 用 `directory.glob(pat)` 不过滤目录，名为 `model.safetensors`
的目录实测让 `has_weights=True`（`_has_any` 已用 `.is_file()`，同文件待遇不一致）。

`allowed_tools` 潜在而非活跃：生产路径 `retry.py` 从不传 `allowed_tools` 参数（永远 `None`），
`or` 短路永远走 `DEFAULT_ALLOWED_TOOLS`；但 `allowed_tools=[]` 语义仍需修正。

`eval.py` 只解析 `relative_gap`，不读 `source_score`/`quantized_score`，而 SKILL.md §5.3
明确让 LLM 自己计算该值。`-inf` 实测判 `within`（fail open）。

**处置**：
- `result_collector.py:240`：`_has_glob` 加 `.is_file()` 过滤，对齐 `_has_any`。
- `result_collector.py:193`：`manifest.read_text()` 的 `OSError` 并入 except 子句。
- `runner.py:277`：`allowed_tools or DEFAULT` 改为 `None` 哨兵判定。
- `eval.py`：拒绝非 finite 值；用原始分数复算 `relative_gap`（容差 0.02），不一致判 `missing`。

- 锚点（修正）：`result_collector.py:193-211,229-245`；`runner.py:277`；`eval.py:131-147`

#### P2-21 · Robustness state view 浅拷贝/no-op 语义误导

`DetectorStateView(store=None)` 实际完全 no-op；save/load/snapshot 只复制外层，
嵌套结构可反向修改 store。

- 锚点：`src/hyperloom/agents/robustness/state_store.py:187-277`
- 建议：无 store 时使用真实 in-memory backing；跨边界 deep copy/immutable value。

#### P2-22 · Robustness source router 把认证/服务错误退化为空数据

4xx、fallback、primary health 等多类错误被折叠成 None/empty；
某些 4xx 还会让 Router 永久保持 HEALTHY，阻止降级。

- 锚点：`src/hyperloom/agents/robustness/sources/server_client.py:107-122`
- 锚点：`src/hyperloom/agents/robustness/sources/base.py:204-324`
- 建议：区分 auth/not-found/transient/schema；health 由真实成功响应驱动。

### Orchestrator / state / bus

#### P2-23 · Canonical fingerprint 对顺序敏感参数排序，last-wins 配置会碰撞

重复 flag 的顺序被排序抹掉，不同最终语义可生成同一 fingerprint，
影响 dedup、结果回连和 resume。

- 锚点：`src/hyperloom/orchestrator/actions/executors/_canonical_fingerprint.py:95-109`
- 建议：按 parser 语义规范化；保留重复 option 的有效顺序。

#### P2-24 · Grid runner 可在 subprocess 非零时仍标 succeeded

只要 measurement 可解析，非零 rc 仍可写成功；overtime 等失败又缺统一
`error_class`/abort marker。

- 锚点：`src/hyperloom/orchestrator/actions/executors/_grid_runner.py:1890-1939,2006-2099`
- 建议：成功必须同时满足 rc、measurement、freshness 和 identity。

#### P2-25 · Server teardown 过度信任 pidfile，nogit patch partial apply 也不自动回滚

普通 teardown 可对 stale/reused PID 发信号；nogit 真 apply 失败后 tree 可能已部分修改，
API 只返回 backups，调用方稍有遗漏就留下脏树。

- 锚点：`src/hyperloom/orchestrator/actions/executors/_server_lifecycle.py:244-299`
- 锚点：`src/hyperloom/orchestrator/actions/executors/_nogit_patch.py:349-410`
- 建议：PID + start-time identity；patch API 自身提供事务上下文。

#### P2-26 · Action registry 对 YAML 类型和值域校验不足

未知 key 静默忽略；字符串 list 会被冻结成字符 tuple；成本/risk/turns/TTL
允许负数或非 finite，10 个 `params_schema` 又没有运行时消费者。

- 锚点：`src/hyperloom/orchestrator/actions/registry.py:167-253,300-302`
- 建议：closed schema + typed decoder；删除或真正接线 advisory dead fields。

#### P2-27 · GPU pool 错误显式列表会 fail-open，重复 acquire 也不幂等

显式 pool 全是坏 token 时可能退回默认 mask/range；相同 holder/task 重试会再占一组 GPU。

- 锚点：`src/hyperloom/orchestrator/bus/gpu_pool.py:147-185,263-293`
- 建议：显式配置全坏时 fail closed；holder/task acquire 返回既有 lease。

#### P2-28 · MessageBus replay 与 resource-lock 坏行处理无上限/无隔离

`replay_for()` 一次加载全部未读事件；一条损坏 expires_at 可阻塞整个 acquire。

- 锚点：`src/hyperloom/orchestrator/bus/message_bus.py:235-250`
- 锚点：`src/hyperloom/orchestrator/bus/resource_lock.py:222-231`
- 建议：分页 replay；坏 lease 隔离并发出 repair event。

#### P2-29 · SQLite connection close 可与活动事务并发

`close()` 不持 `_async_lock`，可在 transaction yield 期间关闭同一连接；
初始化失败也不会关闭已创建 connection。

- 锚点：`src/hyperloom/orchestrator/bus/storage/connection.py:72-83,213-252`
- 建议：close 与 transaction 共用生命周期锁；初始化使用 try/finally。

#### P2-30 · Intent validator 只做 key-presence，review verdict 条件 schema 未执行

required 字段不检查类型/空值/枚举；MCP 早校验和最终 envelope validator 不一致，
空 intents 在不同 backend 又有不同语义。

- 锚点：`src/hyperloom/inference_optimizer/protocol/intent.py:107-185`
- 锚点：`src/hyperloom/orchestrator/roles/mcp_emit_intent.py:89-107`
- 建议：使用同一 discriminated-union schema。

#### P2-31 · FRAMEWORK build pump 先 seen 后处理，旧 unseen build 会永久饥饿

只检查最新 terminal build；最新已 seen 时直接返回，不再寻找更旧 unseen；
处理中又在成功前 stamp seen。

- 锚点：`src/hyperloom/orchestrator/phases/framework.py:4424-4432`
- 建议：按队列扫描最旧 unseen；成功持久化后再 ack。

#### P2-32 · FRAMEWORK 直接写非标准 phase_history 行

绕过统一 transition API，也不做容量裁剪，可能破坏 legacy duration reconstruction。

- 锚点：`src/hyperloom/orchestrator/phases/framework.py:3211-3266,3450-3463`
- 建议：所有 phase event 走单一 recorder 和 schema。

#### P2-33 · SWEEP 的 0 秒预算反而关闭预算门

剩余时间 ≤120 秒时 clamp 得 0；下游把 0 定义为“禁用预算限制”，
本应立即跳过却变成无限预算。

- 锚点：`src/hyperloom/orchestrator/phases/sweep.py:121-143`
- 锚点：`src/hyperloom/orchestrator/kernel/conc_sweep.py:12-13,69-77`
- 建议：用 `None` 表示不限；0 明确表示无预算。

#### P2-34 · Roofline baseline/optimized ceiling 可能不可比，within 也不钳制

量化/dtype 改变时理论 ceiling 不恒定，但表格固定选 baseline ceiling；
within 可超过 100% 并生成负 gap，影响饱和判断。

- 锚点：`src/hyperloom/orchestrator/kernel/roofline_snapshot.py:197-222,349-357,560-584,713-715`
- 建议：分别报告两侧 ceiling；不可比时禁止直接 before/after。

#### P2-35 · Grouped kernel rejection 会被 summary 误报为 IN_FLIGHT

group task 被拒时 ledger 刻意不填 `rejected_kernel_ids`，summary 只看该集合。

- 锚点：`src/hyperloom/orchestrator/kernel/_kernel_decisions.py:757-762`
- 锚点：`src/hyperloom/orchestrator/kernel/attempt_summary.py:337-344`
- 建议：summary 使用 task terminal decision，而不是单一辅助集合。

#### P2-36 · Policy freeform wave 会静默跳过坏 item 后整体放行

非 dict 或空 description 被丢弃，只要剩余部分合法就通过；
domain specialist 的负 `max_turns` 也可通过。

- 锚点：`src/hyperloom/orchestrator/policy/gate.py:1687-1708,1921-1936`
- 建议：列表任一坏 item 都拒绝，并返回精确 index。

#### P2-37 · Proposal scorer 对重复 name、首个 JSON 和 bool score 处理不稳健

同名 proposal 静默覆盖；解析首个 `scores` JSON；`float(True)` 得 1.0；
重复 model slug 还会重复付费调用。

- 锚点：`src/hyperloom/orchestrator/scoring/proposal_scorer.py:100-148,229-251,524-561`
- 建议：proposal 使用稳定 ID；拒绝 bool/重复 name/model；只接受完整尾部 envelope。

### Breakdown / report

#### P2-38 · Recorder 的稳定文件 read-merge-replace 跨进程会 lost update

`RLock` 只保护进程内；同 producer/key 的并发进程可互相覆盖。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/recorder/recorder.py:275-357`
- 建议：使用文件锁/SQLite append log；assembler 再做确定性 merge。

#### P2-39 · Recorder seq 与 slug 不是稳定一一映射

无 key item 调 `_next_seq` 两次，文件名与 envelope seq 不同；
`_slug` 非单射，`a b` 与 `a-b` 可覆盖同一稳定文件。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/recorder/recorder.py:149-159,321-344,401-405`
- 建议：一次分配 seq；slug 加稳定 hash。

#### P2-40 · Canonical attribution 丢现代来源

validation/source breakdown 仍只保留旧桶，遗漏 explore/framework/warm replay/
GEAK/Forge/kernel-unattributed 等来源，无法对账总增益。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/collectors/optimizations.py:420-485`
- 锚点：`src/hyperloom/inference_optimizer/breakdown/collectors/attribution.py:247-397`
- 建议：从统一 operation ledger 聚合，不维护手写桶。

#### P2-41 · V4 optimization subject 使用错误字段名

代码按 `subject["kind"]` 判断 kernel，但规范字段是 `subject_type`，
正常 canonical kernel 的 `kernel_id` 会变空。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/collectors/optimizations.py:998-1001`
- 锚点：`src/hyperloom/inference_optimizer/breakdown/schema.py:2412-2416`
- 建议：subject 使用 typed accessor，并加 producer/collector 对拍。

#### P2-42 · Capability/timeline 会重复计数或折叠不同任务

dedup 有 change 时忽略 task_id；KEEP 按 invocation 行而非唯一 kernel 计数；
micro-only KEEP 也被算正式 kept。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/collectors/timeline.py:229-241,351-377`
- 建议：以 stable operation/task/kernel ID 聚合，并保留 validation tier。

#### P2-43 · Renderer registry 包含无 producer section，skipped section 又被完全丢弃

`decision_journal`、`kernel_profiling` 等当前无生产 producer；
skipped 的 warning/key facts 也不会进入最终报告或 LLM prompt。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/reporters/compose.py:215-220`
- 锚点：`src/hyperloom/inference_optimizer/breakdown/reporters/_renderers/`
- 建议：registry 启动时检查 producer reachability；skipped 原因进入 global flags。

#### P2-44 · LLM narrative 没有 section/长度/事实引用校验

prompt 要求“不编造”只是软约束；响应可写未知 section、超长叙述或无证据数字，
仍直接拼入最终报告。

- 锚点：`src/hyperloom/inference_optimizer/breakdown/reporters/compose.py:223-228`
- 锚点：`src/hyperloom/inference_optimizer/breakdown/reporters/llm_prompt.py:137-145`
- 建议：closed section IDs、长度上限、数值引用必须对应结构化 facts。

### Knowledge / trace / multi-node

#### P2-45 · Local recipe/graph lock 与并发更新契约不足

每次操作新建 `_CidLock`，实例 mutex 不能提供同-cid线程互斥；
simulation fact 写又是无锁 read-modify-write。

- 锚点：`src/hyperloom/orchestrator/knowledge/recipe_kb/local_store.py:175-208`
- 锚点：`src/hyperloom/orchestrator/knowledge/recipe_kb/kg_client.py:898-912`
- 建议：按 cid 共享 lock registry；所有 graph mutation 使用原子 append/事务。

#### P2-46 · Local/remote Recipe store 更新与排序语义不等价

local 替换集合，remote 自动 merge 且无法清空旧值；
remote 搜索也未完整实现 local 的 created/version 排序。

- 锚点：`src/hyperloom/orchestrator/knowledge/recipe_kb/local_store.py`
- 锚点：`src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_remote_client.py:1022-1029`
- 建议：定义 backend-independent conformance suite。

#### P2-47 · LLM usage 配对、reasoning token 与 flush 语义不完整

Codex reasoning tokens 在统一 ledger 丢失；token/conversation 按整秒和 key 配对，
跨秒拆分、同秒覆盖；flush 任一子步骤失败仍永久 `_flushed=True`。

- 锚点：`src/hyperloom/orchestrator/trace/parse_usage.py:490-513`
- 锚点：`src/hyperloom/orchestrator/trace/llm_trace.py:189-265`
- 锚点：`src/hyperloom/orchestrator/trace/langfuse_emitter.py:735-754`
- 建议：使用 call_id 配对；flush 仅在全部步骤成功后完成。

#### P2-48 · Langfuse receipt 非原子，却承担跨进程幂等

普通 `write_text()` 在并发/崩溃时可损坏 receipt，导致重复或永久跳过。

- 锚点：`src/hyperloom/orchestrator/trace/langfuse_emitter.py:1288-1304`
- 建议：atomic write + fsync；receipt 内容包含 payload hash。

#### P2-49 · Multi-node patch 原子替换不保留 mode/owner，finalize 也可半完成

`mkstemp` 的 0600 inode 替换目标，执行位/owner 丢失；
finalize 逐条删除，后续失败会留下半完成状态。

- 锚点：`src/hyperloom/inference_optimizer/multi_node/scripts/patch_path_safety.py:268-315`
- 建议：复制原 stat 元数据；finalize 使用 journal/两阶段提交。

#### P2-50 · AgentX vendored fallback 与规范 mapping 在 null 输入上漂移

fallback `_stat` 对 present-but-null 不回落 avg/default，可能泄漏 None 或 TypeError；
现有 parity 测试只覆盖正常样例。

- 锚点：`src/hyperloom/inference_optimizer/assets/agentx/map_aiperf.py:15-23`
- 锚点：`src/hyperloom/inference_optimizer/agentx/mapping.py:18-29`
- 建议：共享实现或使用同一病理语料做双向对拍。

---

## P3 — 清理项（24 条）

#### P3-01 · SPDX 文件头分布不一致

REUSE aggregate 注解使 CI 合规，但大量新旧 Python 文件没有仓库约定的两行 SPDX，
原审计中的数量也已漂移。建议用脚本统一，不再手工维护计数。

#### P3-02 · `__all__` 口径不统一

部分私有 helper 被导出，部分跨包实际 API 未导出，另有模块完全不定义 `__all__`。
建议只为稳定 facade 维护 `__all__`，friend/private import 单独记录。

#### P3-03 · 多个空 package `__init__.py` 保留无意义 future import

它通常不影响行为，但也不能称“完全无作用”。建议统一模板策略，
不要逐文件把它误列为 release blocker。

#### P3-04 · Dead constants、参数和兼容字段缺少统一清理策略

如未消费的 advisory metadata、历史 counters、无调用 helper、只写不读字段。
建议以 usage/lifecycle 分类：public compatibility 先 deprecate，纯内部 dead code 直接删除。

#### P3-05 · 超长函数/类集中在核心路径

Coordinator、Framework phase、grid runner、request handlers、CLI/preflight、
breakdown exporter 等包含数百至上千行函数。建议按“解析—执行—持久化—报告”拆分，
但不要把长度本身当 correctness blocker。

#### P3-06 · 同类 JSON/request/parser helper 重复

Framework request、Forge/GEMM input、Claude JSONL、header parser、bool/env parser
存在多套近同实现。建议合并到有明确 contract 的 shared helper。

#### P3-07 · 原子写样板重复且 durability 语义不同

多个模块各写 tmp + replace，但 fsync、mode、parent-dir、异常清理不同。
建议统一使用 `common.io` 并明确“原子可见”与“断电耐久”的区别。

#### P3-08 · Import-time env/default 捕获影响测试和长生命周期进程

phase budget、CLI defaults、测试 conftest、provider model 等在 import/parser 构建时读取 env。
建议配置在启动解析后冻结，并允许显式 reload。

#### P3-09 · Logger、变量遮蔽与局部重复 import 风格不一致

root logging/module logger 混用，`site/log/now_iso/succeeded` 等名称跨语义复用。
建议机械 lint/rename，不单独占 release blocker。

#### P3-10 · CLI/docstring/命令名仍含旧树和旧 console-script 名称

包括 `quantization_agent`、sibling agent repo、过时 subcommand/flag/help 文案。
建议从 parser/entrypoint 元数据生成文档片段。

#### P3-11 · 默认模型 slug 和内部部署名字出现在公开 help

proposal scorer、quantization help 等暴露内部默认模型，但实际运行又未必使用这些默认。
建议公开 help 只描述 provider/config resolution。

#### P3-12 · `datetime.utcnow()` 与时间格式散落

个别模块仍使用 deprecated UTC API、无 `Z` 的 gmtime 文本或不同精度 ISO。
建议统一 `common.timeutil`。

#### P3-13 · 测试树存在多对逐字重复文件

已确认多组 coverage/padding、renderer、GBrain、stall、local-store 文件完全重复。
删除前需迁移少量独有 tests，并以 hash/AST 脚本防止再次复制。

#### P3-14 · `asyncio_mode=auto` 下仍有大量冗余 marker

大量 async tests 显式写 `@pytest.mark.asyncio`，同目录又混用无 marker 风格。
建议机械清理裸 marker，仅保留带参数/特殊 loop 语义的标记。

#### P3-15 · 巨型测试文件掩盖职责和重复

若干测试文件超过 2k–5k LOC，混合 helper、unit、integration 和 coverage padding。
建议按生产模块/契约拆分，而不是按“coverage boost”命名。

#### P3-16 · Kernel tests 使用多种直接文件加载/sys.path 注入

包导入、`spec_from_file_location` 和永久 `sys.path.insert` 并存，
同一模块可被加载为多个 identity。建议提供 tests helper，并保留需要的隔离语义。

#### P3-17 · 多个测试用宽断言或动态 skip 掩盖回归

例如接受多个合法 status、对任意 apply 失败直接 skip、只断言字符串/文件存在。
建议每个 deterministic fixture 使用单一预期状态。

#### P3-18 · Checkout-only 架构/packaging 测试不随 wheel 分发

这些门在 CI checkout 有效，但纯 wheel 用户不会运行。建议增加真正构建 wheel 后的 contents/import smoke，
不要把 source-tree 存在等同安装产物存在。

#### P3-19 · 工作流/args 元数据接口与 provenance 不完整

`.audit-workflow.js` 消费 `ids`，现有 args 使用 `units`；manifest 又缺源码 revision/hash/receipt。
建议定义 JSON schema 和不可变 run manifest。

#### P3-20 · Action metadata 的 operational/advisory 文档仍不准确

`pipeline_phase` 对 workspace ownership 是 operational，但对调度排序并非 operational；
其它字段部分只进入 prompt。建议逐字段标注真实 consumer。

#### P3-21 · 报告字段命名把近似量写成精确量

如 launch count 写成 kernel_count、marginal percentage points 写成 total gain、
经验 sustained ceiling 写成理论上界。建议字段名携带单位、基准与估算方法。

#### P3-22 · 可选依赖/实验脚本的安装契约不清晰

matplotlib、numpy、packaging、tomli 等由不同安装路径间接获得。
建议建立明确 extras，并在手动工具启动时 fail-fast 显示安装命令。

#### P3-23 · `e2e` 命名与 CI marker 语义混用

有些测试是组件集成或生产概念的 E2E，并不应被三个真实-runtime marker 排除。
建议用 `integration`/`contract` 命名，marker 只表示真实外部运行时。

#### P3-24 · `code-clean-ref` 自身缺少可复现审计 provenance

原 419 findings、agent receipts、AST diff、merge ID 映射和源码 revision 没有完整落盘，
导致本次只能重建而不能逐字恢复。建议下一轮把这些作为强制产物。

---

## Phase 1 验收要求

每条修复至少满足：

1. 增加一个能在修复前失败、修复后通过的定向测试。
2. 结果/产物写入必须绑定 `session_id + task_id + run_id + start timestamp`。
3. 所有 destructive rollback/kill/revert 路径必须检查并传播失败。
4. 所有 secret、tool input、command、env、manifest 的外送/落盘经过同一脱敏边界。
5. 路径边界统一使用 resolved `Path.relative_to()`，禁止字符串前缀判断。
6. 任一顶层 `status="succeeded"` 必须由本轮、同身份、可验证的成功证据推出。

## 未恢复内容

原文件中的 223 条 P2、71 条 P3、原始 category 表、按包表和 “419 findings” merge 记录
没有可验证的原始 JSON/ID 映射，无法诚实地逐字恢复。本文依据二次审核把仍成立的内容
合并为 50 个 P2 根因和 24 个 P3 清理项；它们不是原条目的逐字还原。
下一轮自动生成时，应为每条 finding 保存稳定 ID、源码验证状态和 merge provenance。
