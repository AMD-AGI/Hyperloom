# Hyperloom ENABLEMENT 自愈回路的 harness 缺陷根因报告

**样本**：DeepSeek-V4.1-Flash 会话 `abe6d402-db29-4dc5-b569-7b8aeeb223fd`（Spur job 159798）
**日期**：2026-09-22
**代码基线**：Hyperloom main `79492aa95`（会话内 `hyperloom-src` 快照）
**InferenceX**：`ix0914` @ `b3c2f1efa`

---

## 0. 结论摘要

这个会话消耗了 **23.0 小时 × 4×MI355X**，产出为零：`baseline_tput=0.0`、`cumulative_gain_validated=0.0`、`current_best={}`、`runs/explore=runs/conc_sweep=runs/roofline=runs/kernel_opt=0`。但它对外报告**成功**：Slurm `JobState=COMPLETED ExitCode=0:0`，日志尾部 `ROBUST_TASK_EXIT_CODE=0`。

导致这个结果的**不是**模型问题、也不是最初那个 GPU 掩码问题。真正的问题是：ENABLEMENT 的自愈回路在诊断上**成功了**，在交付上**完全失效**。

40 轮 specialist 精确定位了根因、写出了补丁、逐条自验（包括 ctypes 直接探测 HIP、字节级比对补丁结果），但**一个补丁都没有落地**：`runs/integrate_patch=0`、`runs/integrate=0`、`patches/` 目录为空。日志中 203 行提到 `integrate_patch`，其中 203 行是 `denied`。

本报告列出 9 个可独立复现、可独立修复的 harness 缺陷。它们共同的后果是：**一个能自己诊断出根因的系统，无法把修复送到被修复的对象上**，于是把整个预算烧在同一个 gap 上打转——单个 `baseline:server_init_dead` gap 占了 40 轮中的 15 轮。

---

## 1. 事实基线

### 1.1 作业与会话

| 项 | 值 |
|---|---|
| Spur JobId | 159798 |
| UserId / QOS | `zgong` / `amd-hyperloom-qos` |
| 节点 | `crsuse2-m2m-119`（4 卡，`ROCR_VISIBLE_DEVICES=4,5,6,7`） |
| 起止 | 2026-09-21T02:57:35 → 2026-09-22T02:03:30 |
| Slurm 结果 | `JobState=COMPLETED` `ExitCode=0:0` |
| session_id | `DeepSeek-V4.1-Flash_20260921T030227Z_d6908b5c` |
| `stop_reason` | `time_exhausted` |
| tick | 344 |
| 预算 / 实耗 | `max_minutes=1380` / `elapsed_charged_sec=82818.7`（23.0 h） |
| `benchmark_mode` | `agentx` |
| `tp` / `ep` / `conc` | 4 / 1 / 32 |
| `crash_count` | 0 |
| `enablement` 完整性 | 37 字段，未被截断 |
| 会话根目录 | `/shared_nfs/hyperloom-slurm/safe-opt-spur-hlqos-3models-e2e-deepseek-ai-DeepSeek-V4-1-Flash/abe6d402-db29-4dc5-b569-7b8aeeb223fd/DeepSeek-V4.1-Flash/20260921T030227Z-71226912/` |
| 作业日志 | `/shared_nfs/hyperloom/safe-opt-spur-hlqos-3models-e2e-deepseek-ai-DeepSeek-V4-1-Flash-abe6d402-db29-4dc5-b569-7b8aeeb223fd-159798.out`（421,164 字节） |

完整产物清单见[附录 B](#附录-b三个-case-的产物位置)。

### 1.2 阶段轨迹

```
2026-09-21T03:02:29   (fresh)      -> PRELUDE       phase_entered
2026-09-21T03:07:05   PRELUDE      -> ENABLEMENT    enablement_entered
2026-09-22T02:02:35   ENABLEMENT   -> CLOSE         time_exhausted
```

**22 小时 55 分钟全部在 ENABLEMENT 内。**

### 1.3 产物统计

| 目录 | 数量 |
|---|---|
| `runs/baseline` | 1 |
| `runs/specialist` | 41 |
| `runs/integrate_patch` | **0** |
| `runs/integrate` | **0** |
| `patches/` | **0** |
| `runs/explore` | 0 |
| `runs/conc_sweep` | 0 |
| `runs/roofline` | 0 |
| `runs/kernel_opt` | 0 |
| `specialist_patch_verdicts` | 22 条，**全部 `approve`** |

### 1.4 触发一切的那次 baseline 失败

`last_baseline`：`decision=no_promote`、`status=failed`、`error_class=server_init_dead`、`ts=2026-09-21T03:07:05`。

`runs/baseline/a76221786804426a91d5d463e684377b/benchmark_vllm_20260921_030504/server.log` 中四个 worker 同时失败：

```
(Worker pid=19745) ERROR 09-21 03:06:54 [multiproc_executor.py:944]
  File "/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py", line 501, in _lazy_init
(Worker pid=19745) ERROR 09-21 03:06:54 [multiproc_executor.py:944]
  RuntimeError: No CUDA GPUs are available
```

父进程（EngineCore）存活，worker 全灭。specialist 在第 33 轮用 20 行 ctypes 探针（`libamdhip64.so` / `hipGetDeviceCount`）复现并定性：

| 环境 | rc | count |
|---|---|---|
| `ROCR=4,5,6,7` 单独 | 0 | 4 |
| `ROCR=4,5,6,7` + `HIP_VISIBLE_DEVICES=4,5,6,7` | 100 (`hipErrorNoDevice`) | **0** |
| `ROCR=4,5,6,7` + `CUDA_VISIBLE_DEVICES=4,5,6,7` | 100 | **0** |
| `ROCR=4,5,6,7` + `HIP_VISIBLE_DEVICES=0,1,2,3` | 0 | 4 |

即 **ROCr 双掩码**：ROCr 把物理 4-7 重编号为逻辑 0-3 后，HIP 层再按 4-7 去找，落在可见范围之外。父进程之所以不受影响，是因为 vLLM 用 amdsmi/sysfs 枚举，二者都不看这两个变量。

制造这个双掩码的是 recipe 自己（已核验，`dsv41flash_fp4_mi355x_vllm_mtp.sh:48-50`）：

```bash
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
```

同族 recipe（`vllm_mi300x.sh` / `atom_mi355x.sh` / `xdit_bench_common.sh`）用的是重编号后的形式（`HIP/CUDA = 0..n-1`），只有这一个抄错了。

> **注**：这个"一行之差"本身是 InferenceX 侧的 bug，修起来是一行。本报告的重点不是它，而是**为什么一个能在 12 分钟内定位到这一行的系统，花了 23 小时也没能改掉它**。

---

## 2. 缺陷清单

### H1 —— `integrate_patch` 的可打补丁根目录不包含 AgentX 实际执行的 recipe 树

**严重程度**：阻塞级。这是所有交付失败的单点根因。

specialist 在第 11 轮（`gap.harness.integrate_patch_root_allowlist`）直接在会话解释器里执行了实际解析器，给出实测结果：

- `hyperloom.orchestrator.framework.paths.resolve_kernel_search_roots()` 返回**恰好**三项：
  ```
  /usr/local/lib/python3.12/dist-packages/aiter
  /usr/local/lib/python3.12/dist-packages/aiter_meta
  /usr/local/lib/python3.12/dist-packages/vllm
  ```
- `resolve_session_framework_root()` 返回 `''`（`FRAMEWORK_REPO_PATH` / `<FW>_REPO_PATH` / `<FW>_DIR` 均未设置）
- `integrate_patch._resolve_framework_root` 的 `candidate_roots` 正是上面这个列表，因此 `manifest.json` 里 `dependencies.inferencex.path` 记录的 `/shared_nfs/hyperloom/chenluo/ix0914/InferenceX` **从来不是候选**
- 把一个头部为 `benchmarks/single_node/agentic/dsv41flash_fp4_mi355x_vllm_mtp.sh` 的补丁喂给 `patch_safety.resolve_patch_apply_root`，得到 `root=None, reason='no_matching_root'`
- `runner.py:1420` 将其转为 `patch_safety_ungrounded`，**在到达 integrate_patch 之前就把补丁从 `patches_written` 里剔除**

**源码核验**（Hyperloom `79492aa95`）：

```
orchestrator/framework/paths.py:54    FRAMEWORK_SOURCE_PACKAGES: tuple[str, ...] = (...)
orchestrator/framework/paths.py:165   def _merge_roots(*groups)
orchestrator/framework/paths.py:388   def _env_source_roots()
orchestrator/framework/paths.py:399     os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", "")
orchestrator/framework/paths.py:412   def resolve_session_framework_root()
orchestrator/framework/paths.py:530   def resolve_kernel_search_roots()
orchestrator/specialists/patch_safety.py:133   "no_matching_root",  # no candidate holds every pre-image
orchestrator/specialists/patch_safety.py:317   def resolve_patch_apply_root(...)
orchestrator/specialists/patch_safety.py:415     return PatchRootResolution(None, "no_matching_root")
orchestrator/specialists/patch_safety.py:768     "patch_safety_ungrounded:" + ...
```

允许打补丁的根目录是从**框架 Python 包树**（`FRAMEWORK_SOURCE_PACKAGES`）推导的。但在 AgentX 模式下，真正决定服务如何启动的是 InferenceX 的 recipe 脚本，它不在任何框架包树里。

**后果**：AgentX 会话中，任何针对 recipe 的 enablement 补丁在物理上不可能落地。而 recipe 恰恰是 AgentX 下最需要修的对象——本例的根因就在 recipe 第 49 行。

**已验证的零代码改动缓解**（specialist 第 11 轮实测）：在**编排进程**上设置

```
INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS=/shared_nfs/hyperloom/chenluo/ix0914/InferenceX
```

重跑解析器后 roots 变为 `[aiter, aiter_meta, vllm, InferenceX]`，同一个 recipe 补丁解析出 `root=/shared_nfs/.../InferenceX, reason=''`。注意三点：该变量是**冒号分隔列表**（要追加不要覆盖）；条目必须是**绝对路径**否则被丢弃并告警；它在解析时刻通过 `os.environ` 读取，因此必须在编排进程上，而不是服务进程上。等价替代：`--framework-path <InferenceX>`（会设置 `FRAMEWORK_REPO_PATH`），或在任务里声明 `task.params.framework_source_root`。

**负面对照**（缩小了 gap 范围）：以 `vllm/...` 为头部的补丁不受影响，能正常解析。

**建议**：AgentX 模式下，把 `manifest.json` 里 `dependencies.inferencex.path` 自动并入可打补丁根目录。当前需要操作者手工设置环境变量才能让自愈回路具备最基本的可达性，这个前置条件在任何文档和 preflight 里都没有提示。

---

### H2 —— PolicyGate 的 critic-verdict 门禁与 specialist 的 verdict 记录对不上

**严重程度**：阻塞级。

会话日志中 PolicyGate 否决统计：

| rule | 次数 |
|---|---|
| `integrate_patch_requires_critic_verdict` | **91** |
| `specialist_freeform_empty_description` | 17 |
| `duplicate_idempotency_key_running` | 9 |
| `request_target` | 4 |
| `execution_order` | 1 |

否决原文有两种形态：

```
rule=integrate_patch_requires_critic_verdict reason=integrate_patch.params names no Critic review subject
rule=integrate_patch_requires_critic_verdict reason=integrate_patch: no Critic verdict on record for subject='073a62a022ba48b5991f4d60b742cf34'
```

同时 `state.json` 里 `specialist_patch_verdicts` 有 **22 条记录，全部是 `approve`**。也就是说 Critic 确实评审了、确实通过了，但门禁按 `subject` 去查时查不到。

**源码位置**：`orchestrator/policy/gate.py:1093,1109,1129,1144` 四处抛出该 rule，判定函数是 `_validate_integrate_patch_critic_gate`。`orchestrator/loop/dispatcher.py:469,497` 说明该否决是**会重试的**——于是 91 次。

**后果**：`subject` 标识在"Critic 记录 verdict"和"integrate_patch 声明评审对象"两侧不是同一套键，导致一个已获批准的补丁被反复否决 91 次。重试机制把一个标识不匹配的 bug 放大成了对预算的持续消耗。

**建议**：让两侧共用同一个 subject 键的构造函数；并且在同一 subject 连续被否决 N 次后停止重试、把它升级为一条显式的 harness 故障，而不是静默重试到预算耗尽。

---

### H3 —— 幂等键在前一个任务永不完成时形成永久自锁

**严重程度**：高。

9 次 `duplicate_idempotency_key_running` 否决，键名可读性很好，恰好暴露了回路在反复尝试什么：

```
duplicate idempotency_key='enablement-bootfix-revalidate-1'
duplicate idempotency_key='enablement-bootprobe-inplace-r1'
duplicate idempotency_key='enablement-launch-probe-1'
duplicate idempotency_key='enablement-launch-probe-rocr-demask-live-1'
duplicate idempotency_key='enablement-launch-probe-unmodified-1'
```

这些键的语义是"同一个幂等键已有一个任务在运行中"。但那个"运行中"的任务因为 H1/H2 永远不会完成，于是这个键被永久占用，后续所有同名尝试全部被拒。

**关联证据**：`ExecutionCleanupUnconfirmed` 在 03:34:30 出现——
```
dispatcher: task=f3090595e38c4f2ca33417d34040855b GPU cleanup unconfirmed; retaining capacity
dispatcher: f3090595e38c4f2ca33417d34040855b (specialist) raised:
  ExecutionCleanupUnconfirmed('task=f3090595e38c4f2ca33417d34040855b: physical cleanup unconfirmed')
```
以及会话结束时的 `dispatcher: shutdown cleanup unconfirmed (0 pending, 1 interrupted); retaining database and ownership`。有任务在整个会话期间一直挂在 "interrupted" 状态。

**建议**：幂等键占用需要租约与超时。一个既不完成也不释放的任务不应该永久封锁它的键。

---

### H4 —— recipe 的 `extra_args` 注入通道是死的，提案静默退化为 baseline 重跑

**严重程度**：高（且**静默**，这是最危险的部分）。

specialist 第 5 轮（`harness_arg_injection_and_offload_param_names`）的原话：

> **HARNESS ARG INJECTION IS DEAD.**

已核验的事实：

- `dsv41flash_fp4_mi355x_vllm_mtp.sh:103` 起构造 `VLLM_CMD=( vllm serve ... )`，是一个**完全字面量**的 bash 数组
- `:126` `printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"`
- `:138` `"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &`
- 全文 `grep -nE '\$@|EXTRA_ARGS|GPU_MEMORY_UTILIZATION'` **零命中**——没有 `"$@"`、没有 `${EXTRA_ARGS[@]}`、没有配置文件入口
- 调用方 `aiperf_client.sh:108,116` 以 `bash "${BENCH_DIR}/${BUILTIN}"` 调用，**零位置参数**
- 用 baseline 落盘的 `vllm_command.txt` 反查：argv 就是那 18 个字面 flag，别无其他

**后果**：本会话中每一个依赖 `extra_args` 的提案都是静默 no-op，benchmark 出来的是一次**精确的 baseline 重跑**。如果 baseline 本身是 `server_init_dead`，这些提案连"跑出一个数"都做不到；如果 baseline 能跑，那更糟——它们会产出看起来合法、实则与提案无关的测量值。

**建议**：harness 在把带 `extra_args` 的变体交给一个不具备参数入口的 recipe 之前，应当先探测该入口是否存在，探测不到就以显式错误拒绝，而不是跑一次无意义的测量。这类"提案被静默丢弃"的失败模式，在数据上与"提案无效果"不可区分。

---

### H5 —— recipe 无保护地覆写 9 个环境变量，对应的 `extra_envs` 被静默丢弃

**严重程度**：中高（同样是静默）。

已核验，`dsv41flash_fp4_mi355x_vllm_mtp.sh` 在第 51,52,59,64,65,67,72,73,74 行**不带 `${X:-}` 保护**地重新导出：

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export AITER_TRITON_LOG_LEVEL=ERROR
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export OMP_NUM_THREADS=1
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1
```

外加第 48-50 行把 `ROCR_VISIBLE_DEVICES` 复制进 `HIP_VISIBLE_DEVICES`（即 H1/1.4 的双掩码源头）。

对这 10 个名字设置 `extra_envs` 会被**无声吞掉**。其余名字（`VLLM_WORKER_MULTIPROC_METHOD`、`VLLM_ROCM_USE_AITER_FP8*`、`AITER_*`、`NCCL_*`、`VLLM_WEIGHT_OFFLOADING_*` 等）确实能生效，因为 recipe 是作为子 bash 调用、继承环境。

所以 `extra_envs` **是**一条可用通道，只是带着一份没有任何人声明过的黑名单。

**建议**：把这份"被 recipe 覆写的名字"清单变成可发现的事实——由 harness 在启动前从 recipe 静态提取，并在变体与之冲突时拒绝或至少告警。

---

### H6 —— `env_grant_requests` 是有文档、无实现的死契约

**严重程度**：中（浪费 specialist 轮次，且误导）。

`orchestrator/enablement/mandate.py:314-331` 的 `ENABLEMENT_ENV_GRANT_GUIDANCE` 明确告诉 specialist：

> `PYTHONPATH` 和 `LD_LIBRARY_PATH` 是 blocked 但 **GRANTABLE** 的；
> "To ask for one, add an `env_grant_requests` array to your final `specialist_done`, each entry `{"name": ..., "value": ..., "reason": ...}`"；
> 该值会被 **PREPENDED**（前置）而非替换。

这段文本在第 475-476 行被注入到 specialist 的 prompt 里。

**实测**：全树搜索 `env_grant`，**只有一处命中**，就是 `mandate.py:321` 这段散文本身（另两处命中是它的 `.pyc`）：

```
orchestrator/enablement/mandate.py:321:    "To ask for one, add an `env_grant_requests` array to your final "
```

没有解析器、没有 schema 字段、没有消费者。`integrate_patch.py:1729` 只有一句孤立注释（"Blocked env names this round was granted, each bound to one value."），底下没有任何属性。

**后果**：一个遵照 mandate 行事的 specialist 会被静默忽略。它会以为自己申请了 grant，实际什么也没发生，然后在下一轮基于错误前提继续推理。

**补充事实**（同一轮核验）：真实的合并语义是**后者整体覆盖**（last-wins overwrite），从不 append/prepend——`common/env_safety.py:383-389` 的 `build_benchmark_env` 先复制 `os.environ` 再逐层 `env.update()`，`_grid_runner.py:474-475` 逐条 `envs[k]=v`。唯一会前置的名字是 `PATH` / `PYTHONPATH` / `LD_LIBRARY_PATH`，且只通过 harness 自有的保留字段（`variant.overlay_pythonpath`、`apply_runtime_override` 的 `path_prefix` / `pythonpath_prefix(es)` / `ld_library_path_prefix`，见 `_grid_runner.py:339-392,481-497`），这些字段在 `_RUNTIME_ENV_RESERVED` 中，specialist 的 payload 触及不到。

也就是说，mandate 描述的 "prepended" 语义与实现相反。

**建议**：要么实现 `env_grant_requests` 的消费端，要么把这段 guidance 从 mandate 里删掉。当前状态是最坏的——它在主动误导 agent。

---

### H7 —— 阻塞条件解除后不会重跑 baseline

**严重程度**：高。

第 32 轮（`enablement_boot_never_exercised`，11:35:00Z）给出了 **LIVE PROOF**：在与 sealed baseline **完全相同**的环境下（`ROCR_VISIBLE_DEVICES=HIP_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES=4,5,6,7`），在一个全新解释器里 `import vllm.platforms.rocm` 会发出两条 `rocm.py:174` remap 告警，把 `HIP_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES` 改写为 `'0,1,2,3'`，随后 `torch.cuda.device_count()` 返回 **4**（sealed baseline 时是 0）。

第 35 轮（`boot-e2e-serve-unverified`，14:01:55Z）进一步确认：`vllm/platforms/rocm.py:127` 的 `_resolve_rocr_double_mask()` 由 `_sync_hip_cuda_env_vars()`（:193）在模块导入时（:210）无条件调用；直接探测三种掩码组合，`device_count()` 全部返回 4，而非 0。fork 安全性假设也被排除（父进程调 `device_count()` 并 `import aiter` 后 fork，子进程仍得 4）。

**换句话说：到 11:35Z，最初那个阻塞 baseline 的失败模式在当前安装树上已经不复存在了。**（第 33 轮 11:41Z 的裸 HIP ctypes 探针仍显示双掩码致命，与此不矛盾——那条路径不经过 vLLM 的 import-time remap。两个结论在各自的层次上都正确。）

会话此时还有 **14.5 小时**预算。但它从未重跑 baseline，一直留在 ENABLEMENT 直到 `time_exhausted`。

**建议**：ENABLEMENT 需要一个"重新验证阻塞前提"的定期动作。一个 baseline 因环境条件失败后，该条件可能在会话生命周期内自行消失（本例就是：邻居作业退出、或安装树被更新）。当前设计把首次 baseline 的失败当作在整个会话内恒为真的事实。

---

### H8 —— 基于瞬时环境事实推导出的硬约束不会失效

**严重程度**：高（污染了约一半的轮次）。

时间线：

- **03:18Z** research scout 实测节点显存：8 张卡各用掉 231–234 GB / 309.22 GB（这是当时同节点上 4 个野容器占的，见附录 A）。由此推导出硬上限 `gpu_memory_utilization <= 0.2444`，并得出结论"**weight offload 是必须的**"。
- 随后约 20 轮围绕这个上限展开：`--cpu-offload-gb`、`--offload-backend prefetch`、`UVAOffloader.post_init` 补丁、`auto_offload_sizer_zero_kv_reserve`、`gap.enablement.auto_offload_sizing`、`gpu_memory_cotenancy_headroom`（4 轮）……
- **08:18Z** 野容器被清理（外部动作），8 张卡显存回落到 0。
- **11:32Z / 11:40Z / 13:59Z** 第 32、33、35 轮各自独立实测：每张卡用掉约 0.298 GB / 309.22 GB，`free/total = 0.999`。

specialist 第 32 轮的定性很直接：

> The entire premise of ~20 prior rounds (gmu ceiling <= 0.244, 'weight offload is mandatory', KV floor pressure) is **void as of now**.

第 33 轮补充了一条重要的反向判断：现在再强制 offload 反而是**确定的性能回退**——"would put 1.72 GiB/rank/layer of H2D on the critical path of every decode step (~29 ms/layer/step at ~64 GB/s) and is a guaranteed regression on the graded E2E normalised interactivity P90"。

**后果**：约一半的预算花在优化一个已经不存在的约束上。harness 没有任何机制在"推导出约束的那次测量"过期时让该约束失效。`pruned_families` 最终只留下 `['offload']`——也就是说系统最后确实把 offload 这条路剪掉了，但那已经是几乎所有轮次都花完之后的事。

**建议**：由环境测量推导出的约束应当携带该测量的时间戳与来源，并在被重新测量推翻时自动失效。至少，研究提示（research hints）应当在被后续实测证伪时被显式撤回，而不是继续留在上下文里。

---

### H9 —— 零测量的会话报告成功

**严重程度**：高（这是让前面 8 条得以长期隐身的原因）。

已观测到的事实：

| 信号 | 值 |
|---|---|
| `baseline_tput` | 0.0 |
| `baseline_accuracy` | 0.0 |
| `cumulative_gain_validated` | 0.0 |
| `current_best` | `{}` |
| `stop_reason` | `time_exhausted` |
| Slurm `JobState` | `COMPLETED` |
| Slurm `ExitCode` | `0:0` |
| 日志尾部 | `ROBUST_TASK_EXIT_CODE=0` |

Final summary 里确实印了一行警告：

```
cumulative_gain_val  : 0.00% ⚠ never validated — no `explore` KEEP has landed yet
```

但这只是人读的文本，没有进入任何机器可判的出口状态。

有意思的是，harness 里**存在**这类门禁，只是没用在会话自身的结局上——日志里有一条：

```
rule=execution_order reason=action='report' denied: baseline must run first
```

也就是说"没有 baseline 就不许出报告"这条规则是有的，但"没有 baseline 的会话不该算成功"这条没有。

**后果**：从调度器、Robust API、任何看 exit code 的自动化视角看，这是一次成功的 23 小时优化会话。上面 8 个缺陷因此可以长期存在而不触发任何告警。

**建议**：`stop_reason=time_exhausted` 且 `baseline_tput == 0` 应当映射为非零退出码，或至少一个独立的、机器可判的 `no_measurement` 结局状态。

---

## 3. 横向对照：这不是孤例

| 会话 | 模型 | baseline 失败原因 | 结局 | 消耗 |
|---|---|---|---|---|
| `abe6d402` / 159798 | DeepSeek-V4.1-Flash | `server_init_dead`（ROCr 双掩码） | `time_exhausted`，产出 0 | 23.0 h |
| `b5689f28` / 159796 | GLM-5.2-MXFP4 | `subprocess_nonzero`（显存 gate，邻居占卡） | `emergency`（`crash_count=25`） | 3.9 h |
| `d021904a` / 160819 | GLM-5.2-MXFP4 | `subprocess_nonzero`（`KV_OFFLOAD_BACKEND_METADATA must contain valid JSON`） | 已于 17.6 h 处手动取消 | 17.6 h |

三个会话、两个模型、三种完全不同的 baseline 失败原因，但**后续行为完全一致**：进入 ENABLEMENT，跑几十轮 specialist，`runs/integrate_patch=0`，最终耗尽预算或崩溃，产出为零。

其中 `b5689f28` 还额外触发了另一条已单独记录的缺陷：`SharedState.apply_changes()`（`shared_state.py:1713`）用 `setattr` 把 agent 提供的裸 dict 直接写进类型化字段 `enablement`，顶掉了 `EnablementRound` 对象，导致每个 tick 抛 `AttributeError: 'dict' object has no attribute 'origin'`，累计 63 次、`crash_count` 到 25 后 emergency 退出。本例的 `enablement` 是完整的 37 字段，说明那条路径是可以不触发的——但它存在。

三者合计约 **44.5 GPU-小时 × 4×MI355X**，有效产出为零。

三个 case 的会话根目录与作业日志：

| case | 会话根目录 | 作业日志 |
|---|---|---|
| `abe6d402` / 159798 | `.../DeepSeek-V4-1-Flash/abe6d402-.../DeepSeek-V4.1-Flash/20260921T030227Z-71226912/` | `...-abe6d402-...-159798.out` |
| `b5689f28` / 159796 | `.../amd-GLM-5-2/b5689f28-.../GLM-5.2-MXFP4/20260921T025941Z-6e133f59/` | `...-b5689f28-...-159796.out` |
| `d021904a` / 160819 | `.../amd-GLM-5-2/d021904a-.../GLM-5.2-MXFP4/20260921T083515Z-9e133991/` | `...-d021904a-...-160819.out` |

会话根目录的公共前缀是 `/shared_nfs/hyperloom-slurm/safe-opt-spur-hlqos-3models-e2e-`，作业日志的公共前缀是 `/shared_nfs/hyperloom/safe-opt-spur-hlqos-3models-e2e-`。逐文件清单见[附录 B](#附录-b三个-case-的产物位置)。

---

## 4. 优先级建议

| 优先级 | 缺陷 | 一句话修法 | 预期收益 |
|---|---|---|---|
| P0 | **H1** | AgentX 模式下把 `manifest.json` 的 `dependencies.inferencex.path` 并入可打补丁根目录 | 自愈回路从"物理不可达"变为"可达"。这是其他一切的前提 |
| P0 | **H9** | `time_exhausted` + `baseline_tput==0` → 非零退出码 | 让这类失败立刻可见，而不是靠人翻 state.json |
| P1 | **H2** | 两侧共用同一个 subject 键构造；连续否决 N 次后升级为显式故障 | 消掉 91 次无意义重试 |
| P1 | **H7** | ENABLEMENT 定期重验阻塞前提，条件消失即重跑 baseline | 本例可在 11:35Z 恢复，挽回 14.5 h |
| P1 | **H4** | 交付前探测 recipe 的参数入口，不存在就显式失败 | 消掉"静默 no-op 提案"这一整类不可观测失败 |
| P2 | **H3** | 幂等键加租约超时 | 打破永久自锁 |
| P2 | **H8** | 环境推导约束携带测量时间戳，被推翻即失效 | 本例可挽回约 20 轮 |
| P2 | **H6** | 实现 `env_grant_requests` 消费端，或删掉那段 guidance | 停止误导 agent |
| P3 | **H5** | 从 recipe 静态提取"被覆写的环境变量名"清单并暴露给变体校验 | 让静默丢弃变为显式冲突 |

另有两条不属于 harness、但阻塞了本次样本的上游/配置问题，应单独提：

- **InferenceX**：`dsv41flash_fp4_mi355x_vllm_mtp.sh:48-50` 应改用重编号后的形式（`HIP/CUDA = 0..n-1`），与 `vllm_mi300x.sh` / `atom_mi355x.sh` / `xdit_bench_common.sh` 一致。一行修复，解掉 1.4 节的根因。
- **HLD 下发链路**：`160819` 的 `KV_OFFLOAD_BACKEND_METADATA` 未给出合法 JSON；`ep` 固定为 1，而 GLM 应为 4。

---

## 附录 A：外部干扰因素（已排除，但影响了本会话的前半段）

本会话运行在 `crsuse2-m2m-119`。该节点在 2026-09-20T14:35:27 起被 4 个**不属于任何 Slurm 作业**的容器占据，镜像 `lmsysorg/sglang:v0.5.20-rocm724-mi35x`，label `inference_testing.worker_pid=1040~1043`，各以 `--tp 2` 跑 `amd/MiniMax-M3-MXFP4`，两卡一组共占满 8 张卡，每卡约 222 GB / 294.9 GB。Slurm 对此无感知，仍把该节点的 8 张卡标记为可调度。

这些容器在 2026-09-21T08:18Z 被手动清理，取证存档于 `/shared_nfs/hyperloom/chenluo/evidence-orphans-m2m-119-20260921.txt`。

需要明确的是：**它们不是本报告 9 条缺陷的原因**。它们只解释了 H8 里那个 231–234 GB 的初始测量，以及为什么 `b5689f28`（GLM，钉 0-3 卡）的显存 gate 会拒绝启动。本会话 baseline 的直接死因是 ROCr 双掩码，与占卡无关——第 33、35 轮在卡完全空闲的条件下仍然复现了双掩码的致命性。

---

## 附录 B：三个 case 的产物位置

所有产物均落在共享盘，容器销毁后仍可访问。以下大小为 2026-09-22 02:5x 实测值。

三处公共前缀，下文用 `$HLS` / `$HLL` 代指：

```
$HLS = /shared_nfs/hyperloom-slurm/safe-opt-spur-hlqos-3models-e2e-      # 会话根
$HLL = /shared_nfs/hyperloom/safe-opt-spur-hlqos-3models-e2e-            # 作业日志
```

### B.1 case A —— DeepSeek-V4.1-Flash，job 159798（本报告主样本）

```
会话根目录
  $HLS-deepseek-ai-DeepSeek-V4-1-Flash/abe6d402-db29-4dc5-b569-7b8aeeb223fd/
    DeepSeek-V4.1-Flash/20260921T030227Z-71226912/

顶层
        612,168  state.json                  # 全部状态字段、40 轮 specialist_rounds、enablement 37 字段
        972,265  session_breakdown.json
        184,206  research_hints.json         # H8 的 gmu<=0.2444 陈旧约束就在这里
        167,554  research_hints.md
          2,184  manifest.json               # dependencies.inferencex.path —— H1 里"从来不是候选"的那个路径

reports/
         98,160  final.json                  # highlights[].payload.result.specialist_done 提案全文
         40,177  final.md
         10,542  optimization_journal.json
          1,727  kernel_optimization_summary.json

reports/trace/
    120,662,193  conversations.jsonl         # 115 MB，40 轮 specialist 的完整对话
        627,574  llm_calls.jsonl
        469,570  specialist_intel.jsonl      # specialist 的每次工具调用（ctypes 探针、rocm-smi 实测都在这）
         22,557  decision_trace.jsonl
          3,480  proposal_task_map.jsonl
          1,263  langfuse_receipt.json

runs/  （baseline=1, specialist=41, 其余全 0）
  runs/baseline/a76221786804426a91d5d463e684377b/
    benchmark_vllm_20260921_030504/
         32,098  server.log                  # No CUDA GPUs are available ×4 worker（1.4 节）
            751  vllm_command.txt            # 证明 argv 只有 18 个字面 flag（H4）
          2,836  summary.txt
          5,347  benchmark_stderr.log
          3,256  benchmark_report.json
  runs/specialist/                           # 41 个目录，每个含 prompt.md / system_prompt.md /
                                             #   specialist_done.json / process.log / worktree/
  runs/integrate  runs/kernel_opt  runs/params  runs/profile  runs/backends   # 全部为空 —— H1 的直接证据
  patches/                                   # 0 个文件 —— 补丁从未落地

作业日志
        421,164  $HLL-deepseek-ai-DeepSeek-V4-1-Flash-abe6d402-db29-4dc5-b569-7b8aeeb223fd-159798.out
                                             # PolicyGate 否决全在这（91/17/9/4/1）
```

### B.2 case B —— GLM-5.2-MXFP4，job 159796（`emergency`，`crash_count=25`）

```
会话根目录
  $HLS-amd-GLM-5-2/b5689f28-7284-40f4-a3f7-ba1e60e38c9b/
    GLM-5.2-MXFP4/20260921T025941Z-6e133f59/

顶层
        309,773  state.json                  # enablement 被截断成 1 字段（只剩 accepted_config）
        302,985  session_breakdown.json
         82,275  research_hints.json
          2,156  manifest.json

reports/
        182,235  final.json                  # highlights[29]/[36] 是那两个 TP=8 + 8 卡 UUID 的提案
         40,464  final.md

reports/trace/
      9,591,050  conversations.jsonl
        198,572  llm_calls.jsonl
        169,859  specialist_intel.jsonl      # 第 222/223/253 行是 GPU UUID 探测
         10,021  decision_trace.jsonl

runs/  （baseline=1, specialist=18, 其余全 0）
  runs/baseline/8a00598562c048c9ab694b8a1710088e/
    benchmark_sglang_20260921_030133/
         39,506  benchmark_stderr.log        # 90 次 vram%max=76 轮询 + exit 1
          2,375  summary.txt
          2,711  benchmark_report.json
                                             # 注意：无 server.log —— 服务从未启动，死在 recipe 的显存 gate
  patches/                                   # 0 个文件

作业日志
        176,145  $HLL-amd-GLM-5-2-b5689f28-7284-40f4-a3f7-ba1e60e38c9b-159796.out
                                             # 第 842/851/864 行起是 63 次 'dict' object has no attribute
```

### B.3 case C —— GLM-5.2-MXFP4，job 160819（17.6 h 处手动取消）

```
会话根目录
  $HLS-amd-GLM-5-2/d021904a-c64d-46b4-be55-396811a3b467/
    GLM-5.2-MXFP4/20260921T083515Z-9e133991/

顶层
        536,777  state.json
        927,163  session_breakdown.json
        176,660  research_hints.json
          2,156  manifest.json

reports/
        218,519  final.json
         63,649  final.md
         10,076  optimization_journal.json

reports/trace/
     80,322,118  conversations.jsonl         # 77 MB，39 轮
        549,214  llm_calls.jsonl
        385,116  specialist_intel.jsonl
         21,491  decision_trace.jsonl

runs/  （baseline=1, specialist=39, 其余全 0）
  runs/baseline/fe0d7b2d2f9841aa9230abd7613c35bc/
    benchmark_sglang_20260921_083740/
      7,224,159  server.log                  # 7 MB —— 这个 case 服务起来了
         28,831  benchmark_stderr.log        # 末尾 KV_OFFLOAD_BACKEND_METADATA must contain valid JSON
          2,382  summary.txt
  patches/                                   # 0 个文件

作业日志
        345,547  $HLL-amd-GLM-5-2-d021904a-c64d-46b4-be55-396811a3b467-160819.out
```

### B.4 关于 session package zip

作业日志尾部会打印一行 `Artifact package : /workspace/hyperloom-session-packages/<session_id>.zip`，例如：

```
case A: /workspace/hyperloom-session-packages/DeepSeek-V4.1-Flash_20260921T030227Z_d6908b5c.zip
case B: /workspace/hyperloom-session-packages/GLM-5.2-MXFP4_20260921T025941Z_c4886903.zip
case C: （无 —— 会话被 scancel 中断，CLOSE 阶段未执行到打包步骤）
```

**这是容器内路径**，`/workspace` 未映射到共享盘，容器销毁后 zip 即不存在，宿主机上找不到。要归档请直接压缩上面的会话根目录。

### B.5 旁证：手动链路的 GLM，job 160817

不属于本报告的三个 case，但它是同一时间窗内唯一 baseline 成功的对照，且暴露了 roofline 的下一层缺陷（`capture_status_missing`）：

```
会话根目录  /shared_nfs/hyperloom-slurm/plan34_glm52/06d993ef-4d22-4a64-9818-ba2690bc1c16/
              GLM-5.2-MXFP4/20260921T083418Z-b3d20ae1/
  runs/roofline/                             # 4 个目录，每个的 agentx-profile/<capture_id>/ 下
                                             #   只有 trace-manifest.json，没有 capture-status.json
作业日志     /shared_nfs/hyperloom/chenluo/logs/hl35-glm52-160817.out   （37,854 字节）
```

---

## 附录 C：源码与 recipe 引用

```
recipe 与调用方
  /shared_nfs/hyperloom/chenluo/ix0914/InferenceX/benchmarks/single_node/agentic/
    dsv41flash_fp4_mi355x_vllm_mtp.sh              # :48-50 双掩码, :103-138 字面 argv
  /shared_nfs/hyperloom/chenluo/ix0914/InferenceX/benchmarks/aiperf_client.sh   # :108,:116

Hyperloom 源码（main 79492aa95）
  src/hyperloom/orchestrator/framework/paths.py           # :54,:165,:388,:399,:412,:530
  src/hyperloom/orchestrator/specialists/patch_safety.py  # :133,:317,:415,:768
  src/hyperloom/orchestrator/policy/gate.py               # :1093,:1109,:1129,:1144
  src/hyperloom/orchestrator/loop/dispatcher.py           # :469,:497
  src/hyperloom/orchestrator/enablement/mandate.py        # :314-331,:475
  src/hyperloom/orchestrator/state/shared_state.py        # :1713 (apply_changes)
  src/hyperloom/common/env_safety.py                      # :222,:383-389
  src/hyperloom/orchestrator/actions/executors/_grid_runner.py  # :339-392,:474-475,:481-497
```
