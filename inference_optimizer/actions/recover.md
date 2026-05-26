# `recover` action playbook

## Purpose

Release GPU memory and process leaks left behind by a crashed inference
server so the optimizer can keep making progress on the remaining time
budget. The action is the inference_optimizer counterpart of the
robustness-agent `gpu_memory_leaked` signal (see the
`gpu-leak-robustness-fix` design notes).

Two failure modes are in scope:

1. A `sglang.launch_server` / `vllm.entrypoints` / `vllm.EngineCore` /
   `Magpie` / `benchmark_serving` PID is still alive after its parent
   process died (orphaned), pinning VRAM and competing with the next
   server start.
2. The process is gone but ROCm KFD driver tables still attribute
   allocations to a dead PID, so every visible GPU reports near-zero
   free MiB. `validate_stack` / `backends` / `params` then loops on
   `RuntimeError: Engine core initialization failed ... Free memory on
   device cuda:N (0.0/...)`.

`recover` is **not** for recovering from a workload-level KEEP/REVERT
regression — that path goes through `integrate` REVERT instead.

## Who delegates this action

* **Robustness** — primary caller. When the `gpu_memory_leaked`
  detector trips (all visible GPUs at >=99% memory + no live owner
  process for `min_consecutive_ticks` ticks), the ActionLadder emits
  `delegate{action_name="recover", params={force_gpu_cleanup: True,
  reason: "gpu_memory_leaked", evidence: {...}}}` with a tick-indexed
  `idempotency_key`. PolicyGate accepts it under
  `ROBUSTNESS_DELEGATE_ACTIONS`.
* **Orchestration** — optional. The action is listed in
  `FULL_ENABLED_ACTIONS` / `NO_KERNEL_ENABLED_ACTIONS` so the
  Orchestration prompt can propose it after operator-visible crashes
  (e.g. a `crash_count_high` alert from Robustness that did not auto-
  delegate, or a manual escalation in the inbox). Orchestration may
  also emit `delegate{action_name="recover", params={...}}` directly.

## Inputs (task.params)

| Key                  | Type     | Default | Description |
|----------------------|----------|---------|-------------|
| `reason`             | string   | `""`    | Trigger label echoed into `result.json`. Robustness sets `"gpu_memory_leaked"`; Orchestration may use `"crash_after_revert"` etc. |
| `force_gpu_cleanup`  | bool     | `False` | Walk the SIGTERM -> SIGKILL ladder against every PID matching an inference-server owner pattern. Robustness sets `True`. When `False` the executor only probes the GPUs and returns a `needs_review` diagnostic. |
| `evidence`           | object   | `{}`    | Optional per-GPU evidence carried over from the symptom (free MiB, consecutive_hits, owner patterns). Stored verbatim in `result.json`; the executor does not branch on it. |

## Tiered cleanup

```
                  +-------------------------------+
  pre-probe   --> | rocm-smi --showmeminfo vram   |
                  +-------------------------------+
                                |
                                v
  soft kill   --> SIGTERM all owners matching:
                    sglang.launch_server / sglang.srt /
                    vllm.entrypoints / vllm serve /
                    EngineCore / Magpie / benchmark_serving
                  wait SERVER_KILL_WAIT_S (5s), then SIGKILL
                  any survivors. Skipped when
                  force_gpu_cleanup=False.
                                |
                                v
  mid-probe   --> rocm-smi (same shape)
                                |
                                v
  hard reset  --> only when ALL hold:
                    * force_gpu_cleanup=True
                    * HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1
                    * mid-probe still shows leaked VRAM
                  Runs `rocm-smi --gpureset --gpu=all` with a 30s
                  timeout. Captures stdout/stderr/returncode for the
                  audit log; never raises.
                                |
                                v
  post-probe  --> rocm-smi (skipped when no gpureset attempted)
```

Out of scope (by design — would kill the live optimizer or require
privileges the sandbox typically lacks):

* `pkill -f sglang` / `pkill -f vllm` (violates kernel-agent IR-5).
* Reloading the `amdgpu` kernel module or restarting the pod.
* Touching `~/.claude/config.json`, Ray, or any other long-lived
  runtime service.

## Output (`runs/recover/<task_id>/result.json`)

```yaml
state:                "succeeded" | "needs_review"
reason:               <echoed from params.reason>
force_gpu_cleanup:    <echoed bool>
allow_reset_env:      <bool — HYPERLOOM_RECOVER_ALLOW_GPU_RESET resolved>
killed_pids:
  - pid:    <int>
    cmd:    <str — cmdline at discovery time>
    pattern: <str — owner pattern that matched>
    signal: "TERM" | "KILL"
pre_free_mb_per_gpu:  [{gpu_id, vram_total_mb, vram_used_mb, free_mb}, ...]
mid_free_mb_per_gpu:  [...]
post_free_mb_per_gpu: [...]
gpureset_attempted:   <bool>
gpureset_result:      {returncode, stdout, stderr, error?, timeout_s?}
error_class:          <str — only on state=needs_review>
workspace:            <str — runs/recover/<task_id>/>
result_path:          <str — workspace/result.json>
```

`state == "succeeded"` requires every visible GPU to report
`free_mb >= 500` (matches the robustness-agent leak detector's
free_mb_threshold).

## Failure handling

When the executor returns `state == "needs_review"`, the recover task
itself is still marked `succeeded` by the SubAgentRunner (the dict
shape is structurally valid). The ladder cooldown
(`cooldown_ticks=5`) prevents Robustness from immediately re-firing
`gpu_memory_leaked` -> `delegate(recover)`. Once the cooldown elapses,
the escalate_strategy_change advisory from the same tick instructs
Orchestration to fall back to a deterministic `report` proposal so the
session can finalize at the last validated gain instead of burning
budget on doomed validate_stack retries.
