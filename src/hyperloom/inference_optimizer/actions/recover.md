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
   free MiB. `explore` then loops on
   `RuntimeError: Engine core initialization failed ... Free memory on
   device cuda:N (0.0/...)`.

`recover` is **not** for recovering from a workload-level KEEP/REVERT
regression — that path goes through `integrate` REVERT instead.

## Who delegates this action

* **Robustness** — the *only* caller. When the `gpu_memory_leaked`
  detector trips (all visible GPUs at >=99% memory + no live owner
  process for `min_consecutive_ticks` ticks), the ActionLadder emits
  `delegate{action_name="recover", params={force_gpu_cleanup: True,
  reason: "gpu_memory_leaked", evidence: {...}}}` with a tick-indexed
  `idempotency_key`. PolicyGate accepts it under
  `DELEGATE_ACTION_SOURCE_ALLOWLIST` (derived from `ROBUSTNESS_DELEGATE_ONLY_ACTIONS`
  in `policy/gate.py`), and additionally requires a non-empty `reason` +
  `evidence` on the payload via `DELEGATE_ACTION_REQUIRED_PAYLOAD` — missing
  either is denied as `rule="delegate_action_evidence"`. The robustness-agent
  envelope has its own pre-emit guard, `ROBUSTNESS_DELEGATE_ACTIONS`
  (`agents/robustness/role/envelope.py`).

  `recover` is a `ROBUSTNESS_DELEGATE_ONLY_ACTIONS` member
  (see `protocol/action_surfaces.py`): it is **not** in
  `FULL_ENABLED_ACTIONS` / `NO_KERNEL_AGENT_ENABLED_ACTIONS`, is subtracted
  from `PHASE_LLM_PROPOSABLE_ACTIONS`. PolicyGate denies any Orchestration
  `propose_action` (`rule="propose_action_source"`) or `delegate`
  (`rule="delegate_action_source"`), and denies any robustness `delegate` for
  actions outside `ROBUSTNESS_DELEGATE_ONLY_ACTIONS` (`rule="role"`).
  Orchestration that observes a crash must emit an ALERT and let the robustness
  action-ladder escalate; it can no longer self-trigger `recover`.

## Inputs (task.params)

| Key                  | Type     | Default | Description |
|----------------------|----------|---------|-------------|
| `reason`             | string   | required (non-empty) | Trigger label echoed into `result.json`; Robustness sets `"gpu_memory_leaked"`. PolicyGate rejects an empty value on the delegate path (`delegate_action_evidence`); the executor-side `""` default only applies to a hypothetical non-delegate invocation. |
| `force_gpu_cleanup`  | bool     | `False` | Walk the SIGTERM -> SIGKILL ladder against every PID matching an inference-server owner pattern. Robustness sets `True`. When `False` the executor only probes the GPUs and returns a `needs_review` diagnostic. |
| `evidence`           | object   | required (non-empty) | Per-GPU evidence carried over from the symptom (free MiB, consecutive_hits, owner patterns). Stored verbatim in `result.json`; the executor does not branch on it. An empty `{}` is denied on the delegate path, same rule as `reason`. |

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
  mid-probe   --> rocm-smi (same shape); decides success
```

Out of scope (by design — would kill the live optimizer, affect other
tenants, or require privileges the sandbox typically lacks):

* `pkill -f sglang` / `pkill -f vllm` (violates kernel-agent IR-5).
* Hard GPU reset (`rocm-smi --gpureset`) — tenant-affecting, never issued.
* Reloading the `amdgpu` kernel module or restarting the pod.
* Touching `~/.claude/config.json`, Ray, or any other long-lived
  runtime service.

## Output (`runs/recover/<task_id>/result.json`)

```yaml
state:                "succeeded" | "needs_review"
reason:               <echoed from params.reason>
force_gpu_cleanup:    <echoed bool>
killed_pids:
  - pid:    <int>
    cmd:    <str — cmdline at discovery time>
    pattern: <str — owner pattern that matched>
    signal: "TERM" | "KILL"
pre_free_mb_per_gpu:  [{gpu_id, vram_total_mb, vram_used_mb, free_mb}, ...]
mid_free_mb_per_gpu:  [...]
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
shape is structurally valid), but the robustness event detector raises a
`recover_unsuccessful` HIGH symptom (`signals/event.py`). The ActionLadder
turns that into a HIGH alert carrying the evidence; Orchestration owns the
decision to finalize at the last validated gain instead of burning budget on
further doomed recover attempts. The `gpu_memory_leaked` cooldown
(`cooldown_ticks=5`) is per-dedup_key and only suppresses re-firing of that
symptom.
