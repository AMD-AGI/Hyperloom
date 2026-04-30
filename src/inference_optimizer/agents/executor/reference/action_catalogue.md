# Action Catalogue — Static Reference

**Always-fresh source**: the conductor injects an "Available actions for
this mode" table into your prompt every turn. That table reflects live
metadata + lane availability — trust it over this static snapshot.

This file is for orientation: **which 21 actions exist, by family**.

> Authoritative metadata: `src/inference_optimizer/actions/_meta/<name>.yaml`
> Markdown bodies (the per-action prompts SubAgentRunner injects):
> `src/inference_optimizer/actions/<name>.md`

## Family overview

| Family | Purpose | Quick | Guided | Marathon |
|---|---|:-:|:-:|:-:|
| `prep` | Setup / classify / target / baseline | ✓ | ✓ | ✓ |
| `analysis` | Profile / collect evidence | ✓ | ✓ | ✓ |
| `shallow` | Backends / params / sweep / report | ✓ | ✓ | ✓ |
| `deep_kernel` | Kernel-opt / integrate / operator tuning | ✗ | partial | ✓ |
| `long` | Comm / compiler tuning | ✗ | ✗ | ✓ |
| `creative` | Dream / re-explore | ✗ | ✗ | ✓ |
| `resilience` | Recover from checkpoint | ✗ | ✗ | ✓ |

## All 21 actions (by family, alphabetical within family)

### prep

| name | quick | guided | marathon | notes |
|---|:-:|:-:|:-:|---|
| `baseline` | ✓ | ✓ | ✓ | First-light measurement; **MUST be first delegate** |
| `classify` | ✓ | ✓ | ✓ | Detect model class (dense / MoE+MLA / etc.) |
| `setup` | ✓ | ✓ | ✓ | Session bootstrap |
| `target_analysis` | ✓ | ✓ | ✓ | Ingest TARGET_DIR for cross-system comparison |

### analysis

| name | quick | guided | marathon | notes |
|---|:-:|:-:|:-:|---|
| `profile` | ✓ | ✓ | ✓ | Capture `filtered-TP-0.trace.json.gz`; soft-skip if existing trace fresh |

### shallow

| name | quick | guided | marathon | notes |
|---|:-:|:-:|:-:|---|
| `backends` | ✓ | ✓ | ✓ | Try alternate attention/GEMM backends; accuracy_risk=0.10 |
| `bench_runner` | ✓ | ✓ | ✓ | Re-bench against running server (no restart) |
| `param_sweep_run` | ✓ | ✓ | ✓ | CONC × ISL/OSL × mem-fraction grid |
| `params` | ✓ | ✓ | ✓ | Single-param test; `kv-cache-dtype fp8` ↑ accuracy_risk to 0.30 |
| `report` | ✓ | ✓ | ✓ | Final summary; **terminates** the run gracefully |
| `sweep` | ✓ | ✓ | ✓ | Final candidate sweep before report |

### deep_kernel

| name | quick | guided | marathon | notes |
|---|:-:|:-:|:-:|---|
| `deep_kernel_analysis` | ✗ | ✗ | ✓ | Per-kernel deep dive |
| `integrate` | ✗ | ✓ | ✓ | **Owned by kernel agent (Plan A)** — emit `request{target_agent="kernel", kind="apply_patch"}`, NOT `delegate(integrate)`. PolicyGate denies direct delegation. |
| `kernel_opt` | ✗ | ✓ | ✓ | **Owned by kernel agent (Plan A)** — emit `request{target_agent="kernel", kind="run_optimization"}`. PolicyGate denies `delegate(kernel_opt)`. |
| `operator_tuning` | ✗ | ✗ | ✓ | Per-op autotune |
| `vendor_kernel_config` | ✗ | ✗ | ✓ | Vendor-specific kernel config knobs |

### long

| name | quick | guided | marathon | notes |
|---|:-:|:-:|:-:|---|
| `comm_optimization` | ✗ | ✗ | ✓ | NCCL / RCCL / SHMEM tuning |
| `compiler_tuning` | ✗ | ✗ | ✓ | torch.compile / triton.autotune envelope |

### creative

| name | quick | guided | marathon | notes |
|---|:-:|:-:|:-:|---|
| `dream` | ✗ | ✗ | ✓ | Sage generates "what-if" hypotheses; no side effects |
| `re_explore` | ✗ | ✗ | ✓ | Re-score previously discarded candidates |

### resilience

| name | quick | guided | marathon | notes |
|---|:-:|:-:|:-:|---|
| `recover` | ✗ | ✗ | ✓ | Resume a crashed sub-task from checkpoint |

## Quick-mode action allowlist (when no ActionRegistry present)

If your run starts with `--no-action-registry` (rare), PolicyGate falls
back to `DEFAULT_QUICK_ACTION_ALLOWLIST`:

```
{baseline, server_lifecycle_restart, param_sweep_run,
 bench_runner, profile, diagnostic_probe, kb_query, report}
```

`kernel_opt` / `integrate` etc. are NOT on this
list and will be rejected even before `mode_gate=0` fires.

## Lane requirements (concurrency)

Most actions require one or more of these lanes (max 1 holder each):

- `server_lifecycle` — held during server kill / start / restart
- `workspace_mutation` — held during `Edit` / patch
- `benchmark_lane` — held during `bench_runner` / `param_sweep_run` / `eval_runner`
- `profile_lane` — held during `profile`

If your candidate's lane is held by another in-flight task, the
scheduler's `lane_available=0` factor zeros its score. The conductor
also surfaces this via `lease_acquire_failed` events. Either wait or
pick a different action.
