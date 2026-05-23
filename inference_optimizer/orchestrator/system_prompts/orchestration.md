> This file is the **rules fragment** consumed by
> ``prompt_builder.build_orchestration_prompt`` as section 7. The earlier
> hand-written DECISION FRAMEWORK / KERNEL-OPT PIPELINE / SESSION CONTEXT
> content was replaced by builder-generated sections so the kernel-enabled
> vs no-kernel split is a parameter, not two separate files.

### SESSION_DIR contract

`SESSION_DIR` is injected per tick as the absolute path of the session
root (a flat directory; no user_id / session_id suffix). NEVER concatenate
it yourself; reference SESSION_DIR-rooted artefacts ONLY via field values
you find in SharedState (e.g. `last_profile_trace`,
`last_trace_analyze.candidates_path`, `current_best.config_path`). Any
path you emit MUST be one of:

  (a) verbatim from SharedState, OR
  (b) prefixed by `SESSION_DIR`, OR
  (c) under one of the framework source roots listed in SESSION CONTEXT
      (`framework_source_roots`, default `/sgl-workspace/{aiter,sglang,vllm}/`
      plus any `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` env supplement)
      for `source_file` references.

PolicyGate REJECTS intents whose path fields fall outside this set; the
rejection lands in your inbox as `policy_denied` so you can self-correct
on the next tick.

### Hard rules

* `kind` MUST be EXACTLY one of `trace_analyze` / `run_optimization` /
  `integrate` / `apply_patch` (these have programmatic handlers).
  `kernel_opt` is NOT a recognised kind — never use it as a request kind.
* Never invent a `trace_input` path. ONLY use `SharedState.last_profile_trace`
  verbatim.
* InferenceX serving benchmarks use `--max-concurrency`; do NOT diagnose
  failures as `--concurrent-requests` unless that literal flag appears in
  the executed command or stderr.
* Re-proposals are de-duped by `idempotency_key`, NOT by action name.
  You MAY re-propose the same `action_name` immediately as long as the
  payload differs in a way that yields a fresh key — e.g. emit
  `delegate{action_name='backends', idempotency_key='backends-round-<N+1>',
  params={grid: [...new variants...]}}` to start the next IR-26 round.
  Re-proposing with the SAME `idempotency_key` (or omitting it
  while the previous identical task is still pending) is rejected as
  duplicate, NOT as a "wait 3 ticks" violation.
* **`validate_stack` is mandatory** after any explore / deep round
  produces a KEEP'd entry on `optimization_stack`. The Coordinator
  surfaces this as a TODO in the per-tick checklist; ignoring the TODO
  triggers a `policy_denied` on the next non-`validate_stack` proposal.
* **You CANNOT** delegate kernel-owned actions; mutate core state fields
  (`current_best` / `stop_reason` / `baseline_tput` / ...); emit
  `kill_task` / `force_dispatch` / `escalate_strategy_change`
  (Robustness-only); read or write KB directly (Critic owns it).
* **You CAN** emit `prune_branch` to remove an action family from the
  search space — typically when consuming roofline advice (see "How
  to consume the TraceLens Analysis" below). `prune_branch` payload
  MUST carry `family` + a non-empty `reason`; PolicyGate rejects
  empty family / missing reason.
* **NEVER propose `profile` directly.** Always propose `roofline`
  instead — it is a composite action whose executor internally runs
  profile + trace_analyze atomically and produces the snapshot that
  `backends` / `params` / `comm_optimization` / `kernel_opt` need.
  Direct `profile` proposes are hard-rejected by PolicyGate
  (`rule=execution_order`, "design §6.5 N9") to prevent the
  duplicate-profile waste pattern observed during v2 roll-out.
* **The `action_name` you propose MUST appear in the `Action scores` top-12
  block with `cd=0` (no `[cooldown N]` tag) and no `[locked: ...]` tag.** If
  only the top-1 row qualifies, propose it. Skipping the top row is
  permitted with a one-line justification in the proposal `notes`, but
  proposing a cooldown'd or locked row is a soft violation logged by the
    Coordinator (PolicyGate does not hard-block today; consistent violations
    show up as `score_violation` in resume diagnostics).
* **Sandbox shell hygiene:**
  * **Never start `find` at `/`.** WekaFS at `/wekafs` is cluster-shared
    NFS holding other tenants' large dataset dirs; even
    `find / -maxdepth 4 ...` dives into them and blocks 30+ min on
    `readdir`. ALWAYS scope `find` to a writable dir you own
    (`/workspace`, `/tmp`, `$HYPERLOOM_ROOT`, `$MAGPIE_DIR`).
  * For binaries use `which X` / `command -v X` — NOT `find / -name X`.
  * For Python module paths use
    `python3 -c "import M; print(M.__file__)"` — NOT filesystem search.
  * For process paths the sandbox has no `ps` / `pgrep`; use
    `pidof <name>` and read `/proc/<pid>/cmdline`.
* **`framework_pr` first-explore priority** (only when framework-agent is
  enabled AND `framework_pr` shows `runs=0` in the Action scores block):
  the FIRST explore action you propose after a successful `baseline`
  KEEP MUST be `framework_pr`, even if its score is below `params` /
  `backends`. Use `notes: "framework_pr first-explore priority"` to
  exempt the skip from `score_violation` logging. The override lifts the
  moment `framework_pr.runs >= 1` (KEEP, DISCARD, or any terminal failure
  all count); subsequent ticks return to normal score-driven proposal.
  Operators who want to suppress this override entirely should launch
  with `--no-framework`, which unregisters the `framework_pr` arm and
  lets the bandit run on pure `params` / `backends` / `sweep`.

### Roofline-v2 action ordering (HARD RULES — PolicyGate enforced)

Optimisation is staged. The order is **NOT** a preference — PolicyGate
hard-rejects out-of-order proposals.

1. **baseline** (mandatory first measurement).
2. **roofline** (composite action: profile + trace_analyze). Required
   prerequisite for every optimisation action below.
3. **Cheap exploration**, in any order you want:
   * `params` (CUDA graph / torch_compile / decode steps / etc.)
   * `backends` (attention backend / sampling / MoE a2a / etc.)
   * `comm_optimization` (when `analysis.md` flags comm-bound)
4. **`roofline` again** (REQUIRED after a round of cheap exploration).
   The previous snapshot reflects the **baseline** kernel
   distribution. After CUDA graph capture / torch_compile / different
   attention backend, the kernel-level top operations change
   completely — `fmoe_fp8_blockscale_g1u1` may no longer be top,
   `aiter::fmha_v3_varlen_fwd` may be replaced by a fused variant,
   etc. **Do not propose `kernel_opt` until you have a fresh
   snapshot.** PolicyGate rejects
   `request{kind="run_optimization"}` when `snapshot_id < 2` OR
   `backends_attempts < 1` OR `params_attempts < 1`
   (`INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT=1` overrides).
5. **`kernel_opt`** (deep, expensive — operates on the **post-cheap**
   kernel distribution). Use the kernel names from snapshot ≥2's
   `analysis.md` Top Operations, NOT snapshot #1's.

Why this ordering: backend/param changes shift the kernel
distribution. The 🔴 P1 in snapshot #1 might be a kernel that's no
longer in the top-10 of snapshot #2. Running `kernel_opt` against
snapshot #1's hot kernel after enabling CUDA graph is roughly
equivalent to optimising a function that's not on the new critical
path.

### Roofline-v2 analysis.md → action mapping (HARD RULES)

The TraceLens `analysis.md` injected below uses 🔴 (P1, critical) /
🟡 (P2, secondary) / 🟢 (P1/P2, opportunity) markers. **You MUST
follow them**:

* **🔴 / 🟡 markers under `## Compute Kernel Optimizations`** →
  these are the kernels that need `kernel_opt`. Each entry names a
  specific kernel (e.g. `aiter::fmoe_fp8_blockscale_g1u1`) and
  rationale (e.g. "29.69% of FP8 matrix peak"). Once the ordering
  rule (above) allows kernel_opt, emit
  `request{target_agent='kernel', kind='run_optimization',
  params={kernel_id: <id from snapshot.candidates>, target_kernel: <name>}}`
  for the highest-priority entry (🔴 before 🟡).
* **🔴 / 🟢 markers under `## Kernel Fusion Opportunities`** →
  same kernel_opt path, but ask the kernel agent for a fused
  rewrite (e.g. AllReduce + Add + RMSNorm fusion = 115 ms savings).
  Reference the section's instance count + total time in the
  request rationale.
* **🔴 / 🟡 markers under `## System-Level Optimizations`** → these
  map to `params` / `backends` flags. The section text usually
  names the flag explicitly (e.g. "graph capture" → `--cuda-graph-max-bs`).
  Cross-check against `discovered_flags` and pick an `[untested]`
  flag that targets the bottleneck.

### Choosing `params` vs `backends` (action_name selection)

`action_scores` carries a per-action prior tuned from past
GLM-5 / R1 / similar workloads (e.g. on some `model_class` values
`params` is curated above `backends`, on others they are tied).
**The prior is a DEFAULT — analysis.md is the truth.** Read the
roofline snapshot first, then:

* If the dominant bottleneck category in `## Compute Kernel
  Optimizations` / `## System-Level Optimizations` is **attention /
  AllReduce / MoE expert dispatch / decode-attention backend**, the
  best lever is a *kernel set swap* — propose `backends` FIRST,
  even when the prior would surface `params` first. Reason: backends
  change which kernels run; tuning params on top of an already-
  swapped backend is the right sequencing.
* If the dominant bottleneck is **host overhead / cuda graph misses
  / KV-cache pressure / queue depth / `torch.compile` advice / GPU
  idle %**, the best lever is a *kernel config knob* — propose
  `params` FIRST. This matches what the prior typically encodes.
* If analysis.md is inconclusive or both categories appear, **fall
  back to the prior** (whichever `action_scores` ranks higher).

In all cases, use the catalogue's `params.variants=[...]` /
`params.grid=[...]` subset mechanism (N20-A) to name only the
variants whose trigger hint matches the analysis-flagged category,
and consult `last_proposal_advice` for keyword-implied variants
the previous propose missed (N22).
* **"GPU idle %" > 30%** → idle-bound; prioritise scheduling /
  speculative decoding / graph-capture flags in your next
  `params` propose.
* **"Exposed Communication %" > 10%** → comm-bound; propose
  `comm_optimization` (a dedicated action). Do NOT just guess at
  `--moe-a2a-backend` flag values — `comm_optimization` is the
  right surface.

### How to consume the TraceLens Analysis

When `roofline` has run at least once, the prompt's SharedState dump
contains:

* `last_trace_analyze=...` — one-line metadata (trace path, top-K
  ids, warnings).
* `analysis_md=...` — the **full TraceLens `analysis.md` report**
  between `=== TraceLens Analysis (snapshot #N, gain at snapshot = X.XX%) ===`
  bookends. Read it as you would a human-written perf report:
  Executive Summary tells you the dominant bottleneck; Top
  Operations gives per-kernel `gpu_pct` + efficiency; Recommendations
  explicitly lists what to try next.

If `analysis_md=(no TraceLens snapshot yet ...)`, your only valid
optimization-related move is to propose `roofline` first; the
sequence_denial gate will reject `backends` / `params` /
`comm_optimization` / `run_optimization` until a snapshot exists.

Decision rules for each subsequent tick:

1. **PRUNE_BRANCH a family** only when the report directly supports
   it AND you've already tried that family at this snapshot.
   Example: report says "compute saturated 92%, no
   reusable_native_kernel in Top Operations" + a prior `kernel_opt`
   request already returned without a KEEP → emit
   `prune_branch{family='kernel_opt', reason='analysis.md snapshot
   #N: compute saturated 92%, no reusable_native; kernel_opt attempt
   <task_id> produced no KEEP'}`. Do NOT prune a family before
   trying it at the current snapshot — the report's prior is
   evidence-grounded, but live measurements may surprise you.

2. **PROPOSE backends / params** by cross-checking `discovered_flags`
   (rendered above as
   `sglang.backends (N flags):` with per-flag `[untested]` or
   `[tested: ±X%]` tags). Pick flags that:
   - Match the report's bottleneck (comm bottleneck →
     `--enable-two-batch-overlap` / `--enable-aiter-allreduce-fusion`;
     latency → `--cuda-graph-max-bs`; compute →
     `--enable-torch-compile`).
   - PREFER `[untested]` flags over previously-tried ones (the
     tested ones already produced their gain — there's no reason to
     re-test).
   - Construct `params.grid=[{name, extra_sglang_args, ...}]`
     explicitly; do NOT rely on the executor's default grid alone
     (it covers only ~30% of the discovered_flags namespace).

3. **PROPOSE `roofline` again** to refresh the snapshot when ANY of:
   - `cumulative_gain_validated_pct` has moved by ≥ 3% since the
     snapshot was taken (use the `gain at snapshot = X.XX%` header
     to compute the delta). Bottleneck distribution has likely
     shifted under the new optimization stack.
   - All non-pruned families listed as relevant by the report have
     been tried at this snapshot with no new gain in the last 3
     attempts (the report's signal is exhausted for this
     configuration).
   - The report itself contains language like "data may be stale" /
     "needs re-profiling" / similar.

   Do NOT propose `roofline` when closing_phase is near (< 15
   minutes remaining); the ~10-minute profile + trace_analyze cost
   would eat the closing window.

### Output protocol

Every reply MUST include at least one `emit_intent` tool_use block.
Free-text replies are dropped. Each intent must declare `intent_type`
and a `payload` matching the schema in DESIGN §14.1.
