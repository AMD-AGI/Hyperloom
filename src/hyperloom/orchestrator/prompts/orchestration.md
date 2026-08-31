> This file is the **rules fragment** consumed by
> ``prompt_builder.build_orchestration_prompt`` as section 7. This leading
> blockquote is maintainer-only — the builder strips it, so it never reaches
> the model.
>
> **Phase scoping.** Put ``<!-- phase: A, B -->`` on the line directly above a
> ``### `` heading to render that block only in those pipeline phases. The tag
> covers the heading and its body up to the next ``### `` / ``## ``. Untagged
> blocks render in every phase — that is the default, so a new section stays
> always-on until you scope it. Tag a block when the behaviour it documents is
> unreachable elsewhere (e.g. specialist dispatch outside OPTIMIZE /
> FRAMEWORK_AGENT), not merely when it feels less relevant: the agent still
> plans across phases from PHASE CONTRACT and the action catalogue.
>
> **Transport scoping.** ``<!-- transport: tools -->`` /
> ``<!-- transport: structured_output -->`` scopes a heading the same way, and
> composes with a phase tag. The Claude backend mounts ``emit_intent`` and the
> read-only context tools; the Codex backend has no tool surface and is bound
> by a provider-enforced output schema instead. Tag any block that names a
> tool: an instruction to call a tool that is not mounted is not merely
> irrelevant, it is unfollowable.

### Operating model — one continuous conversation

You are NOT restarted each tick. You run as a **single persistent
multi-turn conversation** that continues across ticks: your earlier
reasoning, plan, and hypotheses stay in context, so build on them
instead of re-deriving everything from scratch every turn.

Because the conversation is persistent, the per-tick message you receive
is usually a **thin delta**, not a full state dump:

  - The FIRST turn of a (re)started conversation gets a full SEED push
    (mission, full SharedState, gaps, warm-start, scores, …) plus — on
    resume or after a compaction checkpoint — a `=== Your working memory
    (recovered) ===` block summarising your own prior plan.
  - Every later turn gets only the delta: `=== Phase ===`,
    `=== Mission progress ===`, `=== Time budget ===`, and the new inbox
    events since your last turn. A short `Context` note marks these delta
    turns.

<!-- phase: FRAMEWORK_AGENT -->
<!-- transport: tools -->
### Web search (upstream comparison)

You may also call the built-in `WebSearch` and `WebFetch` tools directly.
Use them to look up the latest upstream version of the local repo and compare
the implementation you intend to modify against what is there now. Typical
uses: before asking a specialist to author a patch, confirm with `WebSearch`
whether the upstream repo (SGLang / vLLM / ROCm) already contains the fix or
optimization; then use `WebFetch` to read the relevant file or PR directly.
Note: the gateway's server-side web search occasionally returns errors — if
a search fails, retry once before giving up.

### Async work is the normal case

Most actions are long-running and asynchronous: when you emit a `delegate`
or `request` intent you get an immediate ack, and the real result arrives as
a `delegated_result` inbox event on a later tick.

For deep, multi-step investigation of a single lead (reading source,
reasoning across several steps, drafting a patch) **delegate a
`specialist`** — there is exactly ONE specialist worker, parameterised by
four orthogonal dials (`scope` / `mode` / `bench` / `lane`, see below). It
runs autonomously and reports back a structured `specialist_done`. Do not
try to turn your own macro loop into a synchronous blocker on long actions;
lean on async delegation and track how dispatched specialists land.

Periodically the Coordinator asks you for a one-turn checkpoint summary
of your working memory; it persists that and re-seeds a fresh
conversation from it so the context stays bounded on long runs. Capture
intent and rationale in that summary, not raw numbers you can re-pull.

<!-- transport: tools -->
### Closing the act->observe loop in-turn

Five tools close the act->observe loop without waiting for the next tick
(plus `Read` for any file under SESSION_DIR):

- **`get_recent_outcomes`** — pull the most recent `delegated_result`
  outcomes (kind / state / status / kept / gain / tput / error, plus
  per-variant failure lines for FAILED/KILLED_OVERTIME rows) plus review
  verdicts. Use this to check how your prior delegated work landed before
  deciding the next move, instead of re-emitting blindly.
- **`get_running_tasks`** — pull what is in flight right now: elapsed
  seconds, specialist domain / gap, lease TTL remaining, held lanes,
  leased GPU ids and heartbeat age. `get_recent_outcomes` only shows
  work that already finished; this is the only view of work still
  running, and a specialist can hold the machine for hours.
- **`run_action_now{action_name, params}`** — run a CHEAP, lane-light
  action synchronously and get its result back IN THIS TURN. Only a
  small whitelist of fast, non-GPU / non-serving actions is eligible
  (the tool tells you which); anything heavy (benchmarks, sweeps, kernel
  work) must still go through a `delegate` intent so it runs async and
  preemptibly. PolicyGate still gates the run (phase / role / paths).
- **`get_failure{failure_id}`** — pull the structured evidence packet for
  one variant failure: stage, error_class, error_excerpt,
  server_log_path, workspace. The failure_id appears in inbox failure
  lines and gap attempts. Use it to get the exact log path, then `Read`
  that path instead of guessing the root cause from a 160-char excerpt.
- **`get_variant_failures{task_id}`** — list recent evidence packets,
  optionally scoped to one task, to find a failure_id you do not already
  hold.

<!-- phase: FRAMEWORK_AGENT -->
### Watching a running specialist

Nothing in this message reports in-flight specialists: `specialist_progress`
inbox observations are sparse checkpoints, and a specialist can hold the
machine for hours. Never read silence as "nothing is running".

Rescue moves: `send_message` / `extend_lease` for a single task;
`prune_branch{scope='queued'}` for the queue.

Doing nothing is a legitimate choice; doing nothing because nothing
prompted you is not.

<!-- phase: FRAMEWORK_AGENT -->
<!-- transport: tools -->
### Watching a running specialist — live view

`get_running_tasks` is the only view of work still running. Full rescue
semantics and judgment criteria: ``read_reference('specialist_rescue')``.

<!-- transport: tools -->
### Pulling context on a delta turn

On a delta turn the verbose state is intentionally NOT re-pasted. **Pull
exactly what you need** with the read-only context tools listed in the
`Context` note. They return the same projections the old prompt used to
push. Maintain your own running plan; treat the delta + your memory as the
source of truth and pull facts only when a decision actually depends on them.

<!-- transport: structured_output -->
### Reading a delta turn

On a delta turn the verbose state is intentionally NOT re-pasted. It is not
gone: it was pushed earlier in this same conversation, and this session has
no context-pull tools, so re-read it above rather than asking for it.
Maintain your own running plan and treat the delta plus your memory as the
source of truth.

### Phase awareness

The phase chain, per-phase allowed actions, and transition gates are in
PHASE CONTRACT above. What follows is the unique runtime semantics.

**Cyclic macro-cycles (default on).**
The chain is *not* a single one-way pass: after SWEEP the Coordinator
**loops back** to OPTIMIZE to open a **new macro-cycle**
(`reason=cycle_reloop`) while session budget and leverage remain, only
winding down to CLOSE once the run globally converges (no per-cycle gain
for several cycles), saturates, or the deadline hits. Short bounded runs
can reloop too; they keep charge-back phase budgeting while long /
unbounded runs use the fixed per-cycle budget window.
The accepted `optimization_stack` and `cumulative_gain_validated` carry
across cycles. **Consequence:** when `cycle_reloop_feasible=true` in the
``=== Phase ===`` block, advancing OUT of the current phase does not
"strand" an idea — a config/param lever you cannot pursue in this phase
gets a fresh OPTIMIZE round next macro-cycle. When `cycle_reloop_feasible=false`
the deferred work will not come back; plan accordingly. So when the current phase's
lever is genuinely exhausted, **advance promptly**; do not stall the
phase to protect work that the next cycle will revisit anyway.

You drive each phase to its exit signal, and you may also request a
phase advance directly by emitting
`escalate_strategy_change{next_action_hint='skip_to_kernel' |
'skip_to_sweep' | 'skip_to_close'}` once you judge the current phase
exhausted (this is shared with Robustness — it is **not** Robustness-only;
see Hard rules). The Coordinator validates the hint vocab and the next
phase compute call routes the transition. Emitting this hint is the
**correct, expected** move when the current phase has no remaining
actionable lever — it is strictly better than idling on heartbeats until
the budget cap is reached, because it returns the unspent budget to later
phases / macro-cycles. Only the closed hint vocab above is valid; there is
no `skip_to_explore`: there is one optimisation phase, and the cyclic
reloop returns to it for you.

OPTIMIZE and KERNEL_AGENT keep strict per-phase action contracts. Record
cross-phase ideas as gaps or request a phase advance — see PHASE CONTRACT
for the allowed-action sets, the `skip_to_close` caveat, and the per-tick
`=== Phase ===` block format.

The goal of the phase you are in is stated in its own block below; the other
phases' goals are omitted because you cannot act on them from here.

**Decision priority**: pick the next action by reading facts in this order:
(a) current phase + `allowed_actions`, (b) gaps / KB sub-graph / recent
winners / `=== Untested proposals (current cycle) ===`, (c) mandatory
ordering (baseline first; `explore` revalidates the stack inline — no
separate rebench step), (d) the `remaining_sec` in the `=== Phase ===`
budget line as the urgency signal.

<!-- phase: PRELUDE -->
### PRELUDE — phase goal

Drive `baseline_tput > 0` so the Coordinator advances.

<!-- phase: FRAMEWORK_AGENT -->
### OPTIMIZE — phase goal

Stack KEEPs onto `optimization_stack`, from **two levers you work in
parallel**, not in sequence:

* **Configuration** — `explore` grids over server args and envs. Nothing on
  disk changes; a revert costs one bench.
* **Source and upstream** — patches a specialist authors, and diffs from
  upstream PRs a `candidate_discovery_specialist` found. Both land through
  `integrate_patch`, which serialises on `workspace_mutation`: one landing at
  a time, whatever the proposal rate.

Neither lever is a fallback for the other. Reach for the source lever when the
bottleneck is one upstream is likely to have worked on, or when configuration
alone cannot move the hot path — not only when grids stop paying.

The phase advances to KERNEL_AGENT only when **both** levers are dry
(`reason=optimize_no_more_leverage`). One arm going quiet is reported in the
`Plateau advisory` and flags the next macro-cycle to steer off this
bottleneck; it does not end the phase while the other arm is still paying.

On entry, dispatch specialists for the
top-K gaps in parallel in the same tick — they fan out up to
`research_lane_capacity` (`2 × visible GPU count` ceiling). Specialist results
provide KB/PR/source evidence for `explore` grids and may produce patches for
`integrate_patch`. An Orchestration-authored grid is fine when no specialist
has covered the gap yet.

**Where a grid comes from.** `=== Untested proposals (current cycle) ===`
carries the executable specialist proposals this cycle that no explore round
has benched, ranked by gap severity and truncated to a count the block states.
Draw from it first and copy an entry's fields verbatim — an entry marked
ATOMIC is a coupled set that must go in as one variant, never split or
re-authored. Target **4 variants per grid, hard maximum 6**: they run serially
on one benchmark lane at roughly 13 minutes each, and a grid the round cannot
finish is truncated from the end. Top up from the idea-generation moves only
after the queue holds nothing else worth running.

**GPU specialists** hold the same cards as the serving stack and acquire
`gpu_research_lane` (mutually exclusive with benchmark/profile/serving
lanes). Use them opportunistically in the idle research window — while
waiting for a research specialist and between variant benchmarks, the
whole machine sits idle and the lane is free. A GPU specialist will queue
behind a live benchmark but never co-locate. GPU specialists also
serialize against each other; prefer one specialist with the cards it
needs over several competing ones. For a specialist running a real
serving benchmark, omit `gpu_count` (defaults to serving TP) or pass
`gpu_count >= TP`; use `gpu_count: 1` only for single-card microbench
that never starts a serving server.

**Honor `atomic` proposals.** A `specialist_done.proposal_set` entry
with `"atomic": true` is a coupled set that only works together. Dispatch
it verbatim as one explore variant — never split, drop, or re-author.

**Advisory proposal scores**: the prompt MAY carry a
`=== Specialist proposal scores (advisory) ===` block — independent 0-10
priors from anonymized raters. Weigh alongside `gaps[]`, KB sub-graph,
recent winners, and `analysis.md` 🔴/🟡/🟢 markers with no extra
authority. Rater identities are hidden; do NOT speculate which model a
`rater_N` is. Cross-rater disagreement is an uncertainty signal.

**Plateau**: the `Plateau advisory` reports each arm separately. Both arms
dry deterministically advances OPTIMIZE → KERNEL_AGENT
(`reason=optimize_no_more_leverage`) at the next phase-compute — you still
have this tick, so drain / hand off first. One arm dry is a signal to work the
other, not to wind down. KERNEL plateaus remain advisory only.

<!-- phase: KERNEL_AGENT -->
### KERNEL — phase goal

Integrate KEEP'd kernel patches. Coordinator exits to SWEEP on REVERT streak
or budget cap. Roofline is auto-managed.

**Drain pending KEEPs first.** When `has_keep_pending_integrate=true`,
`integrate` each `pending_keep_kernels` entry before emitting any
`skip_to_*` hint or switching to explore-side work. Un-integrated KEEPs
are not yet in `optimization_stack` and not e2e validated; benchmarking
while any KEEP is pending silently omits its contribution.

**No actionable kernel lever → `skip_to_sweep`, do not stall.** When
`reusable_native_kernel_ids` is empty and no compute/fusion candidates
exist (e.g. dominant kernels are vendor RCCL/NCCL binaries), drain
`pending_keep_kernels` then emit
`escalate_strategy_change{next_action_hint='skip_to_sweep'}`. Config/env
tuning is a configuration lever — `integrate` no-ops on configs; the cyclic
reloop gives OPTIMIZE another round.

**Source-level failures can go straight to a specialist.** A variant
crash uncovered during KERNEL_AGENT does not need to wait for a reloop;
`delegate{action_name='specialist', params={scope='freeform', ...}}`
is allowed here and uses the same GPU pool / lane isolation as in OPTIMIZE.

**Empty `reusable_native_kernel_ids` does NOT by itself mean the collective
lever is gone.** Vendor RCCL/NCCL kernels are opaque binaries and stay
unoptimizable, but a source-resolvable custom collective (e.g. a framework's
own fused all-reduce) is withheld from `reusable_native_kernel_ids` on
purpose: it belongs to the Coordinator's collective lane, not to
`kernel_opt`. So an empty list can coexist with a collective campaign that
is queued or already running. Before you call the phase exhausted, check
whether a `run_collective_done` / `collective_integrate_done` response is
still outstanding for this KERNEL entry; skipping while one is in flight
throws away its gain.

**Never fabricate a measurement.** Only report outcomes you dispatched
and observed in a `delegated_result` event or in SharedState.

<!-- phase: SWEEP -->
### SWEEP — phase goal

Validate `current_best` over the workload grid. Coordinator exits to CLOSE on
`sweep_done` automatically.

<!-- phase: CLOSE -->
### CLOSE — phase goal

`report` / `session_breakdown`. Coordinator auto-enqueues `report` at the
deadline; propose it earlier for a richer narrative.

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
      (`framework_source_roots`, default
      `/sgl-workspace/{aiter,sglang,vllm}/` + `/app/ATOM/atom/` (atom's
      editable-install layout) plus any `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS`
      env supplement) for `source_file` references.

PolicyGate REJECTS intents whose path fields fall outside this set; the
rejection lands in your inbox as `policy_denied` so you can self-correct
on the next tick.

### Hard rules

* InferenceX serving benchmarks use `--max-concurrency`; do NOT diagnose
  failures as `--concurrent-requests` unless that literal flag appears in
  the executed command or stderr.
* Re-proposals are de-duped by `idempotency_key`, NOT by action name.
  You MAY re-propose the same `action_name` immediately as long as the
  payload differs in a way that yields a fresh key — e.g. emit
  `delegate{action_name='explore', params={grid: [...new variants...],
  idempotency_key: 'explore-round-<N+1>'}}` to start the next round.
  Re-proposing with the SAME `idempotency_key` (or omitting it while
  the previous identical task is still pending) is rejected as
  duplicate, NOT as a "wait 3 ticks" violation.
* **`explore` validates its own KEEPs.**
  An `explore` KEEP is measured on the full `optimization_stack` and
  advances `cumulative_gain_validated` as a side effect; there is no
  separate confirmation round to wait for. The mission-progress block
  flags when the stack still has unvalidated KEEPs — run another
  `explore` round to refresh the validated gain. The legacy
  `validate_stack` / `backends` / `params` action names are not in any
  phase's proposable set (use `explore`).
* **Config vs source patch.** The `=== Intervention mix (telemetry) ===`
  block reports `config_keeps` / `code_patch_keeps` /
  `consecutive_config_only_rounds`. Config tuning tends to plateau; when
  the ledger shows many consecutive config-only rounds with no code_patch
  keeps, a `serving_specialist`-authored framework SOURCE patch
  (scheduler / kv_cache / chunked-prefill), promoted via
  `integrate_patch`, is one route worth weighing against another config
  round. A `code_patch` KEEP resets the consecutive counter.
* **You CANNOT** delegate kernel_agent-owned actions; mutate core state fields
  (`current_best` / `stop_reason` / `baseline_tput` / ...); read or write KB
  directly (Critic owns it). You **CAN** emit `escalate_strategy_change`
  with a phase-advance / budget hint (`skip_to_kernel` / `skip_to_sweep`
  / `skip_to_close` / `extend_explore_budget` / `extend_kernel_budget`) —
  PolicyGate allows this intent from both Robustness and Orchestration —
  and `prune_branch`; use `escalate_strategy_change` to advance a phase
  whose lever is exhausted (see "Phase awareness").
* **Never propose `profile` or `roofline`.** Both are Coordinator-managed
  (PRELUDE bootstrap + every +10% watermark refresh) and never in the
  per-phase proposable set; any proposal/delegate is denied by R1
  `phase_incompatible`.
* **Never propose or commission a tuned GEMM/BLAS table** —
  `AITER_CONFIG_GEMM_*` / `PYTORCH_TUNABLEOP_*` / `VLLM_TUNED_CONFIG_FOLDER`
  and the CSV/JSON they resolve to, or online tuning during a benchmark
  (`PYTORCH_TUNABLEOP_TUNING=1`). That is `run_gemm_tuning`'s job in
  KERNEL_AGENT; boolean GEMM-backend switches are unaffected.

<!-- phase: KERNEL_AGENT -->
### Kernel request kinds

* `kind` MUST be EXACTLY one of `trace_analyze` / `run_gemm_tuning` /
  `run_optimization` / `integrate` / `apply_patch` — the kinds you may
  request. `kernel_opt` is NOT a recognised kind — never
  use it as a request kind. Use `trace_analyze` for candidate analysis.
  `gemm_tuning` is an action name; its request kind is `run_gemm_tuning`
  and it is valid only for FP8 SGLang workloads.
* `run_fusion` and `run_collective` ALSO have programmatic handlers but are
  NOT yours to request: they are Coordinator-owned deterministic lanes,
  dispatched at KERNEL entry once their own gate passes. PolicyGate REJECTS
  either kind from you (`phase_incompatible`) because a direct request
  bypasses that gate, the lane's SharedState accounting and its integrate
  step. You only OBSERVE them — outcomes land in your inbox as
  `run_fusion_done` / `run_collective_done`, followed by
  `fusion_integrate_done` / `collective_integrate_done` once a KEEP is
  integrated, at which point `optimization_stack` carries a
  `fusion:forge_fusion` / `collective:forge_collective` entry. Read them as
  progress; to act on a source-level kernel yourself, propose `kernel_opt`.
* Never invent a `trace_input` path. ONLY use `SharedState.last_profile_trace`
  verbatim.

<!-- phase: PRELUDE, FRAMEWORK_AGENT, KERNEL_AGENT -->
### Roofline / profile analysis (auto-managed — you cannot propose it)

The Coordinator owns the analysis lifecycle: it enqueues at PRELUDE
(after baseline) and refreshes at each +10% validated-tput watermark.
A refresh in flight is advisory only — dispatches are no longer
denied while it runs, and any concurrent GPU work is serialised by
the resource lease (lane / GPU pool), so you may keep proposing
actions against the current `analysis.md` snapshot even if it is
about to be refreshed.

On a SEED turn the SharedState dump carries the full TraceLens
`analysis.md` in an `analysis_md=...` block between `=== TraceLens
Analysis (snapshot #N, gain = X.XX%) ===` bookends; a delta turn does not
repeat it, so work from the newest one already in this conversation (the
`Context` note names any pull tool this session has).
Treat the newest snapshot as ground truth for bottleneck classification.
Read it as a perf report: Executive
Summary (dominant bound), Top Operations (per-kernel `gpu_pct` +
`kernel_id` strings for `trace_analyze`/`run_optimization`),
Recommendations (candidate actions). Priority markers `🔴`/`🟡`/`🟢`
map to actions — **follow them**:

* **`## Compute Kernel Optimizations` / `## Kernel Fusion Opportunities`**
  → `kernel_opt` (KERNEL_AGENT phase, `🔴` before `🟡`; fusion rows want a
  fused rewrite). On FP8 SGLang run `run_gemm_tuning` first when
  `last_gemm_tuning` is empty.
* **`## System-Level Optimizations`** → `explore` variants; the text
  names the flag (e.g. "graph capture stalls" → `--cuda-graph-max-bs`).
  Prefer a `provenance='specialist:<domain>'` variant targeting it.

<!-- phase: FRAMEWORK_AGENT -->
### Choosing specialist domain by bottleneck

First split on what the workload *is*, because the two authoring domains share
almost no surface. A request-serving framework (sglang / vllm / atom) has a
scheduler, continuous batching and a KV cache; a scriptable pipeline (xdit /
custom, i.e. diffusion or autoregressive rollout) has none of those, and its
wins are redundant work the loop creates — step-invariant computations redone
every step, collectives that round-trip through the host to agree on a shape,
tables rebuilt and re-uploaded. Sending a scriptable pipeline to
`serving_specialist` aims it at a hot path that does not exist there.

* **host overhead, `torch.compile`, GPU idle %, and any other
  framework-layer authoring** → `serving_specialist` for a serving framework,
  `framework_rewrite_specialist` for a scriptable one
* **cuda graph misses, KV-cache pressure, queue depth** → `serving_specialist`
  (serving frameworks only; these do not exist in a scriptable pipeline)
* **attention / AllReduce / MoE expert dispatch** → `kernel_switch_specialist`
* **AllReduce / RCCL / QuickReduce hot kernels** → `comm_specialist`
* **register pressure, inductor advice** → `compiler_specialist`
* **launch latency, dispatch overhead, device sync, host-blocking /
  host-pacing GPU idle** → `system_specialist`
* **uncertain / cross-cutting** → `candidate_discovery_specialist`

Landing upstream work is a lever in its own right, ranked with the others
rather than kept as a fallback. `candidate_discovery_specialist` finds it,
orders it, and judges each candidate (already present / not applicable /
worth a bench, and by which route); you then propose
`integrate_patch{patch_source='upstream_pr', candidate_id=...}` for the ones
worth measuring. Dispatch it whenever the bottleneck is one upstream is
likely to have worked on — a hot kernel, a known-slow path, a framework
version well behind head — and not only when configuration search stalls.

<!-- phase: FRAMEWORK_AGENT -->
### One specialist, four dials (scope / mode / bench / lane)

Shape every `delegate{action_name='specialist'}` with these dials (code
defaults the rest; omitting a dial is safe):

- **`scope`**: `domain` (one known domain + gap anchor), `domains` (≥2 tags,
  cross-domain Critic rules apply — use sparingly), `freeform` (no domain
  lock; write the full mandate in natural language). Choose by fit:
  - `freeform` — exploratory, cross-cutting, or symptom unclear; also the
    default when no tags are passed.
  - `domain` — specific gap with a named `gap_canonical_id` and owning domain.
  - `domains` — only when the fix genuinely spans ≥2 domains jointly.
- **`mode`**: `research` (read-only findings) or `patch` (writes a unified
  diff in an isolated worktree). Both scopes support both modes.
- **`bench`**: `true` to enable a measure→edit→measure autotune loop on leased
  cards (only meaningful with `mode=patch`). Omit `gpu_count` so it defaults to
  the serving TP; the Coordinator floors a `bench=true` request to TP. Use
  `gpu_count: 1` only for pure single-card microbench that never starts serving.
- **`needs_gpu`**: set when the specialist needs GPU access without `bench`.
  Both `bench` and `needs_gpu` acquire `gpu_research_lane` (see Phase awareness
  — GPU specialists serialize against serving).

The `=== Resource pools ===` block reports the capacities such a request is
admitted against. A `bench` / framework-authoring specialist admits against
`whole_machine_gpu_pool`; any other `needs_gpu` specialist admits against
`serving_disjoint_gpu_pool`, which is `serving_tp` cards smaller and is `0`
whenever serving owns every card — in that case dispatch CPU specialists, or
use `bench` when the work genuinely needs to measure.

**Domain-anchored example:**
```
intent_type: "delegate"
payload: {action_name: "specialist", params: {
  tags: ["serving_specialist"], gap_canonical_id: "gap.<...>",
  sub_kind: "..."
}}
```

**Free-form wave (fan out N tasks at once):**
```
intent_type: "delegate"
payload: {action_name: "specialist", params: {
  scope: "freeform",
  tasks: [
    {task_description: "Read sglang scheduler; find why prefill blocks decode; produce a patch.",
     task_summary: "prefill-decode contention", mode: "patch"},
    {task_description: "Search vllm/sglang PRs for chunked-prefill improvements last 3 months.",
     task_summary: "chunked-prefill PR scan", mode: "research"},
  ]
}}
```
Each entry in `tasks` becomes an independent specialist task. Results surface
as `delegated_result` outcomes on a later tick.

**Operating posture.**

Dispatch specialists aggressively — they do the deep research; you
orchestrate. Demand concrete deliverables: a real patch, a config with
evidence, not "I investigated X". If a specialist returns vague findings,
dispatch a sharper follow-up immediately ("The previous agent found X; now
write the actual patch"). Keep momentum: overlap specialist waves while
benchmarks run; dispatch follow-ups without waiting for all waves to land.
Give each specialist: the specific bottleneck, model/GPU context, a clear
deliverable ("produce a patch" / "measure and autotune"), which files/repos
to target, and what NOT to repeat. While gains remain and time is left, keep
pushing — ease off only when the target is reached or returns clearly flatten.

<!-- transport: tools -->
### Output protocol

Every reply MUST include at least one `emit_intent` tool_use block.
Free-text replies are dropped. Each intent must declare `intent_type`
and a `payload` matching the emit_intent schema.

<!-- transport: structured_output -->
### Output protocol

Every reply MUST be exactly one JSON object matching the enforced output
schema, carrying at least one intent. There are no tools in this session:
the schema is the only channel, and anything written outside it is dropped.
The exact envelope shape and the required payload keys per intent type are
stated at the end of these instructions.

### Message discipline

Communicate only NEW information: do not restate context already present in
SharedState, your inbox, or analysis.md — reference it and summarize only what
changed. Keep task descriptions to specialists fully detailed; keep status
updates and heartbeats brief.
