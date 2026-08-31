# `specialist` action playbook

## Purpose

Dispatch an LLM specialist sub-agent on the **research_lane** to investigate
one canonical gap in depth. The specialist reads advisory RecipeKB
warm-start facts, source-backed research hints, and the framework source
roots under `INFERENCEX_PATH`, and can query the PR Monitor on demand (over
the shared `PR_QUERY_REPOS` allowlist) via `mcp__pr_monitor__*` tools.
It then emits exactly one `specialist_done` intent on exit (Inv-5.3 single
exit protocol).

The specialist is the runtime implementation of the Domain Specialists in
the Arbor paper (arXiv:2606.12563) — "Arbor" being the research name for
this orchestration, so a specialist here *is* an Arbor Domain Specialist.
Unlike the deterministic Python executors (`baseline` / `explore`
/ …) the specialist is *LLM-driven*: SpecialistRunner spawns a fresh
Claude subprocess with the per-tag prompt context and an optional isolated
`runs/specialist/<task_id>/worktree/` (a `git worktree` when
`INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` resolves to a git checkout).

**Containment model (as implemented)**: The specialist runs under
`--permission-mode bypassPermissions` and has access to all built-in
tools except `KillShell` and `SlashCommand` (which are denied via
`--disallowedTools` to enforce the prompt rule against global process
cleanup). The framework checkout listed in `framework_source_roots` is
readable and writable from the subprocess. There is no file-system
containment. Behavioural constraints are enforced by the prompt iron-rules
(MUST NOT write directly to `framework_source_roots`; patches go through
`integrate_patch`) and by post-hoc Critic + PolicyGate review.
The worktree is a best-effort isolation aid for deployments where the
framework is installed as a git checkout; it is absent when the framework
is pip-installed (the default; logged as `no_git_framework_source_root`).

## Inv-5.1 update (specialist-as-patch-author, PR-A1)

Prior to this plan specialists were "research-only" — they could emit
`proposal_set` entries that fed the next `explore` grid but could not
write source patches. The new contract is:

- Specialists MAY write source patches into their own
  `runs/specialist/<task_id>/worktree/`.
- The patch files (`*.patch` / `*.diff`) are produced via
  `git diff > patches/NNN_<slug>.patch` inside the worktree.
- The physical `git apply` against `INFERENCEX_PATH` is the job of the
  `integrate_patch` action — never the specialist itself. Inv-3 (single
  tenant GPU) and Inv-1 (Coordinator-only fact-layer writer) are
  preserved by routing every source-tree mutation through that single
  orchestrator-side step.

## Who delegates this action

* **Orchestration** only. PolicyGate's `specialist_dispatch_source` rule
  rejects `delegate{action_name='specialist'}` from any other role.
  Robustness can recommend a specialist via
  `escalate_strategy_change{hint='need_specialist:<domain>'}`; the next
  Orchestration tick picks it up.

## When to delegate

Specialist is available in `FRAMEWORK_AGENT` and `KERNEL_AGENT`.
Inside FRAMEWORK_AGENT the Orchestration LLM should dispatch
specialists as the **primary** entry into a round:

1. Read `SharedState.gaps[]` (or fall back to `last_action_failures` +
   `explore_search.winners_history` when gaps are empty).
2. Pick the top-K gaps (K up to `min(len(gaps), research_lane_capacity)`;
   `research_lane_capacity` is capped at `2 × visible GPU count`).
3. For each gap, emit one `delegate{action_name='specialist', params={...}}`
   intent in the **same** tick. Claude's tool API supports multiple
   `emit_intent` tool calls per turn, so a single LLM response fans
   out N specialists.

The Coordinator routes each delegate to the TaskRegistry; SpecialistRunner
pulls them off the `research_lane` and runs them in parallel up to the
lane capacity. By default specialists are CPU/research subprocesses. For
GPU experiments, autotuning, profiling, or a real serving + benchmark loop,
set `needs_gpu=true` and `gpu_count=N`; the Coordinator allocates serving-
disjoint visible devices from the separate specialist GPU pool and injects GPU
visibility env vars into the subprocess. On its own leased cards a GPU
specialist is free to write/run scripts, profile, autotune, and start/restart
its own servers (on any port that is NOT the production serving port 8888); the
only hard boundary is the production serving process / its cards / port 8888.

GPU requests are governed identically for **every** scope: a `scope=freeform`
dispatch that sets `needs_gpu` clears the same `gpu_specialist_ceiling`
checks (`specialist_gpu_pool_disabled` / `specialist_gpu_request_exceeds_capacity`)
as a domain-anchored one — freeform is not a GPU loophole. Pair
`mode=patch + bench=true + needs_gpu` to give a freeform specialist a real
measure → edit → measure loop on its own cards inside its worktree.

## Inputs (task.params)

| Key                  | Type     | Required | Description |
|----------------------|----------|----------|-------------|
| `tags`               | list     | yes*     | Canonical knowledge-domain tags (`framework`, `kernel`, `communication`, `compiler`, `systems`, `pr_intelligence`, `research_scout`). |
| `domain`             | string   | yes*     | Backward-compatible single-tag alias using a SpecialistDomain key. Required only when `tags` is absent. |
| `gap_canonical_id`   | string   | yes      | Stable id of the gap (e.g. `gap.framework.cuda_graph_inefficient.session-<sid>`). |
| `gap_symptom`        | string   | no       | Human summary of the symptom (e.g. `decode kernel at 30% HBM peak`). |
| `gap_layer`          | string   | no       | Layer label (`kernel` / `framework` / `communication` / …). |
| `gap_evidence`       | object   | no       | Profile path + numeric metrics forwarded into the prompt. |
| `max_turns`          | int      | no       | Optional turn cap (default 1000, hard ceiling 1000; `0` means unbounded). Depth is primarily bounded by the wall-clock budget. |
| `needs_gpu`          | bool     | no       | Request the specialist GPU pool for wall-budgeted on-GPU work (servers on a non-8888 port, profiling, autotune, benchmark loops). Default false. |
| `gpu_count`          | int      | no       | Number of GPUs to allocate when `needs_gpu=true` (default = serving TP so a TP-coupled gap is reproducible; set explicitly to override, e.g. 1 for a single-card kernel probe). |

## EMIT format

```
delegate{
  action_name = 'specialist',
  params = {
    tags              = ['framework'],
    domain            = 'serving_specialist',  # optional legacy alias
    gap_canonical_id  = 'gap.framework.cuda_graph_inefficient.session-<sid>',
    gap_symptom       = 'cuda graphs disabled at TP=8 on FP8',
    gap_layer         = 'framework',
    gap_evidence      = { profile_trace = '<from SharedState.last_profile_trace>' },
    needs_gpu         = false,
    # Omit max_turns for the default wall-budgeted run; set it only to cap a probe early.
  },
  idempotency_key = 'specialist-framework-<gap_id>-<round_idx>',
}
```

## Specialist contract (subprocess side)

Each specialist subprocess sees:

* A per-domain system prompt with:
  - identity + autonomy scope
  - hardware context
  - the gap statement + evidence
  - optional advisory KB context, warm-start lessons / pitfalls, and
    source-backed research hints
  - a PR-query capability block: the `mcp__pr_monitor__*` tools plus the
    shared `PR_QUERY_REPOS` repo allowlist (self-serve, no pre-warmed feed)
  - framework source root hints
  - the worktree path
  - allocated GPU ids when `needs_gpu=true`
* A tool whitelist including `Read / Grep / Glob / Bash / Edit / Write
  / MultiEdit / TodoWrite / WebSearch / WebFetch / Task` and the relevant
  MCP servers, scoped to the worktree directory (`--add-dir <worktree>`).
  GPU specialists run their measure → edit → measure / autotune loops via the
  broad `Bash`/`Write`/`Edit` grant on their leased cards (the legacy capped
  `run_bench` micro-bench tool has been retired).
  `Task` is limited to `subagent_type="hyperloom-leaf"`: leaves inherit the
  parent specialist's visible devices and their tool set omits `Task`, so
  fan-out is single-layer and stays within the parent's lane/GPU lease.
* A 60s heartbeat protocol — the subprocess writes
  `runs/specialist/<task_id>/heartbeat.json`; SpecialistRunner reaps an agent
  after approximately 5 minutes if neither heartbeat nor process-log activity
  ever appears. Once either activity file has appeared, it reaps the agent
  after approximately 10 minutes of subsequent silence. The staleness
  threshold is currently fixed and has no operator override.
* The exit contract:
  - one `specialist_done` intent (the SpecialistRunner harvests it from
    stdout's stream-json transcript) with payload schema
    `{ gap_canonical_id, domain, tags, proposal_set, empty, summary, ... }`.
  - optionally `patches_written: [<paths in worktree/patches/>]`.
  - `new_findings`, `residual_questions`, `confidence` (0..1).

## Output (specialist_done payload)

```
{
  "gap_canonical_id": "<echoed back verbatim>",
  "domain":           "<echoed back verbatim>",
  "proposal_set": [
    { "name": "...", "extra_args": "...", "extra_envs": {...}, "rationale": "...", "atomic": false },
    ...
  ],
  "patches_written": ["patches/001_cuda_graph_fix.patch"],   // PR-A2+: optional
  "empty":            false,
  "summary":          "short one-liner ≤ 480 chars",
  "confidence":       0.7,
  "new_findings":     ["..."],
  "residual_questions": ["..."]
}
```

When the specialist times out / dies / fails parse, SpecialistRunner
synthesises an empty `specialist_done` payload (`empty=true`,
`proposal_set=[]`, reason filled). The Coordinator never blocks waiting
for a missing specialist result.

## Followup action

If `specialist_done.patches_written` is non-empty, Orchestration should
propose `delegate{action_name='integrate_patch', params={specialist_task_id: <id>}}`
in a subsequent tick. The Coordinator will run apply → restart → gate
serially on the serving lane.

If `specialist_done.proposal_set` is non-empty and the specialist did
not write source patches, Orchestration uses the proposals as the grid
for the next `delegate{action_name='explore', params={grid: [...]}}`
round. Each variant inherits `provenance='specialist:<domain>'` for
audit; variants are deduped by content fingerprint within a single grid.

**`atomic` (do-not-split) flag.** A specialist sets `"atomic": true` on a
proposal whose `extra_args` / `extra_envs` are a **coupled set that only
works together** — e.g. enabling MTP/speculative decoding REQUIRES a paired
`--gpu-memory-utilization` reduction to leave headroom for the draft model,
so the two flags must be benched as one variant. When `atomic` is true,
Orchestration MUST dispatch that proposal **verbatim as a single explore
variant** — do NOT decompose its flags into separate variants, drop any of
them, or re-derive your own version with only part of the coupling. Splitting
an atomic proposal silently defeats the specialist's fix (each half fails or
shows no gain on its own). Non-atomic proposals may still be curated, merged,
or reordered as usual.
