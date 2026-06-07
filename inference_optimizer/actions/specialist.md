# `specialist` action playbook

## Purpose

Dispatch an LLM specialist sub-agent on the **research_lane** to investigate
one canonical gap in depth. The specialist reads advisory RecipeKB
warm-start facts, source-backed research hints, the PR Monitor feed for the
same tag set, and the framework source roots under `INFERENCEX_PATH`.
It then emits exactly one `specialist_done` intent on exit (Inv-5.3 single
exit protocol).

Specialists are the Hyperloom-native counterpart of Arbor's specialist
agents. Unlike the deterministic Python executors (`baseline` / `explore`
/ …) the specialist is *LLM-driven*: SpecialistRunner spawns a fresh
Claude subprocess with the per-tag prompt context, an isolated
`runs/specialist/<task_id>/worktree/` (a `git worktree` rooted at
`INFERENCEX_PATH`), and a tight tool whitelist.

## Inv-5.1 update (Arbor-into-Hyperloom plan, PR-A1)

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

The current phase MUST be `EXPLORE`. Inside EXPLORE the Orchestration
LLM should dispatch specialists as the **primary** entry into a round:

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
short GPU experiments or microbenchmarks, set `needs_gpu=true` and
`gpu_count=N`; the Coordinator allocates visible devices from the separate
specialist GPU pool and injects GPU visibility env vars into the subprocess.
GPU specialists must not launch persistent serving servers or Magpie loops.

## Inputs (task.params)

| Key                  | Type     | Required | Description |
|----------------------|----------|----------|-------------|
| `tags`               | list     | yes*     | Canonical knowledge-domain tags (`framework`, `kernel`, `communication`, `compiler`, `systems`, `pr_intelligence`, `research_scout`). |
| `domain`             | string   | yes*     | Backward-compatible single-tag alias using a SpecialistDomain key. Required only when `tags` is absent. |
| `gap_canonical_id`   | string   | yes      | Stable id of the gap (e.g. `gap.framework.cuda_graph_inefficient.session-<sid>`). |
| `gap_symptom`        | string   | no       | Human summary of the symptom (e.g. `decode kernel at 30% HBM peak`). |
| `gap_layer`          | string   | no       | Layer label (`kernel` / `framework` / `communication` / …). |
| `gap_evidence`       | object   | no       | Profile path + numeric metrics forwarded into the prompt. |
| `max_turns`          | int      | no       | Hard cap on LLM turns (default 8, hard ceiling 16). |
| `needs_gpu`          | bool     | no       | Request the specialist GPU pool for a short GPU experiment / microbenchmark. Default false. |
| `gpu_count`          | int      | no       | Number of GPUs to allocate when `needs_gpu=true` (default 1). |

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
    max_turns         = 8,
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
  - PR feed for `domain.pr_repos`
  - framework source root hints
  - the worktree path
  - allocated GPU ids when `needs_gpu=true`
* A tool whitelist including `Read / Grep / Glob / Bash / Edit / Write
  / MultiEdit / WebSearch / WebFetch` and the relevant MCP servers,
  scoped to the worktree directory (`--add-dir <worktree>`).
* A 60s heartbeat protocol — the subprocess writes
  `runs/specialist/<task_id>/heartbeat.json`; SpecialistRunner reaps
  stale agents after 5 minutes (`HEARTBEAT_STALE_THRESHOLD_S`).
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
    { "name": "...", "extra_args": "...", "extra_envs": {...}, "rationale": "..." },
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
audit; the canonical explore ledger dedups by content fingerprint.
