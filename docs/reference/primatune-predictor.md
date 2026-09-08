---
myst:
    html_meta:
        "description": "HTTP contract between Hyperloom's FRAMEWORK predictor pump and an external first-pass tuning model. Covers the request and response bodies, field provenance, decision points, proposal ranking, and attribution."
        "keywords": "Hyperloom, predictor, PrimaTune, first-pass tuning, FRAMEWORK phase, explore, untested proposals, server args, provenance, attribution, AMD GPU, ROCm, LLM inference"
---
# Predictor HTTP contract

Hyperloom can consult an external *first-pass tuning* model at each FRAMEWORK
decision point and file its answer on the untested-proposal queue, alongside the
proposals its own specialists produce. Orchestration reads that queue, composes
the `explore` grid, and dispatches any source-change mandate itself; the
predictor schedules nothing and creates no tasks.

This topic is the wire contract for that call plus the runtime rules the pump
obeys. It is a reference, not a tutorial: the predictor is off by default and a
session that never sets an endpoint behaves exactly as before, down to the bytes
of the orchestration prompt.

The design constraint that shapes everything below: **the model must not run on
the machine under test.** IR-1 requires every visible GPU to be idle before a
serving launch, so a co-resident model would corrupt the very benchmark it is
trying to improve. Hyperloom therefore only builds a JSON document and POSTs
it; rendering, generation and answer parsing all happen on the far side.

---

## Boundary

Hyperloom sends **its own field names**. It does not construct the predictor's
prompt, and it does not adopt the predictor's vocabulary. The consumer owns the
mapping from these names into whatever shape its renderer wants.

That split is deliberate. A renamed key on this wire would be a second,
untested copy of a mapping the consumer already maintains, and the failure mode
of getting it wrong is silent: a prompt renderer that reads a key nobody sent
omits the sentence and reports nothing. Keeping Hyperloom on its native
vocabulary means a drift shows up as a missing key in one place, on the side
that owns the renderer.

```text
Hyperloom                                    Predictor service
---------                                    -----------------
SharedState + analysis.md + source map
  -> request body (native names)  --POST-->  map -> render -> generate
  <-------------------------------  200  --  parse + repair -> action
  -> untested-proposal queue rows
```

## Endpoint

```text
POST <endpoint>/v1/predict
Content-Type: application/json
```

`<endpoint>` comes from `--primatune-endpoint` or `$HYPERLOOM_PREDICTOR_ENDPOINT`.
There is no discovery and no default: without an endpoint the pump does not run.

## Request body

`schema` is the only field a consumer may branch on. Every other key is
best-effort: absent means "Hyperloom could not determine this", and the
consumer must degrade rather than fail.

```json
{
  "schema": "hyperloom.predictor_request.v1",
  "session_id": "Qwen-Qwen3-8B_20260902T064801Z_ae4ae116",
  "identification": {
    "model_name": "Qwen-Qwen3-8B",
    "model_class": "dense",
    "gpu_type": "mi300x",
    "framework": "vllm",
    "framework_version": "0.22.0",
    "precision": "fp8",
    "tp": 4,
    "ep": 1,
    "nodes": 1,
    "model_info": {
      "model_type": "qwen3",
      "attention_type": "gqa",
      "num_hidden_layers": 36,
      "hidden_size": 4096,
      "head_dim": 128,
      "is_moe": false
    }
  },
  "workload": {
    "isl": 8192,
    "osl": 1024,
    "conc": 64,
    "max_model_len": 13312
  },
  "phase": {
    "phase": "FRAMEWORK_AGENT",
    "phase_reason": "prelude_done",
    "phase_elapsed_seconds": 41.2,
    "macro_cycle": 1
  },
  "performance": {
    "baseline_tput": 1820.4,
    "current_best_tput": 1901.7,
    "cumulative_gain_validated": 4.46,
    "keep_threshold_pct": 1.0,
    "optimization_stack": [
      {
        "candidate_extra_server_args": "--enable-chunked-prefill",
        "extra_envs": {"VLLM_USE_AITER": "1"},
        "tput": 1901.7
      }
    ]
  },
  "evidence": {
    "profile_available": true,
    "profile_age_sec": 312,
    "roofline": {
      "roofline_mem_ceiling_tok_per_sec": 4210.0,
      "roofline_cmp_ceiling_tok_per_sec": 9880.0,
      "roofline_bound_kind": "memory",
      "achieved_tok_per_sec": 1901.7,
      "gap_to_roofline_pct": 54.8,
      "hbm_bw_gbps": 5300.0,
      "peak_achievable_tflops": 1307.0,
      "n_ops_memory_bound": 22,
      "n_ops_total": 31
    },
    "window": {
      "total_gpu_time_ms": 263.98,
      "gpu_busy_pct": 71.4,
      "gpu_idle_pct": 22.1,
      "exposed_comm_pct": 6.5
    },
    "operators": {
      "top_bottleneck_category": "gemm",
      "attribution_pct": null,
      "category_pct": {"gemm": 41.2, "attention": 28.7, "moe": 0.0},
      "top3_cumulative_pct": 82.6
    },
    "hot_kernels": [
      {
        "name": "torch_gemm",
        "args": "16x4096x12288 bf16",
        "call_count": 1440,
        "time_us": 38.2,
        "gpu_pct": 14.1,
        "efficiency_percent": 61.0,
        "arithmetic_intensity": 118.4,
        "bound_type": "compute",
        "kernel_category": "gemm",
        "source_file": "tuned_gemm.py",
        "source_line": 395,
        "source_function": "torch_gemm"
      }
    ]
  }
}
```

### Where each block comes from

`identification`, `workload`, `phase` and `performance` are read straight off
`SharedState`; `model_info` is forwarded verbatim from
`summarize_model_config()`. Three of them need a note:

- `keep_threshold_pct` is `resolve_keep_threshold(state)`, not a constant. The
  bar decays with the macro-cycle (`0.1 + 0.9/N`) and doubles on multi-node, so
  a hardcoded `1.0` is only right on the first single-node cycle.
- `current_best_tput` is `current_best["tput"]`, which is also the grading
  anchor (`resolve_grading_anchor_tput()`). Once the stack is non-empty,
  candidates are scored against the reigning champion rather than the baseline.
- `optimization_stack[].candidate_extra_server_args` is **that step's own**
  args. The sibling `extra_server_args` is the accumulation up to that step;
  sending it would make every row repeat all preceding flags.

`evidence` is assembled rather than read. `roofline` comes from
`roofline_snapshots[-1]`, whose optional `perfmodel_breakdown` sub-dict carries
`hbm_bw_gbps` and `peak_achievable_tflops` and whose `ops[]` list is counted to
get `n_ops_memory_bound` / `n_ops_total`. The `window` and `operators` blocks
are parsed out of `last_trace_analyze["analysis_md_text"]` — the Executive
Summary and System-Level Signals tables, plus the per-P-item data tables
aggregated by category. `hot_kernels` is `hot_kernels_top15[:8]` with
`source_line` / `source_function` joined in from `kernel_source_resolution.json`.

`profile_available` reuses Hyperloom's own test for having evidence at all:
truthy `analysis_md_text` or a non-empty `hot_kernels_top15`. When it is
`false`, the four sub-blocks are omitted entirely.

Category names in `category_pct`, `top_bottleneck_category` and
`hot_kernels[].kernel_category` are the report's **canonical** spellings, which
the renderer produces with `canonical_category()`: attention appears as `SDPA`,
matrix multiply as `GEMM`. Consumers should map from those, not from the raw
names a profiler emitted.

### Two conventions that carry meaning

**Each evidence block is all-or-nothing.** A block is sent complete or not at
all. A half-populated block would let the consumer render a sentence about a
window whose duration it does not know, and prompt shapes that never occurred
in a training corpus are worse than an honestly absent block.

**`attribution_pct: null` is not zero.** The deterministic TraceLens path
leaves op-attribution coverage unset, and a consumer should read `null` as "this
report has no attribution column". A genuine `0.0` means every kernel failed to
attribute, which is the case where per-kernel efficiency should be distrusted.
Do not coerce one into the other.

## Response body

```json
{
  "schema": "primatune.predictor_response.v1",
  "parsed": true,
  "action": {
    "server_args": {"--max-num-batched-tokens": "16384"},
    "envs": {"VLLM_ROCM_USE_AITER": "1"},
    "source_change": null
  },
  "actions": [
    {
      "server_args": {"--max-num-batched-tokens": "16384"},
      "envs": {"VLLM_ROCM_USE_AITER": "1"},
      "source_change": null
    },
    {"server_args": {"--kv-cache-dtype": "fp8"}, "envs": {}, "source_change": null}
  ],
  "meta": {
    "model": "primatune-dpo-star-r3",
    "phase_rendered": "EXPLORE",
    "prompt_chars": 4339,
    "finish_reason": "stop",
    "dropped_flags": ["--not-a-real-flag"],
    "samples": 8,
    "chosen_index": 0
  }
}
```

- `parsed` — whether an action was recovered. `false` stops the chain for this
  decision point; it is a normal outcome, not an error.
- `action.server_args` / `action.envs` — already validated against the
  framework's flag catalogue on the consumer side. Hyperloom forwards them into
  `extra_args` / `extra_envs` without re-checking spelling.
- `action.source_change` — prose describing a source edit, or `null`. This is
  **not** a diff; see [The patch channel](#the-patch-channel).
- `actions` — optional, and present when the service samples more than once
  per request. Every distinct proposal, best-first, deduplicated by the service;
  `actions[0]` is `action`. A service without sampling omits it, and Hyperloom
  reads `action` instead, so neither side needs a schema version to branch on.
  Each entry becomes one variant of the same explore round. The service already
  deduplicates samples; Hyperloom does not re-rank or truncate. The entries are
  graded in order with each KEEP folded onto the stack before the next is
  graded, so they are a greedy stacking attempt rather than a set of
  alternatives. Envs that belong to the other serving stack (`SGLANG_*` on
  vLLM, `VLLM_*` on SGLang) are dropped when the grid is built — the consumer's
  `repair()` already strips illegal flags, but it does not strip envs.
- `meta` — advisory. Logged for the shadow-mode comparison and ignored by
  control flow. `dropped_flags` is the useful one: a high rate means the
  consumer's catalogue disagrees with the framework actually installed.
  `samples` reports how many completions the answer was drawn from.

Sampling is worth the extra variants because the head of the distribution is
not where the value was. Replayed against a real session's FRAMEWORK entry at
40 samples, the flag that carried +30% in that session appeared in 11 of them
while greedy decoding proposed no launch flags at all.

Any non-200, a body that fails to parse, or a timeout is treated exactly like
`parsed: false`. The predictor is never allowed to fail a session.

## Runtime behaviour

### Decision points

The predictor answers one question at a time, and its answer is a function of
the request. So asking twice at an unchanged state buys the same proposals for
the price of a second request. `predictor_asked_keys` on `SharedState` records
which questions have been asked, keyed by

```text
c{macro_cycle}-s{stack_depth}-r{len(roofline_snapshots)}
```

That key is what makes the pump safe to call from the phase-entry hook and from
every tick. Each component moves for a reason worth a fresh answer:

- **stack depth** — a KEEP changes `current_best`, `optimization_stack` and
  `cumulative_gain_validated`. The stack is what the answer is conditioned on:
  the same AITER backend switch measured -1.17% on a bare baseline and +2.68%
  stacked on fp8 KV cache in this fleet.
- **macro-cycle** — a `cycle_reloop` re-enters the phase against a different
  stack.
- **roofline generation** — a landed roofline is new evidence. It is also the
  only thing that earns a second look inside a cycle whose first answer landed
  no KEEP, which matters because nothing else defers to the predictor any more.

All three are **pulled** by the pump on its own tick. Nothing outside
`orchestrator/predictor/` pushes to it, and in particular the writeback that
promotes a KEEP or a roofline does not know the predictor exists.

An answer marks its decision point spent whatever came back. A predictor that
declined, or one whose every proposal was already measured, has answered the
question; re-POSTing it next tick would only spend the request again.

### Why it runs at phase entry as well as on the tick

The tick runs the orchestration reactor *before* the FRAMEWORK pump. A
prediction made only on the tick path would therefore miss the phase's first
orchestration turn entirely and sit unread for a full tick. `_on_enter_framework`
calls the pump inside the entry hook, which runs before any reactor pass, so the
first orchestration turn of the phase already sees the queue rows.

This is a visibility ordering, not a priority one. The pump takes no lease,
holds no lane and denies nothing.

### Choosing what reaches the queue

The service samples N times and returns every distinct proposal, but in
*sampling* order — its `chosen` is merely the first sample that parsed. The head
of `actions[]` therefore carries no quality signal, and ranking is the
consumer's job.

`meta.candidates` carries every raw sample, which makes the model's own
self-consistency measurable at no cost: how many samples proposed a variant is a
real signal, and it is surfaced to orchestration as `votes=k/n`. The pump then
applies, in order:

1. the duplicate filters below, against the stack and `explore_search.tested`
2. a sort by vote count, with the sampling order as the stable tie-break
3. **flag-family de-duplication** — a proposal is reduced to the set of knobs it
   moves, ignoring their values, and only the best-voted member of a family
   survives. At N=8, three of eight proposals differed only in `--block-size`;
   without this, three of four slots would have measured one sweep.
4. truncation to `pump.MAX_PROPOSALS` (4), matching the grid size orchestration
   is told to target, so the predictor and the LLM specialists contribute on
   symmetric terms

Everything ranking or family de-duplication set aside is recorded on the round
under `dropped`, with a reason, so a finished session still shows what the model
proposed before the consumer narrowed it.

Historical `explore_search.tested` **is** an eligibility gate here. A proposal
is dropped when its delta was already measured, when its launched recipe matches
one already measured (`already_tested_launch` — the case that caught round-2
`--quantization fp8` on an `fp8_e4m3` champion), when it duplicates another
proposal in the same answer, or when it is already on the stack. Cross-framework
envs (`SGLANG_*` on vLLM and the reverse) are stripped: the service's own repair
drops illegal flags but not envs.

### The queue row

The answer is filed with `record_specialist_round` — the same ledger a finished
specialist writes its `proposal_set` into — as `domain="primatune"` with
`priority=1`. It then surfaces in `=== Untested proposals (current cycle) ===`,
which is outside the SEED gate and so reaches orchestration on every FRAMEWORK
tick.

`priority` is a new primary sort key on that queue and exists because a
predictor round carries no gap: on gap severity alone it would rank below every
gap-anchored specialist proposal. The default is `0`, so a queue with no
prioritised round sorts exactly as it did before the key existed.

Two further keys are recorded and never rendered: `dropped` (above) and
`predict_meta` (`latency_ms`, `prompt_chars`, `samples`, `actions_returned`).

### Gating

The pump returns immediately, before any HTTP call, when:

- the endpoint is unset, or the mode is `off`
- the session is not in `FRAMEWORK_AGENT`
- `framework` is not one of `sglang` / `vllm`. Flag catalogues exist only for
  those two; the consumer cannot validate an answer for the others
- this decision point is already in `predictor_asked_keys`

`mode=shadow` is the default: it renders, calls, parses and logs, then queues
nothing and leaves the decision point unspent. Shadow mode costs no benchmark
time and is the only way to see whether the request above lands inside the
consumer's trained distribution before spending benchmark cycles on it.

## The patch channel

`action.source_change` is prose, not a diff, so it cannot reach
`integrate_patch` directly — that path needs a real patch and a Critic verdict.
It is offered on the queue as an opaque **mandate id**, and orchestration
dispatches the specialist:

```text
source_change -> queue row {mandate_id, mandate}
              -> orchestration delegates specialist{scope=freeform, mode=patch,
                 primatune_mandate_id=...}
              -> Coordinator substitutes the verbatim mandate
              -> patches_written -> Critic -> integrate_patch
              -> apply + bench + accuracy gate -> KEEP/REVERT
```

Carrying an id rather than the text is what keeps the mandate faithful: an LLM
asked to relay prose can reword it, and one asked to relay an opaque token
cannot. `_resolve_first_pass_mandate` in `loop/intent_router.py` looks the id up
and overwrites `task_description` with the stored text. An id it cannot resolve
is left alone and logged — the dispatch proceeds as the ordinary LLM-authored
specialist it looks like, with no predictor credit.

A `task_description` must still arrive: PolicyGate's freeform gate rejects an
empty one before the substitution runs, so whatever orchestration wrote is a
placeholder the Coordinator replaces.

Three details of the resolved params matter:

- **`mode="patch"` is set explicitly.** A free-form specialist defaults to
  `research`, because the profile resolves the mode before the domain is
  assigned and so `FREEFORM_DOMAIN.default_mode` never applies. Without it there
  is no worktree, no patch-writing instruction, and no `patches_written`.
- **`domain` is left unset.** `_forward_integrate_source` overwrites
  `provenance` with `specialist:<domain>` when a domain is present, which would
  erase the attribution label.
- **The mandate is sanitized on our side.** It is interpolated into a one-line
  markdown quote (`> {desc}`) exactly as given, so a newline in it could leave
  the quote and let model-authored text forge a section header in the
  specialist's own prompt. The pump runs it through `flatten_for_prompt`, which
  folds every line separator and defangs code fences and angle brackets, then
  caps the length.

Only the newest mandate of the current cycle is offered. An older one answered a
stack the session has already moved past, and offering every one of them would
grow the block without bound.

## Attribution

Orchestration composes the grid, so a variant it copied off the queue arrives
labelled `llm_direct` like anything else it authored. The label is recovered
deterministically instead of being asked for:
`_stamp_first_pass_provenance` in `loop/proposals.py` fingerprints every grid
variant and re-stamps `provenance="primatune"` on the ones matching a recorded
proposal. A variant whose flags were edited no longer matches, so the edit keeps
its own credit. The grid is **not** reordered — that stays orchestration's call.

`lever_kind` stays inside its own five-value closed set: the config channel
reports `config`, the patch channel stamps `source_patch`. An unknown lever is
silently reduced to an empty string, so inventing a value there would lose the
attribution rather than extend it.

### Reading the numbers afterwards

Three things are worth knowing before comparing proposers in
`session_breakdown.json`.

**Use `decision_trace`, not `optimizations.summary_by_agent`.** The headline
agent bucket is resolved per *operation*, and one explore task is one operation
however many variants it benchmarked. A grid mixing predictor and LLM proposals
has no single owner, so `_round_provenance` reports none and the bucket credits
neither. `decision_trace` is per-variant and carries `provenance`, `outcome` and
`gain_pct` on each entry.

**The predictor spends no tokens, and that is not a gap in the accounting.** It
is a plain HTTP POST to a service on another host, not a gateway LLM call, so it
is deliberately not a `VALID_COMPONENTS` member and appears in neither
`llm_calls.jsonl` nor `token_usage`. Its cost is GPU seconds on the predictor's
own host, which Hyperloom cannot see; `predict_meta.latency_ms` on each round is
the only figure recorded from this side. Specialist spend, by contrast, is fully
attributed — `token_usage.by_component["specialist"]` and per-`task_id` in
`token_usage.timeline`.

**Going first is worth something.** The grading anchor is `current_best`, so
every KEEP raises the bar for whatever is measured afterwards. Part of any early
proposer's measured contribution is having gone first. That is inherent to the
loop rather than introduced here, but it is worth remembering before concluding
that a later proposer underperformed.

## Configuration

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--primatune-endpoint URL` | `HYPERLOOM_PREDICTOR_ENDPOINT` | unset | Service base URL. Unset disables the pump. |
| `--primatune-mode MODE` | `HYPERLOOM_PREDICTOR_MODE` | `shadow` | `off`, `shadow` (predict + log), `active` (file on the proposal queue). |
| `--no-primatune` | — | — | Force `off` regardless of the other two. |
| — | `HYPERLOOM_PREDICTOR_TIMEOUT_SEC` | `120` | Per-request timeout. A timeout reads as a declined answer. |
| — | `HYPERLOOM_PREDICTOR_PHASE_LABEL` | `EXPLORE` | Value sent as `phase.phase`. See below. |

How much of one answer reaches the queue is not an operator knob either: it is
`pump.MAX_PROPOSALS`, held equal to the grid size the orchestration prompt asks
for so neither proposer is handed a structurally larger share. Leftover
`HYPERLOOM_PREDICTOR_MAX_CHAIN` / `HYPERLOOM_PREDICTOR_MAX_VARIANTS` values in
the environment are ignored; the losing-streak cap they configured no longer
exists, because nothing defers to the predictor.

### Why the phase label is configurable

`HYPERLOOM_PREDICTOR_PHASE_LABEL` defaults to `EXPLORE` even though the live
phase is `FRAMEWORK_AGENT`. Before the two were merged, configuration search
*was* `EXPLORE` and only source landing was `FRAMEWORK_AGENT`; the pump feeds
the configuration arm, so `EXPLORE` describes the decision being made rather
than the enclosing phase. Consumers trained before the merge saw that name for
the overwhelming majority of comparable decisions.

Set the variable to `FRAMEWORK_AGENT` to pass the live phase through instead.
The request otherwise stays identical, and `meta.phase_rendered` in the
response records which label the consumer actually rendered.
