---
myst:
    html_meta:
        "description": "HTTP contract between Hyperloom's FRAMEWORK-entry predictor pump and an external first-pass tuning model. Covers the request and response bodies, field provenance, the greedy chain, gating, and attribution."
        "keywords": "Hyperloom, predictor, PrimaTune, first-pass tuning, FRAMEWORK phase, explore, server args, provenance, attribution, AMD GPU, ROCm, LLM inference"
---
# Predictor HTTP contract

Hyperloom can consult an external *first-pass tuning* model at FRAMEWORK entry
and turn its answer into ordinary `explore` variants and specialist mandates.
This topic is the wire contract for that call plus the runtime rules the pump
obeys. It is a reference, not a tutorial: the predictor is off by default and a
session that never sets an endpoint behaves exactly as before.

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
  -> explore variants / specialist mandate
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
  "meta": {
    "model": "primatune-dpo-star-r3",
    "phase_rendered": "EXPLORE",
    "prompt_chars": 4339,
    "finish_reason": "stop",
    "dropped_flags": ["--not-a-real-flag"]
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
- `meta` — advisory. Logged for the shadow-mode comparison and ignored by
  control flow. `dropped_flags` is the useful one: a high rate means the
  consumer's catalogue disagrees with the framework actually installed.

Any non-200, a body that fails to parse, or a timeout is treated exactly like
`parsed: false`. The predictor is never allowed to fail a session.

## Runtime behaviour

### The greedy chain

The predictor is a single-shot predictor, not a search. But a KEEP changes
`current_best`, `optimization_stack` and `cumulative_gain_validated` — a new
state, and one the model is equally equipped to answer. So the pump re-fires
after each accepted variant:

```text
FRAMEWORK entry -> reprofile -> predict -> explore task -> KEEP
      ^                                                     |
      |________________ stack depth + 1 ____________________|
```

Termination is a property of the idempotency key rather than a counter. The key
is `primatune-c{macro_cycle}-s{stack_depth}`, and the task registry returns an
existing row for a known key without regard to its state. No KEEP means the
stack depth is unchanged, which means the key repeats, which means nothing is
enqueued and the chain stops. Because `coordinator.db` is durable, a resumed
session does not re-benchmark a prediction it already tried.

Two caps bound the chain independently, whichever comes first: a step count and
a fraction of the FRAMEWORK budget.

### Going first is free

`_on_enter_framework` runs synchronously at phase entry, while an
orchestration-proposed `explore` has to wait for the next tick's reactor pass.
The entry prediction therefore takes the serving lane's lease first without any
suppression of existing logic. Nothing about the LLM's proposal path changes.

### Budget must be checked, not assumed

Phase budgets are cumulative across entries, so a `cycle_reloop` does not hand
out a fresh 40% and can re-enter a phase with nothing left. The chain re-fires
whenever `macro_cycle` changes, so the idempotency key alone will not hold it
back: the pump reads the remaining budget first and declines when the phase has
already spent more than `HYPERLOOM_PREDICTOR_BUDGET_PCT` of it.

That default of 25% also encodes the intent. This is a *first-pass* tuner: it
should spend its predictions early, while the cheap wins are unclaimed, and
leave the rest of the phase to the search.

### Gating

The pump returns immediately, before any HTTP call, when:

- the endpoint is unset, or the mode is `off`
- `framework` is not one of `sglang` / `vllm`. Flag catalogues exist only for
  those two; the consumer cannot validate an answer for the others
- FRAMEWORK budget is exhausted
- the chain has hit its step or budget cap

`mode=shadow` is the default and goes one step further: it renders, calls, parses
and logs, then enqueues nothing. Shadow mode costs no GPU time and is the only
way to see whether the request above lands inside the consumer's trained
distribution before spending benchmark cycles on it.

## The patch channel

`action.source_change` is prose, not a diff, so it cannot go to
`integrate_patch` directly — that path needs a real patch and Critic approval.
Instead it becomes the mandate of a free-form specialist, which writes the diff
and then rejoins the normal route:

```text
source_change -> freeform specialist -> patches_written -> Critic
              -> integrate_patch -> apply + bench + accuracy gate -> KEEP/REVERT
```

The specialist prompt needs no changes; its free-form mandate block was built
to carry exactly this kind of externally supplied instruction. Three details do
matter:

- **`mode="patch"` must be set explicitly.** A free-form specialist defaults to
  `research`, because the profile resolves the mode before the domain is
  assigned and so `FREEFORM_DOMAIN.default_mode` never applies. Without it there
  is no worktree, no patch-writing instruction, and no `patches_written`.
- **`domain` must be left unset.** `_forward_integrate_source` overwrites
  `provenance` with `specialist:<domain>` when a domain is present, which would
  erase the attribution label.
- **The mandate is sanitized on our side.** `task_description` is interpolated
  into a one-line markdown quote (`> {desc}`) exactly as given. A newline in it
  would leave the quote, so model-authored text could forge a section header in
  the specialist's own prompt. The pump runs it through `flatten_for_prompt`,
  which folds every line separator and defangs code fences and angle brackets,
  then caps the length.

## Attribution

Adopted variants are attributed to a dedicated agent bucket, following the
`warm_replay` precedent — the existing case of a non-LLM external source owning
headline credit. Three closed sets carry it: `_SOURCES` in the optimizations
collector, `AgentBucket` in the breakdown schema, and `_AGENT_BY_ACTION` in the
recorder.

`lever_kind` stays inside its own five-value closed set: the config channel
reports `config`, the patch channel stamps `source_patch`. An unknown lever is
silently reduced to an empty string, so inventing a value there would lose the
attribution rather than extend it.

The point of the bucket is measurement. With it, `session_breakdown.json`
answers how much validated gain the predictor produced next to `default_grid`
and `llm_direct`, under the same KEEP threshold and the same accuracy gate.

One caveat when reading those numbers: the grading anchor is `current_best`, so
every KEEP the chain lands raises the bar for whatever explores afterwards. Part
of a chain's measured contribution is having gone first. That is inherent to the
loop rather than introduced here, but it is worth remembering before concluding
that free exploration underperformed.

## Configuration

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--primatune-endpoint URL` | `HYPERLOOM_PREDICTOR_ENDPOINT` | unset | Service base URL. Unset disables the pump. |
| `--primatune-mode MODE` | `HYPERLOOM_PREDICTOR_MODE` | `shadow` | `off`, `shadow` (predict + log), `active` (enqueue). |
| `--primatune-max-chain N` | `HYPERLOOM_PREDICTOR_MAX_CHAIN` | `3` | Step cap for the greedy chain. |
| `--no-primatune` | — | — | Force `off` regardless of the other two. |
| — | `HYPERLOOM_PREDICTOR_BUDGET_PCT` | `25` | The chain stops proposing once the phase has spent more than this share of its budget. |
| — | `HYPERLOOM_PREDICTOR_TIMEOUT_SEC` | `120` | Per-request timeout. Exceeding it ends the chain. |
| — | `HYPERLOOM_PREDICTOR_PHASE_LABEL` | `EXPLORE` | Value sent as `phase.phase`. See below. |

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
