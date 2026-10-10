---
myst:
    html_meta:
        "description": "Reference for the PrimaTune predictor integration: an external first-pass tuner Hyperloom asks for launch-configuration proposals in FRAMEWORK_AGENT. Covers the off, shadow and active modes, the HTTP contract, how answers reach the untested-proposal queue, attribution and the source-change channel."
        "keywords": "Hyperloom, PrimaTune, predictor, first-pass tuner, FRAMEWORK_AGENT, untested proposals, explore, vLLM, SGLang, AMD GPU, ROCm, inference optimization"
---

# PrimaTune predictor

PrimaTune is a fine-tuned model that reads a session's state and proposes launch
configurations: server flags, environment variables, and occasionally a source
change. Hyperloom can ask a PrimaTune service for proposals at each FRAMEWORK_AGENT
decision point. The answers join the untested-proposal queue beside the
specialists' proposals and are measured and graded the same way, against the same
KEEP threshold.

The integration is off unless an endpoint is configured. A session that sets none
behaves exactly as before.

## Turning it on

```bash
hyperloom optimize ... \
  --primatune-endpoint http://<predictor-host>:8973 \
  --primatune-mode active
```

| Flag | Environment variable | Default | Meaning |
|---|---|---|---|
| `--primatune-endpoint` | `HYPERLOOM_PREDICTOR_ENDPOINT` | Unset (off) | Base URL of the service. The client posts to `<endpoint>/v1/predict`. |
| `--primatune-mode` | `HYPERLOOM_PREDICTOR_MODE` | `shadow` | `off`, `shadow` or `active`; see below. |
| none | `HYPERLOOM_PREDICTOR_TIMEOUT_SEC` | `900` | Per-request timeout, at least 1 second. |

The CLI exports the flags into its own process environment, and the predictor reads
that environment on every tick. A flag applies to the run it is passed to: a
`--resume-from` relaunch needs the flags again, unless the variables are exported in
the shell, which an absent flag leaves in place. An invalid environment value logs a
warning and falls back to its default.

### Modes

| Mode | Asks the service | Queues proposals | Costs benchmark time |
|---|---|---|---|
| `off` | No | No | No |
| `shadow` | Yes | No; the answer is logged | No |
| `active` | Yes | Yes | Yes, through the ordinary explore lane |

An endpoint alone selects `shadow`, so pointing a session at a service never spends
benchmark time until `active` is chosen. Use `shadow` to check that a service answers
and that its proposals look sane for a model before letting them be measured.

## How a session uses it

**Decision points.** The predictor is asked once per decision point, keyed
`c<macro_cycle>-s<stack_depth>-r<roofline_count>`. A KEEP, a new macro-cycle and a
new roofline each change what the answer is conditioned on. A key counts as asked as
soon as its request goes out, so it is never asked twice, not even after a failure.
Asked keys persist in `state.json` as `predictor_asked_keys`.

**When it asks.** Only in an open FRAMEWORK_AGENT phase, only for `vllm` and `sglang`
(the frameworks the service has flag catalogues for), and only while the phase can
still bench one variant, about ten minutes.

**Off the tick loop.** The request runs in a worker thread. The tick that finds it
finished files the answer, so a slow service never stalls the Coordinator. One
request is in flight at a time. An answer that arrives after the macro-cycle has moved
on is filed under the cycle it was asked in, so it never reaches the new cycle's
queue.

**Filing an answer.** In `active` mode an answer becomes one round in
`specialist_rounds`, with `domain: "primatune"` and `round_id` equal to the decision
point. Each proposal becomes a queue row after four steps:

1. It is reduced to what it changes on top of the current champion. An exact
   `(flag, value)` echo of the champion is dropped, while the same flag with another
   value is kept, since that is the change.
2. Environment variables of the other serving stack are dropped: `SGLANG_*` on vLLM
   and `VLLM_*` on SGLang.
3. A delta this session has already benched, or already has on the queue, is
   dropped. A proposal that reduces to nothing is dropped too.
4. Rows are ordered by how many of the service's samples proposed them. Each knob
   family takes one slot, and at most six rows are queued per answer.

A row's `reason` is `PrimaTune <votes>/<samples>: <rationale>` when the service
explains its proposal, and `predictor: <knobs>` when it does not; a row keeps
600 characters of the rationale. The untested-proposal block shows a predictor
row's reason whole, against 80 characters for a specialist row: a specialist's
findings reach orchestration in their own section, while the reason is
everything a predictor row says.

The round carries `priority: 1`. Specialist rounds carry none, so predictor rows sit
at the head of the untested-proposal queue, which the Coordinator benches from
whenever no explore task is queued or running. Orchestration usually keeps the
benchmark lane busy with its own grids, so that rarely happens; the block's
header therefore makes predictor rows an exception to orchestration's rule of
dispatching only variants the queue does not list, and asks it to put the rows it
judges worth a slot into its next grid verbatim, name and `provenance: primatune`
included, and to say why when it skips one. Which rows run stays orchestration's
choice. They are graded like any other variant.

## Attribution

Predictor rows carry `provenance: "primatune"`. The provenance travels with the
explore variant made from a row and onto that variant's attempt, and the session
breakdown records `primatune` as the variant's producer. A grid that mixes predictor
rows with specialist rows is recorded as the orchestration agent's, as for any mixed
grid; each variant keeps its own provenance on its attempt.

## Source-change channel

When an answer describes a source edit, the round also carries `mandate_id` and
`mandate`, the predictor's prose, flattened to one line and capped at 4000
characters. The untested-proposal block shows the newest mandate of the macro-cycle
that no specialist has taken, with the dispatch to use:

```text
delegate{action_name='specialist', params={scope:'freeform', mode:'patch',
         primatune_mandate_id:'<id>', task_description:'<one line>'}}
```

The router replaces `task_description` with the mandate's own text. It also sets
`scope: freeform`, `mode: patch`, `provenance: primatune` and
`lever_kind: source_patch`, and removes `domain`, so the patch keeps the predictor's
provenance on its way to `integrate_patch`. Once the specialist task is created, the
mandate is marked consumed and is no longer offered. An unknown id is left alone: the
dispatch proceeds as an ordinary specialist.

## HTTP contract

### Request

`POST <endpoint>/v1/predict`, JSON, schema `hyperloom.predictor_request.v1`. The
body carries exactly the fields the service reads, in Hyperloom's own spelling. A
block the session cannot fill completely is omitted rather than half-filled.

| Section | Fields |
|---|---|
| top level | `schema`, `session_id` |
| `identification` | `model_name`, `model_class`, `gpu_type`, `framework`, `framework_version`, `precision`, `tp`, `ep`, and `model_info` with `model_type`, `attention_type`, `num_hidden_layers`, `num_experts`, `hidden_size`, `head_dim` |
| `workload` | `isl`, `osl`, `conc`, `max_model_len` |
| `phase` | `phase` (always `EXPLORE`, the label the predictor was trained on for the configuration arm), `phase_reason`, `phase_elapsed_seconds` |
| `performance` | `baseline_tput`, `current_best_tput`, `cumulative_gain_validated`, `keep_threshold_pct`, and `optimization_stack` with one row per KEEP: `candidate_extra_server_args`, `extra_envs`, `tput` |
| `evidence` | `profile_available`, `profile_age_sec`, `roofline`, `window`, `operators`, `hot_kernels` |

The evidence blocks come from these sources:

- **`roofline`**, from the latest roofline snapshot: the memory and compute
  ceilings, the bound kind, achieved throughput, the gap to the roofline, HBM
  bandwidth and peak TFLOPS when the perf model recorded them, and the operator
  counts.
- **`window`** (total GPU time, busy, idle, and exposed communication) and
  **`operators`** (the per-category GPU-time split, the top bottleneck, and the
  attribution coverage). They are parsed from `analysis.md` when it has the
  layout the bypass route renders. Otherwise, as on the TraceLens route, where a
  model writes the report, the window comes from TraceLens's `analysis.json` and
  the split from the `summary.json` beside it, with busy taken as everything but
  idle, as the bypass report counts it.
- **`hot_kernels`**: up to eight rows of `hot_kernels_top15`. Each row adds the
  operand `args`, the `call_count` and `time_us` from the report's P-item tables
  (the count and time from `summary.json` when the report has none), and the
  `source_file`, `source_line` and `source_function` that TraceLens resolved in
  `kernel_source_resolution.json`.

A bound label of `unknown` is sent as absent, because the service renders labels as
`<label>-bound`.

### Response

| Field | Meaning |
|---|---|
| `schema` | `primatune.predictor_response.*`; any other value is treated as no answer. |
| `parsed` | `false` when the service declined to answer. |
| `actions` | Every distinct proposal: `server_args` (flag to value, `true` for a bare flag), `envs`, `source_change`, and optionally `rationale`, the proposal's mechanism and what it rests on. A service that does not sample may send a single `action` instead. |
| `meta.candidates` | The raw samples. They are used only to count votes. |
| `meta.samples`, `meta.prompt_chars` | Recorded on the queue round. |

A transport error, an HTTP error status, a malformed body or an unexpected schema
is logged and treated as no answer. The decision point still counts as asked.

## Reading a session

| Where | What to look for |
|---|---|
| Log | `predictor: asked <endpoint> at decision point <key>`, then `predictor: queued N row(s) at <key> after <ms>ms`; in shadow mode, `predictor (shadow): key=...` with every action. |
| `state.json` | `predictor_asked_keys`, and `specialist_rounds` entries with `domain: "primatune"`, each with its `proposal_set` (`votes`, `samples`, `reason`) and `predict_meta` (latency, samples, prompt size). |
| Attempts and breakdown | `provenance: "primatune"` on the variants measured from predictor rows, and producer `primatune` on their framework-event proposals. |
