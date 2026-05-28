# quantization-agent — Architecture Design Specification

## 0. Document Purpose and Background

The goal: *"embed Quark as a sub-agent inside Hyperloom… and let the module
also stand alone to quantize models on its own."*

This agent is a **thin Hyperloom → Quark adapter**:

- The sole entrypoint is the coroutine `quantize_via_prompt(prompt, ...)`,
  which feeds a natural-language prompt to a Claude Agent SDK session.
- The session loads Quark's three skills (`quark-torch-ptq` /
  `quark-torch-result-validator` / `quark-torch-llm-eval`) and the
  agent's own `SKILL.md` (runtime contract).
- Translating natural language into a `quant_plan`, filling defaults,
  and surfacing CRITICAL STOPs are all done by the Quark skills.
- The agent does two things only: answer checkpoints per `SKILL.md`,
  and scan the workspace to enforce the artifact contract.

This document records the design's **structure, contracts, and trade-offs**;
runtime behavior is authoritative in `../SKILL.md`.

---

## 1. How Hyperloom Invokes Quark

```
  Hyperloom side                            Quark side (read-only)
  ┌────────────────────────────┐            ┌────────────────────────────┐
  │  quantization-agent        │  prompt    │                            │
  │  (thin adapter)            │ ─────────▶ │  quark-torch-ptq                 │
  │                            │            │  quark-quantization-       │
  │  · answer per SKILL.md     │ ◀──── ?    │       result-validator     │
  │  · _result_collector        │  answer ─▶ │  quark-torch-llm-eval            │
  │                            │            │                            │
  │                            │ ◀───────── │  artifacts + reports       │
  └────────────────────────────┘            └────────────────────────────┘
```

The three arrows are: ① the initial prompt; ② Quark's three CRITICAL
STOPs plus the warning checkpoints SKILL.md may raise (answered per
SKILL.md: auto → default → escalate to operator); ③ artifacts and
reports written to disk after Quark finishes. Quark stays read-only;
all workflow logic lives in the Quark repo.

---

## 2. Agent Internal Invocation Flow

```
   caller
     │  prompt
     ▼
   quantize_via_prompt
     │
     ▼
   ┌──────────────────────────────────────────────────┐
   │  Claude Agent SDK session (cwd = workspace)      │
   │    loads  : quantization-agent/SKILL.md          │
   │             + quark-torch-ptq / validator / llm-eval   │
   │    runs   : Intake → Plan → Manifest →           │
   │             Execute → Validate → (Eval)          │
   │    answers: checkpoints per SKILL.md             │
   └──────────────────────────────────────────────────┘
     │  artifacts on disk
     ▼
   _result_collector             ──  artifact-presence-as-truth
     │
     ▼
   QuantSkillRunResult
```

Only two layers of deterministic code remain, plus one runtime contract:

| Module | Responsibility |
|---|---|
| `quantize_via_prompt` | Coroutine entrypoint: open SDK session, hand the prompt to the Quark skills, collect results |
| `SKILL.md` (runtime contract) | Loaded by the SDK; dictates checkpoint-answering protocol, `acceptable_eval_gap` default, artifact naming |
| `_result_collector` | Scans workspace artifacts and classifies the run as `success` / `partial` / `failed` |

Verdict does not depend on the SDK exit code; it is derived from on-disk artifacts in two layers:

1. **Presence** — contract artifacts are all there (decides `failed` vs `partial`/`success`).
2. **Content parse** — `validation_report.md` per-step `ok` / `FAIL` / `skipped` (decides whether `partial` escalates to `failed`; full tier rules in §5.4).

---

## 3. Agent Input/Output Specification

The agent exposes a single coroutine entry point — `quantize_via_prompt`. A
minimal call:

```python
from quantization_agent import quantize_via_prompt

result = await quantize_via_prompt(
    "Quantize Qwen/Qwen3-8B with mxfp4; self-attention and kv-cache as fp8; "
    "exclude lm_head; write to /scratch/qwen3-8b-mxfp4",
    workspace="/tmp/wks-xxx",
)
print(result.run_result.status, result.run_result.quantized_model_dir)
```

The input is a **natural-language prompt**. The translation from natural
language to `quant_plan` and the filling of defaults are done entirely
inside Quark `quark-torch-ptq` at the Plan checkpoint (the user confirms or
amends it there).

### 3.1 Input Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `prompt` | str | ✅ | Natural-language quantization request; must contain at least the model, output dir, and a rough scheme |
| `workspace` | str \| Path | ✅ | Working directory for intermediate artifacts; also the Quark workflow's cwd |
| `quark_root` | str \| Path \| None | — | Explicit Quark checkout; otherwise auto-discovered via `$QUARK_ROOT` → `/scratch/kewang/workspace/Quark` |
| `interactive` | bool \| None | — | Whether to allow stdin escalation (CRITICAL STOP answers). `None` = auto-detect a tty |
| `acceptable_eval_gap` | float \| None | — | Acceptable numerical eval gap (default `0.03`). Anything larger triggers a warning checkpoint. Equivalently expressible in the prompt ("accept 5% gap"); priority rules in §5.2 |
| `max_requantize_attempts` | int | — | Retry cap for Ask-class failures (#3 / #6 / #16 / #26 in §A), default `1`. Set to `0` to fail-fast (no retry); raise it to let the agent try more times (useful when inference_optimizer wants to manage a global retry budget). Bounded by the persistent counter `<workspace>/requantize_attempts.txt` |

**Prompt example** (covering common fields — users only need to write
the bits they care about; anything omitted is filled with defaults by the
Quark skill):

```
Quantize /scratch/models/Qwen3-30B-A3B with mxfp4:
- self-attention as fp8
- kv-cache as fp8; exclude lm_head
- calibrate with pileval, 128 samples, seq_len 512
- export HF format; evaluation accepts 5% gap
- write to /scratch/qwen3-30b-mxfp4
```

Quark turns the prompt into a `quant_plan` at the Plan checkpoint and
asks the operator to confirm. See §4 for the scheme vocabulary the
prompt may use.

### 3.2 Return Structure

`QuantSkillRunResult` exposes 3 fields. Anything else (audit detail,
raw scores, traceback strings, artifact paths) stays in logs or
fixed-name files inside `<workspace>` — it does not pollute the return
value.

| Field | Type | Meaning |
|---|---|---|
| `status` | str | `success` / `partial` / `failed` (semantics below) |
| `quantized_model_dir` | Path \| None | Quantized result directory on `success` / `partial`; `None` on `failed` |
| `assessment` | `Assessment` | Primary outcome + the full per-attempt trail (structure below) |

`Assessment` fuses four facts the caller would otherwise have to dig out of
logs — *which outcome class this call landed in, how many attempts it took,
whether a retry rescued an earlier failure, and the one numeric we care
about* — into a single object:

```python
@dataclass
class Assessment:
    final:     str | None              # final primary outcome class ID (row name in Appendix §A); None for a clean one-shot success
    attempts:  list[str | None]        # per-attempt outcome class IDs, in chronological order; len = number of pipeline runs performed
    recovered: bool                    # True iff some earlier attempt failed but a later retry succeeded
                                       # (i.e. attempts has a failure ID before a final None / success tag)
    eval_gap:  float | None = None     # eval-stage relative_gap value; populated only when final ∈ {eval_gap_exceeded, eval_gap_accepted}
```

Field semantics:

- **`final`** — the headline outcome the caller acts on. On `failed` /
  `partial` it is the root cause (`exec_oom`, `checkpoint_aborted`,
  `eval_gap_exceeded`, `upstream_change_required`,
  `unclassified_failure`, …). On `success` "with a story" it is the
  narrative tag (e.g. `eval_gap_accepted`). On a clean success it is
  `None`. Full enumeration in Appendix §A.
- **`attempts`** — the outcome ID of each full pipeline run
  (Intake → Execute → Validate → Eval), oldest first. `len(attempts) - 1`
  is the number of extra retries actually performed (bounded by §3.1
  `max_requantize_attempts`).
- **`recovered`** — whether the diagnose-fix-retry loop did real work for
  this call. Callers typically use it to filter "do I need a human to
  look at this run later?".
- **`eval_gap`** — the only numeric we currently surface in the return
  value, pulled into the struct so callers do not have to parse
  `<workspace>/eval_report.json`. `None` for every outcome other than
  the two eval-gap classes.

Typical shapes:

| Scenario | `final` | `attempts` | `recovered` | `eval_gap` |
|---|---|---|---|---|
| Clean one-shot success | `None` | `[None]` | `False` | `None` |
| First attempt OOM, retry succeeded | `None` | `["exec_oom", None]` | `True` | `None` |
| Two attempts, both OOM | `"exec_oom"` | `["exec_oom", "exec_oom"]` | `False` | `None` |
| Over threshold, non-interactive → partial | `"eval_gap_exceeded"` | `["eval_gap_exceeded"]` | `False` | `0.052` |
| Over threshold, operator accepted | `"eval_gap_accepted"` | `["eval_gap_accepted"]` | `False` | `0.041` |

If the caller needs artifact paths, build them directly from the §3.3
outbound contract's fixed names: `workspace / "validation_report.md"`,
`workspace / "eval_report.json"`, etc. — and use `Path.exists()` to
probe presence. Keeping the path layout knowledge on the caller side
means the agent can reshape its workspace without breaking caller
reporting fields.

**`status` semantics** (full tier definitions in §5.4):

| status | Meaning | How the caller should react |
|---|---|---|
| `success` | All artifacts present + every MUST-validate step ok | Hand `quantized_model_dir` to the downstream step; record `assessment.final` if non-null (e.g. `eval_gap_accepted`) for audit |
| `partial` | Model is loadable but audit / eval artifacts missing or warned | Check `assessment.final` to decide whether to accept |
| `failed` | Model missing or MUST-validate failed | Treat as failure; `assessment.final` is the root cause |

### 3.3 Internal Contracts (with Quark)

The agent maintains **one inbound and one outbound contract** with Quark:

- **Hyperloom → Quark**: send the user prompt as the first SDK session
  message. Quark's three CRITICAL STOP checkpoints are answered per the
  SKILL.md protocol (auto / default / escalate). Optionally write
  `session_context.json` as an audit trace.
- **Quark → Hyperloom**: artifact list is
  `session_context.json` / `model_analysis.json` / `quant_plan.json` /
  `run_manifest.yaml` / `<output_dir>/{config.json, *.safetensors, tokenizer*}` /
  `validation_report.md`; when §5.2 evaluation is enabled, `eval_report.md`
  is added. `_result_collector` scans for these and classifies `status`.

---

## 4. Scheme Vocabulary Reference

The prompt is free text; Quark `quark-torch-ptq` translates it into a valid
`quant_plan` and gets user confirmation at the Plan checkpoint. This
section is a **reference list of the terms commonly used when writing
prompts**. **The authoritative definitions live in Quark
`quark-llm-ptq-workflow/SKILL.md` and `quant_plan.schema.json`.**

### 4.1 Common Scheme Terms

| Term | Meaning | Applicable modules |
|---|---|---|
| `fp8` | 8-bit float (E4M3) | weight / activation / kv_cache |
| `mxfp4` | OCP MX 4-bit float | weight / activation (mostly MoE/MLP) |
| `int8` / `int4` | integer quantization | weight |
| `awq` / `gptq` / `smoothquant` | algorithms (not dtypes) | combine with `int*`/`fp8` |
| `none` / unspecified | module keeps its original dtype (bf16/fp16) | any |

### 4.2 Module Names Commonly Referenced

Prompts can simply say "self-attention", "MoE experts", "plain MLP",
"kv-cache", "lm_head" — Quark maps these to `attention_scheme`,
`layer_overrides[".*\.experts\..*"]`,
`layer_overrides[".*\.mlp\.(?!experts\.|shared_expert).*"]`,
`kv_cache_scheme`, and `exclude_layers` respectively. You do **not** need
to write regexes in the prompt.

### 4.3 Known Constraints

- `kv_cache` is only accepted as `fp8` (or unset) in the currently
  shipped `quark-llm-ptq-workflow` examples and `quantize_quark.py`. Other
  values are rejected at the Plan checkpoint.
- If the prompt names a module/scheme combination Quark does not support,
  the Plan checkpoint will list the reasons and ask for changes.

### 4.4 Defaults

Fields the user doesn't mention in the prompt (calibration samples,
sequence length, export format, evaluation_intent, …) are filled in by
the Quark `quark-torch-ptq` skill inside the Plan checkpoint and shown to the
operator for confirmation.

---

## 5. Post-Quantization Validation and Evaluation

This chapter follows "what we check → how it propagates → how we gate":

- §5.1 **Structural validation** — four byte-level sanity checks
- §5.2 **Accuracy evaluation** — source vs quantized PPL/accuracy comparison
- §5.3 **Failure propagation** — how validation outcomes become `status` and process exits
- §5.4 **Gating policy + recovery** — artifact tiers, recovery matrix, retry cap

### 5.1 Structural Validation

After Execute, SKILL.md instructs the SDK to invoke Quark's
`quark-torch-result-validator` skill in cost order
**4 → 1 → 3 → 2** against `source_model_dir = model_path` and
`quantized_model_dir = output_dir`, writing `<workspace>/validation_report.md`.

| # | Subcommand | What it proves |
|---|---|---|
| 4 | `fuzzy` | Per-pattern `dtype_counts`; mixed dtypes inside one pattern flag layers not quantized as planned |
| 1 | `auxiliary` | tokenizer / generation_config / etc. carried over; no missing / mismatched / extra |
| 3 | `config` | `config.json` deep-equal after stripping `quantization_config` keys |
| 2 | `md5` | Byte-identical MD5 spot-check on `exclude_layers` tensors |

These are **structural / byte-level** checks only — they do not measure
numerical quality (perplexity, accuracy).

### 5.2 Accuracy Evaluation

§5.1 covers only structural sanity; this section is the **accuracy-layer**
check — compare the source model against the quantized model on the same
test set, and surface a confirmation request when the gap exceeds the
threshold.

**Mechanism**

- After Validate, SKILL.md appends a prompt segment that invokes the Quark
  `quark-torch-llm-eval` skill.
- The skill uses **whichever inference backend — vLLM or SGLang — is
  already installed in the current environment** (selected by the skill
  per availability; preference order in `quark-torch-llm-eval/SKILL.md`), spins
  up two offline engines (source + quantized), and runs the same test set
  (default: PPL on wikitext-2 + small gsm8k sample). **The skill performs
  no `pip install`** — missing backends surface as `eval_env_unavailable`
  (see Appendix §A.8) for the caller to handle at the SHOULD-have tier.
- Output lands at `<workspace>/eval_report.md` and `eval_report.json`; the
  latter carries `source_score` / `quantized_score` / `relative_gap`.

**Gap threshold**

`acceptable_eval_gap` is resolved in priority order (highest first):

1. **Python argument** — explicit `quantize_via_prompt(..., acceptable_eval_gap=)`
2. **Prompt declaration** — user writes "accept 5% gap" etc. in the prompt
3. **SKILL.md default** — `0.03` (relative gap 3%)

When the caller passes the Python argument, it wins outright; prompt and SKILL.md are not consulted.

**Decision rules**

| Relative gap | Operator response | status | Notes |
|---|---|---|---|
| `≤ threshold` | — | `success` | passes through |
| `> threshold` | `y` | `success` | `assessment.final=eval_gap_accepted` |
| `> threshold` | `n` | `failed` | explicit rejection is an error |
| `> threshold` | non-interactive | `partial` | `assessment.final=eval_gap_exceeded`; upstream gating decides |

When the gap is exceeded, SKILL.md prints the gap / per-metric / threshold to stderr before asking y/n.

**Eval-failure fallbacks** — when eval cannot even produce a gap (distinct from "gap exceeded"; full enumeration in Appendix §A.8):

| Failure | Trigger | Handling |
|---|---|---|
| `quantized_load_failed` | vLLM/SGLang cannot load the quantized model | escalate to MUST-validate `failed` (downstream cannot serve it either) |
| `eval_oom` | source + quantized exceed VRAM together | retry serial-load; still OOM → `partial` + `assessment.final=eval_oom` |
| `eval_env_unavailable` | neither vLLM nor SGLang installed, or test set missing | `partial` + `assessment.final=eval_skipped`; not a blocker |

**Result-collector behavior**

- `eval_report.md` / `eval_report.json` belong to the SHOULD-have tier (§5.4)
  — absence is a warning only, not a blocker.
- The caller can read the raw `relative_gap` straight off
  `assessment.eval_gap` (the agent already pulls it from
  `<workspace>/eval_report.json`). The agent also encodes the threshold
  decision into `assessment.final`: `eval_gap_exceeded` when over
  threshold in non-interactive mode (status=`partial`), or
  `eval_gap_accepted` when the operator accepted it (status=`success`).

### 5.3 Failure Propagation to inference_optimizer

§5.1 / §5.2 give "pass or fail"; this section gives "once it has failed, how the signal leaves the agent". The outcome reaches inference_optimizer through four layers:

| Layer | Produces | Signal to the next layer |
|---|---|---|
| `quark-torch-result-validator` / `quark-torch-llm-eval` | `validation_report.md` / `eval_report.json` | Per-step `ok` / `FAIL` / `skipped`; `relative_gap` |
| `_result_collector.collect_artifacts` + `_assessment.classify_attempt` + `_assessment.derive_status` | `QuantSkillRunResult` | `status ∈ {success, partial, failed}` + `assessment.final` (validation_report per-step state parsed per §5.4, then collapsed into the primary outcome ID) |
| `quantization_request_handlers.py` (prelude adapter) | payload dict | `success` → `status="ok"`; `partial` → tiered ok / failed per §5.4; `failed` → `status="failed"` |
| `_run_quantization_prelude` | process exit / `args.model` rewrite | `SystemExit(3)` if `status != "ok"`; otherwise continue with `quantized_model_dir` |

The adapter uses §5.4's tier classification to decide ok vs failed:

| QuantRunResult.status | Trigger | adapter status | inference_optimizer behavior |
|---|---|---|---|
| `success` | Artifacts present + every MUST-validate step ok | `ok` | Proceed; `args.model` ← `quantized_model_dir` |
| `partial` (SHOULD/NICE missing) | Model loads, MUST-validate ok, only audit / eval missing | `ok` | Continue; payload carries `quant_status="partial"` |
| `partial` (MUST-validate FAIL/SKIPPED) | md5 / config FAIL, or SKIPPED under STRICT | `failed` | `SystemExit(3)` |
| `failed` | `run_manifest.yaml` or quantized weights missing | `failed` | `SystemExit(3)`; stderr prints `assessment.final` |

Early failures (`workspace_unwritable` / `quark_root_missing` / `sdk_runtime_error` / `checkpoint_aborted` etc. — full enumeration in Appendix §A.1 / §A.9) also produce `status="failed"` with a specific `assessment.final` and `SystemExit(3)`.

> **`SystemExit(3)` ownership**: raised by `_run_quantization_prelude` in `inference_optimizer/cli.py`. Direct `quantize_via_prompt` callers (orchestrator handler, tests, ad-hoc Python drivers) never see a process exit — they get a `QuantSkillRunResult` and dispatch on `status` themselves.

### 5.4 Gating Policy and Failure Recovery

The table below tiers artifacts by what downstream actually needs, with an explicit gating action per tier:

| Tier | Contents | Consequence if missing | Gating |
|---|---|---|---|
| **MUST-have** (load) | `config.json`, `*.safetensors`/`*.bin` (+ `model.safetensors.index.json` when sharded), `tokenizer*` | vLLM cannot load | abort |
| **MUST-validate** (correctness) | validation_report Step 3 config + Step 2 md5 = ok | model loads but results lie | FAIL → abort; SKIPPED → abort by default, `HYPERLOOM_QUANT_STRICT_VALIDATION=0` downgrades to warn |
| **SHOULD-have** (cheaply patchable) | validation_report exists + Step 1/4 ok; `model_analysis.json` / `quant_plan.json` / `run_manifest.yaml` | hard-contract violation but recoverable in seconds | FAIL → run the recovery matrix (patch + re-run validator) |
| **NICE-to-have** | `session_context.json` | harmless | warn |

> **Tier semantics**: tiers classify *which recovery path a violation takes*, not "how important the file is". Step 1 auxiliary enforces a hard contract — **"any non-weight file present in the source must be present in the quantized output"** (covers `special_tokens_map.json` / `generation_config.json` / `chat_template.jinja` etc.). It sits in SHOULD-have not because failure is acceptable, but because the fix path is "`cp` from source + re-run Step 1" (seconds) — not worth aborting an already-completed 30-minute quantization. SKIPPED is the only state that drops to warn.

**Failure handling overview** — after `_result_collector` parses the artifacts, SKILL.md's checkpoint protocol drives the LLM per Appendix §A's classification:

- **Auto-recover** (13 rows in §A.10): LLM patches files / re-runs the validator / fixes the plan inside the sandbox — **no human interruption**.
- **Auto-fail** (10 rows): environment-hard errors or semantic violations — immediate `failed`, no retry (retry is meaningless).
- **Ask** (6 rows): decision points. `interactive=True` asks the operator; `interactive=False` **auto-retries** for 4 of them (#3 / #6 / #16 / #26) using the SKILL.md `Recovery` fix hypothesis, and immediately classifies the other two (#2 lacks user info, #21 is an acceptance decision).
- **Catch-all #30 `unclassified_failure`**: any failure not in rows 1–29 is routed through #30, where the agent diagnoses at runtime (logs + SKILL.md Recovery + stage context) and auto-routes to one of the three categories above. Details in §A.9.

**Diagnose before retry**: every retry-eligible row (Ask-class + #30) must emit a concrete, executable fix hypothesis before quark-torch-ptq is re-invoked, and any patch must stay inside `<workspace>` / agent-controlled state; **no file under `quark_root` may be modified** (Quark is treated as an immutable upstream). Full flow in §A.10.

Per-failure recovery actions and retry triggers are documented in Appendix §A.1–§A.9.

**Re-quantization cap**: controlled by the §3.1 input parameter `max_requantize_attempts` (default `1`); applies only to Ask-class #3/#6/#16/#26, bounded by the persistent counter `<workspace>/requantize_attempts.txt`.

| Mode | Behavior before retry | Counter behavior |
|---|---|---|
| `interactive=False` (CI) | Skip confirmation; classify as `failed` once cap is reached | Counter increments; stops at `max_requantize_attempts` |
| `interactive=True` (operator present) | Print the fix hypothesis on stderr and ask the operator `y/n` before retrying | Same; operator decline → `SystemExit(3)` |

**Caller override**: inference_optimizer can pass `max_requantize_attempts=0` to make the agent classify immediately (useful when the global retry budget has already been spent), or raise it to allow more attempts. **The Auto-recover category (13 rows) is not affected by this parameter** — those are sub-second actions inside the same SDK session and do not count against the Ask-class counter.

Principle: blind retry just reproduces the failure; "retry only when SKILL.md provides a fix hypothesis" binds each retry to an explicit fix. CI can neither burn two 30-minute jobs blindly nor lose recoverable output to transient errors.

---

## 6. Operator-Confirmation Checkpoints (Summary)

The deterministic boundary is pushed down to a minimum; the cost is that
"should we keep going?" decisions get delegated to the operator. This
section enumerates **every point in the design that asks the operator
for y/n, prompt amendment, or exit confirmation** — so ops can plan
staffing and CI can be configured to never block.

| # | Checkpoint | Source | Trigger | Auto-skip | Decline effect |
|---|---|---|---|---|---|
| 1 | Intake CRITICAL STOP | Quark `quark-torch-ptq` | parsed model structure shown for confirmation | prompt says "accept Intake defaults" + `interactive=False` | `failed`, `assessment.final=checkpoint_aborted` |
| 2 | Plan CRITICAL STOP | Quark `quark-torch-ptq` | natural-language → `quant_plan` translation shown for confirmation | prompt says "execute generated plan"; `interactive=False` auto-accepts | same as above |
| 3 | Manifest CRITICAL STOP | Quark `quark-torch-ptq` | run_manifest shown before execute | prompt says "accept default manifest"; `interactive=False` auto-accepts | same as above |
| 4 | Eval gap warning (§5.2) | SKILL.md | relative gap > `acceptable_eval_gap` (default 3%) | raise `acceptable_eval_gap` or declare in prompt | `n` → `failed`; non-interactive → `partial` + `eval_gap_exceeded` |
| 5 | Requantize warning (§A.10) | SKILL.md | Ask-class #3/#6/#16/#26 and catch-all #30 — confirmation **after diagnosis + fix hypothesis**, before re-invoking quark-torch-ptq | `interactive=False` skips confirmation and the counter auto-retries (bounded by §3.1 `max_requantize_attempts`); diagnosis yielding no hypothesis → no retry, immediate `failed` | `n` → `SystemExit(3)`, root cause + fix hypothesis summary on stderr |

**Semantics of `interactive`**: `quantize_via_prompt(..., interactive=...)`
is the **only** knob that decides whether checkpoints 1–5 may escalate
to stdin. `None` (default) auto-detects a tty; `True` forces stdin on;
`False` forces the "non-interactive fallback strategy" written into
SKILL.md (essentially "follow what the prompt or defaults say; otherwise
decline"). Auto-fail failures (§A.10) exit directly and never hit a checkpoint.

**MUST-validate SKIPPED is not a checkpoint**: the §5.4 abort / warn branch is decided by `HYPERLOOM_QUANT_STRICT_VALIDATION` at result-collector time — no runtime y/n. CI sets it once at deployment.

**Recommended CI recipe**: `interactive=False` + the prompt spells out
all Intake/Plan/Manifest choices + `acceptable_eval_gap` either set large
or declared explicitly in the prompt + `HYPERLOOM_QUANT_STRICT_VALIDATION`
left at default. With this, only genuinely fatal errors will halt a run
— nothing will sit waiting for a human at a checkpoint.

---

## 7. Agent Positioning and Differences vs Other Agents

quantization-agent is a **one-shot prelude before the reactor loop**: called once by `inference_optimizer/cli.py::_run_quantization_prelude` when `--quantize "<prompt>"` is set, invokes `quantize_via_prompt`, returns `quantized_model_dir`, writes it back to `args.model`, then exits. It does not participate in any subsequent tick.

```
   User CLI               quantization-agent        Coordinator         reactor loop
   --quantize "<prompt>" ──►  (one-shot)       ──►  (bootstrap)    ──►  re-invoked per tick
                             returns quantized_model_dir                orchestration / kernel
                             rewrites args.model                        critic / robustness
```

Compared with reactor-loop agents:

| Dimension | Reactor-loop agents | quantization-agent |
|---|---|---|
| Examples | `orchestration` / `kernel` / `critic` / `robustness` | this agent |
| Lifecycle | Re-invoked every tick | Called once before the loop boots |
| Output channel | `emit_intent` → PolicyGate → dispatch | Returns a Python `dict` directly to the caller |
| Registered in `AgentRole` | ✅ | ❌ |

---

## 8. Read-Only Guarantees on the Quark Repository

The agent treats the Quark repo as strictly read-only: `quantization-agent/SKILL.md` and the Quark workflow SKILL.md files both forbid edits to `quark/`, `examples/`, `tools/`, `docs/`, `tests/`, `pyproject.toml`, `requirements.txt`. Any custom script (new flag, new model template) is written under `<workspace>/` and referenced from `run_manifest.yaml`, so reproducibility against a clean Quark install is preserved.

---

## 9. References and Related Materials

- `quantization-agent/SKILL.md` — **runtime contract**: checkpoint-answering protocol, Quark skill invocation order, eval gap threshold
- `quantization-agent/__init__.py` — exports the `quantize_via_prompt` coroutine entrypoint
- `quantization_agent/_result_collector.py` — artifact harvest (the deterministic boundary)
- `inference_optimizer/orchestrator/quantization_request_handlers.py` — programmatic dispatch (the `quantize_via_prompt` adapter shim)
- `inference_optimizer/cli.py::_run_quantization_prelude` — `--quantize "<prompt>"` pre-hook
- Quark `quark-torch-ptq/SKILL.md` — PTQ workflow contract this agent drives
- Quark `quark-torch-result-validator/SKILL.md` — the four structural checks invoked in §5.1
- Quark `quark-torch-llm-eval/SKILL.md` — accuracy evaluation skill invoked in §5.2
- `kernel-agent/tools/tracelens_skill_runner.py` — the architectural template

---

## Appendix A. Failure-Handling Matrix (by stage)

This appendix enumerates every failure quantization-agent may surface across the pipeline, with each row's **category** (Auto-recover / Auto-fail / Ask), **specific description**, and **CI / interactive behavior**. Principles:

- **Auto-recover** — LLM self-heals inside the sandbox; no human disturbance.
- **Auto-fail** — unfixable or retry-useless; immediately `failed`.
- **Ask** — decision point. CI auto-retries (only when SKILL.md `Recovery` provides a fix hypothesis and the LLM has written out a concrete fix action); interactive mode asks the user. Retry budget is controlled by the §3.1 input parameter `max_requantize_attempts` (default `1`) and persisted in `<workspace>/requantize_attempts.txt` — exceeding it means immediate `failed`.
- **Catch-all #30** — any failure not in rows 1–29 lands on #30, where the agent diagnoses at runtime and maps it to one of the three categories above.

**Universal constraint**: every retry must be preceded by a diagnostic pass that produces a **concrete** fix hypothesis; **patches may only touch files inside `<workspace>` — never under `quark_root`** (Quark is treated as an immutable upstream so shared checkouts stay clean). Full flow in §A.10.

### A.1 Stage 1 · Pre (preflight)

| # | Failure | Category | Specific description | CI=False | interactive=True |
|---|---------|----------|---------------------|----------|------------------|
| 1 | `quark_root_missing` | Auto-fail | `quark_root` path doesn't exist / unreadable / not a git checkout; probed once at startup | immediate **failed** | immediate **failed** |
| 7 | `quark_skill_unavailable` | Auto-fail | Any of `quark_root/.claude/skills/{quark-torch-ptq,validator,llm-eval}/SKILL.md` missing or registration fails | immediate **failed** | immediate **failed** |
| 8 | `intent_parse_failed` | Auto-recover | LLM cannot produce a valid `hyperloom_quant_intent` from the prompt: missing model_path / target_dtype / output_dir, or type errors | LLM self-corrects ≤2 times; still fails → **failed** | After 2 self-correct attempts, asks user to amend prompt |
| 23 | `workspace_unwritable` | Auto-fail | `workspace` path cannot be mkdir-ed, or exists but `os.access(W_OK)=False`; probed via touch in Pre to avoid a 30-minute quant exposing it later | immediate **failed** (with the specific PermissionError) | immediate **failed** |

### A.2 Stage 2 · Intake (quark-torch-ptq Step 1)

| # | Failure | Category | Specific description | CI=False | interactive=True |
|---|---------|----------|---------------------|----------|------------------|
| 9 | `model_path_unreachable` | Auto-fail | quark-torch-ptq Step 1 tries to load source model but `model_path` is unreachable / unreadable / lacks `config.json` | immediate **failed** | immediate **failed** |
| 10 | `analysis_artifact_invalid_or_missing` | Auto-recover | `model_analysis.json` not produced, or schema-invalid / JSON-corrupt | LLM re-runs Step 1 → continue | same |

### A.3 Stage 3 · Plan (quark-torch-ptq Step 2)

| # | Failure | Category | Specific description | CI=False | interactive=True |
|---|---------|----------|---------------------|----------|------------------|
| 2 | `checkpoint_aborted` | Ask | Any Intake/Plan/Manifest CRITICAL STOP triggered without a "accept default" line in the prompt under non-interactive mode | If prompt declares defaults → auto-accept; else **failed** (missing info, retry pointless) | Ask user to accept / rewrite prompt |
| 11 | `plan_artifact_invalid_or_missing` | Auto-recover | `quant_plan.json` missing or fails schema (scheme not in whitelist, layer_overrides regex invalid, etc.) | LLM re-runs Step 2 → continue | same |

### A.4 Stage 4 · Manifest (quark-torch-ptq Step 3)

| # | Failure | Category | Specific description | CI=False | interactive=True |
|---|---------|----------|---------------------|----------|------------------|
| 12 | `manifest_artifact_invalid_or_missing` | Auto-recover | `run_manifest.yaml` missing, or parsed but lacks `outputs.quantized_model_dir` | LLM re-runs Step 3 → continue | same |

### A.5 Stage 5 · Exec (quark-torch-ptq Step 4a, PTQ compute)

| # | Failure | Category | Specific description | CI=False | interactive=True |
|---|---------|----------|---------------------|----------|------------------|
| 3 | `exec_oom` | Ask | quark-torch-ptq Step 4a OOMs during PTQ; root causes: batch too large / seq_len too long / kv_cache quant flag | LLM reduces batch/seq_len + auto-retry once → success / **failed** | same + confirm params before retry |
| 4 | `exec_model_load_failed` | Auto-fail | Exec-stage source model load fails (distinct from #9: path exists but weights corrupted / dtype unsupported / transformers missing) | immediate **failed** | immediate **failed** |
| 5 | `exec_calibration_data_missing` | Auto-fail | calibration dataset (pileval / wikitext / custom) unreachable or sample count 0 | immediate **failed** | immediate **failed** |

### A.6 Stage 6 · Export (quark-torch-ptq Step 4b, writing to disk)

| # | Failure | Category | Specific description | CI=False | interactive=True |
|---|---------|----------|---------------------|----------|------------------|
| 6 | `export_crashed` | Ask | Step 4b crashes while serializing quantized state_dict → safetensors; common causes: disk full / NFS jitter / temp-file contention / config write conflict | LLM auto-retries once → success / **failed** | same + confirm before retry |

### A.7 Stage 7 · Validate (quark-torch-result-validator)

| # | Failure | Category | Specific description | CI=False | interactive=True |
|---|---------|----------|---------------------|----------|------------------|
| 13 | `validator_self_test_failed` | Auto-fail | `run_validation.py self-test` exits non-zero; Quark validation script broken or import errors | immediate **failed** | immediate **failed** |
| 14 | `must_have_config_missing_or_invalid` | Auto-recover | `<quantized_dir>/config.json` missing or JSON-corrupt, or lacking vLLM-required core fields (`model_type` / `architectures`) | LLM copies from source + restores `quantization_config` → re-run Step 3 → continue | same |
| 15 | `must_have_tokenizer_missing` | Auto-recover | `tokenizer.json` / `tokenizer_config.json` or other required tokenizer files missing | LLM `cp`s the full tokenizer set from source → re-run Step 1 → continue | same |
| 16 | `must_have_weights_missing` | Ask | No `*.safetensors` / `*.bin` under `<quantized_dir>`; Step 4 produced no weights | LLM auto-retries once → success / **failed** | same + confirm before retry |
| 17 | `must_validate_config_mismatch` | Auto-recover | Step 3 `config` FAIL: after stripping `quantization_config`, non-quant fields still differ | LLM fixes fields (whitelist additions / business-field copy-over from source) → re-run Step 3 → continue | same |
| 18 | `must_validate_md5_mismatch` | Auto-fail | Step 2 `md5` FAIL: tensors named in `exclude_layers` show byte-level MD5 diff from source — **promised untouched yet touched** | immediate **failed** (semantic violation) | immediate **failed** |
| 19 | `should_have_aux_missing` | Auto-recover | Step 1 `auxiliary` FAIL: non-weight files present in source but missing in quantized dir (`special_tokens_map.json` / `generation_config.json` / `chat_template.jinja`, etc.) | LLM `cp`s missing files → re-run Step 1 → continue | same |
| 20 | `nice_to_have_skipped` | Auto-recover | PyYAML missing causing manifest-parse degrade; or `session_context.json` missing — artifacts that don't affect downstream | record note → continue success | same |
| 25 | `validation_report_absent` | Auto-recover | `validation_report.md` not produced at all — validator wasn't called by SKILL.md, or the call crashed entirely; distinct from "produced but some step FAILed" | LLM invokes `quark-torch-result-validator` once → continue | same |
| 26 | `fuzzy_check_failed` | Ask | Step 4 `fuzzy` FAIL: same pattern shows mixed dtypes, or safetensors header anomalies; mostly plan errors (mixed precision), rarely file corruption | LLM cross-checks `model_analysis.json` to fix plan + auto-retries once → success / **failed** | same + confirm before retry |
| 27 | `must_validate_skipped` | Auto-recover | Step 2 md5 or Step 3 config SKIPPED due to environment (source unreachable, quant_config has no exclude, etc.) — **distinct from FAIL**: unproven ≠ violated | `HYPERLOOM_QUANT_STRICT_VALIDATION=1` (default) → **failed**; `=0` → partial + warning | same (one-time env var decides) |

### A.8 Stage 8 · Eval (quark-torch-llm-eval)

| # | Failure | Category | Specific description | CI=False | interactive=True |
|---|---------|----------|---------------------|----------|------------------|
| 21 | `eval_gap_exceeded` | Ask | quark-torch-llm-eval finishes with `relative_gap > acceptable_eval_gap` — acceptance decision, not retryable | auto → **partial** + `eval_gap_exceeded` | Ask user to accept partial / re-plan |
| 22 | `eval_env_unavailable` | Auto-recover | Neither vLLM nor SGLang installed, or test dataset unreachable, or no GPU at all; **the skill does NOT auto-install backends** | skip eval → success + note `eval_skipped` | same |
| 28 | `quantized_load_failed` | Auto-fail | Eval-stage vLLM/SGLang fails to load the quantized model (distinct from #4 which loads the *source* model) — downstream serving will fail identically, equivalent to a MUST-validate violation | escalate to MUST-validate **failed** | same |
| 29 | `eval_oom` | Auto-recover | source + quantized dual-engine doesn't fit on the same GPU; distinct from #3: happens during vLLM/SGLang inference, not PTQ operator compute | auto-switch to serial loading; still OOM → **partial** + note `eval_oom` | same |

### A.9 Cross-stage

| # | Failure | Category | Specific description | CI=False | interactive=True |
|---|---------|----------|---------------------|----------|------------------|
| 24 | `sdk_runtime_error` | Auto-fail | Claude Agent SDK session itself raises (rate-limit / network / context overflow / authentication); unrelated to Quark or validation logic | immediate **failed** with sdk traceback summary | immediate **failed** |
| 30 | `unclassified_failure` | Auto-recover\* | Any failure not matching rows 1–29: e.g. new error class from a Quark upgrade, novel SKILL.md output, unfamiliar traceback. The agent **must** diagnose first (read stderr / stage context / corresponding SKILL.md `Recovery` table) and then **auto-decide** which known category applies: patch in workspace → Auto-recover; retryable with a fix hypothesis → Ask (bounded by §3.1 `max_requantize_attempts`); otherwise Auto-fail. **Editing any file under `quark_root` is forbidden** — Quark is treated as immutable upstream. | Diagnose → if patchable, patch workspace + retry; otherwise **failed** + `assessment.final=unclassified_failure` (diagnosis summary attached) | Same; operator may override the auto-selected strategy via stderr prompt |

### A.10 Distribution and retry mechanism

> **Orthogonality**: the "tier" in §5.4 (MUST-have / MUST-validate / SHOULD-have / NICE-to-have) describes **the consequence of violating a contract**; the "category" in this appendix (Auto-recover / Auto-fail / Ask) describes **the handling behavior**. E.g. #14 `must_have_config_missing` is tier = MUST-have (can't load without it) and category = Auto-recover (LLM can rebuild it); #18 `must_validate_md5_mismatch` is tier = MUST-validate (semantic violation) and category = Auto-fail (unfixable).

**Diagnose → Fix → Retry flow**: every retry-eligible row (Ask-class plus catch-all #30) must run an explicit diagnostic pass before invoking quark-torch-ptq again:

1. **Diagnose**: read the failing stage's stderr, artifacts, and the corresponding SKILL.md `Recovery` column to locate the root cause.
2. **Form a fix hypothesis**: it must be **concrete and executable** (e.g. `batch_size 32→16`, `exclude lm_head`, `bump calibration samples to 256`, `amend prompt to "execute generated plan"`). Vague "just try again" doesn't count.
3. **Apply the patch**: limit the change to files under `<workspace>` or the agent's own controllable state (prompt, `quant_plan.json`, env vars). **No file under `quark_root` may be modified** — Quark is treated as an immutable upstream; any failure whose only fix is to patch Quark is reclassified as Auto-fail with `assessment.final=upstream_change_required`.
4. **Retry**: re-run the failed stage with the patched inputs; counter increments by 1.

If diagnosis cannot produce a fix hypothesis, the row is classified `failed` immediately — no blind retry.

**Category distribution** (30 rows total):

| Category | Count | Row numbers |
|----------|-------|-------------|
| Auto-recover | 13 | 8, 10, 11, 12, 14, 15, 17, 19, 20, 22, 25, 27, 29 |
| Auto-fail | 10 | 1, 4, 5, 7, 9, 13, 18, 23, 24, 28 |
| Ask | 6 | 2, 3, 6, 16, 21, 26 |
| Auto-recover\* (catch-all, runtime-classified) | 1 | 30 |

**Retry mechanism**: Ask-class rows (and catch-all #30 when routed to Ask) that auto-retry are bounded by the §3.1 input parameter `max_requantize_attempts` (default `1`), persisted via `<workspace>/requantize_attempts.txt`. Callers can override (e.g. `max_requantize_attempts=0` for fail-fast). Auto-recover rows (13 total) are *not* counted against this cap — they run inside the same SDK session and complete in seconds.

**CI retry-triggering rows** (Ask-class #3 / #6 / #16 / #26 only):

| # | Source of fix hypothesis |
|---|--------------------------|
| 3 `exec_oom` | quark-torch-ptq SKILL.md Recovery: reduce batch / seq_len |
| 6 `export_crashed` | Empirical: clear tmp + retry with same params (usually transient) |
| 16 `must_have_weights_missing` | Union of #3 / #6 (PTQ produced no weights) |
| 26 `fuzzy_check_failed` | LLM corrects `layer_overrides` in plan against `model_analysis.json` |

**Ask-class rows that do NOT auto-retry**:

- **#2** `checkpoint_aborted` — missing info is *user decision data*; a retry still lacks it.
- **#21** `eval_gap_exceeded` — acceptance call, not retryable; same plan yields same gap.
