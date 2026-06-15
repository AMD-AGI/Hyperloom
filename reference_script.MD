# Plan B (revised) — `--reference-script` in, `current_setting.sh` out

## Goal

Two halves of one recipe loop:

1. **Input** — operator passes a reference launch recipe (e.g. InferenceX
   `benchmarks/single_node/fixed_seq_len/minimaxm3_fp8_mi300x.sh`) at launch.
   The optimizer lifts the `vllm serve` flags + whitelisted `export`s and uses
   them as the **lowest-priority base** server args / envs for the first
   `baseline` and every later restart — so exotic architectures (MiniMax-M3 MSA,
   etc.) boot instead of dead-ending at `baseline_failed`.
2. **Output** — after every KEEP, hyperloom re-renders `current_setting.sh`, a
   **read-only** recipe in the same format the parser accepts. It records the
   current best route for audit / manual reproduction, and can be fed straight
   back in via `--reference-script` on the next run (the loop closes).

Priority order (low → high), unchanged from the original intent:
```
reference-script base  <  YAML template base  <  per-task extra_server_args / extra_envs
```

## What changed vs the first draft (and why)

The first draft was directionally right but had four problems the codebase made
clear on inspection. This revision fixes all of them:

| # | First draft | Problem found | Revision |
|---|---|---|---|
| R1 | "skip tokens containing `$`" | leaves an **orphan flag**: `--served-model-name $NAME` → dangling `--served-model-name` pollutes args / breaks vLLM | parse as **(flag, value) pairs**; dropping a `$`-value drops its flag too |
| R2 | recipe applied to any model | recipe is model+GPU-specific (`--language-model-only`, `block-size 128` are M3-isms); wrong model gets wrong flags | **model-gate**: record the recipe's model in manifest + SharedState; skip injection (WARN) when current `MODEL_PATH` differs |
| R3 | process-env `HYPERLOOM_REFERENCE_*`, "ignored on --resume" | resume rebuilds from `state.json`, NOT env → restart loses the reference → `baseline_failed` again, contradicting the Goal | store reference in **SharedState fact-layer** → `asdict`-serialized into `state.json`, auto-restored on resume; no process env at all |
| R4 | env hand-off so callers don't thread a param | the module header (`_workload_envs.py:204`) documents the team is **migrating away from process-wide env** toward per-task params to kill races; new global env is a regression | reference flows as **per-task `params`** (same channel as `extra_server_args`), and `materialize_config_with_envs` gains two explicit kwargs |

Plus R5 (anchors corrected — the first draft's line numbers were stale) and R6
(extra gaps, below).

---

## Architecture decisions (confirmed)

- **`current_setting.sh` is a read-only derived artifact.** It is rendered FROM
  `state.current_best` / `state.optimization_stack`; it is never read back to
  restore state. Resume always restores from `state.json`. This means zero
  drift — the same class of bug `_workload_envs.py`'s header warns about
  (baseline vs grid diverging) cannot recur, because there is exactly one
  source of truth.
- **Reference lives in SharedState, dispatched per-task.** No
  `HYPERLOOM_REFERENCE_*` env. `materialize_config_with_envs` learns the
  reference layer via explicit kwargs, fed from each task's `params`.
- **`current_setting.sh` is a valid recipe.** Output is `export K=V` lines +
  a `vllm serve <flags>` line in the exact shape the parser consumes, so
  `--reference-script out/current_setting.sh` round-trips on the next run.

---

## Necessary changes

### 1. New module: `inference_optimizer/reference_script.py`

Dependency-light parser + renderer. Public API:

```python
def parse_reference_script(
    source: str,            # local path or http(s) URL
    *,
    framework: str,         # "vllm" | "sglang" | "atom"
) -> ReferenceRecipe: ...   # (server_args: str, envs: dict[str,str], model: str|None)

def render_reference_script(
    *,
    framework: str,
    server_args: str,
    envs: dict[str, str],
    model: str | None,
) -> str: ...               # the current_setting.sh text
```

`ReferenceRecipe` is a small frozen dataclass `(server_args, envs, model)`.

**Parser behavior** (deliberately narrow):
- Resolve `source`: local read, or `http(s)` fetch. **Reuse the existing
  fetcher** `baseline_comparison/inferencex_client.py:131 _fetch_raw` (TLS
  context + timeout already handled) rather than a bare `urllib` call. Fail
  soft → return an empty recipe `("", {}, None)` with a WARN if unreachable.
  Never abort launch.
- Find the framework entrypoint line(s), honoring `\` line-continuations:
  vllm `vllm serve ...`; sglang `python -m sglang.launch_server`; atom entry.
- **Pair-aware flag extraction (R1).** Tokenize the entrypoint command, then
  walk it as (flag, value) pairs:
  - `--flag value`: if `value` contains `$`, drop **both** tokens. Else keep.
  - `--flag=value`: if RHS contains `$`, drop the whole token. Else keep.
  - bare `--flag` (store-true, next token is another `--flag` or EOL): keep.
  - Always drop the positional model path, `--port`, `--result-*`, and any
    token still carrying an unresolved `$`.
  Join survivors into one `server_args` string.
- **Capture the model (R2):** record the positional model arg (or
  `--model`/`--served-model-name`) basename even though it's dropped from
  `server_args`, so the caller can model-gate. If it contains `$`, model = None
  (ungated; WARN).
- **Whitelisted exports** — `export KEY=VALUE` whose VALUE has no `$`. Start
  with an **explicit enumerated list, not `VLLM_*` glob (R6)**:
  `VLLM_USE_BREAKABLE_CUDAGRAPH`, `VLLM_USE_TRITON_FLASH_ATTN`,
  `VLLM_FP8_PADDING`, `VLLM_ROCM_USE_AITER`, `SGLANG_*` perf flags by explicit
  name, `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS`. A bare `VLLM_*` glob would
  drag in `VLLM_CACHE_ROOT`-style path vars that won't exist in our sandbox.
  Everything else ignored.
- Never emit secrets / paths.

Rationale unchanged: the shell is **not executed**; we only lift static,
fully-resolved flags. Anything dynamic (TP/CONC/ISL/OSL/model/port) is skipped —
the optimizer's normal env seeding owns those.

### 2. CLI flag: `inference_optimizer/cli.py`

- **Add argument** to the `optimize` subparser, next to `--isl/--osl`
  (verified anchor: `cli.py:4921-4923`):
  ```
  --reference-script PATH_OR_URL
      Reference launch recipe (.sh). Its serve flags + whitelisted exports
      seed the baseline server args at lowest priority (EXPLORE can override).
      Model-gated: ignored if the run's model differs from the recipe's.
  ```
- **Parse + stash into SharedState (R3/R4)**, on a fresh (non-resume) launch,
  after `framework` is resolved (`cli.py:4152`) and `PRECISION` is set
  (verified anchors `cli.py:4009` / `4211`; the first draft's
  `_mirror_workload_flags_to_env` / line 3847 **does not exist**):
  ```python
  if getattr(args, "reference_script", None):
      from .reference_script import parse_reference_script
      recipe = parse_reference_script(args.reference_script, framework=framework)
      state.reference_server_args = recipe.server_args
      state.reference_envs = recipe.envs
      state.reference_model = recipe.model
      state.reference_source = args.reference_script
  ```
  (These are new SharedState fields — see §3.) Persisted by the existing
  `state.save(...)`; **restored on resume for free** via `asdict`/`from_dict`.
- **Record provenance** in `manifest.json` via the existing
  `write_manifest(session_dir, args=args)` (verified anchor `cli.py:4237`,
  `manifest.py:430`): add `reference_script`, `reference_server_args`,
  `reference_envs`, `reference_model`. Audit-only.

### 3. SharedState fields: `inference_optimizer/orchestrator/shared_state.py`

Add four fact-layer fields near `current_best` (`shared_state.py:376`):
```python
reference_server_args: str = ""
reference_envs: dict[str, str] = field(default_factory=dict)
reference_model: str = ""
reference_source: str = ""
```
- Serialized automatically (`to_dict` → `asdict`, `save` at `:912`).
- Restored automatically (`from_dict` at `:681`).
- **Add `reference_server_args` + `reference_envs` to the `fact_layer_keys`
  integrity tuple (`shared_state.py:770`)** so a resume that silently drops
  them trips Inv-10.1 instead of quietly losing the reference (this is exactly
  the R3 failure mode, now guarded).

### 4. Per-task dispatch: `inference_optimizer/orchestrator/coordinator.py`

When building `params` for a `baseline` task (and any restart that rebuilds
baseline args), inject the reference as the lowest layer — **model-gated (R2)**:
```python
ref_model = (state.reference_model or "").strip().lower()
run_model = Path(str(resolved_model or "")).name.lower()
if state.reference_server_args and (not ref_model or ref_model in run_model):
    params["reference_server_args"] = state.reference_server_args
    params["reference_envs"] = dict(state.reference_envs or {})
elif state.reference_server_args and ref_model and ref_model not in run_model:
    log.warning(
        "reference-script recipe is for model %r but run model is %r; "
        "skipping reference injection (flags may not apply).",
        ref_model, run_model,
    )
```
This rides the same per-task `params` channel as the existing
`params["base_extra_args"]` injection (`coordinator.py:3174/3186`).

### 5. Consume in `materialize_config_with_envs`
File: `inference_optimizer/orchestrator/action_executors/_workload_envs.py`.

Add two kwargs:
```python
def materialize_config_with_envs(
    ..., reference_server_args: str = "", reference_envs: dict[str, Any] | None = None,
):
```
Insert the **reference base as the lowest-priority layer**, BEFORE the existing
`extra_server_args` merge (verified anchor `:405-415`) and BEFORE the per-model
workarounds (`:418+`) so those `setdefault`/`"flag" not in existing` guards see
the reference flags and don't double them:
```python
# --- reference-script base (lowest priority) ---
ref_args = (reference_server_args or "").strip()
if ref_args:
    from ._grid_runner import merge_server_args
    framework_env = server_args_env_name(bench.get("framework"))
    existing = str(envs.get(framework_env, "")).strip()
    envs[framework_env] = merge_server_args(ref_args, existing) if existing else ref_args
for k, v in (reference_envs or {}).items():
    envs.setdefault(str(k), str(v))   # never clobber YAML/CLI
```
The existing `extra_server_args` merge then runs after (last-wins), and the
final `dedup_vllm_server_args` (verified anchor `:559-566`) collapses dup
single-value flags so a variant override survives. **No dedup change needed.**

Baseline's call site (`baseline.py:619`) passes the new kwargs from `params`:
```python
reference_server_args=read_extra_server_args(params, key="reference_server_args"),
reference_envs=dict(params.get("reference_envs") or {}),
```
(or a small explicit `params.get` — `read_extra_server_args` lives at
`compat/payload_aliases.py:82`).

### 6. Render `current_setting.sh` on KEEP (the output half)

Single choke point: **`SharedState.save()` (`shared_state.py:912`)** is called
after every `current_best`/`optimization_stack` mutation (verified: writers at
`coordinator.py:2522`, `:3064`, and the canonical `_lift_to_current_best`).
Rather than hook every writer, render once in `save()` as a best-effort
derived artifact:
```python
# after os.replace(tmp, path), best-effort, never blocks save
try:
    if self.current_best:
        from ..reference_script import render_reference_script
        text = render_reference_script(
            framework=os.environ.get("FRAMEWORK", "sglang"),
            server_args=str(self.current_best.get("extra_server_args") or ""),
            envs=dict(self.current_best.get("extra_envs") or {}),
            model=self.reference_model or os.environ.get("MODEL_PATH"),
        )
        (Path(session_dir) / "current_setting.sh").write_text(text, encoding="utf-8")
except Exception:  # noqa: BLE001 — derived artifact, never fatal
    pass
```
- Read-only: nothing ever parses it back. Resume restores from `state.json`.
- Same format as the parser input → `--reference-script <sd>/current_setting.sh`
  round-trips next run.

### 6b. Multi-node per-round restart (Gap closed post-impl)

Single-node bakes the reference base into the materialized YAML, but the
multi-node path restarts the server every round via
``restart_server_for_round`` (`baseline.py` `_run_single_benchmark`), which does
NOT read the YAML. Without threading the reference there, exotic-arch models
re-fail on every MN round. Fix: resolve the model-gated reference base in
``_run_single_benchmark`` too (prefer ``params``, else ``_resolve_reference_base``),
merge it UNDER the per-task args (lowest priority, last-wins) and pass
``extra_env=reference_envs``.

### 6c. Single model-gate (Gap closed post-impl)

The baseline executor and discovery must use the SAME model comparison or they
drift. Exposed ``models_compatible(reference_model, run_model)`` in
`reference_script.py` (reuses the normalized, version-aware ``_model_tier``);
``_resolve_reference_base`` calls it instead of a naive substring check, so
``minimaxm2.5`` vs ``minimaxm3`` is blocked consistently in both places.

### 7. Auto-discovery: find a matching InferenceX single-node recipe

When **no `--reference-script` is given**, try to discover one from the
InferenceX checkout. This is a **tiered** capability, not "find-and-use" —
because the model→script mapping is fuzzy and a near-name mismatch
(`minimaxm2.5` vs `minimaxm3`, `kimik2` vs `kimik2.5`) can turn a working
baseline into `baseline_failed` if model-specific flags get applied wrongly.

**Reality of the tree** (verified on disk, e.g.
`/wekafs/zgong/InferenceX/benchmarks/single_node/`):
- Filenames: `{model}_{precision}_{gpu}[_{framework}][_{variant}].sh`
  (e.g. `dsr1_fp8_mi300x.sh`, `minimaxm2.5_fp8_mi300x.sh`,
  `gptoss_fp4_mi300x.sh`, `glm5_fp8_mi355x_atom.sh`).
- **The script does NOT hardcode the model** — `MODEL` is an
  `check_env_vars`-required external var (`hf download "$MODEL"`,
  `--model-path=$MODEL`). So content-matching the model is impossible; the
  **only key is the filename's leading segment**, which is a human-chosen
  short alias (`dsr1`/`gptoss`/`minimaxm2.5`), NOT the HF repo id / wekafs path
  that hyperloom carries in `MODEL_PATH`. No alias map exists today.

**New function** in `reference_script.py`:
```python
def discover_reference_script(
    inferencex_path: str, *, model_path: str, precision: str,
    gpu_type: str, framework: str,
) -> tuple[str | None, str]:   # (path_or_None, match_tier)
```
- glob `<inferencex>/benchmarks/single_node/*.sh` (and `fixed_seq_len/`),
  parse each filename into `(model_seg, precision, gpu, framework?, variant?)`.
- Filter to `precision == PRECISION` and `gpu == GPU_TYPE` (already
  canonicalized mi325x→mi300x). framework: prefer an exact `_{framework}`
  suffix; a plain (suffix-less) sglang script also matches sglang.
- **Tier the model match** (normalize both sides: lowercase, strip `.`/`-`,
  take the leading alnum run of `Path(model_path).name`):
  - **`exact`** — normalized model_seg `==` normalized model alias.
  - **`fuzzy`** — one is a prefix/substring of the other (catches
    `kimik2.5` vs a `kimi-k2.5-instruct` path) BUT differing trailing
    version digits demote to `none` (so `minimaxm2.5` ≠ `minimaxm3`).
  - **`none`** — no usable match.
- **Action by tier (the safety gate):**
  - `exact` → auto-seed `state.reference_*` (lowest priority; EXPLORE still
    overrides). Log the source + tier.
  - `fuzzy` → **do NOT auto-apply.** Record the candidate path into
    `research_hints` / log so the operator can confirm via
    `--reference-script`. (Mirrors how research-scout surfaces findings into
    prompts, `coordinator.py:5302`.)
  - `none` → nothing (see §8, 0-degrade).
- **CLI always wins**: only run discovery when `args.reference_script` is unset.
- **Fail-soft**: any glob / IO / parse error → return `(None, "none")`, never
  raise, never block launch.

Wiring: call from `cli.py` right after the explicit-flag parse (§2), still on
the non-resume path, before `state.save(...)`.

> Follow-up (separate PR, still in scope conceptually): let research-scout's
> coordinator harvest (`coordinator.py:4452+`) promote a `fuzzy`/scout-found
> recipe to `state.reference_*` after its own confidence check. Kept out of
> this PR to bound scope.

### 8. 0-fallback / 0-degrade guarantee (no reference, none found)

When no recipe is supplied AND discovery returns `none`, the run must be
**byte-for-byte identical to today's behavior**. This falls out of the design
because reference is a pure, lowest-priority, opt-in additive layer:

- `state.reference_server_args` stays `""`, `state.reference_envs` stays `{}`.
- The `if ref_args:` block in `materialize_config_with_envs` (§5) is skipped
  entirely → framework env, per-model workarounds (`_workload_envs.py:418+`),
  context-length / watchdog / MoE injections, and dedup all run unchanged.
- baseline still resolves `default_baseline_config()` + the generic Magpie
  `{framework}_{gpu}.sh` (`_workload_envs.py:115`) exactly as now.
- `current_setting.sh` render in `save()` (§6) is `try/except`-wrapped
  best-effort; a render failure never affects `state.json` persistence.

**Two disciplines that lock this in (enforced by tests, see below):**
1. **Empty-reference identity:** `materialize_config_with_envs(...)` with
   `reference_server_args=""` / `reference_envs=None` produces output
   field-for-field equal to a call that omits the kwargs entirely.
2. **Discovery is total-fail-soft:** every IO/glob/parse path in
   `discover_reference_script` is guarded; a broken/missing
   `$INFERENCEX_PATH` yields `(None, "none")`, not an exception.

---

## Tests

1. `tests/test_reference_script.py` (new):
   - parse the M3 recipe text → `--block-size 128`,
     `--attention-backend TRITON_ATTN`, `--no-enable-prefix-caching`,
     `--language-model-only` in `server_args`; `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
     in `envs`; `--port`, `$MAX_MODEL_LEN`, model path dropped.
   - **(R1) orphan-flag:** `--served-model-name $NAME --block-size 128` →
     `server_args` has `--block-size 128` and **no** `--served-model-name`.
   - **(R1) `--flag=$VAR`** form dropped whole.
   - line-continuation `\` recipe parses identically to single-line.
   - **(R6) whitelist:** `export VLLM_CACHE_ROOT=/x` is NOT carried;
     `export VLLM_USE_BREAKABLE_CUDAGRAPH=0` is.
   - unreachable URL / missing file → empty recipe, no raise.
   - **round-trip:** `render_reference_script(parse(...))` re-parses to the
     same `server_args`/`envs`.
2. `tests/test_baseline_param_overrides.py` (extend):
   - `materialize_config_with_envs(reference_server_args="--block-size 128
     --attention-backend TRITON_ATTN", extra_server_args="--attention-backend
     ROCM_FLASH")` → final `EXTRA_VLLM_ARGS` has `--block-size 128` once and
     `--attention-backend ROCM_FLASH` once (override wins).
   - `reference_envs` does not clobber a YAML/CLI-set env (`setdefault`).
3. `tests/test_shared_state*.py` (extend or new):
   - **(R3) resume:** save a state with `reference_server_args` set, reload via
     `from_dict`, assert the field survives.
   - **(R2) model-gate** logic (unit on the coordinator helper, or a thin
     extracted function): ref_model mismatch → no `params["reference_*"]`.
4. discovery + 0-degrade (`tests/test_reference_script.py`, same file):
   - **(§7 tiers)** with a fixture `single_node/` dir: `dsr1_fp8_mi300x.sh` +
     model `.../DeepSeek-R1` precision fp8 gpu mi300x → tier `exact`, returns
     that path; `minimaxm2.5_*` vs a `minimaxm3` model → tier `none`
     (version-digit demotion); a substring case → tier `fuzzy`, path returned
     but caller must not auto-apply.
   - **(§7 fail-soft)** missing / non-dir `$INFERENCEX_PATH` → `(None,"none")`,
     no raise.
   - **(§8 identity)** `materialize_config_with_envs(..., reference_server_args="")`
     output equals the same call without the kwarg (deep-equal the loaded YAML).

---

## Touched files summary

| File | Change |
|---|---|
| `inference_optimizer/reference_script.py` | NEW — `parse_reference_script()` (pair-aware, model-capturing) + `render_reference_script()` + `discover_reference_script()` (tiered, fail-soft) |
| `inference_optimizer/orchestrator/shared_state.py` | NEW fields `reference_server_args/envs/model/source`; add to fact-layer integrity tuple; render `current_setting.sh` in `save()` |
| `inference_optimizer/cli.py` | add `--reference-script`; parse → SharedState (non-resume); auto-discovery when unset (§7); manifest provenance |
| `inference_optimizer/orchestrator/coordinator.py` | model-gated per-task `params["reference_*"]` injection on baseline dispatch |
| `inference_optimizer/orchestrator/action_executors/_workload_envs.py` | two new kwargs; reference base as lowest-priority layer (before extra_server_args + per-model workarounds) |
| `inference_optimizer/orchestrator/action_executors/baseline.py` | pass the new kwargs from `params` |
| `inference_optimizer/tests/test_reference_script.py` | NEW |
| `inference_optimizer/tests/test_baseline_param_overrides.py` | extend (precedence) |
| `inference_optimizer/tests/test_shared_state*.py` | extend (resume survival, model-gate) |
| (follow-up) research-scout harvest | auto-seed `state.reference_*` when a matching recipe is found and no CLI value |

## Out of scope (intentionally)

- Executing the reference shell script.
- Parsing dynamic/`$VAR` flags or client-side bench args (TP/CONC/ISL/OSL stay
  owned by existing CLI/env seeding).
- Changing Magpie's `{framework}_{gpu_type}.sh`.
- Reading `current_setting.sh` back to restore state (it is derived; resume uses
  `state.json`).
