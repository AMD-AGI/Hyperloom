# atom_gap2.md — behavioural parity gaps (atom vs sglang/vllm)

Audit date: 2026-05-28
Branch: `feature/zhenggong/atom` HEAD = `0b36e80`
Scope: every place in the live code where `--framework atom` is
treated DIFFERENTLY from sglang / vllm. Multi-node TP is the one
acknowledged exception.

Companion to [`atom_gap1.md`](atom_gap1.md) which audited
plan-vs-code drift. This file audits **atom-vs-other-frameworks**
behaviour parity.

---

## Classification legend

* **GAP-A (bug)** — a real bug; atom misses behaviour it should
  have. Must fix to reach parity.
* **GAP-B (stale doc/comment)** — docs claim atom behaves
  differently when in fact the live code is at parity. Fix to
  reduce operator confusion.
* **GAP-C (annotation / quality)** — atom lacks the optional
  profiler / trace annotations sglang+vllm get via custom server
  patches. By-design today; closing requires new design work
  (TraceLens patch set for atom, or `--mark-trace` evaluation).
* **GAP-D (UX default)** — atom's launcher / config doesn't ship
  the sensible cold-start defaults sglang/vllm get. Quick fix.
* **GAIN** — atom has something sglang/vllm don't (asymmetric the
  other way). Note for the record; no action needed unless full
  symmetry is desired.
* **EXPECTED** — intentional exception (multi-node TP, per-framework
  EXTRA_*_ARGS env names, dedicated `rocm/atom` image,
  vLLM-wire bench client `--backend vllm`). No action.

---

## Severity-ranked gap inventory

### B1 (GAP-A, high) — `kernel_request_handlers._load_materialized_workload_metadata` reads `EXTRA_SGLANG_ARGS` on atom sessions

**File:** `inference_optimizer/orchestrator/kernel_request_handlers.py:281`

```python
server_key = "EXTRA_VLLM_ARGS" if framework == "vllm" else "EXTRA_SGLANG_ARGS"
```

**What goes wrong:** for an atom session the materialised YAML
carries `envs.EXTRA_ATOM_ARGS` (e.g. `--trust-remote-code
--level 3 --enable-expert-parallel`). The handler reads
`envs.EXTRA_SGLANG_ARGS` → empty string → kernel metadata
(`server_args` / `server_args_argv`) drops every atom-side flag.
Downstream TraceLens / GEAK / Cursor prompts that consume
`runtime_args.server_args` see an EMPTY string, which means
specialist proposals are graded against a non-existent flag
context.

**Why it slipped:** the ternary predates atom; the `else` branch
was historically correct because only sglang and vllm existed.
Phases 2/3/4 of `atom_plan/` did not touch this site.

**Fix:** route through the existing single source of truth for
the per-framework env name — `_grid_runner.server_args_env_name`:

```python
from .action_executors._grid_runner import server_args_env_name
server_key = server_args_env_name(framework)  # EXTRA_{SGLANG,VLLM,ATOM}_ARGS
```

**Test:** parametrise the existing
`test_kernel_request_handlers_units.py` over sglang / vllm / atom,
asserting `server_args` round-trips the materialised
`EXTRA_<FRAMEWORK>_ARGS` value for all three.

**Effort:** 10 min code + 15 min tests.

---

### B2 (GAP-A, medium) — `tracelens_skill_runner._FRAMEWORK_PKG_FALLBACK_ROOTS` lacks atom

**File:** `kernel-agent/tools/tracelens_skill_runner.py:917-926`

```python
_FRAMEWORK_PKG_FALLBACK_ROOTS: dict[str, tuple[str, ...]] = {
    "aiter":  ("/sgl-workspace/aiter",),
    "sglang": ("/sgl-workspace/sglang/python", "/sgl-workspace/sglang"),
    "vllm":   ("/usr/local/lib/python3.12/dist-packages", ...),
}
```

**What goes wrong:** kernel-agent's offline source-file resolver
walks this fallback table when `import <pkg>` hasn't run (CSV-only
parses, static-analysis paths). atom kernels referenced by a
TraceLens CSV under `/app/ATOM/atom/...` or `site-packages/atom/...`
fall through to the "could not resolve" branch and the kernel-opt
proposal is silently rejected as
`source file not resolved`.

**Why it slipped:** Phase 2.5 wired
`kernel_request_handlers._REUSABLE_SOURCE_ROOTS` AND
`tracelens_analysis._REUSABLE_SOURCE_ROOTS` for atom but missed
the analogous fallback table in `tracelens_skill_runner`.

**Fix:**

```python
_FRAMEWORK_PKG_FALLBACK_ROOTS: dict[str, tuple[str, ...]] = {
    "aiter":  ("/sgl-workspace/aiter",),
    "sglang": ("/sgl-workspace/sglang/python", "/sgl-workspace/sglang"),
    "vllm":   ("/usr/local/lib/python3.12/dist-packages",
               "/usr/local/lib/python3.10/dist-packages",
               "/opt/venv/lib/python3.10/site-packages",
               "/sgl-workspace/vllm"),
    "atom":   ("/app/ATOM",
               "/usr/local/lib/python3.12/dist-packages",
               "/usr/local/lib/python3.10/dist-packages",
               "/opt/venv/lib/python3.10/site-packages",
               "/opt/venv/lib/python3.12/site-packages"),
}
```

**Test:** add `"atom"` to the existing fallback-roots
parametrisation in `kernel-agent/tools/test_tracelens_skill_runner.py`
(if it exists) or add a fresh test that resolves a relative
`atom/model_engine/model_runner.py` against a synthetic atom
fallback root.

**Effort:** 5 min code + 10 min test.

---

### B3 (GAP-A, low) — `infer_analysis_mode` excludes atom from `"inference"` default

**File:** `kernel-agent/tools/tracelens_skill_runner.py:95-102`

```python
def infer_analysis_mode(framework: str, requested: str) -> str:
    requested = (requested or "").strip().lower()
    if requested and requested != "default":
        return requested
    if (framework or "").strip().lower() in {"vllm", "sglang"}:
        return "inference"
    return requested or "default"
```

**What goes wrong:** atom sessions default to `"default"` analysis
mode instead of the `"inference"` mode used for sglang/vllm. The
difference is TraceLens-internal but biases kernel-grouping
heuristics toward generic torch profiles. Atom traces are produced
by the same torch profiler API and the same chrome-trace JSON
shape, so the inference-mode grouping should apply.

**Fix:**

```python
if (framework or "").strip().lower() in {"vllm", "sglang", "atom"}:
    return "inference"
```

**Test:** parametrise the existing analysis-mode test (if any) or
add `test_infer_analysis_mode_atom_returns_inference`.

**Effort:** 2 min code + 5 min test.

---

### B4 (GAP-B, high) — root `CLAUDE.md` IR-8 entry is entirely pre-Phase-1/2/3

**File:** `CLAUDE.md:147`

> **IR-8 (atom)**: `--framework atom` is single-node only
> (`--nodes 1`) and auto-tightens `--no-kernel --no-framework
> --no-enable-roofline` on entry. atom in Magpie v1 has no
> torch_profiler integration (profile/roofline executors
> short-circuit to `status="skipped"`) and no sglang/vllm-
> equivalent source patcher, so kernel-agent / framework-agent
> paths are wired off. Baseline + EXPLORE (specialist +
> default_grid) still run.

**Reality (post-Phase-1/2/3):**

* `--framework atom` is single-node only — ✓ still true.
* "auto-tightens `--no-kernel --no-framework --no-enable-roofline`" — ✗ all three lifted.
* "atom in Magpie v1 has no torch_profiler integration" — ✗ Phase 1 wired the `--torch-profiler-dir` bridge in `atom_mi*x.sh`.
* "profile/roofline executors short-circuit to `status='skipped'`" — ✗ short-circuit removed; tests pin the no-longer-skip behaviour.
* "no sglang/vllm-equivalent source patcher … kernel-agent / framework-agent paths are wired off" — ✗ Phase 2 opened kernel-agent (atom source roots in allowlist + reusable roots + help-text probe); Phase 3 opened framework-agent (atom repo URL).

**Why it slipped:** the doc was authored before `atom_plan/` and
not updated alongside Phases 1–3.

**Fix:** rewrite the IR-8 entry. Proposed text:

> **IR-8 (atom)**: `--framework atom` is single-node only
> (`--nodes>=2` fails fast in
> `_apply_atom_auto_tighten` / `_assert_atom_single_node`). After
> `atom_plan/` Phases 1–3 every other auto-tighten was lifted:
> kernel-agent runs on atom (source roots
> `/app/ATOM/atom/` + `aiter/` shared with sglang/vllm),
> framework-agent runs on atom (repo
> `https://github.com/ROCm/ATOM.git`), and profile / roofline /
> TraceLens run on atom (Magpie `atom_mi*x.sh` bridges
> `PROFILE=1` to atom's `--torch-profiler-dir`; atom writes
> standard `*.pt.trace.json.gz` traces TraceLens consumes
> unchanged). The single remaining behavioural difference vs
> sglang/vllm is multi-node TP (atom upstream lacks it).

**Effort:** 5 min.

---

### B5 (GAP-B, medium) — `baseline_atom.yaml` header claims auto-tighten still applies

**File:** `inference_optimizer/scripts/configs/baseline_atom.yaml:30-31, 69`

```yaml
#   - Kernel-agent / framework-agent are auto-disabled (no atom source
#     patcher); cli.py auto-sets --no-kernel and --no-framework.
...
  profiler:
    torch_profiler:
      enabled: false           # atom has no profiler wiring in Magpie v1
```

**Reality:** both lines stale post-Phase-1/2/3.

**Fix:** rewrite both blocks. Proposed:

```yaml
#   - Kernel-agent + framework-agent + profile / roofline / TraceLens
#     all run on atom after atom_plan/ phases 1-3. The only remaining
#     auto-tighten in cli.py is the --nodes>=2 fail-fast guard.
...
  profiler:
    torch_profiler:
      enabled: false           # baseline doesn't profile; profile_atom.yaml does
```

**Effort:** 5 min.

---

### B6 (GAP-B, low) — `orchestration.md` framework-roots example omits atom

**File:** `inference_optimizer/orchestrator/system_prompts/orchestration.md:139-140`

```text
  (c) under one of the framework source roots listed in SESSION CONTEXT
      (`framework_source_roots`, default `/sgl-workspace/{aiter,sglang,vllm}/`
      plus any `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` env supplement)
      for `source_file` references.
```

**Reality:** Phase 2.1 added `/app/ATOM/atom/` to
`_DEFAULT_SOURCE_ROOTS`. Runtime is correct; only the prompt
prose is stale, so the LLM may believe atom paths fall outside
the allowlist (they don't).

**Fix:** add atom to the example string:

```text
default `/sgl-workspace/{aiter,sglang,vllm}/` + `/app/ATOM/atom/`
```

**Effort:** 2 min.

---

### C1 (GAP-C) — no atom equivalent of `_validate_trace_structure` check 5

**File:** `inference_optimizer/orchestrator/action_executors/profile.py:256-266`

```python
# --- Check 5 (Deval): sglang kernel_shape_profiler presence ---
if framework.lower() == "sglang" and main_text is not None:
    if "kernel_shape_profiler" not in main_text:
        issues.append(...)
```

**What this catches on sglang:** the SGLang TraceLens
shape-discovery patch (`_server_patcher.py` PR #207) injects
`kernel_shape_profiler` annotation events. If those events are
missing post-profile, the patch didn't reach the live SGLang
build → operator-facing warning.

**Atom equivalent:** none. atom doesn't have a TraceLens server
patch today (Phase 6.3 investigated `--mark-trace` and deferred
the YAML enable pending live verification). When the
`--mark-trace` follow-up lands the analogous check would be
something like:

```python
if framework.lower() == "atom" and main_text is not None:
    if "record_function" not in main_text:
        issues.append(...)
```

**Fix proposal:** defer until phase 6.3 follow-up runs the live
`--mark-trace`-on vs -off comparison; if Outcome A (TraceLens
sees the annotations), wire both the YAML edit AND the check 5
analogue together.

**Effort:** 1-2 h once live signal is available.

---

### C2 (GAP-C) — atom lacks TraceLens annotation-rich profile path

**Files:**
* `inference_optimizer/orchestrator/action_executors/_workload_envs.py:268-303`
* `inference_optimizer/scripts/configs/profile_atom.yaml`
* `kernel-agent/tools/_server_patcher.py` (sglang/vllm patches)

**What sglang gets:** `_server_patcher.ensure_sglang_patched_for_*`
injects `--enable-shape-discovery-for-cuda-graph-profile` + the
TraceLens #207 patch on top of sglang server source.

**What vllm gets:** vLLM's
`--profiler-config.detailed_trace_annotation True` (TraceLens #194
patched build) gets passed via `PROFILER_EXTRA_BODY`.

**What atom gets:** atom uses native torch profiler via the
`/start_profile` + `/stop_profile` HTTP endpoints, no extra
annotations. The result is correct chrome traces but with less
inference-stack-aware grouping than the patched sglang/vllm
builds produce.

**Closing path (deferred per Phase 6.3 outcome):**
1. Enable `--mark-trace` in `EXTRA_ATOM_ARGS` of `profile_atom.yaml`.
2. Confirm `with record_function("<prefix>"):` blocks show in
   the atom trace JSON as named `cpu_op` events.
3. If TraceLens consumes them well (Outcome A in
   `atom_plan/phase6_atom_ux_polish/6.3_mark_trace_investigation.md`),
   wire the YAML edit + a `_validate_trace_structure` check
   (C1 follow-up).

**Effort:** part of Phase 7's `TODO[atom-mark-trace]`. No code
change until live signal.

---

### D1 (GAP-D, low) — Magpie `atom_mi*x.sh` doesn't inject sensible default flags

**Files:**
* `Magpie/Magpie/scripts/benchmark/sglang_mi300x.sh:73-89`
* `Magpie/Magpie/scripts/benchmark/sglang_mi355x.sh:70-86`
* `Magpie/Magpie/scripts/benchmark/atom_mi300x.sh`, `atom_mi355x.sh`
  (no analogous block)

**What sglang gets:** `DEFAULT_ARGS=""` block prepends
`--mem-fraction-static=0.8` and `--disable-radix-cache` if the
operator hasn't supplied them via `EXTRA_SGLANG_ARGS`.

**Why:** these two flags are empirically known cold-start
defaults that improve OOM resilience and avoid radix-cache
overhead on small models. Both are sglang-specific (atom has its
own equivalents).

**Atom equivalent flags worth defaulting on:**
* `--level 2` — atom's torch.compile + cudagraph bracket is
  usually a safe cold-start choice (level 3 needs longer warmup).
* `--enforce-eager` → NO; debug-only.
* `--kv_cache_dtype fp8` — only on FP8-shipped models; can't
  default without model class.

**Fix (conservative):** add `DEFAULT_ARGS=""` block in
`atom_mi*x.sh` injecting `--level 2` when not already in
`EXTRA_ATOM_ARGS`. atom's `_atom_default_grid` (Phase 6.2)
already includes `atom_level_2` as a baseline variant, so the
default would harmonise the cold-start with the EXPLORE seed.

**Effort:** 10 min × 2 scripts + a small Magpie test.
**Risk:** atom upstream may change `--level` semantics. Same risk
as sglang's `--mem-fraction-static` default; manageable.

---

## Asymmetries that are by-design (no action)

| Item | Why kept asymmetric |
|---|---|
| Per-framework env names `EXTRA_SGLANG_ARGS` / `EXTRA_VLLM_ARGS` / `EXTRA_ATOM_ARGS` | Magpie contract: per-framework slots routed via the correct wrapper script |
| Dedicated `rocm/atom:latest` image vs sglang/vllm dev images | Phase 5; atom upstream's own published runtime image |
| Bench client `--backend vllm` for atom | atom HTTP API is vLLM-wire-compatible; refactor explicitly out-of-scope per `atom_plan/00_overview.md` |
| atom multi-node TP guard (`--nodes>=2` fails fast) | atom upstream lacks multi-node TP |
| No atom TraceLens **server** patch in `_server_patcher.py` | atom uses native torch profiler; the patch set is sglang/vllm-specific (annotation/shape-discovery on patched server source) — closing requires authoring an atom-aware patch set (Phase 6.3 / TraceLens #atom) |
| `_default_grid_for_framework("atom", ...)` exists but sglang/vllm return `[]` | Phase 6.2 PARITY GAIN. Closing would require designing programmatic seed grids for sglang and vllm too |

---

## PARITY GAINs (atom has, sglang/vllm don't)

| Item | Where |
|---|---|
| Programmatic cold-start grid `_atom_default_grid` | `explore.py:209-321`; sglang/vllm rely on LLM-emitted variants |
| Per-framework KB partition helpers | `framework-agent/kb.py::path_for_framework` — available for all three but only atom session populates it today |
| Per-framework specialist hint blocks (serving / kernels / dist) | `specialist_prompt_builder.py::_is_atom` — atom-specific paths; sglang/vllm specialists get the legacy single block |

**Recommended posture:** keep the GAINs. They reflect lessons
learned during atom enablement and don't break sglang/vllm. If
parity is meant strictly bidirectional, the natural next step is
to write `_sglang_default_grid` / `_vllm_default_grid` analogues
and `_is_sglang` / `_is_vllm` focus-block branches — separate
workstream beyond this gap report's scope.

---

## Suggested commit cadence (fix order)

| Commit | Gap(s) | Subject |
|---|---|---|
| H1 | B1 | `fix(atom): kernel_request_handlers reads EXTRA_ATOM_ARGS on atom sessions` |
| H2 | B2 + B3 | `fix(atom): tracelens_skill_runner adds atom fallback roots + inference mode` |
| H3 | B4 + B5 + B6 | `docs(atom): refresh IR-8 description across CLAUDE.md / baseline_atom.yaml / orchestration.md` |
| H4 | D1 | `chore(magpie): atom_mi*x.sh injects --level 2 default` (Magpie repo) |
| (deferred) | C1 + C2 | Held for Phase 7 live verification → `--mark-trace` decision |

H1 + H2 are the only ones that change live behaviour. H3 is doc-
only. H4 is a small Magpie ergonomics improvement.

---

## Bottom line

After `atom_plan/` Phases 1–6 the live behaviour for
`--framework atom` is **already very close to parity** with
sglang/vllm. Three real bugs remain:

1. **B1** silently drops atom's server-flag context from the
   kernel-opt metadata path (highest impact — affects every
   kernel proposal on atom).
2. **B2** silently rejects atom kernel source files in the
   offline classifier path.
3. **B3** picks the wrong TraceLens analysis mode on atom traces.

The rest is documentation drift (B4-B6) plus a deferred
profile-annotation parity question (C1+C2) and one minor UX
default (D1).

None of the remaining gaps require new design work — every fix
is a small, well-scoped patch. **Closing B1+B2+B3 brings atom to
functional parity** with sglang/vllm for the end-to-end
optimisation loop.
