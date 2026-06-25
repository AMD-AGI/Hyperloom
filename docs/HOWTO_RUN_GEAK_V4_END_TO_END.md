# How to run GEAK v4 end-to-end (analysis.md → kernel speedup → combined-patch E2E)

A complete, copy-paste walkthrough for a newcomer. It takes you from a TraceLens
`analysis.md` to per-kernel optimized patches and a final **end-to-end (E2E)
serving throughput** number with **all optimized patches applied together**.

It generalizes to **any workload/model** (not just Llama). Anywhere you see
`<...>` substitute your own value.

---

## 0. Mental model (read this first)

```
TraceLens analysis.md  ─┐
                        ├─►  HL dispatch  ──►  GEAK v4 workflow  ──►  per-kernel patch
HL kernel_candidates ──┘     (builds the          (optimizes each        (.diff / .py)
                              PROMPT)               kernel, 1/GPU)              │
                                                                               ▼
                                              apply ALL patches combined  ──► E2E A/B
                                              (apply_and_bench primitive)      tok/s Δ
```

Two ideas you must internalize:

1. **The PROMPT is the only interface to GEAK.** Everything GEAK needs — the device
   source, the captured serving shapes, the bottleneck hypothesis, and the
   *workload running config* (TP / input-len / output-len / concurrency) — is
   baked into ONE dispatch prompt. GEAK's own preprocessor builds the harness from
   that prompt. You do **not** hand GEAK a test file, a harness, or an op_test.

2. **The final number that matters is COMBINED E2E.** A per-kernel microbenchmark
   speedup (e.g. 1.26×) does not automatically translate to end-to-end serving
   gain (Amdahl's law). So after optimizing every hot kernel, you apply **all**
   the optimized patches **together** and remeasure the serving throughput.

GEAK **v4** specifically is an agentic optimizer (Director / TechLead / specialist
Engineers running budgeted rounds with independent verify + integrate) that runs
via the Claude Code **Workflow** tool — a `.js` orchestration script, not a shell
binary. **No installation/build of GEAK v4 is required**; you just run the script.

---

## 1. Repos, branches & PRs to clone/set up

Clone these three repos at the **exact branches** below. All changes are pushed to
remote — nothing is local-only.

| Repo | Remote | Branch (clone this) | PR | Purpose |
|---|---|---|---|---|
| **Hyperloom (HL)** | `github.com/AMD-AGI/Hyperloom` | `feat/dispatch-prompt-extra-context-clean` | [#703](https://github.com/AMD-AGI/Hyperloom/pull/703) | builds `kernel_candidates.json` + the enriched dispatch prompt; ships the `apply_and_bench.py` E2E primitive; enforces GEAK prompt-only dispatch; **runs the combined E2E autonomously** after the GEAK batch (commit `f2216a9`) |
| **GEAK v4** | `github.com/AMD-AGI/GEAK` | `feat/kernel-workflow-from-analysis` | [#306](https://github.com/AMD-AGI/GEAK/pull/306) | the v4 from-analysis workflow variant (`kernel_workflow_from_analysis.js`); branched off `GEAK_v4` |
| **GEAK v3** (only if you also run v3) | `github.com/AMD-AGI/GEAK` | `fix/ccache-298-on-322` | [#299](https://github.com/AMD-AGI/GEAK/pull/299) | ccache + MAX_JOBS so aiter/CK `.cu` recompiles finish in budget; pinned sha `39472353` |

```bash
# HL
git clone -b feat/dispatch-prompt-extra-context-clean \
  https://github.com/AMD-AGI/Hyperloom.git HL_fresh

# GEAK v4 (the optimizer this doc runs)
git clone -b feat/kernel-workflow-from-analysis \
  https://github.com/AMD-AGI/GEAK.git GEAK_v4_fresh

# GEAK v3 (optional — only if you also want the v3 side of the comparison)
git clone -b fix/ccache-298-on-322 \
  https://github.com/AMD-AGI/GEAK.git GEAK_pr299
```

> GEAK **v4 needs no build/install** — the workflow is the `.js` script run via the
> Claude Code Workflow tool. HL needs its normal `kernel-agent/scripts/install.sh`
> (it produces the runtime env + GEAK config used for dispatch).

## 2. Prerequisites / environment

| Thing | Value used in our runs | Notes |
|---|---|---|
| GEAK v4 repo | `<clone>/GEAK_v4_fresh` | no install; run the workflow `.js` directly |
| v4 from-analysis script | `<clone>/GEAK_v4_fresh/kernel_workflow/kernel_workflow_from_analysis.js` | the variant that starts from analysis.md (see §7) |
| HL checkout | `<clone>/HL_fresh` (branch `feat/dispatch-prompt-extra-context-clean`) | candidates + dispatch prompt + `apply_and_bench.py` |
| Kernel repo under optimization | `/sgl-workspace/aiter` | the repo whose `.cu` / `.py` kernels get rewritten |
| Serving stack | sglang (or vllm) | for the E2E A/B |
| GPUs | 8× MI300X (gfx942) | v4 runs 1 kernel per GPU in parallel |
| Model gateway | core42 gateway; `GEAK_USER=<you>@amd.com`, `ANTHROPIC_CUSTOM_HEADERS` | otherwise HTTP 400 (see your env setup script) |

**Always clean stray processes before a run** (zombies hold GPU memory and skew
latency):
```bash
pkill -f 'geak_vs_forge_driver|kernel_optimization|minisweagent|/opt/venv/bin/geak' 2>/dev/null
# verify GPUs idle (0% use, ~0.3GB baseline):
rocm-smi --showuse | grep -i 'GPU use'
rocm-smi --showmeminfo vram | grep -i 'Used'
```

---

## 3. Input: the TraceLens `analysis.md` (the authoritative agent report)

This is the **only** trace-derived input. It is the TraceLens **agent** report
(not a deterministic bypass) and it lists the hot operations, their captured
argument shapes, %E2E, roofline efficiency, and the bottleneck hypothesis.

Example (Llama-3.1-8B):
```
/wekafs/devalshah/TraceLens/tracelens_geak_workspace/top_11_models/
  meta-llama-Llama-3.1-8B-Instruct_20260613T205925Z_selected_16.08pct/tracelens/analysis.md
```

For **your** workload, point at the corresponding `.../tracelens/analysis.md`. The
sibling `perf_report_csvs/` directory is also consumed (per-kernel CSVs).

> The op identified in `analysis.md` is canonical. E.g. for Llama the attention op
> is `sglang_profiler::attention_paged_attention_ragged` (source
> `aiter_backend.py`, device kernel `attention_ragged.cu`) with a 10-tensor arg
> spec. GEAK must optimize **that** op — never a guessed substitute.

---

## 4. Build `kernel_candidates.json` with HL (the dispatch payload)

HL parses `analysis.md` into structured candidates (the hot kernels, their
captured shapes, source paths, dedup'd task groups) and writes
`kernel_candidates.json`. This is the **same** file that feeds GEAK v3, so v4 is
an apples-to-apples optimizer on identical info.

```bash
cd /wekafs/sapmajum/PROJECTS/OUTS/mixtral_autonomous

python3 geak_vs_forge_driver.py \
  --analysis  <.../tracelens/analysis.md> \
  --csv-dir   <.../tracelens/perf_report_csvs> \
  --framework sglang \
  --backend   geak \
  --model     <your-model-name> \
  --session-dir <ABS/path/to/session_dir> \
  --enrich \
  --serving-config "framework=sglang, TP=1, ISL=1024, OSL=1024, concurrency=64, num_prompts=320, dtype=bf16, gpu=MI300X (gfx942)" \
  --top-k 10
```

What the flags do:
- `--enrich` + `--serving-config` → attaches the authoritative **WORKLOAD
  CONTEXT** block (your real serving config) to every kernel's dispatch prompt.
  This is the #703 enrichment — it tells the optimizer the live decode context so
  its harness matches the workload (without it, the optimizer must *guess* the
  context and can optimize the wrong regime).
- Output: `<session-dir>/kernel-agent-run/kernel_candidates.json` plus, per
  kernel, a rendered dispatch prompt under
  `<session-dir>/kernel-agent/runs/<id>/prompts/geak-*.md`.

> **Get your `--serving-config` right.** Use your workload's ACTUAL serving
> parameters (read them from your serving/CI config): framework, tensor-parallel
> degree (TP), input seq len (ISL), output seq len (OSL), concurrency,
> num_prompts, kv-cache dtype, etc. This is the single most important knob for
> making kernel wins translate to E2E.

**Sanity-check the prompt is prompt-only + enriched:**
```bash
PF=$(ls -t <session-dir>/kernel-agent/runs/*/prompts/geak-*.md | head -1)
grep -c 'WORKLOAD CONTEXT'                 "$PF"   # -> 1  (enrichment present)
grep -cE 'test_pa.py|test_quant.py|Known benchmark/test' "$PF"   # -> 0  (no op_test leaked)
```

---

## 5. Run GEAK v4 on each hot kernel (1 kernel per GPU)

v4 runs via the Claude Code **Workflow** tool. Invoke the from-analysis variant
once per target kernel (`seed_target` selects the task group: `tg001`, `tg002`, …).

```jsonc
// Workflow tool call
{
  "scriptPath": "/wekafs/sapmajum/PROJECTS/GEAK_v4_fresh/kernel_workflow/kernel_workflow_from_analysis.js",
  "args": {
    "kernel_path":            "/sgl-workspace/aiter",          // repo holding the kernel source
    "workflow_dir":           "/wekafs/sapmajum/PROJECTS/GEAK_v4_fresh/kernel_workflow",
    "exp_root":               "<ABS/path>/v4_<workload>/exp",  // where v4 writes its run artifacts
    "analysis_md_path":       "<.../tracelens/analysis.md>",   // SAME file from §3
    "kernel_candidates_path": "<session-dir>/kernel-agent-run/kernel_candidates.json",  // from §4
    "dispatch_prompt_path":   "<session-dir>/.../prompts/geak-<id>.md",  // the EXACT enriched prompt from §4 (authoritative)
    "seed_target":            "tg001",                          // tg001=attention, tg002=quant, ...
    "budget":                 6,                                 // optimize rounds
    "max_no_improve":         2,                                 // early-stop after N stale rounds
    "gpu_ids":                "0,1,2,3,4,5,6,7",
    "task":                   "Optimize <kernel>; bottleneck/shapes are in the provided analysis (task_group tgNNN). Build a standalone harness; keep all build/JIT inside the isolated EVAL_DIR workspace."
  }
}
```

Key arg notes:
- `dispatch_prompt_path` is the **authoritative** input — v4 reads the verbatim HL
  dispatch prompt FIRST (device source + VERBATIM captured shapes "do NOT invent"
  + WORKLOAD CONTEXT). This is what makes v4 build its harness from the real
  serving shapes instead of self-inventing them.
- `analysis_md_path` + `kernel_candidates_path` seed v4's roadmap (replacing
  stock v4's own Analyze/Profile — see §7).
- Run it **once per kernel**: `seed_target: "tg001"`, then again `"tg002"`, etc.
  Each occupies its own GPU; you can launch them concurrently.

**What v4 returns:** `final_geomean` (Director-validated FULL_BENCHMARK verified
speedup) and the paths to the report + the optimized patch
(`<exp_root>/.../final_patch.diff`). v4 builds its own harness + baseline, runs
budgeted optimize rounds, independently verifies, and writes the patch.

Collect, per kernel: the verified speedup and the `final_patch.diff` path. You'll
apply all of them together in §6.

---

## 6. Combined E2E: apply ALL optimized patches together, then remeasure

This is the deliverable number. There are two ways to get it — **autonomous
(default, recommended)** and **manual (fallback)**. Both use the same single shared
gate-less primitive `apply_and_bench.py` (HL `kernel-agent/tools/`), which:
1. applies each patch to its target (handles aiter `.cu` rebuild with
   `AITER_REBUILD=1` + jit/cpp_itfs cache invalidation; also handles `.diff` via
   `git apply` and full-source-file replacement),
2. warm-serves the model and runs an A/B throughput benchmark (baseline vs
   patched), N reps, median,
3. verifies **engagement proof** (the patched kernel is actually on the live
   serving path — e.g. aiter JIT rebuild markers), and
4. reports the delta and reverts the source. **No KEEP/REVERT/NEEDS_REVIEW
   policy** — it is pure measurement.

### 6a. Autonomous (default) — HL runs it for you, no manual step

When you dispatch the GEAK batch through HL (the `geak_vs_forge_driver.py` path in
§4), HL **automatically** runs the combined E2E once all kernels finish: it collects
each kernel's best patch, applies them ALL together, rebuilds, warm-serves the A/B,
and attaches the result to `result_geak.json` under a `combined_e2e` block. **No
manual `apply_and_bench` call.** (HL commit `f2216a9` on #703.)

- It is **GEAK-only and opt-in**: the driver sets `payload["combined_e2e"]=True` and
  passes the serving knobs parsed from `--serving-config`. It only fires when the
  backend is geak, a servable `model_path` is present, and ≥1 GPU is visible —
  otherwise it is a silent no-op (OOB/forge never trigger it).
- It applies each kernel's **best microbench patch** regardless of the per-kernel
  KEEP/REVERT verdict (the E2E A/B is the arbiter).
- Result: `result_geak.json` → `combined_e2e: {baseline_median_tok_s,
  patched_median_tok_s, delta_pct, engagement_proof[...]}`. The driver also prints
  the delta. **Nothing else to run** — skip to §6c (interpreting).

So the normal flow is: run §4 (dispatch through HL) and read the `combined_e2e`
block. The manual path below is only for re-measuring an existing run, a custom
patch selection, or v4 runs driven outside HL's batch handler.

### 6b. Manual (fallback) — explicit `apply_and_bench`

Apply **all** kernels at once with repeated `--pair PATCH:TARGET`:

```bash
cd /wekafs/sapmajum/PROJECTS/HL_fresh/kernel-agent

python3 tools/apply_and_bench.py \
  --pair "<exp_root>/.../attention/final_patch.diff:/sgl-workspace/aiter/csrc/kernels/attention_ragged.cu" \
  --pair "<exp_root>/.../quant/final_patch.diff:/sgl-workspace/aiter/csrc/kernels/quant_kernels.cu" \
  --model    <your-model-name> \
  --backend  sglang \
  --tp 1 --isl 1024 --osl 1024 --conc 64 --num-prompts 320 \
  --reps 3 \
  --aiter-rebuild \
  --backup-root <ABS/path>/e2e_backups \
  --out-dir     <ABS/path>/e2e_combined \
  --gpu 0
```

- One `--pair` per optimized kernel = **combined** application (this is what the
  user wants: all patches together → one final E2E number).
- `--aiter-rebuild` forces the aiter `.cu` JIT to recompile against the patched
  source (otherwise it silently serves the prebuilt baseline `.so` and you'd
  measure ~1.00×).
- Match `--tp/--isl/--osl/--conc/--num-prompts` to the `--serving-config` you used
  in §4, so the E2E config equals the optimization target config.
- Single-kernel mode also exists: `--patch-path <p> --target-file <f>`.

**Output:** a JSON with `baseline_median_tok_s`, `patched_median_tok_s`,
`delta_pct`, and `engagement_proof[].engaged=true`. The `delta_pct` is your
combined E2E result.

### 6c. Interpreting the combined E2E number

> **Interpreting E2E:** on host/decode-bound configs (e.g. an 8B model at TP=1,
> ~55% GPU), even large kernel speedups can yield a near-flat E2E delta — that's
> Amdahl, not a bug. Report the combined `delta_pct` honestly and note the
> bound. Treat |Δ| within ~2–3% as flat (serving noise).

---

## 7. (Background) why the "from-analysis" v4 variant exists

Stock v4 (`kernel_workflow.js`) runs its OWN `Analyze` (re-derives the roadmap)
and `Profile` (rocprof) phases. That would make v4 optimize against a *different*
analysis than the one in your `analysis.md`. The variant
`kernel_workflow_from_analysis.js` replaces those two phases with a **`Seed`**
phase: the TechLead reads ONLY `analysis_md_path` + `kernel_candidates_path` (+
the authoritative `dispatch_prompt_path`) and maps them into v4's roadmap — no
re-trace, no re-profile. Everything after (Setup, Benchmark/harness build,
Optimize loop, Verify, Merge, Report) is byte-identical to stock v4.

Use the variant whenever you want v4 to act on a pre-computed TraceLens/HL
analysis (the normal case here, and required for a fair v3-vs-v4 comparison).

---

## 8. End-to-end checklist (TL;DR)

1. [ ] Clean zombies; confirm GPUs idle.
2. [ ] Locate your workload's `analysis.md` (+ `perf_report_csvs/`).
3. [ ] Run `geak_vs_forge_driver.py … --enrich --serving-config "<your real config>"`
       → `kernel_candidates.json` + enriched dispatch prompts.
4. [ ] Verify each prompt: `WORKLOAD CONTEXT` present (=1), no op_test path (=0).
5. [ ] For each hot kernel: run `kernel_workflow_from_analysis.js` via Workflow
       with `seed_target=tgNNN` + `dispatch_prompt_path=<that kernel's prompt>`.
       Collect verified speedup + `final_patch.diff`.
6. [ ] Combined E2E: if you dispatched through HL (step 3), it runs
       **autonomously** — read `result_geak.json` → `combined_e2e` (§6a). Otherwise
       run `apply_and_bench.py` manually (one `--pair` each, `--aiter-rebuild`,
       serving config matching step 3) — §6b.
7. [ ] Record: per-kernel verified speedups + the combined E2E `delta_pct`
       (with engagement_proof=true).

---

## 9. Common pitfalls

- **Wrong serving config in `--serving-config`** → optimizer targets the wrong
  decode regime → kernel win doesn't transfer to E2E. Use the real values.
- **Forgetting `--aiter-rebuild`** → E2E measures the baseline `.so` → ~1.00×.
- **Applying patches one at a time** → you only learn per-kernel E2E. The
  requested deliverable is **combined** (all patches together).
- **Stray processes from a prior run** → hold GPU memory, corrupt latency. Always
  clean first.
- **Editing budgets** → don't. Use GEAK's default wall-clock budget unless you
  have a specific reason; document it if you ever change it.
- **Leaking an op_test into the prompt** → GEAK is PROMPT-ONLY; never pass a
  test_command/harness/op_test. HL #703 enforces this for the geak backend by
  default; just verify with the grep in §4.
```
