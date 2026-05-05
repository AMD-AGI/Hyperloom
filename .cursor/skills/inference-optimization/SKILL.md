---
name: inference-optimization-magpie
description: |
  Autonomous DFS-guided inference optimization for LLM serving on AMD MI355X GPUs.
  Uses heuristic-scored depth-first search to systematically explore optimization actions
  (backend switches, server params, kernel optimization, target comparison) and maximize
  throughput per GPU (tok/s/GPU) with accuracy as a hard constraint.
globs:
  - "**/inference*optim*"
  - "**/benchmark*"
  - "**/sglang*"
  - "**/vllm*"
---

# Inference Optimization — DFS Orchestrator

## Overview

This skill optimizes LLM inference serving throughput (tok/s/GPU) on AMD MI355X GPUs.
It runs a **depth-first search** over optimization actions, guided by a heuristic scoring
function. The search is fully autonomous — no human prompting required.

**Primary objective:** maximize `tok/s/GPU`
**Hard constraint:** accuracy (numerical correctness) must not degrade
**Optional target:** if an external baseline is provided (e.g. NVIDIA B200), the target
gap acts as an urgency multiplier on all action scores.

## Execution Mode

This skill supports two execution modes. **Read the mode-specific document for your mode
before starting:**

- **Local mode** (Cursor IDE, direct shell): see [`modes/LOCAL.md`](modes/LOCAL.md)
- **Claw mode** (SaFE RayJob, `exec_on_gpu`): see [`modes/CLAW.md`](modes/CLAW.md)

**Auto-detection:** `GEAK_LOCAL=true` → local mode (default). Claw client context → claw mode.

## Iron Rules (non-negotiable)

These rules apply to ALL modes. Violating any invalidates the optimization run.

### IR-1: Submit ALL kernel candidates in parallel

The kernel-opt action MUST submit `GEAK_TOP_CANDIDATES` (default 5) candidates to ALL active backends (`KERNEL_OPT_BACKENDS`) **simultaneously**. Submitting only 1 candidate when multiple exist, or submitting to backends sequentially instead of in parallel = violation.

### IR-2: NEVER modify kernel source before GEAK submission

Submit the kernel source **exactly as extracted**. Do NOT strip decorators, change strides, replace `@triton_heuristics` with `@triton.jit`, or make any "cleanup" edits. GEAK's agent handles kernel adaptation internally.

### IR-3: Integration (Phase 8) is MANDATORY

After GEAK returns optimized kernels, you MUST execute the integrate action (patch → re-baseline → decide). Skipping means GEAK results are never validated end-to-end. Re-baseline uses `magpie benchmark --benchmark-config` with the same YAML envs as the original baseline. See `actions/integrate.md` for details.

### IR-4: Always kill_server + check_gpu_memory before server launch

Every server launch must be preceded by killing any existing server process and verifying GPU memory is free.

### IR-5: Safe process management

**NEVER use `pkill -f sglang`** — it kills Ray workers in claw mode. Only use:

```bash
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
# or for vLLM:
kill $(pgrep -f 'python.*-m vllm.entrypoints') 2>/dev/null
```

Wait `SERVER_KILL_WAIT_S` seconds between kill and relaunch. Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR` after profiling.

### IR-6: Use `patch_inductor.py --target-file` for Inductor patching

Always use `scripts/patch_inductor.py` with `--target-file`. The `--cache-dir` option has been removed.

**CRITICAL:** When GEAK changes block sizes or warp counts, you MUST also pass `--best-config` with the updated tiling parameters. Patching only the kernel `.py` without updating `.best_config` causes numerical corruption (garbled output). See `actions/integrate.md` for details.

### IR-7: NEVER modify GEAK configuration

GEAK is an external service — treat it as **read-only infrastructure**. The skill MUST NOT
modify any GEAK configuration files, settings, or parameters beyond what is passed as
arguments to `geak_create_task`. Specifically:

- **Do NOT** modify GEAK server config, workspace settings, or API configuration
- **Do NOT** write to or alter any files under the GEAK config/settings directories
- **Do NOT** change `KERNEL_OPT_WORKSPACE`, `GEAK_STEP_LIMIT`, or other constants
  at runtime (use the values from the constants table above or user overrides)
- **Do NOT** modify the GEAK MCP server configuration (`cursor_mcp_config.json`, etc.)
- **Do NOT** modify any test data, results, or configuration files belonging to GEAK
  (e.g., `tests/test_data/`, `server/config.py`, `server/templates/`)

The ONLY interaction allowed is through these GEAK MCP tool calls:
`geak_get_model_config` (read-only), `geak_create_task`, `geak_submit_task`,
`geak_get_task`, `geak_get_outputs`, `geak_download_file`, `geak_list_tasks`.

**NEVER call `geak_set_model_config`** — the LLM backend is pre-configured by the
administrator. Changing it risks setting a non-existent model and breaking all tasks.

Violation = immediate run invalidation.

**Additional mode-specific Iron Rules are defined in [`modes/CLAW.md`](modes/CLAW.md) (IR-8 through IR-11) and [`modes/LOCAL.md`](modes/LOCAL.md) (IR-12).**

## Kernel Optimization & Tooling Constants

All values below are the **single source of truth**. All actions reference these by name.

| Constant | Value | Description |
|----------|-------|-------------|
| `KERNEL_OPT_BACKENDS` | `geak,codex` | Comma-separated active backends. Any combination of: `geak`, `codex`, `claude`, `llm`. User can override in prompt. |
| `OOB_ROUND_ITERATIONS` | 3 | Iterations per Codex/Claude round (submit → local benchmark → feedback → re-submit). Best result wins. |
| `KERNEL_OPT_IMAGE` | *(provided by CI or user)* | Framework image for all kernel-opt backends (GEAK + OOB). One image per run, determined by framework (SGLang/vLLM). |
| `KERNEL_OPT_WORKSPACE` | `control-plane-moe` | SaFE workspace for kernel-opt backends (GEAK + OOB). User can override. |
| `GEAK_STEP_LIMIT` | 100 | Max agent steps per GEAK task |
| `GEAK_MAX_RETRIES` | 3 | Max submission retries per kernel |
| `GEAK_MAX_SUBMISSIONS` | 15 | Total GEAK submissions budget per run |
| `GEAK_TOP_CANDIDATES` | 5 | Number of top kernel candidates to submit |
| `GEAK_CONSECUTIVE_DISCARDS` | 5 | Stop after this many consecutive discards |
| `GEAK_WALL_CLOCK_MIN` | 120 | Max wall-clock minutes for kernel-opt action |
| `GEAK_POLL_INTERVAL_S` | 60 | Seconds between GEAK task status polls |
| `GEAK_POLL_TIMEOUT_MIN` | 15 | Max minutes to poll a single GEAK task |
| `MIN_GPU_PCT` | 3 | Minimum GPU time % to consider a kernel as GEAK candidate |
| `SERVER_KILL_WAIT_S` | 10 | Seconds to wait between server kill and relaunch |
| `FILTERED_TRACE_NAME` | `filtered-TP-0.trace.json.gz` | Preferred trace file for TraceLens analysis |

**ALWAYS pass `KERNEL_OPT_IMAGE` to all kernel-opt backends (GEAK + OOB), regardless of kernel type.** For kernels whose source exists in the image (e.g., `/sgl-workspace/aiter/`), the pod uses the same image. For runtime-generated kernels (e.g., `/tmp/torchinductor_root/` from `torch.compile`), do NOT include `kernel_url`/`kernel_repo` in the prompt; copy files to shared NFS or rely on `files[].content` only.

**Claw-mode constants are in [`modes/CLAW.md`](modes/CLAW.md).**

### Magpie Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAGPIE_PATH` | auto-resolved (see below) | Magpie installation path (auto-mirrored to `/tmp/Magpie` when read-only) |
| `MAGPIE_RUN_MODE` | `local` | Benchmark execution mode (local, no Docker) |
| `MAGPIE_DFS_NUM_PROMPTS_MULTIPLIER` | 3 | NUM_PROMPTS = CONC × this for DFS loop (short runs) |
| `MAGPIE_BASELINE_NUM_PROMPTS_MULTIPLIER` | 10 | NUM_PROMPTS = CONC × this for baseline (default) |
| `MAGPIE_PROFILE_NUM_PROMPTS` | `CONC * 10` | NUM_PROMPTS for profiling runs (steady-state windowing controls trace size) |
| `MAGPIE_TIMEOUT_S` | 3600 | Bash/YAML timeout for all magpie benchmark calls (must cover server startup) |

### TraceLens Profiler Constants

Profiling for TraceLens analysis requires specific settings beyond basic torch profiling.
See `actions/profile.md` for the full profiling procedure.

| Constant | Formula / Value | Description |
|----------|-----------------|-------------|
| `MAX_ITERS` | `min(1024, max(256, 16 * OSL / CONC))` | Execution steps to profile (steady-state window size) |
| `DELAY_ITERS` | `(((R+1)/2) * 5 * OSL) - (MAX_ITERS/2)` | Step at which profiling begins (R = NUM_PROMPTS / CONC) |
| `PROFILE_NUM_PROMPTS` | `CONC * 10` | Overrides InferenceX default via env var (prevents cap to max_concurrency) |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 1200 | Required for vLLM when profiling is enabled |

**SGLang-specific env vars (set in Magpie YAML `envs`):**

| Env var | Value | Purpose |
|---------|-------|---------|
| `SGLANG_PROFILE_WITH_STACK` | `true` | Enable callstack recording in traces |
| `SGLANG_PROFILE_RECORD_SHAPE` | `true` | Enable tensor shape recording in traces |
| `PROFILE_EXTRA_BODY` | JSON string | Override `extra_body` in InferenceX benchmark_serving.py start_profile request |

**SGLang server args (append to `EXTRA_SGLANG_ARGS`):**
- `--enable-profile-cuda-graph` — enable graph capture tracing
- `--enable-shape-discovery-for-cuda-graph-profile` — shape discovery for captured graphs

**vLLM server args (append to `EXTRA_VLLM_ARGS`):**
- `--profiler-config.capture_torch_profiler_dir <dir>` — graph capture trace output directory
- `--profiler-config.detailed_trace_annotation True` — enable detailed annotations
- `--profiler-config.delay_iterations <N>` — start profiling after N steps
- `--profiler-config.max_iterations <N>` — profile for N steps
- `--profiler-config.ignore_frontend True` — skip frontend profiling overhead

## Magpie Integration

All throughput benchmarks, profiling, and parameter sweeps use **Magpie** (`magpie benchmark`
CLI) as the sole execution engine. Magpie wraps InferenceX with structured results, built-in
TraceLens trace analysis, and gap analysis.

**Prerequisites — Install Magpie:**

`actions/setup.md` Step 2 resolves `MAGPIE_PATH` with the following priority and then
mirrors it to `/tmp/Magpie` when the source is read-only:

1. Caller-provided `MAGPIE_PATH` env var (must point to an existing directory)
2. `/hyperloom/users/8cf535bc3ad11fa15e48157cf3b3f726/Magpie` (current workspace checkout)
3. First match of `/shared_nfs/*/Magpie` (per-user NFS layout)
4. Legacy default `/shared_nfs/Magpie`

After resolution `pip install -e "$MAGPIE_PATH"` runs once if `magpie` is not yet on PATH,
and a `from Magpie.modes.benchmark.config import SweepMatrix` sanity check warns when the
installed Magpie predates `sweep_matrix` (required by `actions/sweep.md`).

```bash
# Already handled by setup.md _resolve_magpie_path + _mirror_if_readonly helpers.
# Manual fallback (only if setup was skipped):
if ! command -v magpie &>/dev/null; then
    cp -r "$MAGPIE_PATH" /tmp/Magpie 2>/dev/null || true
    pip install -e /tmp/Magpie 2>&1 | tail -3
fi
```

**Key pattern for all benchmarks — generate YAML config, then run:**
```bash
cat > "$RESULT_DIR/config.yaml" <<EOF
benchmark:
  framework: $FRAMEWORK
  model: $MODEL
  precision: fp8
  run_mode: local
  runner_type: $RUNNER_TYPE
  inferencex_path: $INFERENCEX_PATH
  benchmark_script: ${FRAMEWORK}_${RUNNER_TYPE}.sh
  envs:
    TP: $TP
    CONC: $CONC
    ISL: $ISL
    OSL: $OSL
    RANDOM_RANGE_RATIO: 0.5
    EXTRA_SGLANG_ARGS: "..."
    NUM_PROMPTS: ...
  timeout_seconds: 3600
  profiler:
    torch_profiler:
      enabled: false
EOF
magpie benchmark --benchmark-config "$RESULT_DIR/config.yaml" -o "$RESULT_DIR/<action_name>"
```

**CI baseline YAML template (with placeholders for `render_prompt`):**

All CI benchmark runs use this template. Install Magpie first: `pip install -e $MAGPIE_PATH`.
ALL benchmarks (baseline, DFS, profile, sweep) MUST use `magpie benchmark --benchmark-config <yaml>`.
Do NOT launch servers manually or use `run_baseline.sh` / `run_sweep.sh`.
Magpie handles server lifecycle (start, health check, benchmark, cleanup) automatically.

```yaml
benchmark:
  framework: $FRAMEWORK
  model: $MODEL
  precision: fp8
  run_mode: local
  runner_type: $RUNNER_TYPE
  inferencex_path: $INFERENCEX_PATH
  benchmark_script: ${FRAMEWORK}_${RUNNER_TYPE}.sh
  envs:
    TP: $TP
    CONC: $CONC
    ISL: $ISL
    OSL: $OSL
    RANDOM_RANGE_RATIO: 0.8
    NUM_PROMPTS: $NUM_PROMPTS
    MAX_MODEL_LEN: 4096
  profiler:
    torch_profiler:
      enabled: false
  timeout_seconds: 3600
```

**CI benchmark rules:**
- Set `RANDOM_RANGE_RATIO: 0.8` in YAML envs for ALL benchmarks. Do NOT use 1.0.
- Set `NUM_PROMPTS` equal to `CONC * 10` in YAML envs. Do NOT use `CONC * 3`.
- Read the InferenceX benchmark script to extract server configuration (attention backends,
  env vars, memory settings, expert parallelism flags). Put these as `EXTRA_{FRAMEWORK}_ARGS`
  in the YAML envs section. Do NOT run the InferenceX script directly.

**CRITICAL — Bash timeout:** Every `magpie benchmark` call includes server startup (up
to 45 min for large models). Set Bash `timeout` to at least **3600 seconds (1 hour)**.
Without this, the Claw executor kills the process before the server finishes loading.

**`benchmark_script`:** Always specify the Magpie-authored script (e.g.,
`sglang_mi355x.sh`) to ensure `EXTRA_SGLANG_ARGS` / `EXTRA_VLLM_ARGS` are respected.
Without this field, Magpie may select an InferenceX native script that ignores extra args.

**Results:** Each run produces `benchmark_report.json` in the workspace with structured
`throughput` (output_throughput, request_throughput, duration_seconds) and `latency`
(TTFT, TPOT, ITL, E2EL — each with mean/median/p99/std in ms).

**Server lifecycle:** Magpie launches and kills the server within each benchmark call.
For accuracy evaluation (GSM8K), start a dedicated server manually (see action docs).

**DFS short runs:** Set `NUM_PROMPTS: $((CONC * 3))` in the YAML `envs` to reduce
benchmark duration. Default is `CONC * 10`.

## Architecture

```
SKILL.md (this file)          — DFS orchestrator: loop, heuristic, dispatch
actions/*.md                   — Self-contained action modules (11 actions)
kernel-opt/                    — Per-backend kernel optimization references
  geak.md                      — GEAK MCP (remote GPU pod)
  codex.md                     — Codex via OOB GPU Optimizer MCP
  claude.md                    — Claude Code via OOB GPU Optimizer MCP
  llm.md                       — LLM Proxy (direct API)
kb/                            — RAG knowledge base (JSONL + query/ingest scripts)
scripts/                       — Baseline/profiling/accuracy shell scripts
modes/                         — Mode-specific execution details (LOCAL.md, CLAW.md)
KNOWLEDGE-BASE.md              — Legacy KB (archived, seeded into kb/entries.jsonl)
```

## Common Pitfalls (validated from CI logs)

These are recurring errors observed in production CI runs. **Read before executing.**

1. **PATH: Always `export PATH="/opt/venv/bin:$PATH"` first.** The system python3
   (`/usr/bin/python3`) does NOT have sglang/vllm/numpy. Every bash command must
   prepend the venv. Failure mode: `ModuleNotFoundError: No module named 'sglang'`.

2. **Never override user-specified TP.** If the prompt says TP=8, use TP=8. Do NOT
   auto-detect GPU_COUNT and override to TP=1 — large models (120B+) cannot run on
   a single GPU. Failure mode: OOM or server crash.

3. **vLLM flags differ from SGLang.** Common mistake: `--disable-log-requests` is NOT
   a valid vLLM flag. Use `--disable-log-stats` for vLLM. Always check `vllm serve --help`
   before using unfamiliar flags. Failure mode: `unrecognized arguments` → server crash.

4. **Use `magpie benchmark --benchmark-config` instead of manual server launch.** Magpie
   handles server startup, health wait, benchmark, and cleanup in a tested sequence. Manual
   launch skips health checks and often hits Exit code 144 (SIGTERM from stale processes).

5. **Never call `geak_set_model_config`.** See IR-7. GEAK LLM backend is pre-configured.

## DFS Search Tree

**Phases:** SETUP → CLASSIFY → TARGET ANALYSIS (optional) → BASELINE (+ GSM8K accuracy) → PROFILE → HEURISTIC SCORING → DFS LOOP (pick highest-scored action → execute → re-score → repeat) → SWEEP → REPORT

**DFS loop actions:** backends, params, kernel-opt, integrate, sweep — scored by heuristic, popped highest-first. Each action can push sub-actions (e.g., PROFILE pushes GEAK candidates, BACKENDS pushes combination tests). The agent explores depth-first along the most promising branch and backtracks if scores shift.

The agent is NOT limited to pre-defined actions. Ad-hoc actions can be created and scored with the same heuristic if profiling reveals unexpected bottlenecks or KB suggests novel techniques.

This is a single-agent sequential loop. Each action runs to completion before re-scoring. Parallelism is within actions (e.g., GEAK submits 5 kernels in parallel), not between them.

## Autonomy Rules

**Execute autonomously — no human confirmation needed.** Do NOT ask the user before:
- Creating/stopping RayJob on SaFE (claw mode)
- Running baseline/profiling scripts via Ray (claw mode) or locally
- Submitting GEAK tasks
- Killing/restarting servers (inside RayJob or locally)
- Patching kernels (Inductor cache or source files)
- Reverting failed patches

**Autonomy means don't ask permission, NOT skip steps.** Every numbered step in the
Orchestrator Loop (1–11) is **MANDATORY**, including:
- Step 3: TARGET ANALYSIS (if target data provided)
- Step 4: KB WARM-UP (always — query KB before proceeding to baseline)
- Step 11: KNOWLEDGE HOOK (always — ingest findings after report)

Skipping any mandatory step invalidates the run. Present the **final optimization report**
to the user once all steps are complete.

## Heuristic Scoring Function

Every candidate action is scored by:

```
score = (expected_tput_gain_per_gpu / cost_minutes)
        × (1 - accuracy_risk)
        × (1 - crash_risk)
        × target_gap_multiplier
```

| Component | Source | Range |
|-----------|--------|-------|
| `expected_tput_gain_per_gpu` | KB lookup + model class priors | 0–100+ tok/s/GPU |
| `cost_minutes` | Estimated wall-clock time | 2–120 min |
| `accuracy_risk` | From KB (kernel mods = 0.15, backends = 0.1, params = 0.0) | 0.0–1.0 |
| `crash_risk` | From KB (vendor kernel mods = 0.5, scheduling = 0.05) | 0.0–1.0 |
| `target_gap_multiplier` | `1 + min(target_gap_pct, 100) / 100` | 1.0–2.0 |

### Initial Score Priors (by model class)

| Action | Dense | MoE+MLA | MoE+SWA | MoE+MLA+NSA |
|--------|-------|---------|---------|-------------|
| backends | 3 | **9** | **8** | **10** |
| params | 5 | 6 | 7 | 5 |
| kernel-opt (GEAK) | **8** | 2 | 2 | 2 |
| torch.compile | **7** | 0 | 0 | 0 |
| sweep | 1 | 1 | 1 | 1 |

Scores update after each action based on measured results.

### Score Update Rules

After each action completes:

1. **Action succeeded (gain > 0%):** Boost similar actions. E.g., if `backends` gained +5%,
   boost remaining untested backends by 1.5×. Boost `combined_test` score.
2. **Action failed (gain ≤ 0%):** Reduce similar actions by 0.5×.
3. **After 2+ backend wins:** Push `combined_backends_test` with score = sum(individual scores) × 1.5
4. **After all backends tested:** Push `re-profile` (to discover new GEAK targets)
5. **After kernel opt kept:** Push `re-profile + next-kernel` with boosted score
6. **After kernel opt discarded:** Reduce remaining kernel scores by 0.7×
7. **When all action scores < 1.0:** Proceed to sweep → report

## State Schema

The orchestrator maintains this state throughout the run:

```python
state = {
    "model_name": "",
    "model_class": "",           # dense / moe_mla / moe_swa / moe_mla_nsa
    "framework": "sglang",
    "tp": 8,
    "gpu_type": "MI355X",

    "baseline_tput_per_gpu": 0.0,
    "current_tput_per_gpu": 0.0,
    "cumulative_gain_pct": 0.0,

    "target_tput_per_gpu": None,  # from target-analysis, if available
    "target_gap_pct": None,

    "torch_compile_status": None,  # success / failed / skipped
    "accuracy_reference": None,    # path to reference output
    "baseline_accuracy": None,     # GSM8K exact_match score (0.0–1.0) from baseline eval
    "accuracy_threshold": 0.01,    # max allowed accuracy drop (absolute) before REVERT

    "action_stack": [],            # priority stack of (score, action_name, params)
    "completed_actions": [],       # log of (action_name, gain_pct, status)
    "kernel_candidates": [],       # from profiling
    "winning_backends": [],        # from backend exploration
    "winning_params": [],          # from param tuning

    "total_wall_minutes": 0,
    "total_geak_submissions": 0,
    "consecutive_discards": 0,
}
```

## Orchestrator Loop

```
PROCEDURE optimize():

  1. SETUP
     → Execute actions/setup.md
     → Set MODEL, TP, CONC, FRAMEWORK, paths

  2. CLASSIFY
     → Execute actions/classify.md
     → Set model_class, torch_compile_viable, initial score priors

  3. TARGET ANALYSIS (if $TARGET_DIR provided)
     → Execute actions/target-analysis.md
     → Set target_tput_per_gpu, target_gap_pct, target_gap_multiplier

  4. KB WARM-UP
     → Query KB for this model: python3 kb/kb_query.py --model "$MODEL_NAME" --top-k 20
     → Apply KB-informed adjustments to score priors

  5. BASELINE
     → Execute actions/baseline.md
     → Set baseline_tput_per_gpu, torch_compile_status, accuracy_reference
     → Run GSM8K eval → set baseline_accuracy (MANDATORY — this is the accuracy floor)

  6. PROFILE
     → Execute actions/profile.md
     → Populate kernel_candidates with (name, gpu_pct, source)

  7. BUILD ACTION STACK
     → Score all candidate actions using the heuristic
     → Push onto action_stack sorted by score (highest first)

  8. DFS LOOP:
     WHILE action_stack is not empty AND NOT stopping_criteria_met():
       a. Pop highest-scored action
       b. Execute the action (dispatch to actions/*.md)
       c. ACCURACY GATE: if action has accuracy_risk > 0:
          - Run GSM8K eval via scripts/eval_accuracy.sh
          - Compare new score against state.baseline_accuracy
          - If drop > accuracy_threshold (default 0.01): REVERT, mark FAIL
          - See "Accuracy Gate Protocol" section below
       d. Measure result: new_tput_per_gpu
       e. Update state: current_tput_per_gpu, cumulative_gain_pct
       f. RE-SCORE all remaining actions (gains shift after each optimization)
       g. Push any new sub-actions discovered during execution
       h. Log to completed_actions

  9. SWEEP
     → Execute actions/sweep.md (full ISL/OSL/CONC parameter sweep)

 10. REPORT
     → Execute actions/report.md (generate optimization report + KB contribution)

 11. KNOWLEDGE HOOK
     → The .cursor/hooks/knowledge-sink.py hook fires automatically
     → Ingests any new knowledge discovered during the run
```

## Accuracy Gate Protocol

Actions are gated by their `accuracy_risk` value. **Baseline GSM8K accuracy is measured
during step 5 (BASELINE) and stored in `state.baseline_accuracy`.** Every subsequent action
with `accuracy_risk > 0` must pass the accuracy gate before KEEP.

### Which actions trigger the gate

| accuracy_risk | Actions | Gate required |
|:-------------:|---------|:-------------:|
| 0.0 | Server scheduling params (decode-steps, cuda-graph-max-bs, mem-fraction, chunked-prefill) | No |
| 0.05–0.15 | Kernel modifications (GEAK), GEMM tuning | **Yes** |
| 0.1 | Backend switches (aiter, alter, attention backends) | **Yes** |
| 0.3 | Precision-affecting params (kv-cache-dtype fp8, quantization changes) | **Yes** |

### Gate procedure

For any action with `accuracy_risk > 0`, after the throughput benchmark succeeds:

1. **Run GSM8K eval** against the running server using InferenceX's lm-evaluation-harness:
   ```bash
   EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
     RESULTS_DIR="$RESULT_DIR/eval_gsm8k_${ACTION_NAME}" \
     bash "$SKILL_ROOT/scripts/eval_accuracy.sh"
   ```

2. **Extract the score** from the eval summary:
   ```bash
   new_accuracy=$(python3 -c "
   import json, glob
   f = sorted(glob.glob('$RESULT_DIR/eval_gsm8k_${ACTION_NAME}/eval_summary_gsm8k.json'))[-1]
   d = json.load(open(f))
   scores = list(d['scores'].values())[0]
   print(scores.get('exact_match,strict-match', scores.get('exact_match,none', 0)))
   ")
   ```

3. **Compare with baseline:**
   ```
   accuracy_drop = baseline_accuracy - new_accuracy
   if accuracy_drop > accuracy_threshold (default 0.01 = 1 percentage point):
       REVERT immediately
       Log to KB: accuracy_risk=1.0 for this action+model
       Mark action as FAIL (accuracy degradation)
   else:
       KEEP — accuracy within tolerance
   ```

### Kernel-level pre-check (optional, for GEAK/kernel mods only)

Before the full GSM8K eval, a fast micro-benchmark sanity check can catch obvious breakage:
```python
assert torch.allclose(original_output, optimized_output, atol=1e-3, rtol=1e-3)
```
This does NOT replace the GSM8K gate — it's an early-exit optimization.

### Actions that skip the gate

**setup, classify, profile, sweep, report** — these are read-only and never modify the
serving computation path. Pure scheduling params (accuracy_risk=0.0) also skip.

## Stopping Criteria

| Condition | Action |
|-----------|--------|
| All action scores < 1.0 | Proceed to sweep |
| Cumulative gain > 25% | Proceed to sweep |
| 5 consecutive discards across all actions | Proceed to sweep |
| Wall clock > 180 min total | Proceed to sweep |
| Target exceeded (gap ≤ 0%) | Proceed to sweep |
| 2+ server crashes | Emergency stop, report partial results |

## KB Integration

Before each action, query the KB for relevant knowledge:

```bash
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME $ACTION_NAME" --top-k 5 --compact
```

After each action with new findings, ingest into KB:

```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category $CATEGORY --model "$MODEL_NAME" \
    --action "$WHAT_WAS_DONE" --lesson "$KEY_TAKEAWAY" \
    --tags $TAGS --gain $GAIN --status $STATUS
```

The KB provides:
- **Prior knowledge:** Skip actions already known to fail for this model class
- **Score calibration:** Adjust expected gains based on past results
- **Conflict detection:** Flag contradictory information automatically

## Action Dispatch

| Action | Module | When |
|--------|--------|------|
| Setup | [`actions/setup.md`](actions/setup.md) | Always first |
| Classify | [`actions/classify.md`](actions/classify.md) | Always second |
| Target Analysis | [`actions/target-analysis.md`](actions/target-analysis.md) | If `$TARGET_DIR` provided |
| Baseline | [`actions/baseline.md`](actions/baseline.md) | After classify |
| Profile | [`actions/profile.md`](actions/profile.md) | After baseline |
| Backend Exploration | [`actions/backends.md`](actions/backends.md) | DFS loop |
| Server Params | [`actions/params.md`](actions/params.md) | DFS loop |
| Kernel Optimization | [`actions/kernel-opt.md`](actions/kernel-opt.md) | DFS loop |
| Integration | [`actions/integrate.md`](actions/integrate.md) | Per-kernel sub-action |
| Parameter Sweep | [`actions/sweep.md`](actions/sweep.md) | After DFS loop |
| Report | [`actions/report.md`](actions/report.md) | Always last |

## Reference: Critical Lessons

These are the most important validated lessons. Full details in KB and action modules.

1. **Backend switches outperform parameter sweeps.** GLM-5: backends +16.2% vs params <1%.
   Always explore backends BEFORE sweeping parameters.

2. **Combination synergies can be super-linear.** Two +3% backends → +16.2% combined.
   Always test winners together.

3. **torch.compile is prerequisite for large GEAK wins.** With compile: up to +14.72%.
   Without: ≤1.76%.

4. **Benchmark fairness is critical.** Kimi-K2.5 "+40.4%" was invalid — actually +0.81%
   after controlling for CONC mismatch. Always save baseline config and reuse.

5. **Server param tuning can dominate.** Kimi vLLM +84% from gpu-mem + max-num-seqs.
   CUDA graph coverage +35% when misconfigured.

6. **GEAK cannot beat vendor kernels.** Never submit `Cijk_*` or `aiter::*` to GEAK.

7. **MUST patch STANDALONE files, not graph modules.** Graph module patching = 0%.
   Standalone patching = +9%.

8. **Use Python AST for source patching.** Naive regex deletes module-level variables.

## Reference: Process Management

- **Never use `pkill -f "sglang.launch_server"` inside scripts** — kills the script itself.
- **Wait `SERVER_KILL_WAIT_S` seconds** (default 10) between server kill and relaunch.
- **Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR`** after profiling.
- **Always use filtered traces** for TraceLens (raw: 349MB, filtered: 5MB).
- TraceLens does NOT support `rocprofv3` format — only PyTorch Kineto.

## Reference: Benchmark Metrics

Metrics come from Magpie's `benchmark_report.json` (structured JSON):

| Metric | JSON Path | Unit | Meaning |
|--------|-----------|------|---------|
| `output_throughput` | `throughput.output_throughput` | tok/s | Output tokens per second |
| `tput_per_gpu` | derived: `throughput.output_throughput / TP` | tok/s/GPU | Primary optimization target |
| `request_throughput` | `throughput.request_throughput` | req/s | Requests per second |
| `completed_requests` | `throughput.completed_requests` | count | Successfully completed requests |
| `mean_tpot_ms` | `latency.tpot.mean_ms` | ms | Time Per Output Token (decode latency) |
| `p99_tpot_ms` | `latency.tpot.p99_ms` | ms | P99 decode latency |
| `mean_ttft_ms` | `latency.ttft.mean_ms` | ms | Time to First Token (prefill latency) |
| `p99_ttft_ms` | `latency.ttft.p99_ms` | ms | P99 prefill latency |
| `mean_itl_ms` | `latency.itl.mean_ms` | ms | Inter-Token Latency |
| `mean_e2el_ms` | `latency.e2el.mean_ms` | ms | End-to-End Latency |

### Extracting metrics from Magpie results

```bash
WORKSPACE=$(ls -td "$RESULT_DIR"/<action>/benchmark_* | head -1)
python3 -c "
import json
d = json.load(open('$WORKSPACE/benchmark_report.json'))
tp = $TP
print(f'output_throughput: {d[\"throughput\"][\"output_throughput\"]:.2f} tok/s')
print(f'tput_per_gpu: {d[\"throughput\"][\"output_throughput\"]/tp:.2f} tok/s/GPU')
print(f'TPOT mean: {d[\"latency\"][\"tpot\"][\"mean_ms\"]:.2f} ms')
print(f'TTFT mean: {d[\"latency\"][\"ttft\"][\"mean_ms\"]:.2f} ms')
"
```

## Reference: Server Parameter Tables

See [`KNOWLEDGE-BASE.md`](KNOWLEDGE-BASE.md) for validated parameter tables (SGLang, vLLM)
and model-specific configurations. Query the KB for the latest:

```bash
python3 $SKILL_ROOT/kb/kb_query.py --category server_params --compact
```

## Reference: vLLM Integration

All scripts support `FRAMEWORK=vllm`. Parameter mapping:

| SGLang | vLLM | Notes |
|--------|------|-------|
| `--model-path` | `vllm serve <model>` (positional) | — |
| `--mem-fraction-static 0.8` | `--gpu-memory-utilization 0.85` | — |
| `--disable-radix-cache` | `--no-enable-prefix-caching` | For random benchmarks |
| `--enable-torch-compile` | Default ON (level=3) | Disable: `--enforce-eager` |
| `SGLANG_TORCH_PROFILER_DIR` | `VLLM_TORCH_PROFILER_DIR` | Set before server launch |
