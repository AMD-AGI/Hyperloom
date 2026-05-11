---
name: inference-optimization
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

This skill supports one execution mode. **Read the remote-mode document before starting:**

- **Remote mode only** (SaFE RayJob + Ray Dashboard REST/MCP control plane): see [`modes/REMOTE.md`](modes/REMOTE.md)

There is no local-mode path in this skill. All GPU-side work runs inside a newly created RayJob for this skill execution; never attach to or reuse an existing RayJob.

## Iron Rules (non-negotiable)

These rules apply to ALL modes. Violating any invalidates the optimization run.

### IR-1: Submit ALL selected kernel candidates through OOB

The kernel-opt action MUST submit `OOB_TOP_CANDIDATES` (default 5) candidates to all active OOB agents in `KERNEL_OPT_BACKENDS` (`codex`, `claude`) **simultaneously** where capacity allows. Submitting only 1 candidate when multiple exist, or submitting OOB agents sequentially when both are enabled = violation.

### IR-2: NEVER modify kernel source before OOB submission

Submit the kernel source **exactly as extracted**. Do NOT strip decorators, change strides, replace `@triton_heuristics` with `@triton.jit`, or make any "cleanup" edits. OOB agents handle adaptation in their isolated workspaces.

### IR-3: Integration (Phase 8) is MANDATORY

After OOB returns optimized kernels, you MUST execute the integrate action (patch → re-baseline → decide). Skipping means OOB results are never validated end-to-end. Re-baseline uses `run_baseline.sh` — there is no `run_benchmark.sh`. See `actions/integrate.md` for details.

### IR-4: Always kill_server + check_gpu_memory before server launch

Every server launch must be preceded by killing any existing server process and verifying GPU memory is free.

### IR-5: Safe process management

**NEVER use `pkill -f sglang`** — it kills Ray workers in remote mode. Only use:

```bash
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
# or for vLLM:
kill $(pgrep -f 'python.*-m vllm.entrypoints') 2>/dev/null
```

Wait `SERVER_KILL_WAIT_S` seconds between kill and relaunch. Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR` after profiling.

### IR-6: Use `patch_inductor.py --target-file` for Inductor patching

Always use `scripts/patch_inductor.py` with `--target-file`. The `--cache-dir` option has been removed.

**CRITICAL:** When an OOB optimized kernel changes block sizes or warp counts, you MUST also pass `--best-config` with the updated tiling parameters. Patching only the kernel `.py` without updating `.best_config` causes numerical corruption (garbled output). See `actions/integrate.md` for details.

### IR-7: OOB is the ONLY kernel optimization backend

Kernel optimization MUST use OOB via `oob_ray_submit.py run` inside the RayJob.
Do not use any non-OOB backend, direct LLM API calls, ad-hoc scripts, or alternate kernel-optimization clients.
See [`modes/REMOTE.md`](modes/REMOTE.md).

Treat OOB configuration as **read-only infrastructure**. The skill MUST NOT
modify any OOB configuration files, settings, or parameters at runtime. Specifically:

- **Do NOT** modify OOB auth config, workspace settings, or API configuration
- **Do NOT** write to or alter any files under OOB config/settings directories except through bootstrap
- **Do NOT** change `KERNEL_OPT_WORKSPACE`, `OOB_ROUND_ITERATIONS`, or other constants
  at runtime (use the values from the constants table above or user overrides)
- **Do NOT** bypass OOB by calling `openai.Client`, `curl /messages`, or a direct LLM proxy from the orchestrator

The ONLY allowed kernel-optimization interaction is through `oob_ray_submit.py run`,
which invokes `oob run -a codex` or `oob run -a claude` under RayJob execution.

Violation (changing backend or bypassing OOB) = immediate run invalidation.

### IR-7b: Orchestrator MUST NOT write kernel optimization code itself

All kernel optimization MUST go through the configured `KERNEL_OPT_BACKENDS`
(`codex`, `claude`). The orchestrator Agent's role is to **prepare
prompts, submit tasks to backends, verify results, and integrate** — NEVER to
directly author optimized Triton/HIP/CUDA kernels.

Even if the orchestrator is the same LLM model as a backend (e.g., Claude
orchestrating via OOB Claude), it MUST still use the backend toolchain. Backends
provide isolated workspaces, GPU-side validation, reproducible trajectories, and
Ray-managed GPU scheduling that direct in-chat generation lacks.

Violation = immediate run invalidation.

**Additional remote-mode Iron Rules are defined in [`modes/REMOTE.md`](modes/REMOTE.md) (IR-8 through IR-11).**

## Kernel Optimization & Tooling Constants

All values below are the **single source of truth**. All actions reference these by name.

| Constant | Value | Description |
|----------|-------|-------------|
| `KERNEL_OPT_BACKENDS` | `codex,claude` | Comma-separated active OOB agents. Allowed values: `codex`, `claude`. User can override in prompt. |
| `OOB_ROUND_ITERATIONS` | 3 | Iterations per Codex/Claude round (submit → RayJob benchmark/validation → feedback → re-submit). Best result wins. |
| `KERNEL_OPT_IMAGE` | *(provided by CI or user)* | Framework image used by the RayJob where OOB runs. One image per run, determined by framework (SGLang/vLLM). |
| `KERNEL_OPT_WORKSPACE` | `control-plane-moe` | SaFE workspace for the RayJob and OOB output. User can override. |
| `OOB_MAX_RETRIES` | 3 | Max OOB submission retries per kernel |
| `OOB_TOP_CANDIDATES` | 5 | Number of top kernel/code candidates to submit |
| `OOB_CONSECUTIVE_DISCARDS` | 5 | Stop after this many consecutive discards |
| `OOB_WALL_CLOCK_MIN` | 30 | Max wall-clock minutes for kernel-opt action |
| `OOB_POLL_INTERVAL_S` | 60 | Seconds between OOB task status polls when external polling is needed |
| `OOB_POLL_TIMEOUT_MIN` | 15 | Max minutes to poll a single OOB task when external polling is needed |
| `MIN_GPU_PCT` | 3 | Minimum GPU time % to consider a kernel/code path as an OOB candidate |
| `SERVER_KILL_WAIT_S` | 10 | Seconds to wait between server kill and relaunch |
| `FILTERED_TRACE_NAME` | `filtered-TP-0.trace.json.gz` | Preferred trace file for TraceLens analysis |

**All OOB work runs inside the RayJob image.** For kernels whose source exists in the image (e.g., `/sgl-workspace/aiter/`), include the pod-visible path in the prompt. For runtime-generated kernels (e.g., `/tmp/torchinductor_root/` from `torch.compile`), do NOT include `kernel_url`/`kernel_repo` in the prompt; copy files to shared NFS or rely on `files[].content` only.

**Remote-mode constants are in [`modes/REMOTE.md`](modes/REMOTE.md).**

## Path Ownership Invariant

Remote mode has two write domains:

- **RayJob writes:** all benchmark results, traces, TraceLens raw outputs, OOB
  workspaces, kernel candidates, re-baseline artifacts, and sweep outputs
  produced inside the RayJob MUST use pod-visible `/wekafs/...` paths.
- **Sandbox writes:** the Claw sandbox must treat `/wekafs/...` as read-only
  input. Any sandbox-created file must be under `/workspace/hyperloom/`.
- **Final deliverables:** the final optimization report, CI metrics, manifest,
  and result summaries MUST be present under `/workspace/hyperloom/`, even if
  they point back to raw RayJob artifacts stored under `/wekafs/...`.

## Architecture

```
SKILL.md (this file)          — DFS orchestrator: loop, heuristic, dispatch
actions/*.md                   — Self-contained action modules (11 actions)
kernel-opt/                    — Per-backend kernel optimization references
  codex.md                     — Codex via OOB Ray-scheduled CLI
  claude.md                    — Claude Code via OOB Ray-scheduled CLI
kb/                            — RAG knowledge base (JSONL + query/ingest scripts)
scripts/                       — Baseline/profiling/accuracy shell scripts
modes/                         — Remote-mode execution details (REMOTE.md)
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

4. **Use `run_baseline.sh` instead of manual server launch.** The script handles
   server startup, health wait, benchmark, and profiling in a tested sequence. Manual
   launch skips health checks and often hits Exit code 144 (SIGTERM from stale processes).

5. **Never bypass OOB.** See IR-7. Kernel optimization is invoked only via
   `oob_ray_submit.py run` inside the RayJob.

6. **Record start/end timestamps for ALL external calls.** Before invoking
   any external component (OOB, TraceLens, or future approved backends),
   run `python3 $SCRIPTS_DIR/trace_action.py --component <name> --action start`.
   After the component finishes, run `--action end`. This enables per-message cost
   attribution. If the specific backend skill already includes tracing steps, follow
   those. If not, apply this rule as a fallback. Failure to trace does NOT block
   execution — skip if the script is unavailable.

7. **PD/MoRI profiling defaults to immediate capture with explicit output.** Set
   `SGLANG_TORCH_PROFILER_DIR=$TRACE_DIR` before launching the profiled backend,
   pass `/start_profile` an `output_dir` equal to `$TRACE_DIR`, and do not pass
   `start_step` / `num_steps` by default. After `/stop_profile`, poll `$TRACE_DIR`
   until trace count and total size are stable; if empty, inspect the backend log
   for `Traces are saved to:` before rerunning the profile flow.

## DFS Search Tree

**Phases:** SETUP → CLASSIFY → TARGET ANALYSIS (optional) → BASELINE (+ GSM8K accuracy) → PROFILE → HEURISTIC SCORING → DFS LOOP (pick highest-scored action → execute → re-score → repeat) → SWEEP → REPORT

**DFS loop actions:** backends, params, kernel-opt, integrate, sweep — scored by heuristic, popped highest-first. Each action can push sub-actions (e.g., PROFILE pushes OOB candidates, BACKENDS pushes combination tests). The agent explores depth-first along the most promising branch and backtracks if scores shift.

The agent is NOT limited to pre-defined actions. Ad-hoc actions can be created and scored with the same heuristic if profiling reveals unexpected bottlenecks or KB suggests novel techniques.

This is a single-agent sequential loop. Each action runs to completion before re-scoring. Parallelism is within actions (e.g., OOB submits candidates to Codex/Claude concurrently), not between them.

## Autonomy Rules

**Execute autonomously — no human confirmation needed.** Do NOT ask the user before:
- Creating/stopping RayJob on SaFE (remote mode)
- Running baseline/profiling scripts inside the RayJob
- Submitting OOB tasks
- Killing/restarting servers inside the RayJob
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
| kernel-opt (OOB) | **8** | 2 | 2 | 2 |
| torch.compile | **7** | 0 | 0 | 0 |
| sweep | 1 | 1 | 1 | 1 |

Scores update after each action based on measured results.

### Score Update Rules

After each action completes:

1. **Action succeeded (gain > 0%):** Boost similar actions. E.g., if `backends` gained +5%,
   boost remaining untested backends by 1.5×. Boost `combined_test` score.
2. **Action failed (gain ≤ 0%):** Reduce similar actions by 0.5×.
3. **After 2+ backend wins:** Push `combined_backends_test` with score = sum(individual scores) × 1.5
4. **After all backends tested:** Push `re-profile` (to discover new OOB targets)
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
    "total_oob_submissions": 0,
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
| 0.05–0.15 | OOB-generated kernel/code modifications, GEMM tuning | **Yes** |
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

### Kernel-level pre-check (optional, for OOB kernel/code mods only)

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

3. **torch.compile exposes more Triton surface for OOB optimization.** With compile,
   runtime-generated Inductor kernels can be extracted and submitted through OOB;
   without compile, focus OOB on framework source, scheduling, and integration changes.

4. **Benchmark fairness is critical.** Kimi-K2.5 "+40.4%" was invalid — actually +0.81%
   after controlling for CONC mismatch. Always save baseline config and reuse.

5. **Server param tuning can dominate.** Kimi vLLM +84% from gpu-mem + max-num-seqs.
   CUDA graph coverage +35% when misconfigured.

6. **Do not treat vendor kernels as directly editable unless source is available.**
   For `Cijk_*` or `aiter::*`, prefer framework scheduling/config changes or source-backed OOB tasks.

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

| Metric | Unit | Meaning |
|--------|------|---------|
| `output_throughput` | tok/s | Output tokens per second |
| `tput_per_gpu` | tok/s/GPU | `output_throughput / TP` |
| `mean_tpot_ms` | ms | Time Per Output Token (decode latency) |
| `mean_ttft_ms` | ms | Time to First Token (prefill latency) |

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
