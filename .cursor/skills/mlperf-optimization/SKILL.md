---
name: mlperf-optimization
description: |
  Autonomous DFS-guided MLPerf training optimization for GPT-OSS-20B on AMD MI355X GPUs.
  Uses heuristic-scored depth-first search to systematically explore optimization actions
  (fusion flags, parallelism configs, training params, kernel optimization, runtime tunables)
  and minimize time-to-train (wall seconds to reach validation loss target) with MLPerf
  compliance as a hard constraint.
globs:
  - "**/mlperf*"
  - "**/primus*"
  - "**/megatron*"
  - "**/torchrun*"
  - "**/training_optimization/mlperf*"
---

# MLPerf Training Optimization — DFS Orchestrator

## Overview

This skill optimizes MLPerf GPT-OSS-20B training time-to-target on AMD MI355X GPUs.
It runs a **depth-first search** over optimization actions, guided by a heuristic scoring
function. The search is fully autonomous — no human prompting required.

**Primary objective:** minimize `time_to_train` (wall seconds from `run_start` to `run_stop`
with status `success`, reaching validation loss target)
**Hard constraint:** validation log perplexity must reach target (3.34) — run is invalid otherwise
**Hard constraint:** MLPerf compliance logging must be preserved (:::MLLOG format)
**Secondary metric:** `ms_per_iter` (derived from consecutive MLLOG train_loss timestamps)
**Optional target:** if a prior run or external baseline is provided, the target gap acts
as an urgency multiplier on all action scores.

## MLPerf Benchmark Context

- **Benchmark:** GPT-OSS-20B (gpt-oss-20b)
- **Model:** 20B parameter GPT with Mixture of Experts (32 experts, top-k=4)
- **Dataset:** c4/en/3.0.1 (pre-tokenized, ~80GB)
- **Quality target:** validation log perplexity = 3.34
- **Eval frequency:** every 12,288 samples (eval_interval adjusts with GBS)
- **Eval size:** 1024 sequences (64 eval iters × MBS × GPUs)
- **Precision:** BF16 / FP8 hybrid (E4M3 activations/weights, E5M2 gradients)
- **Ruleset:** MLPerf Training 5.1.0

## Architecture

```
SKILL.md (this file)                — DFS orchestrator: loop, heuristic, dispatch
actions/*.md                         — Self-contained action modules (12 actions)
kb/                                  — RAG knowledge base (JSONL + query/ingest scripts)
scripts/common.sh                    — Tiered trial runner + metric extraction helpers
scripts/trial_monitor.py             — Stdin log filter + progress display + anomaly detection
scripts/apply_quiet_config.sh        — YAML quiet/restore for noise reduction
scripts/run_baseline.sh              — Standalone baseline script
scripts/run_sweep.sh                 — GBS × LR sweep script
scripts/run_trial.sh                 — CLI wrapper for run_mlperf_trial
scripts/run_profile.sh               — Profiling run script
```

## DFS Search Tree

```
                        ┌──────────┐
                        │  SETUP   │
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │ CLASSIFY │
                        └────┬─────┘
                             │
                   ┌─────────▼─────────┐
                   │  TARGET ANALYSIS   │ ← optional, if target provided
                   └─────────┬─────────┘
                             │
                        ┌────▼─────┐
                        │ BASELINE │
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │ PROFILE  │
                        └────┬─────┘
                             │
              ┌──────────────▼──────────────┐
              │      HEURISTIC SCORING      │
              │   score each candidate action│
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   PICK HIGHEST-SCORED ACTION │
              │                              │
              │  ┌──────────┐ ┌───────────┐ │
              │  │ FUSION   │ │PARALLELISM│ │
              │  │ FLAGS    │ │           │ │
              │  └────┬─────┘ └─────┬─────┘ │
              │       │             │        │
              │  ┌────▼────┐  ┌────▼─────┐  │
              │  │ PARAMS  │  │ KERNEL   │  │
              │  │         │  │  OPT     │  │
              │  └────┬────┘  └────┬─────┘  │
              │       │            │         │
              │  ┌────▼────┐ ┌────▼─────┐   │
              │  │RUNTIME  │ │ HYPERP   │   │
              │  │TUNABLES │ │  TUNE    │   │
              │  └────┬────┘ └────┬─────┘   │
              │       │           │          │
              │  ┌────▼────┐ ┌───▼──────┐   │
              │  │INTEGRATE│ │  SWEEP   │   │
              │  └─────────┘ └──────────┘   │
              └──────────────┬──────────────┘
                             │
                      ┌──────▼──────┐
                      │  RE-SCORE   │ ← update heuristic, loop back
                      └──────┬──────┘
                             │
                     ┌───────▼───────┐
                     │ STOPPING MET? │
                     └───────┬───────┘
                             │ yes
                        ┌────▼────┐
                        │ REPORT  │
                        └─────────┘
```

**How the DFS works:** The orchestrator maintains a **priority stack** of candidate actions.
After each action completes, the stack is re-scored and the highest-scored action is popped
next. This is DFS because each action can push new sub-actions (e.g., PROFILE pushes
GEAK candidates, FUSION-FLAGS pushes combination tests). The agent explores depth-first
along the most promising branch, but can backtrack if scores shift.

**Exploration beyond the tree:** The agent is NOT limited to the pre-defined actions. If
profiling reveals an unexpected bottleneck or a KB query suggests a novel technique, the
agent can create ad-hoc actions and score them with the same heuristic. The tree above is
the default starting structure — the agent should actively look for opportunities outside it.

## Heuristic Scoring Function

Every candidate action is scored by:

```
score = (expected_time_reduction_pct / cost_minutes)
        × (1 - convergence_risk)
        × (1 - crash_risk)
        × target_gap_multiplier
```

| Component | Source | Range |
|-----------|--------|-------|
| `expected_time_reduction_pct` | KB lookup + model class priors | 0–30% |
| `cost_minutes` | Estimated wall-clock for one trial | 3–60 min |
| `convergence_risk` | Likelihood of failing to reach 3.34 target | 0.0–1.0 |
| `crash_risk` | From KB (parallelism changes = 0.3, env vars = 0.05) | 0.0–1.0 |
| `target_gap_multiplier` | `1 + min(target_gap_pct, 100) / 100` | 1.0–2.0 |

### Initial Score Priors (GPT-OSS-20B MoE on MI355X)

| Action | Score | Rationale |
|--------|-------|-----------|
| fusion-flags | **9** | Highest impact for MoE: permute fusion, GA fusion |
| hyperparams | **8** | GBS, LR, warmup can dramatically change time-to-target |
| parallelism | 6 | EP/TP/DeepEP configuration |
| runtime-tunables | 5 | System-level knobs (NUMA, hugepages, NCCL) |
| params (training) | 5 | MBS, recompute, overlap flags |
| kernel-opt (GEAK) | 3 | MoE model → limited kernel opt opportunity |
| sweep | 1 | Final exploration of operating points |

Scores update after each action based on measured results.

### Score Update Rules

After each action completes:

1. **Action succeeded (gain > 0%):** Boost similar actions. E.g., if `fusion-flags` gained
   +1.5%, boost remaining untested fusion flags by 1.5×.
2. **Action failed (gain ≤ 0%):** Reduce similar actions by 0.5×.
3. **After 2+ fusion flag wins:** Push `combined_fusion_test` with score = sum(individual) × 1.5
4. **After all fusion flags tested:** Push `re-profile` (to discover new kernel targets)
5. **After kernel opt kept:** Push `re-profile + next-kernel` with boosted score
6. **After kernel opt discarded:** Reduce remaining kernel scores by 0.7×
7. **When all action scores < 1.0:** Proceed to sweep → report

## State Schema

The orchestrator maintains this state throughout the run:

```python
state = {
    "model_name": "GPT-OSS-20B",
    "model_class": "moe_gqa",           # MoE + GQA + SWA
    "framework": "primus",
    "num_gpus": 8,
    "tp": 1, "pp": 1, "ep": 1,          # parallelism config (from config)
    "gpu_type": "MI355X",

    "baseline_ms_per_iter": 0.0,
    "baseline_time_to_train": 0.0,       # wall seconds (MLLOG run_start → run_stop)
    "current_ms_per_iter": 0.0,
    "current_time_to_train": 0.0,
    "cumulative_gain_pct": 0.0,

    "global_batch_size": 32,             # can be tuned (unlike training-optimization)
    "micro_batch_size": 2,
    "seq_length": 8192,
    "eval_interval": 384,                # iterations between eval

    "target_eval_loss": 3.34,            # MLPerf quality target
    "baseline_eval_loss": None,          # from baseline run
    "fp8_mode": "hybrid",

    "target_time_to_train": None,        # from target-analysis, if available
    "target_gap_pct": None,

    "config_yaml": "",                   # path to training YAML
    "config_sh": "",                     # path to shell config
    "kept_overrides": [],                # accumulated config overrides
    "kept_patches": [],                  # code patches that improved perf
    "kept_env_vars": {},                 # environment variable changes

    "action_stack": [],                  # priority stack of (score, action_name, params)
    "completed_actions": [],             # log of (action_name, gain_pct, status)
    "kernel_candidates": [],             # from profiling
    "winning_flags": [],                 # from fusion flag exploration
    "winning_params": [],                # from param tuning

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
     → Set framework, config paths, GPU count
     → Setup symlinks, verify data paths, kill stale processes
     → Validate trial_monitor.py and quiet_yaml functions
     → === REVIEW CHECKPOINT RC-1 ===

  2. CLASSIFY
     → Execute actions/classify.md
     → Set model_class, initial score priors, parallelism topology

  3. TARGET ANALYSIS (if $TARGET_DIR or target numbers provided)
     → Execute actions/target-analysis.md
     → Set target_time_to_train, target_gap_pct, target_gap_multiplier

  4. KB WARM-UP
     → Query KB for this model: python3 kb/kb_query.py --model "GPT-OSS-20B" --top-k 20
     → Apply KB-informed adjustments to score priors

  5. BASELINE
     → Execute actions/baseline.md
     → Uses Tier 1 trial: run_mlperf_trial "baseline" 1
     → Parse TRIAL_RESULT for baseline_ms_per_iter, GBS verification
     → === REVIEW CHECKPOINT RC-2 ===

  6. PROFILE
     → Execute actions/profile.md (Tier 1 with profiling enabled)
     → Populate kernel_candidates with (name, gpu_pct, source)
     → === REVIEW CHECKPOINT RC-3 ===

  7. BUILD ACTION STACK
     → Score all candidate actions using the heuristic
     → Push onto action_stack sorted by score (highest first)

  8. DFS LOOP:
     SET dfs_iteration_count = 0
     WHILE action_stack is not empty AND NOT stopping_criteria_met():
       a. Pop highest-scored action
       b. Execute the action with Tier 1 trial
       c. Parse TRIAL_RESULT: check status (nan → REVERT, no_data → skip)
       d. Measure: new_ms_per_iter from TRIAL_RESULT
       e. Compute gain: compute_gain_pct(baseline, new_ms_per_iter)
       f. CONVERGENCE GATE:
          - If gain > 1%: Escalate to Tier 2 trial for convergence validation
          - If Tier 2 TRIAL_RESULT shows eval_loss diverging: REVERT
       g. Update state: current_ms_per_iter, cumulative_gain_pct
       h. RE-SCORE all remaining actions
       i. Push any new sub-actions discovered during execution
       j. Log to completed_actions
       k. INCREMENT dfs_iteration_count
       l. IF dfs_iteration_count % 3 == 0:
          → === REVIEW CHECKPOINT RC-4 ===

  9. PRE-SWEEP REVIEW
     → === REVIEW CHECKPOINT RC-5 ===
     → Execute actions/sweep.md (Tier 2 trials for GBS × MBS sweep)
     → === REVIEW CHECKPOINT RC-6 ===

 10. REPORT
     → Execute actions/report.md:
       a. Run Tier 3: run_mlperf_trial "final" 3
          - Do NOT interrupt. Wait for training to naturally finish.
          - Training ends when eval_loss reaches 3.34 (status=success)
            or all iterations are exhausted (status=aborted).
       b. === REVIEW CHECKPOINT RC-7 ===
          - Verify run_stop status from MLLOG
          - Extract time-to-train (TTT) as the primary result
          - Extract final eval_loss
       c. Generate optimization report with TTT comparison vs baseline projection

 11. KNOWLEDGE HOOK
     → Ingest any new knowledge discovered during the run via kb_ingest.py
```

## Convergence Gate Protocol (CRITICAL)

**Every action that modifies training config** must pass the convergence gate:

1. **Quick check (Tier 1):** Run 100-iteration trial. Verify loss is not NaN/Inf and ms/iter
   is measurable. If the `trial_monitor.py` emits `[ALERT] NaN`, REVERT immediately.
2. **Check loss trajectory:** The `[ITER N/100] loss=X.XX` progress lines must show decreasing loss.
3. **Convergence validation (Tier 2):** For winners (gain > 1%) or high-risk changes
   (GBS/LR/FP8), run a 100-iteration trial with eval enabled. Check that eval_loss
   is on track toward 3.34.
4. **Full verification (Tier 3):** Only for the final report — run the complete training
   to confirm the target is actually reached.

**What invalidates a run:**
- `TRIAL_RESULT` shows `status=nan` (NaN/Inf detected by trial_monitor)
- `TRIAL_RESULT` shows `status=no_data` (zero iterations completed)
- Validation loss diverging or not converging toward 3.34
- MLPerf compliance check failure (:::MLLOG lines missing or malformed)

**Actions that do NOT need convergence gate:** setup, classify, profile (read-only), 
kernel-opt (same config, different kernel code), runtime-tunables (system-level only).

## Trial Tier Protocol

All training runs during optimization use one of three trial tiers. This enables rapid
iteration without waiting for full MLPerf runs.

### Tier 1: Quick Trial (ms/iter measurement)

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | 100 |
| `PRIMUS_EVAL_INTERVAL` | 10000 (suppress eval) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | **1** (every iteration — CRITICAL for short trials) |
| `stderr_sink_level` | WARNING (quiet YAML) |
| `log_interval` | 999999 (quiet YAML) |
| Timeout | 60 minutes |
| Output | ms/iter, loss trajectory, `TRIAL_RESULT` line |

**Use for:** Initial measurement of any config change, fusion flag testing, crash detection.

### Tier 2: Convergence Trial

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | 500 |
| `PRIMUS_EVAL_INTERVAL` | 50 (triggers multiple evals) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | **1** |
| `stderr_sink_level` | WARNING (quiet YAML) |
| `log_interval` | 999999 (quiet YAML) |
| Timeout | 120 minutes |
| Output | ms/iter, loss curve, eval_loss, `TRIAL_RESULT` line |

**Use for:** Validating winners, hyperparameter tuning, FP8 stability checks, sweep.

### Tier 3: Full Verification (Run to Convergence)

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | Original config value (do NOT override) |
| `PRIMUS_EVAL_INTERVAL` | Original config value (do NOT override) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | 32 (original) |
| `stderr_sink_level` | DEBUG (original YAML, NOT quieted) |
| `log_interval` | 32 (original YAML, NOT quieted) |
| Timeout | **None** — run must complete naturally |
| Exit condition | `run_stop` with `status=success` (target reached) or all iterations exhausted |
| Output | **time-to-train (TTT)**, final eval_loss, full MLLOG compliance log |

**Use for:** Final report generation only. The training MUST run until the model either
reaches the target validation loss of 3.34 (`status=success`) or exhausts all iterations
(`status=aborted`). Do NOT interrupt or early-stop a Tier 3 run — the entire purpose is
to measure the actual time-to-target under the optimized configuration.

**After Tier 3 completes**, extract from `run_stop` MLLOG event:
- `status=success` → target reached, TTT is the primary result metric
- `status=aborted` → did not converge, report the final eval_loss and analyze why

### Tier Escalation

```
Tier 1 (100 iters)  →  gain > 1%?  →  Tier 2 (500 iters)  →  converges?  →  KEEP
                   →  gain ≤ 0%?  →  DISCARD
                   →  NaN/crash?  →  REVERT + log to KB
```

### MLLOG_TRAIN_LOSS_LOG_FREQ Override (CRITICAL)

The shell config sets `MLLOG_TRAIN_LOSS_LOG_FREQ=32`, which means `train_loss` events
are only emitted at iterations divisible by 32. For short trials, this yields very few
events and makes `extract_ms_per_iter` unreliable. Tier 1 and 2 MUST override this to `1`.

### Running a Trial

All actions use `run_mlperf_trial` from `scripts/common.sh`:

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "label" <tier> [train_iters] [extra_env]
```

Output is filtered through `trial_monitor.py`. Raw log is preserved at
`$RESULT_DIR/attempt_<label>_raw.log`. The last line of output is:

```
TRIAL_RESULT label=<label> ms_per_iter=<N> gbs=<N> last_loss=<N> iters=<N> status=<ok|nan|no_data>
```

### Output Filtering

Training output passes through a three-stage filter pipeline:

1. **Source reduction:** Quiet YAML (`stderr_sink_level: WARNING`, `log_interval: 999999`)
   suppresses Megatron INFO/DEBUG logs. `PYTHONWARNINGS=ignore` suppresses Python warnings.
2. **Stream filter:** `trial_monitor.py` reads stdin, passes through only :::MLLOG lines,
   RESULT lines, and errors. Replaces `train_loss` MLLOG with compact `[ITER N/M]` progress.
3. **Raw preservation:** Full unfiltered output is always saved to `*_raw.log` for debugging.

This significantly reduces log volume per trial without affecting MLPerf
compliance. All :::MLLOG lines are preserved verbatim in the raw log.

## Review Checkpoint Protocol

The orchestrator MUST pause at these checkpoints to verify state consistency. At each
checkpoint, print a `=== REVIEW CHECKPOINT RC-N ===` header and list what was verified.
If any check fails, STOP and investigate before proceeding.

| Checkpoint | After | Must Verify |
|------------|-------|-------------|
| RC-1 | setup | Env clean, symlinks valid, no stale processes, `quiet_yaml`/`restore_yaml` functional |
| RC-2 | baseline | ms/iter in expected range (1000–5000 ms for this model), loss decreasing, GBS matches config |
| RC-3 | profile | Profile trace exists, top-5 kernels identified, compute vs comm breakdown plausible |
| RC-4 | Every 3 DFS iterations | Cumulative gain positive, no false-positive gains from noise, all reverts were clean |
| RC-5 | Before sweep | Summarize all kept optimizations, verify each individual gain, check they compose correctly |
| RC-6 | After sweep | Optimal config identified, projected TTT is <= baseline TTT |
| RC-7 | After Tier 3 completes | `run_stop` status is `success` (target 3.34 reached), TTT extracted, MLLOG compliance log complete. If `status=aborted`, analyze final eval_loss and document why target was not reached. |

## Stopping Criteria

| Condition | Action |
|-----------|--------|
| All action scores < 1.0 | Proceed to sweep |
| Cumulative ms/iter gain > 15% | Proceed to sweep |
| 5 consecutive discards across all actions | Proceed to sweep |
| Wall clock > 180 min total | Proceed to sweep |
| Target exceeded (gap ≤ 0%) | Proceed to sweep |
| 2+ training crashes | Emergency stop, report partial results |

## KB Integration

Before each action, query the KB for relevant knowledge:

```bash
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B $ACTION_NAME" --top-k 5 --compact
```

After each action with new findings, ingest into KB:

```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category $CATEGORY --model "GPT-OSS-20B" \
    --action "$WHAT_WAS_DONE" --lesson "$KEY_TAKEAWAY" \
    --tags $TAGS --gain $GAIN --status $STATUS
```

## Action Dispatch

| Action | Module | Trial Tier | When |
|--------|--------|-----------|------|
| Setup | [`actions/setup.md`](actions/setup.md) | — | Always first (RC-1) |
| Classify | [`actions/classify.md`](actions/classify.md) | — | Always second |
| Target Analysis | [`actions/target-analysis.md`](actions/target-analysis.md) | — | If target provided |
| Baseline | [`actions/baseline.md`](actions/baseline.md) | Tier 1 | After classify (RC-2) |
| Profile | [`actions/profile.md`](actions/profile.md) | Tier 1 | After baseline (RC-3) |
| Fusion Flags | [`actions/fusion-flags.md`](actions/fusion-flags.md) | Tier 1 → 2 | DFS loop |
| Parallelism | [`actions/parallelism.md`](actions/parallelism.md) | Tier 1 → 2 | DFS loop |
| Training Params | [`actions/params.md`](actions/params.md) | Tier 1 → 2 | DFS loop |
| Hyperparameter Tuning | [`actions/hyperparams.md`](actions/hyperparams.md) | Tier 2 | DFS loop |
| Runtime Tunables | [`actions/runtime-tunables.md`](actions/runtime-tunables.md) | Tier 1 | DFS loop |
| Kernel Optimization | [`actions/kernel-opt.md`](actions/kernel-opt.md) | Tier 1 → 2 | DFS loop |
| Integration | [`actions/integrate.md`](actions/integrate.md) | Tier 1 → 2 | Per-kernel sub-action |
| Parameter Sweep | [`actions/sweep.md`](actions/sweep.md) | Tier 2 | After DFS loop (RC-5/6) |
| Report | [`actions/report.md`](actions/report.md) | Tier 3 | Always last (RC-7) |

## Reference: MLPerf Run Commands

### Local Run (inside container, no Docker wrapper)

```bash
cd /root/Hyperloom-plus-mlperf/training_optimization/mlperf
source config_MI355X_1x8x1_fp8.sh
bash setup_container_symlinks.sh
source config_MI355X_1x8x1_fp8.sh && bash run_and_time.sh
```

### Tiered Trial Run (for optimization loop)

```bash
source "$SKILL_ROOT/scripts/common.sh"

# Tier 1: Quick (100 iters, no eval, MLLOG_TRAIN_LOSS_LOG_FREQ=1)
run_mlperf_trial "test_name" 1

# Tier 2: Convergence (500 iters, eval enabled)
run_mlperf_trial "validate_name" 2

# Tier 2 with custom iters
run_mlperf_trial "validate_name" 2 300

# Tier 1 with extra env vars
run_mlperf_trial "gbs64" 1 100 "PRIMUS_GLOBAL_BATCH_SIZE=64 PRIMUS_LR=5.6e-4"

# Tier 3: Full verification
run_mlperf_trial "final" 3
```

**WARNING:** Never use raw `torchrun` or `bash run_and_time.sh` directly in actions.
Always use `run_mlperf_trial` to ensure log filtering, MLLOG_TRAIN_LOSS_LOG_FREQ override,
quiet YAML, and TRIAL_RESULT output.

### Key Environment Variables

```bash
# System
DGXSYSTEM=MI355X_1x8x1
GPUS_PER_NODE=8
NNODES=1
MASTER_PORT=29501

# Paths
PRIMUS_PATH=/workspace/Primus
EXP=/root/mlperf_primus/conf/gpt_oss_20B-pretrain-fp8.yaml
DATADIR=/shared_nfs/huangwei/gpt_oss_20b/data
MODELDIR=/shared_nfs/huangwei/gpt_oss_20b/model
LOGDIR=/root/mlperf_primus/logs

# Training
PRIMUS_MICRO_BATCH_SIZE=2
PRIMUS_GLOBAL_BATCH_SIZE=32
PRIMUS_LR=4.0e-4
PRIMUS_TRAIN_ITERS=1200000
PRIMUS_FP8_RECIPE=hybrid

# MLPerf
MLLOG_TARGET_EVAL_LOSS=3.34
MLLOG_TRAIN_LOSS_LOG_FREQ=32   # Overridden to 1 for Tier 1/2 trials!
MLLOG_SUBMISSION_BENCHMARK=gpt-oss-20b

# TE/ROCm
NVTE_CK_USES_FWD_V3=1
NVTE_CK_USES_BWD_V3=1
NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1
```

## Reference: MLLOG Output Format

All metrics are emitted as structured JSON lines prefixed with `:::MLLOG`:

```json
:::MLLOG {"namespace": "", "time_ms": 1775725581676, "event_type": "POINT_IN_TIME", "key": "train_loss", "value": 11.846, "metadata": {"samples_count": 0, "lr": 3.125e-06}}
```

### Key MLLOG Events

| Event Key | Type | Meaning |
|-----------|------|---------|
| `cache_clear` | POINT_IN_TIME | Cache cleared before run |
| `init_start` / `init_stop` | INTERVAL | Initialization phase |
| `run_start` / `run_stop` | INTERVAL | Training phase; `run_stop` has `status: success/aborted` |
| `epoch_start` / `epoch_stop` | INTERVAL | Epoch boundary |
| `block_start` / `block_stop` | INTERVAL | Training block (between evals) |
| `eval_start` / `eval_stop` | INTERVAL | Evaluation phase |
| `train_loss` | POINT_IN_TIME | Per-iteration training loss; metadata has `samples_count`, `lr` |
| `eval_loss` | POINT_IN_TIME | Validation loss at eval checkpoint |
| `train_samples` | POINT_IN_TIME | Total samples processed |
| `global_batch_size` | POINT_IN_TIME | GBS logged at init |
| `gradient_accumulation_steps` | POINT_IN_TIME | GA steps logged at init |

### Extracting ms/iter from MLLOG

**IMPORTANT:** `MLLOG_TRAIN_LOSS_LOG_FREQ` must be `1` for Tier 1/2 trials.
With the default value of 32, only iterations divisible by 32 emit `train_loss` events.
`run_mlperf_trial` handles this automatically.

For manual extraction from a raw log file:

```bash
source "$SKILL_ROOT/scripts/common.sh"
extract_ms_per_iter "$RESULT_DIR/attempt_label_raw.log"
```

Or read from the `TRIAL_RESULT` line (preferred):

```
TRIAL_RESULT label=baseline ms_per_iter=1643.9 gbs=32 last_loss=9.82 iters=100 ttt=134.8 run_status=aborted status=ok
```

### Extracting Time-to-Train

```bash
source "$SKILL_ROOT/scripts/common.sh"
extract_time_to_train "$RESULT_DIR/attempt_label_raw.log"
# Output: 134.4	aborted
```

### TRIAL_RESULT Line Format

Every `run_mlperf_trial` call ends by printing a structured result line:

```
TRIAL_RESULT label=<name> ms_per_iter=<float> gbs=<int> last_loss=<float> iters=<int> ttt=<float> run_status=<str> status=<str>
```

| Field | Type | Meaning |
|-------|------|---------|
| `label` | string | Trial identifier |
| `ms_per_iter` | float | Average ms/iter (warmup-excluded) |
| `gbs` | int | Global batch size from MLLOG |
| `last_loss` | float | Final training loss value |
| `iters` | int | Number of train_loss events observed |
| `ttt` | float | Time-to-train in seconds (from `run_start` to `run_stop`, 0.0 if not available) |
| `run_status` | string | MLLOG run_stop status: `success` (target reached), `aborted` (all iters done), `unknown` |
| `status` | enum | `converged` = target reached, `ok` = normal, `nan` = NaN/Inf, `no_data` = zero iterations |

Parse in shell:

```bash
eval "$(parse_trial_result "$(grep TRIAL_RESULT $filtered_log)")"
echo "ms/iter: $TRIAL_MS_PER_ITER, status: $TRIAL_STATUS"
```

## Reference: Validation Loss Output

```
validation loss at iteration N on validation set | lm loss value: 9.753977E+00 | lm loss PPL: 1.722258E+04
```

Parse regex: `lm loss value:\s*([\d.Ee+-]+)`

## Reference: RESULT Line

```
RESULT,GPT_OSS_20B,,176,AMD,2026-04-09 09:05:12 AM
```

Format: `RESULT,<model>,<extra>,<total_seconds>,<org>,<start_time>`

## Reference: Config Override Syntax

For MLPerf, overrides are set via environment variables BEFORE sourcing the config:

```bash
export PRIMUS_GLOBAL_BATCH_SIZE=64
export PRIMUS_MICRO_BATCH_SIZE=4
export PRIMUS_LR=8.0e-4
source config_MI355X_1x8x1_fp8.sh
```

Or modify the YAML config directly for Primus-level overrides (fusion flags, MoE settings).

## Reference: Critical Lessons

1. **GBS is tunable in MLPerf** (unlike training-optimization where GBS is fixed).
   Larger GBS means fewer iterations to process the same data, but each iteration is slower.
   The optimal GBS balances iteration time vs convergence speed.

2. **Eval overhead matters.** At GBS=32, eval runs every 384 iters. Each eval takes ~30s
   (64 eval iters). Reducing eval frequency saves wall time but risks overshooting target.

3. **FP8 stability is critical.** FP8 hybrid mode can cause loss spikes or NaN with certain
   configs. Always verify convergence after FP8-related changes.

4. **DeepEP can overlap communication.** With `moe_enable_deepep=true` and
   `turbo_deepep_num_cu=64`, expert parallelism communication overlaps with compute.

5. **Sliding window attention pattern matters.** The model uses alternating sliding/full
   attention (window_size=[128,0]). Do NOT change this — it affects model quality.

6. **Port conflicts after killing runs.** Increment `--master_port` (29502, 29503, ...)
   after killing training processes.

7. **Primus patches are auto-applied.** 16 patches are applied at startup for ROCm
   compatibility (permute fusion, FP8 context, TopK router, etc.).

8. **hipBLASLt GEMMs dominate (60–70% GPU time).** Gains come from reducing everything else.

## Reference: Process Management

- **Kill lingering processes:** `pkill -9 -f "train.py"` and `pkill -9 -f "torchrun"` before retrying.
- **Wait 5+ seconds** between kill and relaunch.
- **Increment `MASTER_PORT`** if the previous port is still bound.
- **Symlinks must be set** before each run: `bash setup_container_symlinks.sh`

## Reference: Training Metrics

| Metric | Unit | Meaning |
|--------|------|---------|
| `time_to_train` | seconds | Wall time from run_start to run_stop (primary metric) |
| `ms_per_iter` | ms | Milliseconds per training iteration (derived from MLLOG) |
| `train_loss` | float | Cross-entropy training loss per iteration |
| `eval_loss` | float | Validation log perplexity (target: 3.34) |
| `samples_count` | int | Total training samples processed |
| `lr` | float | Current learning rate |

## Reference: File Paths

```
/root/Hyperloom-plus-mlperf/training_optimization/mlperf/
├── config_MI355X_1x8x1_fp8.sh         # Shell config (env vars)
├── conf/gpt_oss_20B-pretrain-fp8.yaml  # Training YAML config
├── src/train.py                         # Training entry point
├── setup_container_symlinks.sh          # Symlink setup
├── run_and_time.sh                      # Training launcher
├── runtime_tunables.sh                  # System tuning (CPU, hugepages)
├── dev/                                 # Development scripts
└── patches/                             # Megatron/Primus patches
```
