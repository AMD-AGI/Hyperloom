# MLPerf Optimization — Operational Reference

This file contains operational reference material for the MLPerf optimization skill.
It is loaded on-demand by action modules and for troubleshooting — **not** by the main
DFS orchestrator. For decision-layer content (Iron Rules, Constants, State Schema,
heuristics, orchestrator loop), see [`SKILL.md`](SKILL.md).

## MLPerf Run Commands

### Local Run (inside container, no Docker wrapper)

```bash
cd /root/Hyperloom-plus-mlperf/training_optimization/mlperf
source config_MI355X_1x8x1_fp8.sh
bash setup_container_symlinks.sh
source config_MI355X_1x8x1_fp8.sh && bash run_and_time.sh
```

### Tiered Trial Run (for the optimization loop)

```bash
source "$SKILL_ROOT/scripts/common.sh"

# Tier 1: Quick (100 iters, no eval, MLLOG_TRAIN_LOSS_LOG_FREQ=1)
run_mlperf_trial "test_name" 1

# Tier 2: Convergence (500 iters, eval enabled)
run_mlperf_trial "validate_name" 2

# Tier 2 with custom iters
run_mlperf_trial "validate_name" 2 300

# Tier 3: Long convergence for config selection + projected TTT (2500 iters, eval enabled, 4hr timeout)
run_mlperf_trial "gbs32_ep1" 3
run_mlperf_trial "gbs16_ep1" 3 "" "PRIMUS_GLOBAL_BATCH_SIZE=16 PRIMUS_LR=2.0e-4 PRIMUS_MIN_LR=2.0e-5"  # IR-9: MIN_LR = LR × 0.1

# Tier 1 with extra env vars
run_mlperf_trial "gbs64" 1 100 "PRIMUS_GLOBAL_BATCH_SIZE=64 PRIMUS_LR=5.6e-4"

# Tier 4: Full verification
run_mlperf_trial "final" 4
```

Never use raw `torchrun` or `bash run_and_time.sh` directly in actions (see IR-1 in SKILL.md).
The wrapper ensures log filtering, `MLLOG_TRAIN_LOSS_LOG_FREQ` override, quiet YAML, and
`TRIAL_RESULT` output.

## Key Environment Variables

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

# Training (see SKILL.md IR-9 for MLPerf hyperparameter bounds)
PRIMUS_MICRO_BATCH_SIZE=2
PRIMUS_GLOBAL_BATCH_SIZE=32          # unconstrained
PRIMUS_LR=4.0e-4                     # unconstrained
PRIMUS_MIN_LR=4.0e-5                 # DERIVED: MUST equal PRIMUS_LR × 0.1
PRIMUS_LR_WARMUP_ITERS=128           # unconstrained
PRIMUS_LR_DECAY_ITERS=1199872        # DERIVED: MUST equal PRIMUS_TRAIN_ITERS − PRIMUS_LR_WARMUP_ITERS
PRIMUS_TRAIN_ITERS=1200000           # FIXED by MLPerf (max_steps)
PRIMUS_WEIGHT_DECAY=0.1              # FIXED by MLPerf (do NOT change)
PRIMUS_CLIP_GRAD=1.0                 # FIXED by MLPerf (do NOT change)
PRIMUS_FP8_RECIPE=hybrid

# MLPerf
MLLOG_TARGET_EVAL_LOSS=3.34
MLLOG_TRAIN_LOSS_LOG_FREQ=32   # Overridden to 1 for Tier 1/2/3 trials (IR-3)
MLLOG_SUBMISSION_BENCHMARK=gpt-oss-20b

# TE/ROCm
NVTE_CK_USES_FWD_V3=1
NVTE_CK_USES_BWD_V3=1
NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1
```

## Config Override Syntax

For MLPerf, overrides are set via environment variables **before** sourcing the config.
Remember to maintain the IR-9 derived pairings (`MIN_LR = LR × 0.1`,
`DECAY_ITERS = TRAIN_ITERS − WARMUP_ITERS`):

```bash
export PRIMUS_GLOBAL_BATCH_SIZE=64
export PRIMUS_MICRO_BATCH_SIZE=4
export PRIMUS_LR=8.0e-4
export PRIMUS_MIN_LR=8.0e-5          # = PRIMUS_LR × 0.1 (IR-9)
export PRIMUS_LR_WARMUP_ITERS=256
export PRIMUS_LR_DECAY_ITERS=1199744 # = PRIMUS_TRAIN_ITERS − PRIMUS_LR_WARMUP_ITERS (IR-9)
source config_MI355X_1x8x1_fp8.sh
```

Or modify the YAML config directly for Primus-level overrides (fusion flags, MoE settings).
Do NOT override `PRIMUS_WEIGHT_DECAY`, `PRIMUS_CLIP_GRAD`, AdamW betas/epsilon, dropout,
or `sequence_length` — see `SKILL.md` IR-9.

## MLLOG Output Format

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

### Extracting ms/iter

```bash
source "$SKILL_ROOT/scripts/common.sh"
extract_ms_per_iter "$RESULT_DIR/attempt_label_raw.log"
```

Or read from the `TRIAL_RESULT` line (preferred).

### Extracting Time-to-Train

```bash
source "$SKILL_ROOT/scripts/common.sh"
extract_time_to_train "$RESULT_DIR/attempt_label_raw.log"
# Output: 134.4  aborted
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
| `iters` | int | Number of `train_loss` events observed |
| `ttt` | float | Time-to-train in seconds (0.0 if not available) |
| `run_status` | string | MLLOG `run_stop` status: `success`, `aborted`, `unknown` |
| `status` | enum | `converged` / `ok` / `nan` / `no_data` |

Parse in shell:

```bash
eval "$(parse_trial_result "$(grep TRIAL_RESULT $filtered_log)")"
echo "ms/iter: $TRIAL_MS_PER_ITER, status: $TRIAL_STATUS"
```

## Trial Tier Details

The summary table and escalation graph are in [`SKILL.md`](SKILL.md) § Trial Tier Summary.
Full per-tier parameter tables are below.

### Tier 1: Quick Trial

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | 100 (`TIER1_ITERS`) |
| `PRIMUS_EVAL_INTERVAL` | 10000 (suppress eval) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | **1** (IR-3) |
| `stderr_sink_level` | WARNING (quiet YAML) |
| `log_interval` | 999999 (quiet YAML) |
| Timeout | 60 minutes (`TIER1_TIMEOUT_MIN`) |

**Use for:** Initial measurement of any config change, fusion flag testing, crash detection.

### Tier 2: Convergence Trial

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | 500 (`TIER2_ITERS`) |
| `PRIMUS_EVAL_INTERVAL` | 50 (multiple evals) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | **1** (IR-3) |
| `stderr_sink_level` | WARNING |
| `log_interval` | 999999 |
| Timeout | 120 minutes (`TIER2_TIMEOUT_MIN`) |

**Use for:** Validating winners, hyperparameter tuning, FP8 stability checks, sweep.

### Tier 3: Long Convergence Trial (Config Selection + Projected TTT)

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | 2500 (`TIER3_ITERS`) |
| `PRIMUS_EVAL_INTERVAL` | 50 |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | **1** (IR-3) |
| `stderr_sink_level` | WARNING |
| `log_interval` | 999999 |
| Timeout | 4 hours (`TIER3_TIMEOUT_S` = 14400) |

**Use for:** Comparing GBS/LR/TP/EP/DP candidates during config selection, and projecting
TTT for convergence-affecting actions during the DFS loop. At GBS=32, 2500 iters = 80,000
samples (~6 eval cycles) — enough to fit a curve and project TTT.

### Tier 4: Full Verification (Run to Convergence)

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | Original YAML (do NOT override) |
| `PRIMUS_EVAL_INTERVAL` | Original YAML (do NOT override) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | 32 (original, `MLLOG_TRAIN_LOSS_LOG_FREQ_DEFAULT`) |
| `stderr_sink_level` | DEBUG (original YAML, NOT quieted) |
| `log_interval` | 32 (original, NOT quieted) |
| Timeout | **None** — run must complete naturally (IR-2) |
| Exit | `run_stop status=success` (target reached) or all iters exhausted (`aborted`) |

**Use for:** Final report + config-selection top-2 verification. After completion, extract
from `run_stop`: `status=success` → TTT is the primary result; `status=aborted` → report
final `eval_loss` and analyze.

### MLLOG_TRAIN_LOSS_LOG_FREQ Override

The shell config defaults to `32`, emitting `train_loss` only at iterations divisible by 32.
For short trials this yields too few events and makes `extract_ms_per_iter` unreliable.
Tier 1/2/3 MUST force to `1` (IR-3). `run_mlperf_trial` handles this automatically.

### Output Filtering

Training output passes through a three-stage filter pipeline:

1. **Source reduction:** Quiet YAML (`stderr_sink_level: WARNING`, `log_interval: 999999`)
   suppresses Megatron INFO/DEBUG logs. `PYTHONWARNINGS=ignore` suppresses Python warnings.
2. **Stream filter:** `trial_monitor.py` reads stdin, passes through only `:::MLLOG` lines,
   RESULT lines, and errors. Replaces `train_loss` MLLOG with compact `[ITER N/M]` progress.
3. **Raw preservation:** Full unfiltered output is always saved to `*_raw.log` for debugging.

All `:::MLLOG` lines are preserved verbatim in the raw log (IR-7).

## Validation Loss Output

```
validation loss at iteration N on validation set | lm loss value: 9.753977E+00 | lm loss PPL: 1.722258E+04
```

Parse regex: `lm loss value:\s*([\d.Ee+-]+)`

## RESULT Line

```
RESULT,GPT_OSS_20B,,176,AMD,2026-04-09 09:05:12 AM
```

Format: `RESULT,<model>,<extra>,<total_seconds>,<org>,<start_time>`

## Process Management

- Kill lingering processes: `pkill -9 -f "train.py"` and `pkill -9 -f "torchrun"` (IR-5)
- Wait 5+ seconds between kill and relaunch
- Increment `MASTER_PORT` (29502, 29503, …) if the previous port is still bound
- Symlinks must be re-created: `bash setup_container_symlinks.sh`

## Training Metrics Reference

| Metric | Unit | Meaning |
|--------|------|---------|
| `time_to_train` | seconds | Wall time from `run_start` to `run_stop` (primary metric) |
| `ms_per_iter` | ms | Milliseconds per training iteration (derived from MLLOG) |
| `train_loss` | float | Cross-entropy training loss per iteration |
| `eval_loss` | float | Validation log perplexity (target: `MLPERF_QUALITY_TARGET` = 3.34) |
| `samples_count` | int | Total training samples processed |
| `lr` | float | Current learning rate |

## File Paths

```
/root/Hyperloom-plus-mlperf/training_optimization/mlperf/
├── config_MI355X_1x8x1_fp8.sh          # Shell config (env vars)
├── conf/gpt_oss_20B-pretrain-fp8.yaml  # Training YAML config
├── src/train.py                         # Training entry point
├── setup_container_symlinks.sh          # Symlink setup
├── run_and_time.sh                      # Training launcher
├── runtime_tunables.sh                  # System tuning (CPU, hugepages)
├── dev/                                 # Development scripts
└── patches/                             # Megatron/Primus patches
```

## Lessons

Unified table — both decision-layer heuristics (used by DFS scoring / action selection)
and operational hints (troubleshooting, integration). `SKILL.md § Lessons` points here.

`layer` column:
- `decision` — informs DFS priors and action selection (previously § Critical Lessons).
- `operational` — troubleshooting and integration (previously § Operational Lessons).

| # | Layer | Lesson | Source |
|---|---|---|---|
| 1 | decision | **GBS is tunable in MLPerf** (unlike training-optimization where GBS is fixed). Larger GBS → fewer iterations but each iteration is slower. Optimal balance is via config-selection. | Critical #1 |
| 2 | decision | **Eval overhead matters.** At `DEFAULT_GBS`, eval runs every 512 iters; each eval ~30s (64 eval iters). Reducing frequency saves 2–5 min at zero convergence risk — but requires convergence-speed Dimension 1 validation. | Critical #2 |
| 3 | decision | **TraceLens comm-compute overlap informs tuning priority** (comm-tuning Part B). High overlap (>0.7) suggests smaller gains from Part B — still run Part A (NCCL buffer/channel tuning) unconditionally. | Critical #3 |
| 4 | decision | **ms/iter gain alone is insufficient for convergence-affecting actions.** LR / GBS / warmup can have zero ms/iter impact but dramatically change iterations-to-converge. Always use the three-layer evaluation (ms/iter → loss_efficiency → projected_ttt). The DFS loop classifies and applies this automatically. | Critical #4 |
| 5 | decision | **The LR-schedule shape is FIXED by MLPerf rules (IR-9).** `opt_end_learning_rate = opt_base_learning_rate × 0.1` and `opt_learning_rate_decay_steps = 1_200_000 − warmup_steps`. This means the cosine curve always goes from peak LR down to `LR × 0.1` over `TRAIN_ITERS − WARMUP_ITERS`. Its *shape* is non-negotiable — only its *amplitude* (peak LR) and its *start* (warmup length) move. Do NOT tune `lr_decay_iters` or `min_lr` as independent knobs. | Critical #5 |
| 6 | decision | **Peak LR and warmup are the only LR-schedule knobs.** When `PRIMUS_LR` changes, `PRIMUS_MIN_LR` MUST be updated to `PRIMUS_LR × 0.1`. When `PRIMUS_LR_WARMUP_ITERS` changes, `PRIMUS_LR_DECAY_ITERS` MUST be updated to `PRIMUS_TRAIN_ITERS − PRIMUS_LR_WARMUP_ITERS`. These pairings are IR-9 compliance, not heuristics. | Critical #6 |
| 7 | operational | **FP8 stability matters.** FP8 hybrid can cause loss spikes or NaN with certain configs. Always Tier 2 after FP8 change. (See also Common Pitfall #4 in SKILL.md.) | Operational #1 |
| 8 | operational | **DeepEP can overlap communication.** With `moe_enable_deepep=true` and `turbo_deepep_num_cu=64`, expert parallelism communication overlaps with compute. | Operational #2 |
| 9 | operational | **Primus patches are auto-applied.** 16 patches applied at startup for ROCm compatibility (permute fusion, FP8 context, TopK router). Do NOT manually re-apply. (See also Common Pitfall #5 in SKILL.md.) | Operational #3 |
| 10 | operational | **hipBLASLt GEMMs dominate 60–70% GPU time.** Gains come from reducing everything else. GEAK cannot beat vendor-tuned GEMMs. (See also Common Pitfall #6 in SKILL.md.) | Operational #4 |
| 11 | operational | **NVTE_USE_CAST_TRANSPOSE_TRITON=1 yields ~0.2% improvement.** Enables Triton-based FP8 cast+transpose. Apply early in the throughput-knobs action (fusion-flags sub-section). | Operational #5 |
| 12 | operational | **Current best known TTT is `REFERENCE_TTT_MIN` = 206 min.** Baseline must re-verify, not assume. | Operational #6 |

## Score Update Rules

After every action completes, apply these rules in order. The summary in
[`SKILL.md § Score Update Rules`](SKILL.md#score-update-rules) points here.

1. **Action succeeded (gain > 0%):** Boost similar actions proportional to gain (e.g.,
   fusion-flags +1.5% → boost remaining fusion flags by 1.5×).
2. **Action failed (gain ≤ 0%):** Throughput-only → reduce similar by 0.7×;
   convergence-affecting → reduce by 0.85× (softer penalty — short trials are noisier for
   convergence effects).
3. **After 2+ fusion flag wins:** Push `combined_fusion_test` with score =
   sum(individual) × 1.5.
4. **After all fusion flags tested:** Push `re-profile` via
   [`profile.md § Re-Profile Trigger`](actions/profile.md#re-profile-trigger) to
   discover new kernel targets.
5. **After kernel opt kept:** Push `re-profile + next-kernel` via
   [`profile.md § Re-Profile Trigger`](actions/profile.md#re-profile-trigger) with
   boosted score.
6. **After kernel opt discarded:** Reduce remaining kernel scores by 0.7×, but enforce
   `SCORE_FLOOR_KERNEL_OPT` — `kernel-opt` is never removed from the stack.
7. **After any action with gain > `REPROFILE_TRIGGER_PCT`:** Push `re-profile` onto stack
   with score = `max(remaining_scores) × 0.8`.
8. **After FP8 knob succeeds:** Boost remaining untested FP8 knobs by 1.3×.
9. **After convergence-speed action:** Update `state.eval_overhead_seconds` and
   `state.optimal_eval_interval`.
10. **Stopping combined rule:** When all scores < `STOP_ALL_SCORES_BELOW` AND
    `dfs_iteration_count ≥ MIN_DFS_ITERATIONS` → proceed to sweep → report.
11. **KB penalty cap (IR-4):** KB negative entries reduce `expected_time_reduction_pct` by
    at most `KB_PENALTY_CAP`. Never zero out, never remove from stack.
12. **After LR / warmup change (convergence-speed Dim 2 or 3):** If peak LR improved
    `projected_ttt`, boost remaining peak-LR candidates in the vicinity by 1.3×. If
    warmup change improved `projected_ttt`, boost remaining warmup candidates by 1.3×.
    Per IR-9, `min_lr` and `lr_decay_iters` are derived, not independent — do NOT score
    them as separate sub-dimensions.

## Orchestrator Loop (Full)

The summary in [`SKILL.md § Orchestrator Loop`](SKILL.md#orchestrator-loop) points here.

```
PROCEDURE optimize():

  1. SETUP
     → Execute actions/setup.md
     → Verify env, symlinks, kill stale processes, validate quiet_yaml/trial_monitor
     → MCP self-heal + TraceLens CLI check → populate state.mcp_*, tracelens_cli_available
     → === REVIEW CHECKPOINT RC-1 ===

  2. CLASSIFY
     → Execute actions/classify.md
     → Set model_class, parallelism_topology, initial score priors

  3. TARGET ANALYSIS (if $TARGET_DIR or target numbers provided)
     → Execute actions/target-analysis.md
     → Set target_time_to_train, target_gap_pct, target_gap_multiplier

  4. KB WARM-UP (REFERENCE ONLY — see IR-4)
     → python3 kb/kb_query.py --model "GPT-OSS-20B" --top-k 20
     → Apply priors subject to KB_PENALTY_CAP. Never skip an untested action.

  5. BASELINE (Tier 4 full run — see IR-2)
     → Execute actions/baseline.md → run_mlperf_trial "baseline" 4
     → Set baseline_time_to_train, baseline_ms_per_iter, baseline_eval_loss
     → Compare against REFERENCE_TTT_MIN — verify, do not assume
     → === REVIEW CHECKPOINT RC-2 ===

  6. PROFILE + TRACELENS
     → Execute actions/profile.md (Tier 1 with profiling enabled)
     → Populate kernel_candidates; run TraceLens CLI (MANDATORY)
     → Compute baseline_loss_efficiency and baseline_projected_ttt from baseline Tier 4 log
     → === REVIEW CHECKPOINT RC-3 ===

  7. BUILD ACTION STACK
     → Score all candidate actions; include comm-tuning (Parts A & B) unconditionally
     → Push onto action_stack sorted by score (highest first)

  8. DFS LOOP:
     SET dfs_iteration_count = 0
     WHILE action_stack is not empty AND
           (dfs_iteration_count < MIN_DFS_ITERATIONS OR NOT stopping_criteria_met()):
       a. Pop highest-scored action; classify (throughput-only vs convergence-affecting)
       b. IF throughput-only:
            Tier 1 (TIER1_ITERS) → parse TRIAL_RESULT → measure ms/iter gain
            If gain > CONVERGENCE_GATE_TRIGGER_PCT: escalate to Tier 2 for convergence check
            If Tier 2 shows divergence or status=nan: REVERT
       c. IF convergence-affecting:
            Tier 2 (TIER2_ITERS) → compute loss_efficiency
            If loss_efficiency < baseline × LOSS_EFFICIENCY_DISCARD_RATIO: DISCARD
            Else: Tier 3 (TIER3_ITERS) → compute projected_ttt
            If ttt_gain_pct ≤ TTT_GAIN_KEEP_THRESHOLD_PCT: DISCARD. Else: KEEP
       d. Update state.current_ms_per_iter, cumulative_gain_pct, projected_ttt
       e. Re-score all remaining actions using the appropriate gain metric
       f. If gain > REPROFILE_TRIGGER_PCT: push re-profile with score = max(remaining) × 0.8
       g. Push any new sub-actions; log completed_actions
       h. INCREMENT dfs_iteration_count
       i. IF dfs_iteration_count % REVIEW_CHECKPOINT_INTERVAL == 0: === RC-4 ===

  9. CONFIG SELECTION (GBS/LR/TP/EP/DP)
     → Execute actions/config-selection.md
     → Stage 1 (Tier 1 filter) → Stage 2 (Tier 3 comparison) → Stage 3 (Tier 4 top-2 verification)
     → Apply winner as new baseline for subsequent steps
     → === REVIEW CHECKPOINT RC-4.5 ===

  9.5 CONVERGENCE SPEED
     → Execute actions/convergence-speed.md (3-dimension: eval interval, warmup, peak LR)
     → Update state.optimal_eval_interval, eval_overhead_seconds, lr, min_lr
     → (min_lr tracked only as the IR-9 derived pair of lr; not independently tunable)

  9.6 COMM TUNING (always runs)
     → Execute actions/comm-tuning.md (NCCL/RCCL/DeepEP Part A + TraceLens-guided overlap Part B)

 10. PRE-SWEEP / SWEEP
     → === RC-5 === → Execute actions/sweep.md (Tier 2) → === RC-6 ===

 11. REPORT
     → Execute actions/report.md → Tier 4 final run (IR-2) → TraceLens before/after comparison
     → === RC-7 === → verify run_stop status, extract TTT, generate report

 12. KNOWLEDGE HOOK
     → python3 kb/kb_ingest.py to record lessons learned
```

## State Schema

Full orchestrator state dict. The summary grouping in
[`SKILL.md § State Schema`](SKILL.md#state-schema) points here.

```python
state = {
    "model_class": "",              # from classify (MoE + GQA/SWA specifics)
    "parallelism_topology": {},     # TP, EP, DP from classify
    "framework": "",                # Primus/Megatron
    "gpu_count": 8,                 # from setup

    "baseline_time_to_train": 0.0,  # TTT in seconds (Tier 4 actual)
    "baseline_ms_per_iter": 0.0,    # from baseline Tier 4
    "baseline_eval_loss": 0.0,
    "baseline_loss_efficiency": 0.0,
    "baseline_projected_ttt": 0.0,

    "current_ms_per_iter": 0.0,
    "cumulative_gain_pct": 0.0,
    "projected_ttt": 0.0,

    "target_time_to_train": None,   # from target-analysis
    "target_gap_pct": None,
    "target_gap_multiplier": 1.0,

    "action_stack": [],             # priority stack of (score, action_name, params)
    "completed_actions": [],        # log of (action, gain_pct, gain_type, status)
    "dfs_iteration_count": 0,
    "kernel_candidates": [],        # from profiling

    "mcp_status": {},               # from setup MCP probe
    "mcp_tools": {},
    "tracelens_cli_available": False,
    "oob_available": False,
    "geak_available": False,

    "optimal_eval_interval": None,  # from convergence-speed
    "eval_overhead_seconds": 0.0,
    "lr": None,                     # IR-9 unconstrained; updated by convergence-speed
    "min_lr": None,                 # IR-9 derived: must equal lr × 0.1
    "warmup_iters": None,           # IR-9 unconstrained
    "lr_decay_iters": None,         # IR-9 derived: train_iters − warmup_iters (never tuned independently)

    "total_wall_minutes": 0,
    "consecutive_discards": 0,
}
```
