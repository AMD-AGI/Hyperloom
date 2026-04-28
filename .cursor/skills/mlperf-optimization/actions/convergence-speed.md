# Action: Convergence Speed Optimization

## Overview

MLPerf time-to-train is `ms/iter × iterations_to_converge + eval_overhead`; this action
reduces the second and third terms by shaping the path to `eval_loss = 3.34` and cutting
eval wall time, within the already-chosen GBS/LR/parallelism configuration. It is
**convergence-affecting**: Layer 2 uses `loss_efficiency` (Tier 2) to filter candidates;
Layer 3 uses `projected_ttt` (Tier 3) for keep/discard. Run it after config selection
(Step 9) and before the sweep (Step 10).

## Scope (MLPerf IR-9 compliance)

Under MLPerf GPT-OSS-20B + AdamW rules (see `SKILL.md` IR-9), the vast majority of
hyperparameters are fixed or derived. The only remaining convergence knobs this action
may touch are:

| Dimension | Primus variable | Constraint |
|-----------|------------------|------------|
| Eval cadence | `EVAL_SAMPLES_INTERVAL` / `PRIMUS_EVAL_INTERVAL` | Measurement cadence, not a hyperparameter |
| Warmup steps | `PRIMUS_LR_WARMUP_ITERS` | `unconstrained` |
| Peak LR | `PRIMUS_LR` | `unconstrained`; `PRIMUS_MIN_LR` MUST be updated to `PRIMUS_LR × 0.1` |

**Forbidden in this action (IR-9 violations — do NOT test):**

- `PRIMUS_WEIGHT_DECAY` — fixed at `0.1`.
- `PRIMUS_CLIP_GRAD` — fixed at `1.0`.
- `PRIMUS_MIN_LR` as an independent knob — it is `PRIMUS_LR × 0.1`, period.
- `PRIMUS_LR_DECAY_ITERS` as an independent knob — it is
  `PRIMUS_TRAIN_ITERS − PRIMUS_LR_WARMUP_ITERS`, always.
- `PRIMUS_TRAIN_ITERS` — fixed at `1_200_000`.
- AdamW betas/epsilon, dropout, sequence length.

Any ad-hoc sub-action that proposes to vary those must be rejected before launching a
trial.

## Inputs

- Winning config from config-selection (GBS, LR, EP, TP) with `MIN_LR = LR × 0.1`
- Baseline TTT and ms/iter
- Baseline loss-vs-samples curve (from Tier 3 or Tier 4)
- Current `EVAL_SAMPLES_INTERVAL` and `PRIMUS_LR_WARMUP_ITERS`

## KB Query

```bash
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B convergence warmup eval interval peak LR" --top-k 5 --compact
```

## Procedure

### Dimension 1: Eval interval optimization

Eval overhead is pure wall time with no convergence benefit. Each eval is on the order
of tens of seconds (64 eval iters × MBS × GPUs). Eval cadence does not affect training
math — only measurement cadence.

#### Current configuration

| Setting | Value | Impact |
|---------|-------|--------|
| `EVAL_SAMPLES_INTERVAL` | 12288 | Eval every 12,288 samples |
| `PRIMUS_EVAL_INTERVAL` | `EVAL_SAMPLES_INTERVAL / GBS` | Eval every N iterations |
| Eval duration | ~30s per eval | Multiple minutes over full run |

Use `compute_eval_overhead()` from `common.sh`:

```bash
source "$SKILL_ROOT/scripts/common.sh"
eval_overhead=$(compute_eval_overhead "$RESULT_DIR/attempt_baseline_raw.log")
```

#### Trial verification

If eval interval changes are significant, verify with Tier 3:

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "eval_interval_24576" 3 "" \
    "EVAL_SAMPLES_INTERVAL=24576"
```

Verify the loss trajectory matches baseline (eval interval does not affect training math).

### Dimension 2: Warmup iterations (`opt_learning_rate_warmup_steps`)

**Unconstrained** per MLPerf rules. When `PRIMUS_LR_WARMUP_ITERS` changes,
`PRIMUS_LR_DECAY_ITERS` MUST be updated to `PRIMUS_TRAIN_ITERS − PRIMUS_LR_WARMUP_ITERS`.

| Warmup | Notes |
|--------|-------|
| 64 | Aggressive — higher risk of early NaN with FP8 |
| 128 | Current default |
| 256 | More conservative |
| 512 | Very conservative |

**Stage 1: Tier 2 quick filter (500 iters) — `loss_efficiency`**

```bash
source "$SKILL_ROOT/scripts/common.sh"

# When overriding warmup, always pair with matching decay_iters (IR-9 derived constraint).
run_mlperf_trial "warmup_64"  2 500 \
    "PRIMUS_LR_WARMUP_ITERS=64  PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-64))"
run_mlperf_trial "warmup_128" 2 500 \
    "PRIMUS_LR_WARMUP_ITERS=128 PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-128))"
run_mlperf_trial "warmup_256" 2 500 \
    "PRIMUS_LR_WARMUP_ITERS=256 PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-256))"

baseline_eff=$(compute_loss_efficiency "$RESULT_DIR/attempt_baseline_raw.log")
for label in warmup_64 warmup_128 warmup_256; do
    eff=$(compute_loss_efficiency "$RESULT_DIR/attempt_${label}_raw.log")
    echo "$label: loss_efficiency=$eff (baseline=$baseline_eff)"
done
```

Discard any candidate where `loss_efficiency < baseline_eff × LOSS_EFFICIENCY_DISCARD_RATIO`
(0.85).

**Stage 2: Tier 3 projected TTT for survivors (2500 iters)**

```bash
run_mlperf_trial "warmup_<best>_proj" 3 "" \
    "PRIMUS_LR_WARMUP_ITERS=<best> PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-<best>))"
projected=$(project_ttt "$RESULT_DIR/attempt_warmup_<best>_proj_raw.log" "$GBS")
ttt_gain=$(compute_ttt_gain_pct "$BASELINE_PROJECTED_TTT" "$(echo $projected | cut -f1)")
echo "Warmup <best>: projected_ttt gain = ${ttt_gain}%"
```

**Decision metric:** `projected_ttt` from Tier 3, not `eval_loss` at iteration 500.
KEEP only if `ttt_gain_pct > TTT_GAIN_KEEP_THRESHOLD_PCT` (0%).

**NaN gate:** If warmup < 128 causes NaN in the first 100 iterations (`TRIAL_STATUS=nan`),
discard immediately.

### Dimension 3: Peak LR (`opt_base_learning_rate`)

**Unconstrained** per MLPerf rules. `PRIMUS_MIN_LR` MUST be updated to
`PRIMUS_LR × 0.1` (IR-9 derived constraint — the `opt_end_learning_rate` rule).

Because both `lr_decay_iters` and `min_lr` are locked to `lr` and `train_iters`, peak LR
is the **only** LR-schedule knob available. The cosine schedule anneals from `lr` down
to `lr × 0.1` over `TRAIN_ITERS − WARMUP_ITERS` steps — this shape is fixed; only its
amplitude (`lr`) moves.

| Peak LR | `MIN_LR` (= `LR × 0.1`) | Notes |
|---------|-------------------------|-------|
| 2.0e-4 | 2.0e-5 | Lower — safer, slower early progress |
| 4.0e-4 | 4.0e-5 | Baseline default |
| 5.0e-4 | 5.0e-5 | Moderate bump |
| 6.0e-4 | 6.0e-5 | Aggressive — higher NaN risk under FP8 |
| 8.0e-4 | 8.0e-5 | Very aggressive |

**Stage 1: Tier 2 loss_efficiency filter (500 iters, NaN gate)**

```bash
source "$SKILL_ROOT/scripts/common.sh"

run_mlperf_trial "lr_5e4" 2 500 "PRIMUS_LR=5.0e-4 PRIMUS_MIN_LR=5.0e-5"
run_mlperf_trial "lr_6e4" 2 500 "PRIMUS_LR=6.0e-4 PRIMUS_MIN_LR=6.0e-5"
run_mlperf_trial "lr_8e4" 2 500 "PRIMUS_LR=8.0e-4 PRIMUS_MIN_LR=8.0e-5"

baseline_eff=$(compute_loss_efficiency "$RESULT_DIR/attempt_baseline_raw.log")
for label in lr_5e4 lr_6e4 lr_8e4; do
    eff=$(compute_loss_efficiency "$RESULT_DIR/attempt_${label}_raw.log")
    echo "$label: loss_efficiency=$eff (baseline=$baseline_eff)"
done
```

Discard any candidate where:

- `TRIAL_STATUS=nan` (NaN in first 200 iters → revert immediately, log to KB), OR
- `loss_efficiency < baseline_eff × LOSS_EFFICIENCY_DISCARD_RATIO` (0.85)

**Stage 2: Tier 3 projected TTT for survivors (2500 iters)**

```bash
run_mlperf_trial "lr_<best>_proj" 3 "" "PRIMUS_LR=<best> PRIMUS_MIN_LR=<best×0.1>"
projected=$(project_ttt "$RESULT_DIR/attempt_lr_<best>_proj_raw.log" "$GBS")
ttt_gain=$(compute_ttt_gain_pct "$BASELINE_PROJECTED_TTT" "$(echo $projected | cut -f1)")
echo "LR <best>: projected_ttt gain = ${ttt_gain}%"
```

KEEP only if `ttt_gain_pct > TTT_GAIN_KEEP_THRESHOLD_PCT` (0%).

### Combined validation

Apply all winning convergence tweaks together. Only three knobs can legally appear in
this combination:

```bash
# IR-9: MIN_LR = LR × 0.1 and DECAY_ITERS = TRAIN_ITERS − WARMUP_ITERS must hold.
run_mlperf_trial "convergence_combined" 3 "" \
    "PRIMUS_LR=<best_lr> PRIMUS_MIN_LR=<best_lr_times_0.1> \
     PRIMUS_LR_WARMUP_ITERS=<best_warmup> \
     PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-<best_warmup>)) \
     EVAL_SAMPLES_INTERVAL=<best_eval>"
projected=$(project_ttt "$RESULT_DIR/attempt_convergence_combined_raw.log" "$GBS")
echo "Projected TTT: $projected"
```

Compare against baseline projected TTT.

## Outputs

- Optimal `EVAL_SAMPLES_INTERVAL`
- Optimal `PRIMUS_LR_WARMUP_ITERS` (with matching `PRIMUS_LR_DECAY_ITERS`)
- Optimal `PRIMUS_LR` (with matching `PRIMUS_MIN_LR = PRIMUS_LR × 0.1`)
- Projected TTT savings (seconds and percentage)
- Updated environment variables in `state.kept_env_vars`
- Updated `state.optimal_eval_interval`, `state.eval_overhead_seconds`
- Updated `state.lr`, `state.min_lr` (if changed; always paired)

## Heuristic Update

- Peak LR improvement: boost remaining untested LR candidates in the vicinity by 1.3×
- Warmup improvement: boost remaining untested warmup candidates by 1.3×
- After convergence-speed completes: update `state.eval_overhead_seconds`,
  `state.optimal_eval_interval` (Score Update Rule #9)
- If no dimension improves projected TTT: reduce convergence-speed score by 0.7×

## Failure Handling

- Aggressive warmup (< 128) causes NaN → revert; keep default 128
- Peak LR bump causes NaN in first 200 iters → revert; keep prior LR, log to KB as unsafe
- Eval interval change must not affect loss trajectory (expected) → keep the interval
  that minimizes total wall time
- These are low-risk tunings within MLPerf-compliant bounds — if none improve projected
  TTT, keep baseline defaults
