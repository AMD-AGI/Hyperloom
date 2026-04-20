# Action: Convergence Speed Optimization

## Overview

MLPerf time-to-train is `ms/iter × iterations_to_converge + eval_overhead`; this action reduces the second and third terms by shaping the path to `eval_loss = 3.34` and cutting eval wall time, including secondary hyperparameter fine-tuning (warmup, eval interval, weight decay) within the already-chosen GBS/LR/parallelism configuration. It is **convergence-affecting**: Layer 2 uses `loss_efficiency` (Tier 2) to filter candidates; Layer 3 uses `projected_ttt` (Tier 2.5) for keep/discard. Run it after config selection (Step 9) and before the sweep (Step 10).

## Inputs

- Winning config from config-selection (GBS, LR, EP, TP)
- Baseline TTT and ms/iter
- Baseline loss-vs-samples curve (from Tier 2L or Tier 3)
- Current `eval_interval`, `warmup`, `weight_decay`, and LR schedule settings

## KB Query

```bash
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B convergence LR schedule warmup" --top-k 5 --compact
```

## Procedure

### Dimension 1: Eval interval optimization

Eval overhead is pure wall time with no convergence benefit. Each eval is on the order of tens of seconds (64 eval iters × MBS × GPUs).

#### Current configuration

| Setting | Value | Impact |
|---------|-------|--------|
| `EVAL_SAMPLES_INTERVAL` | 16384 | Eval every 16384 samples |
| `PRIMUS_EVAL_INTERVAL` | 512 (at GBS=32) | Eval every 512 iterations |
| Eval duration | ~30s per eval | ~6.4 min total (~13 evals to converge at 220k iters) |

Use `compute_eval_overhead()` from `common.sh`:

```bash
source "$SKILL_ROOT/scripts/common.sh"
eval_overhead=$(compute_eval_overhead "$RESULT_DIR/attempt_baseline_raw.log")
```

#### Trial verification

If eval interval changes are significant, verify with Tier 2L:

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "eval_interval_32768" 2L "" \
    "EVAL_SAMPLES_INTERVAL=32768"
```

Verify the loss trajectory matches baseline (eval interval does not affect training math).

### Dimension 2: Warmup iterations

| Warmup | Notes |
|--------|-------|
| 64 | Aggressive — higher risk of early NaN with FP8 |
| 128 | Current default |
| 256 | More conservative |

**Stage 1: Tier 2 quick filter (500 iters) — `loss_efficiency`**

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "warmup_64" 2 500 "PRIMUS_LR_WARMUP_ITERS=64"
run_mlperf_trial "warmup_128" 2 500 "PRIMUS_LR_WARMUP_ITERS=128"
run_mlperf_trial "warmup_256" 2 500 "PRIMUS_LR_WARMUP_ITERS=256"

baseline_eff=$(compute_loss_efficiency "$RESULT_DIR/attempt_baseline_raw.log")
for label in warmup_64 warmup_128 warmup_256; do
    eff=$(compute_loss_efficiency "$RESULT_DIR/attempt_${label}_raw.log")
    echo "$label: loss_efficiency=$eff (baseline=$baseline_eff)"
done
```

Discard any candidate where `loss_efficiency < baseline_eff × 0.85`.

**Stage 2: Tier 2.5 projected TTT for survivors (1500 iters)**

```bash
run_mlperf_trial "warmup_<best>_proj" 2.5 "" "PRIMUS_LR_WARMUP_ITERS=<best>"
projected=$(project_ttt "$RESULT_DIR/attempt_warmup_<best>_proj_raw.log" "$GBS")
ttt_gain=$(compute_ttt_gain_pct "$BASELINE_PROJECTED_TTT" "$(echo $projected | cut -f1)")
echo "Warmup <best>: projected_ttt gain = ${ttt_gain}%"
```

**Decision metric:** `projected_ttt` from Tier 2.5, not `eval_loss` at iteration 500. KEEP only if `ttt_gain_pct > 0%`.

**NaN gate:** If warmup < 128 causes NaN in the first 100 iterations (`TRIAL_STATUS`), discard immediately.

### Dimension 3: Weight decay

Weight decay affects regularization strength. Lower values may speed early convergence but risk late-stage instability near the target.

**Stage 1: Tier 2 quick filter (500 iters)**

```bash
run_mlperf_trial "wd_001" 2 500 "PRIMUS_WEIGHT_DECAY=0.01"
run_mlperf_trial "wd_005" 2 500 "PRIMUS_WEIGHT_DECAY=0.05"
run_mlperf_trial "wd_02" 2 500 "PRIMUS_WEIGHT_DECAY=0.2"

for label in wd_001 wd_005 wd_02; do
    eff=$(compute_loss_efficiency "$RESULT_DIR/attempt_${label}_raw.log")
    echo "$label: loss_efficiency=$eff"
done
```

Discard if `loss_efficiency < baseline × 0.85`.

**Stage 2: Tier 2.5 projected TTT for survivors**

```bash
run_mlperf_trial "wd_<best>_proj" 2.5 "" "PRIMUS_WEIGHT_DECAY=<best>"
projected=$(project_ttt "$RESULT_DIR/attempt_wd_<best>_proj_raw.log" "$GBS")
ttt_gain=$(compute_ttt_gain_pct "$BASELINE_PROJECTED_TTT" "$(echo $projected | cut -f1)")
```

KEEP only if `ttt_gain_pct > 0%`.

### Dimension 4: Gradient clipping

With FP8 hybrid precision, gradient magnitudes can spike. Tighter clipping reduces wasted iterations from loss spikes.

```bash
run_mlperf_trial "clip_05" 2 500 "PRIMUS_CLIP_GRAD=0.5"
run_mlperf_trial "clip_15" 2 500 "PRIMUS_CLIP_GRAD=1.5"
```

Only test if baseline Tier 2L runs showed loss spikes.

### Dimension 5: LR schedule (decay window, floor, peak)

Three sub-dimensions interact: **decay window**, **min LR floor**, and **peak LR**. Tune in this order; later knobs assume earlier ones are fixed.

#### 5A: LR decay window (`lr_decay_iters`)

If `lr_decay_iters` is much larger than projected convergence iters (e.g. default 1.2M vs ~7200 actual iters), the cosine schedule is essentially flat — LR barely decays during training. However, setting `lr_decay_iters` too close to convergence iters (1.0–1.2×) is also problematic: cosine decay reaches near-`min_lr` by the convergence point, leaving the model with insufficient per-step learning capacity in the critical final phase where it approaches the loss target. For MLPerf's "reach target loss ASAP" objective, each late-stage iteration at very low LR wastes wall time.

The optimal range is **2–5×** projected convergence iters. This provides meaningful annealing (LR at convergence ≈ 55–91% of peak) while preserving enough late-stage learning rate to efficiently push toward the target.

| Multiplier | LR at convergence (% of peak) | Character |
|------------|-------------------------------|-----------|
| 1.0–1.2× | 10–19% | Too aggressive — late-stage stall |
| **2×** | **~55%** | **Strong annealing, good balance** |
| **3×** | **~78%** | **Moderate annealing, safe default** |
| **5×** | **~91%** | **Gentle annealing** |
| 10×+ | >97% | Essentially no decay |

**Trial plan:** Test three decay windows spanning the 2–5× range.

```bash
source "$SKILL_ROOT/scripts/common.sh"
converge_est=$((state_baseline_projected_iters))  # e.g. 7200

run_mlperf_trial "decay_2x" 2L "" \
    "PRIMUS_LR_DECAY_ITERS=$((converge_est * 2))"

run_mlperf_trial "decay_3x" 2L "" \
    "PRIMUS_LR_DECAY_ITERS=$((converge_est * 3))"

run_mlperf_trial "decay_5x" 2L "" \
    "PRIMUS_LR_DECAY_ITERS=$((converge_est * 5))"
```

**IMPORTANT:** This change **requires** Tier 2L (2500+ iters). Tier 2 (500 iters) is not informative — all decay windows look the same in warmup/early training. Compare `projected_ttt` from Tier 2L.

#### 5B: MIN_LR floor (`min_lr`)

After the decay window is set, `min_lr` determines the LR floor at the end of the cosine schedule. The optimal floor depends on the decay window chosen in 5A: a wider window (e.g. 5×) already keeps late-stage LR high, so min_lr matters less; a tighter window (e.g. 2×) makes min_lr the dominant factor for late-stage learning capacity. Explore **both** lower and higher floors relative to the baseline (10% of peak).

| MIN_LR | Ratio to LR | Effect |
|--------|-------------|--------|
| 1.0e-5 | 2.5% | Deep decay — aggressive late-stage annealing |
| 2.0e-5 | 5% | Moderate-low floor |
| 4.0e-5 | 10% | Baseline default |
| 1.2e-4 | 30% | Elevated floor — preserves late-stage learning capacity |
| 2.0e-4 | 50% | High floor — minimal effective LR range, use with tight (2×) decay window |

```bash
run_mlperf_trial "min_lr_1e5" 2L "" \
    "PRIMUS_LR_DECAY_ITERS=<best_from_5A> PRIMUS_MIN_LR=1.0e-5"
run_mlperf_trial "min_lr_2e5" 2L "" \
    "PRIMUS_LR_DECAY_ITERS=<best_from_5A> PRIMUS_MIN_LR=2.0e-5"
run_mlperf_trial "min_lr_12e5" 2L "" \
    "PRIMUS_LR_DECAY_ITERS=<best_from_5A> PRIMUS_MIN_LR=1.2e-4"
run_mlperf_trial "min_lr_2e4" 2L "" \
    "PRIMUS_LR_DECAY_ITERS=<best_from_5A> PRIMUS_MIN_LR=2.0e-4"
```

If 5A chose a tight window (2×), prioritize the higher-floor trials (1.2e-4, 2.0e-4). If 5A chose a wide window (5×), prioritize the lower-floor trials (1e-5, 2e-5) to explore whether deeper annealing helps.

#### 5C: Peak LR (`lr`)

With a meaningful annealing schedule, the model may tolerate a higher peak LR. Test after 5A and 5B.

```bash
run_mlperf_trial "lr_5e4" 2L "" \
    "PRIMUS_LR=5.0e-4 PRIMUS_MIN_LR=<best> PRIMUS_LR_DECAY_ITERS=<best>"
run_mlperf_trial "lr_6e4" 2L "" \
    "PRIMUS_LR=6.0e-4 PRIMUS_MIN_LR=<best> PRIMUS_LR_DECAY_ITERS=<best>"
```

**NaN gate:** Higher LR with FP8 increases NaN risk. If NaN occurs in the first 200 iters, discard and keep the current LR.

**Decision:** Across 5A→5B→5C survivors, keep the combination with lowest `projected_ttt` from Tier 2L.

### Combined validation

Apply all winning convergence tweaks together:

```bash
run_mlperf_trial "convergence_combined" 2L "" \
    "PRIMUS_LR=<best> PRIMUS_MIN_LR=<best> \
     PRIMUS_LR_DECAY_ITERS=<best> PRIMUS_LR_WARMUP_ITERS=<best> \
     PRIMUS_WEIGHT_DECAY=<best> EVAL_SAMPLES_INTERVAL=<best>"
projected=$(project_ttt "$RESULT_DIR/attempt_convergence_combined_raw.log" "$GBS")
echo "Projected TTT: $projected"
```

Compare against baseline projected TTT.

## Outputs

- Optimal `eval_interval`, `warmup_iters`, `weight_decay`, `clip_grad`
- Optimal `lr`, `min_lr`, `lr_decay_iters` (from Dimension 5)
- Projected TTT savings (seconds and percentage)
- Updated environment variables in `state.kept_env_vars`
- Updated `state.optimal_eval_interval`
- Updated `state.eval_overhead_seconds`
- Updated `state.lr_decay_iters`, `state.min_lr`, `state.lr` (if changed)

## Heuristic Update

- LR decay window improvement: boost `min_lr` and `peak_lr` sub-dimensions by 1.5× (Score Update Rule #12)
- After convergence-speed completes: update `state.eval_overhead_seconds`, `state.optimal_eval_interval` (Rule #9)
- If no dimension improves projected TTT: reduce convergence-speed score by 0.7×

## Failure Handling

- If aggressive warmup (< 128) causes NaN: revert; keep default 128
- If weight decay change degrades the eval loss trajectory: revert; keep default 0.1
- If eval interval change does not affect the loss trajectory (expected): keep the interval that minimizes total wall time
- These are low-risk tunings — if none improve projected TTT, keep defaults
