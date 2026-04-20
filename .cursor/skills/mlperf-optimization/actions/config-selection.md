# Action: Config Selection (GBS / LR / TP / EP / DP)

## Overview

Selects the optimal GBS/LR/EP/TP/DP combination using a three-stage elimination workflow. GBS, LR, and parallelism interact, so they are tested jointly. Tier 1 filters crashes and measures ms/iter; Tier 3 compares convergence; Tier 4 verifies full time-to-train on the top two.

## Inputs

- Baseline TTT and ms/iter (from Tier 4 baseline full run)
- Current config: GBS=32, LR=4.0e-4, EP=1, DP=8, TP=1
- Profile data (compute vs communication breakdown)

## KB Query

```bash
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B GBS LR parallelism config" --top-k 5 --compact
```

## Procedure

The matrix is intentionally small (6 configs). LR is coupled to GBS via sqrt scaling — it is not swept independently.

`LR = 4.0e-4 × sqrt(GBS / 32)` · `MIN_LR = LR / 10` · `WARMUP_ITERS = 128 × (32 / GBS)` · `GA_STEPS = GBS / (MBS × DP)` with MBS=2.

EP: set `expert_model_parallel_size`, `moe_enable_deepep: true`, `use_turbo_deepep: true`, `turbo_deepep_num_cu: 64`. TP: set `tensor_model_parallel_size`, then `DP = GPUS / (TP × PP)` and `GA = GBS / (MBS × DP)` for the target GBS.

### Candidate Matrix

| ID | GBS | LR | EP | TP | DP | GA | Notes |
|----|-----|----|----|----|----|----|-------|
| A (baseline) | 32 | 4.0e-4 | 1 | 1 | 8 | 2 | Current config |
| B | 16 | 2.0e-4 | 1 | 1 | 8 | 1 | Half GBS, LR scaled by 0.5× |
| C | 32 | 4.0e-4 | 8 | 1 | 8 | 2 | EP=8 with DeepEP, same GBS |
| D | 16 | 2.0e-4 | 8 | 1 | 8 | 1 | EP=8, half GBS |
| E | 32 | 4.0e-4 | 4 | 2 | 4 | 4 | TP=2, EP=4, DP shrinks to 4, GA=4 to maintain GBS |
| F | 16 | 2.0e-4 | 4 | 2 | 4 | 2 | TP=2, EP=4, half GBS |

### Stage 1: Tier 1 Quick Filter (100 iters per candidate)

Run each candidate for 100 iterations. Purpose: crash detection, NaN detection, ms/iter measurement. Discard any candidate that crashes, OOMs, or produces NaN loss.

```bash
source "$SKILL_ROOT/scripts/common.sh"

# For each candidate:
# 1. Apply parallelism changes to YAML if needed (EP/TP configs)
# 2. Run Tier 1 trial with GBS/LR overrides
# 3. Parse TRIAL_RESULT
# 4. Revert YAML changes before next candidate

# Candidate A (baseline — already measured, reuse existing result)

# Candidate B: GBS=16
run_mlperf_trial "cfg_B_gbs16" 1 "" \
    "PRIMUS_GLOBAL_BATCH_SIZE=16 PRIMUS_LR=2.0e-4 PRIMUS_MIN_LR=2.0e-5 PRIMUS_LR_WARMUP_ITERS=256"

# Candidate C: EP=8, GBS=32
# (first modify YAML for EP=8, then run)
run_mlperf_trial "cfg_C_ep8_gbs32" 1

# Candidate D: EP=8, GBS=16
run_mlperf_trial "cfg_D_ep8_gbs16" 1 "" \
    "PRIMUS_GLOBAL_BATCH_SIZE=16 PRIMUS_LR=2.0e-4 PRIMUS_MIN_LR=2.0e-5 PRIMUS_LR_WARMUP_ITERS=256"

# Candidate E: TP=2, EP=4, GBS=32
# (first modify YAML for TP=2/EP=4, then run)
run_mlperf_trial "cfg_E_tp2ep4_gbs32" 1

# Candidate F: TP=2, EP=4, GBS=16
run_mlperf_trial "cfg_F_tp2ep4_gbs16" 1 "" \
    "PRIMUS_GLOBAL_BATCH_SIZE=16 PRIMUS_LR=2.0e-4 PRIMUS_MIN_LR=2.0e-5 PRIMUS_LR_WARMUP_ITERS=256"
```

Discard candidates with nan/no_data/crash. Revert YAML between candidates.

### Stage 1.5: Tier 2 Loss Efficiency Filter (500 iters per survivor)

Run each Stage 1 survivor for 500 iterations to compute `loss_efficiency`. This is a quick filter to eliminate candidates that are clearly worse at convergence before investing in costly Tier 3 runs.

```bash
source "$SKILL_ROOT/scripts/common.sh"

baseline_eff=$(compute_loss_efficiency "$RESULT_DIR/attempt_cfg_A_eff_raw.log")

for candidate in survivors; do
    run_mlperf_trial "cfg_${candidate}_eff" 2
    eff=$(compute_loss_efficiency "$RESULT_DIR/attempt_cfg_${candidate}_eff_raw.log")
    echo "Candidate $candidate: loss_efficiency=$eff (baseline=$baseline_eff)"
done
```

**Decision:** Discard any candidate whose `loss_efficiency < baseline_loss_efficiency × 0.7`
(30% worse). This is a looser threshold than the DFS loop's 15% because config-selection
candidates involve larger changes (GBS/EP) where early loss trajectory is less predictive.

### Stage 2: Tier 3 Long Convergence Comparison (2500 iters per survivor)

Run each surviving candidate for 2500 iterations. The dense eval schedule (every 50
iters) produces loss-vs-samples curves for direct comparison.

```bash
# For each surviving candidate:
# 1. Apply parallelism changes to YAML if needed
# 2. Run Tier 3 trial with GBS/LR overrides

run_mlperf_trial "cfg_A_t3" 3
run_mlperf_trial "cfg_B_t3" 3 "" \
    "PRIMUS_GLOBAL_BATCH_SIZE=16 PRIMUS_LR=2.0e-4 PRIMUS_MIN_LR=2.0e-5 PRIMUS_LR_WARMUP_ITERS=256"
# ... etc for each survivor
```

**After each Tier 3 trial completes, extract:**

1. **ms/iter** — from TRIAL_RESULT
2. **samples/sec** — `GBS / (ms_per_iter / 1000)`
3. **Loss-vs-samples curve** — from `train_loss` MLLOG events in raw log:
   ```bash
   extract_losses "$RESULT_DIR/attempt_cfg_A_t3_raw.log"
   ```
4. **Eval loss trajectory** — from `eval_accuracy` MLLOG events
5. **Projected TTT** — from `project_ttt()` helper:
   ```bash
   project_ttt "$RESULT_DIR/attempt_cfg_A_t3_raw.log" 32
   ```

**Ranking:**

1. Primary: `projected_ttt` from `project_ttt()` (lower = better)
2. Tiebreaker: `loss_efficiency` from `compute_loss_efficiency()` (higher = better)
3. Sanity check: ms/iter must be within 2× of baseline (discard gross regressions)

**Select top 2 candidates by projected TTT.**

| Candidate | ms/iter | samples/sec | eval_loss@2500 | projected TTT | Rank |
|-----------|---------|-------------|----------------|---------------|------|
| A | ... | ... | ... | ... | ... |
| B | ... | ... | ... | ... | ... |

### Stage 3: Tier 4 Top-2 Full Verification

Run only the top-2 candidates to full convergence. The actual TTT is the ground truth.

```bash
# Top-1 candidate:
# Apply its config (YAML + env), then:
run_mlperf_trial "cfg_winner1_full" 4

# Top-2 candidate:
# Apply its config, then:
run_mlperf_trial "cfg_winner2_full" 4
```

**Do NOT interrupt these runs.** Wait for `run_stop` with `status=success` or
`status=aborted`. Extract actual TTT from each.

```bash
extract_time_to_train "$RESULT_DIR/attempt_cfg_winner1_full_raw.log"
extract_time_to_train "$RESULT_DIR/attempt_cfg_winner2_full_raw.log"
```

**The candidate with the lowest actual TTT wins.** If both `status=aborted`
(did not converge), the one with the lower final eval_loss is preferred, but
document this as a convergence failure.

### Post-Selection

After selecting the winner:

1. Apply the winning config permanently:
   - Update YAML for parallelism (EP, TP, DeepEP settings)
   - Record winning env vars in `state.kept_env_vars`
   - Update `state.global_batch_size`, `state.tp`, `state.ep`
2. Re-run baseline with winning config (Tier 1) to establish the new baseline ms/iter
   (a full Tier 4 re-baseline is only required if the winning config is substantially different)
3. All subsequent actions (sweep, report) use the winning config

## Outputs

- Winning config (GBS, LR, EP, TP, DP)
- Stage 1 survival table
- Stage 2 comparison table with projected TTT
- Stage 3 actual TTT for top-2 candidates
- Updated state with new baseline

## Heuristic Update

- Winning config differs from baseline: re-baseline ms/iter, update state
- All candidates worse than baseline: keep baseline, skip Stage 3
- After selection: all subsequent actions use winning config

## Failure Handling

- If all candidates except baseline crash in Stage 1: keep baseline, skip Stage 2/3
- If projected TTT for all candidates is worse than baseline: keep baseline, skip Stage 3
- If Stage 3 winner has `status=aborted`: document in report, consider increasing `PRIMUS_TRAIN_ITERS`
- If YAML modification for EP/TP causes NCCL init failure: skip that parallelism config
- Always revert YAML to baseline between candidates to avoid cross-contamination
