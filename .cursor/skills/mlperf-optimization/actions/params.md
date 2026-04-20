# Action: Training Parameter Tuning

## Overview

Explores training configuration parameters that affect per-iteration performance
without changing hyperparameters or parallelism. FP8 knobs: see
[fp8-recipe-tuning.md](fp8-recipe-tuning.md). Gradient clipping: see
[convergence-speed.md](convergence-speed.md).

## Inputs

- Current config and kept_overrides
- Profile data

## KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B training parameters MBS recompute" --top-k 5 --compact
```

## Parameter Matrix

### Memory / Batch Parameters

| Parameter | Range | Impact | Notes |
|-----------|-------|--------|-------|
| `micro_batch_size` | 1–8 | Medium | Larger MBS = fewer GA steps. Must maintain GBS |
| `recompute_granularity` | none/selective/full | Medium | Trades compute for memory |

### Communication Overlap

| Parameter | Current | Try | Notes |
|-----------|---------|-----|-------|
| `overlap_grad_reduce` | true | — | Already enabled |
| `overlap_param_gather` | true | — | Already enabled |
| `use_distributed_optimizer` | true | — | Already enabled |

### PrimusTurbo Parameters

| Parameter | Current | Range | Notes |
|-----------|---------|-------|-------|
| `enable_primus_turbo` | true | — | Master switch, already on |
| `turbo_sync_free_moe_stage` | 0 | 0–3 | Sync-free MoE pipeline |
| `turbo_deepep_num_cu` | 64 | 32–128 | CUs for DeepEP |

## Procedure

### Step 1: Prioritize based on profile

```python
param_priority = []

# MBS tuning (if GA > 2)
if ga_steps > 2 and mbs < 4:
    param_priority.append(("micro_batch_size", 4, "Reduce GA overhead"))

# Sync-free MoE
if turbo_sync_free_moe_stage == 0:
    param_priority.append(("turbo_sync_free_moe_stage", 2, "Enable sync-free MoE"))
```

### Step 2: Test each parameter (Tier 1 → Tier 2)

For each candidate, modify YAML or env var and run a Tier 1 trial:

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "param_NAME" 1
eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_param_NAME.log)")"
gain=$(compute_gain_pct "$baseline_ms_per_iter" "$TRIAL_MS_PER_ITER")
```

For winners (gain > 1%), validate with Tier 2:

```bash
run_mlperf_trial "param_NAME_validate" 2 500
```

### Step 3: GBS verification for MBS changes

When changing MBS, verify GBS is maintained:
```
new_ga = GBS / (new_mbs × dp)
assert new_ga * new_mbs * dp == baseline_gbs
```

## Outputs
- `winning_params`: list of parameters that improved ms/iter
- Per-parameter gain percentages
- Updated config

## Heuristic Update

- MBS improvement: boost remaining MBS candidates
- Sync-free MoE gain > 1%: boost turbo_sync_free_moe_stage=3 candidate
- All params neutral: reduce params score by 0.7x

## Failure Handling

- OOM on larger MBS: revert, try intermediate value
- Loss divergence: revert to prior config
