# Action: Parameter Sweep

## Overview

After the DFS optimization loop completes, run a systematic sweep over key parameters
to find the optimal operating point. This is a read-only measurement phase — no new
optimizations are applied.

## Inputs
- Final `KEPT_OVERRIDES` and `KEPT_PATCHES` from DFS loop
- `$CONFIG_YAML`, `$NUM_GPUS`, `$PRIMUS_ROOT`

## Sweep Dimensions

### Primary: Micro Batch Size

| MBS | GA Steps | Notes |
|-----|----------|-------|
| 1 | GBS/DP | Maximum GA overhead, minimum memory |
| 2 | GBS/(2×DP) | — |
| 4 | GBS/(4×DP) | Typical default |
| 8 | GBS/(8×DP) | Less GA overhead, more memory |
| 16 | GBS/(16×DP) | Minimum GA, may OOM |

**Constraint:** `MBS × DP × GA = GBS` must hold for every row.

### Secondary: Precision (if user allows)

| Precision | Config | Notes |
|-----------|--------|-------|
| BF16 | Default | Baseline precision |
| FP8 | `--config ...-FP8-pretrain.yaml` | 2× memory savings, may need separate config |

### Tertiary: Recompute Strategy

| Strategy | Config | Notes |
|----------|--------|-------|
| None | `recompute_granularity=none` | Fastest if memory allows |
| Selective | `recompute_granularity=selective` | Recompute attention only |
| Full | `recompute_granularity=full` | Maximum memory savings |

## Procedure

### Step 1: Build sweep configs

```python
sweep_configs = []
for mbs in [1, 2, 4, 8, 16]:
    ga = baseline_gbs // (mbs * dp)
    if ga < 1 or mbs * dp * ga != baseline_gbs:
        continue
    sweep_configs.append({
        "micro_batch_size": mbs,
        "gradient_accumulation_steps": ga,
        "label": f"mbs{mbs}_ga{ga}"
    })
```

### Step 2: Run each config

```bash
for config in sweep_configs:
    pkill -9 -f "primus/cli/main.py" 2>/dev/null; sleep 3
    MASTER_PORT=$((MASTER_PORT + 1))

    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
      -m primus.cli.main train pretrain \
      --config "$CONFIG_YAML" \
      $KEPT_OVERRIDES \
      micro_batch_size=${config.mbs} \
      profile=false use_pytorch_profiler=false \
      2>&1 | tee "/tmp/sweep_${config.label}.log"
done
```

### Step 3: Collect results

```
cat >> "$RESULT_DIR/sweep_results.tsv" <<EOF
mbs	ga_steps	ms_per_iter	samples_per_sec	memory_gb	status
EOF
```

Append each sweep result as a TSV row.

### Step 4: Find Pareto-optimal configs

Identify configs that are Pareto-optimal on the ms/iter vs memory frontier.
The best config is the one with lowest ms/iter that fits in GPU memory.

## Outputs
- `$RESULT_DIR/sweep_results.tsv`: full sweep data
- Pareto-optimal configurations
- Recommended production config

## Failure Handling
- OOM on large MBS: record as `oom`, skip
- Ensure GBS is verified for every sweep point
