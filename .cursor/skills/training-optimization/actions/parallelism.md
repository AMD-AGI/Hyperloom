# Action: Parallelism Configuration

## Overview

Explores tensor parallelism (TP), pipeline parallelism (PP), expert parallelism (EP),
data parallelism (DP), and context parallelism (CP) configurations. Changes to parallelism
can significantly affect communication overhead and memory efficiency.

**Caution:** Parallelism changes have **higher crash risk** than fusion flags (new NCCL
groups, different memory layout). Always verify GBS after each change.

## Inputs
- Current parallelism: `tp`, `pp`, `ep`, `dp`
- `NUM_GPUS` total available
- Model class from classify step
- Profile data (communication fraction)

## Parallelism Search Space

### Constraint: `TP × PP × EP_effective ≤ NUM_GPUS`

Note: EP does not consume GPUs in the same way — with `expert_model_parallel_size=8` and
`tensor_model_parallel_size=1`, all 8 GPUs participate in EP. The effective constraint is
`TP × PP ≤ NUM_GPUS / max(1, DP)` where `DP = NUM_GPUS / (TP × PP)`.

### Exploration matrix for 8 GPUs

| Config | TP | PP | EP | DP | Best for |
|--------|----|----|----|----|----------|
| EP-only | 1 | 1 | 8 | 8 | MoE models (default GPT-OSS) |
| TP2+EP4 | 2 | 1 | 4 | 4 | Large hidden dims, reduce GEMM per GPU |
| TP4+EP2 | 4 | 1 | 2 | 2 | Very large hidden dims |
| TP2+PP2+EP2 | 2 | 2 | 2 | 2 | Very large models (memory constrained) |
| TP8 | 8 | 1 | 1 | 1 | Dense models (no MoE) |

### For dense (non-MoE) models

| Config | TP | PP | DP | Best for |
|--------|----|----|-----|----------|
| TP8 | 8 | 1 | 1 | Largest per-GPU compute |
| TP4+DP2 | 4 | 1 | 2 | Balance compute and communication |
| TP2+DP4 | 2 | 1 | 4 | Communication-light workloads |

## Procedure

### Step 1: Evaluate current config against profile

```python
comm_pct = categories["communication"]
if comm_pct < 10:
    # Communication is small — parallelism changes unlikely to help much
    # Reduce priority, but still worth testing if TP > 1
    pass
elif comm_pct > 20:
    # Communication-heavy — parallelism tuning is high priority
    # Consider reducing TP (fewer AllReduce) or increasing EP (AlltoAll may be cheaper)
    pass
```

### Step 2: Test alternative configs

For each candidate parallelism config:

```bash
pkill -9 -f "primus/cli/main.py" 2>/dev/null; sleep 5
MASTER_PORT=$((MASTER_PORT + 1))

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m primus.cli.main train pretrain \
  --config "$CONFIG_YAML" \
  $KEPT_OVERRIDES \
  tensor_model_parallel_size=$NEW_TP \
  pipeline_model_parallel_size=$NEW_PP \
  expert_model_parallel_size=$NEW_EP \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/attempt_parallel_N.log
```

### Step 3: Verify GBS and measure

**CRITICAL:** Parallelism changes affect DP, which affects GBS calculation.
When changing TP/PP, you may need to adjust `micro_batch_size` or
`gradient_accumulation_steps` to maintain the same GBS.

```
GBS = micro_batch_size × data_parallel_size × gradient_accumulation_steps
DP = NUM_GPUS / (TP × PP)
```

If DP changes, adjust MBS or GA to keep GBS constant.

### Step 4: Communication tuning (NCCL env vars)

If parallelism config is kept, try NCCL environment tuning:

| Variable | Options | Impact |
|----------|---------|--------|
| `NCCL_ALGO` | `Ring`, `Tree`, `CollNet` | AllReduce algorithm |
| `NCCL_PROTO` | `Simple`, `LL`, `LL128` | Protocol selection |
| `NCCL_MIN_NCHANNELS` | `4`, `8`, `16` | Minimum channels |
| `RCCL_MSCCL_ENABLE` | `0`, `1` | MSCCL acceleration |

## Outputs
- Best parallelism config (TP, PP, EP, DP)
- Communication overhead change
- Updated `KEPT_OVERRIDES` with parallelism flags

## Failure Handling
- If config crashes (NCCL init failure): likely invalid TP/PP/EP combo, skip
- If GBS changes: recalculate MBS/GA to maintain GBS, retry
- If OOM: reduce micro_batch_size (maintaining GBS via more GA steps)
