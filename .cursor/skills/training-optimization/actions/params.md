# Action: Training Parameter Tuning

## Overview

Explores training configuration parameters that affect performance without changing the
model computation. These are lower-risk than parallelism changes but can still yield
meaningful gains.

## Inputs
- Current config and kept_overrides
- Model class, parallelism config
- Profile data

## Parameter Matrix

### Memory / Batch Parameters

| Parameter | Range | Impact | Notes |
|-----------|-------|--------|-------|
| `micro_batch_size` | 1–16 | Medium | Larger MBS = fewer GA steps = less overhead. Must maintain GBS |
| `sequence_length` | 2048–8192 | High | Only change if explicitly allowed by user |
| `recompute_granularity` | `none`, `selective`, `full` | Medium | Trades compute for memory |
| `recompute_method` | `uniform`, `block` | Low | How activation checkpointing is distributed |
| `recompute_num_layers` | 0–N | Medium | Number of layers to checkpoint |

### Communication Parameters

| Parameter | Range | Impact | Notes |
|-----------|-------|--------|-------|
| `overlap_grad_reduce` | true/false | Low–Medium | Overlap gradient all-reduce with backward |
| `overlap_param_gather` | true/false | Low–Medium | Overlap param gather with forward |
| `use_distributed_optimizer` | true/false | Low | ZeRO-style optimizer sharding |
| `async_tensor_model_parallel_allreduce` | true/false | Low | Async TP all-reduce |

### PrimusTurbo Parameters

| Parameter | Range | Impact | Notes |
|-----------|-------|--------|-------|
| `enable_primus_turbo` | true/false | High | Master switch for Triton kernels |
| `turbo_deepep_num_cu` | 32–128 | Low–Medium | CUs allocated to DeepEP communication |
| `turbo_deepep_use_comm_stream` | true/false | Low | Separate stream for EP communication |
| `turbo_sync_free_moe_stage` | 0–3 | Low–Medium | Sync-free MoE pipeline stages |

### Environment Variables

| Variable | Values | Impact | Notes |
|----------|--------|--------|-------|
| `PYTORCH_TUNABLEOP_ENABLED` | 0/1 | Variable | GEMM autotuning. **VERY slow** (30+ min). Offline only |
| `TORCH_NCCL_BLOCKING_WAIT` | 0/1 | Low | Blocking vs non-blocking NCCL |
| `NCCL_BUFFSIZE` | 4M–128M | Low | NCCL buffer size |

## Procedure

### Step 1: Prioritize based on profile

```python
param_priority = []

# If MBS is small and GA steps > 4, try larger MBS
if ga_steps > 4 and mbs < 8:
    param_priority.append(("micro_batch_size", mbs * 2, "Reduce GA overhead"))

# If recompute is off and memory allows, keep it off
# If recompute is on, try selective instead of full
if current_recompute == "full":
    param_priority.append(("recompute_granularity", "selective", "Less recompute overhead"))

# If overlap flags are off, try turning them on
if not overlap_grad_reduce:
    param_priority.append(("overlap_grad_reduce", True, "Overlap comm with compute"))
```

### Step 2: Test each parameter

For each candidate:
```bash
pkill -9 -f "primus/cli/main.py" 2>/dev/null; sleep 3
MASTER_PORT=$((MASTER_PORT + 1))

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m primus.cli.main train pretrain \
  --config "$CONFIG_YAML" \
  $KEPT_OVERRIDES \
  <param>=<value> \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/attempt_param_N.log
```

### Step 3: GBS verification for MBS changes

**CRITICAL:** When changing `micro_batch_size`, verify:
```
new_ga = GBS / (new_mbs × dp)
assert new_ga * new_mbs * dp == baseline_gbs
```

If the framework auto-adjusts GA, verify from log. If not, explicitly set
`gradient_accumulation_steps` override.

## Outputs
- `winning_params`: list of parameters that improved ms/iter
- Per-parameter gain percentages
- Updated `KEPT_OVERRIDES`

## Failure Handling
- OOM on larger MBS: revert, try intermediate value
- Crash on recompute change: incompatible with certain model features
- GBS mismatch: recalculate GA steps
