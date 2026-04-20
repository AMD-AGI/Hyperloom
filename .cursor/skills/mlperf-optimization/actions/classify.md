# Action: Model Classification

## Overview

Parses the training config to identify model architecture, parallelism topology,
and batch configuration, then sets initial DFS heuristic score priors.

## Inputs
- `$EXP` — training YAML config path

## KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B architecture classification" --top-k 5 --compact
```

## Procedure

### Step 1: Parse the training config

```python
import yaml, os

config_path = os.environ.get("EXP", "/root/mlperf_primus/conf/gpt_oss_20B-pretrain-fp8.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

overrides = config.get("modules", {}).get("pre_trainer", {}).get("overrides", config)
```

### Step 2: Classify architecture

GPT-OSS-20B is a MoE model with GQA and sliding window attention:

```python
num_experts = overrides.get("num_experts", 32)
num_kv_heads = overrides.get("num_query_groups", 8)
num_q_heads = overrides.get("num_attention_heads", 64)
has_moe = num_experts > 1
has_gqa = num_kv_heads > 0 and num_kv_heads < num_q_heads
has_swa = overrides.get("window_size", None) is not None

model_class = "moe_gqa_swa"  # GPT-OSS-20B is MoE + GQA + SWA
```

### Step 3: Extract parallelism and batch config

```python
tp = int(os.environ.get("PRIMUS_TP", overrides.get("tensor_model_parallel_size", 1)))
pp = int(os.environ.get("PRIMUS_PP", overrides.get("pipeline_model_parallel_size", 1)))
ep = int(os.environ.get("PRIMUS_EP", overrides.get("expert_model_parallel_size", 1)))
dp = int(os.environ.get("GPUS_PER_NODE", 8)) // (tp * pp)

gbs = int(os.environ.get("PRIMUS_GLOBAL_BATCH_SIZE", overrides.get("global_batch_size", 32)))
mbs = int(os.environ.get("PRIMUS_MICRO_BATCH_SIZE", overrides.get("micro_batch_size", 2)))
seq_len = overrides.get("seq_length", 8192)
ga_steps = gbs // (mbs * dp)
```

### Step 4: Determine optimization strategy

```python
priors = {
    "fusion-flags": 9,        # Highest for MoE
    "config-selection": 8,    # GBS/LR/parallelism joint tuning
    "convergence-speed": 7,   # LR schedule, warmup, eval interval
    "fp8-recipe-tuning": 6,   # FP8 knobs
    "runtime-tunables": 5,    # System knobs
    "params": 5,              # MBS, recompute, overlap
    "comm-tuning": 6,         # NCCL/RCCL/DeepEP/AllReduce (always runs)
    "kernel-opt": 5,          # All non-vendor kernels explored
    "sweep": 1,               # Final exploration
}
```

### Step 5: Identify key config properties

```python
fp8_mode = overrides.get("fp8", os.environ.get("PRIMUS_FP8_RECIPE", "hybrid"))
enable_primus_turbo = overrides.get("enable_primus_turbo", True)
moe_permute_fusion = overrides.get("moe_permute_fusion", True)
gradient_accumulation_fusion = overrides.get("gradient_accumulation_fusion", True)
moe_enable_deepep = overrides.get("moe_enable_deepep", False)
```

## Outputs
- `model_class`: `moe_gqa_swa` (GPT-OSS-20B)
- Parallelism topology: `tp`, `pp`, `ep`, `dp`
- Batch config: `gbs`, `mbs`, `seq_len`, `ga_steps`
- FP8 mode, Turbo status
- Heuristic score priors
- Flags already enabled (to avoid redundant testing)

## Heuristic Update

N/A — classify sets initial priors, does not update them.

## Failure Handling

- If config cannot be parsed: check YAML syntax, check env var interpolation
- If parallelism values don't match GPU count: verify PRIMUS_TP/PP/EP env vars
