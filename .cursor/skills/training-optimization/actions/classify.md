# Action: Model Classification

## Inputs
- `$CONFIG_YAML` — training YAML config

## Procedure

### Step 1: Parse the training config

Read the YAML config to extract model architecture details:

```python
import yaml

with open(CONFIG_YAML) as f:
    config = yaml.safe_load(f)

# Navigate nested config (Primus uses module-level nesting)
overrides = config.get("modules", {}).get("pre_trainer", {}).get("overrides", config)
model_ref = config.get("modules", {}).get("pre_trainer", {}).get("model", "")
```

### Step 2: Load model-specific config

```python
# Resolve model YAML (e.g., gpt_oss_20B.yaml)
import os
model_yaml_path = os.path.join(PRIMUS_ROOT, "primus/configs/models/megatron", model_ref)
if os.path.exists(model_yaml_path):
    with open(model_yaml_path) as f:
        model_config = yaml.safe_load(f)
```

### Step 3: Classify architecture

```python
has_moe = model_config.get("num_experts", 0) > 1 or overrides.get("num_experts", 0) > 1
num_kv_heads = model_config.get("num_query_groups", model_config.get("num_key_value_heads", 0))
num_q_heads = model_config.get("num_attention_heads", 0)
has_gqa = num_kv_heads > 0 and num_kv_heads < num_q_heads
has_mla = model_config.get("multi_latent_attention", False) or overrides.get("multi_latent_attention", False)
has_swa = model_config.get("sliding_window", 0) > 0 or overrides.get("sink_sliding_window", 0) > 0

if has_moe and has_mla:
    model_class = "moe_mla"
elif has_moe and has_swa:
    model_class = "moe_swa"
elif has_moe:
    model_class = "moe_gqa"
else:
    model_class = "dense"
```

### Step 4: Extract parallelism and batch config

```python
tp = overrides.get("tensor_model_parallel_size", 1)
pp = overrides.get("pipeline_model_parallel_size", 1)
ep = overrides.get("expert_model_parallel_size", 1)
dp = NUM_GPUS // (tp * pp)

gbs = overrides.get("global_batch_size", 512)
mbs = overrides.get("micro_batch_size", 8)
seq_len = overrides.get("seq_length", 4096)
ga_steps = gbs // (mbs * dp)
```

### Step 5: Determine optimization strategy

```python
# Score priors from SKILL.md table
if model_class == "moe_gqa":
    priors = {"fusion-flags": 9, "parallelism": 6, "params": 5, "kernel-opt": 3, "attention-backend": 4}
elif model_class == "moe_mla":
    priors = {"fusion-flags": 8, "parallelism": 6, "params": 5, "kernel-opt": 2, "attention-backend": 4}
elif model_class == "dense":
    priors = {"fusion-flags": 5, "parallelism": 4, "params": 3, "kernel-opt": 7, "attention-backend": 6}
else:
    priors = {"fusion-flags": 8, "parallelism": 6, "params": 5, "kernel-opt": 2, "attention-backend": 7}

primary_strategy = "fusion-flags" if has_moe else "kernel-opt"
```

### Step 6: Check torch.compile viability

```python
# torch.compile is often incompatible with distributed Primus/Megatron (graph breaks)
# but may work for single-GPU toy benchmarks
torch_compile_viable = (tp == 1 and pp == 1 and ep <= 1)
```

## Outputs
- `model_class`: one of `dense`, `moe_gqa`, `moe_mla`, `moe_swa`
- `primary_strategy`: highest-priority action category
- `torch_compile_viable`: bool
- Parallelism topology: `tp`, `pp`, `ep`, `dp`
- Batch config: `gbs`, `mbs`, `seq_len`, `ga_steps`
- Heuristic score priors

## Failure Handling
- If config cannot be parsed: ask user for model details
- If model YAML not found: infer from overrides in training config
