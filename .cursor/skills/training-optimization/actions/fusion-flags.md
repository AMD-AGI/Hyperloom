# Action: Fusion Flag Exploration

## Overview

Fusion flags replace generic PyTorch kernels with fused, hardware-optimized alternatives.
This is the **highest-impact action for MoE models** — each flag is a single config override,
zero code changes, zero crash risk.

## Inputs
- Baseline ms/iter and kept_overrides from prior attempts
- Model class from classify step
- Profile data (which kernels dominate)

## Fusion Flag Matrix

Test each flag individually, then combine winners.

### Tier 1: High-confidence flags (try first)

| Flag | Expected Impact | Model Class | Notes |
|------|----------------|-------------|-------|
| `moe_permute_fusion=true` | +1–2% | MoE only | Fused Triton permute for MoE token dispatch. Replaces `CatArrayBatchedCopy`, `scatter_gather`, `indexFuncLargeIndex` |
| `gradient_accumulation_fusion=true` | +0.3–0.5% | All | Fuses wgrad GEMM with optimizer accumulation. No downside |
| `moe_use_fused_router_with_aux_score=true` | ~0% | MoE only | Fused TopK router. Reduces kernel count, negligible speedup |

### Tier 2: Conditional flags (model/config dependent)

| Flag | Expected Impact | Model Class | Notes |
|------|----------------|-------------|-------|
| `use_turbo_grouped_mlp=true` | -1% to +1% | MoE | Fused SwiGLU. **Regresses with wide FFN dims** (e.g., ffn_hidden_size > 8192) |
| `use_turbo_attention=true` | +0–1% | All | PrimusTurbo attention. Usually already enabled in MI355X configs |
| `apply_rope_fusion=true` | +0–0.5% | All | Fused RoPE embedding. Check if already enabled |
| `moe_shared_expert_overlap=true` | +0–1% | MoE with shared experts | Overlap shared expert compute with routing |

### Tier 3: Attention backend toggles

| Flag | Expected Impact | Model Class | Notes |
|------|----------------|-------------|-------|
| `use_sink_attention=true` | Variable | All | Toggles between Triton and aiter backends. **Must measure** |
| `use_sink_attention=false` | Variable | All | aiter attention. Check `deterministic` flag |

### Flags to AVOID

| Flag | Why |
|------|-----|
| `sink_sliding_window=N` | Triton backend does NOT support sliding window. Crashes. |
| `use_turbo_grouped_mlp=true` with wide FFN | Suboptimal tile config for non-square shapes |

## Procedure

### Step 1: Test each Tier 1 flag individually

For each flag:
```bash
pkill -9 -f "primus/cli/main.py" 2>/dev/null; sleep 3
MASTER_PORT=$((MASTER_PORT + 1))

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m primus.cli.main train pretrain \
  --config "$CONFIG_YAML" \
  $KEPT_OVERRIDES \
  <new_flag>=<value> \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/attempt_fusion_N.log
```

### Step 2: Measure and decide

Extract ms/iter from iterations 6–10. Verify GBS matches baseline.

- **Improved:** KEEP — add to `KEPT_OVERRIDES`
- **Same or worse:** DISCARD — do not add
- **Crashed:** Log crash, skip flag

### Step 3: Combine winners

After testing all Tier 1 flags individually, test all KEPT flags together:
```bash
torchrun ... $ALL_KEPT_FLAGS profile=false use_pytorch_profiler=false
```

If the combined result is better than individual best, the combination is the new baseline.
If worse than expected (sub-additive), test pairwise combinations to find conflicts.

### Step 4: Conditional Tier 2 testing

Based on model class and profile:
- If MoE with narrow FFN: test `use_turbo_grouped_mlp=true`
- If attention > 10% of GPU time: test `use_sink_attention` toggle
- If RoPE not fused: test `apply_rope_fusion=true`

## Outputs
- `winning_flags`: list of flags that improved ms/iter
- Combined gain percentage
- Updated `KEPT_OVERRIDES` string
- KB entries for each tested flag

## Failure Handling
- If a flag crashes: log the error, mark as `crash`, skip
- If a flag changes GBS: mark as `invalid`, revert
- If all flags fail: parallelism or kernel-opt may be more productive
