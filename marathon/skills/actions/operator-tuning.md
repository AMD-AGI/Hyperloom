# Action: Operator Tuning

Run operator-level autotuning for kernels where the framework is using generic default
configurations instead of shape-specific tuned parameters. Covers GEMM shape tuning,
fused MoE tuning, attention parameter tuning, and any other operator config system.

## Inputs
- `state.untuned_shapes` from deep-kernel-analysis (kernels using generic configs)
- `state.kernel_dispatch_map` (library and source information)
- Model configuration (hidden size, head count, expert count, etc.)

## Procedure

### Step 1: Inventory operator tuning tools

Each operator library has its own tuning infrastructure. Discover what's available:

```bash
# Search for tuning scripts in operator libraries
find "$FRAMEWORK_ROOT" -name "*tune*" -o -name "*autotune*" -o -name "*sweep*" | \
    grep -v __pycache__ | sort

# Common patterns:
# aiter:  gemm_a8w8_blockscale_tune.py, fmoe_tune.py
# triton: tuning via @triton.autotune decorator (in-kernel)
# vendor: hipblaslt tuning via env vars
```

Record available tuning tools:
```python
tuning_tools = {
    'gemm_a8w8_blockscale': {
        'library': 'aiter',
        'tool': 'gemm_a8w8_blockscale_tune.py',
        'input': 'M,N,K shapes',
        'output': 'JSON config file with tiling/block params',
    },
    'fmoe': {
        'library': 'aiter',
        'tool': 'fmoe_tune.py',
        'input': 'M,N,K + expert count',
        'output': 'Tuned dispatch config',
    },
    # ... discovered dynamically
}
```

### Step 2: Extract model-specific shapes

```python
# From the model config, derive the exact GEMM shapes
# that will be called during inference:

import json

model_config = json.load(open(f"{MODEL_PATH}/config.json"))
hidden_size = model_config.get('hidden_size', 0)
intermediate_size = model_config.get('intermediate_size', 0)
num_attention_heads = model_config.get('num_attention_heads', 0)
num_key_value_heads = model_config.get('num_key_value_heads', 0)
num_experts = model_config.get('num_local_experts', 0)
num_layers = model_config.get('num_hidden_layers', 0)

# Common GEMM shapes for transformer inference:
# QKV projection: M=batch_tokens, N=3*head_dim*num_heads, K=hidden_size
# Attention output: M=batch_tokens, N=hidden_size, K=head_dim*num_heads
# Gate projection: M=batch_tokens, N=intermediate_size, K=hidden_size
# Up projection:   M=batch_tokens, N=intermediate_size, K=hidden_size
# Down projection: M=batch_tokens, N=hidden_size, K=intermediate_size

gemm_shapes = [
    # (N, K) pairs — M varies with batch and is typically autotuned per-M
    (3 * hidden_size // num_attention_heads * num_attention_heads, hidden_size),
    (hidden_size, hidden_size),
    (intermediate_size, hidden_size),
    (hidden_size, intermediate_size),
]
if num_key_value_heads != num_attention_heads:
    # GQA: separate Q and KV projections
    head_dim = hidden_size // num_attention_heads
    gemm_shapes.append((num_key_value_heads * head_dim * 2, hidden_size))
    gemm_shapes.append((num_attention_heads * head_dim, hidden_size))

print(f"Model GEMM shapes (N, K): {gemm_shapes}")
```

### Step 3: Check current tuning coverage

```python
# For each GEMM shape, check if shape-specific tuning exists:

for N, K in gemm_shapes:
    # Check the operator library's config system
    # (implementation is library-specific)

    # Example for aiter GEMM configs:
    import glob
    config_pattern = f"*N={N}-K={K}*"
    matches = glob.glob(f"{OPERATOR_LIB_ROOT}/**/configs/**/{config_pattern}", recursive=True)

    if matches:
        print(f"  N={N}, K={K}: TUNED ({matches[0]})")
    else:
        print(f"  N={N}, K={K}: GENERIC DEFAULT — tuning opportunity")
        state['untuned_shapes'].append({
            'N': N, 'K': K,
            'library': 'aiter',
            'status': 'untuned',
        })
```

### Step 4: Run autotuning

For each untuned shape, run the library's tuning tool:

```bash
# Generic autotuning pattern:
for shape in "${UNTUNED_SHAPES[@]}"; do
    N=$(echo $shape | cut -d, -f1)
    K=$(echo $shape | cut -d, -f2)

    echo "=== Tuning N=$N K=$K ==="

    # Run the tuning tool (library-specific)
    # Common patterns:
    #   python3 $TUNE_SCRIPT --N $N --K $K --arch $GPU_ARCH --output $CONFIG_DIR/
    #   python3 $TUNE_SCRIPT --shape "$N,$K" --device 0 --warmup 10 --iters 100

    # Capture the tuned config
    # Most tools write a JSON/CSV file with optimal tiling parameters

    echo "Tuned config written to $CONFIG_DIR"
done
```

### Step 5: Validate tuned configs

```bash
# a) Run micro-benchmark with generic vs tuned config
python3 -c "
import torch, time

# Benchmark with generic config (baseline)
# ... library-specific benchmark code ...
generic_time = benchmark(N=$N, K=$K, config='generic')

# Benchmark with tuned config
# ... library-specific benchmark code ...
tuned_time = benchmark(N=$N, K=$K, config='tuned')

speedup = generic_time / tuned_time
print(f'N={$N}, K={$K}: generic={generic_time:.3f}ms, tuned={tuned_time:.3f}ms, speedup={speedup:.2f}x')
"

# b) If micro-benchmark shows improvement, deploy and E2E benchmark
if [ $(python3 -c "print(1 if $SPEEDUP > 1.05 else 0)") -eq 1 ]; then
    echo "Micro-benchmark shows ${SPEEDUP}x improvement — deploying for E2E test"
    # Copy tuned config to the operator library's config directory
    # Restart server and run E2E benchmark
fi
```

### Step 6: Deploy and measure E2E impact

```bash
# Install tuned configs
cp "$TUNED_CONFIG" "$OPERATOR_LIB_CONFIG_DIR/"

# Restart server with tuned configs
# (framework-rebuild.md if the library needs reinstall,
#  or just server restart if configs are read at runtime)

# Run E2E benchmark
bash "$SKILL_ROOT/scripts/run_benchmark.sh"
```

## Outputs
- Tuned config files for each untuned shape
- Micro-benchmark results (per-shape speedup)
- E2E benchmark result
- Updated `state.untuned_shapes` with tuning status

## Heuristic Update

- **Micro-benchmark shows >5% speedup:** Deploy and proceed to E2E. Boost operator-tuning
  score for remaining untuned shapes.
- **Micro-benchmark shows ≤5% speedup:** Generic config was already near-optimal for this
  shape. Reduce expected gain for similar shapes.
- **E2E shows gain after deploying tuned configs:** KEEP. Boost operator-tuning globally.
  Log the relationship between shape and optimal tiling for KB.
- **E2E regression despite micro-benchmark gain:** Possible interference with other
  optimizations. Revert, investigate, and log to KB as a pitfall.

## Notes on Multi-Node

At multi-node scale (>1 node), GEMM shapes may change due to:
- Different batch sizes per rank (data parallelism)
- Different expert assignments (expert parallelism)
- Pipeline stage differences (pipeline parallelism)

Check shapes per rank, not just globally. Tuning for the wrong M dimension wastes time.
