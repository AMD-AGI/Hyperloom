# Action: Try ROCm Runtime Environment Variables

## Why this is the cheapest action
Env vars don't require a rebuild — just re-run the benchmark with the variable
set. Often produces single-digit-percent gains for free, especially on AMD.

## Catalog (try in this order, scored highest first)

| Env var | Default | Try | Why |
|---|---|---|---|
| `HSA_ENABLE_SDMA` | 1 | `0` | Disables SDMA, often faster for small transfers on MI3xx |
| `GPU_MAX_HW_QUEUES` | 4 | `8`, `16` | More concurrent kernel streams |
| `HIP_FORCE_DEV_KERNARG` | 0 | `1` | Allocates kernel args in device memory; lower launch overhead |
| `HSA_FORCE_FINE_GRAIN_PCIE` | 0 | `1` | Fine-grain PCIe coherence; helps small-message workloads |
| `HSA_ENABLE_PEER_SDMA` | 1 | `0` | Disable peer SDMA on single-GPU runs |
| `AMD_DIRECT_DISPATCH` | 1 | `1` | Confirm enabled (huge win on launch-bound kernels) |
| `HIPBLASLT_TUNING_FILE` | unset | `$RESULT_DIR/blaslt.tune` | Persists hipBLASLt autotune across runs |
| `ROCBLAS_LAYER` | 0 | `0` | Make sure no logging overhead is on |
| `MIOPEN_FIND_MODE` | 5 | `1` | Faster MIOpen heuristic mode |
| `OMP_NUM_THREADS` | nproc | `1`, `nproc/2` | If launch-bound on host |

For PyTorch projects also try:
| `PYTORCH_TUNABLEOP_ENABLED` | 0 | `1` | GEMM autotune (skip on first run, slow) |
| `TORCH_BLAS_PREFER_HIPBLASLT` | 1 | `1` | Confirm hipBLASLt is preferred |
| `TORCHINDUCTOR_CACHE_DIR` | `/tmp/...` | `$RESULT_DIR/inductor` | Reproducible compile cache |

## Procedure

### Step 1: Pick one env var to test
```bash
ENV_NAME="$1"      # e.g. HSA_ENABLE_SDMA
ENV_VAL="$2"       # e.g. 0
ATTEMPT_DESCRIPTION="env: $ENV_NAME=$ENV_VAL"
```

### Step 2: Re-run benchmark with the var set (no rebuild)
```bash
export "$ENV_NAME=$ENV_VAL"
ATTEMPT_ID=$((ATTEMPT_ID + 1))
# Trigger baseline.md (which will source kept_env.sh and pick up the new export)
```

### Step 3: Apply correctness gate
Trigger `correctness.md`. Env vars normally don't affect correctness, but they
sometimes do (e.g. fine-grain PCIe) — never skip the gate.

### Step 4: Keep or revert
If KEEP, append to `$RESULT_DIR/kept_env.sh`:
```bash
echo "export $ENV_NAME=$ENV_VAL" >> "$RESULT_DIR/kept_env.sh"
```
If REVERT, just `unset $ENV_NAME`.

## Combination Rule
After 2+ wins, push a `combined-env-test` action with all winners simultaneously.
Sometimes individual wins compose, sometimes they fight; only direct measurement
tells you.

## Outputs
- `$RESULT_DIR/kept_env.sh` updated
- Entry in `results.tsv`
