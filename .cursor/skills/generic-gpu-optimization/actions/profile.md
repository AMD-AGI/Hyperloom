# Action: Profile and Identify Hot Kernels

## Inputs
- `$BENCH_COMMAND`, `$KERNEL_LANGS`, `$PROJECT_CLASS`
- `$BUILD_DIR`

## Decision: which profiler

| project_class      | kernel_langs contain | Use |
|--------------------|----------------------|-----|
| pytorch-script     | torch-compile/triton | torch.profiler (`profile_torch.sh`) |
| triton-collection  | triton               | torch.profiler |
| hip-cmake-bench    | hip                  | rocprofv3 (`profile_rocm.sh`) |
| hpc-app            | hip                  | rocprofv3 |
| any                | mixed                | rocprofv3 (catches everything) |

## Procedure

### Step 1: Run profiler
```bash
case "$PROJECT_CLASS" in
  pytorch-script|triton-collection)
      "$SKILL_ROOT/scripts/profile_torch.sh" > "$RESULT_DIR/profile.log" 2>&1
      TRACE="$RESULT_DIR/profiles/torch_trace.json"
      ;;
  *)
      "$SKILL_ROOT/scripts/profile_rocm.sh" > "$RESULT_DIR/profile.log" 2>&1
      TRACE="$RESULT_DIR/profiles/rocprof.json"
      ;;
esac

[ -f "$TRACE" ] || { echo "ERROR: no trace produced"; exit 1; }
```

### Step 2: Top-20 kernel breakdown
```python
import json, sys
from collections import defaultdict

trace_path = sys.argv[1]
trace = json.load(open(trace_path))

# rocprofv3 vs torch.profiler schema
events = trace.get("traceEvents", trace.get("kernels", trace))

kernel_time = defaultdict(float)
kernel_count = defaultdict(int)
for e in events:
    if isinstance(e, dict):
        # torch.profiler
        if e.get("cat") == "kernel" and "dur" in e:
            kernel_time[e["name"]] += e["dur"]
            kernel_count[e["name"]] += 1
        # rocprofv3 has KernelName + DurationNs
        elif "KernelName" in e:
            kernel_time[e["KernelName"]] += e.get("DurationNs", 0) / 1000.0  # to us
            kernel_count[e["KernelName"]] += 1

total = sum(kernel_time.values())
print(f"\nTop-20 kernels ({total/1e6:.2f}s total):\n")
for name, t in sorted(kernel_time.items(), key=lambda x: -x[1])[:20]:
    print(f"  {name[:80]:80s}  {t/1000:>8.1f}ms  {t/total*100:>5.1f}%  {kernel_count[name]:>5d}x")
```

### Step 3: Extract GEAK candidates
A kernel becomes a candidate if ALL hold:
- ≥ 3% of total GPU time
- Source can be located in `$REPO_ROOT` (NOT in `/opt/rocm/`, NOT in
  `site-packages/`)
- Name does NOT match the vendor exclusion list:
  ```
  Cijk_*           hipBLASLt
  rocblas_*        rocBLAS
  ck::*            Composable Kernel
  aiter::*         AITER
  rccl_*           RCCL
  cufft_*/hipfft_* hipFFT
  ```

For each candidate, find source:
```bash
# HIP kernel
KNAME="select_k_radix_kernel"
SRC=$(rg -l "void ${KNAME}|__global__.*${KNAME}" "$REPO_ROOT" --glob '*.{hip,cu,cuh,h,cpp}' | head -1)

# Triton kernel (Python)
SRC=$(rg -l "@triton.jit" "$REPO_ROOT" --glob '*.py' | xargs rg -l "def ${KNAME}" 2>/dev/null | head -1)

# Inductor-generated Triton (torch.compile)
SRC=$(ls /tmp/torchinductor_*/*/triton/${KNAME}*.py 2>/dev/null | head -1)
```

### Step 4: Save candidates list
```bash
jq -n --argjson candidates "$CANDIDATES_JSON" \
    '{candidates: $candidates}' > "$RESULT_DIR/kernel_candidates.json"
```

Each candidate has: `{name, gpu_pct, source_path, kernel_lang, sample_shapes}`.

## Outputs
- `$RESULT_DIR/profile.log`
- `$RESULT_DIR/profiles/{torch_trace.json|rocprof.json}`
- `$RESULT_DIR/kernel_candidates.json` — fed to `kernel-opt.md`
- Top-20 breakdown printed to console for the user to read

## Failure Handling
- `rocprofv3` not installed: try `rocprof` (v2) or `omnitrace`. If none available,
  skip profile and proceed with env-vars + compile-flags only (will still produce
  gains, just blind).
- Trace file empty: re-run with longer `--benchmark_min_time` (5s).
