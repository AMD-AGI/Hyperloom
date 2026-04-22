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

The exact JSON schema differs between profilers. Use the helper below — it
recognizes torch.profiler Chrome traces, rocprofv3 (`rocprofiler-sdk-tool`),
rocprof-v2 CSV-derived JSON (`{"kernels": [...]}`), and falls back to a
generic `traceEvents` walker.

```python
import json, sys
from collections import defaultdict

def kernel_breakdown(trace_path):
    data = json.load(open(trace_path))
    times, counts = defaultdict(float), defaultdict(int)

    # ---- rocprofv3 schema ------------------------------------------------
    if isinstance(data, dict) and "rocprofiler-sdk-tool" in data:
        top = data["rocprofiler-sdk-tool"][0]
        sym = {s["kernel_id"]: s["kernel_name"]
               for s in top.get("kernel_symbols", [])
               if s.get("kernel_name")}
        for d in top.get("buffer_records", {}).get("kernel_dispatch", []):
            kid = d.get("kernel_id") or d.get("dispatch_info", {}).get("kernel_id")
            name = sym.get(kid, f"<id_{kid}>")
            dur_ns = d["end_timestamp"] - d["start_timestamp"]
            times[name] += dur_ns / 1000.0   # ns -> us
            counts[name] += 1
        return times, counts

    # ---- rocprof v2 fallback (our wrapper writes {kernels: [{KernelName,DurationNs}]})
    if isinstance(data, dict) and "kernels" in data:
        for e in data["kernels"]:
            times[e["KernelName"]] += e.get("DurationNs", 0) / 1000.0
            counts[e["KernelName"]] += 1
        return times, counts

    # ---- torch.profiler Chrome trace -------------------------------------
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    for e in events:
        if isinstance(e, dict) and e.get("cat") == "kernel" and "dur" in e:
            times[e["name"]] += e["dur"]   # already us
            counts[e["name"]] += 1
    return times, counts


trace_path = sys.argv[1]
times, counts = kernel_breakdown(trace_path)
total = sum(times.values())
print(f"\nTop-20 kernels ({total/1e3:.2f} ms total GPU time):\n")
for name, t in sorted(times.items(), key=lambda x: -x[1])[:20]:
    short = name if len(name) <= 80 else name[:77] + "..."
    print(f"  {short:80s}  {t:>9.1f}us  {t/total*100:>5.1f}%  {counts[name]:>5d}x")
```

The helper is also available as `$SKILL_ROOT/scripts/analyze_profile.py` so the
agent doesn't have to inline this every run.

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
