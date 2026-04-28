# Action: Local Test — Compilation, Correctness, Micro-Benchmark

Verifies optimized kernel code locally before generating a merge-ready patch.
This is the critical quality gate: no kernel passes to the orchestrator without
compilation, correctness, and (when possible) performance validation.

## Prerequisites

- Optimized kernel source from OOB backend or self-fix
- Original kernel source (for correctness comparison)
- Test shapes from trace or work queue entry
- GPU access (may be unavailable if inference server is running)

## Test Pipeline

```
1. COMPILATION CHECK  — can the code compile/import?
2. CORRECTNESS CHECK  — does it produce the same output as original?
3. MICRO-BENCHMARK    — is it actually faster? (multi-shape)
4. VERDICT            — pass / fail / deferred
```

---

## Step 1: Compilation Check

### Triton Kernels (`@triton.jit`, `@triton_heuristics`)

```python
import sys, importlib

def check_triton_compilation(optimized_source, filename="optimized_kernel.py"):
    """Compile Triton kernel source in an isolated namespace."""
    try:
        ns = {}
        code = compile(optimized_source, filename, "exec")
        exec(code, ns)
        
        kernel_fn = None
        for name, obj in ns.items():
            if hasattr(obj, 'run') or (callable(obj) and name.startswith(('triton_', '_'))):
                kernel_fn = (name, obj)
                break
        
        if kernel_fn is None:
            return False, "NO_KERNEL — no kernel function found in compiled output"
        
        return True, f"OK — found kernel '{kernel_fn[0]}'"
    except SyntaxError as e:
        return False, f"SYNTAX_ERROR — {e}"
    except ImportError as e:
        return False, f"IMPORT_ERROR — {e}"
    except Exception as e:
        return False, f"COMPILE_ERROR — {type(e).__name__}: {e}"
```

### HIP/C++ Kernels

```bash
# Compile with hipcc and check for errors
hipcc -O3 --amdgpu-target=gfx950 -c -o /tmp/test_kernel.o $KERNEL_SOURCE_FILE

# If the kernel is part of sgl_kernel, do a full build check:
cd /sgl-workspace/sglang/sgl-kernel
python setup_rocm.py build_ext --inplace 2>&1 | tail -50
```

```python
import subprocess

def check_hip_compilation(source_file):
    """Compile a HIP kernel with hipcc."""
    result = subprocess.run(
        ["/opt/rocm/bin/hipcc", "-O3", "--amdgpu-target=gfx950",
         "-c", "-o", "/tmp/test_kernel.o", source_file],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        return True, "OK — hipcc compilation successful"
    return False, f"HIPCC_ERROR — {result.stderr[:500]}"
```

### Python Dispatch Changes

```python
def check_dispatch_fix(modified_file_path):
    """Verify a Python dispatch change imports and runs correctly."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", modified_file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return True, f"OK — module loaded from {modified_file_path}"
    except Exception as e:
        return False, f"IMPORT_ERROR — {type(e).__name__}: {e}"
```

### sgl_kernel Rebuild

When the optimized kernel is a C++/HIP change to `sgl_kernel`:

```python
def rebuild_sgl_kernel():
    """Rebuild sgl_kernel from source. Returns (success, message)."""
    result = subprocess.run(
        ["python", "setup_rocm.py", "install"],
        capture_output=True, text=True, timeout=600,
        cwd="/sgl-workspace/sglang/sgl-kernel",
    )
    if result.returncode == 0:
        return True, "OK — sgl_kernel rebuilt successfully"
    return False, f"BUILD_FAIL — {result.stderr[:1000]}"
```

### aiter JIT Kernel

For aiter JIT kernels, clear the cache and trigger recompilation:

```python
import shutil

def clear_aiter_jit_cache(kernel_name):
    """Clear JIT cache for a specific kernel, forcing recompilation."""
    cache_dir = f"/sgl-workspace/aiter/aiter/jit/build/{kernel_name}/"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        return True, f"OK — cleared JIT cache at {cache_dir}"
    return True, f"OK — no JIT cache found at {cache_dir} (will compile on first use)"
```

---

## Step 2: Correctness Check

**Requires GPU access.** If the inference server is running on **all** GPUs, skip
this step and mark as `correctness: "deferred"`. When `TP < GPU_COUNT`, use a free
GPU via `get_test_device()` — do NOT default to device 0 (the server's GPU).

### GPU Availability Check

The inference server uses GPUs `0..TP-1`. When `TP < GPU_COUNT`, the remaining
GPUs are free for micro-benchmarks. Always check **all** devices — not just
device 0 — and use the GPU lock file if present.

```python
import torch, os, json

GPU_LOCK_PATH = "/tmp/.marathon_gpu_lock.json"

def _read_gpu_lock():
    """Read the GPU lock file to find which GPUs the orchestrator is using."""
    try:
        with open(GPU_LOCK_PATH) as f:
            lock = json.load(f)
        if lock.get("holder") and lock.get("gpus"):
            return set(lock["gpus"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None

def find_free_gpu():
    """Find a GPU not used by the inference server.

    Returns (device_id, reason) — device_id is an int if a free GPU exists,
    or None if all GPUs are busy.

    Resolution order:
      1. Read /tmp/.marathon_gpu_lock.json (authoritative, written by orch)
      2. Fall back to TP env var (server uses devices 0..TP-1)
      3. Fall back to per-device memory probe
    """
    total_gpus = torch.cuda.device_count()
    if total_gpus == 0:
        return None, "NO_GPU — torch sees 0 devices"

    locked = _read_gpu_lock()
    if locked is not None:
        for dev in range(total_gpus):
            if dev in locked:
                continue
            mem_free, mem_total = torch.cuda.mem_get_info(dev)
            if (mem_total - mem_free) / mem_total < 0.20:
                return dev, f"OK — GPU {dev} free (lock says {sorted(locked)} busy)"
        return None, f"ALL_LOCKED — lock={sorted(locked)}, all others >20% used"

    server_tp = int(os.environ.get("TP", str(total_gpus)))
    server_gpus = set(range(min(server_tp, total_gpus)))

    for dev in range(total_gpus):
        if dev in server_gpus:
            continue
        mem_free, mem_total = torch.cuda.mem_get_info(dev)
        if (mem_total - mem_free) / mem_total < 0.20:
            return dev, f"OK — GPU {dev} free (TP={server_tp}, devices {sorted(server_gpus)} assumed busy)"

    for dev in range(total_gpus):
        mem_free, mem_total = torch.cuda.mem_get_info(dev)
        if (mem_total - mem_free) / mem_total < 0.20:
            return dev, f"OK — GPU {dev} free (all-scan, server may be down)"

    return None, f"GPU_BUSY — all {total_gpus} devices >20% VRAM used"

def gpu_available():
    """Legacy compat wrapper. Returns True if any GPU is free."""
    dev, _ = find_free_gpu()
    return dev is not None

def get_test_device(kernel_name=None):
    """Return the device string for torch tensors during testing.

    Sets HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES so torch operations
    target the free GPU, not the server's GPU.

    Resolution order:
      1. Try find_free_gpu() for an immediately available GPU
      2. If none free, request temporary access from orchestrator (IR-25)
      3. If request times out after 60min total (2 attempts), return None
    """
    dev, reason = find_free_gpu()
    if dev is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(dev)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
        return "cuda:0", reason

    # No free GPU — request temporary access from orchestrator
    dev, reason = request_gpu_access(kernel_name or "unknown")
    if dev is not None:
        return dev, reason  # request_gpu_access already set env vars
    return None, reason

import time, datetime

GPU_REQUEST_PATH = os.path.join(os.environ.get("SESSION_DIR", ""),
                                "kernel_manager/gpu_request.json")

def request_gpu_access(kernel_name, estimated_duration_s=300):
    """Request exclusive GPU access from the orchestrator (IR-25).

    Writes a pending request, polls for the orchestrator to grant it
    (by killing the inference server), then returns a device.

    Returns (device_str, reason) — device_str is "cuda:0" if granted,
    or None if the request timed out.
    """
    if not GPU_REQUEST_PATH or GPU_REQUEST_PATH.startswith("/kernel_manager"):
        return None, "GPU_REQUEST_SKIP — SESSION_DIR not set"

    req = {"status": "pending", "requester": "kernel-manager",
           "kernel": kernel_name,
           "since": datetime.datetime.utcnow().isoformat() + "Z",
           "estimated_duration_s": estimated_duration_s}
    os.makedirs(os.path.dirname(GPU_REQUEST_PATH), exist_ok=True)
    with open(GPU_REQUEST_PATH, "w") as f:
        json.dump(req, f)

    # Try twice: 30min initial + 30min retry = 60min max.
    # The micro-benchmark MUST run — do not give up easily.
    for attempt in range(2):
        if attempt > 0:
            req["since"] = datetime.datetime.utcnow().isoformat() + "Z"
            req["attempt"] = attempt + 1
            with open(GPU_REQUEST_PATH, "w") as f:
                json.dump(req, f)
        for _ in range(60):  # 60 * 30s = 30min per attempt
            time.sleep(30)
            try:
                with open(GPU_REQUEST_PATH) as f:
                    r = json.load(f)
                if r.get("status") == "granted":
                    _write_gpu_lock_km([0], os.getpid())
                    os.environ["HIP_VISIBLE_DEVICES"] = "0"
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    return "cuda:0", "GPU_GRANTED by orchestrator"
            except (FileNotFoundError, json.JSONDecodeError):
                pass
    # 60min total — last resort
    try: os.remove(GPU_REQUEST_PATH)
    except FileNotFoundError: pass
    return None, "GPU_REQUEST_TIMEOUT — orchestrator did not grant in 60min"

def _write_gpu_lock_km(gpus, pid):
    """Write GPU lock as kernel-manager (temporary holder during benchmarks)."""
    lock = {"holder": "kernel-manager", "gpus": list(gpus), "pid": pid,
            "since": datetime.datetime.utcnow().isoformat() + "Z",
            "purpose": "micro-benchmark"}
    with open(GPU_LOCK_PATH, "w") as f:
        json.dump(lock, f)

def _release_gpu_lock_km():
    """Release GPU lock held by kernel-manager."""
    try: os.remove(GPU_LOCK_PATH)
    except FileNotFoundError: pass

def release_gpu_after_benchmark():
    """Release GPU lock and signal orchestrator to restart its server.

    Call this after ALL micro-benchmarks are done for the current kernel.
    The orchestrator polls gpu_request.json and will restart the inference
    server when it sees status: 'released'.
    """
    _release_gpu_lock_km()
    try:
        with open(GPU_REQUEST_PATH) as f:
            r = json.load(f)
        r["status"] = "released"
        r["released_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        with open(GPU_REQUEST_PATH, "w") as f:
            json.dump(r, f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
```

### Correctness Verification

```python
def verify_correctness(original_source, optimized_source, test_shapes, dtype=torch.bfloat16):
    """Compare outputs of original vs optimized kernel.

    Args:
        original_source: str, the original kernel source code
        optimized_source: str, the optimized kernel source code  
        test_shapes: dict with keys like "xnumel", "r0_numel", "M", "N", "K"
        dtype: torch dtype for test tensors

    Returns: (passed: bool, details: str)
    """
    ATOL = 1e-2
    RTOL = 1e-2

    orig_ns, opt_ns = {}, {}
    exec(compile(original_source, "original.py", "exec"), orig_ns)
    exec(compile(optimized_source, "optimized.py", "exec"), opt_ns)

    kernel_name = None
    for name in opt_ns:
        if hasattr(opt_ns[name], 'run') or (callable(opt_ns.get(name)) and name.startswith(('triton_', '_'))):
            kernel_name = name
            break

    if kernel_name not in orig_ns:
        return False, f"NAME_MISMATCH — '{kernel_name}' not found in original"
    
    original_fn = orig_ns[kernel_name]
    optimized_fn = opt_ns[kernel_name]
    
    xnumel = test_shapes.get("xnumel", 1)
    r0_numel = test_shapes.get("r0_numel", None)
    shape = (xnumel, r0_numel) if r0_numel else (xnumel,)
    
    import re
    func_match = re.search(r"def \w+\(([^)]+)\)", original_source)
    if not func_match:
        return False, "PARSE_ERROR — cannot extract function signature"
    
    params = [p.strip().split(':')[0].strip() for p in func_match.group(1).split(',')]
    in_ptrs = [p for p in params if p.startswith('in_ptr')]
    out_ptrs = [p for p in params if p.startswith('out_ptr')]
    
    in_tensors = [torch.randn(shape, device="cuda", dtype=dtype) for _ in in_ptrs]
    out_orig = [torch.empty(shape, device="cuda", dtype=dtype) for _ in out_ptrs]
    out_opt = [torch.empty(shape, device="cuda", dtype=dtype) for _ in out_ptrs]
    
    constexprs = {k: v for k, v in test_shapes.items() if k not in ("dtype",)}
    args_orig = in_tensors + out_orig + list(constexprs.values())
    args_opt = in_tensors + out_opt + list(constexprs.values())
    
    try:
        original_fn(*args_orig)
        optimized_fn(*args_opt)
        torch.cuda.synchronize()
    except Exception as e:
        return False, f"RUNTIME_ERROR — {type(e).__name__}: {e}"
    
    for i, (o, g) in enumerate(zip(out_opt, out_orig)):
        if not torch.allclose(o, g, atol=ATOL, rtol=RTOL):
            max_diff = (o - g).abs().max().item()
            return False, f"CORRECTNESS_FAIL — out_ptr{i} max diff={max_diff:.6f} (atol={ATOL}, rtol={RTOL})"
    
    return True, "OK — outputs match within tolerance"
```

---

## Step 3: Micro-Benchmark

**Requires GPU access.** The micro-benchmark MUST run.
When `TP < GPU_COUNT`, use a free GPU via `get_test_device()`.
When `TP == GPU_COUNT`, request temporary GPU access from the orchestrator (IR-25),
which will wait up to 60min total (2 × 30min attempts). Only defer as a last resort.

### Multi-Shape Benchmark

Test at multiple batch sizes to catch occupancy-dependent regressions (the DeepSeek-R1
lesson: +44% micro at one shape, -19.9% E2E due to register pressure at scale).

```python
def micro_benchmark(original_source, optimized_source, test_shapes, dtype=torch.bfloat16):
    """Multi-shape micro-benchmark. Returns (avg_speedup, per_shape_results) or (None, reason).
    
    CRITICAL: test_shapes MUST come from the trace or work queue entry. Never fabricate.
    """
    if not gpu_available():
        return None, "GPU_BUSY — defer to integration phase"
    
    orig_ns, opt_ns = {}, {}
    try:
        exec(compile(original_source, "original.py", "exec"), orig_ns)
        exec(compile(optimized_source, "optimized.py", "exec"), opt_ns)
    except Exception as e:
        return None, f"COMPILE_FAIL — {e}"
    
    kernel_name = None
    for name in opt_ns:
        if hasattr(opt_ns[name], 'run') or (callable(opt_ns.get(name)) and name.startswith(('triton_', '_'))):
            kernel_name = name
            break
    
    if not kernel_name or kernel_name not in orig_ns:
        return None, f"NAME_MISMATCH — kernel not found in both files"
    
    original_fn = orig_ns[kernel_name]
    optimized_fn = opt_ns[kernel_name]
    
    base_xnumel = test_shapes.get("xnumel", 1)
    r0_numel = test_shapes.get("r0_numel", None)
    results = []
    
    for mult in [1, 4, 16, 64]:
        xnumel = base_xnumel * mult
        shape = (xnumel, r0_numel) if r0_numel else (xnumel,)
        
        import re
        func_match = re.search(r"def \w+\(([^)]+)\)", original_source)
        params = [p.strip().split(':')[0].strip() for p in func_match.group(1).split(',')]
        in_ptrs = [p for p in params if p.startswith('in_ptr')]
        out_ptrs = [p for p in params if p.startswith('out_ptr')]
        
        in_tensors = [torch.randn(shape, device="cuda", dtype=dtype) for _ in in_ptrs]
        out_orig = [torch.empty(shape, device="cuda", dtype=dtype) for _ in out_ptrs]
        out_opt = [torch.empty(shape, device="cuda", dtype=dtype) for _ in out_ptrs]
        
        constexprs = {k: v for k, v in test_shapes.items() if k not in ("dtype",)}
        constexprs["xnumel"] = xnumel
        args_orig = in_tensors + out_orig + list(constexprs.values())
        args_opt = in_tensors + out_opt + list(constexprs.values())
        
        try:
            for _ in range(20):
                original_fn(*args_orig)
                optimized_fn(*args_opt)
            torch.cuda.synchronize()
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(200):
                original_fn(*args_orig)
            end.record()
            torch.cuda.synchronize()
            orig_ms = start.elapsed_time(end) / 200
            
            start.record()
            for _ in range(200):
                optimized_fn(*args_opt)
            end.record()
            torch.cuda.synchronize()
            opt_ms = start.elapsed_time(end) / 200
            
            speedup = orig_ms / opt_ms if opt_ms > 0 else 0
            results.append({
                "xnumel": xnumel, "mult": mult,
                "speedup": speedup,
                "orig_ms": round(orig_ms, 4),
                "opt_ms": round(opt_ms, 4),
            })
        except Exception as e:
            results.append({"xnumel": xnumel, "mult": mult, "speedup": 0, "error": str(e)})
    
    if any(r.get("speedup", 0) < 0.95 for r in results):
        return None, results  # regression at some shape
    
    avg_speedup = sum(r.get("speedup", 0) for r in results) / len(results)
    return round(avg_speedup, 3), results
```

---

## Step 4: Verdict

| Compilation | Correctness | Micro-Benchmark | Verdict | Next Step |
|:-----------:|:-----------:|:---------------:|---------|-----------|
| PASS | PASS | > 1.05x avg, no shape regresses | **PASS** | Generate merge-ready patch |
| PASS | PASS | Deferred (GPU busy) | **PASS-DEFERRED** | Generate patch with `micro_benchmark: "deferred"` |
| PASS | Deferred | Deferred | **PASS-DEFERRED** | Generate patch, orchestrator verifies during integration |
| PASS | FAIL | — | **FAIL** | Feed correctness error to OOB as feedback |
| PASS | PASS | < 1.05x avg or regression at any shape | **FAIL-PERF** | Feed performance data to OOB as feedback |
| FAIL | — | — | **FAIL-COMPILE** | Feed compilation error to OOB as feedback |

### Building Feedback for Iterative Refinement

When a test fails, build a feedback string for the next OOB round:

```python
def build_feedback(compilation_result, correctness_result, benchmark_result):
    """Build feedback string for OOB resubmission."""
    feedback_parts = []
    
    comp_ok, comp_msg = compilation_result
    if not comp_ok:
        feedback_parts.append(f"COMPILATION FAILED: {comp_msg}")
        return "\n".join(feedback_parts)  # no point checking further
    
    if correctness_result:
        corr_ok, corr_msg = correctness_result
        if not corr_ok:
            feedback_parts.append(f"CORRECTNESS FAILED: {corr_msg}")
            return "\n".join(feedback_parts)
    
    if benchmark_result:
        speedup, details = benchmark_result
        if speedup is None and isinstance(details, list):
            regressed = [r for r in details if r.get("speedup", 0) < 0.95]
            feedback_parts.append(
                f"PERFORMANCE REGRESSION at shapes: "
                + ", ".join(f"xnumel={r['xnumel']} ({r['speedup']:.2f}x)" for r in regressed)
            )
        elif speedup is not None and speedup < 1.05:
            feedback_parts.append(
                f"INSUFFICIENT SPEEDUP: {speedup:.2f}x (need >1.05x). "
                f"Try a different optimization approach."
            )
    
    if not feedback_parts:
        return None  # no issues
    
    return "\n".join(feedback_parts) + "\nFix the issue and try a different approach."
```

---

## Build System Integration

After a kernel passes all tests, prepare the build environment before generating the
patch. The patch itself is generated by `actions/patch-gen.md`, but the manager must
know which build steps are needed.

### Determining Rebuild Requirements

| Kernel Location | Patch Type | Rebuild Required | Command |
|----------------|------------|:----------------:|---------|
| `sglang/python/**/*.py` | Python dispatch | No | Server restart only |
| `sgl_kernel/__init__.py`, `*.py` wrappers | Python wrapper | No | Server restart only |
| `sgl-kernel/csrc/**/*.cu` | C++/HIP | **Yes** | `cd /sgl-workspace/sglang/sgl-kernel && python setup_rocm.py install` |
| `aiter/aiter/**/*.py` | Python dispatch | No | Server restart only |
| `aiter/aiter/jit/**/*.cpp` or `.hip` | JIT source | **Clear cache** | `rm -rf /sgl-workspace/aiter/aiter/jit/build/<kernel>/` |
| `/tmp/torchinductor_*/**/*.py` | Inductor Triton | No | Clear `~/.triton/cache` + restart |
| `sglang/srt/layers/**/*.py` (Triton `@triton.jit`) | Triton source | No | Clear `~/.triton/cache` + restart |

### Cache Clearing Checklist

Before declaring a test complete, ensure the right caches are cleared so the
optimized kernel actually loads on the next server start:

```python
import shutil, subprocess

def clear_caches_for_patch(patch_type, kernel_name=None):
    """Clear relevant caches based on patch type."""
    cleared = []
    
    if patch_type in ("triton-source", "inductor-triton"):
        triton_cache = os.path.expanduser("~/.triton/cache")
        if os.path.exists(triton_cache):
            shutil.rmtree(triton_cache)
            cleared.append(f"Triton cache: {triton_cache}")
    
    if patch_type == "jit-source" and kernel_name:
        jit_dir = f"/sgl-workspace/aiter/aiter/jit/build/{kernel_name}/"
        if os.path.exists(jit_dir):
            shutil.rmtree(jit_dir)
            cleared.append(f"aiter JIT cache: {jit_dir}")
    
    if patch_type in ("inductor-triton",):
        inductor_cache = "/tmp/torchinductor_root"
        if os.path.exists(inductor_cache):
            cleared.append(f"Note: Inductor cache at {inductor_cache} should NOT be cleared "
                         "(it contains the standalone files we're patching)")
    
    if patch_type in ("python-dispatch", "python-wrapper"):
        subprocess.run(
            ["find", "/sgl-workspace", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
            capture_output=True, timeout=30,
        )
        cleared.append("Python __pycache__ directories")
    
    return cleared
```

---

## Full Test Orchestration

Putting it all together — the complete test pipeline for a single kernel result:

```python
def run_full_test(original_source, optimized_source, test_shapes, kernel_type, kernel_name=None):
    """Run the complete test pipeline. Returns a test report dict.

    GPU access strategy (IR-25):
      1. Try find_free_gpu() for an immediately available GPU
      2. If none free, request_gpu_access() asks the orchestrator to
         temporarily kill the server and grant GPU time
      3. Waits up to 60min (2 × 30min attempts) — micro-benchmark must run
      4. If GPUs were borrowed, release_gpu_after_benchmark() signals
         the orchestrator to restart its server
      5. Only defer as absolute last resort (60min timeout with no grant)
    """
    report = {
        "compilation": None,
        "correctness": None,
        "micro_benchmark": None,
        "verdict": None,
        "feedback": None,
        "caches_cleared": [],
        "gpu_borrowed": False,
    }
    
    # Step 1: Compilation (no GPU needed)
    if kernel_type in ("triton", "inductor-triton", "triton-source"):
        comp_ok, comp_msg = check_triton_compilation(optimized_source)
    elif kernel_type in ("cpp-hip", "hip-kernel"):
        comp_ok, comp_msg = check_hip_compilation(optimized_source)
    else:
        comp_ok, comp_msg = True, "OK — Python source (no compilation needed)"
    
    report["compilation"] = (comp_ok, comp_msg)
    if not comp_ok:
        report["verdict"] = "FAIL-COMPILE"
        report["feedback"] = build_feedback(report["compilation"], None, None)
        return report
    
    # Acquire GPU for steps 2-3 (uses IR-25 request if no free GPU)
    device, gpu_reason = get_test_device(kernel_name)
    gpu_ok = device is not None
    borrowed = "GPU_GRANTED" in gpu_reason if gpu_ok else False
    report["gpu_borrowed"] = borrowed

    # Step 2: Correctness (requires GPU)
    if gpu_ok:
        corr_ok, corr_msg = verify_correctness(original_source, optimized_source, test_shapes)
        report["correctness"] = (corr_ok, corr_msg)
        if not corr_ok:
            report["verdict"] = "FAIL-CORRECTNESS"
            report["feedback"] = build_feedback(report["compilation"], report["correctness"], None)
            if borrowed:
                release_gpu_after_benchmark()
            return report
    else:
        report["correctness"] = (None, f"DEFERRED — {gpu_reason}")
    
    # Step 3: Micro-benchmark (requires GPU)
    if gpu_ok:
        speedup, details = micro_benchmark(original_source, optimized_source, test_shapes)
        report["micro_benchmark"] = (speedup, details)
        if speedup is None:
            if isinstance(details, str) and "GPU_BUSY" in details:
                report["verdict"] = "PASS-DEFERRED"
            else:
                report["verdict"] = "FAIL-PERF"
                report["feedback"] = build_feedback(
                    report["compilation"], report["correctness"], report["micro_benchmark"]
                )
                if borrowed:
                    release_gpu_after_benchmark()
                return report
        elif speedup < 1.05:
            report["verdict"] = "FAIL-PERF"
            report["feedback"] = build_feedback(
                report["compilation"], report["correctness"], report["micro_benchmark"]
            )
            if borrowed:
                release_gpu_after_benchmark()
            return report
        else:
            report["verdict"] = "PASS"
    else:
        report["micro_benchmark"] = (None, f"DEFERRED — {gpu_reason}")
        report["verdict"] = "PASS-DEFERRED"
    
    # Release borrowed GPU back to orchestrator (triggers server restart)
    if borrowed:
        release_gpu_after_benchmark()

    # Step 4: Clear caches for the patch type
    patch_type_map = {
        "triton": "triton-source",
        "inductor-triton": "inductor-triton",
        "cpp-hip": "cpp-rebuild",
        "hip-kernel": "cpp-rebuild",
        "python-dispatch": "python-dispatch",
    }
    pt = patch_type_map.get(kernel_type, "python-dispatch")
    report["caches_cleared"] = clear_caches_for_patch(pt, kernel_name)
    
    return report
```
