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

**Requires GPU access.** If the inference server is running on the GPU, skip this step
and mark as `correctness: "deferred"`.

### GPU Availability Check

```python
import torch

def gpu_available():
    """Check if the GPU is free for testing."""
    try:
        mem_free, mem_total = torch.cuda.mem_get_info()
        usage_pct = (mem_total - mem_free) / mem_total * 100
        if usage_pct > 20:
            return False  # server likely running
        return True
    except RuntimeError:
        return False
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

**Requires GPU access.** If GPU is busy, mark as `micro_benchmark: "deferred"`.

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
    """Run the complete test pipeline. Returns a test report dict."""
    report = {
        "compilation": None,
        "correctness": None,
        "micro_benchmark": None,
        "verdict": None,
        "feedback": None,
        "caches_cleared": [],
    }
    
    # Step 1: Compilation
    if kernel_type in ("triton", "inductor-triton", "triton-source"):
        comp_ok, comp_msg = check_triton_compilation(optimized_source)
    elif kernel_type in ("cpp-hip", "hip-kernel"):
        comp_ok, comp_msg = check_hip_compilation(optimized_source)  # source is a file path
    else:
        comp_ok, comp_msg = True, "OK — Python source (no compilation needed)"
    
    report["compilation"] = (comp_ok, comp_msg)
    if not comp_ok:
        report["verdict"] = "FAIL-COMPILE"
        report["feedback"] = build_feedback(report["compilation"], None, None)
        return report
    
    # Step 2: Correctness (if GPU available)
    if gpu_available():
        corr_ok, corr_msg = verify_correctness(original_source, optimized_source, test_shapes)
        report["correctness"] = (corr_ok, corr_msg)
        if not corr_ok:
            report["verdict"] = "FAIL-CORRECTNESS"
            report["feedback"] = build_feedback(report["compilation"], report["correctness"], None)
            return report
    else:
        report["correctness"] = (None, "DEFERRED — GPU busy")
    
    # Step 3: Micro-benchmark (if GPU available)
    if gpu_available():
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
                return report
        elif speedup < 1.05:
            report["verdict"] = "FAIL-PERF"
            report["feedback"] = build_feedback(
                report["compilation"], report["correctness"], report["micro_benchmark"]
            )
            return report
        else:
            report["verdict"] = "PASS"
    else:
        report["micro_benchmark"] = (None, "DEFERRED — GPU busy")
        report["verdict"] = "PASS-DEFERRED"
    
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
