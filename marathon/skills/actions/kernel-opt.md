# Action: Deep Kernel Optimization (Marathon)

**DFS role:** Scores 6-8 for Marathon depending on model class. Marathon kernel-opt
builds on Sprint's basic fire-and-forget approach with deeper analysis:
- **Dispatch tracing prerequisite** — deep-kernel-analysis.md MUST run first
- **Strategy B' (register-constrained)** — OOB submissions with explicit VGPR/occupancy limits
- **Strategy G (GEMM shape tuning)** — deferred to operator-tuning.md
- **Kernel rewrite + rebuild cycle** — modify source, rebuild library, verify dispatch, E2E test

Multi-round kernel optimization loop using configurable backends.

Backend references:
- [`../kernel-opt/geak.md`](../kernel-opt/geak.md) — GEAK MCP (remote GPU pod)
- [`../kernel-opt/codex.md`](../kernel-opt/codex.md) — Codex via OOB Agent MCP
- [`../kernel-opt/claude.md`](../kernel-opt/claude.md) — Claude Code via OOB Agent MCP
- [`../kernel-opt/llm.md`](../kernel-opt/llm.md) — LLM Proxy (direct API)

## Inputs
- Kernel candidates from `profile.md` (kernel_name, gpu_pct, source_location)
- Current best config (backends + params)
- `baseline_tput_per_gpu` (after backends + params)
- `compile_probe_results` from `classify.md` (if torch.compile failed — lists which submodule types compile)

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME kernel optimization" --top-k 5 --compact
python3 $SKILL_ROOT/kb/kb_query.py --category kernel_optimization --compact
```

## Procedure

**FLOW GUARD:** Do NOT skip this action if candidates exist. Running sweep with unoptimized kernels wastes compute.

### Step 1: Locate kernel source

**Strategy A (torch.compile mode):** Extract from STANDALONE kernel files in Inductor cache.
```bash
find /tmp/torchinductor_root -name "*.py" | while read f; do
    if grep -q "@triton_heuristics" "$f" && \
       ! grep -q "async_compile\|def call(" "$f"; then
        echo "STANDALONE: $f"
    fi
done
```

**Strategy B (no torch.compile):** Find framework source kernels.
```bash
find /opt/venv -path "*/sglang/srt/layers/*.py" -exec grep -l "@triton.jit" {} \;
find /sgl-workspace/aiter -name "*.py" -exec grep -l "@triton.jit" {} \;
```

**Strategy B' (register-constrained OOB):** When prior unconstrained optimization showed
micro-benchmark gains but E2E regression due to register pressure / occupancy drop.
Submit to OOB agents (Codex, Claude) with explicit constraints:

```python
REGISTER_CONSTRAINED_PROMPT_ADDITION = """
CRITICAL CONSTRAINT: The previous optimization attempt achieved {micro_speedup}x
micro-benchmark speedup but REGRESSED E2E throughput by {e2e_regression}% because
of register pressure causing occupancy to drop from {old_occupancy} to {new_occupancy}.

You MUST:
1. Keep VGPR usage under {max_vgprs} registers per thread (current: {current_vgprs})
2. Target occupancy >= {min_occupancy} waves per CU
3. Use shared memory (LDS) instead of extra registers where possible
4. Prefer smaller tile sizes that fit within the register budget
5. If using Triton: set num_warps and num_stages conservatively

The kernel MUST be both faster in micro-benchmark AND maintain occupancy.
"""
```

**When to use Strategy B':**
- KB shows prior unconstrained optimization hit register pressure
- Tags: `register-pressure-fixable`, `occupancy-degraded`
- Deep kernel analysis (Step 2 of Marathon) identified the specific register budget
- **Backend:** OOB (Codex, Claude) — they can follow register constraints. GEAK handles
  register constraints internally if supported, otherwise skip GEAK for B'.

**Strategy C (selective submodule compilation):** When full-model `torch.compile` failed
but `classify.md` identified compilable submodule types, selectively compile those
submodules to generate Inductor Triton targets for GEAK.

torch.compile failures on MoE+MLA/SWA models are caused by specific components (MLA
FP8 attention, SWA memory pool, MoE expert dispatch). Normalization, activations, and
gate projections almost always compile successfully.

```python
import torch, importlib, re

SGLANG_PATH = importlib.import_module("sglang").__file__.rsplit("/", 1)[0]

# Wrap compilable submodules BEFORE server warm-up:
# (Do this in a custom launch script or monkey-patch at import)
def selective_compile(model):
    """Wrap compile-safe submodules to generate Inductor Triton targets."""
    compiled_modules = []
    for name, module in model.named_modules():
        module_type = type(module).__name__
        # These are safe on virtually all architectures:
        if module_type in ("RMSNorm", "LayerNorm", "SiLU", "GELU", "GELUActivation"):
            try:
                compiled = torch.compile(module, fullgraph=False)
                # Replace the submodule in the parent
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, compiled)
                compiled_modules.append(name)
            except Exception:
                pass  # skip if compile fails
    return compiled_modules
```

After the server warms up with selective compilation, Inductor cache at
`/tmp/torchinductor_root` will contain standalone Triton files for the compiled
submodules. Extract them with the Strategy A `find` command, then submit to GEAK/OOB
as usual.

**When to use Strategy C:**
- Full-model `torch.compile` failed (MoE+MLA, MoE+SWA, etc.)
- `classify.md` compile probe found ≥1 compilable submodule type
- Profile shows elementwise/normalization ops taking >5% GPU time as C++ dispatches
  (`vectorized_elementwise_kernel`)

**What Strategy C generates:** Inductor Triton for RMSNorm, activation functions, and
other elementwise ops. These are the same ops that showed +15% from GEAK on Qwen3
(RMSNorm dual-loop → single-pass). On non-compile models, these ops run as
`vectorized_elementwise_kernel` (C++) — invisible to GEAK without this step.

**Strategy D (call stack patching):** Rewrite Python dispatch logic in aiter/framework
to select faster kernel variants for model-specific shapes. No kernel source needed —
modify the Python wrappers that choose which compiled kernel to invoke.

```python
# Example: aiter fused_moe dispatch rewrite
# File: /sgl-workspace/aiter/aiter/ops/fused_moe.py
# The dispatch selects kernel config based on (M, N, K, topk, quant_type)
# For specific model shapes, force the optimal config instead of runtime search

AITER_PATH = importlib.import_module("aiter").__file__.rsplit("/", 1)[0]
DISPATCH_FILES = [
    f"{AITER_PATH}/ops/fused_moe.py",
    f"{AITER_PATH}/ops/gemm.py",
    f"{AITER_PATH}/fused_moe.py",
]
```

**When to use Strategy D:**
- T2 (aiter/CK dispatch) kernels >20% GPU time in profile
- Known model shapes that could benefit from specialized dispatch paths
- FP8 bypass logic that skips tuned kernels (see GLM-5 `use_cfg` fix)
- **Backend:** OOB (Claude preferred for multi-file analysis, Codex for focused patches)

**Strategy E (framework scheduling optimization):** Modify the inference framework's
scheduling, batching, and token management code to reduce idle time and improve throughput.

```python
# Target files in SGLang:
SGLANG_PATH = importlib.import_module("sglang").__file__.rsplit("/", 1)[0]
SCHEDULING_FILES = [
    f"{SGLANG_PATH}/srt/managers/schedule_batch.py",
    f"{SGLANG_PATH}/srt/managers/tp_worker.py",
    f"{SGLANG_PATH}/srt/managers/tp_worker_overlap_thread.py",
    f"{SGLANG_PATH}/srt/model_executor/forward_batch_info.py",
]

# Target files in vLLM:
VLLM_PATH = importlib.import_module("vllm").__file__.rsplit("/", 1)[0]
SCHEDULING_FILES_VLLM = [
    f"{VLLM_PATH}/core/scheduler.py",
    f"{VLLM_PATH}/engine/async_llm_engine.py",
    f"{VLLM_PATH}/worker/model_runner.py",
]
```

**When to use Strategy E:**
- T3 (framework scheduling) shows >10% idle time in profile
- High communication overhead (T4 >15%) suggests scheduling can overlap better
- `--enable-mixed-chunk` and `--num-continuous-decode-steps` already maximized
- **Backend:** OOB Claude (multi-file reasoning required), applied via `pip install -e .`

**Strategy F (kernel sequence fusion):** Write entirely new Triton or HIP kernels that
fuse multi-kernel sequences identified during per-layer analysis.

```python
# From profile: identify repeating multi-kernel patterns
# Example (gpt-oss-120b): KV cache ops = 6 small kernels (24us/layer)
# Fusion target: single kernel doing split + rope + cache_write
# See KNOWLEDGE-BASE.md "Per-Layer Kernel Sequence Analysis" for methodology

FUSION_CANDIDATES = [
    {"name": "kv_cache_fused", "sequence": ["index_elementwise", "elementwise", "cache_write"],
     "estimated_savings_us": 20, "layers": 78, "risk": "high"},
    {"name": "moe_routing_fused", "sequence": ["topkGatingSoftmax", "fill", "cast"],
     "estimated_savings_us": 10, "layers": 78, "risk": "medium"},
]
```

**When to use Strategy F:**
- Per-layer kernel sequence analysis shows 3+ small kernels that could be fused
- Total fusion savings > 2% E2E (estimated: savings_per_layer × num_layers / total_step_time)
- Existing fusion kernels in aiter/SGLang can be extended rather than written from scratch
- **Backend:** GEAK (needs GPU to compile + test HIP), OOB Claude (for Triton fusion kernels)

**Claw mode:** Kernel source lives on the RayJob. Use `exec_on_gpu` for all find/cat commands. See [`../modes/CLAW.md`](../modes/CLAW.md) "Kernel Optimization" section.

### Step 1b: Generate test harness for each candidate

Before submitting to any backend, build a test harness for each candidate kernel. This
harness is used for local verification (Codex/Claude/LLM) and as a sanity check for
GEAK results. **The test harness must use shapes extracted from the trace or standalone
file — never fabricated by the agent.**

**Why this matters:** Micro-benchmarks with fabricated shapes can be misleading.
DeepSeek-R1 GEAK showed +44% micro-benchmark but -19.9% E2E because register pressure
only manifests at real occupancy levels. Test inputs must match actual inference shapes.

#### For Inductor standalone kernels (Strategy A/C):

The standalone file itself contains all shape and dtype information:

```python
import re, torch

def generate_inductor_test_harness(standalone_file):
    """Extract shapes, dtypes, and constexprs from an Inductor standalone file.
    Returns a dict with test tensors and launch args."""
    content = open(standalone_file).read()

    # 1. Extract size_hints → real tensor dimensions
    size_hints = {}
    m = re.search(r"size_hints=\{([^}]+)\}", content)
    if m:
        for pair in re.findall(r"'(\w+)':\s*(\d+)", m.group(1)):
            size_hints[pair[0]] = int(pair[1])
    xnumel = size_hints.get('x', 1)
    r0_numel = size_hints.get('r0_', None)

    # 2. Extract dtype from inductor_meta or function body
    dtype = torch.bfloat16  # default for LLM inference
    if "fp16" in content or "float16" in content:
        dtype = torch.float16
    if "fp8" in content.lower() or "e4m3" in content:
        dtype = torch.float8_e4m3fn

    # 3. Count input/output pointers from function signature
    func_match = re.search(r"def \w+\(([^)]+)\)", content)
    if func_match:
        params = [p.strip().rstrip(':') for p in func_match.group(1).split(',')]
        in_ptrs = [p for p in params if p.startswith('in_ptr')]
        out_ptrs = [p for p in params if p.startswith('out_ptr')]

    # 4. Build test tensors matching real shapes
    shape = (xnumel, r0_numel) if r0_numel else (xnumel,)
    test_inputs = {f"in_ptr{i}": torch.randn(shape, device="cuda", dtype=dtype)
                   for i in range(len(in_ptrs))}
    test_outputs_orig = {f"out_ptr{i}": torch.empty(shape, device="cuda", dtype=dtype)
                         for i in range(len(out_ptrs))}
    test_outputs_opt = {f"out_ptr{i}": torch.empty(shape, device="cuda", dtype=dtype)
                        for i in range(len(out_ptrs))}

    # 5. Extract constexpr scalars (xnumel, r0_numel, etc.)
    constexprs = {"xnumel": xnumel}
    if r0_numel:
        constexprs["r0_numel"] = r0_numel

    return {
        "shape": shape, "dtype": dtype,
        "in_ptrs": test_inputs,
        "out_ptrs_orig": test_outputs_orig,
        "out_ptrs_opt": test_outputs_opt,
        "constexprs": constexprs,
        "standalone_file": standalone_file,
    }
```

#### For framework Triton kernels (Strategy B):

Extract shapes from the profiling trace kernel events, or inspect the calling code:

```python
def generate_framework_test_harness(kernel_name, trace_path, source_file):
    """Build test inputs from trace events for a framework Triton kernel."""
    import gzip, json
    with gzip.open(trace_path) as f:
        trace = json.load(f)

    # Find kernel launch events to extract arg shapes
    for e in trace.get('traceEvents', []):
        if e.get('cat') == 'kernel' and kernel_name in e.get('name', ''):
            args = e.get('args', {})
            # Trace events may contain grid, block, shared_mem info
            break

    # Fallback: parse the @triton.jit function signature from source
    content = open(source_file).read()
    # ... extract parameter names, find callers to determine shapes
    # This is model-specific — log shapes for the agent to fill in
    return None  # Agent must inspect callers and fill in shapes manually
```

**Framework kernels require manual shape inspection.** The agent MUST:
1. Find the caller of the kernel in the framework source
2. Log the tensor shapes passed at call sites
3. Use those shapes in the test harness

#### Micro-benchmark: the complete implementation

This is the single reference implementation. All backend docs (`geak.md`, `codex.md`,
`claude.md`, `llm.md`) and the collect step (Step 3) call this function.

**When it runs:** During `kernel-opt-collect` (Step 3), after downloading a backend's
result. The server MAY be running on the GPU at this point — the micro-benchmark
handles this.

**GPU availability constraint:** If the inference server is running, it occupies the
GPU. Two options:
1. **Server is stopped** (between DFS actions, or after a server restart): run
   micro-benchmark directly on local GPU. This is the preferred path.
2. **Server is running** (typical during collect between other DFS actions): skip
   micro-benchmark, mark result as `unverified`. Verify during `kernel-opt-integrate`
   (Step 4) after the server is killed. This means integration may try a kernel that
   ultimately fails — that's OK, the E2E benchmark in integrate.md catches it.

```python
import torch, os, importlib, copy, re

def micro_benchmark_filter(optimized_code, test_harness, original_code=None):
    """Run micro-benchmark using trace-derived test harness.
    Returns (avg_speedup, per_shape_results) or (None, error_reason).
    Call with original_code=None to auto-extract from test_harness["standalone_file"].
    """
    # 0. Check GPU availability
    try:
        torch.cuda.mem_get_info()
    except RuntimeError:
        return None, "GPU_BUSY — server is running; defer to integrate"

    # 1. Load original kernel from standalone file
    if original_code is None:
        original_code = open(test_harness["standalone_file"]).read()

    # 2. Compile both kernels in isolated namespaces
    try:
        orig_ns, opt_ns = {}, {}
        exec(compile(original_code, "original.py", "exec"), orig_ns)
        exec(compile(optimized_code, "optimized.py", "exec"), opt_ns)
    except Exception as e:
        return None, f"COMPILE_FAIL — {e}"

    # 3. Find the kernel functions by name
    kernel_name = None
    for name in opt_ns:
        if hasattr(opt_ns[name], 'run') or callable(opt_ns.get(name)):
            if name.startswith('triton_') or name.startswith('_'):
                kernel_name = name
                break
    if not kernel_name:
        return None, "NO_KERNEL — could not find kernel function in optimized code"

    original_fn = orig_ns.get(kernel_name)
    optimized_fn = opt_ns.get(kernel_name)
    if not original_fn or not optimized_fn:
        return None, f"NAME_MISMATCH — kernel '{kernel_name}' not found in both files"

    # 4. Multi-shape benchmark
    base_xnumel = test_harness["constexprs"].get("xnumel", 1)
    results = []

    for mult in [1, 4, 16, 64]:
        xnumel = base_xnumel * mult
        shape = (xnumel,) + test_harness["shape"][1:]
        dtype = test_harness["dtype"]

        # Build tensors for this shape
        in_tensors = [torch.randn(shape, device="cuda", dtype=dtype)
                      for _ in test_harness["in_ptrs"]]
        out_orig = [torch.empty(shape, device="cuda", dtype=dtype)
                    for _ in test_harness["out_ptrs_orig"]]
        out_opt = [torch.empty(shape, device="cuda", dtype=dtype)
                   for _ in test_harness["out_ptrs_opt"]]

        constexprs = dict(test_harness["constexprs"])
        constexprs["xnumel"] = xnumel

        args_orig = in_tensors + out_orig + list(constexprs.values())
        args_opt = in_tensors + out_opt + list(constexprs.values())

        # Correctness check (at base shape only)
        if mult == 1:
            try:
                original_fn(*args_orig)
                optimized_fn(*args_opt)
                torch.cuda.synchronize()
                for o, g in zip(out_opt, out_orig):
                    if not torch.allclose(o, g, atol=1e-2, rtol=1e-2):
                        max_diff = (o - g).abs().max().item()
                        return None, f"CORRECTNESS_FAIL — max diff={max_diff:.6f}"
            except Exception as e:
                return None, f"RUNTIME_FAIL — {e}"

        # Timing
        try:
            for _ in range(20):  # warmup
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
            results.append({"xnumel": xnumel, "speedup": speedup,
                            "orig_ms": orig_ms, "opt_ms": opt_ms})
        except Exception as e:
            results.append({"xnumel": xnumel, "speedup": 0, "error": str(e)})

    # 5. Verdict: reject if ANY shape regresses >5%
    if any(r["speedup"] < 0.95 for r in results):
        return None, results  # regression at some shape

    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    return avg_speedup, results
```

**What this catches vs what it misses:**

| Catches (reject early, save 15 min) | Misses (only E2E catches) |
|---|---|
| Compilation errors | Server interaction effects (scheduling, batching) |
| Numerical correctness failures | Multi-GPU communication overhead |
| Obvious regressions at any batch size | CUDA graph capture failures |
| Register pressure at higher occupancy | Memory fragmentation under load |

**Micro-benchmark is a FILTER, not the final arbiter.** It catches obviously broken or
slower kernels quickly (saves 15+ min of E2E restart time). The real truth is the E2E
benchmark in `integrate.md` — that's what determines KEEP/REVERT.

**For the OOB iterative loop** (Codex/Claude feedback between iterations): if the server
is running and GPU is busy, use `unverified` results. Feed compilation success/failure
as feedback, but defer the actual speedup measurement. The OOB agent still gets useful
feedback ("compiled successfully" or "COMPILE_FAIL — NameError: name 'libdevice' ..."),
which is enough to improve subsequent iterations. Full speed verification happens later.

### Step 2: Submit to all backends (fire-and-forget)

`KERNEL_OPT_BACKENDS` (default `geak,codex`) controls which backends run. User can
override in prompt (e.g., `"Use only geak"`, `"Use geak,codex,claude"`).

| Backend | MCP | Full round latency | GPU on pod | Reference |
|---------|-----|--------------------|------------|-----------|
| `geak` | GEAK (`geak_create_task`) | 10–30 min | Yes | [`../kernel-opt/geak.md`](../kernel-opt/geak.md) |
| `codex` | OOB Agent (`agent_create_task(agent="codex")`) | 2–6 min (3 iters) | No | [`../kernel-opt/codex.md`](../kernel-opt/codex.md) |
| `claude` | OOB Agent (`agent_create_task(agent="claude")`) | 3–15 min (3 iters) | No | [`../kernel-opt/claude.md`](../kernel-opt/claude.md) |
| `llm` | Direct OpenAI API (LLM Proxy) | 1–30s | No | [`../kernel-opt/llm.md`](../kernel-opt/llm.md) |

**For each candidate kernel, fire off all active backends and RETURN IMMEDIATELY:**

```
                     ┌─ geak:  geak_create_task + geak_submit_task → record task_id
                     │
candidate kernel ────┼─ codex: agent_create_task + agent_submit_task → record task_id
                     │
                     ├─ claude: agent_create_task + agent_submit_task → record task_id
                     │
                     └─ llm:   openai API call → result arrives immediately, store it
                     
All MCP submissions are fire-and-forget. Do NOT poll. Do NOT wait.
Record task IDs in state.pending_kernel_tasks. Return to DFS loop.
```

**The DFS loop CONTINUES with other actions** (backend switches, param tuning,
vendor-kernel-config) while kernel optimization runs asynchronously. Kernel tasks
are remote MCP calls — they need no local resources.

**Exception:** The `llm` backend returns immediately (1–30s). Run its micro-benchmark
filter right away. If it passes, store as a `pending_result` — but do NOT integrate yet.
Wait for other backends to potentially produce better results.

**Per-backend submission:**

1. **`geak`**: `geak_create_task` + `geak_submit_task` → store `task_id`. No polling yet.
2. **`codex`**: `agent_create_task(agent="codex")` + `agent_submit_task` → store `task_id`. The iterative refinement loop (3 iterations with local feedback) runs during the **collect** phase, not here.
3. **`claude`**: `agent_create_task(agent="claude")` + `agent_submit_task` → store `task_id`. Same as codex.
4. **`llm`**: Direct API call → result is immediate. Micro-benchmark filter → store as `pending_result`.

See each backend reference for full prompt templates and MCP tool details.

#### Prompt rules — shared across all backends

These rules apply to **every** kernel optimization submission regardless of backend.

1. **Kernel path — conditional on image availability:**
   - If the kernel source file **exists in the Docker image** (e.g., `/sgl-workspace/aiter/...`, `/opt/venv/...`), **MUST include** the kernel's absolute file path and repo path in the prompt. Example: `"The kernel source file is at /sgl-workspace/aiter/jit/core/compile.py"`, `"The kernel repo is at /sgl-workspace/aiter/"`.
   - If the kernel source is **runtime-generated** (e.g., `/tmp/torchinductor_root/...` from `torch.compile` Inductor cache), **DO NOT include** `kernel_url` or `kernel_repo` in the prompt. These files only exist in the running inference server's ephemeral storage, not in the Docker image. Instead, copy kernel files to a shared NFS path and reference the NFS path, OR omit these paths entirely and rely on `files[].content`.
   - **How to tell:** paths under `/tmp/`, `/root/.cache/`, or any `torchinductor_*` directory are runtime-generated. Paths under `/sgl-workspace/`, `/opt/`, `/usr/` are part of the image.
2. **1.5x minimum speedup target** — Always include: `"The kernel MUST be optimized to at least 1.5x speedup."` in the prompt.
3. **No broad filesystem searches** — Always say: `"Do NOT search the filesystem with find / or grep -r /"`.
4. **Embed full source in files** — All backends receive the kernel source via `files[].content` (or inline in the prompt for `llm`).

#### Prompt rules — backend-specific

5. **`geak` only: mode and max_rounds** — Include: `"Use homogeneous mode. Set max_rounds to 1."` GEAK's optimization engine uses these parameters. Other backends ignore them.
6. **`geak` only: framework image** — Use `GEAK_IMAGE_SGLANG` or `GEAK_IMAGE_VLLM`. In claw mode, use `GEAK_IMAGE_SGLANG_RAY` instead (see [`../modes/CLAW.md`](../modes/CLAW.md)). In local mode with `GEAK_LOCAL=true`, image is optional.
7. **`codex` / `claude` only: explicit output filename** — Include: `"Write the COMPLETE optimized file to optimized_kernel.py."` These backends need an explicit output path.

### Step 3: Collect results (between other DFS actions)

**This step runs BETWEEN other DFS actions, not blocking them.** The DFS loop pushes
a `kernel-opt-collect` sub-action onto the stack. When it gets popped (or between
other actions), poll pending tasks and collect completed results.

```python
for task in state.pending_kernel_tasks:
    if task.backend == "geak":
        result = geak_get_task(task_id=task.task_id)
    else:  # codex, claude
        result = agent_get_task(task_id=task.task_id)

    if result["status"] == "completed":
        # Download optimized kernel
        code = download_result(task)
        # Run micro-benchmark filter (local, doesn't need server restart)
        speedup = micro_benchmark_filter(code, task.test_harness)
        if speedup and speedup > task.best_speedup:
            task.best_result = {"code": code, "speedup": speedup, "backend": task.backend}
        task.status = "collected"
    elif result["status"] == "failed":
        task.status = "failed"
    # else: still running — check again later
```

**For Codex/Claude:** The iterative refinement loop (submit → local benchmark → feedback)
runs here during collect, NOT during submit. Each OOB_ROUND_ITERATIONS iteration is
one poll cycle. See [`../kernel-opt/codex.md`](../kernel-opt/codex.md) for the loop.

**Re-push if tasks are still running:** If GEAK tasks are still pending after collecting
OOB results, push `kernel-opt-collect` back onto the stack with a lower score. The DFS
loop will get to it after finishing higher-priority code-level actions.

**Do NOT wait for slow backends to "win."** A Codex result arriving in 2 minutes with
+5% micro-speedup does NOT beat a GEAK result arriving in 20 minutes with +15%.
Finishing first is irrelevant. Collect results as they arrive, keep the best so far,
but always wait for the full `KERNEL_OPT_WALL_CLOCK_MIN` budget before integrating
(unless all tasks have completed or failed).

### Step 4: Integrate the best result per kernel

**This is the only blocking step.** Integration requires a server restart, so it must
happen when the DFS loop is ready to use the server for benchmarking.

Push `kernel-opt-integrate` for each kernel that has a passing result. When popped:

1. Pick the best result across all backends (highest micro-benchmark speedup)
2. Verify function name + signature matches original exactly
3. If mismatch: re-submit with stricter prompt (max 3 attempts)
4. Patch standalone files using AST-based replacement
5. Clear binary caches (.so/.json/Triton cache)
6. Kill server, wait 10s, restart
7. **E2E benchmark** with EXACTLY same config as baseline
8. E2E speedup is the **final truth** — NOT micro-benchmark, NOT which backend finished first

| Outcome | Action |
|---------|--------|
| `actual_e2e > 0` | **KEEP**. Update baseline_tput. Log winning backend in `backend_wins`. |
| `actual_e2e <= 0` | **REVERT**. Restore backup. If another backend's result exists, try it next. |
| Crashed | **REVERT**. Log as crash. Try next-best backend result if available. |

**Late arrivals:** If GEAK finishes AFTER an OOB result was already integrated and kept,
compare GEAK's micro-benchmark against the kept result's E2E gain. If GEAK's
micro-benchmark suggests significantly better performance (>2× the kept gain), it MAY
be worth re-integrating. Otherwise, skip — the E2E-validated result stands.

### Step 5: Re-profile after kept optimizations

Kernel rankings shift after optimization. Re-profile to find new bottlenecks.

### Stopping criteria

| Condition | Action |
|-----------|--------|
| All T1-T3 candidates processed AND no new >1% candidates in any tier | Stop |
| Cumulative E2E gain > 25% | Stop — excellent |
| `KERNEL_OPT_CONSECUTIVE_DISCARDS` (7) discards across all backends+strategies | Stop — diminishing returns |
| 3+ crashes during patching | Stop — environment unstable (raised from 2 for multi-day) |
| Wall clock > `KERNEL_OPT_WALL_CLOCK_MIN` (180 min) | Stop — time budget (raised from 120 for deep opt) |
| Total submissions (all backends) > `KERNEL_OPT_MAX_SUBMISSIONS` (25) | Stop — cost budget (raised from 15) |

## Accuracy Validation
After EACH kept kernel patch, run the accuracy gate:
```bash
curl -s http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'$MODEL'","prompt":"The capital of France is","max_tokens":20,"temperature":0}' \
  | python3 -c "
import json, sys
new = json.load(sys.stdin)
ref = json.load(open('$RESULT_DIR/accuracy_reference.json'))
if new['choices'][0]['text'].strip() == ref['choices'][0]['text'].strip():
    print('ACCURACY: PASS')
else:
    print(f'ACCURACY: FAIL — expected [{ref[\"choices\"][0][\"text\"]}] got [{new[\"choices\"][0][\"text\"]}]')
    sys.exit(1)
"
```

Kernel modifications have accuracy_risk = 0.15 (reduction kernels) to 0.05 (pointwise).

**After throughput benchmark passes, run the GSM8K accuracy gate:**
```bash
EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
  RESULTS_DIR="$RESULT_DIR/eval_gsm8k_kernel_${KERNEL_NAME}" \
  bash "$SKILL_ROOT/scripts/eval_accuracy.sh"
```
Compare `exact_match` against `state.baseline_accuracy`. If accuracy drops by more than
`accuracy_threshold` (default 0.01): REVERT the kernel patch immediately, mark FAIL.

## Outputs
- Per-kernel results: (kernel_name, speedup, e2e_gain, status, winning_backend)
- `cumulative_gain_pct`: total improvement from all kept kernels
- `backend_wins`: dict of backend → number of KEEP results (update `state.backend_wins`)
- Updated baseline with all kept patches applied

## Heuristic Update
- Each kept kernel: boost similar kernel type scores (other reduction kernels likely optimizable too)
- Each discarded kernel: reduce scores **for that specific strategy+backend combination only**
- After 2+ discards from one backend on vendor kernels: reduce THAT BACKEND's score for vendor kernels, but do NOT reduce scores for other backends or strategies. A GEAK discard does NOT mean OOB agents will also fail.
- After 2+ discards across ALL backends for the SAME kernel: reduce that kernel's score to near-zero
- If the discard reason was register pressure: add `register-pressure-fixable` tag and push a new `kernel-opt-submit` with register-constrained prompt to OOB backends (Strategy B')
- Re-profiled new candidates get fresh scores based on new gpu_pct
- **NEVER zero ALL kernel-opt scores based on one backend's failures.** Each backend and strategy is an independent optimization path.

## Marathon: Kernel Rewrite + Rebuild Cycle

When OOB agents produce an optimized kernel for a compiled extension (C++/HIP/CUDA),
Marathon follows this cycle to deploy:

```
1. Deep kernel analysis identified: kernel_name, source_file, build_system, library
2. OOB agent produces: optimized_source (Triton .py, C++ .cu/.hip, or HIP .hip)
3. Patch the source file in the library's source tree
4. Rebuild the library (see actions/framework-rebuild.md)
5. Verify the correct kernel path is active (dispatch check)
6. Micro-benchmark (if GPU available)
7. E2E benchmark
8. KEEP/REVERT decision
```

This cycle is different from Sprint's basic kernel-opt which only patches standalone
Triton files (no library rebuild needed). Marathon operates at the compiled extension
level, which requires framework rebuilds.

**Rollback on REVERT:**
```bash
# Restore original source
git checkout -- "$SOURCE_FILE"
# Rebuild library with original code
pip install -e "$LIBRARY_DIR" --no-deps
# Verify dispatch is back to original path
python3 -c "from lib import kernel; print(kernel.__file__)"
```

## Failure Handling

All backends are treated equally. If one fails, the others still race.

- **GEAK** fails (workspace unavailable): retry on alternate workspace (3 attempts), other backends unaffected
- **Codex/Claude** fails (task error): other backends unaffected, log failure
- **LLM** fails (API error): other backends unaffected, retry with different model
- **All backends** produce wrong signature: re-submit all with explicit constraint
- Register OOM during Triton compile: reduce block sizes in prompt for next round
- Server crash after patch: revert, log crash, skip kernel — attribute to winning backend in `backend_wins`
