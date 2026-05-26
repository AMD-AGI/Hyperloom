---
name: unittest-harness
description: Generate a GEAK-compatible 4-mode test harness for a kernel before GEAK optimization. Follows the same format as GEAK's own test discovery pipeline.
---

# Unittest Harness Skill

Generate a GEAK-compatible test harness for a kernel so that GEAK can evaluate
correctness and performance of optimized candidates. The harness is passed to
`kernel_optimization.py` via `--test-command`.

This skill is executed by the **main agent** (Claude Code) — it does NOT call
GEAK's API or miniswe. You read these instructions, search for tests, generate
the harness, validate it, and output the `test_command` string.

---

## Inputs

From the calling context (kernel_optimization candidate):

| Field | Description |
|-------|-------------|
| `source_file` | Absolute path to the kernel source (.py / .cu / .hip) |
| `kernel_name` | Name of the kernel function |
| `input_shapes` | TraceLens-resolved per-arg shapes (may be empty) |
| `input_dtypes` | TraceLens-resolved per-arg dtypes (may be empty) |
| `env_vars` | Runtime environment variables from profiling |
| `benchmark_files` | Known benchmark/test files for this kernel |

## Output

- A harness Python file written to the working directory
- A `test_command` string: `python3 <harness> --correctness && python3 <harness> --benchmark`

The **first command in the chain MUST include a mode flag** (e.g.
`--correctness`). GEAK's per-patch validator (`SaveAndTest`) runs the
test_command verbatim without appending a mode flag — a bare
`python3 <harness>` causes every patch to fail with
`argparse: error: one of the arguments ... is required` and the round
produces zero usable patches. See §4.2.1 for details.

If the skill fails after retries, output nothing — the caller omits
`--test-command` and GEAK falls back to its own test discovery.

---

## Phase 1 — Test Discovery

Search for existing tests/benchmarks that can serve as a harness.

### 1.1 Locate repo root

From `source_file`, walk up to find `.git` or `pyproject.toml`:

```bash
d="$(dirname "$SOURCE_FILE")"
while [ "$d" != "/" ]; do
  [ -d "$d/.git" ] || [ -f "$d/pyproject.toml" ] && break
  d="$(dirname "$d")"
done
echo "REPO_ROOT=$d"
```

### 1.2 Search for test files

**First**, check the candidate's `benchmark_files` field — these are pre-discovered
test/benchmark files from TraceLens analysis. Read each file that exists on disk
and evaluate it as a harness source. This is the **preferred** source because
these files already import and call the kernel correctly.

**Then**, search the repo more broadly:

```bash
find "$REPO_ROOT" -type f -name '*.py' \( \
  -name 'test_*.py' -o -name '*_test.py' -o \
  -name 'bench*.py' -o -name '*_benchmark.py' \
\) 2>/dev/null | head -50
```

Then grep for kernel-specific references:

```bash
grep -rl "$KERNEL_NAME" "$REPO_ROOT" --include='*.py' | head -20
```

### 1.3 Evaluate candidates

For each candidate test file, check:
1. Does it reference the kernel by name or import path?
2. Does it contain `argparse` / `ArgumentParser`?
3. Does it contain all 4 required flags (`--correctness`, `--profile`,
   `--benchmark`, `--full-benchmark`) as string literals (not just in comments)?
4. Does it contain output markers (`GEAK_SHAPES_USED`, `GEAK_RESULT_LATENCY_MS`)?

Run static validation on promising candidates:

```bash
python3 "$REPO_ROOT/kernel-agent/skills/unittest/validate_harness.py" "$CANDIDATE" --static
```

If a candidate passes static validation, go to **Phase 3** (validation).
Otherwise, proceed to **Phase 2** (generation).

### 1.4 Extract reusable test patterns

Even if no candidate passes as a complete harness, extract useful patterns
from existing test files:

- **Reference implementations**: `def ref_*`, `def torch_*`, golden-value
  computations
- **Input creation patterns**: how tensors are created, shapes, dtypes,
  device placement
- **Tolerance values**: `atol`, `rtol`, `tol_err_ratio` specific to this
  kernel
- **Import paths**: how the kernel is imported (package path, `sys.path`
  setup)
- **Shape configurations**: `ALL_CONFIGS`, `EVAL_CONFIGS`, `CTX_LENS`,
  `BATCH_SIZES`, `test_cases`, parameter lists

---

## Phase 1.5 — Adapt Existing Test to GEAK Format

**When Phase 1 found an existing test that calls the kernel but does NOT pass
GEAK static validation** (missing 4-mode flags or output markers), adapt it
rather than writing from scratch. This is the **primary path** — most repos
have tests that work but aren't in GEAK format.

### 1.5.1 Read and analyze the existing test

Read the full source of the best candidate test file. Identify:

| Component | What to look for | Example |
|-----------|-----------------|---------|
| **Kernel import** | `from X import Y`, `import X` | `from aiter import rmsnorm2d_fwd` |
| **Reference impl** | `F.rms_norm`, `torch.nn.functional.*`, `def ref_*` | `F.rms_norm(input, weight)` |
| **Input creation** | `torch.randn(...)`, `torch.empty(...)` | `torch.randn(m, n, dtype=torch.bfloat16, device="cuda")` |
| **Correctness check** | `allclose`, `checkAllclose`, `assert_close` | `checkAllclose(ref, out, atol=0.01)` |
| **Perf measurement** | `@perftest`, `torch.cuda.Event`, `do_bench` | `@perftest(num_warmup=50, num_iters=200)` |
| **Shape params** | `-m`, `-n`, `BATCH_SIZES`, default shape lists | `[8, 256, 2048, 2560, 32768]` |

### 1.5.2 Replace shapes/dtypes with TraceLens real values

The candidate JSON has `input_shapes` from actual inference profiling.
Additionally, the TraceLens category metrics file
(`tracelens/category_data/<category>_metrics.json`) contains **all distinct
call patterns** observed during profiling, with per-operation shape and call
count. Use both sources to extract every real shape.

**Where to find shapes (priority order):**

1. **TraceLens category metrics** (`category_data/<cat>_metrics.json` →
   `operations[].Input Dims` / `operations[].args`): lists every distinct
   shape with call count and duration. This is the authoritative source.
2. **Candidate `input_shapes`**: per-arg shapes from the candidate JSON.
   May only show the dominant call pattern.
3. **Candidate `shapes`**: flattened shape list (less structured).

**Shape parsing rules:**
- `"(256,128) bf16"` → shape tuple `(256, 128)`, dtype `torch.bfloat16`
- `"(256,128) f16"` / `"fp16"` → `torch.float16`
- `"(256,128) f32"` / `"fp32"` → `torch.float32`
- `"(256,128) i8"` / `"int8"` → `torch.int8`
- `"(256,128) f8e4m3"` → `torch.float8_e4m3fnuz`

**Build `ALL_CONFIGS` — target exactly 10 configs:**

When the caller merges multiple same-kernel candidates into a single
`run_optimization` request (see `kernel-agent/SKILL.md` →
"Merging same-kernel candidates"), `candidate.input_shapes` already
contains the union of TraceLens-observed shapes across all merged
kernel_ids. The harness builder doesn't need to re-merge — it just
deduplicates by `(ndim, shape_tuple, dtype)` so the same shape
observed at multiple call sites isn't tested twice.

1. **Start with all TraceLens real shapes** (typically 2-5 distinct patterns).
   These are the most important — they reflect actual production traffic.
2. **Expand to 10** by scaling the batch dimension (first dim) of the
   highest-frequency real shape. Use powers-of-2 or inference-realistic
   values (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, ...).
   - Keep the non-batch dimensions and dtype identical to the real shape.
   - Prefer values that bracket the real batch size: a few smaller, a few
     larger.
   - Skip shapes that would OOM on a single GPU (rule of thumb: total
     tensor bytes < 2 GB).
3. **Deduplicate** — if a scaled variant matches an existing real shape,
   drop it.
4. **Sort** by total element count (smallest first) so `_pick()` samples
   a representative spread.

Example for rmsnorm with 3 real shapes:
```python
# TraceLens actual runtime shapes:
#   (1024,128) bf16 — 1152 calls, 5.65ms (decode, large batch)
#   (256,128)  bf16 — 1152 calls, 5.42ms (decode, small batch)
#   (64,4096)  bf16 — 32 calls, 0.16ms  (prefill, full hidden dim)
ALL_CONFIGS = [
    # --- Real TraceLens shapes ---
    (256, 128, torch.bfloat16),     # real: decode small batch
    (1024, 128, torch.bfloat16),    # real: decode large batch
    (64, 4096, torch.bfloat16),     # real: prefill
    # --- Scaled from (M, 128) decode pattern ---
    (1, 128, torch.bfloat16),       # single-token decode
    (8, 128, torch.bfloat16),       # small batch
    (64, 128, torch.bfloat16),      # medium batch
    (512, 128, torch.bfloat16),     # between real shapes
    (2048, 128, torch.bfloat16),    # large batch stress
    # --- Scaled from (M, 4096) prefill pattern ---
    (16, 4096, torch.bfloat16),     # short prefill
    (256, 4096, torch.bfloat16),    # long prefill
]
```

### 1.5.3 Inject environment variables

From the candidate's `env_vars` field, set defaults so the harness runs in
the same environment as the profiled workload:

```python
# Inject env vars from TraceLens profiling context
import os
for k, v in {
    "CONC": "64", "ISL": "256", "OSL": "256", "TP": "2",
    # ... from candidate["env_vars"]
}.items():
    os.environ.setdefault(k, v)
```

Only include vars that affect kernel behavior (CONC, ISL, OSL, TP, etc.).
Exclude `ROCR_VISIBLE_DEVICES` (managed by GEAK) and anything with KEY/TOKEN.

### 1.5.4 Wrap in GEAK 4-mode structure

**Copy the FIXED BOILERPLATE from the Phase 2.4 template verbatim.** Only modify
the 5 kernel-specific functions (`ALL_CONFIGS`, `setup_inputs`, `run_kernel`,
`run_ref`, `config_str`). Do NOT rewrite the benchmark loop, correctness check,
mode functions, or main — these are invariant across all kernels.

Concretely:
1. **Copy verbatim** from the Phase 2.4 template: the shebang, imports, sys.path
   block, WARMUP/ITERATIONS constants, `_pick()`, `check_correctness_val()`,
   `benchmark_kernel()`, `mode_correctness()`, `mode_benchmark()`,
   `mode_profile()`, and `main()`
2. **Fill in** `ALL_CONFIGS` with shapes from 1.5.2
3. **Fill in** `setup_inputs(cfg)` using the original test's tensor creation
4. **Fill in** `run_kernel(inputs)` using the original test's kernel call
5. **Fill in** `run_ref(inputs)` using the original test's reference impl
6. **Fill in** `config_str(cfg)` to label configs human-readably

The result should be a standalone script that passes `validate_harness.py --static`.

Proceed to **Phase 3** (validation).

---

## Phase 2 — Harness Generation (from scratch)

### 2.1 Collect shapes and dtypes

Search for shape/dtype information in this priority order. Use the first
source that provides usable data:

1. **TraceLens candidate data**: `input_shapes` and `input_dtypes` from the
   candidate JSON. These reflect real inference traffic.

2. **Profile trace files**: Look for trace JSON/CSV files in the session
   directory. Parse tensor shapes from trace events.

3. **Model structure / launch scripts**: Read the model config or serving
   launch script to infer typical shapes (batch size, sequence length,
   hidden dim, num heads, etc.).

4. **Repo default test shapes**: Search for existing test files in the repo
   that define shapes for this kernel:
   ```bash
   grep -rn 'ALL_CONFIGS\|EVAL_CONFIGS\|CTX_LENS\|BATCH_SIZES\|test_cases\|SHAPES\|M\s*=\|N\s*=\|K\s*=' \
     "$REPO_ROOT" --include='*.py' | grep -i "$KERNEL_NAME" | head -20
   ```
   Also check the kernel file itself for `EVAL_CONFIGS` or shape constants.

### 2.2 Determine the kernel import path

This is critical — a wrong import path will make the harness fail immediately.

1. **Read the source file** and identify:
   - Is the kernel a `@triton.jit` function?
   - Is it a Python wrapper that calls a lower-level kernel?
   - What module is it part of?

2. **Find the package path** from the repo root:
   ```bash
   # Example: source_file = /sgl-workspace/aiter/aiter/ops/triton/rope.py
   # repo_root = /sgl-workspace/aiter
   # package_path = aiter.ops.triton.rope
   ```
   Convert the filesystem path relative to repo root into a dotted
   package path by replacing `/` with `.` and dropping `.py`.

3. **Verify the import works**:
   ```bash
   cd "$REPO_ROOT" && python3 -c "from $PACKAGE_PATH import $KERNEL_NAME; print('OK')"
   ```

### 2.3 Collect environment variables

Merge from candidate `env_vars` and current environment. Include variables
matching these patterns:
- `SGLANG_*`, `VLLM_*`, `AITER_*`, `TRITON_*`
- `HIP_*`, `ROCR_*`, `CUDA_*`, `HIPBLASLT_*`

Exclude any key containing `KEY`, `TOKEN`, or `SECRET`.

### 2.4 Generate harness file

Write a Python harness that follows the **exact GEAK format**. The template
below has two clearly marked sections:

- **FIXED BOILERPLATE** — copy verbatim, do NOT modify
- **ADAPT TO YOUR KERNEL** — fill in for the specific kernel

**Template** (copy the FIXED sections exactly, only change ADAPT sections):

```python
#!/usr/bin/env python3
import argparse
import os
import sys
import math
import torch

# ══════════════════════════════════════════════════════════════════════
# ██  FIXED BOILERPLATE — copy verbatim, do NOT modify               ██
# ══════════════════════════════════════════════════════════════════════

# --- sys.path setup (REQUIRED: GEAK patches land in GEAK_WORK_DIR) ---
REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get("GEAK_REPO_ROOT", "<ACTUAL_REPO_ROOT>"),
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- Fixed constants (from GEAK contract) ---
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def check_correctness_val(out_ref, out_kernel, dtype=torch.float16):
    tol_map = {
        torch.float32: (1e-4, 1e-4, 0.05),
        torch.float16: (1e-2, 1e-2, 0.10),
        torch.bfloat16: (1e-2, 1e-2, 0.10),
        torch.float8_e4m3fnuz: (5e-2, 5e-2, 0.20),
        torch.float8_e5m2fnuz: (5e-2, 5e-2, 0.20),
    }
    rtol, atol, max_err_ratio = tol_map.get(dtype, (1e-2, 1e-2, 0.20))
    isClose = torch.isclose(out_ref, out_kernel, rtol=rtol, atol=atol)
    err_ratio = 0.0 if isClose.all() else (~isClose).sum().item() / out_ref.numel()
    x, y = out_ref.double(), out_kernel.double()
    denom = (x * x + y * y).sum().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max(denom, 1e-12)
    return err_ratio <= max_err_ratio, err_ratio, cos_diff


def benchmark_kernel(inputs):
    """Benchmark with GPU events. Returns median latency in ms."""
    def fn():
        run_kernel(inputs)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))
    latencies.sort()
    return latencies[len(latencies) // 2]


def mode_correctness(indices):
    print(f"Running correctness check on {len(indices)} configs...")
    all_pass = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            inputs = setup_inputs(cfg)
            out = run_kernel(inputs)
            ref = run_ref(inputs)
            passed, err_ratio, cos_diff = check_correctness_val(ref, out)
            status = "PASS" if passed else "FAIL"
            print(f"  [{idx}] {label}  err_ratio={err_ratio:.4f} cos_diff={cos_diff:.2e}  {status}")
            if not passed:
                all_pass = False
        except Exception as e:
            print(f"  [{idx}] {label}  ERROR: {e}")
            all_pass = False
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={indices}")
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


def mode_benchmark(indices):
    print(f"Running benchmark on {len(indices)} configs...")
    latencies = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            inputs = setup_inputs(cfg)
            ms = benchmark_kernel(inputs)
            print(f"  {label}  {ms:.4f}ms")
            latencies.append(ms)
        except Exception as e:
            print(f"  {label}  ERROR: {e}")
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={indices}")
    if latencies:
        geo_mean = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")
    else:
        print("No successful benchmarks")
        sys.exit(1)


def mode_profile(indices):
    print(f"Running profile on {len(indices)} configs...")
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = config_str(cfg)
        try:
            inputs = setup_inputs(cfg)
            run_kernel(inputs)
            print(f"  {label}  OK")
        except Exception as e:
            print(f"  {label}  ERROR: {e}")
        finally:
            torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={indices}")


def main():
    parser = argparse.ArgumentParser(description="Test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    total = len(ALL_CONFIGS)
    print(f"Total configs: {total}")
    if args.correctness:
        mode_correctness(_pick(ALL_CONFIGS, 25))
    elif args.benchmark:
        mode_benchmark(_pick(ALL_CONFIGS, 25))
    elif args.full_benchmark:
        mode_benchmark(list(range(total)))
    elif args.profile:
        mode_profile(_pick(ALL_CONFIGS, 5))


if __name__ == "__main__":
    main()

# ══════════════════════════════════════════════════════════════════════
# ██  ADAPT THESE TO YOUR KERNEL — the only parts you should change  ██
# ══════════════════════════════════════════════════════════════════════
#
# Place these BETWEEN the sys.path/constants block and the FIXED
# functions above. The 5 items to fill in:
#
# 1. Kernel import:
#        from <actual.package.path> import <actual_kernel_function>
#    MUST use package path, NOT importlib.util.
#
# 2. ALL_CONFIGS — list of tuples, one per test case shape:
#        ALL_CONFIGS = [
#            (M, N, dtype),          # from TraceLens
#            (M2, N2, dtype),        # scaled variant
#            ...                     # target ≥ 6 configs
#        ]
#
# 3. setup_inputs(cfg) — create input tensors:
#        def setup_inputs(cfg):
#            torch.manual_seed(42)
#            M, N, dtype = cfg
#            x = torch.randn(M, N, dtype=dtype, device="cuda")
#            w = torch.randn(N, dtype=dtype, device="cuda")
#            return {"x": x, "w": w, "eps": 1e-6}
#
# 4. run_kernel(inputs) — call the kernel:
#        def run_kernel(inputs):
#            return my_kernel(inputs["x"], inputs["w"], inputs["eps"])
#
# 5. run_ref(inputs) — reference implementation:
#        def run_ref(inputs):
#            x = inputs["x"].float()
#            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + inputs["eps"])
#            return (x / rms * inputs["w"].float()).to(inputs["x"].dtype)
#
# 6. config_str(cfg) — human-readable label:
#        def config_str(cfg):
#            M, N, dtype = cfg
#            return f"M={M} N={N} {dtype}"
```

### 2.5 Key rules (validate_harness.py enforces rules 1-3 as hard failures)

**These 3 rules cause hard validation failures if violated:**

1. **GEAK_WORK_DIR MUST be first in sys.path**. GEAK places its patched
   kernel candidate in `GEAK_WORK_DIR`. If the harness doesn't add it to
   `sys.path` before importing, the ORIGINAL unmodified kernel is tested
   instead of the patch → GEAK thinks every candidate is 1.00x → no
   optimization happens. This is the #1 cause of "GEAK did nothing" bugs.

2. **GPU event timing ONLY**. Use `torch.cuda.Event(enable_timing=True)`,
   NEVER `time.perf_counter()` or `time.time()`. GPU kernels execute
   asynchronously — wall-clock timing measures Python overhead, not kernel
   latency. `validate_harness.py` rejects any harness using wall-clock timing.

3. **GEAK_BENCHMARK_ITERATIONS from env**. The iteration count MUST be read
   from `os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")`. GEAK uses
   this to control evaluation speed vs. accuracy. Hardcoding the value
   breaks the contract.

**Additional rules (important but not enforced by static validation):**

4. **Output markers**: All modes print `GEAK_SHAPES_USED=<indices>`.
   Benchmark modes also print `GEAK_RESULT_LATENCY_MS=<geometric_mean>`.

5. **Geometric mean for latency**: `GEAK_RESULT_LATENCY_MS` must be the
   geometric mean — `math.exp(sum(math.log(x) ...) / n)`, NOT arithmetic
   mean.

6. **Import via package path** (`from X import Y`), NOT `importlib.util`.
   Triton kernels have deep import chains that break with `exec_module`.

7. **Fixed seed**: `torch.manual_seed(42)` in `setup_inputs`.

8. **4 mutually exclusive modes** via argparse:
   | Mode | Configs | Correctness | Performance |
   |------|---------|-------------|-------------|
   | `--correctness` | `_pick(ALL_CONFIGS, 25)` | Yes | No |
   | `--profile` | `_pick(ALL_CONFIGS, 5)` | No | No |
   | `--benchmark` | `_pick(ALL_CONFIGS, 25)` | No | Yes |
   | `--full-benchmark` | All | No | Yes |

---

## Phase 3 — Validation + Retry

### 3.0 Pre-validation self-check

Before running `validate_harness.py`, visually confirm **every item** below.
These are the most common reasons for harness rejection:

- [ ] **GEAK_WORK_DIR is first in sys.path** — without this, GEAK's patched
  kernel is never tested (the harness imports the original instead)
- [ ] **`benchmark_kernel()` uses `torch.cuda.Event(enable_timing=True)`** —
  NOT `time.perf_counter()` or `time.time()` (GPU kernels are async)
- [ ] **ITERATIONS reads `GEAK_BENCHMARK_ITERATIONS` env** — GEAK controls
  this value; hardcoding it breaks the evaluation contract
- [ ] **`GEAK_RESULT_LATENCY_MS` uses geometric mean** —
  `math.exp(sum(math.log(x) for x in latencies) / len(latencies))`
- [ ] **ALL_CONFIGS has ≥ 6 entries** — TraceLens real shapes + scaled variants
- [ ] **All 4 modes print `GEAK_SHAPES_USED`** marker
- [ ] **`--benchmark` and `--full-benchmark` print `GEAK_RESULT_LATENCY_MS`**
- [ ] **Kernel import uses package path** (`from X import Y`), not importlib
- [ ] **`torch.manual_seed(42)` in `setup_inputs`** — reproducible tensors
- [ ] **`run_kernel` calls the kernel under test**, not the reference impl

If any item is missing, fix it before proceeding. `validate_harness.py` will
catch most of these, but self-checking first saves a retry cycle.

### 3.1 Static validation

```bash
python3 "$SKILL_DIR/validate_harness.py" "$HARNESS_PATH" --static
```

where `$SKILL_DIR` is the absolute path to `kernel-agent/skills/unittest/`.

If static check fails, read the errors, fix the harness, and retry.

### 3.2 Runtime validation

```bash
python3 "$SKILL_DIR/validate_harness.py" "$HARNESS_PATH" --all
```

This runs correctness and benchmark modes with `GEAK_BENCHMARK_ITERATIONS=5`
(fast validation). Each mode has a 300s timeout. Runtime validation
short-circuits: if correctness fails, benchmark is skipped.

### 3.3 Retry logic

If validation fails:
1. Read the JSON error output — it includes `stderr_tail` and `stdout_tail`
   to diagnose the issue
2. Fix the harness. Common issues:
   - **ImportError**: wrong package path, missing `sys.path` entry, or
     `GEAK_WORK_DIR` not handled
   - **Shape mismatch**: tensor dimensions don't match kernel signature
   - **Dtype mismatch**: kernel expects bf16 but harness creates fp16
   - **Missing dependency**: kernel needs a helper module not imported
   - **OOM**: configs too large for GPU memory — reduce shapes or add
     `torch.cuda.empty_cache()` between configs
3. Re-validate with `--all`

Maximum 3 attempts. If all fail, **do not pass `--test-command`** — let GEAK
use its own test discovery.

### 3.4 Diagnosing common failures

| Error pattern | Likely cause | Fix |
|---------------|-------------|-----|
| `ModuleNotFoundError` | Wrong import path | Verify with `python3 -c "from X import Y"` |
| `RuntimeError: expected ... got ...` | Tensor shape/dtype mismatch | Check kernel signature carefully |
| `CUDA out of memory` | Configs too large | Reduce batch sizes, add `empty_cache()` |
| `CORRECTNESS FAILED` | Bad reference impl or wrong tolerance | Compare against existing test's reference |
| `Timed out after 300s` | Kernel too slow at given shapes | Reduce config space, smaller shapes |
| Missing `GEAK_SHAPES_USED` | Mode function doesn't print marker | Check print statement after loop |

---

## Phase 4 — Output

### 4.1 Harness file location

Write the harness into the GEAK output directory's `unittest/` subdirectory:

```
$USER_DATA_PATH/kernel-agent/runs/<session_id>/unittest/<kernel_id>_harness.py
```

`kernel_optimization.py` will copy this file into the GEAK attempt's output
directory (`geak/<session_id>/<attempt_id>/unittest/`) for traceability.

### 4.2 Output values

After successful validation, output:

1. **Harness file path**: The absolute path to the generated harness
2. **test_command**: The command string to pass via `--test-command`:
   ```
   python3 <harness_path> --correctness && python3 <harness_path> --benchmark
   ```

The caller (`kernel_optimization.py`) will pass this as
`--test-command "<test_command>"` to GEAK.

#### 4.2.1 test_command MUST include a mode flag (GEAK SaveAndTest contract)

The first command in the `test_command` chain MUST include one of the four
mode flags (`--correctness` recommended). This is because GEAK has TWO
different consumers of `test_command`, and they call it differently:

| Consumer | Behavior | Mode-flag handling |
|----------|----------|--------------------|
| `Preprocessor` (Step 3 of GEAK's preflight) | Re-invokes the harness explicitly with each of the 4 modes to collect baselines | Appends its own mode flag — your trailing flag is ignored here |
| `SaveAndTest` (sub-agent's per-patch validator) | Runs the test_command **verbatim** after each generated patch | Does NOT append a mode flag — relies on you to provide one |

If the test_command is bare `python harness.py` (no flag), `SaveAndTest` will
fail every patch with:
```
usage: harness.py [-h] (--correctness | --benchmark | --full-benchmark | --profile)
harness.py: error: one of the arguments ... is required
```
GEAK then marks every candidate as broken and the optimization round produces
zero usable patches.

**Always include `--correctness` in the first command** of the chain. It runs
fast (~5 s) and is what GEAK's sub-agent should check on every patch attempt
anyway.

#### 4.2.2 Avoid `SameFileError` in the harness-copy step

`kernel_optimization.py` copies any `.py` files referenced in `test_command`
into `<out_dir>/unittest/` for traceability. If your harness is **already
inside** `<out_dir>/unittest/` (the case when this skill writes there
directly), the copy step will hit `shutil.SameFileError`.

The caller is responsible for guarding the copy with an `os.path.samefile`
check (already fixed in `kernel_optimization.py` L1403-1410), but if you
build your own dispatcher, mirror the same guard:
```python
if _dst.exists() and _dst.resolve() == Path(src).resolve():
    continue  # source and destination are the same file
_shutil.copy2(src, _dst)
```

---

## Fallback Behavior

If the skill cannot produce a valid harness (no existing tests found,
generation fails, validation fails after 3 retries):

- Do NOT pass `--test-command` to `kernel_optimization.py`
- GEAK will fall back to its own 6-layer test cascade (discovery →
  UnitTestAgent → shape fixer → raw fallback)
- Log the failure reason for debugging

This ensures the optimization pipeline never blocks on harness generation.

---

## Language-Specific Notes

### Triton kernels (.py with @triton.jit)

- Import via package path: `from aiter.ops.triton.rope import rope_fwd`
- Add repo root to `sys.path` for import resolution
- `GEAK_WORK_DIR` must be first in `sys.path` so patched kernels are tested
- Reference implementation: use equivalent PyTorch operations
- If the kernel has a wrapper → inner kernel split, both the wrapper and
  the `@triton.jit` function must be importable

### HIP/C++ kernels (.cu, .hip, .cpp)

- If Python bindings exist (pybind11 / `torch.ops`), call them directly
- If no bindings, create a Python wrapper using `subprocess` to call the
  compiled binary
- The harness must still output GEAK markers to stdout
- For aiter JIT modules: `import aiter.jit.module_<name>` triggers
  ninja compilation on first import (~60-90s). Account for this in
  timeout expectations.

### Multi-GPU kernels

- Do NOT generate harnesses for multi-GPU collective kernels
  (`is_multigpu: True`)
- These use `torchrun --nproc_per_node=N` which is incompatible with
  GEAK's sandbox
- Let the caller fall back to the legacy `torchrun` benchmark path
