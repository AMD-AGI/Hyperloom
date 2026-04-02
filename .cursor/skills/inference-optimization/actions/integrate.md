# Action: Integrate, Benchmark, Decide

Per-kernel integration phase. Called by `kernel-opt.md` for each GEAK result.

## Inputs
- GEAK output: optimized kernel source file
- Original kernel location (Inductor standalone file or framework source)
- Baseline server config + benchmark params

## Procedure

**Claw mode:** All patch, re-baseline, and revert commands go through `exec_on_gpu`. See [`../modes/CLAW.md`](../modes/CLAW.md) "Integrate" section for wrapper syntax and multi-node patching.

### Choose patching strategy

| Condition | Strategy |
|-----------|----------|
| torch.compile active + Inductor kernel | **Strategy A**: Standalone file patching |
| Framework source kernel or no torch.compile | **Strategy B**: Direct source edit with AST |

### Strategy A: Standalone File Patching

**CRITICAL:** Patch STANDALONE files, NOT graph module inline source. Graph module patching = 0% E2E. Standalone patching = +9% E2E (validated 2026-03-21).

Standalone files: `@triton_heuristics` present, `async_compile` absent, `def call(` absent.

```python
import os, re, glob, shutil

CACHE_DIR = "/tmp/torchinductor_root"

def is_standalone_kernel(content):
    return ("@triton_heuristics" in content and
            "async_compile" not in content and
            "def call(" not in content)

def patch_standalone_kernels(kernel_name, geak_source_path, target_signature_pattern):
    with open(geak_source_path) as f:
        geak_body = f.read()

    body_match = re.search(r'(def ' + kernel_name + r'\([^)]+\):)\n(.*)', geak_body, re.DOTALL)
    if not body_match:
        return 0, 0
    geak_function_body = body_match.group(2)

    patched, skipped = 0, 0
    for root, dirs, files in os.walk(CACHE_DIR):
        for f in files:
            if not f.endswith(".py") or f.endswith(".bak"):
                continue
            fpath = os.path.join(root, f)
            content = open(fpath).read()
            if kernel_name not in content or target_signature_pattern not in content:
                continue
            if not is_standalone_kernel(content):
                skipped += 1; continue

            m = re.search(r"size_hints=\{'x': (\d+)", content)
            xnumel = int(m.group(1)) if m else 4

            # Single-pass safety: R0_BLOCK must equal r0_numel
            m_hint = re.search(r"'r0_':\s*(\d+)", content)
            m_body = re.search(r"r0_numel\s*=\s*(\d+)", content)
            if m_hint and m_body and int(m_body.group(1)) > int(m_hint.group(1)):
                continue

            shutil.copy2(fpath, fpath + ".bak")
            adapted_body = re.sub(r'xnumel\s*=\s*\d+', f'xnumel = {xnumel}', geak_function_body, count=1)
            sig_pattern = r'(def ' + kernel_name + r'\([^)]+\):)\n.*'
            new_content = re.sub(sig_pattern, r'\1\n' + adapted_body, content, flags=re.DOTALL)
            open(fpath, 'w').write(new_content)
            patched += 1

    for so in glob.glob(f"{CACHE_DIR}/**/*.so", recursive=True): os.remove(so)
    for j in glob.glob(f"{CACHE_DIR}/**/*.json", recursive=True): os.remove(j)
    triton_cache = os.path.expanduser("~/.triton/cache")
    if os.path.exists(triton_cache): shutil.rmtree(triton_cache)
    return patched, skipped
```

**Alternative: Use `patch_inductor.py` (recommended, IR-8):**

```bash
# Patch a single standalone kernel file (kernel source only)
python3 $SCRIPTS_DIR/patch_inductor.py patch \
    --kernel-name <kernel_name> \
    --geak-file <geak_output.py> \
    --target-file <standalone_file_path>

# Patch kernel source AND update .best_config tiling parameters
python3 $SCRIPTS_DIR/patch_inductor.py patch \
    --kernel-name <kernel_name> \
    --geak-file <geak_output.py> \
    --target-file <standalone_file_path> \
    --best-config '{"XBLOCK": 4, "R0_BLOCK": 2048, "num_warps": 4}'

# Revert if benchmark shows regression (reverts both .py and .best_config):
python3 $SCRIPTS_DIR/patch_inductor.py revert --target-file <standalone_file_path>
```

`patch_inductor.py` preserves the original `@triton_heuristics` decorator and `inductor_meta` — it only replaces the `@triton.jit def kernel_name(...)` function body. This is critical because `inductor_meta` contains launcher configuration that Triton's CachingAutotuner depends on.

**`.best_config` updates (CRITICAL):** Inductor standalone kernels have a companion `.best_config` file in the same directory that controls tiling parameters (`XBLOCK`, `R0_BLOCK`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages`, etc.). When GEAK optimizes a kernel with different block sizes or warp counts, the `.best_config` MUST be updated to match. Patching only the `.py` without updating `.best_config` causes numerical corruption (garbled model output) because the autotuner launches the kernel with mismatched tiling parameters. Use `--best-config` to update both atomically.

**How to determine `.best_config` values:** Read the GEAK-optimized kernel source for block size constants (e.g., `XBLOCK: tl.constexpr`, `R0_BLOCK`, `BLOCK_K`) and `num_warps`/`num_stages` in the `@triton.jit` decorator or meta. These values go into the `--best-config` JSON.

**Signature Validation (auto-enforced):** `patch_inductor.py` automatically rejects patches when the GEAK kernel and target file have different function signatures (different parameter count/names). The same kernel name can map to multiple shape variants with different signatures — e.g. a `triton_red_fused_*` kernel may have 3-param (no residual) and 5-param (with residual) variants. Patching the wrong variant causes `AttributeError` crashes during torch.compile recompilation. If `patch_inductor.py` reports a signature mismatch, skip that file and find the correct variant.

### Strategy B: Direct Source Edit with AST

**CRITICAL:** Use Python AST for function boundary detection — aiter source has module-level variables between functions.

```python
import ast

def get_function_line_range(source, func_name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            start = node.lineno - 1
            end = node.end_lineno
            if node.decorator_list:
                start = node.decorator_list[0].lineno - 1
            return start, end
    return None, None
```

### Re-Baseline (CRITICAL FAIRNESS)

**Use `run_baseline.sh` to re-baseline.** There is no `run_benchmark.sh` — `run_baseline.sh` is used for all phases (initial baseline, re-baseline after patching, backend tests). Just change `RESULT_DIR` to distinguish outputs.

```bash
# Kill server, extend health timeout for torch.compile recompilation
kill_server
export HEALTH_TIMEOUT=1800

# Re-baseline with EXACTLY same env vars as baseline, only change RESULT_DIR
export RESULT_DIR="$RESULT_DIR/optimized_${KERNEL_NAME}"
bash $SCRIPTS_DIR/run_baseline.sh
```

**MUST use EXACTLY the same server config AND benchmark params as baseline:**
- `--num-continuous-decode-steps` must match
- `--mem-fraction-static` must match
- `--cuda-graph-max-bs` must match
- `--max-concurrency $CONC` must match (NEVER omit)
- `--num-prompts $((CONC * 3))` must match

**Red flag:** If TPOT INCREASES, throughput gain is from higher batching, NOT faster kernels.

### Decide

```python
actual_e2e = (new_tput - baseline_tput) / baseline_tput * 100
if actual_e2e > 0:
    # KEEP: update baseline
    baseline_tput = new_tput
else:
    # REVERT: restore .bak files
    revert_all_bak_files()
```

### Re-Baseline: torch.compile recompilation timeout

**CRITICAL:** `patch_inductor.py` clears ALL `.so` binary cache files and `~/.triton/cache`. Server restart triggers FULL torch.compile recompilation (5-30 minutes). Set `HEALTH_TIMEOUT=1800` to allow completion before health check times out.

## Accuracy Validation
Integration patches modify computation — accuracy_risk = 0.15. **After the re-baseline
benchmark passes, run the GSM8K accuracy gate:**
```bash
EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
  RESULTS_DIR="$RESULT_DIR/eval_gsm8k_integrate_${KERNEL_NAME}" \
  bash "$SKILL_ROOT/scripts/eval_accuracy.sh"
```
Compare `exact_match` against `state.baseline_accuracy`. If accuracy drops by more than
`accuracy_threshold` (default 0.01): REVERT the patch immediately, mark REVERT.

## Outputs
- `actual_e2e_pct`: measured gain
- `status`: KEEP / REVERT / CRASH
- Updated or restored kernel files
