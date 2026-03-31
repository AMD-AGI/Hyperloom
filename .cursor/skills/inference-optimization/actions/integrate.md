# Action: Integrate, Benchmark, Decide

Per-kernel integration phase. Called by `kernel-opt.md` for each GEAK result.

## Inputs
- GEAK output: optimized kernel source file
- Original kernel location (Inductor standalone file or framework source)
- Baseline server config + benchmark params

## Procedure

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

### Re-Benchmark (CRITICAL FAIRNESS)

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

## Accuracy Validation
Run accuracy gate after each patch. Compare output text with `accuracy_reference.json`.

## Outputs
- `actual_e2e_pct`: measured gain
- `status`: KEEP / REVERT / CRASH
- Updated or restored kernel files
