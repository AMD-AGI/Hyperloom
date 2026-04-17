---
name: llm-inference-kernel-optimization
description: LLM-based kernel optimization for inference serving, run in parallel with GEAK. Sends kernel source to an LLM (Claude, GPT) via the PRISM SAFE LLM proxy. Both GEAK and LLM are submitted simultaneously for each candidate kernel; the result with the best micro-benchmark speedup wins. Referenced by SKILL.md Phase 7 (optimization loop).
---

# LLM Inference Kernel Optimization — Deep Reference

This document provides detailed reference material for LLM-based kernel optimization in the inference optimization loop defined in `SKILL.md`. LLM and GEAK are **run in parallel** for each candidate kernel — same candidate selection, same integration paths, but the result with the best micro-benchmark speedup wins. See also [`geak.md`](geak.md) for GEAK-specific details.

## Relationship to GEAK

| | GEAK | LLM (this skill) |
|---|---|---|
| **Backend** | GEAK CLI → remote GPU pod with AI agent (REST API) | PRISM SAFE LLM proxy → Claude / GPT |
| **Latency** | 10–30 min (pod scheduling + agent) | 1–30s (direct API call) |
| **GPU access** | Yes — hardware-in-loop micro-benchmark | No — LLM writes code, you benchmark in your serving env |
| **Output** | Verified kernel (compiled + benchmarked on pod) | Unverified kernel (must compile + benchmark locally) |
| **Best for** | Complex HIP kernels, final polish, high-confidence | Fast iteration, Triton rewrites, GEAK pods overloaded |
| **Cost** | GPU pod time (shared cluster) | API tokens (~$0.01–0.50 per call) |

**Strategy: always run both in parallel.** For every candidate kernel, submit to GEAK and LLM simultaneously. LLM results arrive in seconds; GEAK results arrive in minutes. While waiting for GEAK, verify + micro-benchmark LLM results locally. When both are done, pick the winner by micro-benchmark speedup.

**Where each backend shines** (both still run; this explains which tends to win):
- **LLM wins more often on**: Triton structural rewrites (dual-loop → single-pass), simple block-size tuning, kernels where multi-model diversity finds creative solutions
- **GEAK wins more often on**: Complex HIP/C++ kernels, cases needing compile → test → fix iteration on real hardware, kernels with subtle correctness constraints

**Fallback behavior:**
- If GEAK fails (pod timeout, all 3 retries exhausted) → use best LLM result
- If all LLM models fail (compilation errors after 3 reflection rounds) → use GEAK result
- If both fail → skip kernel, move to next candidate

## Prerequisites

- `pip install openai httpx` (likely already installed)
- `LLM_PROXY_API_KEY` set in `.env` (starts with `ak-`)
- A profiling trace analyzed (TraceLens or manual kernel breakdown from SKILL.md Phase 3-5)
- Kernel source code extracted (from Inductor cache or framework source)

## Available Models (validated 2026-03-28)

Gateway: `https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy/v1`

| Model | Provider | Status | Latency | Recommendation |
|---|---|---|---|---|
| `claude-opus-4-6` | Anthropic | **Working** | ~24s | Best for complex structural optimizations. Produces multi-variant solutions. |
| `claude-opus-4.5` | Anthropic | **Working** | ~1s | Good for quick block-size tuning, simple rewrites. |
| `gpt-4.1` | OpenAI | **Working** | ~2s | Fast but may use invalid Triton APIs. Always verify. |
| `gpt-5.2` | OpenAI | **Broken** | — | 400 BadRequest. Do not use. |

## Inference Kernel Categories (Same as GEAK)

| Kernel pattern | Framework | Source available? | LLM target? |
|----------------|-----------|-------------------|-------------|
| `Cijk_Ailk_Bljk_*` | hipBLASLt | No (compiled) | No — vendor BLAS |
| `aiter::fmha_v3_fwd` | aiter | No (.so) | No — vendor attention |
| `moe_ck2stages_gemm*` | aiter | No (.so) | No — vendor fused MoE |
| `triton_*` from SGLang | SGLang | Yes (Python) | **Yes** |
| `triton_poi_*`, `triton_red_*` | torch.compile | Yes (Inductor cache) | **Yes** — primary target |
| `vectorized_elementwise_kernel` | PyTorch | No (C++) | Maybe — try torch.compile first |
| Custom HIP `__global__` | User code | Yes | **Yes** |

## LLM Optimization Flow

### Step 1: Extract kernel source (same as GEAK)

**Strategy A (torch.compile mode):** Extract from STANDALONE Inductor files.

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

### Step 2: Build the prompt

The prompt quality directly determines output quality. Include all available context:

```python
SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer specializing in AMD ROCm and Triton. "
    "Target hardware: AMD MI355X (gfx950, CDNA4, wavefront size 64, 256KB LDS per CU, "
    "304 CUs, HBM3e ~8TB/s, MFMA bf16). "
    "Context: LLM inference serving (decode path). "
    "When optimizing kernels, return the COMPLETE optimized file in a ```python code block. "
    "Focus on: eliminating redundant memory loads, optimal block sizes for wavefront 64, "
    "vectorized loads for HBM3e, register pressure management."
)
```

**For RMSNorm / reduction kernels (highest-impact target):**

```python
USER_PROMPT = f"""Optimize this Triton kernel for AMD MI355X (gfx950).

HARDWARE: gfx950, 304 CUs, HBM3e ~8TB/s, MFMA bf16, wavefront 64, 65536 VGPRs per CU.
SHAPES: xnumel={xnumel} (batch rows), r0_numel={r0_numel} (hidden_dim).
CURRENT: {gpu_pct}% of GPU time, called {call_count} times per forward pass in LLM decode.

MANDATORY CONSTRAINTS:
1. Function name MUST be EXACTLY: `{original_function_name}`. Do NOT rename.
2. Function signature MUST be IDENTICAL to original.
3. The decorator MUST be preserved.
4. R0_BLOCK <= {r0_block}. Do NOT increase beyond original value.
5. MUST produce numerically identical output.

CRITICAL OPTIMIZATION — TRUE SINGLE-PASS:
The original kernel has TWO loops that BOTH read from in_ptr0:
  Loop 1: load input → compute sum of squares
  Loop 2: RE-LOAD input → normalize with rsqrt → multiply weight → store

Since R0_BLOCK = r0_numel = {r0_numel}, each loop executes exactly ONCE. The data fits
in registers. ELIMINATE THE SECOND LOOP ENTIRELY:
  1. Load ALL inputs ONCE
  2. Compute sum of squares + rsqrt
  3. Normalize and store
  4. ZERO for-loops in the result

WARNING: This optimization is ONLY valid when R0_BLOCK = r0_numel.

```python
{kernel_source}
```

Return the COMPLETE optimized file."""
```

**For general kernels:**

```python
USER_PROMPT = f"""Optimize this Triton kernel for AMD MI355X (gfx950).

HARDWARE: gfx950, 304 CUs, HBM3e ~8TB/s, MFMA bf16, wavefront 64.
SHAPES: {shapes_from_trace}
CURRENT: {gpu_pct}% of GPU time, called {call_count} times per forward pass.

MANDATORY CONSTRAINTS:
1. Function name MUST be EXACTLY: `{original_function_name}`. Do NOT rename.
2. Function signature MUST be IDENTICAL to original.
3. Decorators MUST be preserved.
4. Do NOT increase block sizes beyond 2x original values.

OPTIMIZATION TARGETS (prioritized):
1. Eliminate redundant memory loads (merge dual-pass into single-pass)
2. Hoist loop-invariant computations
3. Adjust BLOCK sizes to match exact dimensions
4. Use libdevice.rsqrt (NOT tl.math.rsqrt)
5. Simplify grid indexing when dimensions are small

```python
{kernel_source}
```

Return the COMPLETE optimized file."""
```

### Step 3: Call the LLM

```python
from openai import OpenAI
import httpx

http_client = httpx.Client(verify=False, timeout=180)

client = OpenAI(
    base_url="https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy/v1",
    api_key=os.environ["LLM_PROXY_API_KEY"],
    http_client=http_client,
)

response = client.chat.completions.create(
    model="claude-opus-4-6",   # or claude-opus-4.5 / gpt-4.1
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
    max_tokens=8192,
    temperature=0.0,
)

optimized_code = response.choices[0].message.content
```

### Step 4: Extract and save the code

```python
import re

code_match = re.search(r'```python\n(.*?)```', optimized_code, re.DOTALL)
if code_match:
    with open("solution.py", "w") as f:
        f.write(code_match.group(1))
```

### Step 5: Verify compilation

```python
try:
    exec(open("solution.py").read())
    print("Compilation: PASS")
except Exception as e:
    print(f"Compilation: FAIL — {e}")
    # Feed error back for reflection (see below)
```

### Step 6: Parallel multi-model optimization (optional)

For maximum coverage, try multiple models on the same kernel simultaneously:

```python
import concurrent.futures

models = ["claude-opus-4-6", "claude-opus-4.5", "gpt-4.1"]

def optimize_with_model(model):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_tokens=8192,
        temperature=0.0,
    )
    return model, resp.choices[0].message.content

with concurrent.futures.ThreadPoolExecutor() as pool:
    results = list(pool.map(optimize_with_model, models))

# Test each result for compilation + correctness, pick the best
for model, code in results:
    try:
        # extract code, compile, correctness check, micro-benchmark
        ...
    except Exception:
        continue
```

## Integration Paths (Same as GEAK)

After the LLM returns optimized code, integrate using the same paths as [`geak.md`](geak.md):

### Strategy A: Standalone File Patching (torch.compile mode)

Use the `patch_standalone_kernels()` function from SKILL.md Phase 8a. The function:
- Finds all standalone kernel files matching the kernel name
- Adapts xnumel per file
- Checks r0_numel safety (single-pass only when R0_BLOCK = r0_numel)
- Backs up originals, patches, clears binary cache

```python
patch_standalone_kernels(kernel_name, "solution.py", target_signature_pattern)
```

Then kill and restart the server.

### Strategy B: Direct Source Edit (no torch.compile)

Use AST-based function replacement from SKILL.md Phase 8a Strategy B:

```python
replace_function_ast(original_source, func_name, geak_source)
```

Then clear `__pycache__` and restart.

## Reflection Loop

If the first attempt fails or regresses, feed results back to the LLM:

| Issue | Append to conversation |
|-------|----------------------|
| Compilation error | `Your solution failed to compile: {error}. Fix it.` |
| Correctness failure | `Your solution produces wrong output (max diff={diff}). Fix the computation logic.` |
| Performance regression | `Your solution is correct but {X}% slower. The bottleneck is {detail}. Eliminate redundant memory loads.` |
| Improvement < target | `Your solution achieves {Z}% speedup. Target is 10%+. Try: {suggestion}.` |

Max 3 reflection iterations per kernel. Use `temperature=0.0` for deterministic output.

## Parallel Race with GEAK

**Both backends run simultaneously for every candidate kernel.** The winner is determined by micro-benchmark speedup after correctness verification.

### Why parallel is better than sequential

1. **No wall-clock penalty**: GEAK pod scheduling (5–30 min) is the bottleneck. LLM calls (1–30s) complete during GEAK's wait time, so running both costs no extra time.
2. **Diversity wins**: LLM multi-model (3 models) + GEAK = 4 independent optimization attempts per kernel. More attempts = higher chance of finding a good structural optimization.
3. **Complementary strengths**: LLM excels at structural rewrites (dual-loop → single-pass); GEAK excels at hardware-verified tuning. The better result wins.
4. **Graceful degradation**: If one backend fails (GEAK pod timeout, LLM compilation error), the other still produces a result.

### LLM advantages (tend to make LLM the winner for Triton kernels)

1. **Speed**: 1–30s vs 10–30 min. Enables 20+ iterations in the time GEAK does 1.
2. **Multi-model**: Try Claude + GPT in parallel, pick the best.
3. **Reflection loop**: Feed compilation/correctness errors back instantly.
4. **Cheap experimentation**: Test wild ideas (aggressive fusion, entirely new algorithms) without GPU cost.

### GEAK advantages (tend to make GEAK the winner for complex kernels)

1. **Hardware verification**: GEAK compiles and benchmarks on real GPU — output is pre-validated.
2. **Iteration on hardware**: GEAK's mini-swe-agent does compile → test → fix cycles on the actual target GPU.
3. **HIP/C++ support**: GEAK can optimize non-Python kernels.

### Workflow (Phase 7 of SKILL.md)

For each candidate kernel:
1. Submit to GEAK **and** LLM (all 3 models) simultaneously
2. LLM results arrive first → verify compilation + correctness + micro-benchmark each
3. GEAK result arrives later → verify + micro-benchmark
4. **Compare**: pick the result with the best micro-benchmark speedup
5. Patch the winner → E2E benchmark → keep/revert

If one backend produces no valid result, the other wins by default. If neither produces a valid result, skip the kernel.

## Knowledge Base: LLM Inference Kernel Lessons

### Gateway Configuration (validated 2026-03-28)

```
Base URL: https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy/v1
Key format: ak-... (PRISM SAFE API key, set as LLM_PROXY_API_KEY in .env)
SSL: Must use httpx.Client(verify=False)
```

### Model-Specific Observations

- **claude-opus-4-6**: Produces thorough solutions with multiple variants (single-pass + multi-block fallback). Uses `tl.math.rsqrt` correctly. Best for RMSNorm dual-loop → single-pass merges. ~24s latency.
- **claude-opus-4.5**: Concise, correct, fast (~1s). Good enough for block-size tuning.
- **gpt-4.1**: Fast (~2s) but uses invalid Triton APIs (`tl.shared_memory`, `tl.barrier()`). Always test compilation before benchmarking. Better for brainstorming optimization ideas than producing runnable code.

### Same Integration Caveats as GEAK

- **MUST patch STANDALONE files, NOT graph module inline source** (Strategy A)
- **Clear ALL binary caches** (.so, .json, ~/.triton/cache) after patching
- **Kill server and restart** — SGLang loads kernels at startup
- **Wait 10+ seconds** between server kill and relaunch
- **r0_numel > R0_BLOCK**: single-pass is UNSAFE — do NOT eliminate loops
- **Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR`** after profiling

### When LLM Kernel Optimization Is Not Worth It

Same criteria as GEAK (see [`geak.md`](geak.md)):
- Kernel is <3% of total GPU time
- Kernel is from vendor library (aiter, hipBLASLt, CK)
- All compute is in vendor C++/ASM (>50% GPU time)
- Model is GEMM-dominated with vendor BLAS
