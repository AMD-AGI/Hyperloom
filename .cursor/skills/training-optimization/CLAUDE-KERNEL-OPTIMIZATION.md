---
name: llm-kernel-optimization
description: Optional LLM-based kernel optimization that extends the workload-optimization skill. Instead of sending kernels to GEAK MCP (remote GPU pod), this flow sends kernel source to an LLM (Claude, GPT, etc.) via your existing LiteLLM gateway and gets back optimized code. No CLI tools or GPU required for the LLM call. Faster turnaround than GEAK (no pod scheduling). Use alongside the main workload-optimization skill as an alternative to GEAK when you want faster iteration or GEAK pods are unavailable.
---

# LLM Kernel Optimization — Optional Extension

## Relationship to Main Skill and GEAK

This skill is an **alternative** to `GEAK-KERNEL-OPTIMIZATION.md`. Both extend the main `SKILL.md` with kernel-level optimization, but they use different backends:

| | GEAK | LLM (this skill) |
|---|---|---|
| **Backend** | GEAK MCP → remote GPU pod | LiteLLM gateway → Claude / GPT / any model |
| **Install needed** | GEAK MCP configured | Just `pip install openai` (likely already installed) |
| **Latency** | 10–30 min (pod scheduling + execution) | 30s–5 min (direct API call) |
| **GPU access** | Yes — dedicated pod with target GPU | No — LLM writes code, you benchmark in your training env |
| **Micro-benchmark** | GEAK runs its own benchmarks | You run benchmarks via `torchrun` (same as other optimization attempts) |
| **Best for** | Standalone kernel optimization with hardware-in-loop | Fast iteration, quick rewrites, when GEAK pods are overloaded |

**When to use LLM instead of GEAK:**
- GEAK pods are overloaded or unavailable (queue > 30 min)
- You want faster turnaround and can benchmark in your training environment
- The kernel optimization is more about algorithmic restructuring than low-level tuning
- You want to iterate quickly on Triton block sizes or fusion patterns

**When to prefer GEAK:**
- You need hardware-in-loop micro-benchmarking before E2E integration
- The kernel requires compiled HIP/C++ changes that need a dedicated GPU environment
- You want GEAK's specialized kernel optimization agent (mini-swe-agent)

You can use **both** in parallel: kick off GEAK for complex kernels while using LLM for simpler ones.

## Prerequisites

**Minimum (LiteLLM backend — default, recommended):**
- `pip install openai` (likely already installed)
- `LITELLM_API_KEY` and `LITELLM_BASE_URL` in your `.env` file (already configured in agentic-rc)
- That's it. No CLI tools, no Node.js, no separate auth flows.

**Optional (agentic backends — only if you want the agent to read/write files autonomously):**
- **Claude Code SDK**: `pip install claude-code-sdk` + `ANTHROPIC_API_KEY` or `claude auth login`
- **Codex CLI**: `npm install -g @openai/codex` + `OPENAI_API_KEY` or `codex login`

**For integration testing (needs GPU):**
- A TraceLens analysis already completed (trace file + kernel profile from main skill)
- The source code for the candidate kernel (Triton `.py`, HIP `.hip`/`.cu`, or extracted from Python)

## Phase 1: Identify Candidates (Same as GEAK)

Use the same candidate selection rules from `GEAK-KERNEL-OPTIMIZATION.md`:

| Kernel type in profile | Applicable? | Why |
|------------------------|-------------|-----|
| `Cijk_*` (hipBLASLt GEMM) | **No** | Vendor BLAS; hand-tuned MFMA |
| `aiter::fmha_v3_*` | **No** | Vendor attention; already optimized |
| `triton_*` / `triton::` | **Yes** | Triton kernels have Python source |
| Custom HIP kernels (`__global__`) | **Yes** | Primary target |
| `vectorized_elementwise_kernel` | **Maybe** | Try `torch.compile` first |

**Decision rule:** Top-5 by GPU time, has modifiable source, NOT vendor BLAS.

## Phase 2: Extract Kernel Source (Same as GEAK)

Follow the same extraction steps from `GEAK-KERNEL-OPTIMIZATION.md` Phase 2:
- For Triton: `rg "@triton.jit"` in the workload tree
- For HIP: `rg "void <kernel_name>"` in `.hip`/`.cu` files
- For torch.compile: extract from `/tmp/torchinductor_*/`
- For Inductor: use `inductor_extract.py` if available

Save the kernel source to a working directory (e.g., `/tmp/claude_kernel_opt/<kernel_name>/`).

## Phase 3: Run the LLM Optimization

### Option A: Helper Script with LiteLLM Gateway (Default — Simplest)

The helper script at `.cursor/agents/executors/agent_kernel_optimizer.py` uses your existing LiteLLM gateway. **No extra CLI tools needed** — just the `openai` Python package (already installed).

It reads credentials from your `.env` file automatically (`LITELLM_API_KEY`, `LITELLM_BASE_URL`, `LITELLM_MODEL`).

```bash
# Simplest — uses .env config, default model (claude-opus-4-6)
python3 .cursor/agents/executors/agent_kernel_optimizer.py \
  --kernel-source /path/to/kernel.py \
  --task-dir /tmp/kernel_opt/triton_rmsnorm/

# With profiling context (better results)
python3 .cursor/agents/executors/agent_kernel_optimizer.py \
  --kernel-source /path/to/kernel.py \
  --task-dir /tmp/kernel_opt/triton_rmsnorm/ \
  --hardware "MI355X (gfx950, CDNA4)" \
  --shapes "x: [8192, 2880] bf16, weight: [2880] bf16" \
  --gpu-time-pct 4.2 \
  --call-count 128

# With multiple refinement iterations
python3 .cursor/agents/executors/agent_kernel_optimizer.py \
  --kernel-source /path/to/kernel.py \
  --task-dir /tmp/kernel_opt/triton_rmsnorm/ \
  --max-iterations 3

# Dry run (show prompt, no LLM call, no GPU needed)
python3 .cursor/agents/executors/agent_kernel_optimizer.py \
  --kernel-source /path/to/kernel.py \
  --task-dir /tmp/kernel_opt/triton_rmsnorm/ \
  --dry-run

# Explicit model/credentials (override .env)
python3 .cursor/agents/executors/agent_kernel_optimizer.py \
  --kernel-source /path/to/kernel.py \
  --task-dir /tmp/kernel_opt/triton_rmsnorm/ \
  --model claude-opus-4-6 \
  --base-url https://tw325.primus-safe.amd.com/llm-gateway/v1 \
  --api-key sk-...
```

The script:
1. Reads the kernel source and your `.env` config
2. Copies source to `task_dir/baseline.py`
3. Sends it to the LLM via a standard OpenAI-compatible chat completions call
4. Extracts the optimized kernel code from the LLM response
5. Writes it to `task_dir/solution.py`
6. Saves metadata to `task_dir/agent_result.json`

**No GPU needed for the LLM call.** GPUs are only needed later when you benchmark the result via `torchrun`.

### Option B: Inline Python Call (for scripting)

```python
from pathlib import Path
from openai import OpenAI

kernel_source = Path("kernel.py").read_text()

client = OpenAI(
    base_url="https://tw325.primus-safe.amd.com/llm-gateway/v1",
    api_key="sk-...",  # or os.environ["LITELLM_API_KEY"]
)

response = client.chat.completions.create(
    model="claude-opus-4-6",
    messages=[
        {"role": "system", "content": "You are an expert GPU kernel engineer..."},
        {"role": "user", "content": f"Optimize this kernel for MI355X:\n```python\n{kernel_source}\n```\nReturn optimized code in a ```python code block."},
    ],
    max_tokens=8192,
    temperature=0.0,
)

optimized_code = response.choices[0].message.content
# Extract code from ```python ... ``` block and write to solution.py
```

### Option C: Claude Code SDK (Agentic — Needs CLI Installed)

Only use this if you have Claude Code CLI installed and want the agent to autonomously read/write files:

```bash
python3 .cursor/agents/executors/agent_kernel_optimizer.py \
  --kernel-source /path/to/kernel.py \
  --task-dir /tmp/kernel_opt/triton_rmsnorm/ \
  --agent claude --model claude-sonnet-4-6
```

Requires: `pip install claude-code-sdk` + `ANTHROPIC_API_KEY` or `claude auth login`.

## Phase 4: Validate and Integrate

After the agent writes `solution.py`, follow the same integration flow as GEAK:

### 4a. Correctness Check (Mandatory)

```python
import torch

# Load both kernels
exec(open("baseline.py").read())  # defines original_fn
exec(open("solution.py").read())  # defines optimized_fn

# Test with representative inputs
x = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)
orig_out = original_fn(x)
opt_out = optimized_fn(x)
assert torch.allclose(orig_out, opt_out, atol=1e-2, rtol=1e-2), \
    f"Correctness failed! max diff: {(orig_out - opt_out).abs().max()}"
print("Correctness: PASS")
```

### 4b. Micro-benchmark (Optional but Recommended)

```python
import torch, time

# Warmup
for _ in range(50):
    original_fn(x)
    optimized_fn(x)
torch.cuda.synchronize()

# Benchmark
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(200):
    original_fn(x)
end.record()
torch.cuda.synchronize()
orig_ms = start.elapsed_time(end) / 200

start.record()
for _ in range(200):
    optimized_fn(x)
end.record()
torch.cuda.synchronize()
opt_ms = start.elapsed_time(end) / 200

print(f"Original:  {orig_ms:.3f} ms")
print(f"Optimized: {opt_ms:.3f} ms")
print(f"Speedup:   {(orig_ms - opt_ms) / orig_ms * 100:.1f}%")
```

If the micro-benchmark shows regression, skip integration.

### 4c. Integrate into Training

Use the same paths as GEAK integration (see `GEAK-KERNEL-OPTIMIZATION.md` Phase 4):

**Path A (Inductor):** Patch into Inductor cache via `inductor_extract.patch_kernel_in_file()`

**Path B (Manual Triton):** Replace the kernel in the training stack source file

**Path C (HIP):** Use `torch.utils.cpp_extension.load_inline()`

### 4d. E2E Benchmark

This is a normal optimization attempt in the main skill's loop:

```bash
torchrun --nproc_per_node=<NUM_GPUS> --master_port=<PORT> \
  -m primus.cli.main train pretrain \
  --config <CONFIG_YAML> \
  <all_kept_overrides> \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/attempt_claude_kernel.log
```

Extract ms/iter from iterations 6–10. Keep or revert based on the main skill's DECIDE rules.

## Phase 5: Reflection Loop (Optional)

Apex uses a reflection loop where agent results are graded and fed back for improvement. You can do the same:

1. **Agent writes solution.py** (Phase 3)
2. **You test correctness** (Phase 4a)
3. **If compile/correctness failure**, feed back to agent:
   ```
   Your previous solution failed: <error>. Fix and write a corrected solution.py.
   ```
4. **If performance regression**, feed back:
   ```
   Your solution is correct but slower (X ms vs Y ms baseline).
   Focus on: <specific bottleneck from profiling>. Write an improved solution.py.
   ```
5. **If improvement < target**, feed back:
   ```
   Your solution achieves Z% speedup. Target is 10%+.
   Consider: <optimization suggestion>. Write an improved solution.py.
   ```

The helper script supports this via the `--max-iterations` flag, which re-runs the agent with reflection prompts.

## Reporting

In the final optimization report, add:

```markdown
## LLM Kernel Optimization (Optional Flow)

### Candidates Identified
| Kernel | GPU Time % | Source Type | Model |
|--------|-----------|-------------|-------|
| triton_rmsnorm | 4.2% | Triton | claude-opus-4-6 |
| custom_silu_mul | 3.1% | HIP | claude-opus-4-6 |

### Results
| Kernel | Original (ms) | Optimized (ms) | Speedup | Integrated? |
|--------|--------------|----------------|---------|-------------|
| triton_rmsnorm | 0.85 | 0.74 | +12.9% | Yes |
| custom_silu_mul | 0.61 | 0.63 | -3.3% | No (regression) |

### Details
- triton_rmsnorm: claude-opus-4-6 via LiteLLM, 1 iteration, 45s
- custom_silu_mul: claude-opus-4-6 via LiteLLM, 2 iterations, 1.8 min
```

## Knowledge Base: LLM Kernel Optimization Lessons

### Model Choice

- **claude-opus-4-6** (via LiteLLM): Strong at Triton code with AMD-specific patterns (MFMA, LDS, wavefront-aware tiling). Good at structural optimizations.
- **claude-sonnet-4-6** (via LiteLLM or Claude Code SDK): Faster, cheaper, good enough for straightforward block-size tuning.
- Any model available on your LiteLLM gateway works — the script uses a standard OpenAI-compatible API.
- The LLM has no GPU access during optimization — it reasons about performance but can't profile. Always benchmark the output.

### Prompt Quality Matters

The agent's output quality scales directly with context provided:
- **Always include**: hardware target (gfx950), dtype (bf16), exact input shapes, current kernel time
- **Include if available**: TraceLens roofline data, memory bandwidth utilization
- **Include if available**: the calling code (how the kernel is launched, grid dimensions)
- **Include the baseline source** in the prompt — don't just point to a file path

### Agent May Produce Non-Compiling Code

LLM-generated kernels may:
- Use CUDA-only APIs on ROCm (e.g., `__syncthreads()` instead of `__syncthreads()` — actually the same, but other CUDA intrinsics differ)
- Import modules not available in your environment
- Use Triton APIs from a different version

Always compile/import first before benchmarking. Fix obvious issues if possible.

### Agent May Not Respect AMD Architecture

Common agent mistakes on AMD ROCm:
- Using NVIDIA-specific block sizes (e.g., warp size 32 instead of wavefront size 64)
- Suggesting shared memory sizes > 64 KB per CU
- Using CUDA graph APIs not supported on ROCm
- Not accounting for MFMA instruction latency

Include AMD architecture details in the prompt to mitigate this.

### Faster Iteration Than GEAK

Typical timing comparison:
- **GEAK**: 10–30 min (2–15 min pod scheduling + 3–10 min agent)
- **LLM via LiteLLM**: 30s–5 min (direct API call, no scheduling)

Use LLM calls for fast iteration during the optimization loop. Use GEAK for final polish or when you need hardware-in-loop micro-benchmarks.

### When LLM Kernel Optimization Is Not Worth It

Same as GEAK:
- Kernel is < 2% of total GPU time (even 50% speedup = < 1% E2E)
- Kernel is a single simple op (element-wise add, copy) — bandwidth-bound
- Kernel is from a vendor library (aiter, hipBLASLt, CK)
- Workload is GEMM-dominated (> 60% GEMM)
