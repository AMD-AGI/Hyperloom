# Action: Kernel Optimization

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

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME GEAK kernel optimization" --top-k 5 --compact
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

**Claw mode:** Kernel source lives on the RayJob. Use `exec_on_gpu` for all find/cat commands. See [`../modes/CLAW.md`](../modes/CLAW.md) "Kernel Optimization" section.

### Step 2: Submit candidates to active backends in parallel

`KERNEL_OPT_BACKENDS` (default `geak,codex`) controls which backends run. User can
override in prompt (e.g., `"Use only geak"`, `"Use geak,codex,claude"`).
All active backends run **simultaneously** for every candidate kernel.

| Backend | MCP | Full round latency | GPU on pod | Reference |
|---------|-----|--------------------|------------|-----------|
| `geak` | GEAK (`geak_create_task`) | 10–30 min | Yes | [`../kernel-opt/geak.md`](../kernel-opt/geak.md) |
| `codex` | OOB Agent (`agent_create_task(agent="codex")`) | 2–6 min (3 iters) | No | [`../kernel-opt/codex.md`](../kernel-opt/codex.md) |
| `claude` | OOB Agent (`agent_create_task(agent="claude")`) | 3–15 min (3 iters) | No | [`../kernel-opt/claude.md`](../kernel-opt/claude.md) |
| `llm` | Direct OpenAI API (LLM Proxy) | 1–30s | No | [`../kernel-opt/llm.md`](../kernel-opt/llm.md) |

**For each candidate kernel, launch all active backends CONCURRENTLY (not sequentially):**

```
                     ┌─ geak:  single task → poll → done
                     │
candidate kernel ────┼─ codex: iterative loop (10 iters, each: submit→benchmark→feedback)
                     │
                     ├─ claude: iterative loop (10 iters, same as codex)
                     │
                     └─ llm:   single API call → done
                     
All 4 branches run in parallel. Each branch is independent.
Wait for ALL branches to finish. Collect best result from each.
Pick the one with highest verified speedup → Step 3.
```

**Per-backend execution:**

1. **`geak`**: `geak_create_task` + `geak_submit_task` → poll until done (single submission, GEAK verifies on-pod)
2. **`codex`**: iterative refinement loop — `OOB_ROUND_ITERATIONS` (3) iterations, each: submit → download → **local benchmark** → feed result back as context. Take best speedup from all iterations. See [`../kernel-opt/codex.md`](../kernel-opt/codex.md).
3. **`claude`**: same iterative refinement loop as codex, with `agent="claude"`. See [`../kernel-opt/claude.md`](../kernel-opt/claude.md).
4. **`llm`**: `openai.Client.chat.completions.create` (multi-model parallel). See [`../kernel-opt/llm.md`](../kernel-opt/llm.md).

**IMPORTANT:** Do NOT run backends sequentially (geak first, then codex, then claude...).
Launch all active backends at the same time. The iterative loops within codex/claude
are internal to each backend — they do NOT block other backends.

Use prompt templates per kernel type:
- **RMSNorm / reduction kernels**: TRUE single-pass template (eliminate second loop, 2x memory reduction)
- **General dual-loop kernels**: Merge redundant memory loads
- **Template/GEMM kernels**: Low priority, Inductor autotuner already near-optimal

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

### Step 3: Verify + Patch each result individually

For each winning backend output:
1. Verify function name + signature matches original exactly
2. If mismatch: re-submit with stricter prompt (max 3 attempts)
3. Patch standalone files using AST-based replacement
4. Clear binary caches (.so/.json/Triton cache)
5. Kill server, wait 10s, restart
6. Benchmark with EXACTLY same config as baseline

### Step 4: Decide per kernel

| Outcome | Action |
|---------|--------|
| `actual_e2e > 0` | **KEEP**. Update baseline_tput. |
| `actual_e2e <= 0` | **REVERT**. Restore backup. |
| Crashed | **REVERT**. Log as crash. |

### Step 5: Re-profile after kept optimizations

Kernel rankings shift after optimization. Re-profile to find new bottlenecks.

### Stopping criteria

| Condition | Action |
|-----------|--------|
| All top 5 processed AND no new >3% non-vendor candidates | Stop |
| Cumulative E2E gain > 15% | Stop — excellent |
| 5 consecutive discards | Stop — diminishing returns |
| 2+ crashes during patching | Stop — environment unstable |
| Wall clock > 120 min | Stop — time budget |
| Total submissions > `GEAK_MAX_SUBMISSIONS` | Stop — cost budget |

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

Kernel modifications have accuracy_risk = 0.15 (reduction kernels) to 0.05 (pointwise). REVERT immediately on accuracy failure.

## Outputs
- Per-kernel results: (kernel_name, speedup, e2e_gain, status, winning_backend)
- `cumulative_gain_pct`: total improvement from all kept kernels
- Updated baseline with all kept patches applied

## Heuristic Update
- Each kept kernel: boost similar kernel type scores (other reduction kernels likely optimizable too)
- Each discarded kernel: reduce scores for that kernel type
- After 2+ discards on vendor-type kernels: reduce all kernel-opt scores to near-zero
- Re-profiled new candidates get fresh scores based on new gpu_pct

## Failure Handling
- GEAK workspace unavailable: retry on alternate workspace (3 attempts)
- Codex/Claude task fails: fall back to other active backends
- All backends produce wrong signature: re-submit with explicit constraint
- Register OOM during Triton compile: reduce block sizes in prompt
- Server crash after patch: revert, log crash, skip kernel
