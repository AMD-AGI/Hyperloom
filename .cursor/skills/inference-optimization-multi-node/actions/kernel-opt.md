# Action: Kernel Optimization

Multi-round kernel/code optimization loop using OOB Codex and Claude backends.

Backend references:
- [`../kernel-opt/codex.md`](../kernel-opt/codex.md) — Codex via `oob_ray_submit.py run` CLI
- [`../kernel-opt/claude.md`](../kernel-opt/claude.md) — Claude Code via `oob_ray_submit.py run` CLI

## Inputs
- Kernel candidates from `profile.md` (kernel_name, gpu_pct, source_location)
- Current best config (backends + params)
- `baseline_tput_per_gpu` (after backends + params)

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME OOB kernel optimization" --top-k 5 --compact
python3 $SKILL_ROOT/kb/kb_query.py --category kernel_optimization --compact
```

## Procedure

**FLOW GUARD:** Do NOT skip this action if candidates exist. Running sweep with unoptimized kernels wastes compute.

### Step 0: Tracing setup (once per backend)

Use `trace_action.py` to record start/end for each external component:

**OOB (Codex/Claude)** — if `codex` or `claude` is in `KERNEL_OPT_BACKENDS`:
1. `python3 $SCRIPTS_DIR/trace_action.py --component oob --action start --agent <codex|claude>`
2. After ALL OOB iterations: `python3 $SCRIPTS_DIR/trace_action.py --component oob --action end`

See each backend's skill doc for details. OOB header injection is automatic via
`auth_proxy.py` / the bootstrap auth proxy when configured.

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

**Remote mode:** Kernel source lives on the RayJob. Use `exec_on_gpu` for all find/cat commands; it submits work through Ray Dashboard REST (`POST /api/jobs/`). See [`../modes/REMOTE.md`](../modes/REMOTE.md) "Kernel Optimization" section.

### Step 2: Submit candidates to OOB backends in parallel

| Backend | Invocation | Full round latency | GPU on pod | Reference |
|---------|------------|--------------------|------------|-----------|
| `codex` | `python $SKILL_ROOT/scripts/oob_ray_submit.py run -a codex` | 2–6 min (3 iters) | No | [`../kernel-opt/codex.md`](../kernel-opt/codex.md) |
| `claude` | `python $SKILL_ROOT/scripts/oob_ray_submit.py run -a claude` | 3–15 min (3 iters) | No | [`../kernel-opt/claude.md`](../kernel-opt/claude.md) |

`KERNEL_OPT_BACKENDS` (default `codex,claude`) controls which OOB agents run. User can
override in prompt (e.g., `"Use only codex"`, `"Use codex,claude"`). All active OOB
agents run **simultaneously** for every candidate kernel where capacity allows.

```
candidate kernel ────┬─ codex:  iterative loop (3 iters, each: submit→benchmark→feedback)
                     └─ claude: iterative loop (3 iters, same as codex)

Both branches run in parallel when both are enabled. Each branch is independent.
Wait for ALL branches to finish. Collect best result from each.
Pick the one with highest verified speedup → Step 3.
```

**Per-backend execution:**

1. **`codex`**: iterative refinement loop — `OOB_ROUND_ITERATIONS` (3) iterations, each: submit → read workspace → benchmark/validate → feed result back as context. Take best speedup from all iterations. Each iteration is one blocking `python $SKILL_ROOT/scripts/oob_ray_submit.py run -a codex -p "$PROMPT" -f kernel.py --max-turns 20 --no-live --json` call; read `$(jq -r .workspace)`/`optimized_kernel.py` from the result (no `output/` subdir). See [`../kernel-opt/codex.md`](../kernel-opt/codex.md).
2. **`claude`**: same iterative refinement loop as codex, with `agent="claude"` (or `python $SKILL_ROOT/scripts/oob_ray_submit.py run -a claude -p "$PROMPT" -f kernel.py --max-turns 30 --no-live --json`). See [`../kernel-opt/claude.md`](../kernel-opt/claude.md) — note Claude's prompt should include the MANDATORY CONSTRAINTS block from `claude.md` (function signature / block size / output filename).

**IMPORTANT:** Do NOT run enabled OOB backends sequentially when both are available.
Launch all active OOB backends at the same time. The iterative loops within codex/claude
are internal to each backend — they do NOT block other backends.

Use prompt templates per kernel type:
- **RMSNorm / reduction kernels**: TRUE single-pass template (eliminate second loop, 2x memory reduction)
- **General dual-loop kernels**: Merge redundant memory loads
- **Template/GEMM kernels**: Low priority, Inductor autotuner already near-optimal

See each OOB backend reference for full prompt templates and CLI execution details.

#### Prompt rules — shared across all backends

These rules apply to **every** OOB kernel optimization submission.

1. **Kernel path — conditional on image availability:**
   - If the kernel source file **exists in the Docker image** (e.g., `/sgl-workspace/aiter/...`, `/opt/venv/...`), **MUST include** the kernel's absolute file path and repo path in the prompt. Example: `"The kernel source file is at /sgl-workspace/aiter/jit/core/compile.py"`, `"The kernel repo is at /sgl-workspace/aiter/"`.
   - If the kernel source is **runtime-generated** (e.g., `/tmp/torchinductor_root/...` from `torch.compile` Inductor cache), **DO NOT include** `kernel_url` or `kernel_repo` in the prompt. These files only exist in the running inference server's ephemeral storage, not in the Docker image. Instead, copy kernel files to a shared NFS path and reference the NFS path, OR omit these paths entirely and rely on `files[].content`.
   - **How to tell:** paths under `/tmp/`, `/root/.cache/`, or any `torchinductor_*` directory are runtime-generated. Paths under `/sgl-workspace/`, `/opt/`, `/usr/` are part of the image.
2. **1.5x minimum speedup target** — Always include: `"The kernel MUST be optimized to at least 1.5x speedup."` in the prompt.
3. **No broad filesystem searches** — Always say: `"Do NOT search the filesystem with find / or grep -r /"`.
4. **Embed full source in files** — OOB receives the kernel source via `-f`, `files[].content`, or inline prompt content.

#### Prompt rules — backend-specific

5. **Framework image** — OOB tasks run inside the RayJob image selected by `KERNEL_OPT_IMAGE` (provided by CI or user).
6. **Workspace** — Use `KERNEL_OPT_WORKSPACE` (default `"control-plane-moe"`) and pod-visible shared paths for OOB inputs/outputs.
7. **Explicit output filename** — Include: `"Write the COMPLETE optimized file to optimized_kernel.py."` OOB agents need an explicit output path.

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
| Total submissions > OOB budget from `OOB_TOP_CANDIDATES × enabled_backends × OOB_ROUND_ITERATIONS` | Stop — cost budget |

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
- Updated baseline with all kept patches applied

## Heuristic Update
- Each kept kernel: boost similar kernel type scores (other reduction kernels likely optimizable too)
- Each discarded kernel: reduce scores for that kernel type
- After 2+ discards on vendor-type kernels: reduce all kernel-opt scores to near-zero
- Re-profiled new candidates get fresh scores based on new gpu_pct

## Failure Handling
- OOB workspace unavailable: retry on a pod-visible shared workspace (3 attempts)
- Codex task fails: use Claude result if available, otherwise re-submit with stricter constraints
- Claude task fails: use Codex result if available, otherwise re-submit with stricter constraints
- All backends produce wrong signature: re-submit with explicit constraint
- Register OOM during Triton compile: reduce block sizes in prompt
- Server crash after patch: revert, log crash, skip kernel
