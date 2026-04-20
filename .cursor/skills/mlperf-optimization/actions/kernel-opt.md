# Action: Kernel Optimization

Multi-round kernel optimization loop using configurable backends running in parallel.

Backend references:
- [`../kernel-opt/geak.md`](../kernel-opt/geak.md) — GEAK MCP (remote GPU pod)
- [`../kernel-opt/oob-codex.md`](../kernel-opt/oob-codex.md) — Codex via OOB Agent MCP
- [`../kernel-opt/oob-claude.md`](../kernel-opt/oob-claude.md) — Claude Code via OOB Agent MCP

## Inputs

- `kernel_candidates` from `profile.md` (kernel_name, gpu_pct, source_location)
- Current best config (kept_overrides + kept_patches)
- `baseline_ms_per_iter` (after prior DFS actions)

## KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B kernel optimization GEAK OOB" --top-k 5 --compact
python3 $SKILL_ROOT/kb/kb_query.py --category kernel_optimization --compact
```

## Eligibility Rules

| Kernel Type | Optimize? | Reason |
|-------------|-----------|--------|
| `Cijk_*` (hipBLASLt GEMM) | **No** | Vendor BLAS, hand-tuned MFMA |
| `aiter::fmha_v3_*` | **No** | Vendor attention, optimized for gfx950 |
| `triton_*` / `_permute_kernel` | **Yes** | Triton kernels have Python source |
| Custom HIP `__global__` | **Yes** | Primary optimization target |
| `cast_transpose` Triton | **Yes** | FP8 cast+transpose, can be optimized |
| NCCL kernels | **No** | Communication, not compute |

**Decision rule:** all non-vendor candidates from profile step, prioritized by GPU time + has modifiable source = optimization candidate. No hard GPU% threshold — `low_gpu_pct` flag is advisory context.

## Procedure

**FLOW GUARD:** Do NOT skip this action if candidates exist. Running sweep with
unoptimized kernels wastes compute.

### Step 1: Locate kernel source

**Primus/TransformerEngine Triton kernels:**
```bash
rg "@triton.jit" "$PRIMUS_ROOT/"
rg "def <kernel_name>" "$PRIMUS_ROOT/"
python3 -c "import primus_turbo; import os; print(os.path.dirname(primus_turbo.__file__))"
```

**torch.compile generated Triton:**
```bash
ls /tmp/torchinductor_*/*/triton/*.py | head -20
```

**Custom HIP kernels:**
```bash
rg "void <kernel_name>" "$PRIMUS_ROOT/" --glob "*.{hip,cu,cuh}"
```

### Step 2: Submit candidates to active backends in parallel

`KERNEL_OPT_BACKENDS` (default `geak,oob-claude,oob-codex`) controls which backends
run. User can override in prompt (e.g., `"Use only geak"`, `"Use geak,oob-codex"`).
All active backends run **simultaneously** for every candidate kernel.

| Backend | MCP | Full round latency | GPU on pod | Reference |
|---------|-----|--------------------|------------|-----------|
| `geak` | GEAK (`geak_create_task`) | 10–30 min | Yes | [`../kernel-opt/geak.md`](../kernel-opt/geak.md) |
| `oob-claude` | OOB Agent (`agent_create_task(agent="claude")`) | 3–15 min (3 iters) | No | [`../kernel-opt/oob-claude.md`](../kernel-opt/oob-claude.md) |
| `oob-codex` | OOB Agent (`agent_create_task(agent="codex")`) | 2–6 min (3 iters) | No | [`../kernel-opt/oob-codex.md`](../kernel-opt/oob-codex.md) |

**For each candidate kernel, launch all active backends CONCURRENTLY (not sequentially):**

```
                     ┌─ geak:       single task → poll → done
                     │
candidate kernel ────┼─ oob-claude: iterative loop (3 iters, each: submit→benchmark→feedback)
                     │
                     └─ oob-codex:  iterative loop (3 iters, same as oob-claude)

All 3 branches run in parallel. Each branch is independent.
Wait for ALL branches to finish. Collect best result from each.
Pick the one with highest verified speedup → Step 3.
```

**Per-backend execution:**

1. **`geak`**: `geak_create_task` + `geak_submit_task` → poll until done (single submission, GEAK verifies on-pod)
2. **`oob-claude`**: iterative refinement loop — `OOB_ROUND_ITERATIONS` (3) iterations, each: submit → download → **local benchmark** → feed result back as context. Take best speedup from all iterations. See [`../kernel-opt/oob-claude.md`](../kernel-opt/oob-claude.md).
3. **`oob-codex`**: same iterative refinement loop as oob-claude, with `agent="codex"`. See [`../kernel-opt/oob-codex.md`](../kernel-opt/oob-codex.md).

**IMPORTANT:** Do NOT run backends sequentially (geak first, then claude, then codex...).
Launch all active backends at the same time. The iterative loops within oob-claude/oob-codex
are internal to each backend — they do NOT block other backends.

Use prompt templates per kernel type:
- **RMSNorm / reduction kernels**: TRUE single-pass template (eliminate second loop, 2x memory reduction)
- **General dual-loop kernels**: Merge redundant memory loads
- **Template/GEMM kernels**: Low priority, vendor autotuner already near-optimal

See each backend reference for full prompt templates and MCP tool details.

#### Prompt rules — shared across all backends

1. **Kernel path — conditional on image availability:**
   - If the kernel source file **exists in the Docker image** (e.g., `/workspace/Primus/...`, `/opt/...`), **MUST include** the kernel's absolute file path in the prompt.
   - If the kernel source is **runtime-generated** (e.g., `/tmp/torchinductor_root/...`), **DO NOT include** `kernel_url` or `kernel_repo`. Rely on `files[].content` only.
2. **1.5x minimum speedup target** — Always include: `"The kernel MUST be optimized to at least 1.5x speedup."`
3. **No broad filesystem searches** — Always say: `"Do NOT search the filesystem with find / or grep -r /"`
4. **Embed full source in files** — All backends receive the kernel source via `files[].content`.

#### Prompt rules — backend-specific

5. **`geak` only: mode and max_rounds** — Include: `"Use homogeneous mode. Set max_rounds to 1."` See [`../kernel-opt/geak.md`](../kernel-opt/geak.md).
6. **`geak` only: framework image** — Use `GEAK_IMAGE` (Primus training image). See [`../kernel-opt/geak.md`](../kernel-opt/geak.md).
7. **`oob-codex` / `oob-claude` only: explicit output filename** — Include: `"Write the COMPLETE optimized file to optimized_kernel.py."`

### Step 3: Verify + Patch each result individually

For each winning backend output:
1. Verify function name + signature matches original exactly
2. If mismatch: re-submit with stricter prompt (max 3 attempts)
3. Patch source file with backup (see integration paths in each backend reference)
4. Clear Python `__pycache__` and Triton binary cache

### Step 4: Decide per kernel

| Outcome | Action |
|---------|--------|
| Best speedup > 0 | Dispatch to [`actions/integrate.md`](integrate.md) for full training benchmark |
| All backends produce ≤ 0 speedup | **DISCARD**. Log as discard. |
| All backends crashed | **DISCARD**. Log as crash. |

### Step 5: Re-profile after kept optimizations

Kernel rankings shift after optimization. Re-profile to find new bottlenecks.

### Stopping criteria

| Condition | Action |
|-----------|--------|
| All candidates from profile step processed | Stop |
| Cumulative E2E gain > 15% | Stop — excellent |
| 5 consecutive discards | Stop — diminishing returns |
| 2+ crashes during patching | Stop — environment unstable |
| Wall clock > 120 min | Stop — time budget |
| Total submissions > `GEAK_MAX_SUBMISSIONS` | Stop — cost budget |

## Outputs

- Per-kernel results: (kernel_name, speedup, e2e_gain, status, winning_backend)
- `cumulative_gain_pct`: total improvement from all kept kernels
- Updated baseline with all kept patches applied

## Heuristic Update

- Each kept kernel: boost similar kernel type scores by 1.3x, push re-profile (Rule #5)
- Each discarded kernel: reduce remaining kernel scores by 0.7x (Rule #6)
- After 2+ discards on vendor-type kernels: reduce all kernel-opt scores by 0.5x (floor at 0.5)

## Failure Handling

- GEAK workspace unavailable: retry on alternate workspace (up to `GEAK_MAX_RETRIES`)
- OOB-Claude task fails: other parallel backends (GEAK, Codex) continue independently
- OOB-Codex task fails: other parallel backends (GEAK, Claude) continue independently
- All backends produce wrong signature: re-submit with explicit constraint (max 3 attempts)
- Training crash after patch: revert, log crash, skip kernel
- All backends unreachable: skip entire action, log to KB
