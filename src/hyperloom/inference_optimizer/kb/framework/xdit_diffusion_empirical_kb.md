# xDiT Diffusion Model Optimization — Empirical KB

**Synthesized:** 2026-05-20  
**Sources:** 1 session (FLUX.2-dev BF16, 8× MI355X, session 20260520-220555)  
**Framework:** xDiT (Ulysses Sequence Parallelism for diffusion transformers)  
**Hardware:** AMD MI355X (gfx950, 256 CUs, 309 GB HBM3e)  
**Model:** FLUX.2-dev (BF16, 56 transformer blocks, 48 heads, 3072 hidden dim)

---

## Quick Reference: Best Configuration for FLUX.2-dev BF16 on MI355X

### Env Vars
```bash
AMDGCN_USE_BUFFER_OPS=1     # +0.29% (marginal, directionally correct)
```

### Required Code Patches (xDiT bugs — fix before any optimization)
See recipe `skills/kb/recipes/flux2-dev-bf16_mi355x.json` for detailed patch descriptions.

1. **`runner_models/flux.py`**: Add `_compile_model()` override with `reduce-overhead` mode
2. **`transformer_flux2.py`**: Add `combine_qkv_a2a=True` to `USP()` in dual-stream processors
3. **`usp.py`**: Use `.reshape()` instead of `.contiguous().view()` in combined QKV path

**Net gain from patches: +3.44%**

---

## 1. Architecture Context for FLUX.2-dev + Ulysses SP

### Compute Profile at 1024×1024, Ulysses=8

| Component | Fraction of Total FLOPS | Notes |
|---|---|---|
| GEMM (QKV + MLP projections) | ~85% | M=576, BF16, hipBLASLt |
| AllToAll communication (A2A) | ~10% | 2.53MB per-pair, XGMI |
| Attention compute | **~2%** | 576 tok/GPU × 6 heads/GPU |
| Norm/activation (Triton) | ~3% | RMSNorm, GELU, SiLU |

**Critical implication:** Attention backend optimizations are ineffective. With only 576 tokens/GPU and 6 heads/GPU (after 8-way Ulysses split), attention is ~2% of FLOPS. Any quantization overhead in approximate attention exceeds the compute savings.

### FLUX.2 Block Structure
- **8 dual-stream blocks** (`transformer_blocks`): separate Q, K, V, enc_Q, enc_K, enc_V projections (6 separate GEMMs per block)
- **48 single-stream blocks** (`single_transformer_blocks`): fused `to_qkv_mlp_proj` (1 GEMM for QKV+MLP) — already optimal
- **4608 tokens total** at 1024×1024 → **576 tokens/GPU** after SP split

### AllToAll Communication Pattern
- Per dual-stream block (post-patch): 2 A2A ops (combined QKV input + output)
- Per single-stream block: 2 A2A ops (combined QKV input + output via to_qkv_mlp_proj)
- Per-pair message size: **2.53MB** → RCCL's native algorithm is optimal at this size
- MSCCL routing overhead > MSCCL savings for 2.53MB at 8 GPUs

---

## 2. xDiT-Specific Bugs Found in FLUX.2 Implementation

### Bug 1: Missing `combine_qkv_a2a=True` in dual-stream blocks

**Symptom:** 4 AllToAll collectives per dual-stream block instead of 2  
**Root cause:** `xFuserFlux2AttnProcessor` called `USP(q, k, v)` without `combine_qkv_a2a=True`  
**Fix:**
```python
# Before (4 A2A per block):
hidden_states = USP(query, key, value)
# After (2 A2A per block):
hidden_states = USP(query, key, value, combine_qkv_a2a=True)
```
**Note:** The single-stream `xFuserFlux2ParallelSelfAttnProcessor` already had this correct — only dual-stream was wrong.

### Bug 2: Wrong `torch.compile` mode for FLUX.2

**Symptom:** FLUX.2 using `mode="default"` (no cudagraphs) while FLUX.1 used `reduce-overhead`  
**Root cause:** `xFuserFlux2Model` had no `_compile_model()` override; base class used default mode  
**Fix:**
```python
def _compile_model(self, input_args: dict) -> None:
    torch._inductor.config.reorder_for_compute_comm_overlap = True
    self.pipe.transformer = torch.compile(
        self.pipe.transformer,
        mode="reduce-overhead",
        dynamic=False,
    )
    input_args["num_inference_steps"] = 2
    self._run_timed_pipe(input_args)
```
**Impact:** `reduce-overhead` enables cudagraph capture, eliminating Python kernel dispatch overhead across 28 denoising steps × 56 blocks × many kernels.

### Bug 3: Missing `set_input_parameters()` in `_run_pipe()`

**Symptom:** USP runtime state not initialized; may cause incorrect Ulysses split behavior  
**Fix:** Add at start of `_run_pipe()`:
```python
batch_size = self.config.batch_size if self.config.batch_size else 1
get_runtime_state().set_input_parameters(
    batch_size=batch_size,
    num_inference_steps=input_args["num_inference_steps"],
    max_condition_sequence_length=input_args["max_sequence_length"],
    split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
)
```

### Bug 4: Unnecessary `.contiguous()` before reshape in combined QKV path

**Symptom:** Extra buffer allocation in `usp.py:_combined_qkv_all_to_all()`  
**Fix:**
```python
# Before:
qkv = qkv.permute(1, 2, 3, 0, 4, 5).contiguous().view(3, b, h // world_size, -1, d)
# After:
qkv = qkv.permute(1, 2, 3, 0, 4, 5).reshape(3, b, h // world_size, -1, d)
```

### Bug 5: `.to(query.dtype)` no-ops after attention

**Root cause:** AITER BF16 attention returns BF16 by contract; the cast is always a no-op  
**Fix:** Remove `.to(query.dtype)` after `hidden_states.flatten(2, 3)` in both processor classes  
**Note:** Confirmed via AITER source code review. Removing is a correctness improvement and code cleanup.

---

## 3. What Does NOT Work for xDiT Diffusion Models

### 3.1 Attention Backend Experiments — AVOID for Ulysses≥4

All tested backends regressed. Root cause: attention is <5% of FLOPS when tokens/GPU is small.

| Backend | Delta | Reason |
|---|---|---|
| `XDIT_ATTENTION_BACKEND=aiter_fp8` | -7.0% | FP8 quant overhead at M=576 |
| `XDIT_ATTENTION_BACKEND=aiter_sage` | -8.6% | SAGE approximation overhead |
| `XDIT_ATTENTION_BACKEND=aiter_sage_v2` | -8.66% | MXFP4 quant overhead |

**Rule:** Never test attention backends for diffusion models when `tokens/GPU < 2000`.

### 3.2 Communication Collective Tuning — RCCL Native is Optimal

| Change | Delta | Reason |
|---|---|---|
| `RCCL_MSCCL_ENABLE=1 + NCCL_MAX_NCHANNELS=32` | -13.0% | MSCCL overhead > savings for 2.53MB per-pair |
| Async A2A (USP_deferred closure) | 0.0% | torch.compile graph break at closure boundary; GELU window ~1µs vs A2A ~50-100µs |

**Rule:** For 2.53MB per-pair AllToAll at 8 GPUs over XGMI, RCCL native is optimal. MSCCL breakeven requires larger messages (>10MB per-pair or more GPUs).

### 3.3 GEMM Backend Changes — hipBLASLt Already Optimal

| Change | Delta | Reason |
|---|---|---|
| `torch.backends.cuda.preferred_blas_library("ck")` | CRASH | CK has no device_gemm for FLUX.2 GEMM shapes (M=576, non-standard N) |
| `PYTORCH_TUNABLEOP_TUNING=1` + torch.compile | CRASH | GPU memory fault: writes to read-only cudagraph memory |
| `coordinate_descent_tuning=True` | 0.0% | Triton autotune irrelevant (AITER/hipBLASLt handle >95% of compute) |

**Rule:** For M=576 BF16, hipBLASLt heuristic selects the optimal kernel. CK BLAS is UNSAFE for FLUX.2.

### 3.4 CPU Dispatch Optimizations — Ineffective with Cudagraphs

| Change | Delta | Reason |
|---|---|---|
| `HIP_FORCE_DEV_KERNARG=1` | -0.29% | cudagraphs already eliminate per-kernel CPU dispatch overhead |
| `OMP_NUM_THREADS=1` | 0.0% | Fully GPU-bound |
| `AMD_DIRECT_DISPATCH=1` | (not tested alone) | CRASHES if combined with AMDGCN_USE_BUFFER_OPS=1 |

**Rule:** With `torch.compile(reduce-overhead)`, CPU-side kernel dispatch optimizations provide no benefit and can regress.

---

## 4. Remaining Opportunities (Not Yet Tested)

### 4.1 Fused QKV Projection for Dual-Stream Blocks (Estimated +2-5%)

The 8 dual-stream `transformer_blocks` each have 6 separate linear projections:
- `to_q`, `to_k`, `to_v` (for image tokens)
- `add_q_proj`, `add_k_proj`, `add_v_proj` (for text encoder tokens)

These can be fused into a single GEMM with concatenated weight matrix. Implementation requires modifying `diffusers.models.transformers.transformer_flux2.Flux2Attention._get_qkv_projections()`. The single-stream blocks already do this via `to_qkv_mlp_proj`.

Expected impact: reduces 6 GEMMs → 1 per dual-stream block × 8 blocks × 28 steps = 1344 GEMM launches eliminated. At M=576, each GEMM is fast so launch overhead matters.

### 4.2 hipBLASLt Epilog Fusion for GELU in Single-Stream Blocks (Estimated +1-3%)

The 48 single-stream blocks apply `attn.mlp_act_fn(mlp_hidden_states)` (GELU) as a separate kernel after the combined projection GEMM. hipBLASLt's epilog API can fuse this activation into the projection GEMM itself.

### 4.3 torch.compile(mode="max-autotune") (Uncertain, Risky)

Enables exhaustive Triton kernel search for norm/activation ops. Risk: may pick Triton GEMM over hipBLASLt for M=576 (Triton is ~20% slower). Only test if norm/activation ops are confirmed >10% of runtime via profiling.

---

## 5. Quality Gate Notes

FLUX.2-dev BF16 quality thresholds (tight, same-precision):
- LPIPS < 0.05
- SSIM > 0.95
- MSE < 0.002

All code patches kept in this session passed quality gate. All attention backends and communication changes also passed quality gate but were reverted for throughput regression.

The quality gate format in bench output: `quality_gate_passed: true/false`. Check both this field AND throughput vs baseline.

---

## 6. Compatibility Notes

| Feature | Compatible with torch.compile(reduce-overhead)? |
|---|---|
| AITER attention backends | Yes |
| combine_qkv_a2a=True | Yes |
| AMDGCN_USE_BUFFER_OPS=1 | Yes |
| PYTORCH_TUNABLEOP_TUNING=1 | **NO — GPU memory fault** |
| CK BLAS backend | **NO — device_gemm crash** |
| USP_deferred async A2A | **NO — graph break at closure** |
| coordinate_descent_tuning | Yes (but zero gain) |
| HIP_FORCE_DEV_KERNARG=1 | Yes (but slight regression) |

---

## 7. Framework Agent Integration (FRAMEWORK Phase)

xDiT is fully supported by the Hyperloom framework agent's PR discovery and
apply pipeline. Unlike serving frameworks (sglang/vllm), xDiT is a
**scriptable** (server-less) workload with the following characteristics:

### Source Layout & Patch Mechanics

| Property | Value |
|---|---|
| Source root | `/app/xDiT/` (editable install via `pip install -e .`) |
| Python package | `xfuser` (importable; `xfuser/__init__.py` in source root) |
| Git origin | `https://github.com/xdit-project/xDiT.git` |
| Reinstall after patch? | **No** — pure-Python editable install; `git apply` takes immediate effect |
| Server restart needed? | **No** — scriptable workload (each bench is a fresh process) |

### How PR Discovery Works for xDiT

1. `fa phase-discover` searches xDiT's GitHub repo for open PRs matching
   the performance gaps (parallelism improvements, communication optimizations,
   compile mode changes, etc.).
2. The Critic gate evaluates candidate relevance.
3. `FrameworkAgentExecutor` fetches the unified diff (via checkout-head worktree
   or `diff_url`), applies it to `/app/xDiT/` with `git apply`, and runs a
   Magpie benchmark against the patched source.
4. **KEEP** (delta >= threshold): commits the patch to the live tree.
   **REVERT** (regression / no gain): `git reset --hard` back to pre-apply HEAD.

### CLI Usage

To enable framework agent PR discovery for xDiT sessions, omit `--no-framework-agent`:

```bash
python -m hyperloom.inference_optimizer.cli optimize \
    --framework xdit \
    --model /wekafs/models/FLUX.1-dev \
    --gpu-type mi300x \
    --tp 8 \
    --precision bf16 \
    --no-kernel \
    --max-hours 2
```

The `--no-framework-agent` flag disables the FRAMEWORK_AGENT phase entirely. Only use it
when the framework agent binary (`fa`) is unavailable or xDiT upstream PR
exploration is not desired.

### Relevant Search Keywords for xDiT PRs

The framework agent uses these keywords when searching for xDiT optimization PRs:

- `USP`, `Ulysses`, `sequence parallelism`
- `AllToAll`, `all_to_all`, `communication overlap`
- `torch.compile`, `reduce-overhead`, `cudagraph`
- `FLUX`, `diffusion`, `transformer`
- `pipeline parallel`, `cfg parallel`
- `performance`, `throughput`, `latency`
