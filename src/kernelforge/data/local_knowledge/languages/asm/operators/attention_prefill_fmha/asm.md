---
title: attention_prefill_fmha on ASM/MFMA — SOTA card
kind: sota_card
operator: attention_prefill_fmha
backend: asm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/ROCm/aiter
  - https://arxiv.org/abs/2511.08083
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py
---

# attention_prefill_fmha × ASM/MFMA

## TL;DR (one-line decision)
> Hand-written MFMA assembly is the **peak** prefill attention — the ceiling aiter ships for its hottest
> MHA/MLA paths and that everything else (CK-Tile, TileLang, Triton) is measured against. Reach for it
> only for the last 10–20% over CK-Tile on a *hot, fixed* shape, or to diagnose why a higher-level FMHA
> underperforms. For almost all work use **aiter** (which *is* the asm path) or CK-Tile; authoring raw asm
> attention by hand is expert-only and brittle.

## SOTA implementation(s)
| impl | source | gens / dtypes | measured perf (`value @ hw, lib, date`) | when it's best |
|---|---|---|---|---|
| aiter asm MHA/MLA kernels | `ROCm/aiter` (`hsa/` HSACO blobs + `csrc` launchers) | gfx942/950; bf16/fp16/fp8 | the peak path under `aiter.flash_attn_func` / `mla_decode_fwd` — source of the **1.2–4.4× TPS vs generic FA** and **17× MLA-decode** vendor numbers (MI300X, 2025–2026) | hot fixed shapes at scale |
| MFMA-intrinsic FMHA (`__builtin_amdgcn_mfma_*`) | author via HIP/CK | gfx942/950 | scheduler-friendly; near-asm with `sched_group_barrier` | when you must author + want compiler RA |

**Real call site** — aiter's Python wrappers thunk straight into asm kernels (`aiter/mla.py`):
```python
aiter.mla_decode_stage1_asm_fwd(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, ...)  # stage-1 asm
aiter.mla_prefill_asm_fwd(...)        # prefill asm
aiter.mla_prefill_ps_asm_fwd(...)     # persistent-scheduler prefill asm (tile_q = 256)
torch.ops.aiter.paged_attention_rocm(...)  # paged decode asm (csrc launcher)
```
These `*_asm_fwd` ops resolve to pre-built HSACO in `hsa/` — the asm *is* the product; the Python layer
only picks the variant and lays out scratch buffers.

## Config space / knobs (hand-scheduling, not a runtime table)
| lever | choice | effect |
|---|---|---|
| MFMA shape | **16×16×16** ≫ 32×32×8 (fp8 16×16×32) | 16×16 clocks higher → higher achievable FLOPs, lower power |
| VGPR/AGPR split | 256/256 (one wave/SIMD) | max register budget; spill = occupancy collapse |
| LDS layout | XOR-swizzle Q/K/V tiles | avoid bank conflicts on `ds_read_b128` |
| LDS budget | **64 KB/CU MI300X, 160 KB/CU MI350X** | tile-size ceiling; MI350X fits bigger tiles |
| `s_waitcnt vmcnt/lgkmcnt` | place before dependent `v_mfma` | overlap the two GEMMs with softmax |
| `s_setprio`, `sched_group_barrier`/IGLP | interleave VMEM/MFMA/VALU | hide latency |
| CDNA4 `q_waitcnt` | async load queue (gfx950) | decouple global loads |

See `languages/asm_mfma/`.

## Numerics / parity
fp32 accumulate in the MFMA pipe; identical math to FA-2 but a different reduction order than CK/Triton →
bf16 tie-flips benign. fp8 uses fp8 MFMA + per-tile scale — **FNUZ on gfx942, OCP e4m3 on gfx950**;
feeding the wrong dialect mis-scales ~2× (silent). See [numerics.md](numerics.md).

## Integration (rebind seam)
Not authored at the serving call site — it enters through **aiter** (`flash_attn_func`, paged/MLA decode).
To *use* the asm path you select aiter (`--attention-backend aiter`); to *replace* it you ship a new asm
kernel into aiter's catalog or load an HSACO. **Verify it engaged:** `AITER_LOG_MORE=1` (asm vs Triton
fallback); rocprofv3 shows the asm kernel name, not a Triton `_attn_fwd_*`.

## Pitfalls & anti-patterns
- Hand-written MFMA in **inline `asm volatile`** is not recognized by `SchedGroupMask` → defeats the SW
  pipeliner; use intrinsics + `sched_group_barrier` to *guide* the compiler instead.
- 32×32 MFMA clocks lower than 16×16 (power) → 16×16×16 yields higher achievable FLOPs.
- Spilling past the 256 VGPR / 256 AGPR budget collapses occupancy — the #1 cause of an asm FMHA
  underperforming.
- Sizing tiles for MI350X's 160 KB LDS then running on MI300X (64 KB) won't launch — gen-specific.
- Maintenance cost is high; only worth it for a hot shape at scale (why aiter, not individual users,
  owns these).

## Worked example
Diagnosing a slow DeepSeek MLA decode at scale:
1. `AITER_LOG_MORE=1` confirms `mla_decode_stage1_asm_fwd` fired (not Triton MLA).
2. `llvm-objdump` the HSACO → want `v_mfma_*16x16`, `buffer_load_dwordx4`, `ds_read_b128`,
   `s_waitcnt lgkmcnt(1)` before `v_mfma`, no `scratch_`/`v_accvgpr` spam.
3. If you see `scratch_`, the kernel spilled past 256 VGPR — that's the regression. The fix lives in
   aiter (file an issue / tune the variant), not at the call site.

## How to verify (bench + oracle)
Disassemble (`amdclang++ --offload-device-only -S` / `llvm-objdump`): want `v_mfma_*16x16`,
`buffer_load_dwordx4`, `ds_read_b128`, no `scratch_`. `amd_matrix_instruction_calculator` for A/B/C/D
register layouts. Isolated bench vs CK-Tile at the fixed shape; gate = *measured* win over CK that
justifies the maintenance, AND greedy temp=0 parity.

## Alternatives / cross-links
./ck.md (use first) · ./aiter.md (ships asm) · ./triton.md · ../../mla_attention/backends/aiter.md ·
`languages/asm_mfma/overview.md`, `mfma_intrinsics.md`, `pitfalls.md` · `languages/hipkittens/` ·
`hardware/mi300x.md`, `hardware/mi350x.md` · ../overview.md.

## Sources
- aiter ships raw asm for the fastest paths — on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py` (`mla_decode_stage1_asm_fwd`, `mla_prefill_asm_fwd`, `mla_prefill_ps_asm_fwd`) + `hsa/` HSACO blobs.
- HipKittens (peak AMD kernels are raw asm; 256/256 VGPR/AGPR; CDNA4 q_waitcnt): https://arxiv.org/abs/2511.08083 ; https://github.com/ROCm/aiter
- MFMA shapes / waitcnt / 16×16 vs 32×32 / LDS per CU: CDNA3 ISA Ch.7 https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf ; `languages/asm_mfma/overview.md`
