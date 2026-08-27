---
title: mla_attention on HIP — SOTA card
kind: sota_card
operator: mla_attention
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode, prefill]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends/mla
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py
---

# mla_attention × HIP

## TL;DR
There is **no standalone hand-written HIP MLA kernel** to recommend as the SOTA — vLLM's MLA backends
(`ROCM_AITER_MLA`, `ROCM_AITER_TRITON_MLA`, `TRITON_MLA`) all route to **aiter asm** or **Triton**, not to
a bespoke `csrc/rocm` MLA kernel. So for MLA, "HIP" means: the absorbed asm `mla_decode_fwd` is *itself*
the low-level (asm, not C++ HIP) kernel, and authoring a fresh HIP MLA kernel is an expert-only Tier-C
move. **Use aiter MLA** (aiter); reach for raw HIP/asm only to modify the absorbed decode
kernel for a hot fixed shape.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter asm `mla_decode_fwd` (the low-level path) | `ROCm/aiter@a6bb49937:aiter/mla.py` + asm blobs | gfx942/950; bf16/fp16/fp8 | the tuned ceiling (17× decode kernel) | use via aiter |
| Author a HIP MLA decode kernel | author via `hip_levers/` | gfx942/950 | shape-specific; only if you out-schedule the asm | hot fixed shape, expert |

## Config space / knobs
If authoring HIP: the latent (512) + rope (64) two-part score, MQA over the latent (absorbed), splitKV +
reduce, `matrix_instr_nonkdim=16`, `waves_per_eu` (memory-bound → 3-4), LDS layout for the 512-wide latent
(64 KB gfx942 / 160 KB gfx950), fp8 latent scale (fnuz). See `hip_levers/` and `languages/asm_mfma/`.

## Numerics / parity
fp32 two-part score accumulate; absorption exact in bf16; fp8 accuracy gate (MLA eval-sensitive). See
[numerics.md](numerics.md).

## Integration (rebind seam)
The asm path enters through aiter (`mla_decode_fwd`); a hand-authored HIP MLA kernel would need to be
wired into aiter's catalog or called directly and e2e-gated. Confirm engagement with `AITER_LOG_MORE=1`.

## Pitfalls & anti-patterns
- Don't reinvent the absorbed decode in HIP unless you can prove a measured win over the aiter asm path.
- Inline-asm MFMA defeats `SchedGroupMask` — use intrinsics + `sched_group_barrier`.
- 512-wide latent is LDS/register heavy — spilling collapses occupancy.
- fp8 MLA accuracy regressions — gate.

## How to verify
Disassemble (want `v_mfma_*16x16`, `buffer_load_dwordx4`, no `scratch_`); greedy temp=0 parity vs Triton
MLA reference; isolated decode bench vs aiter asm — the gate is a measured win that justifies the
maintenance.

## Alternatives / cross-links
aiter (use this) · triton (reference) · ck ·
`hip_levers/`, `languages/asm_mfma/` · ../overview.md.

## Sources
- vLLM MLA backends route to aiter asm / Triton (no bespoke HIP MLA): https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends/mla ; https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- aiter asm `mla_decode_fwd` is the low-level path: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py`.
