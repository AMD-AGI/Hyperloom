---
title: weight_shuffle — overview
kind: operator_overview
operator: weight_shuffle
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8, int8, int4, fp4]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/shuffle.py
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
  - ROCm/aiter@b467ce342:aiter/ops/moe_op.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/utils/shuffle.py
---

# weight_shuffle

## TL;DR
`shuffle_weight` (and friends) permutes a weight tensor **offline** into the MFMA/WMMA-native tile order the
tuned GEMM/MoE kernels expect ("bpreshuffle"). It reshapes `[…, N, K]` into per-instruction tiles
(default `layout=(16, 16)` → 16×16 MFMA), permutes so each wavefront lane's operand elements are
contiguous, and tags the result `is_shuffled = True` (`shuffle.py:204`). Kernels then issue wide contiguous
loads straight into MFMA/WMMA operand registers with no runtime transpose/swizzle. Consumers select a
preshuffle kernel variant from that tag: `tuned_gemm` sets `bpreshuffle = B.is_shuffled` (`tuned_gemm.py:353`)
and CK MoE reads `is_shuffled=getattr(w, "is_shuffled", False)` (`moe_op.py:574`, `:614`).

## What it is
CDNA MFMA (and gfx1250 WMMA) matrix instructions read their operands from a fixed lane×VGPR arrangement:
each of the 64 wavefront lanes owns a specific sub-block of the tile, and the K (contraction) elements a
lane needs must sit in adjacent registers. A plain row-major `[N, K]` weight does **not** land in that
arrangement, so a naive kernel would transpose/gather per tile at runtime. `shuffle_weight` pre-applies
that permutation on the host so the on-device load is a straight, coalesced copy into MFMA operand form —
the single reason tuned/asm GEMM and grouped-MoE kernels ship a dedicated `bpreshuffle` (preshuffled-B)
variant.

Default-layout mechanics (`shuffle.py:191-208`, `layout=(16, 16)`):
- `IN, IK = layout`; `BK = IK*2 = 32`; `BN = IN = 16`; `K = 16 // element_size` (bf16/fp16 → 8, fp8/int8 → 16;
  `use_int4` → 32) (`shuffle.py:191-193`).
- `x.view(-1, N/BN, BN, K_dim/BK, BK/K, K).permute(0, 1, 3, 4, 2, 5)` reorders each `BN×BK` block into
  lane/register order, then flattens back to the original shape (`shuffle.py:199-203`).
- The K-per-tile `BK` (= `2*IK`) packs two MFMA K-steps so a lane loads its full operand contiguously.

## Entry points (API)
| symbol | path:line | signature (from source) | purpose |
|---|---|---|---|
| `shuffle_weight` | `aiter/ops/shuffle.py:147` | `shuffle_weight(x, layout=(16, 16), use_int4=False, is_guinterleave=False, gate_up=False, pad_k_to=0)` | Main MFMA-native B preshuffle; sets `is_shuffled=True`. `is_guinterleave` path is the a16w4/gated packing (`:173-189`). |
| `shuffle_weight_NK` | `aiter/ops/shuffle.py:218` | `shuffle_weight_NK(x, inst_N, inst_K, use_int4=False)` | Same idea for an arbitrary MFMA instruction shape: `kPerLane = inst_K // (64 // inst_N)` (×2 for int4). |
| `shuffle_weight_a16w4` | `aiter/ops/shuffle.py:211` | `shuffle_weight_a16w4(src, NLane, gate_up)` | Back-compat wrapper → `shuffle_weight(..., layout=(NLane, 16), is_guinterleave=True, gate_up=gate_up)`. |
| `shuffle_scale` | `aiter/ops/shuffle.py:338` | `shuffle_scale(src, experts_cnt=None, is_guinterleave=False, gate_up=False)` | Shuffles microscale (e8m0/MXFP4) block-scales to match the shuffled weight tiles; fp32 scales pass through (`:346`). |
| `moe_shuffle_weight` | `aiter/ops/shuffle.py:122` | `moe_shuffle_weight(src, experts_cnt=None, is_guinterleave=False, gate_up=False, layout=(16, 16))` | Arch-aware MoE B shuffle (gfx1250 row-interleave + WMMA vs `shuffle_weight` lane interleave). |
| `shuffle_weight` (triton) | `aiter/ops/triton/utils/shuffle.py:41` | `shuffle_weight(x, layout=(16, 16), use_int4=False, is_guinterleave=False, gate_up=False, pad_k_to=0, arch=None)` | Arch dispatcher: gfx1250 WMMA TDM layout else the base shuffle. |

## Dispatch / backends
Two families, chosen by GPU arch (`get_gfx()` / `get_arch()`):
- **CDNA MFMA (gfx942/gfx950)** — the `shuffle.py` base `shuffle_weight` `(16, 16)` layout. `moe_shuffle_weight`
  routes non-gfx1250 archs here with lane-level gate/up interleave (`shuffle.py:142-144`; the comment names
  "other archs (e.g. MI355)" at `:134`).
- **gfx1250 WMMA** — `shuffle_weight_gfx1250` (`shuffle.py:78-113`) uses the `(N//16, 16, K//32, 2, 16)` →
  `(N//16, K*16)` TDM-optimal view; the arch-aware wrappers branch on `get_gfx() == "gfx1250"`
  (`shuffle.py:138`, `:419`). gfx1250 also has the mxfp8fp4 / F4 preshuffles (`shuffle_mxfp8fp4_*`,
  `shuffle_weight_f4`, `shuffle_scale_f4`) in the same file.

Downstream selection of the preshuffled kernel:
- `tuned_gemm.gemm_a16w16`: `bpreshuffle = True` iff `B.is_shuffled` (`tuned_gemm.py:353-355`), then
  `get_GEMM_A16W16_config(..., bpreshuffle=bpreshuffle)` picks the tuned solution (`:369-377`).
- CK MoE stages: `ck_moe_stage1_fwd`/`ck_moe_stage2_fwd` pass `is_shuffled=getattr(w, "is_shuffled", False)`
  into the module codegen (`moe_op.py:574`, `:614`).

## Config / knobs
- **`layout=(IN, IK)`** — MFMA/WMMA tile (default `(16, 16)`). Other consumers pass e.g. `(32, 16)`
  (docstring example, `gemm_op_a8w8.py:556`).
- **`use_int4`** — sets `K = 32` (int4-packed) instead of `16 // element_size`; forbidden with `pad_k_to`
  (`shuffle.py:163-164`).
- **`is_guinterleave` + `gate_up`** — gated stage1 packing (a16w4): interleaves the N-lane layout with
  gate/up; `gate_up` folds the 2× gate/up rows (`shuffle.py:173-189`). `is_guinterleave` requires
  `experts_cnt` for `shuffle_scale` (`shuffle.py:366-367`).
- **`pad_k_to`** — right-pad K to a multiple before shuffling; records `aiter_original_k` / `aiter_padded_k`
  on the result (`shuffle.py:159-171`, `:205-207`). Unsupported with `use_int4` / `is_guinterleave`.
- **Divisibility asserts** — `N % BN == 0` and `K % BK == 0` (`shuffle.py:195-196`); gfx1250 requires
  `N % 16 == 0`, `K % 32 == 0` (`shuffle.py:95-96`).

## Numerics / parity
Bit-exact reordering — no arithmetic. `fp4` (`torch.float4_e2m1fn_x2`) is viewed as `uint8` for the
permute and viewed back (`shuffle.py:156-157`, `:90-91`). `shuffle_scale`'s non-interleave path pads rows to
a multiple of 256 and cols to a multiple of 8 with a freshly allocated buffer (`shuffle.py:352-359`);
the guinterleave path pads short K with the neutral e8m0 byte `0x7F` (= scale 1.0) (`shuffle.py:375-379`).
Correctness therefore hinges on the shuffle layout **exactly** matching the consuming kernel's expected
tile order.

## Pitfalls
- **Layout must match the kernel** — a weight shuffled with the wrong `layout`/arch will silently produce
  garbage; the tag `is_shuffled` only signals *that* it was shuffled, not *which* layout.
- **`is_shuffled` is an attribute, not dtype** — it can be dropped by ops that rebuild the tensor (e.g. a
  `.view()`), which then re-selects the non-preshuffle kernel. Consumers use `getattr(..., "is_shuffled", False)`.
- **Scales must be shuffled too** — for MXFP4/e8m0 weights, shuffle the block-scales with the matching
  `shuffle_scale` / `moe_shuffle_scale` or the microscaling misaligns.
- **gfx1250 restrictions** — the triton arch dispatcher raises `NotImplementedError` for
  `use_int4`/`is_guinterleave`/`gate_up`/`pad_k_to` on gfx1250 (`triton/utils/shuffle.py:56-59`).

## Cross-links
- [../dense_gemm/aiter.md](../dense_gemm/aiter.md) — bpreshuffle B is a tuned dense-GEMM input.
- [../scaled_quant_gemm/overview.md](../scaled_quant_gemm/overview.md) — quantized GEMM consuming shuffled B + shuffled scales.
- [../quant_fp4_mxfp/aiter.md](../quant_fp4_mxfp/aiter.md) — MXFP4/e8m0 packing that pairs with `shuffle_scale`.
- [../fused_moe_grouped_gemm/aiter.md](../fused_moe_grouped_gemm/aiter.md) — CK MoE stages that read `is_shuffled`.
- [../grouped_gemm_moe/aiter.md](../grouped_gemm_moe/aiter.md) — grouped-GEMM MoE weight prep.

## Sources
- Weight shuffles: `aiter/ops/shuffle.py:147-208` (`shuffle_weight`), `:211-215` (`a16w4`), `:218-236` (`NK`), `:78-113` (`gfx1250`), `:122-144` (`moe_shuffle_weight`).
- Scale shuffles: `aiter/ops/shuffle.py:338-408` (`shuffle_scale`), `:411-430` (`moe_shuffle_scale`).
- Arch dispatch: `aiter/ops/shuffle.py:138`, `:419`; `aiter/ops/triton/utils/shuffle.py:41-69`.
- Consumers of `is_shuffled` / `bpreshuffle`: `aiter/tuned_gemm.py:353-377`; `aiter/ops/moe_op.py:574`, `:614`; docstring `aiter/ops/gemm_op_a8w8.py:556`.
