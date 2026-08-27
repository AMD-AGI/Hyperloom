---
title: layout_shuffle — tuning (layout choice, the bpreshuffle key, tuned bpreshuffle CSVs)
kind: technique
operator: layout_shuffle
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1, int8]
regimes: [prefill, decode, both]
updated: 2026-06-08
---

# layout_shuffle — tuning

## TL;DR
The shuffle is cheap (one-time); "tuning" means (a) picking the layout that matches the consuming MFMA kernel (asm/CK/FlyDSL each ask for a specific `(IN,IK)`/`pack_n`) and (b) making the `bpreshuffle` dispatch key + tuned CSV actually engage — a shuffled weight needs `bpreshuffle=True` in its tuned row or the fast kernel is silently missed.

The shuffle is cheap (one-time). "Tuning" means **(a) picking the layout that matches the consuming MFMA
kernel** and **(b) making the bpreshuffle dispatch + tuned CSV actually engage**. Getting the layout wrong
gives wrong numerics; getting the key wrong silently loses the fast kernel.

## 1. The shuffle math (what the layout means)
`shuffle_weight(x, layout=(IN, IK), use_int4)` on `[..., N, K]`:
```
BK   = IK * 2
Kel  = 16 // x.element_size()      # 8 for bf16/fp16, 16 for fp8/int8; 32 if use_int4
view → [-1, N/IN, IN, K/BK, BK/Kel, Kel]
permute(0,1,3,4,2,5) → groups (IN, Kel) last  → MFMA fragment order
```
So `IN` = N-lane width of the matrix-core fragment, `IK·2 = BK` = the K block, `Kel` = elements per lane.
- **`(16,16)`** — default; matches mfma_16x16-style fragments.
- **`(32,16)`** — a8w8 asm path (`shuffle_weight(weight, layout=(32,16))`).
- **FlyDSL** uses `_get_flydsl_shuffle_layout(pack_n)` → `(16·pack_n, 16)`.
- **FP4 MoE**: `shuffle_weight_a16w4(src, NLane, gate_up)` with `KPack=16`, `KLane=64//NLane`, and a matching
  `shuffle_scale_a16w4` for the block scale (MXFP4 1×32: `K_Pack=2, N_Pack=2, N_Lane=16`).

**Pick the layout the kernel asks for** — the asm/CK/FlyDSL kernel and the shuffle layout are a matched pair.
FlyDSL even prints the exact `shuffle_weight(b, layout=...)` you must use.

## 2. The bpreshuffle dispatch key (make it engage)
`shuffle_weight` sets `x_.is_shuffled = True`. In `aiter/tuned_gemm.py::gemm_a16w16`, `bpreshuffle` is read
from `B.is_shuffled` and becomes the **9th element of the dispatch key**:
`(cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)`. Consequences:
- The tuned CSV row must have **`bpreshuffle = True`** for a shuffled weight to hit a tuned bpreshuffle
  kernel. A row tuned with `bpreshuffle=False` won't match a shuffled call (and vice versa) — same
  miss-the-key failure mode as `bias` ([[backends/aiter/tuned_gemm.md]]).
- No-hit defaults with `bpreshuffle`: gfx942 → `hipblaslt solidx=-1` (heuristic); gfx950 bf16 with
  `N%64==K%64==0` → `asm`.

## 3. The tuned bpreshuffle CSVs (deploy seam)
aiter ships dedicated bpreshuffle config files ([[backends/aiter/configs_db.md]]):
| op | tuned CSV | env override |
|---|---|---|
| a8w8 bpreshuffle | `a8w8_bpreshuffle_tuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` |
| a8w8 blockscale+bpreshuffle | `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |
| a4w4 block-scale (FP4 BpreShuffle kernels) | `a4w4_blockscale_tuned_gemm.csv` | `AITER_CONFIG_GEMM_A4W4` |

Real shipped FP4 rows are gfx950 BpreShuffle kernels, e.g.
`_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E`. Deploy by env (`:`-mergeable), same merge semantics
as the bf16 GEMM DB.

## 4. Why it helps (the perf rationale)
The matrix core wants its operand fragment in a specific N-lane × K-pack order in LDS. Without pre-shuffle the
kernel must permute on load (extra `ds_*`/cross-lane ops, possible bank conflicts). Pre-shuffling means the
GEMM does a **conflict-free, vectorized `ds_*_b128`** read straight into the MFMA — it's the static-weight
version of the conflict-free LDS staging in [[operators/transpose/tuning.md]]. The cost is paid **once** at
load, amortized over every forward pass.

## 5. Alignment requirements
`shuffle_weight` asserts `N % IN == 0` and `K % BK == 0`; FP4 scale asserts `real_k ≥ 256` (Tile_K). A weight
whose dims don't divide must be padded (the same padding the GEMM kernel needs anyway).

## Verify
After shuffling: `grep 'is tuned on cu_num' server.log` shows the bpreshuffle row hit (with the right
`bpreshuffle=True` libtype). GEMM output `allclose` vs unshuffled-weight GEMM (value-preserving). rocprofv3 →
the bpreshuffle asm/CK/FP4 kernel fires (not the generic path).
