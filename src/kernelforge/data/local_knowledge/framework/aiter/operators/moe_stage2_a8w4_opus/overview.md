---
title: moe_stage2_a8w4_opus — overview
kind: operator_overview
operator: moe_stage2_a8w4_opus
gens: [gfx950]
dtypes: [fp8_e4m3, fp4_e2m1, bf16]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/opus/moe_stage2_a8w4.py
  - ROCm/aiter@b467ce342:aiter/ops/opus/moe_stage2_a8w4_meta.py
  - ROCm/aiter@b467ce342:aiter/ops/opus/moe_stage2_a8w4_fused_adapter.py
---

# moe_stage2_a8w4_opus

## TL;DR
The opus **MoE stage-2 (down-projection)** kernel family for **gfx950 (CDNA4)**: **FP8 activation × MXFP4 weight →
bf16** (or a routed per-slot output). It is the a8w4 stage-2 counterpart to the CK/asm stage-2 used by
`fused_moe`, selected by a per-shape kid. Two output strategies: a **direct-atomic** path that accumulates the
top-k contributions straight into the token output, and a **route-output** path that writes one row per
(token, slot) which a second **reduce** kernel folds down. Two shape families are wired: **K3** (`inter_dim=512`)
and **K5** (`inter_dim=768`). There is **no expert-parallel (EP)** support and it **requires `a2_scale` and
`w2_scale`**.

## What it is
Stage-2 of a two-stage fused MoE (stage-1 = gate/up, stage-2 = down). `inter_states` is the post-activation
intermediate `[token, topk, inter_dim]` in FP8; `w2` is the MXFP4 down weight; the op produces the model-dim
output. The kid encodes {out_mode, block_m, shape_family, route-reduce block_n}, so a single kernel name fully
selects the variant (`moe_stage2_a8w4_meta.py:20-36`). Out modes: `ATOMIC=0`, `BF16=1`, `FP8=2`
(`moe_stage2_a8w4_meta.py:16-18`).

## Entry points (API)
| symbol | path:line | signature (abridged) | purpose |
|---|---|---|---|
| `opus_moe_stage2_a8w4_decode_fwd` | `aiter/ops/opus/moe_stage2_a8w4.py:125` | `(inter_states, w2, a2_scale, w2_scale, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, *, block_m, inter_dim_pad, out=None, kernel_id=-1, return_per_slot=False, route_out_dtype=None) -> Tensor` | the stage-2 GEMM; auto-selects kid from shape family + `block_m` when `kernel_id=-1` |
| `opus_moe_stage2_reduce_token_slot_route_output_fwd` | `moe_stage2_a8w4.py:217` | `(route_out, out=None, *, topk=None, block_n=None) -> Tensor` | folds route-output rows (bf16 `[token,topk,md]` or uint8 MXFP8 `[token·topk, md+md/8]`) down to `[token, md]` bf16 |
| `_opus_moe_stage2_a8w4_decode_fwd_raw` | `moe_stage2_a8w4.py:95` | `@compile_ops("module_moe_opus", fc_name="opus_moe_stage2_a8w4_decode_fwd")` | JIT-bound decode kernel |
| `_opus_moe_stage2_reduce_token_slot_route_output_fwd_raw` | `moe_stage2_a8w4.py:117` | `@compile_ops("module_moe_opus", fc_name="opus_moe_stage2_reduce_token_slot_route_output_fwd")` | JIT-bound reduce kernel |
| `opus_a8w4_stage2_wrapper` | `moe_stage2_a8w4_fused_adapter.py:209` | fused-moe-facing adapter | validates shapes/gfx and drives decode(+reduce) from a tuned `kernelName2` |
| `cfg_is_supported` | `moe_stage2_a8w4_fused_adapter.py:134` | `(kernel_name, *, cfg, gfx, block_m, is_ep, has_stage2_bias=False) -> (bool, str)` | tuner/dispatch guard |

## Dispatch / backends
- **JIT module** `module_moe_opus` (opus in-tree kernels). The `_meta.py` module is intentionally torch-free and is
  the Python source of truth shared with the csrc tuner/codegen (`moe_stage2_a8w4_meta.py:3-8`).
- **Shape family** is resolved from `(model_dim=w2.shape[1], inter_dim=inter_states.shape[2], expert=w2.shape[0],
  topk)` via `opus_a8w4_shape_family_for_shape` (`meta.py:557`); if none matches and `kernel_id=-1`, auto-selection
  raises (`moe_stage2_a8w4.py:151`).
- **Kid selection** when `kernel_id=-1`:
  - route path (`return_per_slot=True`) → `opus_a8w4_decode_kid(out_mode, block_m, shape_family)` (`meta.py:613`),
    where out_mode comes from `route_out_dtype` (`fp8`/`mxfp8`/`uint8`→FP8, `bf16`→BF16, `moe_stage2_a8w4.py:35`).
  - direct-atomic path with `block_m==32` → `opus_a8w4_best_atomic_kid(token_num, shape_family)` (`meta.py:655`),
    which prefers block_m 32 then 16.
- **Route-reduce block_n** is carried by the kid name suffix `_rbn####` and mapped to a tuned
  `OpusA8W4RouteReduceInstance` (thread count etc.) — instances: `full_model_n7168` (bn=7168, 448 thr), `rbn2240`,
  `rbn2304`, `rbn2816`, `rbn3072`, `rbn3584` (`meta.py:211-248`).
- **fused_moe integration**: `opus_a8w4_stage2_wrapper` (adapter `:209`) is the seam the MoE dispatcher calls; it
  runs decode → (for route modes) reduce, validating `w2 == [E, model_dim, inter_dim//2]` (MXFP4 packed,
  `adapter:274-282`).

## Config / knobs
| knob | where | values |
|---|---|---|
| shape family | derived | **K3**: `logical_inter_dim=512`, `inter_dim_pad=128` (eff 384) (`meta.py:170`); **K5**: `768`, pad `128` (`meta.py:179`) |
| `block_m` | call arg / kid | supported `{16, 32, 64}` (`OPUS_A8W4_SUPPORTED_BLOCK_MS`, derived `meta.py:517`) |
| `out_mode` | kid | `ATOMIC=0` (direct accumulate) / `BF16=1` / `FP8=2` (route-output) (`meta.py:16-18`) |
| `return_per_slot` | call arg | request the route-output path (needs a route-output kid) |
| `route_out_dtype` | call arg | `fp8`/`mxfp8`/`uint8` or `bf16`; requires `return_per_slot=True` (`moe_stage2_a8w4.py:156`) |
| `inter_dim_pad` | call arg | must be `0` or the family's `inter_dim_pad` (`adapter:267`) |
| `block_n` (reduce) | reduce arg | `-1` = auto (`_OPUS_MOE_STAGE2_ROUTE_REDUCE_AUTO_BLOCK_N`, `moe_stage2_a8w4.py:24`) |
| kernel contract | fixed | `gfx950_a8w4_decode_v1`: `block_k=256`, MFMA `16×16×128`, scale group `32`, `fp4_values_per_byte=2` (`meta.py:188-204`) |

K3 kids include `2000` (atomic bm16×bn64), `2001-2004`/`2007` (route FP8 bm32/bm64 + rbn), `2005`/`2008` (route BF16
full-N7168), `2006` (atomic bm32×bn128); K5 kids `2100`/`2101` (atomic), `2110` (route BF16), `2111` (route FP8)
(`meta.py:24-36`, `339-492`). K5 is a generalized bring-up family, **not** in the current DSV4-tuned target set
(`meta.py:176-178`).

## Numerics / parity
- fp32 accumulate inside the MFMA; MXFP4 weight carries a 32-element shared exponent (`scale_group_logical_k=32`).
  Requires per-expert weight scale `w2_scale` and activation scale `a2_scale` — both mandatory (`adapter:243`).
- FP8 route-output packs each row as uint8 `[md fp8 | md/8 e8m0 scale]`; the reduce kernel infers fp8 from the
  uint8 dtype (no env flag) and validates `cols % 9 == 0` and `rows % topk == 0` (`moe_stage2_a8w4.py:184-247`).
- BF16 route-output is `[token, topk, md]` bf16, reduced to `[token, md]` bf16 (`moe_stage2_a8w4.py:267-308`).
- No in-repo perf numbers for this op; gate quant end-to-end (this is the lossy down-proj — same caution as fused
  MoE stage-2).

## Pitfalls
- **gfx950 only** — `cfg_is_supported` rejects any other `gfx` (`adapter:145`).
- **No EP** — `expert_mask`/`topk_ids` raise (`adapter:241`, `:147`); **no stage-2 bias** (`bias2` raises,
  `adapter:239`).
- `a2_scale` **and** `w2_scale` are required or the wrapper raises (`adapter:243`).
- `inter_states` must be 3-D `[token, topk, inter_dim]` (`adapter:245`); `w2` must be `[E, model_dim, inter_dim//2]`
  (MXFP4 packed) or shape validation fails (`adapter:274-282`).
- `route_out_dtype` without `return_per_slot=True` raises (`moe_stage2_a8w4.py:156`); auto route-out needs an
  explicit route-output kid at `block_m==64` (`adapter:167-171`).
- Auto direct-atomic only covers bm16/bm32; auto route-out only bm64 (`adapter:168-171`).

## Cross-links
- opus backend family (split-K GEMM + this MoE stage-2, workspace/JIT notes) →
  operators/dense_gemm/backends/opus.md.
- The consuming two-stage fused MoE path (stage-1 asm + stage-2 select, `moe_sorting`, tuned DB) →
  operators/fused_moe_grouped_gemm/aiter.md.
- MXFP4 weight packing / e8m0 block scales → operators/quant_fp4_mxfp/aiter.md; routing/sort upstream →
  operators/moe_routing_topk/aiter.md.

## Sources
- on-box `ROCm/aiter@b467ce342`: `aiter/ops/opus/moe_stage2_a8w4.py` (`opus_moe_stage2_a8w4_decode_fwd:125`,
  `opus_moe_stage2_reduce_token_slot_route_output_fwd:217`, raw ops `:95`/`:117`, fp8 route-out packing `:184-247`),
  `aiter/ops/opus/moe_stage2_a8w4_meta.py` (out modes `:16-18`, kids `:24-36`, K3/K5 contracts `:170-183`, gfx950
  kernel contract `:188-204`, route-reduce instances `:211-248`, kid selection `:557`/`:613`/`:655`),
  `aiter/ops/opus/moe_stage2_a8w4_fused_adapter.py` (`opus_a8w4_stage2_wrapper:209`, `cfg_is_supported:134`,
  shape/scale/EP/bias guards `:239-282`).
