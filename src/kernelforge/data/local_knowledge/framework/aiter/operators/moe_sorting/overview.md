---
title: moe_sorting — overview
kind: operator_overview
operator: moe_sorting
gens: [gfx942, gfx950]
dtypes: [int32, fp32, bf16]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/moe_sorting.py
  - ROCm/aiter@b467ce342:aiter/ops/moe_sorting_opus.py
  - ROCm/aiter@b467ce342:aiter/fused_moe.py
  - ROCm/aiter@b467ce342:csrc/py_itfs_ck/moe_sorting_kernels.cu
  - ROCm/aiter@b467ce342:csrc/py_itfs_cu/moe_sorting_opus_kernels.cu
  - ROCm/aiter@b467ce342:csrc/include/moe_sorting_opus.h
  - ROCm/aiter@b467ce342:csrc/kernels/moe_align_block_size_kernels.cu
---

# moe_sorting

## TL;DR
`moe_sorting` turns per-token top-k routing (`topk_ids`, `topk_weights`) into the expert-contiguous,
block-padded layout the grouped-MoE GEMM consumes: `sorted_token_ids`, `sorted_weights`,
`sorted_expert_ids`, a `num_valid_ids` counter, and a zero-initialized `moe_buf` output. `aiter.fused_moe`
picks one of four backends behind the single `moe_sorting(...)` Python dispatcher — CK, Opus (default),
FlyDSL, or the adaptive `mxfp4_moe_sort` — with priority `CK > FlyDSL > Opus` (`fused_moe.py:57`).

## What it is
Fused MoE runs each expert as a tile of a grouped GEMM. Tokens must first be gathered so that all tokens
routed to expert `e` are contiguous and each expert's run is padded up to a multiple of `unit_size`
(= `block_size`, default `BLOCK_SIZE_M = 32`, `fused_moe.py:55`). `moe_sorting` does that gather in one
launch and additionally:
- writes `sorted_weights` = the `topk_weights` gathered in sorted order (fp32; the CK binding asserts fp32,
  `moe_sorting_kernels.cu:24`),
- writes `sorted_expert_ids` = the expert id per padded block,
- writes `num_valid_ids` = post-pad token count,
- zero-inits `moe_buf` (`[M, model_dim]`) so stage-2 can atomic-add into it.

This is a superset of `moe_align_block_size` (which produces only the id/expert/count arrays and no
weight-gather or output zeroing) — see [../moe_align_block_size/overview.md](../moe_align_block_size/overview.md).

## Entry points (API)
| symbol | path:line | signature (from source) | purpose |
|---|---|---|---|
| `moe_sorting` | `aiter/fused_moe.py:335` | `moe_sorting(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, block_size=BLOCK_SIZE_M, expert_mask=None, num_local_tokens=None, dispatch_policy=0, return_local_topk_ids=False, accumulate=True, flat=False, output_aux=False)` | Backend-selecting dispatcher; returns `(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf[, aux…])`. |
| `moe_sorting_fwd` | `aiter/ops/moe_sorting.py:11` | `moe_sorting_fwd(topk_ids, topk_weights, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf, num_experts, unit_size, local_expert_mask=None, num_local_tokens=None, dispatch_policy=0) -> None` | CK backend (`@compile_ops("module_moe_sorting")`); in-place writes. |
| `moe_sorting_opus_fwd` | `aiter/ops/moe_sorting_opus.py:20` | `moe_sorting_opus_fwd(topk_ids, topk_weights, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf, num_experts, unit_size, local_expert_mask=None, num_local_tokens=None, workspace=None, dispatch_policy=0, local_topk_ids=None, m_indices=None, reverse_sorted=None) -> None` | Opus backend (`develop=True`); default path. Adds `local_topk_ids` / `m_indices` / `reverse_sorted` aux outputs. |
| `moe_sorting_opus_get_workspace_size` | `aiter/ops/moe_sorting_opus.py:11` | `moe_sorting_opus_get_workspace_size(tokens, num_experts, topk, dispatch_policy=0) -> int` | Sizes the Opus scratch buffer (0 for the one-shot kernel). |
| `moe_align_block_size` (HIP) | `csrc/kernels/moe_align_block_size_kernels.cu:134` | `moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad)` | Related vLLM-derived align kernel (no weight gather); Python decl `aiter/ops/moe_op.py:49`. |

## Dispatch / backends
Selection is in `moe_sorting` (`fused_moe.py:335-400`) driven by env flags declared at `fused_moe.py:57-68`:
- `AITER_USE_CK_MOE_SORTING` (`fused_moe.py:59`), `AITER_USE_FLYDSL_MOE_SORTING` (`fused_moe.py:60`),
  `AITER_MOE_SORT_BACKEND` (default `"auto"`, `fused_moe.py:68`).

Order:
1. **FlyDSL** — if not CK, FlyDSL requested + available, and not `return_local_topk_ids` / `flat` /
   `output_aux`, and `dispatch_policy == 0` → `_flydsl_moe_sorting` → `flydsl_moe_sorting_fwd`
   (`fused_moe.py:350-369`, impl `:285-332`).
2. **FLAT pass-through** — `flat=True` (gfx950-only asm 1-stage kernels do in-kernel routing) returns
   *unsorted* topk via `_moe_prepare_unsorted_input` (`fused_moe.py:371-374`, impl `:83-109`).
3. **Adaptive** — inside `_moe_sorting_impl`, when `output_aux` and `_MOE_SORT_BACKEND` not in
   `("opus","ck")` → `_adaptive_moe_sort` → `aiter.mxfp4_moe_sort` (emits `m_indices` + `reverse_sorted`
   and can atomic zero-init) (`fused_moe.py:187-200`, impl `:112-166`). No shape fallback — the kernel is
   codegen'd for a fixed shape set (`fused_moe.py:61-66`).
4. **Opus vs CK** — otherwise `_moe_sorting_impl(..., use_opus=not _USE_CK_MOE_SORTING)`
   (`fused_moe.py:376-390`): Opus calls `moe_sorting_opus_get_workspace_size` + `moe_sorting_opus_fwd`
   (`:235-261`), CK calls `moe_sorting_fwd` (`:262-276`).

Backend C++ bindings:
- CK: `csrc/py_itfs_ck/moe_sorting_kernels.cu:10` delegates to CK `moe_sorting(...)` from
  `moe_sorting_api.hpp` (`:8`; the CK tile implementation is not vendored in this snapshot).
- Opus: `csrc/py_itfs_cu/moe_sorting_opus_kernels.cu:14` delegates to `aiter::moe_sorting_opus`; the
  self-contained kernel is header-only in `csrc/include/moe_sorting_opus.h`.

## Config / knobs
- **`unit_size` / `block_size`** — per-expert padding granularity (default 32). Output buffer sizing:
  `max_num_tokens_padded = topk_ids.numel() + num_experts*block_size - topk` (`fused_moe.py:203`).
- **`dispatch_policy`** (Opus/CK) — `0` = auto-pick, `1` = always single (one-shot) kernel, `2` = always
  multi-pass ("mp") kernel (`moe_sorting_opus.h:1398`). Workspace is `0` for one-shot, else the mp size
  (`moe_sorting_opus.h:1401-1427`).
- **`expert_mask` / `num_local_tokens`** — expert-parallel (EP) masking: mark non-local experts so their
  tokens are dropped from the sort.
- **`return_local_topk_ids=True`** — forces Opus (CK cannot emit local ids) (`fused_moe.py:220-223`).
- **`output_aux=True`** — forces Opus + `dispatch_policy=1` and allocates `m_indices` / `reverse_sorted`
  when routed through the standard path (`fused_moe.py:227-233`).
- **`accumulate`** — when false (FlyDSL stage-2 reduce mode, no mask), `moe_buf` is a `(0,0)` placeholder
  and the caller owns the `[M, topk, model_dim]` intermediate (`fused_moe.py:215-218`).

## Numerics / parity
Sorting is index/metadata movement, not arithmetic: it reorders `topk_ids` (integral) and gathers
`topk_weights` (fp32, enforced at `moe_sorting_kernels.cu:24`). `moe_buf` is allocated in the downstream
compute dtype (`moebuf_dtype`, default bf16 at the `fused_moe` call sites) and zero-initialized so stage-2
atomic-add is well-defined. `num_valid_ids` carries the post-pad token count (mapped to CK's
`p_total_tokens_post_pad`, `moe_sorting_kernels.cu:60`) that bounds the downstream GEMM. No approximation is
introduced by the sort itself.

## Pitfalls
- **Backends are mutually exclusive and env-gated** — CK needs `AITER_USE_CK_MOE_SORTING=1`; the default is
  Opus. FlyDSL is skipped whenever `return_local_topk_ids` / `flat` / `output_aux` are set or
  `dispatch_policy != 0` (`fused_moe.py:350-357`).
- **`topk_weights` must be fp32** — CK asserts it (`moe_sorting_kernels.cu:24`); Opus asserts it
  (`moe_sorting_opus_kernels.cu:31`).
- **Adaptive sort has no shape fallback** — an un-codegen'd shape hits `TORCH_CHECK`; only safe because
  `output_aux` is set for tuned rows routed to the port (`fused_moe.py:61-66`).
- **FLAT path returns unsorted tensors** — the sorted-slot pointers alias `topk_ids` as scratch and are
  unread by the asm kernels (`fused_moe.py:107-109`); gfx950-only (`fused_moe.py:698`).
- **`moe_sorting_opus_fwd` is a `develop` op** (`aiter/ops/moe_sorting_opus.py:11,20`).

## Cross-links
- [../moe_align_block_size/overview.md](../moe_align_block_size/overview.md) — the vLLM-style align path (subset; no weight gather).
- [../fused_moe_grouped_gemm/aiter.md](../fused_moe_grouped_gemm/aiter.md) — consumer of the sorted buffers.
- [../grouped_gemm_moe/aiter.md](../grouped_gemm_moe/aiter.md) — grouped-GEMM stages driven by `sorted_expert_ids`.
- [../moe_routing_topk/aiter.md](../moe_routing_topk/aiter.md) — produces the `topk_ids` / `topk_weights` inputs.
- [../shared_expert_fusion/overview.md](../shared_expert_fusion/overview.md) — shares the same sort/accumulate buffer.

## Sources
- Python decls: `aiter/ops/moe_sorting.py:11-25`, `aiter/ops/moe_sorting_opus.py:11-38`.
- Dispatcher + backend flags: `aiter/fused_moe.py:57-68`, `:112-166`, `:169-282`, `:285-332`, `:335-400`; stage call site `:715-762`.
- CK binding: `csrc/py_itfs_ck/moe_sorting_kernels.cu:10-70`; header `csrc/include/moe_sorting.h:6-17`.
- Opus binding: `csrc/py_itfs_cu/moe_sorting_opus_kernels.cu:14-84`; impl/`dispatch_policy` `csrc/include/moe_sorting_opus.h:1398-1427`, `:3258-3262`.
- Related HIP align kernel: `csrc/kernels/moe_align_block_size_kernels.cu:38-162`; Python decl `aiter/ops/moe_op.py:49-58`.
