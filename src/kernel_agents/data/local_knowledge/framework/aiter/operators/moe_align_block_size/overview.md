---
title: moe_align_block_size — overview
kind: operator_overview
operator: moe_align_block_size
gens: [gfx942, gfx950, gfx1250]
dtypes: [int32, int64]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/triton/moe/moe_align_block_size.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/_triton_kernels/moe/moe_align_block_size.py
  - ROCm/aiter@b467ce342:csrc/kernels/moe_align_block_size_kernels.cu
  - ROCm/aiter@b467ce342:aiter/ops/moe_op.py
  - ROCm/aiter@b467ce342:op_tests/triton_tests/moe/test_moe_align_block_size.py
---

# moe_align_block_size

## TL;DR
`moe_align_block_size` reorders per-token top-k routing ids so all tokens routed to the same expert are
contiguous, and pads each expert's run up to a multiple of `block_size`. It writes `sorted_token_ids`
(flat routing index per slot, padding slots hold the out-of-range sentinel `= topk_ids.numel()`),
`expert_ids` (expert id per padded block), and `num_tokens_post_pad`. This lets the grouped-MoE GEMM launch
one uniform `block_size`-row tile per block with a single `expert_ids[block]` weight lookup. aiter ships a
Triton port (`moe_align_block_size_triton`) and a HIP/vLLM-derived kernel (`moe_align_block_size`).

## What it is
In a fused MoE, expert `e`'s GEMM operand is the set of tokens whose top-k picked `e`. Those tokens are
scattered across the batch, and each expert gets a different count. A tiled GEMM wants **fixed-size,
expert-homogeneous** row blocks. This op computes that mapping:
- gather token→expert assignments so each expert's tokens are contiguous,
- pad every expert's group up to `ceil(count/block_size)*block_size` (dummy slots use the sentinel id),
- emit `expert_ids[block]` = which expert each `block_size`-row block belongs to.

The grouped GEMM then iterates blocks: load `block_size` rows via `sorted_token_ids`, pick the weight with
`expert_ids[block]`. It is the id/expert/count subset of `moe_sorting`, which additionally gathers
`sorted_weights` and zero-inits the MoE output buffer — see
[../moe_sorting/overview.md](../moe_sorting/overview.md).

## Entry points (API)
| symbol | path:line | signature (from source) | purpose |
|---|---|---|---|
| `moe_align_block_size_triton` | `aiter/ops/triton/moe/moe_align_block_size.py:20` | `moe_align_block_size_triton(topk_ids, num_experts, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad) -> None` | Triton 4-stage host wrapper; writes results in place. |
| `moe_align_block_size` (HIP) | `csrc/kernels/moe_align_block_size_kernels.cu:134` | `moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad)` | HIP/vLLM kernel; extra `token_nums` (per-block remaining count) output. |
| `moe_align_block_size` (py decl) | `aiter/ops/moe_op.py:50` | `moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad) -> None` | `@compile_ops("module_moe_asm")` binding for the HIP kernel. |

## Dispatch / backends
Two independent implementations of the same contract (no runtime auto-select between them; the caller picks):
- **Triton** (`moe_align_block_size_triton`) launches four JIT kernels
  (`_triton_kernels/moe/moe_align_block_size.py`):
  1. `_moe_align_block_size_stage1_kernel` (`:43`) — grid `(num_experts,)`; each program counts tokens per
     expert into its row of a `tokens_cnts[(num_experts+1), num_experts]` scratch (`:57-61`).
  2. `_moe_align_block_size_stage2_kernel` (`:64`) — column-wise cumulative sum of the per-program partial counts (`:72-75`).
  3. `_moe_align_block_size_stage3_kernel` (`:78`, grid `(1,)`) — block-padded prefix sum across experts into
     `cumsum[num_experts+1]` and writes `total_tokens_post_pad` (`:88-92`).
  4. `_moe_align_block_size_stage4_kernel` (`:95`) — writes `expert_ids` per block, then scatters each token's
     flat index into `sorted_token_ids[rank_post_pad]` (`:111-122`).
  Host allocs `tokens_cnts`, `cumsum`, `tokens_per_thread = ceil_div(numel, num_experts)`
  (`triton/moe/moe_align_block_size.py:46-85`).
- **HIP** (`moe_align_block_size_kernels.cu`) runs the equivalent as **one block of `num_experts` threads**
  (`<<<1, num_experts, shared_mem>>>`, `:153`) with `shared_mem = (3*num_experts+1)*sizeof(int32)`
  (`:147`): Pass 1 `atomicAdd` counts (`:72-76`), Pass 2 thread-0 block cumsum + write positions (`:84-94`),
  Pass 3 per-expert `expert_ids`/`token_nums` fill (`:102-111`), Pass 4 `atomicAdd`-scatter tokens (`:119-125`).
  `total_tokens_post_pad = cumsum[num_experts] * block_size` (`:93`).

## Config / knobs
- **`block_size`** — alignment/pad granularity (the MoE tile M). Tests exercise `16 / 32 / 64 / 128`
  (`test_moe_align_block_size.py:137-149`).
- **`num_experts`** — sets the Triton grid and the HIP thread count / shared-memory footprint (O(num_experts)).
- **Output sizing (caller-owned):** `max_num_tokens_padded = topk_ids.numel() + num_experts*(block_size-1)`;
  `max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)` (`test_moe_align_block_size.py:78-82`,
  `:114-119`).
- **Sentinel fill (caller-owned):** the caller pre-fills `sorted_token_ids` with `topk_ids.numel()` so padded
  slots point out of range (`test_moe_align_block_size.py:86`, `:118`); the kernels only overwrite valid slots.

## Numerics / parity
Index-only; no arithmetic on activations. `topk_ids` is an integral tensor — the HIP kernel dispatches over
integral types via `VLLM_DISPATCH_INTEGRAL_TYPES` (`moe_align_block_size_kernels.cu:144`), outputs are
`int32`. Parity: `test_moe_align_block_size.py` checks `moe_align_block_size_triton` against a pure-torch
reference (`_torch_moe_align_block_size`) across the shapes above (`:151-163`). Slot ordering within an
expert is not guaranteed identical across backends (HIP uses `atomicAdd` scatter, `:123`), but the
expert grouping, block padding, and post-pad count are.

## Pitfalls
- **`topk_ids` shape convention** — the Triton wrapper's inline comment labels it `[num_tkns, num_experts]`
  (`triton/moe/moe_align_block_size.py:21`), but the aligned semantics use `numel = num_tokens*topk`; the
  test drives it as `[M, top_k]` (`test_moe_align_block_size.py:151-159`). Treat it as flattened routing ids.
- **Caller must pre-fill the sentinel** — kernels do not initialize padding slots of `sorted_token_ids`.
- **HIP kernel is single-block** — `num_experts` threads in one block; large `num_experts` grows shared
  memory linearly and bounds occupancy (`:147`, `:153`).
- **Ids must be in `[0, num_experts)`** — Pass 1 indexes `expert_token_counts[expert_id]` with no bounds
  check (`moe_align_block_size_kernels.cu:74-75`).
- **`token_nums` is HIP-only** — the Triton port does not emit the per-block remaining-count array.

## Cross-links
- [../moe_sorting/overview.md](../moe_sorting/overview.md) — production fused-MoE superset (adds weight gather + output zero-init).
- [../fused_moe_grouped_gemm/aiter.md](../fused_moe_grouped_gemm/aiter.md) — consumes `sorted_token_ids` / `expert_ids` blocks.
- [../grouped_gemm_moe/aiter.md](../grouped_gemm_moe/aiter.md) — grouped-GEMM stages keyed by `expert_ids`.
- [../moe_routing_topk/aiter.md](../moe_routing_topk/aiter.md) — produces `topk_ids`.

## Sources
- Triton wrapper: `aiter/ops/triton/moe/moe_align_block_size.py:20-85`.
- Triton kernels: `aiter/ops/triton/_triton_kernels/moe/moe_align_block_size.py:43-122`.
- HIP kernel + launcher: `csrc/kernels/moe_align_block_size_kernels.cu:38-126`, `:134-162`.
- Python HIP binding: `aiter/ops/moe_op.py:49-58`.
- Parity test: `op_tests/triton_tests/moe/test_moe_align_block_size.py:12-163`.
