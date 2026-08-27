---
title: causal_conv1d — overview
kind: operator_overview
operator: causal_conv1d
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/causal_conv1d_update.py
  - ROCm/aiter@b467ce342:aiter/ops/causal_conv1d_fwd_split_qkv.py
  - ROCm/aiter@b467ce342:aiter/ops/chunk_gated_delta_rule_fwd_h.py
---

# causal_conv1d

## TL;DR
The short **causal depthwise 1-D convolution** that front-runs the token-mixing block in Mamba/SSM and
linear-attention layers. Two aiter kernels cover the two regimes: **`causal_conv1d_update`** for autoregressive
**decode** (seqlen≈1, in-place sliding-window state), and **`causal_conv1d_fwd_split_qkv`** for **prefill**
(varlen `cu_seqlen`, fused Q/K/V split output, bf16, width=4). A related neighbor,
**`chunk_gated_delta_rule_fwd_h`**, is the gated-delta-rule (GDN) chunked hidden-state forward that consumes the
conv outputs in gated-linear-attention models.

## What it is
- **Decode update** (`causal_conv1d_update.py:11`): given one (or a few) new tokens and a persistent
  `conv_state` window, apply the depthwise causal conv (optionally SiLU), updating `conv_state` in place. Supports
  continuous batching (`conv_state_indices`), circular vs linear state, and a `pad_slot_id` skip.
- **Prefill split-qkv** (`causal_conv1d_fwd_split_qkv.py:90`): a varlen HIP conv over channels concatenated as
  `[Q|K|V]`, writing contiguous `q/k/v` outputs in one launch — a drop-in for the Triton
  `causal_conv1d_split_qkv_triton_fn`.
- **Gated-delta neighbor** (`chunk_gated_delta_rule_fwd_h.py:189`): chunked GDN hidden-state `h` forward with
  `h` layout `[V, K]`, the operator these conv outputs typically feed in linear/gated-delta attention.

## Entry points (API)
| symbol | path:line | signature (abridged) | purpose |
|---|---|---|---|
| `causal_conv1d_update` | `aiter/ops/causal_conv1d_update.py:11` | `(x, conv_state, weight, bias, out, use_silu, cache_seqlens, conv_state_indices, pad_slot_id) -> None` | decode-time conv + in-place state update (`module_causal_conv1d_update`) |
| `causal_conv1d_split_qkv_hip_fn` | `aiter/ops/causal_conv1d_fwd_split_qkv.py:90` | `(x, weight, bias, conv_states, query_start_loc, k_dim, v_dim, cache_indices=None, has_initial_state=None, activation="silu", pad_slot_id=-1, block_m=None, metadata=None) -> (q, k, v)` | prefill varlen conv, fused Q/K/V split |
| `causal_conv1d_fwd_split_qkv_hip` | `causal_conv1d_fwd_split_qkv.py:22` | raw `@compile_ops(MD_NAME, develop=True)` op | `module_causal_conv1d_fwd_split_qkv`; `aiter_tensor_t` ABI, outputs written in place |
| `chunk_gated_delta_rule_fwd_h_hip_fn` | `aiter/ops/chunk_gated_delta_rule_fwd_h.py:189` | `(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, chunk_size=64, save_new_value=True, cu_seqlens=None, selected_bv=None, state_dtype=None, use_exp2=True, g_head_major=False) -> (h, v_new, final_state)` | related GDN chunked hidden-state forward (`module_chunk_gdr_fwd_h`) |

`causal_conv1d_update` is star-imported to top level (`aiter/__init__.py:130` → `aiter.causal_conv1d_update`);
`causal_conv1d_split_qkv_hip_fn` and `chunk_gated_delta_rule_fwd_h_hip_fn` are reached via their module paths.

## Dispatch / backends
- All three are JIT-compiled HIP modules (`@compile_ops(..., develop=True)`). The split-qkv and GDN kernels use the
  **`aiter_tensor_t` ABI** (no C++ torch dependency): `develop=True` converts each `torch.Tensor` to a pybind
  `aiter_tensor_t`, and the caller pre-allocates outputs written in place (`causal_conv1d_fwd_split_qkv.py:16-42`).
- **Tile dispatch (split-qkv)**: token-tile `TM` is auto-picked from average sequence length —
  `≤12→8, ≤24→16, ≤48→32, else 64` (`_TILE_SEGMENTS`, `causal_conv1d_fwd_split_qkv.py:50`; `_pick_tile:53`) — unless
  the caller forces `block_m ∈ {8,16,32,64}`. A vectorized `_build_chunk_metadata` builds the flattened
  `(sequence, chunk)` schedule with no Python loop and no device sync (graph-capture safe, `:60`).
- **`bv` dispatch (GDN)**: `_select_bv` chooses value-block width from `{64,32,16}` using per-CU LDS
  (`shared_memory_per_multiprocessor`) and CU count, with an arch LDS fallback map
  `{gfx95: 128 KiB, gfx94: 64 KiB}` (`chunk_gated_delta_rule_fwd_h.py:37`, `_compute_bv:47`).

## Config / knobs
| knob | where | values / constraint |
|---|---|---|
| conv `width` | `causal_conv1d_update` | 2, 3, or 4 (docstring `causal_conv1d_update.py:56`) |
| dtypes | `causal_conv1d_update` | fp16, bf16, fp32 (docstring `:55`) |
| `use_silu` / `activation` | both convs | SiLU/swish activation toggle (`split_qkv.py:170`) |
| circular vs linear state | `causal_conv1d_update` | `cache_seqlens` empty → linear shift; non-empty → circular indexing (`:44-46`) |
| `pad_slot_id` | both convs | skip rows where `conv_state_indices[i]==pad_slot_id`; `PAD_SLOT_ID=-1` (`split_qkv.py:13`) |
| split-qkv `x` layout | `split_qkv` | `[dim, cu_seqlen]` bf16, channels `[Q|K|V]`; `weight [dim, width]`, **width must be 4** (`split_qkv.py:118-121`) |
| `block_m` (TM) | `split_qkv` | force `{8,16,32,64}` or auto from avg seqlen (`:128`) |
| GDN `chunk_size` / K / V | `chunk_gdr` | fixed `chunk_size=64`, `K=V=128`, bf16 (`chunk_gated_delta_rule_fwd_h.py:219-224`) |
| `use_exp2` / `g_head_major` | `chunk_gdr` | gates in log2 space (`RCP_LN2` prescale of `gk`); token- vs head-major `g` (`:216-217`, `:281`) |

## Numerics / parity
- `causal_conv1d_update`: initialize `out` with `torch.zeros_like` (not `empty_like`) when `pad_slot_id` padding is
  used, so padded outputs are zero (`causal_conv1d_update.py:35`).
- split-qkv: **bf16 only** for `x` and `conv_states` (`TypeError` otherwise, `split_qkv.py:113-116`); outputs are
  contiguous `q/k [cu_seqlen, k_dim]`, `v [cu_seqlen, v_dim]` (`:174-176`).
- GDN: `initial_state`/`state_dtype` must be fp32 or bf16 (`chunk_gated_delta_rule_fwd_h.py:127-136`); `gk` supplied
  in natural-log space is prescaled by `RCP_LN2` once before exp2 kernels consume it (`:281-283`); varlen mode
  requires `B==1` flattened input (`:256`).

## Pitfalls
- split-qkv **hard-requires `width==4`** and bf16 — other widths/dtypes raise (`split_qkv.py:113-121`); it is a
  prefill kernel (varlen `query_start_loc`), not a decode-step op.
- `causal_conv1d_update.conv_state` must have `state_len ≥ width-1` and is **modified in place** each call
  (`causal_conv1d_update.py:31-32`, `:51`); pass empty `torch.empty(0,...)` for optional `bias`/`cache_seqlens`/
  `conv_state_indices` rather than `None`.
- GDN varlen path asserts `B==1` and imports `prepare_chunk_offsets` from the Triton GDN utils
  (`chunk_gated_delta_rule_fwd_h.py:251-256`); `chunk_size != 64` or `K/V != 128` raise (`:219-222`).
- These conv kernels carry **no explicit arch guard** in-source (portable HIP); the confirmable arch evidence
  (`gens`) comes from the gated-delta neighbor's LDS arch map (`gfx94`/`gfx95`, `chunk_gated_delta_rule_fwd_h.py:37`).

## Cross-links
- SiLU/activation epilogue → operators/act_and_mul_silu_gelu/aiter.md.
- The conv outputs feed the gated-delta / linear-attention state forward
  (`chunk_gated_delta_rule_fwd_h`, same folder sources) and QKV projections → operators/rope/aiter.md.
- FP8/quant norm that often precedes token-mixing in these blocks →
  operators/collectives_all_reduce/overview.md (fused AR+RMSNorm emits the bf16 mirror GDN in-proj consumes).

## Sources
- on-box `ROCm/aiter@b467ce342`: `aiter/ops/causal_conv1d_update.py` (`causal_conv1d_update:11`, modes/dtypes/width
  docstring `:44-57`), `aiter/ops/causal_conv1d_fwd_split_qkv.py` (`causal_conv1d_split_qkv_hip_fn:90`,
  `causal_conv1d_fwd_split_qkv_hip:22`, `aiter_tensor_t` ABI note `:16-20`, tile dispatch `_TILE_SEGMENTS:50`/
  `_pick_tile:53`, `_build_chunk_metadata:60`, bf16/width guards `:113-121`), `aiter/ops/chunk_gated_delta_rule_fwd_h.py`
  (`chunk_gated_delta_rule_fwd_h_hip_fn:189`, raw op `:94`, `bv` select `:47-90`, LDS arch map `:37`,
  chunk/K/V guards `:219-224`), `aiter/__init__.py:130`.
