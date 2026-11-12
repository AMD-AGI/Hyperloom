# vLLM clone patches

The `vllm/` directory in this repo is a clone of upstream vLLM and is **gitignored** (it is the
build/runtime source tree, not part of this repo's history). A few fixes for the native
Windows + ROCm port are made as **direct edits to that clone**, so they are captured here as
patches for reproducibility. Everything else (the platform plugin, native-kernel builds, run
harness) lives in this repo and is monkeypatched/loaded at runtime without touching vLLM source.

Clone base when these were generated: vLLM `b1388b1` — this IS the `v0.19.1` tag (pip reports it
as `0.19.2.dev0+gb1388b1fb`, a post-tag dev version string).

**These patches are applied automatically** by `python tools/patch_vllm.py vllm` (the same
step that installs the bootstrap import and `sitecustomize.py`): it reads the tree to see which
patches are already there, so it is safe to re-run, and after a partial reset of the clone it
puts back exactly what is missing, down to single files of a multi-file patch. A patch whose
context a later patch rewrote (`native-cache-ops.patch` under `triton-attn-pc-scale.patch`) is
verified on a scratch copy with the later patch reversed. It is the required step for
`vllm serve` to work on Windows (see `windows-serve-windows.patch` below). The tool also warns when the clone is not at the
expected base commit (see below). Patches apply in **alphabetical filename
order** — when a patch depends on another one being applied first, name it so it sorts after
(e.g. `triton-attn-pc-scale.patch` builds on `native-cache-ops.patch`'s if/else store dispatch,
and sorts after it). The manual flow remains as fallback:

```
git -C vllm apply --ignore-whitespace ../patches/vllm/conch-group-size.patch
git -C vllm apply --ignore-whitespace ../patches/vllm/gemma4-moe-weightload.patch
git -C vllm apply --ignore-whitespace ../patches/vllm/hf-fs-windows-path.patch
git -C vllm apply --ignore-whitespace ../patches/vllm/kvarn.patch
git -C vllm apply --ignore-whitespace ../patches/vllm/native-attn-sliding-window.patch
git -C vllm apply --ignore-whitespace ../patches/vllm/native-cache-ops.patch
git -C vllm apply --ignore-whitespace ../patches/vllm/triton-attn-pc-scale.patch
git -C vllm apply --ignore-whitespace ../patches/vllm/triton-attn-qscales.patch
git -C vllm apply --ignore-whitespace ../patches/vllm/windows-serve-windows.patch
```

## windows-serve-windows.patch
Makes the vLLM HTTP server compatible with native Windows: avoids Linux-only socket options,
uses TCP instead of ZeroMQ IPC for local engine sockets, avoids polling Windows process handles
through ZMQ, and uses Windows-compatible signal handling.

## conch-group-size.patch
`conch.py`: conch's Triton W4A16 kernel applies one scale per `block_k == 64` tile, so it is only
numerically correct for `group_size >= 64`. Verified: gs 128/64 give rel_err ~3e-3, gs 32 gives
~0.96 (garbage -> degenerate output). Reverts the whitelist to `[-1, 64, 128]` (drops the wrong
32). group_size 32 (e.g. gemma4 AWQ) is instead routed to `WinRocmW4A16DequantKernel` in the
plugin (a correct dequant->matmul / fused GEMV fallback).

## gemma4-moe-weightload.patch
`gemma4.py` `_weight_iterator`: the fused-3D-expert explosion kept the checkpoint's underscore
quant suffix (`gate_proj_packed`/`_scale`), which never matched `expert_params_mapping`
(`experts.{id}.{proj}.` dotted) -> `KeyError 'layers.0.moe.experts.0.down_proj_packed'`. Rewrites
`_packed`/`_scale` to the canonical dotted `.weight_packed`/`.weight_scale` so compressed-tensors
fused-expert MoE checkpoints load 1:1 with the Linux behavior. Regression-safe: bare/`.weight`
(unquantized) names are untouched.

## hf-fs-windows-path.patch
`model_executor/model_loader/weight_utils.py` (from PR #18): `HfFileSystem` paths are POSIX-style
repo paths, but `os.path.join` uses the OS separator, so on Windows the file-list lookup asked for
`Qwen/Qwen2.5-...Int4\` and failed with "Invalid HF URI", falling back to one request per
pattern. The repo path is now joined with a forward slash.

## kvarn.patch
Port of Huawei **KVarN** (calibration-free KV-cache quant: Hadamard rotation + Sinkhorn
variance-normalisation + asymmetric RTN, K 4-bit per-channel / V 2-or-4-bit per-token, per
128-token tile) as a native vLLM KV-cache-dtype backend on Windows + ROCm (gfx1100). Enable with
`--kv-cache-dtype kvarn_k4v2_g128 --block-size 128`.

Contents (one self-contained patch; new files + integration edits):
- **New files** copied from github.com/huawei-csl/KVarN (Apache 2.0) with two ROCm edits:
  `v1/attention/backends/kvarn_attn.py`, `v1/attention/ops/{kvarn_decode,kvarn_store,triton_kvarn_decode,triton_kvarn_sinkhorn}.py`,
  `model_executor/layers/quantization/kvarn/{__init__,config,sinkhorn}.py`. ROCm edits:
  (1) dropped the `maxnreg` autotune configs in `triton_kvarn_decode.py` (NVIDIA-only; Triton-AMDGPU
  raises "Keyword argument maxnreg unrecognised"), pinned to a single BLOCK_N=32/nw=4 config for fast
  first-run; (2) diagnostic env gates left inert-by-default in `kvarn_attn.py`
  (`KVARN_FORCE_SLOW` = dequant+SDPA reference path, `KVARN_NO_HADAMARD`, `KVARN_GTRACK`,
  `KVARN_RECON_DEBUG`, `KVARN_FAST_FLUSH=0` = legacy per-tile flush).
  Two further edits for this port (issue #26): (3) the fp16 tail pool, rotation scratch and
  decode buffers are allocated during vLLM's memory-profiling pass (`forward` with
  `attn_metadata=None`), so `gpu_memory_utilization` accounts for them instead of them landing
  on top of a KV cache that already took the remaining memory; the plugin's
  `check_and_update_config` caps `max_num_seqs` to what the pool budget (`KVARN_POOL_MEM_FRAC`,
  default 0.08 of GPU memory) supports. (4) `flash_attn_varlen_func` (absent on Windows) is
  replaced by an SDPA stand-in using torch's `causal_lower_right` bias, which reaches the
  aotriton flash kernel with GQA; the previous fallback ran chunked-prefill continuations
  through fp32 SDPA with a boolean mask, i.e. the math kernel (+4.9 GiB per layer at
  2048x8192 on 32 heads), which is what made 8k prefills spill and crawl.
- **Integration edits** to vLLM: register the KVARN backend (`registry.py`); add the 4 kvarn presets
  to `CacheDType` (`config/cache.py`) and `STR_DTYPE_TO_TORCH_DTYPE` (`utils/torch_utils.py`); graft
  `TQFullAttentionSpec` (tile-quant full-attn spec with `tq_slot_size` byte sizing) into
  `v1/kv_cache_interface.py` and register it -> `FullAttentionManager` in
  `v1/core/single_type_kv_cache_manager.py`; `attention.py` `get_kv_cache_spec` returns
  `TQFullAttentionSpec` for `kvarn_*` layers (this branch takes PRECEDENCE over the sliding-window
  branch so `KVARN_QUANT_SLIDING` sliding layers get kvarn byte sizing, not fp16 SlidingWindowSpec).

THE key correctness fix lives OUTSIDE this patch, in `attention.py` too but as the plugin-critical
one-liner `self.impl.layer_name = prefix` (right after `self.impl = impl_cls(...)`): vLLM 0.19's
`Attention.__init__` never propagated the layer name to the impl, so KVarN's metadata builder found
zero impls for its group (`group_impls=0`) -> pool-slot allocation + tile flush never ran -> full
blocks were read back as uninitialised int4 (garbled decode). It is included in the attention.py hunk.

Status: correct end-to-end on gemma-4-26B (compressed-tensors W4A16 MoE); ~40 tok/s decode with
cudagraph (global-only) / real 4.4x KV-capacity win with `KVARN_QUANT_SLIDING=1` but slower pending a
builder D2H-sync refactor. Sliding-window semantics enforced by the decode kernel (`impl.sliding_window`).

## native-cache-ops.patch
`v1/attention/backends/triton_attn.py` `do_kv_cache_update`: prefers the native HIP
`torch.ops._C_cache_ops.reshape_and_cache_flash` (built 1:1 from `csrc/cache_kernels.cu` by
`experiments/vllm_c_ext/build_cache_c.py` into `vllm_win_cache_C.pyd`, loaded by the plugin) over the
Triton kernel for the per-token KV write: one HIP launch instead of a Triton launch per layer per
step. Falls back to the Triton kernel when the extension is not loaded, and always for fp8 caches:
the Windows build of the extension has no `ENABLE_FP8`, so its fp8 `scaled_convert` is a stub
(`assert(false)`, or zeros under NDEBUG) while the Triton kernel quantizes correctly.

## triton-attn-qscales.patch
`v1/attention/backends/triton_attn.py` (from PR #27, @CLVwp): the fp8 KV path asserted
`layer._q_scale_float == 1.0`, but on ROCm this backend never quantizes the query
(`supports_quant_query_input` is CUDA-only, so `layer.query_quant` is None) and never descales
it, so a non-1.0 q_scale is inert here. `--calculate-kv-scales` writes one (it calibrates Q for
the flash-attn / flashinfer fp8-Q paths) and used to kill the engine at startup. The assert is
now conditional on `layer.query_quant` being present, which keeps the upstream behaviour on CUDA.

## native-attn-sliding-window.patch
`csrc/attention/{attention_kernels.cuh, paged_attention_v1.cu}` + `csrc/ops.h`: threads a new
`sliding_window` int param through the generic wave32 paged_attention_v1 (device kernel -> global
kernel -> LAUNCH macro -> launcher -> public op -> ops.h decl). The QK stage gains an in-kernel
sliding-window mask `sw_mask = (sliding_window>0) && ((seq_len-1-token_idx) >= sliding_window)` ->
masked logit set to -FLT_MAX (excluded from the softmax normalizer AND the V accumulation), matching
vLLM's Triton `where((context_len-seq_offset) < SLIDING_WINDOW, S, -inf)`. `sliding_window<=0`
disables it (original full-attention behavior; perf unchanged). v2 is left intact (passes 0). This is
built (NOT compiled into the gitignored vllm tree at runtime) via `experiments/vllm_c_ext/build_attn_c.py`
into `vllm_win_attn_C.pyd` and loaded opt-in by the plugin (`VLLM_WIN_ATTN_NATIVE=1`).

Result: the native decode kernel is ~3.2x faster than Triton `kernel_paged_attention_2d` in isolation
and numerically correct, BUT the end-to-end ROCM_ATTN integration REGRESSES (-9% gemma, -5% ERNIE) --
the path overhead negates the kernel win. Kept
because the kernel is the RDNA3-native `fmha_v3` equivalent and the remaining flash-layout-swap path
would reuse it.

## triton-attn-pc-scale.patch
Issue #28 option 2: optional per-(kv head, channel) K descale for fp8 KV caches, on top of the
plugin's `kv_offsets` mean-removal (see `kv-fp8-audit.md`). The residual of each channel is
normalized to its own fp8 full range at store time and multiplied back at read time:

- `triton_reshape_and_cache_flash.py`: optional `k_scale_channel` tensor + `HAS_K_SCALE_CHANNEL`
  constexpr in `reshape_and_cache_kernel_flash` / wrapper — `tile_pos` already enumerates
  (head, dim) in natural `[Hk, D]` order, so the per-channel load is a flat indexed load before
  the fp8 divide. The `_diffkv` variant is unreachable under TRITON_ATTN and is not touched.
- `triton_unified_attention.py`: optional `k_descale_channel` + `USE_PC_K_SCALE` constexpr in the
  2D and 3D kernels — a loop-invariant `[HEAD_SIZE_PADDED]` load at `kv_head_idx * HEAD_SIZE +
  offs_d`, multiplied where the scalar descale is applied (fp8-K / non-fp8-Q branch only).
- `triton_attn.py`: both call sites pass `getattr(layer, "_kv_k_pc_scale", None)` — the tensor is
  stashed per layer by the plugin's `kv_offsets.py` when the offsets file has a `pc_scales` key
  (absent key = option 1 only; constexpr off = bit-identical codegen, no regression).

Direct kernel check: `run/kv_pc_kernel_check.py` (store round-trip + read descale vs a torch
reference, on/off, bit-exact).
