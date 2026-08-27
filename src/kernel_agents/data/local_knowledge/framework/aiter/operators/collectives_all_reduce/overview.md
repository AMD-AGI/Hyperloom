---
title: collectives_all_reduce — overview
kind: operator_overview
operator: collectives_all_reduce
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/custom_all_reduce.py
  - ROCm/aiter@b467ce342:aiter/ops/quick_all_reduce.py
  - ROCm/aiter@b467ce342:aiter/ops/communication.py
  - ROCm/aiter@b467ce342:aiter/dist/device_communicators/custom_all_reduce.py
  - ROCm/aiter@b467ce342:aiter/dist/communication_op.py
  - ROCm/aiter@b467ce342:csrc/include/custom_all_reduce.cuh
  - ROCm/aiter@b467ce342:csrc/include/quick_all_reduce.cuh
---

# collectives_all_reduce

## TL;DR
aiter ships its own **intra-node, RCCL-bypass** collectives (all-reduce / reduce-scatter / all-gather) built on
peer IPC (or HIP VMM on gfx1250), so tensor-parallel layers reduce over XGMI/PCIe **without launching NCCL**.
The all-reduce kernel picks **one-shot (1-stage)** for small payloads / TP2 and **two-shot (2-stage)** for large
ones (`custom_all_reduce.cuh:3746`). On top of raw AR, aiter fuses the **post-attention residual+RMSNorm (and
optional FP8/MXFP4 quant)** into the same kernel so the TP reduce and the norm epilogue are one launch. A separate
`quick_all_reduce` family does a **quantized (FP8/INT6/INT4/INT3) two-shot** AR for bandwidth-bound cases.

## What it is
Three cooperating layers:
- **`custom_all_reduce`** (`aiter/ops/custom_all_reduce.py`) — pybind wrappers over `module_custom_all_reduce`
  (generic CDNA) and `module_custom_all_reduce_gfx1250` (MI450). Raw AR + reduce-scatter + all-gather, plus
  fused residual-add + RMSNorm + (FP8 / per-group FP8 / MXFP4) quant epilogues, plus fused QK-norm(+RoPE) AR.
- **`fused_allreduce_mhc_post*`** (`module_fused_ar_mhc`) — fused AR + MHC (multi-head-combine) post-mix used by
  attention-output residual paths.
- **`quick_all_reduce`** (`aiter/ops/quick_all_reduce.py`, `module_quick_all_reduce`) — the "QuickReduce"
  quantized two-shot AR (optionally fused with RMSNorm).

The runtime driver is `CustomAllreduce` (`aiter/dist/device_communicators/custom_all_reduce.py:465`); TP entry
points live in `aiter/dist/communication_op.py`.

## Entry points (API)
| symbol | path:line | signature (abridged) | purpose |
|---|---|---|---|
| `all_reduce` | `aiter/ops/custom_all_reduce.py:28` | `(_fa, inp, out, use_new, open_fp8_quant, reg_inp_ptr, reg_inp_bytes)` | out-of-place custom AR; `use_new`→new-kernel path, `open_fp8_quant`→FP8 transport |
| `all_reduce_gfx1250` | `custom_all_reduce.py:253` | same sig, `fc_name="all_reduce"` on `module_custom_all_reduce_gfx1250` | MI450 VMM-based AR |
| `reduce_scatter` | `custom_all_reduce.py:40` | `(_fa, inp, out, m, n, k, split_dim, reg_ptr, reg_bytes)` | fused RS; `split_dim`∈{first=0,last=1,mid=2} |
| `all_gather_reg` / `all_gather_unreg` | `custom_all_reduce.py:54` / `:63` | `(_fa, inp, [reg_buffer,] out, [reg_bytes,] dim)` | AG on pre-registered vs copy-in buffer |
| `fused_allreduce_rmsnorm` | `custom_all_reduce.py:74` | `(_fa, inp, res_inp, res_out, out, w, eps, reg_ptr, reg_bytes, use_1stage, gemma_norm=False)` | AR + residual add + RMSNorm |
| `fused_allreduce_rmsnorm_pad` | `custom_all_reduce.py:90` | + hidden-dim pad | AR+RMSNorm when `out_dim != in_dim` |
| `fused_allreduce_rmsnorm_quant` | `custom_all_reduce.py:106` | + `scale_out`, `bf16_out_ptr` | AR+RMSNorm + **per-token FP8** quant |
| `fused_allreduce_rmsnorm_quant_per_group` | `custom_all_reduce.py:124` | + `group_size` | AR+RMSNorm + **per-group FP8** (block-scale) |
| `fused_allreduce_rmsnorm_mxfp4_quant` | `custom_all_reduce.py:142` | + `scale_out` (e8m0) | AR+RMSNorm + **MXFP4** quant |
| `fused_qknorm_allreduce` / `_rope` | `custom_all_reduce.py:159` / `:174` | `(_fa, qkv_in, q_w, k_w, q_out, k_out, v_out, [cos_sin_cache, position_ids, head_dim, rotary_dim,] eps, reg_ptr, reg_bytes)` | fused QK-norm (+RoPE) then AR, split to q/k/v |
| `fused_allreduce_mhc_post_only` | `custom_all_reduce.py:347` | `(_fa, inp, next_residual, residual_in, post_layer_mix, comb_res_mix, use_new=True, open_fp8_quant=False, reg_ptr=0, reg_bytes=0)` | fused AR + MHC post-mix (no pre/RMSNorm) |
| `fused_allreduce_mhc_post_one_stage` / `_post_split` | `custom_all_reduce.py:362` / `:377` | same sig | 1-stage vs 2-stage split (large-M) MHC-post variants |
| `launch_fused_allreduce_mhc_post_only` / `_split` | `custom_all_reduce.py:427` / `:454` | py helpers; squeeze `post_layer_mix`, alloc `next_residual` | ergonomic launchers returning `next_residual` |
| `qr_all_reduce` | `aiter/ops/quick_all_reduce.py:24` | `(fa, inp, out, quant_level, cast_bf2half=False)` | QuickReduce quantized two-shot AR |
| `qr_all_reduce_rmsnorm` | `quick_all_reduce.py:34` | `(fa, inp, residual_inp, residual_out, out, weight, eps, hidden_dim, quant_level, cast_bf2half=False)` | quantized AR fused with RMSNorm |
| `init_custom_ar` / `init_custom_qr` | `custom_all_reduce.py:16` / `quick_all_reduce.py:14` | handle constructors | build the AR context (`_fa`) |
| `init_dist_env` / `destroy_dist_env` | `aiter/ops/communication.py:23` / `:83` | TP/DP/PCP process-group setup | calls `set_custom_all_reduce(True)` and wires `ca_comm` |
| `tensor_model_parallel_all_reduce` (+ fused variants) | `aiter/dist/communication_op.py:141` | `(input_, use_new=True, open_fp8_quant=False, prefill_support=False)` | TP-group front doors used by serving stacks |

All of `custom_all_reduce.py` and `quick_all_reduce.py` are `from ... import *`-ed into the `aiter` namespace
(`aiter/__init__.py:104-105`), so e.g. `aiter.all_reduce`, `aiter.qr_all_reduce` resolve at top level.

## Dispatch / backends
- **Arch split**: `CustomAllreduce._select_ops()` binds the `_gfx1250` module when the device is MI450, else the
  generic module (`device_communicators/custom_all_reduce.py:469`). gfx1250 uses HIP **VMM** IPC (`_init_gfx1250`,
  `:636`) because `hipIpc` is unavailable there; generic CDNA uses `hipIpcMemHandle` pools (`_init_ipc`, `:706`).
- **One-shot vs two-shot** (`custom_all_reduce.cuh:3746`, `use_new=true` path): `world_size==2` → 1-stage;
  else if fully-connected and `bytes < 160 KiB` (ws≤4) or `< 80 KiB` (ws≤8) → 1-stage (`cross_device_reduce_1stage`,
  `.cuh:377`); otherwise 2-stage (`cross_device_reduce_2stage`, `.cuh:484`). A `write_mode` 2-stage variant
  (`.cuh:570`) is used only for `world_size==8`, `bytes > 512·4096·2`, on **gfx942** (`.cuh:3783`). `use_new=false`
  falls back to the vLLM-derived kernels with slightly different size thresholds (`.cuh:3854`).
- **quick_all_reduce** is **two-shot only** — every path dispatches through `TWOSHOT_*` (`quick_all_reduce.cuh:1573`).
- **Eligibility gate**: `should_custom_ar` (`device_communicators/custom_all_reduce.py:792`) requires byte size % 16,
  weak-contiguity, and a size cap: decode `inp_size ≤ 8192·8192`, prefill `≤ max_size/2` (`_fits_custom_ar_size:773`).
  Supported world sizes are `[2,4,6,8]` (`:467`); returns `None`/falls back otherwise.
- **CUDA-graph capture**: `capture()` records buffer addresses and `flush_graph_buffers` batch-registers them after
  capture (`:732`); the registered vs copy-in path is chosen by `enable_register_for_capturing`.

## Config / knobs
| knob | where | effect |
|---|---|---|
| `use_new` | `all_reduce` arg | select new AR kernels (default `True`) vs vLLM-derived kernels |
| `open_fp8_quant` | `all_reduce` arg | FP8-quantize the AR transport payload |
| `use_1stage` | fused RMSNorm variants | force 1-stage fusion kernel (`allreduce_fusion_kernel_1stage`) vs 2-stage |
| `quant_level` | `qr_all_reduce` arg | `QuickReduceQuantLevel`: `F16=0, FP8=1, INT6=2, INT4=3, INT3=4` (`quick_all_reduce.cuh:1437`) |
| `cast_bf2half` | `qr_all_reduce` arg | reinterpret bf16 as fp16 for the codec |
| `group_size` | per-group quant | FP8 block size; validated as power-of-two threads/group ≤ wavefront (`device_communicators/custom_all_reduce.py:88`) |
| `gemma_norm` | fused RMSNorm variants | Gemma-style `(1+w)` norm weight |
| `prefill_support` | TP front doors | raise the size cap to `max_size/2` for prefill payloads |
| `max_size` | `CustomAllreduce.__init__` | pre-registered IPC buffer size (default `1 GiB`, `:501`) |

## Numerics / parity
- Base AR is exact reduce in the element dtype (bf16/fp16), fp32 not required for the sum-tree; `open_fp8_quant`
  and `quick_all_reduce` trade accuracy for bandwidth by quantizing the transported partials — gate end-to-end.
- Fused per-group FP8 quant uses a butterfly `__shfl_xor` reduction scoped to one 64-lane wavefront, so
  `group_size/PACK_SIZE` must be a power of two ≤ 64 (`device_communicators/custom_all_reduce.py:96-141`).
- MXFP4 fused quant needs hidden `n % 32 == 0` (32-element e8m0 block) (`:149`).
- FP8 codec max is `240.0` (float8_e4m3fnuz on MI300X) (`quick_all_reduce.cuh:581`).

## Pitfalls
- **gfx1250 (MI450)** is **1-stage only** (`_init_gfx1250` comment, `device_communicators/custom_all_reduce.py:642`)
  and caps `world_size ≤ 4` — larger TP raises `RuntimeError` with **no RCCL fallback** (`:556`).
- gfx1250 **cannot register CUDA-graph-captured buffers** cross-rank, so capture is forced onto the copy-in "unreg"
  path (`enable_register_for_capturing=False`, `:614`); the extra `register_input_buffer(signal)` in
  `communication.py` is **skipped on gfx1250** or it deadlocks at startup (`communication.py:65-79`).
- Custom AR must be attached to a **non-NCCL** group (`assert`, `:527`) and to ranks **in the same node** — it
  disables itself across nodes (`:530`).
- IPC metadata exchange requires a **pure-TCP** `torch.distributed` store (no RCCL/gloo/MPI), asserted at
  `IPCBufferPool` init (`:277`).
- During warmup (pre-capture, not capturing) `custom_all_reduce` returns **zeros** to mimic allocation, but
  `reduce_scatter`/`all_gather` run the **real** op there — returning zeros corrupts hash-routed MoE accuracy
  (`:977`).
- The `_write_mode` fast path is **gfx942-only** and `world_size==6` is excluded from the vectorized reduce
  (`DISPATCH_REDUCE` requires `world_size != 6`, `.cuh:3808`).

## Cross-links
- Norm epilogue fused here → operators/rmsnorm/overview.md, operators/fused_add_rmsnorm/aiter.md.
- Output quant formats → operators/quant_dequant_fp8/aiter.md, operators/quant_fp4_mxfp/aiter.md.
- TP relevance: these collectives are the reduce seam for tensor-parallel GEMM/MoE →
  operators/dense_gemm/aiter.md, operators/fused_moe_grouped_gemm/aiter.md.
- QK-norm(+RoPE) fusion neighbor → operators/rope/aiter.md.

## Sources
- on-box `ROCm/aiter@b467ce342`: `aiter/ops/custom_all_reduce.py` (`all_reduce:28`, gfx1250 block `:238-343`,
  `fused_allreduce_mhc_post*:347-480`), `aiter/ops/quick_all_reduce.py` (`qr_all_reduce:24`),
  `aiter/ops/communication.py` (`init_dist_env:23`), `aiter/dist/device_communicators/custom_all_reduce.py`
  (`CustomAllreduce:465`, `_select_ops:469`, `_init_gfx1250:636`, `should_custom_ar:792`, `fused_ar_rms:1092`),
  `aiter/dist/communication_op.py` (TP/EP/DP/PP front doors), `csrc/include/custom_all_reduce.cuh`
  (`cross_device_reduce_{1,2}stage`, dispatch `:3746`), `csrc/include/quick_all_reduce.cuh`
  (`QuickReduceQuantLevel:1437`, twoshot `allreduce:1573`), `csrc/kernels/{custom_all_reduce,fused_ar_mhc_post}.cu`,
  `csrc/kernels/custom_all_reduce_gfx1250.cu`.
