# vLLM on native Windows + AMD ROCm (RDNA3)

Glue and build tooling to run [vLLM](https://github.com/vllm-project/vllm) on **native
Windows** (no WSL2) with **AMD ROCm** on **RDNA3** consumer GPUs. Developed and tested on a
**Radeon RX 7900 XT (gfx1100)**.

This is **not** a fork of vLLM. It is an out-of-tree platform plugin plus a set of
compatibility shims, a set of source patches (`patches/vllm/`), and a build harness that
compiles vLLM's own HIP kernels natively on Windows. Upstream vLLM is cloned and pinned
separately (see Setup).

## Status (honest)

Experimental, but past "it just runs". What currently works on the test machine:

- vLLM imports and **generates correct tokens** on gfx1100, native Windows, single GPU.
- **W4A16 quantized models run** across formats: compressed-tensors, GPTQ-Int4, and AWQ-Int4.
- vLLM's **native exllama W4A16 GEMM** (`_C.gptq_gemm`) is **compiled natively** for Windows
  (GPTQ models otherwise have no kernel on Windows at all).
- A **custom M=1 W4 dequant-GEMV** (Triton) for AWQ-uint4 decode, which has no fast kernel on
  ROCm otherwise (exllama rejects uint4, Marlin is CUDA-only, leaving only the slow `conch` tile).
- **torch.compile / inductor works** (CompilationMode.STOCK_TORCH_COMPILE), and **hipGraph
  decode capture works** (`cudagraph_mode=FULL_DECODE_ONLY`).
- **fp8 KV cache** works (Triton path), ~2x KV-cache capacity / context length. On Qwen2.5 it
  requires the calibrated per-layer K-offset file (see "fp8 KV cache" below); with it, output
  is coherent at 128-8k context.
- **KVarN KV-cache quantization** (calibration-free Hadamard + Sinkhorn + asymmetric RTN, K 4-bit /
  V 2-bit; Huawei's method, Triton kernels ported to gfx1100) runs end-to-end and gives **~4.7x KV
  capacity** at ~fp16 accuracy (demonstrated on Qwen2.5-7B: 999k vs 210k KV tokens, coherent). **WIP:**
  its per-forward workspace over-allocates (~5 GiB), so today it only fits models that leave enough
  headroom (7-9B), and it is ~35% slower — a capacity feature, not a speed one. Not finished; the
  builder memory refactor is pending.

### Performance (measured)

Single-stream decode (batch 1, greedy) on the test machine. Output was verified coherent for
each model. All weights are 4-bit; KV cache fp16 unless noted.

| Model | Quantization | decode (tok/s) | KVarN KV-quant (WIP) | notes |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` (dense, 7B) | GPTQ Int4 | **115** | **4.74x KV** (999k vs 210k tok), 74.7 tok/s, coherent | native exllama GEMM + hipGraph decode, `gpu_memory_utilization=0.9` (spill-verified clean) |
| `cyankiwi/ERNIE-4.5-21B-A3B-Thinking-AWQ-4bit` (**MoE**, 21B / A3B active, head 128) | compressed-tensors W4A16 gs32 | 62.7 → **79.2** | — (14GB weights leave no room for the ~5 GiB KVarN workspace) | stock → +M=1 MoE-decode gather-GEMV + native `wvSplitK` dense. Fits 20GB with ~3 GiB free — spill-free |
| `sahilchachra/Qwythos-9B-Claude-Mythos-5-1M-AWQ` (Qwen3.5 hybrid, 9B) | compressed-tensors W4A16 | **61.7** | — | native exllama + hipGraph, **`gpu_memory_utilization=0.7`** (see note) |
| `casperhansen/deepseek-r1-distill-qwen-14b-awq` (dense, 14B) | AWQ Int4 | **50.3** | — | custom M=1 W4 GEMV, autotuned, util 0.9 (spill-verified clean) |

**Community-validated hardware.** RX 7800 XT (`gfx1101`, 60 CU, ~624 GB/s): **81.9 tok/s** on
Qwen2.5-7B-GPTQ, validated end-to-end by [@CLVwp](https://github.com/CLVwp)
([#6](https://github.com/ThePie88/vLLM-ROCm-Windows/issues/6)) -- consistent with the bandwidth ratio to
the 7900 XT's 115 (decode is memory-bound). `vllm serve` plus the local chat UI (`run/chat_ui.py`) were
validated on the same card.

Numbers re-measured 2026-07-02, cudagraph (`FULL_DECODE_ONLY`) decode, and **verified spill-free** by
polling the Windows GPU shared-memory counter during the run (peak shared == the ~0.76 GiB desktop
baseline). `torch.cuda.mem_get_info()` only reports *dedicated* VRAM free, so it does NOT catch a WDDM
spill on its own.

**VRAM caveat (per-model `gpu_memory_utilization`).** Standard dense models (Qwen, deepseek) are safe at
0.9. The **Qwythos-9B hybrid** (linear-attention Mamba state cache + an **unquantized vision tower** + a
248k vocab) is NOT: vLLM's memory profiling does not account for the Mamba/vision allocations, so at 0.9 it
fills dedicated VRAM with KV and those extras overflow ~2.7 GiB into shared DRAM — a fake, spill-slowed
number (that was the old 39.9). At `util=0.7` it runs entirely in dedicated VRAM (6 GiB free) and is both
clean and faster (61.7). The 26B-class MoEs (gemma-4, 17GB weights) overfill 20GB at any workable util, so
ERNIE-4.5-21B is used as the clean MoE bench. Aggregate throughput scales with concurrency (Qwen2.5-7B,
greedy): ~73 tok/s at batch 4, ~232 at batch 16, ~358 at batch 32.

**KVarN (experimental / WIP).** `--kv-cache-dtype kvarn_k4v2_g128 --block-size 128` runs end-to-end on
gfx1100 (K 4-bit / V 2-bit, calibration-free). On Qwen2.5-7B, vLLM sizes **999,296 KV tokens vs 210,784
in fp16, 4.74x capacity**, generation stays coherent at 74.7 tok/s (~35% slower than the fp16 115: KVarN is
a KV-*capacity* feature, not a speed one), and an 8k prompt prefills in 2.9 s (`max_num_batched_tokens`
2048) or 2.4 s (8192). Memory accounting: the fp16 tail pool (`2*max_num_seqs + ceil(max_num_batched_tokens/128)
+ 8` slots per attention layer) is allocated during vLLM's memory profiling, so `gpu_memory_utilization`
covers it, and the plugin caps `max_num_seqs` to what the pool budget supports (`KVARN_POOL_MEM_FRAC`,
default 8% of GPU memory; the clamp and the pool size are printed at startup). Chunked-prefill
continuations use an SDPA stand-in for `flash_attn_varlen` (torch's `causal_lower_right` bias reaches the
aotriton flash kernel), where the previous fallback materialized fp32 score matrices and spilled. Keep
`gpu_memory_utilization` at 0.8 or `--max-num-seqs` small with KVarN: cudagraph capture runs after the KV
cache has been sized, and on Windows going past VRAM spills to shared memory instead of failing. Models
whose weights leave little headroom (ERNIE, 14 GB) are untested with the new accounting.

**fp8 KV cache** (`--kv-cache-dtype fp8`) works on Qwen2.5 as of PR #30 (issues #25/#28) — with
one calibration step. The problem: a few K channels carry a near-constant offset of ~420 (the
k_proj bias on the lowest-frequency RoPE channels), fp8 stores them with an absolute error of up
to 16 whatever the scale, and the attention output of those layers was off by 0.9-1.0 (max-rel)
from the first tokens. Scales at any granularity, e5m2 and a Hadamard rotation do not help; the
fix subtracts a calibrated per-layer per-channel mean from K before the store (never added back:
q*mean cancels in the softmax) and optionally normalizes each residual channel to its own fp8
full range — attention error drops to ~0.05 max-rel / 0.017 rms-rel offline, generation is
coherent at 128-8k context.

Two steps, from `run\`:

```bat
:: 1. calibrate once (two sweeps; writes offsets + per-layer + per-channel scales)
set VLLM_KV_CTXS=128,2048,8192
set VLLM_KV_PROBE_MEAN=kv_offsets.pt
python kv_scale_probe.py

:: 2. run any fp8 workload with the offsets file
set VLLM_WIN_KV_OFFSETS=kv_offsets.pt
```

Design, measurements and results: [`kv-fp8-audit.md`](kv-fp8-audit.md). Tooling:
`run/kv_scale_probe.py` (amplitudes, per-channel map, layer dump, offset calibration),
`run/kv_quant_schemes.py` (format comparison on a dump), `run/kv_pc_kernel_check.py` (direct
kernel check), `VLLM_WIN_KV_KSCALE` / `VLLM_WIN_KV_VSCALE` (static per-tensor scales, PR #27).
Without the offsets file Qwen2.5 fp8 stays garbled; KVarN remains the long-context *capacity*
option.

Decode is still below the card's ~800 GB/s memory-bandwidth roofline; per-shape GEMV tuning and porting
the rest of the `csrc` kernels are ongoing.

### Not done

- **Single GPU only.** RCCL does not exist on Windows, so tensor/pipeline parallel are out of
  scope; `torch.distributed` is shimmed for the single-process case only.
- KV-cache quantization: fp8 works, including calibrated per-layer offsets/scales on Qwen2.5
  (see "fp8 KV cache" above); KVarN works but stays WIP (see Status); int8-asymmetric per-channel
  K (the theoretically cleanest format, < 0.011 error everywhere) would need a new vLLM cache
  dtype and is not planned.
- Only part of vLLM's kernel suite is built natively so far (see "Native kernels" below).

## Tested stack (pinned, fragile)

This depends on a specific, somewhat experimental combination. Other versions may not work.

- Windows 11, AMD Radeon RX 7900 XT (gfx1100)
- A ROCm-enabled PyTorch **Windows** build: `torch 2.10.0+rocm7.13` (TheRock-class), `torchvision 0.25.0`, Python 3.12
- AMD HIP SDK 7.2 (`C:\HIP-SDK`), MSVC (Visual Studio Build Tools), Windows SDK 10
- `triton-windows` 3.6, `conch-triton-kernels`, `llguidance`, `xgrammar`
- vLLM **v0.19.1** (the newest tag pinned to torch 2.10; v0.20+ requires torch 2.11)

Note: helper scripts contain absolute paths from the author's machine
(`C:\HIP-SDK`, `E:\BuildTools`, `C:\Users\...`). Adjust them for your environment.

## How it works

- `windows_rocm_plugin/` is a pip-installable package providing:
  - `WindowsRocmPlatform` (registered via the `vllm.platform_plugins` entry point) that
    detects the GPU through `torch.cuda` instead of the Linux-only `amdsmi`.
  - A **single-process `torch.distributed` shim** (the Windows ROCm torch wheel is built
    without distributed), plus stubs for `amdsmi`, `uvloop`, `fcntl`,
    `torch._C._distributed_c10d`, and a tokenizer-class compatibility alias.
  - A **`torch.distributed.tensor` stub** that makes the (natively absent) DTensor module
    raise `ModuleNotFoundError` instead of a half-initialized `ImportError`. inductor's graph
    logging guards that import with `except ModuleNotFoundError`; without the stub,
    `torch.compile` dies during compilation. This is what unblocks inductor here.
  - `cops.py`: loads the compiled native kernel library (see below) so `torch.ops._C.*`
    resolve to the real HIP kernels, and registers torch-native fallbacks for any op the
    native build does not provide (so vLLM's unconditional `torch.ops._C.*` bindings work
    either way).
  - `kv_scales.py` / `kv_offsets.py`: static fp8 KV scales and calibrated per-layer K-offset
    removal (+ optional per-channel residual scale) for fp8 KV caches — issues #25/#28, see
    "fp8 KV cache" in Status and `kv-fp8-audit.md`.
- vLLM is installed with `VLLM_TARGET_DEVICE=empty` (no kernels compiled by vLLM's own build),
  then `tools/patch_vllm.py` applies the source patches under `patches/vllm/` (the bootstrap
  import in `vllm/__init__.py`, Windows port fixes, the KVarN backend, the fp8-K per-channel
  descale kernels, ...).

### Native kernels

`experiments/vllm_c_ext/` builds vLLM's **own** `csrc` HIP kernels for Windows. vLLM's
Linux build relies on a CUDA->HIP header redirect that the Windows torch wheel does not ship,
and `cpp_extension`'s hipify orchestrator mishandles Windows paths, so the harness applies
torch's hipify substitution engine (`RE_PYTORCH_PREPROCESSOR` + `PYTORCH_MAP`) to the sources
directly, with a small set of redirect shim headers. Currently built and validated:

- `silu_and_mul`, `rms_norm`, `fused_add_rms_norm`, `rotary_embedding` (fused activation /
  layernorm / RoPE)
- the **W4A16 GPTQ/exllama GEMM** (`gptq_gemm`, `gptq_shuffle`) from
  `csrc/quantization/gptq/q_gemm.cu`, which has a dedicated small-batch path for single-stream
  decode

To select the native exllama GEMM for a compressed-tensors W4A16 model, set
`VLLM_DISABLED_KERNELS=ConchLinearKernel` (vLLM's ROCm kernel selection then falls through
from `conch` to the exllama kernel).

The plugin also ships a **custom M=1 W4 dequant-GEMV** (`awq_gemv.py`, pure Triton) registered
ahead of `conch` for AWQ-uint4 decode. AWQ-uint4 has no fast kernel on ROCm (exllama only
accepts uint4b8; Marlin is CUDA-only), so vLLM falls back to `conch`, whose throughput-shaped
tile is ~20x off memory bandwidth for a single decode row. The GEMV is a true reduction (no
`tl.dot`/split-K/atomicAdd) that reuses `conch`'s weight normalization and delegates prefill
(M>1) back to `conch`; it is `@triton.autotune`d per shape (BLOCK_N/num_warps). On
`casperhansen/deepseek-r1-distill-qwen-14b-awq` it takes decode from 12.2 to 50.9 tok/s.

Two siblings in the same family: `bf16_gemv.py` patches `UnquantizedLinearMethod.apply` with a
small-tile skinny bf16/fp16 GEMV for the M==1 decode step only (`VLLM_WIN_BF16_GEMV=1`) — the
unquantized dense MLP of a hybrid MoE is the biggest per-token weight consumer, and rocBLAS
under-utilises the CUs at M=1 on these shapes. And `moe_decode.py` replaces vLLM's fused-experts
sort/align/grouped-GEMM machinery with a direct loop over the top-k active experts (streaming
W4A16 dequant-GEMV in the `moe_wna16` layout) — that is the "+M=1 MoE-decode gather-GEMV" gain
in the ERNIE row of the performance table above.

### CK ck_tile FMHA (WMMA) for prefill

`experiments/ck_fmha/` builds a native **Composable Kernel `ck_tile` flash-attention** (forward, d128,
fp16 + bf16, causal + GQA, varlen/group-mode) for gfx1100 -- the RDNA3 WMMA attention path that AITER's
Windows gate (`ENABLE_CK=False`) hides but that CK itself supports. It compiles with hipcc + MSVC after a
one-line device-code patch (`std::memcpy` -> `__builtin_memcpy`). Isolated, it runs prefill attention at
~37 TFLOP/s vs ~11 for Triton `unified_attention` (~3.3x).

Wired into vLLM prefill via `VLLM_WIN_CK_PREFILL=1` (`cops.maybe_patch_ck_prefill`, opt-in): pure-prefill
batches with no prior KV context (head 128, no sliding-window / softcap / alibi) route their attention to
the CK varlen kernel; decode, mixed prefill+decode, and sliding-window steps fall through to Triton. The
KV-cache write is a separate step, so decode is untouched. This is a **prefill / TTFT** lever
(compute-bound WMMA), not a single-stream decode one, so the end-to-end win grows with context as the
O(S^2) attention fraction rises. On `ERNIE-4.5-21B-A3B` (bf16, clean paired runs, best-of-3 TTFT):

| prompt tokens | Triton | CK | TTFT speedup |
|---|---|---|---|
| 2059 | 415.7 ms | 381.9 ms | 1.09x |
| 4099 | 850.2 ms | 733.9 ms | 1.16x |
| 6156 | 1379.7 ms | 1113.5 ms | 1.24x |
| 8196 | 1990.6 ms | 1519.0 ms | 1.31x |
| 10253 | 2681.7 ms | 1935.9 ms | 1.39x |

At short prompts (~1k) the win is only ~1.03x -- attention is a small slice of the prefill step (QKV/O
projection + MoE) -- and the curve is still climbing at 10k. bf16 output differs slightly from Triton
(kernel numerics), which can flip greedy tokens. Correctness gate: rel ~1e-4 (fp16) / ~3e-3 (bf16) vs
`scaled_dot_product_attention` across causal, GQA, and multi-sequence varlen.

## Setup

```bat
:: 1. Clone the matching vLLM tag next to this repo's content
git clone --depth 1 --branch v0.19.1 https://github.com/vllm-project/vllm.git vllm

:: 2. Don't let pip replace your ROCm torch, then install vLLM with no kernels
cd vllm
python use_existing_torch.py
set VLLM_TARGET_DEVICE=empty
python -m pip install -e . --no-build-isolation
cd ..

:: 3. Apply the one-line shim import to vLLM, install the plugin and extra deps
python tools\patch_vllm.py vllm
python -m pip install -e windows_rocm_plugin
python -m pip install conch-triton-kernels llguidance xgrammar

:: 4. (optional) Build vLLM's native HIP kernels for Windows
cd experiments\vllm_c_ext
build_run.bat
```

Nothing above needs editing for a different machine: the `build_*.bat` wrappers locate MSVC (via
`vswhere`) and the HIP SDK themselves, resolve their own location, and take the GPU architecture to
compile for from the live device.

### Optional: the CK FMHA prefill kernel

Needed only for `VLLM_WIN_CK_PREFILL=1` (see [CK ck_tile FMHA](#ck-ck_tile-fmha-wmma-for-prefill)).
Clone Composable Kernel next to this repo, generate the FMHA instances, then build:

```bat
:: from the parent directory of this repo
git clone --depth 1 https://github.com/ROCm/composable_kernel

:: one portability fix: std::memcpy is host-only in device code on Windows HIP
:: in composable_kernel\include\ck_tile\core\arch\amd_buffer_addressing_builtins.hpp,
:: replace std::memcpy with __builtin_memcpy (one line, around line 148)

cd composable_kernel\example\ck_tile\01_fmha
python generate.py --targets gfx11 --api fwd --receipt 0 -o ..\..\..\..\ckfmha_gen

:: back in this repo
cd experiments\ck_fmha
build_ck_varlen.bat
```

Use `--targets gfx12` instead of `gfx11` on RDNA4. The build picks up `composable_kernel` and
`ckfmha_gen` as siblings of this repo; point `CK_ROOT` / `CK_FMHA_GEN` elsewhere if you put them
somewhere else.

### Overriding the auto-detection

Set any of these if a probe guesses wrong:

| variable | what it pins | default |
|---|---|---|
| `HIP_PATH` | HIP SDK / ROCm install root | `C:\HIP-SDK`, else newest `C:\Program Files\AMD\ROCm\*` |
| `VCVARS64` | MSVC `vcvars64.bat` | located via `vswhere` |
| `VLLM_WIN_GFX_ARCH` | `--offload-arch` target | the installed GPU's arch (`gfx1100`, `gfx1200`, ...) |
| `VLLM_WIN_BUILD_ROOT` | parent of the scratch build dirs | `C:\`, else `%LOCALAPPDATA%` |
| `VLLM_WIN_BUILD_CLEAN=0` | keep scratch dirs (incremental rebuilds) | wipe before building |
| `CK_ROOT` / `CK_FMHA_GEN` | Composable Kernel checkout / generated FMHA instances | a `composable_kernel` next to this repo |

Resolution lives in `tools/winrocm_paths.py` (Python) and `tools/winrocm_env.bat` (the `.bat`
wrappers). Only RDNA3 (gfx1100) is actually tested here; other targets should build but are unverified.

## Running

Run from `run/` (not the repo root, so the cloned `vllm/` directory does not shadow the
installed `vllm` package).

```bat
cd run
python first_token.py        :: smallest end-to-end smoke test (OPT-125m)
python bench.py              :: decode tok/s + VRAM (configure via VLLM_BENCH_* env vars)
python batch_sweep.py        :: aggregate throughput vs concurrency
python kv_bench.py           :: KV-cache dtype bench: prefill/decode/drift vs fp16 at 128..16k (VLLM_KV_* env vars)
python kv_scale_probe.py     :: per-layer amplitudes, per-channel K map, layer dump, and fp8 KV
                             ::   offset/scale calibration (VLLM_KV_PROBE_* env vars)
python kv_quant_schemes.py   :: offline KV quantization-format comparison on a probe dump
python kv_pc_kernel_check.py :: direct bit-exact check of the fp8 per-channel-scale Triton kernels
```

This is the shortlist — the complete script table (incl. `chat_ui.py`, `precision_check.py`,
`profile_decode.py`, `test_compile.py`, `s5_bench/`) lives in [`run/README.md`](run/README.md).

`bench.py` knobs (env): `VLLM_BENCH_COMPILE=1` enables inductor, `VLLM_BENCH_CGMODE=FULL_DECODE_ONLY`
enables hipGraph decode capture, `VLLM_DISABLED_KERNELS=ConchLinearKernel` selects the native
exllama GEMM.

For a quantized model with a broken tokenizer_class (e.g. some llm-compressor exports):

```bat
python ..\tools\fix_tokenizer_config.py <model-substring>
set HF_HUB_OFFLINE=1
```

## Layout

- `windows_rocm_plugin/` - the out-of-tree platform plugin and compatibility shims
- `tools/` - patch and fixup scripts
- `patches/` - direct edits to the `vllm/` clone, captured as git patches for reproducibility
  (applied automatically by `tools/patch_vllm.py`; documented in `patches/README.md`)
- `run/` - bench / KV-cache tooling / profiling / batch-sweep drivers
- `experiments/` - native `csrc` kernel build harness and standalone HIP/Triton kernel proofs

## Contributing

Issues and pull requests are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0, matching vLLM. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright 2026 ThePie88. The plugin, the native HIP and Triton kernels, their build harnesses
and the benchmark harnesses are original work; every source file carries an SPDX header.
**If you redistribute this code, in source or binary form, section 4 of the license requires you
to keep those headers and to carry the NOTICE contents.** Attribution is the only thing asked in
return, so please credit the project and link back to it.

vLLM itself is not included here and remains under its own license; files under `patches/vllm/`
contain vLLM source and are covered by vLLM's copyright, not the above.
