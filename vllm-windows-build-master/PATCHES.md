# Patch Reference

> **Release selection:** [the Windows releases](docs/releases.md) by Python/CUDA version, installer, and source-build guidance.
>
> **Current patch:** `vllm-windows-v12.patch` targets upstream vLLM v0.33.0
> (`4ba787ec9959e566d3b587a796dff07673324da5`). Its 0.33-specific changes and
> validation are in the [build record](docs/v0.33.0-build-candidate.md#port-changes).

The detailed reference below documents `vllm-windows-v10.patch`, the committed
Windows delta against upstream vLLM 0.27.1; v12 carries the same areas forward
onto 0.33.0. Older patches remain in the repository for
legacy releases.

For build internals, see [docs/build.md](docs/build.md).

**Issue #16 follow-up:** v10 remains the source snapshot for the published
v0.27.1 wheel. It omits native SM120 CUTLASS FP8/NVFP4 kernels but does not fix
their runtime selection or the missing FP8 LM-head partition attributes.
Rebuilding this exact snapshot retains those issues. The
[existing-wheel workaround](docs/troubleshooting.md#blackwell-fp8-nvfp4)
documents the Marlin setting and separate Python edit; neither is incorporated
into v10 by this documentation update.

## Build environment

| | |
|---|---|
| Base | vLLM v0.27.1 (tag `v0.27.1`) |
| Compiler | MSVC 19.43.34810 (Visual Studio 2022 Community 17.13) |
| CUDA | 13.0 Update 2 (`nvcc` 13.0.88) |
| Python | 3.13.11 (`cp313`) |
| PyTorch | 2.13.0+cu130 |
| Triton | triton-windows 3.7.1.post27 |
| Arch list | `7.5;8.6;8.9;12.0` (Turing through Blackwell) |
| CUTLASS | FetchContent + Windows patch applied automatically |
| vllm-flash-attn | FetchContent + vendored CUTLASS submodule patched |
| Generator | Ninja (`MAX_JOBS=8`) |
| GPU | Built for SM 7.5/8.6/8.9/12.0; tested on RTX 3090 (SM 8.6) |

## Diff stats

```text
v10 patch size: 167,335 bytes unified diff against upstream v0.27.1.
SHA-256: 4B6C9CD543414EF3ED1EB7FCBD7F39CE6BE10BDC97D901261977EE905346C988
72 files changed, 1,900 insertions, 287 deletions.
```

Apply it to a clean upstream checkout with:

```bat
git checkout v0.27.1
git apply --check ..\vllm-windows-v10.patch
git apply ..\vllm-windows-v10.patch
```

## v0.27.1 additions

- Ported the Windows build/runtime surface to the stable-libtorch module
  layout and CUDA 13 toolchain used by vLLM 0.27.1.
- Added Windows packaging for the optimized Rust frontend and Rust tool
  parser, plus generated FlashAttention Python payloads.
- Added MSVC fixes for new attention math/types and omitted only the SM120
  CUTLASS specializations whose aligned by-value ABI is unsupported on
  Windows x64.
- Carried forward the Windows-safe CPU/filesystem KV tiers and added the
  hybrid KV block-table fallback required by the 0.27 runtime.
- Added regression coverage for filesystem mapping/tiering, shared mmap, and
  hybrid block tables. The focused suite passed 80 tests with one expected
  platform skip.

The v0.25.1 details below describe the earlier v9 patch and remain as release
history.

## v0.25.1 additions

These changes enable and harden vLLM's native offloading framework on Windows:

| Category | File(s) | Purpose |
|---|---|---|
| Dependencies | `requirements/common.txt` | Treat Windows `AMD64` as x86-64 for `llguidance` and `xgrammar` wheel metadata |
| Native filesystem I/O | `csrc/fs_io.cpp`, `CMakeLists.txt` | Build/package `fs_io_C.pyd` and use Windows-safe `std::filesystem` UTF-8 paths |
| Shared RAM tier | `vllm/v1/kv_offload/cpu/shared_offload_region.py` | Use the system temp directory and Windows `mmap.ACCESS_WRITE`; avoid the creator/joiner resize race |
| GPU/CPU transfer | `vllm/v1/kv_offload/cpu/gpu_worker.py` | Route Windows file-backed mmap restores through native CUDA DMA instead of unsafe Triton host-pointer loads |
| Filesystem tier | `vllm/v1/kv_offload/tiering/fs/io.py` | Add `O_BINARY`, an `os.read` fallback where `readv` is absent, complete-read checks, and safe cleanup |
| Cache namespace | `vllm/v1/kv_offload/file_mapper.py` | Sanitize and bound absolute Windows model paths while retaining the full identifier in the namespace hash |
| Block table | `vllm/v1/worker/block_table.py` | Add a pure-Torch CUDA slot-mapping fallback for builds where Triton is unavailable |
| Tests | upstream offload/block-table tests | Make mmap/filesystem tests portable and cover block mapping without Triton |
| Wheel packaging | `assemble_wheel_cu128_v0.25.1.py` | Assemble the cp313/cu128 wheel and reject missing native/tiering payloads or Windows fix markers |

The release launcher maps opt-in `cpu-lru`, `cpu-arc`, `fs-lru`, and `fs-arc`
modes to upstream `OffloadingConnector`, `CPUOffloadingSpec`, and
`TieringOffloadingSpec`. It is disabled by default; the filesystem tier has no
automatic capacity quota.

## v0.24.0 additions

These hunks were added on top of the v0.23.0 Windows/cu128 work:

| Category | File(s) | Purpose |
|---|---|---|
| Build system | `CMakeLists.txt`, `setup.py` | Skip QuTLASS and DeepGEMM on Windows; both are optional paths and their Linux build assumptions do not hold under MSVC |
| Build system | `setup.py` | Copy generated FlashAttention Python files from the temporary build tree with platform-independent paths so Windows editable builds retain rotary and CuteDSL modules |
| Build system | `CMakeLists.txt` | Skip cooperative TopK on Windows; CUDA 12.8's PTX wrapper signatures do not match the v0.24 cooperative top-k calls under this toolchain |
| Runtime Python | `vllm/model_executor/layers/sparse_attn_indexer.py` | Guard `torch.ops._C.cooperative_topk` so sparse attention falls back when the optional op is absent |
| Runtime Python | `vllm/platforms/cuda.py` | Suppress the expected missing `_qutlass_C` warning on Windows while preserving warnings for real import failures |
| Runtime Python | `vllm/utils/system_utils.py`, `vllm/entrypoints/openai/api_server.py` | Terminate process trees through psutil instead of unavailable `SIGKILL`; reject `--uds` clearly when Windows Python has no `AF_UNIX` |
| Runtime Python | `vllm/v1/worker/gpu/sample/states.py` | Request explicit NumPy `int64` output for the full-range random request seed; Windows C `long` is only 32-bit |
| Runtime Python | `vllm/v1/simple_kv_offload/cuda_mem_ops.py` | Replace the Windows `cuMemcpyBatchAsync` path, which triggers an illegal memory access, with standard per-region `cudaMemcpyAsync` transfers |
| Rust packaging | `setup.py` | Recognize `vllm-rs.exe` and `_rust_*.pyd` as prebuilt Rust artifacts on Windows |
| Wheel packaging | `assemble_wheel_cu128_v0.24.0.py` | Assembles the already-built tree into a cp313/cu128 wheel including `_rust_tool_parser.pyd`, `vllm-rs.exe`, `triton_kernels`, `fmha_sm100`, and generated FlashAttention rotary/CuteDSL Python payloads |
| Requirements | `requirements/cuda.txt` | Keep Linux-only CUDA helper packages out of the Windows install path: FlashInfer, TVM FFI, TileLang, CUDNN frontend, CUTLASS DSL, QuACK, TokenSpeed-MLA, Humming kernels, and fastsafetensors |

## v0.23.0 additions

These hunks were added on top of the v0.21.0 Windows/cu128 work:

| Category | File(s) | Purpose |
|---|---|---|
| Rust frontend | `rust/src/managed-engine/src/process.rs` | Keep Unix process groups/signals on Unix; use Windows process creation flags and `taskkill` for managed Python process-tree shutdown |
| Rust frontend | `rust/src/server/src/listener.rs` | Gate Unix-domain listener and inherited-fd support to Unix; keep TCP listener support on Windows |
| Rust frontend | `rust/src/cmd/src/main.rs` | Use Windows-compatible Ctrl+C shutdown path; disable mimalloc global allocator on Windows to avoid MSVC CRT mismatch with `esaxx-rs` |
| Rust frontend | `vllm/envs.py` | Resolve `VLLM_RUST_FRONTEND_PATH=auto` to `vllm-rs.exe` on Windows |
| Build tooling | build env / docs | `protoc` is required for the Rust frontend; `build.bat`/`run_build.bat` now surface `PROTOC` |
| Wheel packaging | `assemble_wheel_cu128_v0.23.0.py` | Assembles the already-built tree into a cp313/cu128 wheel and writes `RECORD` with `csv.writer` so comma-containing config filenames install under `uv` |

## cu128 / Python 3.13 / Blackwell additions

These hunks were added on top of the original cu126 patch for the
`v0.21.0-win-cu128` build:

| Category | File(s) | Purpose |
|---|---|---|
| API server (Windows) | `entrypoints/openai/api_server.py`, `entrypoints/openai/dp_supervisor.py`, `cli/serve.py`, `cli/launch.py`, `grpc_server.py`, `benchmarks/throughput.py` | Fall back `import uvloop` → `import asyncio as uvloop` (uvloop is Unix-only); set `WindowsSelectorEventLoopPolicy` so pyzmq's `add_reader` works on the Proactor-less loop |
| API server (Windows) | `vllm/v1/engine/utils.py` | `wait_for_engine_startup()`: don't register process sentinels (HANDLEs) with `zmq.Poller` on win32 (`not a socket`); detect dead procs via exit codes |
| API server (Windows) | `vllm/entrypoints/launcher.py` | `add_signal_handler` → `signal.signal` fallback (Unix-only on asyncio) |
| Blackwell build | `CMakeLists.txt` | Skip `minimax_reduce_rms_kernel.cu` on WIN32 (cudafe++ crash; multi-GPU only); marlin host `/Od`; quote CUDA-13 cccl include (gated CUDA≥13) |
| Blackwell build | `csrc/torch_bindings.cpp` | `#ifndef _WIN32` around the minimax op registration |
| Blackwell build | `cmake/external_projects/qutlass.cmake` (include guard in `CMakeLists.txt`) | Skip QuTLASS (NVFP4/MXFP4) on WIN32 — GCC inline-PTX, not MSVC-portable |

> **Note:** v0.21.0 cu128 context:
> These v0.21.0 cu128 fixes are carried forward in `vllm-windows-v9.patch`
> for v0.25.1.

## Files modified

| Category | File | Hunks | Purpose |
|---|---|---|---|
| Build system | `CMakeLists.txt` | 8 | Force CUDA toolkit; MSVC flags incl. `/Usmall`, `/Zc:__cplusplus`, `WIN32_LEAN_AND_MEAN`; auto-apply `cutlass-windows.patch`; link `CUDA::cublas` |
| Build system | `cmake/utils.cmake` | 1 | Quote paths in `file(REAL_PATH)` |
| Build system | `cmake/external_projects/vllm_flash_attn.cmake` | 1 | Auto-apply `vllm-flash-attn-cutlass-windows.patch` after FetchContent |
| Build system | `setup.py` | 5 | Allow Win CUDA, path conversion, find Ninja in venv |
| Build system | `requirements/cuda.txt` | 1 | Comment out flashinfer / cutlass-dsl / quack-kernels / tilelang / fastsafetensors / tokenspeed-mla |
| CUDA kernel | `csrc/attention/merge_attn_states.cu` | 3 | `uint` typedef, `std::isinf` → `isinf` |
| CUDA kernel | `csrc/core/math.hpp` | 1 | `__builtin_clz` → portable bit-twiddle |
| CUDA kernel | `csrc/cumem_allocator.cpp` | 1 | `<BaseTsd.h>`, `SSIZE_T`/`ssize_t` |
| CUDA kernel | `csrc/fused_qknorm_rope_kernel.cu` | 1 | `uint` typedef |
| CUDA kernel | `csrc/mamba/mamba_ssm/selective_scan_fwd.cu` | 3 | Replace nested `BOOL_SWITCH` lambdas with explicit dispatch |
| CUDA kernel | `csrc/moe/grouped_topk_kernels.cu` | 4 | `__attribute((aligned))` → `__align__` |
| CUDA kernel | `csrc/moe/marlin_moe_wna16/generate_kernels.py` | 2 | Flat `if` chain (avoid C1061), `os.remove` |
| CUDA kernel | `csrc/moe/topk_softplus_sqrt_kernels.cu` | 1 | **NEW** Hoist `#ifndef USE_ROCM` out of `DISPATCH_HASH(...)` macro arg |
| CUDA kernel | `csrc/persistent_topk.cuh` | 1 | **NEW** Guard `__attribute__((always_inline))` with `_MSC_VER` |
| CUDA kernel | `csrc/quantization/activation_kernels.cu` | 8 | Designated init, `int4` for `__int128_t`, `int64_t` |
| CUDA kernel | `csrc/quantization/awq/gemm_kernels.cu` | 2 | `__asm__ __volatile__` → `asm volatile` |
| CUDA kernel | `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu` | 1 | `and` → `&&` |
| CUDA kernel | `csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu` | 1 | **NEW** Add `()` to `quant_type_max_v<scalar_out_t>` call |
| CUDA kernel | `csrc/quantization/fused_kernels/layernorm_utils.cuh` | 1 | `quant_type_max_v<T>` → `quant_type_max_v<T>()` |
| CUDA kernel | `csrc/quantization/fused_kernels/quant_conversions.cuh` | 1 | Same |
| CUDA kernel | `csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu` | 1 | `or` → `\|\|` |
| CUDA kernel | `csrc/quantization/marlin/generate_kernels.py` | 1 | Flat `if` chain, `os.remove` |
| CUDA kernel | `csrc/quantization/utils.cuh` | 2 | Variable template → function template |
| CUDA kernel | `csrc/quantization/w8a8/fp8/common.cu` | 1 | `quant_type_max_v` call site fix |
| CUDA kernel | `csrc/quantization/w8a8/fp8/common.cuh` | 1 | Same |
| Runtime Python | `vllm/distributed/parallel_state.py` | 2 | `FakeProcessGroup` + `FileStore` instead of Gloo |
| Runtime Python | `vllm/entrypoints/openai/api_server.py` | 1 | Guard `SO_REUSEPORT` |
| Runtime Python | `vllm/entrypoints/openai/dp_supervisor.py` | 1 | **NEW** `import uvloop` fallback to `asyncio`; selector event-loop policy on Windows |
| Runtime Python | `vllm/envs.py` | 1 | **NEW** Default `VLLM_USE_FLASHINFER_SAMPLER=False` on `win32` |
| Runtime Python | `vllm/model_executor/model_loader/weight_utils.py` | 8 | Custom safetensors reader (numpy mmap + chunked GPU stream) |
| Runtime Python | `vllm/utils/network_utils.py` | 1 | ZMQ IPC → `tcp://127.0.0.1` |
| Runtime Python | `vllm/utils/system_utils.py` | 1 | Force `spawn` multiprocessing |
| Runtime Python | `vllm/v1/engine/core_client.py` | 1 | Force `InprocClient` |
| Runtime Python | `vllm/v1/simple_kv_offload/cuda_mem_ops.py` | 4 | Use CUDA runtime `cudaMemcpyAsync` per region on Windows while preserving the upstream batch path elsewhere |
| Runtime Python | `vllm/v1/worker/gpu/sample/states.py` | 1 | Force `np.int64` seed generation for the full signed 64-bit range on Windows |
| Runtime Python | `vllm/v1/utils.py` | 1 | `import uvloop` → try/except with asyncio fallback |
| TQ integration | `vllm/config/cache.py` | 1 | Add 6 TQ entries to `CacheDType` literal (alongside upstream's 4 `turboquant_*`) |
| TQ integration | `vllm/utils/torch_utils.py` | 1 | Map our 6 TQ dtypes to `torch.uint8` |
| TQ integration | `vllm/v1/attention/backends/triton_attn.py` | 4 | Add our 6 to `supported_kv_cache_dtypes`; hook `forward()` decode + `do_kv_cache_update` encode for our dispatch |
| **NEW file** | `vllm/v1/attention/ops/multi_turboquant_kv.py` | new | 295-line dispatch helper |
| **NEW file** | `cutlass-windows.patch` | new | 5-file CUTLASS 4.4.2 patch applied automatically by CMakeLists.txt |
| **NEW file** | `vllm-flash-attn-cutlass-windows.patch` | new | 5-file vendored CUTLASS patch applied automatically by `vllm_flash_attn.cmake` |

## Dropped from v4 (now obsolete upstream)

| Old patch hunk | Status |
|---|---|
| `csrc/topk.cu` — designated initializer fix in `get_params()` | Upstream removed `FastTopKParams` entirely; the replacement `persistent_topk` uses explicit member assignment, our patch is no longer needed |
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` — `fcntl.flock` → `msvcrt.locking` | Upstream rewrote the file; the new implementation uses ABC + contextlib and no longer touches `fcntl` |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` — `kv_cache_dtype.startswith("fp8")` assert | Upstream now uses `is_quantized_kv_cache(kv_cache_dtype)`, a strictly more general check |

---

## Categorized notes

### CMakeLists.txt — MSVC compiler flags (new in v5)

PyTorch 2.11.0 and CUTLASS 4.4.2 surfaced two new MSVC-specific issues
that didn't exist in earlier toolchains. The patch adds these flags
inside the existing `if(MSVC)` block:

```cmake
add_compile_definitions(WIN32_LEAN_AND_MEAN)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /Usmall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Usmall")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:__cplusplus")
```

- **`/Usmall` + `WIN32_LEAN_AND_MEAN`**: Windows SDK's `rpcndr.h`
  defines `small` as a macro (`typedef char small`). PyTorch 2.11.0's
  `c10/cuda/CUDACachingAllocator.h` line 105 uses `small` as a
  parameter name (`StreamSegmentSize(cudaStream_t s, bool small, size_t sz)`).
  Without these flags, the preprocessor turns that into `bool char,
  size_t sz` and nvcc rejects it as a syntax error.
- **`/Zc:__cplusplus`**: MSVC defaults to reporting `__cplusplus=199711L`
  even when you pass `-std=c++20`. CUTLASS 4.4.2's `platform.h` gates
  `is_unsigned_v` / `is_integral_v` / etc. behind
  `__cplusplus >= 201703L`, so without this flag those aliases are
  invisible and `exmy_base.h` errors with "is_unsigned_v: is not a
  member of cutlass::platform".

### CMakeLists.txt — auto-apply CUTLASS patches

After `FetchContent_MakeAvailable(cutlass)` we add an `execute_process`
block guarded by `if(MSVC AND EXISTS ...)` that runs `git apply` on
`cutlass-windows.patch` inside the just-fetched `${cutlass_SOURCE_DIR}`.
The `--check` pass ensures we don't error if the patch has already been
applied (e.g. on rebuild). The same pattern is mirrored in
`cmake/external_projects/vllm_flash_attn.cmake` for the vendored
CUTLASS submodule inside `vllm-flash-attn`.

### envs.py — flashinfer sampler default (new in v5)

Upstream defaults `VLLM_USE_FLASHINFER_SAMPLER=True`. That triggers an
unconditional `from vllm.v1.attention.backends.flashinfer import
FlashInferBackend` in `vllm/v1/sample/ops/topk_topp_sampler.py:40`,
which in turn does `import flashinfer` at module load. The flashinfer
package has no Windows wheel, so the import raises
`ModuleNotFoundError` and `LLM()` construction fails before any kernel
ever runs.

The patch flips the lambda so the default on `win32` is `False`,
making the Triton sampler the default path on Windows:

```python
"VLLM_USE_FLASHINFER_SAMPLER": lambda: (
    bool(int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"]))
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ
    else (False if sys.platform == "win32" else True)
),
```

### MSVC keyword operators (`and`, `or`, `not`)

C++ accepts these as alternative tokens in code, but MSVC doesn't
enable them by default in CUDA files. The patch replaces every
instance with `&&`, `||`, `!` in three files. There's no other way —
even `/Zc:alternateTokens` doesn't enable them in nvcc.

### MSVC `__builtin_clz`

GCC/Clang have `__builtin_clz` (count leading zeros). MSVC uses
`_BitScanReverse` or `__lzcnt`. The patch replaces it with portable
bit-twiddling code in `csrc/core/math.hpp`:

```cpp
uint32_t v = num - 1;
v |= v >> 1; v |= v >> 2; v |= v >> 4;
v |= v >> 8; v |= v >> 16;
return v + 1;
```

### MSVC variable template + `__device__`

`csrc/quantization/utils.cuh` uses a variable template
`quant_type_max_v<T>` annotated with `MAYBE_HOST_DEVICE`. MSVC's nvcc
can't apply `__host__`/`__device__` attributes to variable templates.
The patch converts it to a function template with the same signature
(`quant_type_max_v<T>()` at call sites). In v0.21.0 the upstream added
a new call site in `fused_silu_mul_block_quant.cu` that needed the
extra `()` too.

### MSVC nested constexpr lambdas

`csrc/mamba/mamba_ssm/selective_scan_fwd.cu` uses nested `BOOL_SWITCH`
macros that each generate a lambda. The compile-time `bool` template
parameters get captured in lambdas, but MSVC doesn't propagate
`constexpr` through lambda captures the way gcc does.

The patch replaces the nested lambdas with an explicit 8-way `if/else`
dispatch tree calling a helper function template
`selective_scan_fwd_dispatch<..., kIsEvenLen, kHasZ, kVarlen>()` with
all booleans as real template parameters.

### MSVC C1061 — blocks nested too deeply

`csrc/quantization/marlin/generate_kernels.py` and the MOE variant
generate C++ kernel selectors with 700+ branches using `if/else if`
chains. MSVC hits `C1061: compiler limit: blocks nested too deeply`.

The patch generates flat `if` statements instead of `else if`. The
selectors are mutually exclusive by construction (each `if` checks a
unique combination of template parameters), so the flat-`if` form is
semantically equivalent to the chain.

### MSVC designated initializers

CUDA files use C99-style designated initializers like
`__nv_bfloat16_raw{.x = 17376}`. MSVC C++ mode doesn't accept these.
Patch converts to positional initialization `__nv_bfloat16_raw{17376}`.

### MSVC `__int128_t` / `__int64_t`

`__int128_t` doesn't exist in MSVC. The patch replaces it with `int4`
(CUDA's built-in 128-bit vector type) for shared-memory pointers in
`csrc/quantization/activation_kernels.cu`. `__int64_t` is replaced
with the standard `int64_t`.

### MSVC `__attribute__((always_inline))` (new in v5)

`csrc/persistent_topk.cuh` (added in v0.21.0) defines
`FLASHINFER_INLINE` as `inline __attribute__((always_inline)) __device__`.
MSVC's nvcc rejects `__attribute__`. The patch wraps the macro in
`#ifdef _MSC_VER` with `__forceinline __device__` as the MSVC variant.

### MSVC preprocessor-in-macro-arg (new in v5)

`csrc/moe/topk_softplus_sqrt_kernels.cu` calls
`DISPATCH_HASH(use_hash, USE_HASH, { ... })` and embeds
`#ifndef USE_ROCM ... #endif` *inside* the trailing braced argument.
Preprocessor directives inside macro arguments are ill-formed C++ even
with `/Zc:preprocessor`. The patch hoists the `#ifndef USE_ROCM` out
to wrap the entire `DISPATCH_HASH(...)` call instead.

### Distributed: FakeProcessGroup + FileStore

PyTorch Windows builds have `GLOO_HAVE_TRANSPORT_TCP=false` and
`GLOO_HAVE_TRANSPORT_UV=false`. All Gloo init methods (`file://`,
`env://`, `tcp://`, `GLOO_SOCKET_IFNAME`) fail with
`makeDeviceForHostname(): unsupported gloo device`.

The patch wires up `torch.testing._internal.distributed.fake_pg.FakeProcessGroup`
as the backend on Windows, with a `FileStore` for the rendezvous. This
works for single-rank single-GPU operation because vLLM only uses the
distributed API for rank/world_size bookkeeping when running on one GPU.

This is the biggest single Windows compatibility blocker the patch
solves.

### Custom safetensors reader (weight_utils.py)

The biggest *runtime* fix. Windows uses commit charge (RAM + pagefile)
for any process allocation. On systems with the pagefile disabled or
small, the upstream safetensors `safe_open` mmap path fails with
`OSError 1455 (The paging file is too small for this operation to
complete)` mid-load. Eager file reads and `torch.empty(nbytes)` paths
also fail because all need ~1.5 GB of contiguous committed memory for
the embedding tensor.

The patch adds `_windows_safetensors_iterator` which:
1. `numpy.memmap` the safetensors file (file-backed, no commit charge)
2. For tensors >256 MB, allocate the destination directly on the GPU
   and stream bytes in 64 MB chunks (avoids the CPU staging buffer)
3. For smaller tensors, return a zero-copy `torch.from_numpy` view
   over the mmap

Works without a pagefile.

### TritonAttention Multi-TurboQuant dispatch

Two hooks added to `vllm/v1/attention/backends/triton_attn.py`:

1. **`do_kv_cache_update`**: if `kv_cache_dtype` is one of our 6 TQ
   methods, call `tq_write_kv_cache` (encode + scatter packed bytes
   to the uint8 cache). Otherwise fall through to standard
   `triton_reshape_and_cache_flash` / per-token-head path.
2. **`forward`**: same check; if TQ, call `tq_decode_active_blocks`
   to gather and decode the active blocks into a compact fp16 cache,
   remap `block_table` to compact indices, then run the standard
   `unified_attention` Triton kernel on the compact cache.

The dispatch helper lives in
`vllm/v1/attention/ops/multi_turboquant_kv.py`. It imports from the
`multi_turboquant` package at function-call time to keep import
overhead zero for non-TQ runs.

The upstream `turboquant_*` variants are handled by the new
`TurboQuantBackend` in `vllm/v1/attention/backends/turboquant_attn.py`
(unmodified by this patchset). Backend selection happens in
`vllm/platforms/cuda.py`: `supports_kv_cache_dtype` on
`TurboQuantBackend` matches any string starting with `turboquant_`, so
our `turboquant25`/`turboquant35` (no underscore after `turboquant`)
keep landing on the TritonAttention path with our hooks.

---

## What's NOT patched (known limitations)

- **NCCL** — No Windows support upstream. No multi-GPU tensor parallelism.
- **FlashInfer** — No Windows wheel; the sampler defaults to Triton.
- **FlashAttention 3** — Has MSVC-incompatible PTX macros. FA2 works fine.
- **FlashAttention 4 (CuteDSL)** — Needs `nvidia-cutlass-dsl`, no Windows wheel.
- **`nvidia-cutlass-dsl`** — No Windows wheel. Used by FA4 / QuACK; FA2 path is fine.
- **DeepGEMM** - skipped on Windows because the current build helper invokes `g++`; callers fall back when the package is unavailable.
- **Remote/distributed KV tiers** - P2P, NIXL, GDS, object-store, and LMCache remote features are not part of the validated Windows release. The shipped addition is local CPU/filesystem tiering only.
- **Filesystem cache quota** - no automatic byte limit or cleanup; users must manage the selected cache root.
- **Cooperative TopK** - skipped on Windows; sparse attention falls back to the persistent TopK path when the op is absent.
- **fastsafetensors** — Linux-only (`io_uring`); our custom Windows safetensors reader replaces it.
- **Gloo distributed** — Worked around with `FakeProcessGroup`.
- **Our 6 Multi-TurboQuant methods' throughput** — Encode/decode runs in PyTorch (no fused Triton kernel). Memory savings real, throughput cost is the trade-off. Upstream `turboquant_*` variants don't pay this cost.

---

## Patch history

Older patches against earlier vLLM versions are kept for legacy
installs:

| Patch file | Base vLLM | Status |
|---|---|---|
| `vllm-windows-v12.patch` | v0.33.0 | **current** |
| `vllm-windows-v10.patch` | v0.27.1 | previous; final v0.27.1 patch |
| `vllm-windows-v9.patch` | v0.25.1 | stale; final v0.25.1 patch |
| `vllm-windows-v8.patch` | v0.24.0 | stale; final v0.24.0 patch including the sampling/KV-copy hotfixes |
| `vllm-windows-v7.patch` | v0.24.0 | stale; original wheel omitted generated FlashAttention Python files |
| `vllm-windows-v6.patch` | v0.23.0 | stale; still works for v0.23.0 builds |
| `vllm-windows-v5.patch` | v0.21.0 | stale; still works for v0.21.0 builds |
| `vllm-windows-v4.patch` | v0.19.1 | stale; still works for v0.19.1 builds |
| `vllm-windows-v3.patch` | v0.19.0 | stale; for v0.19.0 builds |
| `vllm-windows-v2.patch` | v0.17.1 | stale; for v0.17.1 builds |
| `vllm-windows.patch` | v0.14.1 | stale; for v0.14.2 legacy install |
| `cutlass-windows-v0.21.0.patch` | CUTLASS v4.4.2 | legacy standalone copy; bundled into v5 through v9 patchsets |
| `vllm-flash-attn-cutlass-windows-v0.21.0.patch` | vllm-flash-attn `f5bc33cfc` submodule | legacy standalone copy; bundled into v5 through v9 patchsets |
