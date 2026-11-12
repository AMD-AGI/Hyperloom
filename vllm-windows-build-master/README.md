# vllm-windows-build

> **Latest: vLLM 0.33.0** for Python 3.13 / CUDA 13.0 ([release](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu130)). `install.bat` on this branch installs it.
>
> **Python 3.14 / CUDA 13.2:** the same version is also built for Python 3.14 ([release](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu132-py314); CPU TorchAudio).
>
> **Older versions** (0.29.0, 0.27.1 and earlier) remain available on the [Releases page](https://github.com/aivrar/vllm-windows-build/releases);
> see [choosing a release](docs/releases.md).

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue)
![vLLM: v0.33.0](https://img.shields.io/badge/vLLM-v0.33.0-orange)
![CUDA: 13.0](https://img.shields.io/badge/CUDA-13.0-76B900)
![Python: 3.13](https://img.shields.io/badge/Python-3.13-3776AB)
![PyTorch: 2.14](https://img.shields.io/badge/PyTorch-2.14.0-EE4C2C)
![Triton: 3.8](https://img.shields.io/badge/Triton-3.8.0-red)
![GPU: SM 7.5-12.0](https://img.shields.io/badge/GPU-SM%207.5%20%E2%86%92%2012.0-76B900)
![Multi-TurboQuant](https://img.shields.io/badge/Multi--TurboQuant-6%20methods-purple)
![+ Upstream TurboQuant](https://img.shields.io/badge/+%20Upstream%20TurboQuant-4%20variants-purple)

## v0.33.0 Windows release

The native Windows **vLLM 0.33.0 + CUDA 13.0 / Python 3.13** wheel is published
as [`v0.33.0-win-cu130`](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu130) and was GPU-validated on an RTX 3090. It
includes kernels for RTX 20xx/Turing (SM 7.5), RTX 30xx (SM 8.6), RTX 40xx
(SM 8.9), and Blackwell (SM 12.0), FlashAttention 2, the optimized opt-in Rust
frontend, Multi-TurboQuant, and opt-in Windows CPU/filesystem prompt-KV tiers.
The 243,466,236-byte wheel served a real Qwen3.5 9B GPTQ model with
compilation and CUDA graphs on an RTX 3090 and passed a CPU KV-offload restore.
Its SHA-256 is:

```text
28c45d24e246d515d1ae87e8ca286cf7d3b1dadbd8aec4feb685b01b93627321
```

The same version is also built for **Python 3.14 / CUDA 13.2**:
[`v0.33.0-win-cu132-py314`](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu132-py314). That build uses the CPU TorchAudio wheel;
everything else matches.

> **Blackwell FP8/NVFP4 users:** SM 12.0 code in this wheel does not include
> native Blackwell FP4 acceleration. 0.33.0 includes the issue #16 FP8 Marlin
> fallback fixes (automatic Marlin selection was verified with SM 12.0 metadata
> on an RTX 3090), but it has not been run on Blackwell hardware. See the
> [Blackwell notes](docs/troubleshooting.md#blackwell-fp8-nvfp4) if a model
> still fails.

For RTX 20xx users, the launcher provides `--turing-compat`,
which selects `TRITON_ATTN`, float16 KV cache, 32-token blocks, and eager mode.
See [the Turing usage notes](docs/usage.md#rtx-20xx--turing-sm-75), the
[build record](docs/v0.33.0-build-candidate.md), and
[issue #14](https://github.com/aivrar/vllm-windows-build/issues/14).

**Native Windows build of vLLM 0.33.0 - no WSL, no Docker, no Linux VM.**

> **Current build (cu130 / Python 3.13 / SM 7.5-12.0):** updated to
> **vLLM 0.33.0** with RTX 20-/30-/40-/50-series kernels
> (`TORCH_CUDA_ARCH_LIST=7.5;8.6;8.9;12.0`), PyTorch 2.14.0+cu130, CUDA 13.0,
> the Windows API-server/Rust fixes, Turing launcher compatibility profile,
> Multi-TurboQuant, and opt-in CPU/filesystem KV offload.

Ships with **10 KV cache compression methods**: the 6 Multi-TurboQuant
methods (`isoquant`/`planarquant`/`turboquant25/35`) plus the 4
upstream TurboQuant variants that landed in v0.19.2rc0 (`turboquant_k8v4`,
`turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`).

vLLM is the most popular open-source LLM serving engine, but it
officially only supports Linux. This repo provides a **pre-built wheel**
(just download and install), the current v12 source patch, historical
patchsets, and the v0.33.0 build workflow for compiling vLLM natively on
Windows with full CUDA acceleration, Triton support, and Multi-TurboQuant
integration.

## Releases

| Release | vLLM | PyTorch | Triton | KV compression | Download |
|---|---|---|---|---|---|
| **v0.33.0-win-cu130 (latest, Python 3.13)** | 0.33.0 | 2.14.0+cu130 | 3.8.0 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8; SM 7.5/8.6/8.9/12.0, issue #16 Marlin fallback, CPU/filesystem KV offload, optimized Rust frontend | [Release page](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu130) |
| v0.33.0-win-cu132-py314 (Python 3.14) | 0.33.0 | 2.14.0+cu132 | 3.8.0 | Same as above; **CPU TorchAudio** | [Release page](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu132-py314) |
| v0.29.0-win-cu132-py314 (regular, Python 3.14) | 0.29.0 | 2.13.0+cu132 | 3.7.1 | Retains Windows KV features; **CPU TorchAudio**; [validation](https://github.com/aivrar/vllm-windows-build/blob/v0.29.0-win-cu132-py314/docs/v0.29.0-cu132-py314.md) | [Release page](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu132-py314) |
| v0.29.0-win-cu130-rc1 (prerelease, Python 3.13) | 0.29.0 | 2.13.0+cu130 | 3.7.1 | Retains Windows KV features; issue #16 Marlin fallback fixes | [Release page](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu130-rc1) |
| v0.27.1-win-cu130 (older release) | 0.27.1 | 2.13.0+cu130 | 3.7.1 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8; **SM 7.5/8.6/8.9/12.0, Qwen3.5, Turing profile, CPU/filesystem KV offload, optimized Rust frontend + tool parser** | [Release page](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.27.1-win-cu130) |
| v0.26.0-win-cu128 (older release) | 0.26.0 | 2.11.0+cu128 | 3.6.0 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8; SM 7.5/8.6/8.9/12.0, Turing profile, CPU/filesystem KV offload, Rust frontend + tool parser | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.26.0-win-cu128) |
| v0.25.1-win-cu128 | 0.25.1 | 2.11.0+cu128 | 3.6.0 | Multi-TurboQuant + upstream TurboQuant + fp8; CPU/filesystem KV offload, SM 8.6/8.9/12.0 | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.25.1-win-cu128) |
| v0.24.0-win-cu128 | 0.24.0 | 2.11.0+cu128 | 3.6.0 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8 - Python 3.13, Blackwell sm_120, Rust frontend + Rust tool parser included | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.24.0-win-cu128) |
| v0.23.0-win-cu128 | 0.23.0 | 2.11.0+cu128 | 3.6.0 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8 - **Python 3.13, Blackwell sm_120, Rust frontend included** | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.23.0-win-cu128) |
| v0.21.0-win-cu128 | 0.21.0 | 2.11.0+cu128 | 3.6.0 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8 — **Python 3.13, Blackwell sm_120** | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.21.0-win-cu128) |
| v0.21.0-win | 0.21.0 | 2.11.0+cu126 | 3.6.0 | Multi-TurboQuant (6) + upstream TurboQuant (4) + fp8 (Python 3.10) | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.21.0-win) |
| v0.19.1-win | 0.19.1 | 2.10.0+cu126 | 3.6.0 | Multi-TurboQuant (6 methods) + fp8 | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.19.1-win) |
| v0.19.0-win | 0.19.0 | 2.10.0+cu126 | 3.6.0 | Multi-TurboQuant (6 methods) + fp8 | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.19.0-win) |
| v0.17.1-win | 0.17.1 | 2.10.0+cu126 | 3.6.0 | TurboQuant (2 recipes) | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.17.1-win) |
| v0.14.2-win | 0.14.2 | 2.9.1+cu126 | n/a | fp8 only | [Download](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.14.2-win) |

### What's new in v0.33.0

- **Upstream vLLM 0.33.0** with PyTorch 2.14.0, Triton 3.8.0, and CUDA 13.0
  (Python 3.13) or CUDA 13.2 (Python 3.14), with native Windows CUDA, serving,
  Rust, FlashAttention, and KV-offload support carried forward.
- **Issue #16 FP8 Marlin fallback fixes** are included (first released in
  0.29.0); native Blackwell FP4 is still not available on Windows.
- **Rust:** upstream removed the Rust tool-parser extension. The optimized
  `vllm-rs` frontend remains, opt-in through `VLLM_USE_RUST_FRONTEND=1`.
- **Windows fixes new in 0.33.0:** the sleep-mode allocator no longer raises
  during worker shutdown, and the CPU KV-offload region stays pinned.
- **Validated** on an RTX 3090 from the exact installed wheels: real-model
  serving with compilation and CUDA graphs, six KV-cache formats, the FP8
  Marlin regression, CPU-ARC restore, and a clean-folder `install.bat` +
  `launch.bat` run for both Python builds.

See the [v0.33.0 build record](docs/v0.33.0-build-candidate.md) for details.

### What's new in v0.27.1 (older release)

- **Upstream vLLM 0.27.1** and its CUDA 13.0 / PyTorch 2.13.0 / Triton 3.7.1
  baseline, with native Windows CUDA, serving, Rust, and FlashAttention
  packaging carried forward.
- **Qwen3.5 GPTQ serving verified** from the final installed wheel with
  AutoGPTQ/Marlin and a local 9B checkpoint on the RTX 3090.
- **New stable-libtorch layout**: both CUDA and MoE stable-ABI modules are
  built and packaged alongside the allocator, filesystem helper, spinloop,
  FlashAttention 2, Rust frontend, and Rust tool parser.
- **Prompt caching verified**: three 984-token requests produced 1,056 cached
  prompt-token hits out of 2,952 total and completed without request errors.
- **Windows tiered KV cache revalidated**: CPU LRU, forced filesystem eviction
  and restore, and fresh-process persistent reuse each restored 1,056 of 1,452
  prompt tokens with exact generated token IDs.
- **Regression suite**: 80 focused Windows KV/file-mapping/shared-region and
  hybrid-block tests passed, with one expected platform-gated skip.
- **Known Windows build limit**: a few specialized SM120-only CUTLASS FP8 and
  NVFP4/MoE entry points are omitted because their aligned parameter ABI is
  incompatible with MSVC x64. Other SM120 kernels remain in the wheel, but
  automatic FP8 backend selection can still choose an omitted implementation.
  See the [issue #16 workaround](docs/troubleshooting.md#blackwell-fp8-nvfp4).

See the [v0.27.1 release build record](docs/v0.27.1-build-candidate.md) for
the exact wheel hash, measured results, and scope.

### What's new in v0.26.0 (older release)

- **Upstream vLLM 0.26.0** with native Windows CUDA/MSVC, serving, Rust, and
  FlashAttention packaging carried forward from the previous release.
- **SM 7.5/Turing coverage** added to the multi-architecture wheel alongside
  SM 8.6, 8.9, and 12.0.
- **RTX 20xx launcher compatibility** via `--turing-compat`, which forwards
  `TRITON_ATTN`, float16 KV cache, 32-token blocks, eager execution, and the
  RTX 2080 Ti-tested memory profile: 0.89 GPU utilization, one sequence, and
  2,048 batched tokens. Explicit command-line values still win.
- **GPU numbering and console fixes**: `launch.bat` sets
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` and UTF-8 output settings.
- **Windows build fixes**: hybrid-KV block-table fallback, Rust IPC polling,
  and the Linux-only CuTe FA4 target are handled for native Windows.
- **Validation**: direct Qwen3-14B AWQ serving passed on the RTX 3090;
  integrated launcher serving passed on the RTX 3060; 23 tests and 19
  subtests passed.
- **RTX 2080 Ti confirmation**: the issue #14 reporter loaded
  `cyankiwi/Qwen3.5-9B-AWQ-4bit` with the v0.26.0 wheel and the compatibility
  profile. Final request-level confirmation remains pending.
- **Known limits**: direct `.gguf` files are not supported by this launcher;
  use a Hugging Face-format model with `config.json` and compatible weights.

See the [v0.26.0 release build record](docs/v0.26.0-build-candidate.md) for
the wheel hash, test commands, and release scope.

### What's new in v0.25.1 (older release)

- **vLLM v0.25.1 base** - carries forward the native Windows CUDA/MSVC,
  serving, Rust, FlashAttention packaging, and Multi-TurboQuant work.
- **Native Windows prompt-KV offload** - the launcher now offers opt-in
  CPU LRU/ARC modes and tiered RAM + filesystem/NVMe LRU/ARC modes. The
  filesystem tier can reuse matching prompt blocks after a process restart.
- **Windows tiering fixes** - adds shared-file `mmap` support, binary-safe
  filesystem I/O, safe cache namespaces for absolute Windows model paths,
  native DMA for file-backed mmap restores, and a non-Triton Torch block-table
  fallback. The native `fs_io_C.pyd` helper is included in the wheel.
- **Safe defaults** - KV offload is disabled by default. Filesystem modes
  require an explicit cache root and have no automatic disk quota; monitor or
  clean that directory yourself. `launch.bat` fixes `PYTHONHASHSEED=0` so the
  same on-disk cache namespace can be found across restarts.
- **AMD64 dependency metadata fixed** - `llguidance` and `xgrammar` now install
  through the wheel metadata on Windows, whose machine name is `AMD64`.
- **Validated on RTX 3090** - the installed final wheel passed a 24-case
  bidirectional GPU/CPU offload matrix, CPU LRU/ARC reuse, forced filesystem
  eviction and restore, cross-process persistent reuse, and exact-output model
  checks with the local Qwen3-14B AWQ model. A disk restore reused 1,440 prompt
  tokens and produced the same generated token IDs as the baseline.
- **Scope is intentionally smaller than [LMCache](https://github.com/LMCache/LMCache)** -
  this adapts the useful local RAM/filesystem tiering ideas to vLLM's own
  offloading framework; it is not a Windows port of LMCache's remote, P2P,
  NIXL, GDS, or distributed cache stack.

The Windows tiering work was informed by LMCache's published architecture and
feature set, but it does not copy or bundle LMCache code. Thanks to the LMCache
maintainers for making that work public. The design comparison and remaining
roadmap are documented in
[LMCache-inspired KV cache expansion](docs/lmcache-inspired-windows-kv-cache.md).

Use `launch.bat --help` or see [docs/usage.md](docs/usage.md#experimental-kv-offload)
for opt-in examples and limitations.

### What's new in v0.24.0

- **vLLM v0.24.0 base** - carries forward the Windows CUDA/MSVC fixes from
  v0.23.0 and adds the upstream v0.24 engine, model, serving, parser, and
  security fixes.
- **Rust tool parser packaged on Windows** - v0.24 adds a PyO3
  `_rust_tool_parser.pyd` beside `vllm-rs.exe`; the wheel now includes and
  smoke-tests both Rust artifacts.
- **Windows-only skips for Linux-only v0.24 extensions** - QuTLASS,
  cooperative TopK, and DeepGEMM are skipped on Windows. Their callers are
  guarded or fall back to existing paths, and the expected missing QuTLASS
  warning is suppressed on Windows.
- **v0.24 third-party CUDA helper packages included** - the wheel carries
  the generated `vllm.third_party.triton_kernels` and `fmha_sm100` files
  copied by the build.
- **Qwen3-VL FlashAttention packaging fixed** - the rebuilt wheel includes
  the generated rotary, Triton rotary, and CuteDSL Python modules that the
  upstream editable-build copy step dropped on Windows. The v8 patch makes
  that copy path platform-independent, and the assembler now rejects an
  incomplete FlashAttention payload.
- **Windows request sampling fixed** - NumPy now generates the full-range
  internal request seed explicitly as `int64`, avoiding the 32-bit C `long`
  default that caused issue #10's follow-up error on 64-bit Windows.
- **Smoke tested from the final wheel** - installed the assembled wheel,
  imported `vllm`, the stable libtorch CUDA extensions, FA2, `spinloop`,
  `cumem_allocator`, `_rust_tool_parser`, OpenAI API server / DP supervisor
  modules, ran `vllm --help` and `vllm serve --help`, verified
  `VLLM_USE_RUST_FRONTEND=1` resolves `vllm-rs.exe`, and verified the
  intentionally skipped DeepGEMM/cooperative-TopK paths report unavailable.
- **Portable installer repairs Triton runtime compilation support** -
  `install.bat` now adds `Python.h` and `python313.lib` to the embedded
  Python tree, which Triton needs when it JIT-compiles CUDA helpers for
  models such as Qwen3.5. `launch.bat` runs the same repair check before
  starting the server, and both scripts pin Triton to its bundled CUDA
  helper toolkit when present. `launch.bat` also no longer sets the
  removed `VLLM_ATTENTION_BACKEND` environment variable.
- **Installer integrity and repair hardened** - Python, NuGet, and bootstrap
  downloads plus both project release wheels are pinned by exact size and SHA-256.
  Wheels download to a temporary file, stale/truncated local wheels are
  replaced automatically, and the install marker records the verified wheel
  hash only after CUDA, Rust, Qwen3.5/Qwen3-VL, and FlashAttention checks pass.
- **Older Windows PowerShell bootstrap fixed** - pre-Python verification and
  extraction use direct .NET APIs, not `Get-FileHash` or `Expand-Archive`.
  Windows PowerShell 3 or newer is required for the downloader.
- **Concurrent launcher requests fixed** - one dispatcher now owns
  `engine.step()` and routes outputs by request ID. Streaming and
  non-streaming requests can no longer consume each other's engine output.

### What's new in v0.23.0

- **vLLM v0.23.0 base** - carries forward the Windows CUDA/MSVC fixes from
  v0.21.0 and adds the upstream v0.22/v0.23 bug fixes and frontend work.
- **Rust frontend builds on Windows** - added `protoc` support to the build
  flow, made the Rust managed-engine process handling platform-aware, gated
  Unix-only listener/signal paths, disabled mimalloc on Windows to avoid an
  MSVC CRT link mismatch, and fixed `VLLM_RUST_FRONTEND_PATH=auto` to resolve
  `vllm-rs.exe`.
- **Wheel packaging fixed for `uv`** - the v0.23.0 wheel is assembled with
  proper CSV `RECORD` generation, so comma-containing fused-MoE config
  filenames install correctly with `uv`.
- **Smoke tested** - installed with `uv`, imported `vllm`, `_C`,
  `_C_stable_libtorch`, `_moe_C`, `spinloop`, `cumem_allocator`, FA2, the
  OpenAI API server / DP supervisor import surface, `vllm --help`, and
  `vllm serve --help`; also verified `VLLM_USE_RUST_FRONTEND=1` resolves the
  packaged `vllm-rs.exe`.

### What's new in v0.21.0-cu128 (historical Blackwell build)

The notes in this section describe v0.21.0. For current Blackwell
FP8/NVFP4 restrictions, see
[Blackwell troubleshooting](docs/troubleshooting.md#blackwell-fp8-nvfp4).

This is a rebuild of the same vLLM 0.21.0 source for **RTX 50-series
(Blackwell)** plus a set of Windows API-server fixes. Thanks to
[@Dhrhciebcy](https://github.com/aivrar/vllm-windows-build/issues/4) for
the report that surfaced both the Blackwell gap and the API-server bug.

- **Blackwell (sm_120) support** — built with `TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0`
  on **CUDA 12.8 + PyTorch 2.11.0+cu128 + Python 3.13**, so the wheel carries
  sm_86 / sm_89 / **sm_120** kernels (verified with `cuobjdump`). The older
  `v0.21.0-win` wheel (cu126, sm_86 only) fails on a 5090 with
  `no kernel image is available for execution on the device` — that's a
  compute-capability gap, not a Python-version problem.
- **The OpenAI API server now works on Windows.** Previously only the
  in-process `LLM()` path worked; `vllm serve` / `api_server` crashed. Four
  Windows-only bugs fixed: (1) bare `import uvloop` (Unix-only) in six
  server/entrypoint modules → falls back to `asyncio`; (2) `wait_for_engine_startup()`
  registered process *sentinels* (Windows HANDLEs, not sockets) with a
  `zmq.Poller` → `not a socket`, now skipped on win32 with exit-code
  liveness checks; (3) pyzmq needs `loop.add_reader`, absent from the
  Windows Proactor loop → set `WindowsSelectorEventLoopPolicy` (no tornado);
  (4) `loop.add_signal_handler` is `NotImplementedError` on Windows → falls
  back to `signal.signal`. **winloop is no longer needed.**
- **Two Blackwell-only kernels are skipped on Windows** (they don't compile
  under MSVC and aren't usable here anyway): **QuTLASS** (NVFP4/MXFP4
  microscaling quant — uses GCC inline-PTX `asm`) and the **MiniMax**
  multi-GPU all-reduce RMS fusion (needs real multi-GPU comm; Windows uses
  `FakeProcessGroup`). Their vLLM callers are `hasattr`-guarded, so FP4 and
  MiniMax just degrade gracefully. Everything mainstream — FP16/BF16, AWQ,
  GPTQ/Marlin, FP8, and all 10 KV-cache compression methods — is unaffected.
- **Dependency note:** vLLM gates `llguidance` and `xgrammar` on
  `platform_machine == "x86_64"`, but Windows reports `AMD64`, so pip
  silently skips them and vLLM then fails to import. `install.bat` installs
  them explicitly; if installing manually, run
  `pip install "llguidance>=1.7.0,<1.8.0" "xgrammar>=0.2.0,<1.0.0"`.

### What's new in v0.21.0

- **vLLM v0.21.0 base** — 1,157 upstream commits since v0.19.1, including
  the new native TurboQuant attention backend (PR #38479), DeepGEMM
  extension, fastsafetensors prefetch helpers, and v1 engine maturity.
- **PyTorch 2.11.0 + CUDA 12.6** (was 2.10.0). New compiler flags needed
  for MSVC: `/Usmall` to dodge the `rpcndr.h` macro that collides with
  PyTorch's new `bool small` parameter name, and `/Zc:__cplusplus` so
  CUTLASS's `is_unsigned_v` (C++17) actually sees the standard `__cplusplus`
  value.
- **Upstream TurboQuant coexists with Multi-TurboQuant** — the patch
  registers our 6 method names alongside upstream's 4 in `CacheDType`.
  Backend dispatch in `vllm/platforms/cuda.py` routes `turboquant_*` to
  the new `TurboQuantBackend`; ours stay on the existing `TritonAttention`
  backend with the dispatch hooks from the v4 patch.
- **CUTLASS 4.4.2 (vendored + vllm-flash-attn submodule) is now patched
  inline** — two MSVC fixes (`memsetDevice` host/device mismatch, four
  `static constexpr dim3 get_block_shape()` violations). The patches
  ship as `cutlass-windows.patch` and `vllm-flash-attn-cutlass-windows.patch`
  inside `vllm-source/`; `CMakeLists.txt` applies them automatically after
  `FetchContent_MakeAvailable`, so no manual intervention.
- **flashinfer is now silently skipped on Windows** — upstream defaults
  `VLLM_USE_FLASHINFER_SAMPLER=True`, which then unconditionally `import
  flashinfer` (no Windows wheel). The patch flips the default to `False`
  on `win32` so the Triton fallback is used transparently.
- **Smoke-tested end-to-end on RTX 3090, Qwen3-14B-AWQ-4bit** with both
  `kv_cache_dtype=auto` (9.7 tok/s) and `turboquant35` (0.73 tok/s,
  consistent with v0.19.x).

### Carried over from v0.19.x

- **Multi-TurboQuant integration**: 6 KV cache compression methods
  (`isoquant3`, `isoquant4`, `planarquant3`, `planarquant4`,
  `turboquant25`, `turboquant35`) with real uint8 packed storage —
  **2× more KV cache tokens** at the same `gpu_memory_utilization`.
- **Custom Windows safetensors reader**: numpy memory-mapping +
  chunked GPU streaming. Loads a 14B model in seconds and works on
  systems with the Windows pagefile disabled.
- **All 140 CUDA targets compile clean** with MSVC 2022 + CUDA 12.6 +
  Ninja. 36 source files patched + 3 new files (the TQ dispatch helper
  and the two CUTLASS patches).
- **Tests included**: end-to-end validation suite that proves each
  TQ method actually compresses (not a placebo) and each one produces
  unique output from FP16.

### Real numbers

Single 24 GB RTX 3090, Qwen3-14B AWQ-4bit, `gpu_memory_utilization=0.5`:

| KV dtype | Cache tokens | Concurrency @ 512 | vs FP16 |
|---|---|---|---|
| `auto` (fp16) | 16,336 | 31.91× | 1.00× |
| `isoquant3`/`4`, `planarquant3`/`4`, `turboquant25`/`35` | **32,672** | **63.94×** | **2.00×** |

### v0.33.0 GPU validation

The final `vllm-0.33.0-cp313-cp313-win_amd64.whl` was installed outside the
source tree and tested on an RTX 3090 with the local
`Qwen3.5-9B-abliterated-GPTQ-4bit` checkpoint. The Python 3.14 wheel passed
the same checks with identical results.

| Check | Result |
|---|---|
| Wheel | 243,466,236 bytes; SHA-256 `28c45d24e246d515d1ae87e8ca286cf7d3b1dadbd8aec4feb685b01b93627321` |
| Runtime | Python 3.13, vLLM 0.33.0, Torch 2.14.0+cu130, CUDA 13.0, Triton 3.8.0 |
| Native payload | Native modules, expected operators, and `vllm-rs 0.33.0` passed |
| Real-model serving | Ordinary and streaming chat with compilation and CUDA graphs |
| Multi-TurboQuant GPU paths | All six local cache formats passed write/decode checks |
| FP8 Marlin (issue #16) | Padded LM head with/without bias: relative L2 0.223% / 0.276% |
| CPU-ARC restore | 1,056 / 1,452 cached tokens; identical generated token IDs |
| Focused regressions | 46 passed, 24 skipped (POSIX-only or CUDA-without-Triton) |
| Portable install | `install.bat` from an empty folder, then `launch.bat` chat request |

### v0.27.1 GPU and KV-tier validation (older release)

The final `vllm-0.27.1-cp313-cp313-win_amd64.whl` was installed outside the
source tree and tested on an RTX 3090 with the local
`Qwen3.5-9B-abliterated-GPTQ-4bit` checkpoint.

| Check | Result |
|---|---|
| Wheel | 239,025,534 bytes; SHA-256 `7c13ed44e94694478bdd4f5fcca23e2d66ba1e8fa9bccad9fddb8651d1b2447b` |
| Runtime | Python 3.13.11, vLLM 0.27.1, Torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1 |
| Native payload | All packaged CUDA, FlashAttention 2, allocator, filesystem, spinloop, and Rust modules imported |
| API requests | 3 succeeded, 0 errors; 984 prompt + 96 generated tokens each |
| Integrated launcher | `/health` and `/v1/chat/completions` HTTP 200; 17 prompt + 24 generated tokens in 2.918 s |
| Multi-TurboQuant GPU paths | All six local cache formats passed write/decode smoke tests |
| Request wall time | 18.787 s cold, then 12.887 s and 12.872 s |
| Prefix cache | 1,056 hits / 2,952 prompt tokens (35.77%) |
| 4K KV capacity | 11.52 GiB; 208,896 tokens; 51.00x reported concurrency |
| Focused regressions | 80 passed, 1 expected skip, 0 failed |

CPU LRU restored 1,056 of 1,452 tokens after clearing the GPU cache and
reduced the measured request from 3.529 s to 1.027 s. Filesystem LRU restored
the same 1,056 tokens after forced RAM eviction (1.244 s cold, 0.840 s
restored), then produced a persistent first-request hit after a fresh engine
start. All restored runs reproduced the cold run's generated token IDs.

Full commands, filesystem counts, and scope are in the
[v0.27.1 build record](docs/v0.27.1-build-candidate.md).

### v0.26.0 GPU validation (older release)

The v0.26.0 release wheel was validated after installation from
`E:\vllm-windows-build-v2\dist-v0.26.0` (389,473,142 bytes; SHA-256
`a9fd2e5752d885a03c28aaa25472b9cdbe8685b4d3ed1a7ce3999803f0179658`). The
native payload includes SM 7.5, 8.6, 8.9, and 12.0 code, the Rust frontend
and tool parser, FlashAttention 2, and the Windows KV-offload helper.

- Direct `vllm serve` with the local Qwen3-14B AWQ checkpoint returned HTTP 200
  on the RTX 3090.
- The integrated `vllm_launcher.py` path returned HTTP 200 on the RTX 3060
  using the Turing-compatible profile.
- The issue #14 reporter confirmed that the release wheel loads
  `cyankiwi/Qwen3.5-9B-AWQ-4bit` on an 11-GB RTX 2080 Ti with
  `--turing-compat`. Request-level confirmation is still pending.
- The release-contract, wheel-content, launcher, and Windows runtime checks
  passed: **23 tests and 19 subtests**.
- The issue #14 verifier/launcher follow-up passed **27 repository-side unit
  and contract tests**; the published wheel's complete contents, 389,473,142
  byte size, and SHA-256 were revalidated unchanged.

That v0.26.0 run did not support the local Qwen3.5-9B GPTQ directory because
of its older Transformers/vLLM config path. Upstream 0.27.1 resolves that
specific limitation, and the model is now part of the current release
evidence. An 11-GB RTX 2080 Ti can still require lower `--max-model-len` and
concurrency during startup; `--turing-compat` applies those limits.

### v0.25.1 KV-offload validation (older release)

The final `0.25.1+cu128` wheel was also tested on the same RTX 3090 with
Qwen3-14B-abliterated-AWQ-4bit. Request times below are focused eager-mode
correctness measurements, not broad benchmark distributions or performance
guarantees.

| Validation path | Cold request | Cached/restore request | Cached prompt tokens | Result |
|---|---:|---:|---:|---|
| Low-level GPU transfer matrix | — | — | — | **24/24 passed** across both directions, ordinary/shared memory, multiple page sizes, block-size factors, and KV groups |
| CPU LRU and CPU ARC | — | — | 1,440 / 1,451 | Both policies reproduced the baseline token IDs exactly |
| Filesystem LRU after forced RAM eviction | 1.359 s | 1.017 s | 1,440 / 1,451 | Exact output |
| Filesystem ARC after forced RAM eviction | 1.365 s | 0.763 s | 1,440 / 1,451 | Exact output |
| Persistent LRU cache, new engine process | — | 0.963 s | 1,440 / 1,451 | Exact output |
| Persistent LRU cache, installed release wheel | — | 0.806 s | 1,440 / 1,451 | Exact output |

The forced-eviction runs grew the filesystem cache from 90 blocks
(235,929,600 bytes) to 272 blocks (713,031,680 bytes), exceeding the configured
102-block / 256 MiB RAM tier before restore. That verifies the measured hits
came from the filesystem tier rather than a surviving RAM entry.

Full compression benchmarks → [docs/benchmarks.md](docs/benchmarks.md)

KV-offload evidence and scope →
[docs/lmcache-inspired-windows-kv-cache.md](docs/lmcache-inspired-windows-kv-cache.md)

---

## Quick Start

For the recommended portable install, download this release's
**Source code (zip)** (or [choose another release](docs/releases.md)),
extract it, and run `install.bat` followed by `launch.bat`. The manual
commands below are for the default 0.33.0 Python 3.13 release.

### Option A — Pre-built wheel (no compiler needed)

Download
**[vllm-0.33.0-cp313-cp313-win_amd64.whl](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu130)**
and `multi_turboquant-0.1.0-py3-none-any.whl` from the release page,
then:

| Artifact | SHA-256 |
|---|---|
| `vllm-0.33.0-cp313-cp313-win_amd64.whl` | `28C45D24E246D515D1AE87E8CA286CF7D3B1DADBD8AEC4FEB685B01B93627321` |
| `multi_turboquant-0.1.0-py3-none-any.whl` | `5B310E05904B588539D9A8E3374DFA6C160F025F9C2099BA5C7877C79B2FA149` |

```batch
:: Create a Python 3.13 venv
py -3.13 -m venv venv
venv\Scripts\activate

:: Install the exact PyTorch CUDA 13.0 stack used by vLLM 0.33.0
pip install torch==2.14.0 torchaudio==2.11.0 torchvision==0.29.0 ^
    --index-url https://download.pytorch.org/whl/cu130

:: Install Triton for Windows
pip install triton-windows==3.8.0.post29

:: Install the pre-built vLLM wheel
pip install vllm-0.33.0-cp313-cp313-win_amd64.whl

:: Structured-output backends pinned by install.bat
pip install "llguidance>=1.7.0,<1.8.0" "xgrammar==0.2.8"

:: Install Multi-TurboQuant for the 6 KV cache compression methods
pip install multi_turboquant-0.1.0-py3-none-any.whl
```

Or just run **`install.bat`** for a fully self-contained, one-click portable
Python install — it downloads Python 3.13, PyTorch cu130, and the vLLM wheel
itself (no manual download or folder creation needed). If you already have the
`.whl` locally, drop it in `dist-v0.33.0\` next to `install.bat` and the script uses
that instead of downloading.

Fresh installs use Python 3.13.14. Rerunning `install.bat` repairs an existing
portable Python 3.13 install if its dependencies, native/Rust payloads,
FlashAttention modules, marker hash, headers, or import checks are incomplete;
`launch.bat` checks the same release contract before it starts the server.
The installer also downloads the pinned `multi_turboquant-0.1.0` release wheel;
Git is not required for the portable path. Install 0.33.0 into a fresh
folder rather than over an existing 0.27.1 installation.

Both the wheel distribution and `vllm.__version__` report `0.33.0`. The CUDA
compatibility contract is verified separately through Torch 2.14.0+cu130.

### Option B — Build from source

Requires Visual Studio 2022 (Community is fine), CUDA 13.0, Rust, `protoc`,
and a Python 3.13 venv. The build compiles four targets (7.5;8.6;8.9;12.0);
the CUDA compile dominates. Use the worker count that fits the machine and
**do not enable sccache** — it causes intermittent MSVC `cl.exe` crashes
(0xC000001D) on the heavy multi-arch CUDA kernels. The tested machine used
`MAX_JOBS=8`; reduce that value if your available RAM is lower.

Apply `vllm-windows-v12.patch` to upstream vLLM v0.33.0, build the optimized
Rust executable first, then compile the CUDA extensions and assemble the wheel
from the already-built tree with `VLLM_USE_LOCAL_PRECOMPILED=1`. The exact
environment, source commit, commands, and output contract are recorded in
[the release build record](docs/v0.33.0-build-candidate.md).

Full instructions, including all the env vars and prerequisites:
**→ [docs/install.md](docs/install.md)**

---

## Hello world

```python
import os
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

# CUDA + torch DLL search paths
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
os.add_dll_directory(r"C:\path\to\venv\Lib\site-packages\torch\lib")

# Both uvloop and flashinfer fallbacks are baked into the wheel.
# Multi-GPU host? Don't forget CUDA_DEVICE_ORDER + CUDA_VISIBLE_DEVICES
# so vLLM lands on the GPU you actually want.

from vllm import LLM, SamplingParams

llm = LLM(
    model=r"E:\models\Qwen2.5-0.5B-Instruct",
    dtype="float16",
    kv_cache_dtype="auto",        # Fast FP16 baseline
    max_model_len=512,
    gpu_memory_utilization=0.5,
)

outputs = llm.generate(
    ["Explain CUDA streams in three sentences:"],
    SamplingParams(temperature=0.0, max_tokens=32, seed=0),
)
print(outputs[0].outputs[0].text)
```

`auto` is deliberate here: it establishes normal baseline performance. The
first request can include one-time JIT or CUDA-graph setup, so benchmark a
second request in the same process. Add `enforce_eager=True` only when
diagnosing graph/compile compatibility; it disables those optimizations.

For OpenAI-compatible HTTP serving and more usage patterns:
**→ [docs/usage.md](docs/usage.md)**

---

## KV cache compression: 10 methods (6 ours + 4 upstream)

vLLM v0.33.0 on Windows ships with integrated support for **ten** KV cache
compression dtypes. The four `turboquant_*` entries are the new upstream
TurboQuant attention backend (PR #38479, landed in v0.19.2rc0); the six
others come from our [Multi-TurboQuant](https://github.com/aivrar/multi-turboquant)
library and run on the patched `TritonAttention` backend.

| Method | Bits | Family | Calibration | Use case |
|---|---|---|---|---|
| `turboquant_k8v4` | 8.25 / 4.25 | upstream | none | Mixed-precision K/V |
| `turboquant_4bit_nc` | 4.25 | upstream | none | Upstream default |
| `turboquant_k3v4_nc` | 3.25 / 4.25 | upstream | none | More aggressive K |
| `turboquant_3bit_nc` | 3.25 | upstream | none | Most aggressive upstream |
| `isoquant4` | 4.25 | quaternion 4D rotation | none | Quality-first local TQ; offline/memory-first use |
| `planarquant4` | 4.25 | Givens 2D rotation | none | Same memory, simpler transform |
| `isoquant3` | 3.25 | quaternion 4D rotation | none | More aggressive |
| `planarquant3` | 3.25 | Givens 2D rotation | none | More aggressive |
| `turboquant35` | 3.25 | WHT + MSE codebook + QJL | runtime | Calibrated outliers |
| `turboquant25` | 2.25 | WHT + MSE codebook + QJL | runtime | Most compression |

Just pass the method name as `kv_cache_dtype` when constructing an
`LLM` (or `--kv-cache-dtype` to `vllm serve`). Upstream `turboquant_*`
names are routed by `vllm/platforms/cuda.py` to the new
`TurboQuantBackend` (separate cache layout + Triton encode/decode);
ours stay on `TritonAttention` with the dispatch hooks from the v4
patch.

**Trade-off (ours)**: throughput drops ~30-300× with our 6 methods enabled
because the encode/decode runs in PyTorch (no fused Triton kernel yet).
Memory savings are real, throughput cost is the price. Best for
offline / long-context / batch workloads. The upstream variants use
fused Triton kernels and don't pay this cost.  See
**[docs/turboquant.md](docs/turboquant.md)** for the full picture.

---

## What's in the patch

`vllm-windows-v12.patch` is the exact committed delta from upstream tag
`v0.33.0` to the release source. Its SHA-256 is
`cb0d7041f0a3b1c126c1fc89930a861296d62066590d6037ec72ff079eaf4206`.
`vllm-windows-v10.patch` remains the v0.27.1 snapshot and
`vllm-windows-v9.patch` the v0.25.1 snapshot. The breakdown below was written
for v10; v12 carries the same areas forward plus the 0.33-specific changes in
the [build record](docs/v0.33.0-build-candidate.md#port-changes). The
changes touch the Windows build/runtime/Rust frontend and
KV-offload surfaces, and carries the TQ dispatch helper plus two
CUTLASS-vendor patches:

- **Build system**: CMakeLists, cmake/utils, setup.py, requirements/cuda.txt
  (with `/Usmall` + `/Zc:__cplusplus` for MSVC, Linux-only CUDA deps
  commented out, auto-apply of cutlass-windows patches, Windows skips for
  QuTLASS, cooperative TopK, and DeepGEMM)
- **CUDA kernels**: MSVC compatibility for keyword operators,
  designated initializers, `__builtin_clz`, variable templates with
  attributes, nested constexpr lambdas, deeply nested `else if`,
  `__attribute__((aligned))`, `std::isinf`, `__int128_t`, the new
  `persistent_topk.cuh` `__forceinline` swap, `fused_silu_mul_block_quant.cu`
  `quant_type_max_v<T>()` call-syntax, and the `topk_softplus_sqrt_kernels.cu`
  preprocessor-in-macro-arg refactor
- **Runtime Python**: `fcntl` → `msvcrt`, ZMQ IPC → TCP, fork →
  spawn, NCCL → FakeProcessGroup, custom safetensors reader for small
  pagefile systems, `uvloop` fallback, `VLLM_USE_FLASHINFER_SAMPLER`
  default-False on Windows, Windows Rust artifact lookup, optional QuTLASS
  warning suppression, and Windows-safe RAM/filesystem KV offload
- **Multi-TurboQuant integration** (4 + 1 new): 6 new `CacheDType`
  literals, dtype mapping, attention backend dispatch, plus the new
  `vllm/v1/attention/ops/multi_turboquant_kv.py` (295 lines)
- **CUTLASS patches** (2 new files): `cutlass-windows.patch` (5 files
  in CUTLASS 4.4.2: `cuda_host_adapter.hpp` + 4 SM100/SM103 headers
  with `static constexpr dim3` violations) and
  `vllm-flash-attn-cutlass-windows.patch` (5 files in the vendored
  CUTLASS submodule under vllm-flash-attn).

Full per-file breakdown → **[PATCHES.md](PATCHES.md)**

All changes are guarded by `#ifdef _MSC_VER`, `sys.platform == "win32"`,
`if(MSVC ...)`, or similar conditionals. **Zero impact on Linux builds.**

---

## Documentation

| Page | Topic |
|---|---|
| [docs/releases.md](docs/releases.md) | Choose a release, Python version, and matching installer |
| [docs/install.md](docs/install.md) | Install the wheel or build from source |
| [docs/usage.md](docs/usage.md) | Python embedding + HTTP server |
| [docs/turboquant.md](docs/turboquant.md) | Multi-TurboQuant deep dive |
| [docs/benchmarks.md](docs/benchmarks.md) | Historical Multi-TurboQuant benchmarks and current release checks |
| [docs/build.md](docs/build.md) | Patch internals + iterating on builds |
| [docs/architecture.md](docs/architecture.md) | How the integration works |
| [docs/lmcache-inspired-windows-kv-cache.md](docs/lmcache-inspired-windows-kv-cache.md) | LMCache research, implemented native-Windows KV tiers, test evidence, and roadmap |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common errors + fixes |
| [tests/README.md](tests/README.md) | End-to-end test scripts |
| [GitHub Wiki](https://github.com/aivrar/vllm-windows-build/wiki) | Browsable installation, usage, architecture, benchmarks, and KV-offload reference |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Issue reports and pull requests |

---

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 21H2 (x64) | Windows 10 22H2 / Windows 11 |
| GPU | NVIDIA SM 7.5+ (RTX 20/30/40/50, A100, H100) | RTX 3090 / 4090 / A6000 |
| VRAM | 12 GB | 24 GB |
| RAM | 16 GB | 32+ GB |
| CUDA driver | R580+ (CUDA 13.0) | latest |
| Python | 3.13.x | 3.13.14 |
| Compiler (build only) | VS 2022 Community + Win 10 SDK | Same |
| CUDA Toolkit (build only) | 13.0 | 13.0 Update 2 |

For build-from-source, you also need a **Windows pagefile** (system
managed is fine). Without it, large allocations during compilation can
fail. See [docs/troubleshooting.md → OSError 1455](docs/troubleshooting.md#oserror-1455).

---

## Tested with

- RTX 3090 (24 GB, SM 8.6) - v0.33.0 final wheels for Python 3.13 and 3.14: clean-folder `install.bat` and `launch.bat`, Qwen3.5 GPTQ serving with compilation and CUDA graphs, six KV formats, FP8 Marlin regression, and CPU-ARC restore
- RTX 3090 (24 GB, SM 8.6, driver 596.36) - previous v0.27.1 final wheel, native imports, Qwen3.5 GPTQ API serving, prompt-cache metrics, CPU LRU restore, filesystem eviction/restore, and persistent restart reuse
- RTX 3060 (12 GB, SM 8.6) - previous v0.26.0 wheel through the integrated launcher and Turing-compatible profile
- Qwen2.5-0.5B-Instruct, Qwen3-14B-abliterated-AWQ-4bit, and Qwen3.5-9B-abliterated-GPTQ-4bit
- Windows 10 Pro 22H2
- Visual Studio 2022 Community 17.13 (MSVC 14.43)
- CUDA Toolkit 13.0 Update 2 (`nvcc` 13.0.88)
- Python 3.13.11 for the native build/final-wheel tests; portable installer targets Python 3.13.14 (same `cp313` ABI)
- RTX PRO 5000 Blackwell (48 GB, SM 12.0), Windows 11 Pro: the issue #16 reporter confirmed `unsloth/Qwen3.8-27B-NVFP4` serving with compilation, CUDA graphs, and a successful chat request using v0.27.1 plus the [Marlin setting and Python fix](docs/troubleshooting.md#blackwell-fp8-nvfp4). This validates that model's fallback path; native FP4 and other Blackwell configurations remain unvalidated.

### v0.21.0 smoke test (RTX 3090, Qwen3-14B-abliterated-AWQ-4bit)

`kv_cache_dtype=auto` (FlashAttention 2): **20 tokens in 2.06 s,
9.7 tok/s** with `max_model_len=512`, `gpu_memory_utilization=0.92`.
First model load completes in ~24 s after the safetensors cache warms.

`kv_cache_dtype=turboquant35` (Triton attention + Multi-TurboQuant
PyTorch-fallback encode/decode): **20 tokens in 27.39 s, 0.73 tok/s** —
in line with the v0.19.x figure (0.92 tok/s for 5 tokens). All other
Multi-TurboQuant methods (`isoquant3/4`, `planarquant3/4`,
`turboquant25`) should behave the same as in v0.19.x; rerun
`tests/test_tq_real.py` for a full sweep.

### v0.19.1 historical reference

Older Multi-TurboQuant timings on the same hardware (5 decoded tokens,
`gpu_memory_utilization=0.5`):

| Method | Preset | Time (5 tok) | Output tok/s | Status |
|---|---|---:|---:|---|
| `isoquant3` | no_calibration_symmetric | 41.5s | 0.12 | PASS |
| `isoquant4` | no_calibration_quality | 53.0s | 0.09 | PASS |
| `planarquant3` | k_only_planar | 40.5s | 0.12 | PASS |
| `planarquant4` | k_only_planar | 53.0s | 0.09 | PASS |
| `turboquant25` | max_compression | 6.7s | **0.74** | PASS |
| `turboquant35` | speed | 5.4s | **0.92** | PASS |

`turboquant25/35` are ~8× faster than the iso/planar family on the
PyTorch-fallback path. Reproduce with:

```bat
set TQ_METHOD=isoquant3
%VLLM_PYTHON% tests\test_tq_diag.py
```

---

## Limitations

- **Blackwell FP8/NVFP4 requires care.** Native SM120 CUTLASS FP8/NVFP4
  kernels are omitted on Windows. 0.33.0 includes the issue #16 Marlin
  fallback fixes but has not been run on Blackwell hardware; see the
  [Blackwell notes](docs/troubleshooting.md#blackwell-fp8-nvfp4). The fallback
  does not enable native FP4 acceleration or establish a throughput guarantee.
- **Single GPU only.** NCCL doesn't ship with PyTorch on Windows; the
  patch wires up `FakeProcessGroup` for single-rank operation. Multi-GPU
  needs separate vLLM instances + external load balancing.
- **No FlashInfer.** No Windows wheel. The patch defaults
  `VLLM_USE_FLASHINFER_SAMPLER=False` on `win32` so vLLM falls back to
  the Triton sampler transparently.
- **No FlashAttention 3, no FlashAttention 4 (CuteDSL).** FA3 has
  MSVC-incompatible PTX macros, FA4 needs `nvidia-cutlass-dsl` (no
  Windows wheel). FlashAttention 2 works fine.
- **No fastsafetensors.** Linux-only (`io_uring`). The patched
  `weight_utils.py` keeps the in-tree numpy-mmap + chunked-GPU-stream
  reader from v0.19.x for the safetensors path.
- **No DeepGEMM, no Quack, no Tilelang, no TokenSpeed-MLA, no NIXL.**
  None ship Windows wheels; CMake skips DeepGEMM automatically when the
  target arch is below SM 9.0+.
- **Our 6 Multi-TurboQuant methods are still on the PyTorch-fallback
  encode/decode.** Memory savings real, throughput cost real
  (`turboquant35` ≈ 0.73 tok/s on Qwen3-14B). The upstream
  `turboquant_*` variants don't pay this cost — they use the fused
  Triton store/decode kernels that landed in PR #38479.
- **Triton JIT cold-start latency.** First inference with Triton kernels
  (e.g. Qwen3.5 GDN layers) takes ~1-2 minutes for compilation.

---

## Credits

| | |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | The original engine |
| [PyTorch](https://github.com/pytorch/pytorch) | Tensor library + CUDA bindings |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | NVIDIA |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | FA2 kernels |
| [triton-windows](https://github.com/triton-lang/triton-windows) | Triton compiler ported to Windows |
| [Multi-TurboQuant](https://github.com/aivrar/multi-turboquant) | KV cache compression methods (ours) |
| [Upstream TurboQuant](https://github.com/vllm-project/vllm/pull/38479) | TurboQuant attention backend (vLLM PR #38479) |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | GEMM kernels (CUTLASS 4.4.2 with MSVC patches) |
| [TurboQuant paper](https://arxiv.org/abs/2501.06725) | Walsh-Hadamard quantization |

Built with the help of [Claude](https://claude.ai).

---

## License

MIT. See [LICENSE](LICENSE).
