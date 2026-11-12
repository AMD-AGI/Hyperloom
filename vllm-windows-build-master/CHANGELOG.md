# Changelog

## v0.33.0-win-cu132-py314 - 2026-11-08

[Regular release](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu132-py314) for Python 3.14.2, CUDA 13.2 and PyTorch 2.14.0+cu132, built from the same 0.33.0 Windows port source as the Python 3.13 Latest release.

- Same Windows port, fixes and retained features as the Python 3.13 variant below. The 0.29 top-k `FLT_MAX` patch is no longer needed; the FlashAttention Python 3.14 allowlist patch still is.
- The exact wheel passed the same targeted RTX 3090 checks with identical FP8 and KV-format results, plus a pinned CPU-offload restore.
- TorchAudio is CPU-only in this variant; GPU audio processing and audio-model serving were not validated. Native Blackwell FP4 is not included.

See the [Python 3.14 build and validation notes](https://github.com/aivrar/vllm-windows-build/blob/v0.33.0-win-cu132-py314/docs/v0.33.0-cu132-py314.md) and [release selection guide](docs/releases.md).

## v0.33.0-win-cu130 - 2026-11-08

[Regular release](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu130) updating the Windows port to upstream vLLM 0.33.0 for Python 3.13, CUDA 13.0 and PyTorch 2.14.0+cu130. This is the Latest release and the default `master` installer.

- Carries native Windows serving, FlashAttention, ten KV formats, the issue #16 Marlin fallback and opt-in CPU/filesystem cache tiers forward to 0.33.0. Source patch: `vllm-windows-v12.patch`.
- Upstream removed the Rust tool-parser extension; the optimized, opt-in `vllm-rs` frontend remains.
- Fixes a Windows error importing the sleep-mode allocator during worker shutdown (also present in 0.29.0) and keeps CPU KV-offload host memory pinned on Windows.
- The exact wheel passed targeted RTX 3090 real-model serving, KV-format, FP8 Marlin and CPU-offload restore checks. Blackwell hardware was not tested; native Blackwell FP4 is not included.

See the [build record](https://github.com/aivrar/vllm-windows-build/blob/v0.33.0-win-cu130/docs/v0.33.0-build-candidate.md) and [release selection guide](docs/releases.md).

## v0.29.0-win-cu132-py314 - 2026-09-22

[Regular release](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu132-py314) for Python 3.14.2, CUDA 13.2 and PyTorch 2.13.0+cu132. The default `master` installer and the GitHub Latest designation still point to 0.27.1.

- Reuses the byte-identical wheel from the Python 3.14 release candidate. The final tagged source and installer contain the corrected Python 3.14 launcher; the candidate's source ZIP did not.
- Retains native Windows serving, Rust, FlashAttention, ten KV formats and opt-in CPU/filesystem cache tiers. The exact wheel passed targeted RTX 3090 serving and cache checks.
- An [RTX 5090 tester](https://github.com/aivrar/vllm-windows-build/issues/17#issuecomment-5774839645) served a Qwen3.8-27B INT4 checkpoint with FP8 KV and one-token MTP at 131,072 configured context; a 127,987-token input returned the correct hidden label. This is model-specific confirmation, not a speed benchmark.
- TorchAudio is CPU-only in this variant; GPU audio processing and audio-model serving were not validated. Native Blackwell FP4 is not included.

See the [Python 3.14 build and validation notes](https://github.com/aivrar/vllm-windows-build/blob/v0.29.0-win-cu132-py314/docs/v0.29.0-cu132-py314.md) and [release selection guide](docs/releases.md).

## v0.29.0-win-cu130-rc1 - 2026-09-19

[Python 3.13 / CUDA 13.0 prerelease](https://github.com/aivrar/vllm-windows-build/releases/tag/v0.29.0-win-cu130-rc1) updates the Windows port to upstream vLLM 0.29.0 and carries the issue #16 Marlin fallback fixes. The exact wheel passed targeted RTX 3090 real-model serving, KV-format and persistent cache checks. Native Blackwell FP4 remains unavailable. See the [build record](https://github.com/aivrar/vllm-windows-build/blob/v0.29.0-win-cu130-rc1/docs/v0.29.0-build-candidate.md).

## v0.27.1-win-cu130 - 2026-08-21

Release notes: [docs/v0.27.1-release-notes.md](docs/v0.27.1-release-notes.md)

Upstream bump to **vLLM 0.27.1**, CPython 3.13, CUDA 13.0, PyTorch
2.13.0+cu130, Triton Windows 3.7.1.post27, and
`TORCH_CUDA_ARCH_LIST=7.5;8.6;8.9;12.0`.

### New and fixed

- Ported the native Windows runtime, serving, FlashAttention, Rust, sampling,
  and tiered KV-cache changes onto vLLM 0.27.1.
- Packaged vLLM's new stable-libtorch CUDA and MoE modules plus optimized
  `vllm-rs.exe` and `_rust_tool_parser.pyd` artifacts.
- Added MSVC fixes for CUDA 13 attention math/types and guarded the few
  SM120-only CUTLASS FP8/NVFP4 entry points whose aligned by-value parameters
  are incompatible with the Windows x64 ABI.
- Updated the portable installer contract to Torch 2.13.0+cu130, CUDA 13.0,
  Triton Windows 3.7.1.post27, and the 0.27.1 wheel hash.
- Updated the real-model KV harness for vLLM 0.27's `CacheConfig` API.
- Fixed FastAPI request-annotation resolution under Python's postponed
  annotations so JSON embedding/chat requests are not mistaken for a required
  query parameter.
- Exported the exact upstream-v0.27.1 source delta as
  `vllm-windows-v10.patch` (167,335 bytes; SHA-256
  `4b6c9cd543414ef3ed1eb7fcbd7f39ce6be10bdc97d901261977ee905346c988`).

### Validation

- Installed-wheel native imports and Qwen3.5 9B GPTQ OpenAI-compatible API
  serving passed on an RTX 3090.
- Three requests completed without errors and recorded 1,056 prefix-cache hits
  from 2,952 prompt tokens.
- The fixed integrated launcher returned HTTP 200 for health and chat
  completion; the measured request completed in 2.918 s.
- All six Multi-TurboQuant cache formats passed direct GPU write/decode smoke
  tests with the pinned companion wheel.
- CPU LRU, filesystem eviction/restore, and fresh-process persistent reuse each
  restored 1,056 of 1,452 prompt tokens with exact generated token IDs.
- Focused Windows KV and hybrid-block regressions: 80 passed, 1 expected skip,
  0 failed.
- Repository release contracts: 24 passed, 0 failed.
- Final wheel: 239,025,534 bytes; SHA-256
  `7c13ed44e94694478bdd4f5fcca23e2d66ba1e8fa9bccad9fddb8651d1b2447b`.

## v0.26.0-win-cu128 - 2026-07-31

Upstream bump to **vLLM 0.26.0**, targeting CPython 3.13, CUDA 12.8,
PyTorch 2.11.0+cu128, Triton Windows 3.6.0.post26, and
`TORCH_CUDA_ARCH_LIST=7.5;8.6;8.9;12.0`.

### Issue #14 follow-up - 2026-08-02

- Fixed the portable verifier's false failure: wheel metadata is
  `0.26.0+cu128`, while upstream's `vllm.__version__` correctly remains
  `0.26.0`. Both fields are now validated separately.
- Expanded `--turing-compat` with the settings confirmed by the reporter on an
  11-GB RTX 2080 Ti: 0.89 GPU utilization, one sequence, and 2,048 batched
  tokens. Explicit user overrides still take precedence.
- Corrected the launcher guidance for a negative KV-cache budget: raise GPU
  utilization when VRAM is free, or lower context, concurrency, and batched
  tokens.
- Removed GGUF files from model discovery and added an immediate explanation
  that direct `.gguf` input is unsupported, instead of allowing a misleading
  UTF-8/JSON traceback from Transformers.
- The published wheel is unchanged. The reporter confirmed that it loads
  Qwen3.5-9B AWQ on the RTX 2080 Ti; issue #14 remains open pending an actual
  inference response.
- The follow-up passed 27 repository unit/contract tests. Wheel content,
  complete RECORD integrity, exact size, and SHA-256 were revalidated.

### New and fixed

- Added native SM 7.5 coverage for RTX 20xx/Turing while retaining RTX 30xx,
  RTX 40xx, and Blackwell targets.
- Added `--turing-compat` to the Windows launcher. It selects `TRITON_ATTN`,
  a float16 KV cache, 32-token KV blocks, and eager execution for Turing
  compatibility. The individual `--attention-backend`, `--kv-cache-dtype`, and
  `--block-size` options are also forwarded to `LLM`.
- Set deterministic Windows GPU numbering (`CUDA_DEVICE_ORDER=PCI_BUS_ID`) and
  UTF-8 console encoding in `launch.bat`.
- Fixed the Windows hybrid-KV block-table fallback and skipped the Linux-only
  CuTe FA4 target on Windows.
- Fixed Rust mock-engine IPC polling so the v0.26 source builds on Windows
  without `tokio::net::UnixStream`.
- Added local precompiled-extension reuse for wheel assembly after the native
  CUDA build, without downloading an upstream Linux precompiled wheel.

### Validated

- Native wheel imports, Rust frontend/tool parser payloads, and `vllm.exe
  --help` all pass.
- Direct `vllm serve` returned HTTP 200 for local Qwen3-14B AWQ on the RTX 3090.
- The integrated Windows launcher returned HTTP 200 on the RTX 3060 using the
  new compatibility profile.
- The release-contract, launcher, Windows-runtime, bootstrap, artifact, and
  dispatcher tests pass: **23 passed, 19 subtests**.

These GPU checks validate the packaged v0.26 path on Ampere hardware. They do
not replace a final run on the reporter's RTX 2080 Ti. An 11-GB Turing card may
still require lower `--max-model-len` and `--max-num-seqs`; a 65K context request
is not guaranteed to fit. The local Qwen3.5-9B GPTQ checkpoint also remains a
separate Transformers/vLLM config-compatibility issue.

### Artifacts

- Wheel: `vllm-0.26.0+cu128-cp313-cp313-win_amd64.whl`
- Size: `389473142` bytes
- SHA-256:
  `a9fd2e5752d885a03c28aaa25472b9cdbe8685b4d3ed1a7ce3999803f0179658`
- The v0.26.0 release is published as `v0.26.0-win-cu128` with both wheel
  assets attached; the installer points to that release.

### Scope and credit

The local CPU/filesystem KV tiering remains an independent adaptation of vLLM's
offloading framework informed by [LMCache](https://github.com/LMCache/LMCache).
It does not copy or bundle LMCache code and does not claim feature parity;
remote/distributed cache, P2P, NIXL, GDS, object-store tiers, and CacheBlend
remain out of scope.

## v0.25.1-win-cu128 - 2026-07-19

Upstream bump from vLLM 0.24.0 to **vLLM 0.25.1**, targeting CPython
3.13, CUDA 12.8, PyTorch 2.11.0+cu128, Triton Windows 3.6.0.post26, and
`TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0`.

### New

- Added `vllm-windows-v9.patch`, generated against upstream tag `v0.25.1`
  (`752a3a504485790a2e8491cacbb35c137339ad34`) and verified with
  `git apply --check` against a clean worktree.
- Added `assemble_wheel_cu128_v0.25.1.py` with fail-fast checks for all native,
  Rust, FlashAttention, dependency, and KV-offload payloads/fix markers.
- Added opt-in prompt-KV cache modes to `vllm_launcher.py`: CPU LRU, CPU ARC,
  RAM + filesystem LRU, and RAM + filesystem ARC. Prefix caching is enabled
  automatically only when one of these modes is selected.
- Added a release-grade real-model harness for baseline, RAM offload,
  forced-filesystem restore, and cross-process persistent reuse.

### Windows KV offload fixes

- Added Windows shared-file mmap support using the system temporary directory,
  `mmap.ACCESS_WRITE`, and a creator/joiner size-race fix.
- Added binary-safe filesystem block I/O, an `os.read` fallback for Windows,
  complete-read validation, and safe error cleanup.
- Sanitized absolute Windows model paths before forming filesystem cache
  directories while retaining the full path in the configuration hash.
- Routed CPU-to-GPU loads from file-backed mmap through native CUDA DMA on
  Windows. Triton direct host-pointer reads caused an illegal memory access for
  some grouped block shapes.
- Added a non-Triton pure-Torch block-table fallback and portable upstream test
  adjustments.
- Built and packaged `fs_io_C.pyd`.
- Added Windows `AMD64` to the `llguidance` and `xgrammar` requirement markers.

### Validated

- Completed the full native build in the separate v2 build tree and assembled
  a 3,617-file wheel.
- Installed the exact final wheel outside the source tree and verified runtime
  and distribution version `0.25.1+cu128`.
- Passed a 24-case RTX 3090 GPU offload matrix covering both transfer
  directions, ordinary and shared memory, multiple page sizes,
  `block_size_factor` 1 and 3, and multiple KV groups.
- Passed CPU LRU and ARC model tests plus forced filesystem LRU/ARC eviction
  and restore using `Qwen3-14B-abliterated-AWQ-4bit`. The restored request
  reused 1,440 cached prompt tokens and produced exactly the baseline token
  IDs.
- Verified a new process could reuse the persistent filesystem cache and
  repeated that check from the final wheel.
- The observed filesystem-restore times (about 0.76-1.02 seconds versus about
  1.36 seconds cold in this focused test) are validation evidence, not a broad
  performance guarantee.

### Safety and scope

- KV offload remains **disabled by default**.
- Filesystem modes require an explicit cache root and do not impose an
  automatic disk quota; users must monitor and clean that directory.
- `launch.bat` sets `PYTHONHASHSEED=0` before Python starts to keep persistent
  cache keys stable across restarts.
- This adapts local RAM/filesystem tiering ideas researched from LMCache to
  vLLM's native offload framework. It does not claim LMCache feature parity;
  remote cache, P2P, NIXL, GDS, object-store, and distributed tiers are not in
  this release.
- The local text-only `Qwen3.5-9B-abliterated-GPTQ-4bit` model remains blocked
  by the upstream Qwen3.5 registry/config path, so release validation used the
  supported local Qwen3 AWQ model instead of adding an unmerged workaround.

### Artifacts

- Wheel: `vllm-0.25.1+cu128-cp313-cp313-win_amd64.whl`
- Size: `293080424` bytes
- SHA-256:
  `0C4F9B2E36482523FC7B4C092D711AC49B4265EF9F36A7AEEFFF9A667C875339`
- Patch SHA-256:
  `4893BDB35F905237BD0D0D042E365EAFC5B6B4C49809747BE49B42E6D8BF7609`

## v0.24.0-win-cu128 - 2026-07-01

Upstream bump from vLLM 0.23.0 to **vLLM 0.24.0**, still targeting
Python 3.13, CUDA 12.8, PyTorch 2.11.0+cu128, and
`TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0`.

### 2026-07-17 performance quick-start correction

- Changed every Hello World example to the normal `kv_cache_dtype="auto"`
  baseline. The previous README selected `isoquant4`, whose documented
  unfused PyTorch fallback is 30-300× slower and is intended for offline or
  memory-constrained workloads.
- Stopped presenting `enforce_eager=True` as a throughput default; it is now
  documented as a graph/compile compatibility option that disables normal
  optimizations.
- Synchronized the usage guide with `vllm_launcher.py`'s actual defaults and
  added slow-generation troubleshooting guidance.
- Removed the unsupported Windows `expandable_segments` recommendation, which
  only produced a PyTorch warning and did not repair OOM conditions.

### 2026-07-17 Windows sampling hotfix

- Fixed the follow-up failure reported in issue #10:
  `ValueError: low is out of bounds for int32` during request sampling.
- Requested NumPy's explicit `int64` output when vLLM generates a seed across
  the full signed 64-bit range. NumPy otherwise defaults to a C `long`, which
  remains 32-bit on 64-bit Windows.
- Added source, patch, and wheel regression guards and repacked the release
  wheel. Final wheel SHA-256:
  `41E930FBCF994E4FD7E5CB1585F8D277AF3FFDBA0AEE7F5DDE5822DD90E6FBA7`.

### 2026-07-12 reliability audit

- Fixed issue #8 by replacing fragile PowerShell stdout hash capture with a
  Python streaming SHA-256 verifier that always reports the actual digest.
- Pinned exact sizes and SHA-256 digests for Python 3.13.14, its NuGet
  development package, `get-pip.py`, and both project release wheels. Project
  wheels now use temporary files; stale or truncated copies are repaired
  automatically.
- Changed `.vllm-installed` from a timestamp to the verified wheel SHA-256 and
  write it only after dependency, native/Rust, model-import, Triton, and CUDA
  rotary checks pass. `launch.bat` now rejects stale or incomplete installs.
- Made build patching fail closed: missing, partial, or conflicting patches,
  wrong source bases, wrong Python/PyTorch/CUDA versions, missing `protoc`, and
  missing release artifacts now stop the build.
- Removed the unsafe post-build FlashAttention `xcopy`; the patched build and
  assembler own that payload. The assembler now self-validates metadata, ZIP
  integrity, required binaries/modules, CuteDSL import rewrites, and `RECORD`.
- Fixed concurrent streaming/non-streaming output routing in
  `vllm_launcher.py` by giving one dispatcher exclusive ownership of
  `engine.step()`.
- Fixed the portable install's missing `multi_turboquant` dependency. A
  pure-Python wheel built from commit `e2b59ee474132999c2b42d5c96bfc48fcaf850dc`
  is now a pinned release asset, and all six vLLM write/decode paths are tested.
- Added pure-Python regression tests for artifact verification and concurrent
  engine output dispatch.
- Repacked the vLLM wheel with Windows-safe process-tree shutdown and a clear
  `--uds` rejection. Final wheel SHA-256:
  `41E930FBCF994E4FD7E5CB1585F8D277AF3FFDBA0AEE7F5DDE5822DD90E6FBA7`.

### 2026-07-12 legacy PowerShell bootstrap fix

- Fixed issue #9: pre-Python integrity checks no longer require the unavailable
  `Get-FileHash` cmdlet.
- Replaced bootstrap hashing with direct .NET SHA-256 and replaced
  `Expand-Archive` with a .NET ZIP extractor that supports overwrite repair and
  rejects path-traversal entries.
- Forced basic web parsing for every bootstrap download so Windows PowerShell
  does not depend on Internet Explorer or stop at a script-execution prompt.
- Pinned `get-pip.py` to an immutable upstream commit so its verified hash
  cannot drift when PyPA updates the floating download URL.
- Made embedded-Python extraction transactional through `python.part`, so an
  interrupted extraction cannot be mistaken for a valid Python installation.
- Updated the audit workflow to the current GitHub Actions Node 24 runtimes.
- Added regression tests that forbid both cmdlets in `install.bat` and exercise
  exact hash/size verification, paths with spaces, overwrite extraction, and a
  malicious ZIP path.
- Revalidated a complete fresh Python 3.13.14 bootstrap including the Python
  archive, NuGet development files, `get-pip.py`, both project wheels, and CUDA.

### 2026-07-11 packaging hotfix

- Rebuilt the wheel with the generated `vllm_flash_attn.layers`,
  `vllm_flash_attn.ops`, and 48 CuteDSL Python files required by Qwen3-VL
  and other FlashAttention call paths.
- Replaced the Windows-unsafe editable-build path split with
  `Path.relative_to`, so future source builds copy generated Python files
  out of the temporary build directory correctly.
- Added assembler fail-fast checks plus complete wheel ZIP/RECORD and CUDA
  rotary regression tests.
- Updated the installer to accept only the current v0.24.0 wheel, pin its
  SHA256, force-reinstall that wheel without replacing PyTorch, and verify
  the FlashAttention rotary import before marking installation complete.

### New

- **`vllm-windows-v8.patch`** generated against upstream tag `v0.24.0`
  and validated with `git apply --check` on a fresh checkout.
- **`assemble_wheel_cu128_v0.24.0.py`** packages the already-built tree
  into `vllm-0.24.0+cu128-cp313-cp313-win_amd64.whl`.
- **Rust tool parser support**: v0.24 adds `_rust_tool_parser.pyd`, now
  included in the wheel beside `vllm-rs.exe`.

### Fixed

- Skipped QuTLASS on Windows and kept the optional `_qutlass_C` warning
  quiet when the module is intentionally absent.
- Skipped cooperative TopK on Windows and guarded the sparse-attention
  caller so it falls back cleanly when the op is not compiled.
- Skipped DeepGEMM on Windows; upstream's build helper still invokes
  `g++`, which is not available in the MSVC build environment.
- Updated Windows precompiled Rust artifact detection to recognize
  `vllm-rs.exe` and `_rust_*.pyd`.
- Carried forward the v0.23 OpenAI API server, DP supervisor, uvloop,
  selector-event-loop, safetensors, and CUDA/MSVC fixes.

### Verified

- Native CUDA build completed for `_C_stable_libtorch`,
  `_moe_C_stable_libtorch`, `cumem_allocator`, `spinloop`, FA2,
  `vllm-rs.exe`, and `_rust_tool_parser.pyd`.
- Final wheel installed with `uv pip install --reinstall --no-deps`.
- Smoke test imported `vllm 0.24.0+cu128`, all native extensions above,
  OpenAI API server / DP supervisor import surface, `vllm --help`, and
  `vllm serve --help`.
- Verified `VLLM_USE_RUST_FRONTEND=1` resolves the packaged
  `vllm-rs.exe`.
- Verified intentionally skipped paths report unavailable:
  `has_deep_gemm=False` and `has_cooperative_topk=False`.
- Wheel SHA256:
  `41E930FBCF994E4FD7E5CB1585F8D277AF3FFDBA0AEE7F5DDE5822DD90E6FBA7`.

## v0.23.0-win-cu128 - 2026-06-30

Upstream bump from vLLM 0.21.0 to **vLLM 0.23.0**, still targeting
Python 3.13, CUDA 12.8, PyTorch 2.11.0+cu128, and
`TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0`.

### New

- **`vllm-windows-v6.patch`** generated against upstream tag `v0.23.0` and
  validated with `git apply --check` on a fresh checkout.
- **Rust frontend support on Windows**: `vllm-rs.exe` now builds, packages,
  and resolves via `VLLM_USE_RUST_FRONTEND=1`.
- **`assemble_wheel_cu128_v0.23.0.py`** packages the already-built tree into
  `vllm-0.23.0+cu128-cp313-cp313-win_amd64.whl` without another CUDA compile.

### Fixed

- Added Windows Rust fixes for process shutdown, TCP-only listener support,
  Ctrl+C shutdown handling, and the mimalloc/MSVC CRT link mismatch.
- Added `vllm-rs.exe` lookup in `vllm/envs.py`; upstream only checked
  `vllm-rs`, which fails on Windows wheels.
- Added the v0.23 DP supervisor `uvloop` fallback; this module was new after
  the earlier Windows API-server fixes and was caught by CLI import testing.
- Carried forward the `uv` wheel `RECORD` fix by writing `RECORD` with
  `csv.writer`, so comma-containing fused-MoE config filenames install cleanly.

### Verified

- Native CUDA build completed for all six extension modules:
  `_C`, `_C_stable_libtorch`, `_moe_C`, `cumem_allocator`, `spinloop`, FA2.
- `setup.py build_rust --inplace` completed and copied `vllm\vllm-rs.exe`.
- Final wheel installed with `uv pip install --reinstall --no-deps`.
- `vllm --help` and `vllm serve --help` exit cleanly.
- Smoke test imported `vllm 0.23.0+cu128`, all native extensions above, and
  the OpenAI API server / DP supervisor import surface, then resolved
  `VLLM_RUST_FRONTEND_PATH` to the packaged `vllm-rs.exe`.
- Wheel SHA256:
  `53BC2360AC636804DD37FD2FBD8098FCE3C350AEF55B257F0EF0DDF6B24FADB1`.

## v0.21.0-win-cu128 — 2026-05-25

Rebuild of the same vLLM 0.21.0 source for **RTX 50-series (Blackwell,
sm_120)** on **Python 3.13 / CUDA 12.8 / PyTorch 2.11.0+cu128**, plus four
fixes that make the **OpenAI API server start on Windows**. Prompted by
issue #4 (RTX 5090: `no kernel image is available` + `zmq: not a socket`).

### New

- **Blackwell (sm_120) wheel** — `vllm-0.21.0+cu128-cp313-cp313-win_amd64.whl`,
  built with `TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0` so it carries sm_86 / sm_89 /
  sm_120 kernels (verified with `cuobjdump`: 33 sm_120 cubins in `_C.pyd`, 16
  in `_C_stable_libtorch.pyd`). The cu126 wheel was sm_86-only and fails on a
  5090 — a compute-capability gap, not a Python issue.
- **OpenAI API server now works on Windows** — `vllm serve` / `api_server`
  previously crashed; only the in-process `LLM()` path worked. Four Windows
  bugs fixed (see PATCHES.md): uvloop fallback, ZMQ sentinel poller,
  Proactor `add_reader` (`WindowsSelectorEventLoopPolicy`), and
  `add_signal_handler`. **winloop no longer required.**

### Changed

- `install.bat` now provisions **Python 3.13.11 + torch cu128**, and
  explicitly installs `llguidance` + `xgrammar` (vLLM gates them on
  `platform_machine=="x86_64"`, but Windows reports `AMD64`, so pip skips
  them and vLLM fails to import).
- `vllm-windows-v5.patch` regenerated — 46 files, ~2160 lines (adds the
  API-server fixes and the Windows kernel exclusions below).

### Excluded on Windows (skipped, gracefully)

- **QuTLASS** (NVFP4/MXFP4 microscaling quant) — uses GCC inline-PTX `asm`
  in host-reachable code; never ported to MSVC. Only pulled in once sm_120
  is enabled. Its `_qutlass_C` ops are `hasattr`-guarded in vLLM.
- **MiniMax all-reduce RMS fusion** (`minimax_reduce_rms_kernel.cu`) — a
  multi-GPU collective (unusable under Windows `FakeProcessGroup`) that also
  crashes nvcc 12.8's `cudafe++`. Callers are `hasattr`-guarded.
- Mainstream paths unaffected: FP16/BF16, AWQ, GPTQ/Marlin, FP8, and all 10
  KV-cache compression methods.

### Build notes

- **CUDA 12.8** is required for sm_120 (12.6 can't target Blackwell). CUDA
  13.x was tried and abandoned — it repeatedly crashes MSVC's compiler.
- Build with **`MAX_JOBS=2`** and **without sccache** — both trigger
  intermittent `cl.exe` ICEs (0xC000001D) on the heavy 3-arch CUDA TUs.

## v0.21.0-win — 2026-05-19

Major upstream bump. 1,157 commits from v0.19.1 → v0.21.0. PyTorch
2.10.0 → 2.11.0. CUTLASS 4.2.1 → 4.4.2. New native TurboQuant attention
backend in mainline (PR #38479) coexists with our Multi-TurboQuant 6.

### New

- **vLLM v0.21.0 base** — covers the v0.19.2 / v0.20.0 / v0.20.1 /
  v0.20.2 / v0.21.0 release train, including v1 engine maturity,
  zero-bubble DP scheduling, batched chat completions, DeepGEMM
  extension, async-scheduling hardening, and the new
  `TurboQuantBackend` (PR #38479) with four `turboquant_*` KV cache
  variants (`turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`,
  `turboquant_3bit_nc`).
- **10 KV cache compression dtypes in one wheel** — our 6
  Multi-TurboQuant methods *and* the 4 upstream variants both live in
  `CacheDType`. The platform dispatcher in `vllm/platforms/cuda.py`
  routes the upstream names to `TurboQuantBackend` and ours to the
  patched `TritonAttention` backend.
- **PyTorch 2.11.0 + CUDA 12.6** wheels for cp310 win_amd64.
- **`cutlass-windows.patch` and `vllm-flash-attn-cutlass-windows.patch`**
  ship inside the v5 vllm-source patch. `CMakeLists.txt` /
  `vllm_flash_attn.cmake` apply them automatically after FetchContent
  lands the upstream source, so end users don't need to know.
- **Auto-default `VLLM_USE_FLASHINFER_SAMPLER=False` on Windows** —
  upstream defaults this to True, which triggers an unconditional
  `import flashinfer` in the sampler. The patch flips the default on
  `sys.platform == "win32"` so the Triton sampler is used silently.

### Changed

- `vllm-windows-v5.patch` replaces `vllm-windows-v4.patch`. 36 modified
  files + 3 new files (`vllm/v1/attention/ops/multi_turboquant_kv.py`,
  `cutlass-windows.patch`, `vllm-flash-attn-cutlass-windows.patch`).
  ~1918 lines total. Three v4 hunks dropped because they're now obsolete
  upstream: `csrc/topk.cu` designated-initializer fix landed upstream,
  `routed_experts_capturer.py` got a major rewrite that no longer needs
  `fcntl`/`msvcrt` locking, and `triton_reshape_and_cache_flash.py` now
  uses `is_quantized_kv_cache` already.
- `build.bat` now applies `vllm-windows-v5.patch` and sets
  `SETUPTOOLS_SCM_PRETEND_VERSION=0.21.0`.
- README badges, releases table, hello-world snippet, and "What's in
  the patch" section updated to v0.21.0 / PyTorch 2.11.0 numbers.

### Fixed

- **PyTorch 2.11.0's `c10::cuda::CUDACachingAllocator` uses `small` as
  a parameter name** — Windows SDK's `rpcndr.h` defines `small` as a
  macro (`typedef char small`). `bool small` gets preprocessor-replaced
  to `bool char` and nvcc errors out. Patch adds `/Usmall` and
  `WIN32_LEAN_AND_MEAN` to CXX + CUDA flags.
- **MSVC reports `__cplusplus=199711L` by default** — CUTLASS 4.4.2's
  `platform.h` gates `is_unsigned_v` / `is_integral_v` aliases behind
  `__cplusplus >= 201703L`, so `exmy_base.h` fails. Patch adds
  `/Zc:__cplusplus` to CXX + CUDA flags.
- **CUTLASS 4.4.2 `cuda_host_adapter.hpp::memsetDevice` is marked
  `CUTLASS_HOST_DEVICE` but calls a `__host__`-only virtual
  `memsetDeviceImpl`** — nvcc rejects the cross-execution-space call.
  `cutlass-windows.patch` weakens the wrapper to `CUTLASS_HOST`.
- **CUTLASS 4.4.2 SM100/SM103 kernel headers declare `static constexpr
  dim3 get_block_shape()`** — `dim3` is a non-literal type for MSVC
  even with CUDA 12.6's constexpr constructors, so nvcc rejects the
  return type. `cutlass-windows.patch` drops the `constexpr` on the
  four offenders.
- **`csrc/persistent_topk.cuh` (new file in v0.21.0) uses
  `__attribute__((always_inline))`** — GCC-only. Patch wraps the
  `FLASHINFER_INLINE` macro in `#ifdef _MSC_VER` with `__forceinline`
  fallback.
- **`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`
  uses `quant_type_max_v<scalar_out_t>` without `()`** — that's a
  variable-template reference, but the v4 patch already converted
  `quant_type_max_v` to a function template. Patch adds the call parens.
- **`csrc/moe/topk_softplus_sqrt_kernels.cu` `DISPATCH_HASH(...)`
  argument contained `#ifndef USE_ROCM`** — preprocessor directives
  inside macro arguments are ill-formed; even `/Zc:preprocessor`
  doesn't allow it. Patch hoists the `#ifndef` outside the macro call.
- **`requirements/cuda.txt` `fastsafetensors >= 0.2.2` is Linux-only
  (`io_uring`)** — was inherited unchanged from upstream; pip tried to
  build it from source and required `pybind11`. Commented out — the
  Windows safetensors path uses our existing numpy-mmap reader.

### Verified (RTX 3090, Qwen3-14B-abliterated-AWQ-4bit)

| KV dtype | Output tok/s | Notes |
|---|---:|---|
| `auto` (fp16, FA2 backend) | 9.7 | 20 tok in 2.06 s |
| `turboquant35` (ours, PyTorch-fallback) | 0.73 | 20 tok in 27.4 s |

### Known limitations

Unchanged from v0.19.x:

- TQ throughput penalty on our 6 methods (PyTorch-fallback encode/decode).
  Upstream `turboquant_*` variants use fused Triton kernels and don't
  pay this cost.
- Single GPU only (NCCL still unavailable on Windows).
- No FlashAttention 3 or 4, no FlashInfer.
- No DeepGEMM, no Quack, no Tilelang, no TokenSpeed-MLA, no NIXL.

---

## v0.19.1-win — 2026-04-19

Point release. Upstream vLLM bumped to 0.19.1, one new Windows patch,
all six Multi-TurboQuant methods re-verified on RTX 3090.

### New

- **vLLM v0.19.1 base** — upstream point release (CI fixes, minor
  bugfixes, pinned `nixl-cu{12,13}` versions, Jina ColBERT rotary
  recomputation for transformers v5).
- **`tests/test_tq_diag.py`** — faulthandler-guarded diagnostic that
  dumps stack traces if `generate()` stalls past 90s. Catches hangs vs
  genuinely slow runs (PyTorch-fallback decode is slow but terminates).

### Changed

- `vllm-windows-v4.patch` is now the active patch. 34 modified files +
  1 new file (`vllm/v1/attention/ops/multi_turboquant_kv.py`). 1656
  lines. The only functional delta vs v3 is a new hunk in
  `vllm/v1/utils.py` (see Fixed).
- `build.bat` now applies `vllm-windows-v4.patch` and sets
  `SETUPTOOLS_SCM_PRETEND_VERSION=0.19.1`.
- `README.md` version badges, tables, and examples updated to v0.19.1.

### Fixed

- **`vllm/v1/utils.py` unconditional `import uvloop`** — upstream added
  this import at module load. `uvloop` is Unix-only, so the import fails
  on Windows before the `Hello world` snippet's `sys.modules` stub gets
  a chance to run. The patch wraps the import in a `try/except
  ImportError` that aliases `asyncio` as `uvloop` (same `run()`
  signature). This means user code no longer needs to stub `uvloop` at
  all — the fallback is baked into the wheel.

### Verified (RTX 3090, Qwen3-14B-abliterated-AWQ-4bit)

All 6 TQ methods load the model, initialize the KV cache, and generate
coherent output end-to-end. Timings are for 5 decoded tokens with
`max_model_len=512`, `gpu_memory_utilization=0.5`:

| Method | Preset | Time (5 tok) | Output tok/s |
|---|---|---:|---:|
| `isoquant3` | no_calibration_symmetric | 41.5s | 0.12 |
| `isoquant4` | no_calibration_quality | 53.0s | 0.09 |
| `planarquant3` | k_only_planar | 40.5s | 0.12 |
| `planarquant4` | k_only_planar | 53.0s | 0.09 |
| `turboquant25` | max_compression | 6.7s | **0.74** |
| `turboquant35` | speed | 5.4s | **0.92** |

`turboquant25/35` are ~8× faster than the iso/planar methods on
Windows — their algorithm favours the PyTorch-fallback path. All
methods still pay the expected ~30-300× throughput cost vs FP16 until
a fused Triton kernel lands (tracked as a known limitation, not new in
v0.19.1).

### Known limitations

Unchanged from v0.19.0-win:

- TQ throughput penalty from PyTorch-fallback encode/decode.
- Single GPU only (NCCL still unavailable on Windows).
- No FlashAttention 3, no FlashInfer.

---

## v0.19.0-win — 2026-04-12

Major release. vLLM 0.19.0 base, full Multi-TurboQuant integration with
real packed-uint8 storage, custom Windows safetensors reader.

### New

- **vLLM v0.19.0 base** — Gemma 4 support, zero-bubble async scheduling,
  Model Runner V2 maturation, online MXFP8, batched chat completions
  endpoint, ViT full CUDA graphs.
- **Multi-TurboQuant integration** — six KV cache compression methods
  with real packed uint8 storage:
  - `isoquant3` (3.25-bit, no calibration, quaternion 4D rotation)
  - `isoquant4` (4.25-bit, no calibration, quaternion 4D rotation)
  - `planarquant3` (3.25-bit, no calibration, Givens 2D rotation)
  - `planarquant4` (4.25-bit, no calibration, Givens 2D rotation)
  - `turboquant25` (2.25-bit, runtime calibration, WHT + MSE + QJL)
  - `turboquant35` (3.25-bit, runtime calibration, WHT + MSE + QJL)

  All six methods deliver **2× more KV cache tokens** at the same
  `gpu_memory_utilization` (uint8 storage with `head_size` bytes per
  slot vs fp16 with `head_size * 2`). Each method's quantization noise
  is verified to actually affect inference output (not a placebo) via
  `tests/test_tq_real.py`.

- **Custom Windows safetensors reader** — `numpy.memmap` + chunked GPU
  streaming. Loads a 14B model in **6.5 seconds** vs ~189 seconds with
  the upstream mmap path. Works on systems with the Windows pagefile
  disabled.

- **End-to-end test suite** — `tests/test_v19.py` (smoke),
  `tests/test_tq_real.py` (correctness sweep), `tests/test_tq_thorough.py`
  (full benchmark with 8 batched prompts and quality checks).

- **Comprehensive docs** — `docs/install.md`, `docs/usage.md`,
  `docs/turboquant.md`, `docs/benchmarks.md`, `docs/build.md`,
  `docs/architecture.md`, `docs/troubleshooting.md`.

### Changed

- `vllm-windows-v3.patch` is now the active patch (vs v0.17.1's
  `vllm-windows-v2.patch`). 33 modified files + 1 new file
  (`vllm/v1/attention/ops/multi_turboquant_kv.py`, 295 lines).
- `build_wheel.py` now defaults to `--source-dir vllm-source --output-dir dist-v3`.
- `install.bat` now installs PyTorch 2.10.0 + triton-windows 3.6.0.post26.
- `build.bat` now applies `vllm-windows-v3.patch` and uses Ninja generator.

### Fixed

- New patch entries for v0.19.0 source changes:
  - `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu` —
    `and` keyword → `&&`
  - CUTLASS in vllm-flash-attn — guard SM100/103/120 includes behind
    CUDA 12.8+ (vendored fix in `.deps/`)
- Custom safetensors reader resolves `OSError 1455 (The paging file is
  too small for this operation to complete)` on systems with the
  pagefile disabled.

### Known limitations

- TQ throughput drops ~30-300× because encode/decode runs in
  PyTorch-vectorised mode (no fused Triton kernel yet). Memory savings
  are real, throughput cost is the trade-off. Suitable for offline /
  long-context / batch workloads. Online serving should stick with
  `auto` or `fp8`.
- Single GPU only (NCCL still unavailable on Windows; the patch wires
  up `FakeProcessGroup` for single-rank operation).

---

## v0.17.1-win — 2026-03-21

vLLM 0.17.1 base. First release with TurboQuant KV cache compression
(2 recipes) and Triton support via triton-windows.

- vLLM 0.17.1 base
- TurboQuant KV cache compression: `turboquant25` (~6.4× reduction)
  and `turboquant35` (~4.27× reduction)
- Triton kernel support via triton-windows 3.6.0
- Qwen 3.5 (DeltaNet hybrid) support
- 27 patched files

---

## v0.14.2-win — 2026-02-28

First pre-built wheel release.

- vLLM 0.14.2 base
- PyTorch 2.9.1+cu126
- Pre-built wheel + portable Python installer (`install.bat`)
- 26 patched files (no Triton, no TQ)
- Tested with Qwen 2.5, Llama 3.x, Phi-4, xLAM

---

## v0.14.1-win — 2026-02-11

Initial release. Source patches only (no pre-built wheel).
