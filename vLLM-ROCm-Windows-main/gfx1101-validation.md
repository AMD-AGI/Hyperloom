# gfx1101 (RX 7800 XT) validation — full stack on native Windows

End-to-end validation of this repo on **AMD Radeon RX 7800 XT (gfx1101, Navi 32, 16 GB)** —
the first gfx1101 report for this project. Everything below was run and verified on the
machine (Windows 11 26200, Adrenalin 32.0.31041, iGPU + dGPU hybrid).

Author: CLVwp — 2026-09-26.

## Results

| Test | Result |
|---|---|
| `first_token.py` (OPT-125m) | ✅ coherent output |
| Native `csrc` build (`build_run.bat`) | ✅ BUILD_OK in 52.9 s (silu/rms_norm/fused_add_rms_norm/rotary/gptq_gemm/gptq_shuffle, gfx1100-1103 fat binary) |
| `bench.py` Qwen2.5-7B-Instruct-GPTQ-Int4 | ✅ **81.9 tok/s** decode (batch 1, greedy, 128 tok), **28 ms** single-token TTFT, 9.38 GiB VRAM, BENCH_OK |

Reference: 115 tok/s on RX 7900 XT (gfx1100) from the README → ratio 0.71, consistent with
the 624 vs 800 GB/s memory-bandwidth ratio (0.78). No software penalty observed on gfx1101.

## Validated stack

- Windows 11 26200, HIP SDK 7.2 installed (`C:\Program Files\AMD\ROCm\7.2`) — see §3 for why
  the *pip* SDK is used for the native build instead
- venv Python 3.12.6, **ASCII install path** (`C:\AI\vllm-venv`) — see §1
- torch 2.10.0+rocm7.13.0a20260508 + `rocm[libraries,devel]` from TheRock nightly index
  `https://rocm.nightlies.amd.com/v2/gfx110X-all/`
- triton-windows 3.6.0.post26, conch-triton-kernels 1.3, xgrammar 0.2.8, llguidance 1.8.0
- vLLM v0.19.1 (`VLLM_TARGET_DEVICE=empty`) + this repo's plugin

## Environment variables needed at runtime

```bat
set HIP_VISIBLE_DEVICES=1        & :: iGPU is device 0; the 7800 XT is device 1 on hybrid systems
set ROCM_HOME=C:\AI\vllm-venv\Lib\site-packages\_rocm_sdk_devel   & :: see §2
set TMP=C:\AI\tmp                & :: ASCII temp, see §1
set TEMP=C:\AI\tmp
```

## Findings

### 1. Non-ASCII Windows user paths break tvm-ffi / xgrammar DLL loading

With a Windows username containing non-ASCII characters (e.g. `C:\Users\PC-Clément`),
`xgrammar` fails at import: `tvm_ffi`'s `DSOLibrary::Load` passes the DLL path through a
narrow (ANSI) encoding, the `é` becomes `Ã©`, and `xgrammar_bindings.dll` is reported
missing although it exists. Minimal reproducer:

```python
from pathlib import Path
from tvm_ffi.module import load_module
load_module(Path(r'C:\Users\PC-Clément\...\xgrammar_bindings.dll'))  # RuntimeError
load_module(Path(r'C:\ASCII\...\xgrammar_bindings.dll'))            # OK
```

Workaround: install the venv (and point TMP/TEMP) at an ASCII-only path. Upstream candidate:
apache/tvm-ffi. Same class of issue can bite the triton JIT temp dirs, hence the TMP
redirection.

### 2. triton-windows silently misses the HIP includes without ROCM_HOME

`triton/backends/amd/driver.py` resolves the ROCm SDK root **at import time** by shelling
out to `rocm-sdk path --root`; if `rocm-sdk.exe` is not on the process PATH at that moment,
the HIP include dir is never added, and the first Triton JIT kernel fails with
`fatal error: 'hip/hip_runtime.h' file not found`. On this stack the SDK lives inside the
venv, and `rocm-sdk.exe` is not on PATH by default → export `ROCM_HOME` (or `HIP_PATH`)
pointing at the SDK root. The Python package `rocm_sdk` works without PATH; triton could
prefer it over the CLI probe.

### 3. The installed HIP SDK 7.2 toolchain fails the native build; the pip SDK works

The first native-build attempt with the auto-detected HIP SDK 7.2
(`C:\Program Files\AMD\ROCm\7.2`) failed with `cannot find ROCm device library` from
`clang.exe`, even though `--rocm-device-lib-path` pointed at the SDK's own
`amdgcn\bitcode` (which does ship `oclc_isa_version_1101.bc`). Building through the pip
SDK's toolchain instead resolved it, and also keeps compiler/bitcode consistent with the
torch wheel. Fix: run the native build with `HIP_PATH` pointed at the pip SDK:

```bat
set HIP_PATH=C:\AI\vllm-venv\Lib\site-packages\_rocm_sdk_devel
```

`tools/winrocm_env.bat` already honours a pre-set `HIP_PATH`, so no code change needed.
(The follow-up PR adds a `hip_root_for_arch` helper that prefers an SDK whose bitcode
matches the target arch and prints an actionable message otherwise.)

### 4. atomicAdd(half/half2) ambiguous on HIP 7.13 — and the 7.14 guard could never work

vLLM's `csrc/quantization/gptq/compat.cuh` defines `atomicAdd(half*, half)` /
`atomicAdd(half2*, half2)` compat overloads guarded by `#if __CUDA_ARCH__ < 700 ||
defined(USE_ROCM)`. Two problems on HIP 7.13/7.14, which ship native overloads in
`amd_hip_fp16.h`:

1. **HIP 7.13 already clashes** — the existing patch (commit 2fc5f43) only guards against
   HIP >= 7.14, but the pip 7.13 SDK already provides both native overloads.
2. **The guard is structurally ineffective**: hipcc does **not** define `__CUDA_ARCH__` in
   the device pass, and an undefined identifier evaluates to `0` in `#if`, so
   `__CUDA_ARCH__ < 700` is *always true* — the `|| defined(USE_ROCM)` arm never gets a
   chance to be version-gated.

Fix (applied in `experiments/vllm_c_ext/build_c_ext.py`): rewrite both arms as
`(defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600/700) || (defined(USE_ROCM) &&
!(defined(HIP_VERSION) && HIP_VERSION >= 71300000))`, and inject
`#include <hip/hip_version.h>` into the hipified `compat.cuh` so `HIP_VERSION` is actually
visible. Verified: clean build in 52.9 s, kernels load and run on gfx1101.

### 5. build_c_ext.py rewrote shim headers without their SPDX headers

`build_c_ext.py` regenerates `experiments/vllm_c_ext/shim/**` on every run but wrote them
without the SPDX/copyright lines, so a build leaves the working tree dirty (license-notice
loss). Fixed: the generator now writes the SPDX header too.

## From-scratch install order (verified)

```bat
py -3.12 -m venv C:\AI\vllm-venv
C:\AI\vllm-venv\Scripts\python -m pip install -U pip setuptools wheel packaging setuptools_scm huggingface_hub
C:\AI\vllm-venv\Scripts\python -m pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ torch==2.10.0+rocm7.13.0a20260508 "rocm[libraries,devel]"
C:\AI\vllm-venv\Scripts\python -m pip install conch-triton-kernels llguidance xgrammar "triton-windows<3.7"
:: vLLM + plugin (from this repo's root)
cd vllm && python use_existing_torch.py
set VLLM_TARGET_DEVICE=empty
C:\AI\vllm-venv\Scripts\python -m pip install -e . --no-build-isolation
cd .. && C:\AI\vllm-venv\Scripts\python tools\patch_vllm.py vllm
C:\AI\vllm-venv\Scripts\python -m pip install -e windows_rocm_plugin
:: native kernels (HIP_PATH override per §3)
set HIP_PATH=C:\AI\vllm-venv\Lib\site-packages\_rocm_sdk_devel
cd experiments\vllm_c_ext && build_run.bat
```

Notes: setuptools/setuptools_scm must be in the venv before the editable vLLM install
(its setup.py imports them; Python 3.12 venvs ship neither). `use_existing_torch.py` must
run before the vLLM install or pip pulls a CUDA/CPU torch.
