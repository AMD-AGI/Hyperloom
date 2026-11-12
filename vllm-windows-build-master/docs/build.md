# Build From Source

> **Release selection:** [the Windows releases](releases.md) by Python/CUDA version, installer, and source-build guidance.

The current release build is vLLM 0.33.0. Its exact environment, source commit,
build commands, and validation are in the
[v0.33.0 build record](v0.33.0-build-candidate.md), and the short version is in
[install.md](install.md#build-from-source). This repository contains the exact
`vllm-windows-v12.patch`, installer, launcher, validation contract, and release
documentation. The previous 0.27.1 build and the older v0.25.1 patch workflow
are retained below as historical reference.

Apply the current patch to a clean upstream tree with:

```bat
git checkout v0.33.0
git apply --check ..\vllm-windows-v12.patch
git apply ..\vllm-windows-v12.patch
```

## Previous v0.27.1 build (v2 workspace)

The v2 tree is based on upstream v0.27.1 and compiles CUDA targets
`7.5;8.6;8.9;12.0` with Python 3.13, PyTorch 2.13.0+cu130, and CUDA 13.0.
Run the checked-in build script from a developer command prompt:

```bat
cd /d E:\vllm-windows-build-v2
build_cu130_py313_v0.27.1.bat
```

After the native editable build completes, assemble the wheel from the same
tree without downloading a Linux precompiled package:

```bat
cd /d E:\vllm-windows-build-v2\vllm-source-v0.27.1
set VLLM_USE_LOCAL_PRECOMPILED=1
set VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1
set VLLM_VERSION_OVERRIDE=0.27.1
uv build --wheel --no-build-isolation --out-dir E:\vllm-windows-build-v2\dist-v0.27.1
```

Expected output:

```text
E:\vllm-windows-build-v2\dist-v0.27.1\vllm-0.27.1-cp313-cp313-win_amd64.whl
SHA-256: 7c13ed44e94694478bdd4f5fcca23e2d66ba1e8fa9bccad9fddb8651d1b2447b
```

Run `python tests\test_wheel_contents.py` against that file before release.
The complete source snapshot, native changes, exact build log, and GPU
validation are recorded in [v0.27.1-build-candidate.md](v0.27.1-build-candidate.md).

The public source delta can be applied to a clean upstream tree with:

```bat
git checkout v0.27.1
git apply --check ..\vllm-windows-v10.patch
git apply ..\vllm-windows-v10.patch
```

For install-only usage, see [install.md](install.md).

## Historical v0.25.1 patch workflow

## Patch Scope

`vllm-windows-v9.patch` is a unified diff against upstream
`vllm-project/vllm` tag `v0.25.1`
(`752a3a504485790a2e8491cacbb35c137339ad34`).

Main categories:

| Area | Purpose |
|---|---|
| Build system | Allow CUDA builds on Windows, force CUDA 12.8 paths, apply CUTLASS patches, skip Linux-only optional extensions |
| CUDA kernels | MSVC compatibility for GCC-only syntax, `__int128_t`, `__builtin_clz`, macro/preprocessor issues, and generated selector depth |
| Runtime Python | Windows multiprocessing/network/event-loop fixes, safetensors reader, FakeProcessGroup, API server fallbacks |
| Rust artifacts | Build and package `vllm-rs.exe` and `_rust_tool_parser.pyd` |
| Multi-TurboQuant | Carry the 6 local KV-cache compression methods alongside upstream TurboQuant variants |
| KV offload | Native Windows DMA, shared mmap, binary filesystem I/O, safe cache paths, CPU LRU/ARC, and tiered filesystem persistence |

## Required Toolchain

| Component | Version |
|---|---|
| Visual Studio | 2022 Community or newer, C++ workload |
| CUDA Toolkit | 12.8 |
| Python | 3.13.x |
| PyTorch | 2.11.0+cu128 |
| Triton | triton-windows 3.6.0.post26 |
| Rust | MSVC stable toolchain |
| protoc | Required for Rust frontend/tool parser |
| Generator | Ninja |

Recommended environment:

```bat
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0
set VLLM_TARGET_DEVICE=cuda
set CMAKE_BUILD_TYPE=Release
set VLLM_DISABLE_SCCACHE=1
set SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1
set MAX_JOBS=2
set PROTOC=C:\path\to\protoc.exe
```

## Build Phases

### 1. Patch Upstream

```bat
git clone https://github.com/vllm-project/vllm.git vllm-source
cd vllm-source
git checkout v0.25.1
git apply ..\vllm-windows-v9.patch
cd ..
```

### 2. Configure And Build

`python -m pip install -e . --no-build-isolation -v` drives CMake/Ninja through
`setup.py`.

`build.bat` accepts either a clean upstream v0.25.1 tree or a tree with the
complete v9 patch already applied. It stops on a partial/conflicting patch and
verifies the Python, PyTorch, CUDA, `protoc`, native, Rust, FlashAttention, and
third-party payload contract before reporting success.

Expected native artifacts in `vllm-source\vllm\` after a successful build:

```text
_C_stable_libtorch.pyd
_moe_C_stable_libtorch.pyd
_rust_tool_parser.pyd
cumem_allocator.pyd
fs_io_C.pyd
spinloop.pyd
vllm_flash_attn\_vllm_fa2_C.pyd
vllm-rs.exe
third_party\triton_kernels\...
third_party\fmha_sm100\...
```

Intentionally absent on Windows:

```text
_qutlass_C.pyd
_deep_gemm_C.pyd
cooperative_topk op
```

Those paths are optional in this build. vLLM falls back when they are not
available.

### 3. Generate Metadata

If `vllm.egg-info` was not left in the source tree by the editable build,
generate it before assembling a wheel:

```bat
set VLLM_TARGET_DEVICE=cuda
set SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1
python setup.py egg_info
```

Confirm `vllm.egg-info\PKG-INFO` contains:

```text
Version: 0.25.1+cu128
```

### 4. Assemble Wheel

```bat
python assemble_wheel_cu128_v0.25.1.py
```

Output:

```text
dist-v9\vllm-0.25.1+cu128-cp313-cp313-win_amd64.whl
```

### 5. Smoke Test The Wheel

Validate archive completeness and RECORD before installing:

```bat
python tests\test_wheel_contents.py dist-v9\vllm-0.25.1+cu128-cp313-cp313-win_amd64.whl
```

The assembler and wheel-content test verify request-seed generation, all native
release payloads, Windows KV-offload DMA/mmap/filesystem markers, the non-Triton
block-table fallback, AMD64 dependency metadata, ZIP integrity, and RECORD.

Install the wheel from outside the source tree:

```bat
python -m pip install --force-reinstall --no-deps dist-v9\vllm-0.25.1+cu128-cp313-cp313-win_amd64.whl
```

Run:

```bat
python -c "import vllm; print(vllm.__version__)"
vllm --help
vllm serve --help
```

For the issue #7 Qwen3-VL/FlashAttention regression, install the wheel into
an isolated target and run:

```bat
python -m pip install --no-deps --target %TEMP%\vllm-wheel-test dist-v9\vllm-0.25.1+cu128-cp313-cp313-win_amd64.whl
python tests\test_issue7_flash_attn.py --package-root %TEMP%\vllm-wheel-test
```

Required Rust frontend check:

```bat
set VLLM_USE_RUST_FRONTEND=1
python -c "from pathlib import Path; import vllm.envs as e; print(e.VLLM_RUST_FRONTEND_PATH); assert Path(e.VLLM_RUST_FRONTEND_PATH).exists()"
```

## Iterating

If you change Python files, rerun the smoke tests.

If you change CUDA/C++ files:

```bat
python -m pip install -e . --no-build-isolation -v
```

If you change `setup.py` or CMake files, clear the temp build directory
or start from a fresh build temp before rebuilding.

## Regenerating The Patch

From the patched vLLM source tree:

```bat
git diff --binary v0.25.1..HEAD --output=..\vllm-windows-v9.patch -- .
```

Validate against a clean upstream worktree:

```bat
git worktree add --detach ..\patch-check-v0.25.1 v0.25.1
git -C ..\patch-check-v0.25.1 apply --check ..\vllm-windows-v9.patch
```

Also run:

```bat
git diff --check v0.25.1..HEAD -- . ":!cutlass-windows.patch" ":!vllm-flash-attn-cutlass-windows.patch"
```
