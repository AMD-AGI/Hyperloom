# Install vLLM v0.33.0 on Windows

> **Release selection:** [the Windows releases](releases.md) by Python/CUDA version, installer, and source-build guidance.
> These instructions describe the Latest release, 0.33.0 for Python 3.13 / CUDA 13.0. For the
> Python 3.14 build or an older version, use that release's own instructions.

Two paths:

- **Install the pre-built wheel**: no compiler needed; recommended for most users.
- **Build from source**: requires Visual Studio 2022, CUDA 13.0, Python 3.13, and patience.

**Blackwell FP8/NVFP4:** 0.33.0 includes the issue #16 FP8 Marlin fallback
fixes but does not provide native Blackwell FP4 acceleration, and it has not
been run on Blackwell hardware. See the
[Blackwell notes](troubleshooting.md#blackwell-fp8-nvfp4) if a model still fails.

`install.bat` handles the wheel path end to end. The source build is
documented in the [v0.33.0 build record](v0.33.0-build-candidate.md).

## Install The Wheel

### Requirements

| Component | Version | Notes |
|---|---|---|
| Windows | 10 / 11 x64 | Tested on Windows 10 Pro 22H2 |
| GPU | NVIDIA SM 7.5+ | RTX 20/30/40/50, A100, H100 |
| Driver | R580+ | Required by CUDA 13.0 |
| Python | 3.13.x | `install.bat` uses embedded Python 3.13.14 plus headers/libs for Triton |
| PyTorch | 2.14.0+cu130 | CUDA 13.0 runtime from PyTorch wheels |
| Triton | triton-windows 3.8.0.post29 | Installed by `install.bat` |
| Disk | 5 GB+ | Python, PyTorch, Triton, and wheel |

You do not need CUDA Toolkit or Visual Studio to install the pre-built wheel.

### Automated Install

```bat
install.bat
```

The installer downloads embedded Python, adds the Python headers/libs
needed by Triton's runtime compiler, installs PyTorch cu130,
triton-windows, the vLLM wheel, structured-output backends, and verifies
both `import vllm` and Triton's CUDA runtime driver path.

The pre-Python bootstrap requires Windows PowerShell 3 or newer but does not
require `Get-FileHash` or `Expand-Archive`. Hashing and ZIP extraction use .NET
helpers included beside `install.bat`, and downloads use basic parsing without
the Internet Explorer engine. `get-pip.py` is fetched from an immutable PyPA
commit so its pinned size and SHA-256 remain reproducible.

Fresh embedded-Python extraction is staged under `python.part` and renamed only
after the exact Python version runs successfully. A failed extraction removes
the staging directory, so rerunning the installer starts from a clean state.

It caches state in:

- `python\.torch-installed`
- `python\.vllm-installed` (contains the verified vLLM and Multi-TurboQuant SHA-256 values)

Delete those files to force reinstall.

Rerunning `install.bat` repairs an existing portable Python 3.13 directory if
its marker hash, dependencies, native/Rust files, generated FlashAttention
modules, `Include\Python.h`, or `libs\python313.lib` are missing. A stale or
truncated local wheel is deleted and downloaded again through a `.part` file.
`launch.bat` performs the same release-contract checks before starting the
server.
If `python\` is from an older major/minor Python version, delete
`python\` and rerun the installer.

### Manual Install

```bat
py -3.13 -m venv venv
venv\Scripts\activate

pip install torch==2.14.0 torchaudio==2.11.0 torchvision==0.29.0 ^
    --index-url https://download.pytorch.org/whl/cu130

pip install triton-windows==3.8.0.post29
pip install "llguidance>=1.7.0,<1.8.0" "xgrammar==0.2.8"
pip install multi_turboquant-0.1.0-py3-none-any.whl
pip install dist-v0.33.0\vllm-0.33.0-cp313-cp313-win_amd64.whl
```

Or download the wheel from the latest GitHub release:

```text
https://github.com/aivrar/vllm-windows-build/releases/tag/v0.33.0-win-cu130
```

### Verify

```bat
python -c "import importlib.metadata, vllm; print(vllm.__version__); print(importlib.metadata.version('vllm'))"
vllm --help
vllm serve --help
```

Expected runtime and distribution versions:

```text
0.33.0
0.33.0
```

The runtime and distribution metadata both report 0.33.0. `verify_install.py`
also checks the Torch CUDA and Triton versions.

## Build From Source

### Requirements

| Component | Version |
|---|---|
| Visual Studio | VS 2022 with C++ workload (MSVC 19.43 tested) |
| CUDA Toolkit | 13.0 |
| Python | 3.13.x |
| PyTorch | 2.14.0+cu130 |
| Ninja | Available in the venv or on PATH |
| Rust | Current stable MSVC toolchain |
| protoc + Perl | Required for the Rust frontend (`PROTOC`, `OPENSSL_SRC_PERL`) |
| RAM | 32 GB minimum, 64 GB recommended |
| Disk | 30 GB+ |

### Source Tree

Start from a clean upstream v0.33.0 checkout and apply the current patch:

```bat
git clone --branch v0.33.0 https://github.com/vllm-project/vllm.git vllm-source-v0.33.0
cd vllm-source-v0.33.0
git apply --check ..\vllm-windows-v12.patch
git apply ..\vllm-windows-v12.patch
```

### Build

From a Visual Studio x64 developer command prompt, with the venv, CUDA 13.0,
Git (including `usr\bin`) and Rust on `PATH`:

```bat
set TORCH_CUDA_ARCH_LIST=7.5;8.6;8.9;12.0
set VLLM_TARGET_DEVICE=cuda
set CMAKE_BUILD_TYPE=Release
set MAX_JOBS=8
set VLLM_DISABLE_SCCACHE=1
set SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0
set VLLM_RS_BUILD_VERSION=0.33.0

.venv\Scripts\python.exe tools\build_rust.py --release
.venv\Scripts\python.exe setup.py build_ext --inplace -v
```

Build the Rust executable first so `setup.py` reuses it instead of rebuilding
it in debug mode. `MAX_JOBS=8` was used on the tested machine; reduce it if
memory or compiler stability is lower.

### Build A Wheel From An Already-Built Tree

```bat
set VLLM_USE_LOCAL_PRECOMPILED=1
set VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1
set VLLM_VERSION_OVERRIDE=0.33.0
uv build --wheel --no-build-isolation
```

Output: `dist\vllm-0.33.0-cp313-cp313-win_amd64.whl`. Validate it with
`python tests\test_wheel_contents.py` before installing it outside the source
tree. Full details are in the [build record](v0.33.0-build-candidate.md).

## Runtime Environment

Keep single-rank Windows initialization on the loopback interface:

```bat
set VLLM_HOST_IP=127.0.0.1
```

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for this Windows
build. PyTorch reports that expandable segments are unsupported on this
platform, so the setting only adds a warning and does not repair an OOM.

For multi-GPU systems, set the CUDA device ordering explicitly:

```bat
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
```

For more, see [troubleshooting.md](troubleshooting.md).
