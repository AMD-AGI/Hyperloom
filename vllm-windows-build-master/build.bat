@echo off
setlocal DisableDelayedExpansion

:: ============================================================
:: vLLM v0.25.1 Windows Build Script
:: Compiles vLLM from patched source with MSVC + CUDA + Ninja
:: ============================================================

echo.
echo  vLLM v0.25.1 Windows Build
echo  ==========================
echo.

:: -----------------------------------------------------------
:: 1. Check prerequisites
:: -----------------------------------------------------------

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cl.exe not found. Run this from a Visual Studio Developer Command Prompt
    echo         or run vcvars64.bat first:
    echo         "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    exit /b 1
)

if not defined CUDA_HOME (
    echo [ERROR] CUDA_HOME is not set. Point it at your CUDA toolkit, e.g.:
    echo         set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
    exit /b 1
)

if not exist "%CUDA_HOME%\bin\nvcc.exe" (
    echo [ERROR] nvcc.exe not found at %CUDA_HOME%\bin\nvcc.exe
    echo         Make sure CUDA_HOME points to a CUDA 12.8 install (12.8 is the
    echo         first toolkit with Blackwell sm_120; CUDA 13.x crashes MSVC).
    exit /b 1
)

"%CUDA_HOME%\bin\nvcc.exe" --version 2>nul | findstr /C:"release 12.8" >nul
if errorlevel 1 (
    echo [ERROR] CUDA_HOME must point to CUDA 12.8 exactly for this release build.
    exit /b 1
)

:: -----------------------------------------------------------
:: 2. Configuration (edit these for your system)
:: -----------------------------------------------------------

:: Compute capability — multi-arch covers all (drop ones you don't need):
::   RTX 30xx = 8.6, RTX 40xx = 8.9, RTX 50xx = 12.0
if not defined TORCH_CUDA_ARCH_LIST set TORCH_CUDA_ARCH_LIST=8.6;8.9;12.0

:: Parallel compile jobs. Keep at 2: higher parallelism intermittently crashes
:: MSVC cl.exe (0xC000001D) on the heavy 3-arch CUDA TUs. Do not enable sccache.
if not defined MAX_JOBS set MAX_JOBS=2

set "VLLM_TARGET_DEVICE=cuda"
set "SETUPTOOLS_SCM_PRETEND_VERSION=0.25.1"
if not defined CMAKE_BUILD_TYPE set "CMAKE_BUILD_TYPE=Release"
set "VLLM_DISABLE_SCCACHE=1"

:: -----------------------------------------------------------
:: 3. Locate vllm source
:: -----------------------------------------------------------

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%vllm-source-v0.25.1\setup.py" (
    set "VLLM_SRC=%SCRIPT_DIR%vllm-source-v0.25.1"
) else if exist "%SCRIPT_DIR%vllm-source\setup.py" (
    set "VLLM_SRC=%SCRIPT_DIR%vllm-source"
) else if exist "%SCRIPT_DIR%setup.py" (
    set "VLLM_SRC=%SCRIPT_DIR%"
) else (
    echo [ERROR] Cannot find vLLM source. Clone it into vllm-source\ next to this script:
    echo         git clone https://github.com/vllm-project/vllm.git vllm-source
    echo         cd vllm-source ^&^& git checkout v0.25.1
    exit /b 1
)

where git.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git.exe not found on PATH.
    exit /b 1
)

where python.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python.exe not found on PATH. Activate the Python 3.13 build environment.
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] The release wheel requires Python 3.13.
    exit /b 1
)

python -c "import torch; raise SystemExit(0 if torch.__version__.startswith('2.11.0') and torch.version.cuda == '12.8' else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] PyTorch 2.11.0 with CUDA 12.8 is required in the active Python environment.
    exit /b 1
)

git -C "%VLLM_SRC%" merge-base --is-ancestor 752a3a504485790a2e8491cacbb35c137339ad34 HEAD >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Source is not based on upstream vLLM v0.25.1 commit 752a3a504.
    echo         Source: %VLLM_SRC%
    exit /b 1
)

:: -----------------------------------------------------------
:: 4. Apply patch
:: -----------------------------------------------------------

if not exist "%SCRIPT_DIR%vllm-windows-v9.patch" (
    echo [ERROR] vllm-windows-v9.patch not found next to build.bat.
    exit /b 1
)

pushd "%VLLM_SRC%"
git apply --check "%SCRIPT_DIR%vllm-windows-v9.patch" >nul 2>&1
if not errorlevel 1 goto :applyPatch

git apply --reverse --check "%SCRIPT_DIR%vllm-windows-v9.patch" >nul 2>&1
if errorlevel 1 goto :patchConflict
echo Windows patch is already applied cleanly.
goto :patchReady

:applyPatch
echo Applying vllm-windows-v9.patch...
git apply "%SCRIPT_DIR%vllm-windows-v9.patch"
if errorlevel 1 goto :patchConflict
goto :patchReady

:patchConflict
popd
echo [ERROR] The Windows patch is partially applied or conflicts with this source tree.
echo         Use a clean upstream v0.25.1 checkout or restore the fully patched tree.
exit /b 1

:patchReady
popd

if not defined PROTOC (
    if exist "%SCRIPT_DIR%tools\protoc\bin\protoc.exe" (
        set "PROTOC=%SCRIPT_DIR%tools\protoc\bin\protoc.exe"
    )
)

if not defined PROTOC (
    echo [ERROR] PROTOC is not set. The release wheel requires the Rust frontend.
    echo         Install protoc and set PROTOC=...\protoc.exe.
    exit /b 1
)

if not exist "%PROTOC%" (
    echo [ERROR] PROTOC does not exist: %PROTOC%
    exit /b 1
)

:: -----------------------------------------------------------
:: 5. Build
:: -----------------------------------------------------------

echo.
echo Configuration:
echo   CUDA_HOME              = %CUDA_HOME%
echo   TORCH_CUDA_ARCH_LIST   = %TORCH_CUDA_ARCH_LIST%
echo   MAX_JOBS               = %MAX_JOBS%
echo   Source                 = %VLLM_SRC%
if defined PROTOC echo   PROTOC                 = %PROTOC%
echo.
echo Starting build (this can take several hours for 8.6;8.9;12.0)...
echo.

cd /d "%VLLM_SRC%"
python -m pip install -e . --no-build-isolation -v 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed. Check output above for errors.
    exit /b 1
)

:: -----------------------------------------------------------
:: 6. Post-build: verify every release-critical artifact
:: -----------------------------------------------------------

call :requireArtifact "vllm\_C_stable_libtorch.pyd" || exit /b 1
call :requireArtifact "vllm\_moe_C_stable_libtorch.pyd" || exit /b 1
call :requireArtifact "vllm\_rust_tool_parser.pyd" || exit /b 1
call :requireArtifact "vllm\cumem_allocator.pyd" || exit /b 1
call :requireArtifact "vllm\fs_io_C.pyd" || exit /b 1
call :requireArtifact "vllm\spinloop.pyd" || exit /b 1
call :requireArtifact "vllm\vllm-rs.exe" || exit /b 1
call :requireArtifact "vllm\vllm_flash_attn\_vllm_fa2_C.pyd" || exit /b 1
call :requireArtifact "vllm\vllm_flash_attn\layers\rotary.py" || exit /b 1
call :requireArtifact "vllm\vllm_flash_attn\ops\triton\rotary.py" || exit /b 1
call :requireArtifact "vllm\vllm_flash_attn\cute\interface.py" || exit /b 1
call :requireArtifact "vllm\third_party\triton_kernels" || exit /b 1
call :requireArtifact "vllm\third_party\fmha_sm100" || exit /b 1

echo.
echo Build complete!
echo.
echo Recommended environment for single-rank vLLM on Windows:
echo   set VLLM_HOST_IP=127.0.0.1
echo Use kv_cache_dtype=auto for the fast baseline.
echo Enable enforce_eager only for graph/compile troubleshooting.
echo.

endlocal
exit /b 0

:requireArtifact
if exist "%~1" exit /b 0
echo [ERROR] Build completed but required artifact is missing: %~1
exit /b 1
