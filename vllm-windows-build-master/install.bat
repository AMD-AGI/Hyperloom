@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"

echo.
echo  ============================================================
echo          vLLM v0.33.0 Windows Installer
echo     Portable Python 3.13.14 + PyTorch 2.14.0 (cu130) + vLLM 0.33.0
echo  ============================================================
echo.

REM ============================================================
REM  Version Configuration
REM ============================================================
set "PYTHON_VERSION=3.13.14"
set "PYTHON_URL=https://www.python.org/ftp/python/3.13.14/python-3.13.14-embed-amd64.zip"
set "PYTHON_SHA256=90B4E5B9898B72D744650524BFF92377C367F44BD5FBD09E3148656C080AD907"
set "PYTHON_SIZE=10964839"
set "PYTHON_DEV_URL=https://www.nuget.org/api/v2/package/python/3.13.14"
set "PYTHON_DEV_SHA256=9AC15CFA6CAB1115C83D48F2AF55C554EFA4D1BB044BBC4AB1C9D17AD426E16C"
set "PYTHON_DEV_SIZE=14345376"
set "PYTHON_PTH_FILE=python313._pth"
set "PYTHON_PTH_ZIP=python313.zip"
set "PYTHON_LIB_NAME=python313.lib"
set "TRITON_NVIDIA_DIR=%~dp0python\Lib\site-packages\triton\backends\nvidia"
set "GETPIP_URL=https://raw.githubusercontent.com/pypa/get-pip/5e84c8360eaf92009551b3eec69d734137f31cec/public/get-pip.py"
set "GETPIP_SHA256=A341E1A43E38001C551A1508A73FF23636A11970B61D901D9A1CAD2A18F57055"
set "GETPIP_SIZE=2226848"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"

REM Pre-built vLLM wheel (auto-downloaded into dist-v0.33.0\ if not present locally).
REM This exact release artifact is verified by size and SHA-256 before install.
set "WHEEL_NAME=vllm-0.33.0-cp313-cp313-win_amd64.whl"
set "WHEEL_URL=https://github.com/aivrar/vllm-windows-build/releases/download/v0.33.0-win-cu130/vllm-0.33.0-cp313-cp313-win_amd64.whl"
set "WHEEL_SHA256=28C45D24E246D515D1AE87E8CA286CF7D3B1DADBD8AEC4FEB685B01B93627321"
set "WHEEL_SIZE=243466236"
set "WHEEL_FILE=%~dp0dist-v0.33.0\%WHEEL_NAME%"
set "WHEEL_PART=%~dp0dist-v0.33.0\%WHEEL_NAME%.part"

REM Pure-Python Multi-TurboQuant wheel built from commit e2b59ee474132999c2b42d5c96bfc48fcaf850dc.
set "MTQ_NAME=multi_turboquant-0.1.0-py3-none-any.whl"
set "MTQ_URL=https://github.com/aivrar/vllm-windows-build/releases/download/v0.33.0-win-cu130/multi_turboquant-0.1.0-py3-none-any.whl"
set "MTQ_SHA256=5B310E05904B588539D9A8E3374DFA6C160F025F9C2099BA5C7877C79B2FA149"
set "MTQ_SIZE=136429"
set "MTQ_FILE=%~dp0dist-v0.33.0\%MTQ_NAME%"
set "MTQ_PART=%~dp0dist-v0.33.0\%MTQ_NAME%.part"

set "STAGES_TOTAL=5"
if not defined CUDA_DEVICE_ORDER set "CUDA_DEVICE_ORDER=PCI_BUS_ID"

echo  Components to install:
echo    - Python %PYTHON_VERSION% (embedded distribution + headers/libs)
echo    - pip (package manager)
echo    - PyTorch 2.14.0+cu130 + triton-windows (CUDA GPU acceleration)
echo    - vLLM 0.33.0 wheel (pre-built Windows binary)
echo    - Verification
echo.

if not exist "%~dp0verify_bootstrap.ps1" (
    echo  FAILED: verify_bootstrap.ps1 is missing next to install.bat.
    goto :fail
)
if not exist "%~dp0expand_zip.ps1" (
    echo  FAILED: expand_zip.ps1 is missing next to install.bat.
    goto :fail
)
powershell -NoProfile -Command "if ($PSVersionTable.PSVersion.Major -lt 3) { exit 1 }"
if errorlevel 1 (
    echo  FAILED: install.bat requires Windows PowerShell 3 or newer.
    goto :fail
)

REM ============================================================
REM  STAGE 1: Download and Extract Python Embedded
REM ============================================================
echo [1/%STAGES_TOTAL%] Python %PYTHON_VERSION% embedded...
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo          FAILED: Existing python\ is not Python 3.13.
        echo          Delete python\ and rerun install.bat to install Python %PYTHON_VERSION%.
        goto :fail
    )
    echo          SKIP - already installed
    goto :pythonDevFiles
)
echo          Downloading from python.org...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference = 'Stop';" ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "$path = Join-Path $env:TEMP 'vllm-python-embed.zip';" ^
    "Remove-Item $path -Force -ErrorAction SilentlyContinue;" ^
    "Invoke-WebRequest -UseBasicParsing -Uri '%PYTHON_URL%' -OutFile $path"
if errorlevel 1 (
    echo          FAILED: Could not download Python %PYTHON_VERSION%
    echo          URL: %PYTHON_URL%
    goto :fail
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify_bootstrap.ps1" ^
    "%TEMP%\vllm-python-embed.zip" "%PYTHON_SHA256%" %PYTHON_SIZE%
if errorlevel 1 (
    echo          FAILED: Python archive integrity check failed.
    del "%TEMP%\vllm-python-embed.zip" 2>nul
    goto :fail
)
echo          Extracting to python\ ...
if exist "%~dp0python.part" rmdir /S /Q "%~dp0python.part"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0expand_zip.ps1" ^
    "%TEMP%\vllm-python-embed.zip" "%~dp0python.part"
if errorlevel 1 (
    echo          FAILED: Could not extract Python archive
    rmdir /S /Q "%~dp0python.part" 2>nul
    goto :fail
)
del "%TEMP%\vllm-python-embed.zip" 2>nul
if not exist "%~dp0python.part\python.exe" (
    echo          FAILED: python.exe not found after extraction
    rmdir /S /Q "%~dp0python.part" 2>nul
    goto :fail
)
"%~dp0python.part\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:3] == (3, 13, 14) else 1)" >nul 2>nul
if errorlevel 1 (
    echo          FAILED: Extracted Python version is not %PYTHON_VERSION%.
    rmdir /S /Q "%~dp0python.part" 2>nul
    goto :fail
)
move /Y "%~dp0python.part" "%~dp0python" >nul
if errorlevel 1 (
    echo          FAILED: Could not move verified Python into place.
    rmdir /S /Q "%~dp0python.part" 2>nul
    goto :fail
)
echo          OK

:pythonDevFiles
REM Triton JIT compiles a small CUDA driver helper at runtime and needs
REM Python.h plus python313.lib. The embeddable Python zip does not ship them.
echo          Checking Python headers/libs for Triton...
if exist "%~dp0python\Include\Python.h" if exist "%~dp0python\libs\%PYTHON_LIB_NAME%" (
    echo          OK - Python development files present
    goto :stage2
)
echo          Downloading Python headers/libs from NuGet...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference = 'Stop';" ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "$pkg = Join-Path $env:TEMP 'vllm-python-dev.nupkg';" ^
    "$out = Join-Path $env:TEMP 'vllm-python-dev';" ^
    "Remove-Item $pkg -Force -ErrorAction SilentlyContinue;" ^
    "Remove-Item $out -Recurse -Force -ErrorAction SilentlyContinue;" ^
    "Invoke-WebRequest -UseBasicParsing -Uri '%PYTHON_DEV_URL%' -OutFile $pkg"
if errorlevel 1 (
    echo          FAILED: Could not download Python headers/libs.
    goto :fail
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify_bootstrap.ps1" ^
    "%TEMP%\vllm-python-dev.nupkg" "%PYTHON_DEV_SHA256%" %PYTHON_DEV_SIZE%
if errorlevel 1 (
    echo          FAILED: Python development package integrity check failed.
    goto :fail
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0expand_zip.ps1" ^
    "%TEMP%\vllm-python-dev.nupkg" "%TEMP%\vllm-python-dev"
if errorlevel 1 (
    echo          FAILED: Could not extract Python headers/libs.
    goto :fail
)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference = 'Stop';" ^
    "$out = Join-Path $env:TEMP 'vllm-python-dev';" ^
    "New-Item -ItemType Directory -Force -Path '%~dp0python\Include', '%~dp0python\libs' | Out-Null;" ^
    "Copy-Item -Path (Join-Path $out 'tools\include\*') -Destination '%~dp0python\Include' -Recurse -Force;" ^
    "Copy-Item -Path (Join-Path $out 'tools\libs\*') -Destination '%~dp0python\libs' -Recurse -Force"
if errorlevel 1 (
    echo          FAILED: Could not install Python headers/libs.
    goto :fail
)
if not exist "%~dp0python\Include\Python.h" (
    echo          FAILED: Python.h not found after installing development files.
    goto :fail
)
if not exist "%~dp0python\libs\%PYTHON_LIB_NAME%" (
    echo          FAILED: %PYTHON_LIB_NAME% not found after installing development files.
    goto :fail
)
echo          OK

:stage2
REM ============================================================
REM  STAGE 2: Configure Python for site-packages + Install pip
REM ============================================================
echo [2/%STAGES_TOTAL%] Python configuration + pip...

REM Write python313._pth to enable site-packages and import site.
echo          Configuring %PYTHON_PTH_FILE%...
(
    echo %PYTHON_PTH_ZIP%
    echo .
    echo ..
    echo Lib
    echo Lib\site-packages
    echo DLLs
    echo import site
) > "%~dp0python\%PYTHON_PTH_FILE%"

REM Create required directories
if not exist "%~dp0python\Lib\site-packages" mkdir "%~dp0python\Lib\site-packages"
if not exist "%~dp0python\Scripts" mkdir "%~dp0python\Scripts"

if exist "%~dp0python\Scripts\pip.exe" (
    echo          OK - pip already installed
    goto :stage3
)

REM Download and run get-pip.py
echo          Downloading get-pip.py...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference = 'Stop';" ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
    "$ProgressPreference = 'SilentlyContinue';" ^
    "$path = Join-Path $env:TEMP 'get-pip.py';" ^
    "Remove-Item $path -Force -ErrorAction SilentlyContinue;" ^
    "Invoke-WebRequest -UseBasicParsing -Uri '%GETPIP_URL%' -OutFile $path"
if errorlevel 1 (
    echo          FAILED: Could not download get-pip.py
    goto :fail
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify_bootstrap.ps1" ^
    "%TEMP%\get-pip.py" "%GETPIP_SHA256%" %GETPIP_SIZE%
if errorlevel 1 (
    echo          FAILED: get-pip.py integrity check failed.
    goto :fail
)
echo          Installing pip...
"%~dp0python\python.exe" "%TEMP%\get-pip.py" --no-warn-script-location
if errorlevel 1 (
    echo          FAILED: pip installation error
    goto :fail
)
del "%TEMP%\get-pip.py" 2>nul
if not exist "%~dp0python\Scripts\pip.exe" (
    echo          FAILED: pip.exe not found after installation
    goto :fail
)
echo          OK

:stage3
REM ============================================================
REM  STAGE 3: Install PyTorch 2.14.0+cu130 + triton-windows
REM ============================================================
echo [3/%STAGES_TOTAL%] PyTorch 2.14.0+cu130 + triton-windows (~2.5 GB download)...
if exist "%~dp0python\.torch-installed" (
    "%~dp0python\python.exe" -c "import torch, triton; assert torch.__version__.startswith('2.14.0'); assert torch.version.cuda == '13.0'; assert triton.__version__.startswith('3.8.0')" >nul 2>nul
    if not errorlevel 1 (
        echo          SKIP - already installed ^(delete python\.torch-installed to force^)
        goto :stage4
    )
    echo          Existing marker found, but torch/triton import failed - reinstalling...
    del "%~dp0python\.torch-installed" 2>nul
)
echo          Installing PyTorch 2.14.0 from pytorch.org...
"%~dp0python\python.exe" -m pip install torch==2.14.0 torchaudio==2.11.0 torchvision==0.29.0 --index-url %TORCH_INDEX% --no-warn-script-location
if errorlevel 1 (
    echo          FAILED: PyTorch installation error - check output above
    goto :fail
)
echo          Installing triton-windows 3.8.0...
"%~dp0python\python.exe" -m pip install "triton-windows==3.8.0.post29" --no-warn-script-location
if errorlevel 1 (
    echo          FAILED: triton-windows installation error
    goto :fail
)
echo %DATE% %TIME% > "%~dp0python\.torch-installed"
echo          OK

:stage4
REM ============================================================
REM  STAGE 4: Install vLLM Wheel + Dependencies
REM ============================================================
echo [4/%STAGES_TOTAL%] vLLM wheel + dependencies...
if exist "%~dp0python\.vllm-installed" (
    findstr /I /X /C:"WHEEL_SHA256=%WHEEL_SHA256%" "%~dp0python\.vllm-installed" >nul 2>nul
    if not errorlevel 1 findstr /I /X /C:"MTQ_SHA256=%MTQ_SHA256%" "%~dp0python\.vllm-installed" >nul 2>nul
    if not errorlevel 1 "%~dp0python\python.exe" "%~dp0verify_install.py" --root "%~dp0." >nul 2>nul
    if not errorlevel 1 (
        echo          SKIP - already installed ^(delete python\.vllm-installed to force^)
        goto :stage5
    )
    echo          Existing install is stale or incomplete - repairing...
    del "%~dp0python\.vllm-installed" 2>nul
)

REM Only accept the current v0.33.0 wheel. Older local wheels are not
REM compatible substitutes and the original dist-v7 artifact omitted required
REM FlashAttention Python modules.
if not exist "%~dp0verify_artifact.py" (
    echo          FAILED: verify_artifact.py is missing next to install.bat.
    goto :fail
)

if exist "%WHEEL_FILE%" (
    echo          Checking local wheel...
    "%~dp0python\python.exe" "%~dp0verify_artifact.py" "%WHEEL_FILE%" "%WHEEL_SHA256%" %WHEEL_SIZE%
    if errorlevel 1 (
        echo          Local wheel is stale or incomplete - downloading a clean copy.
        del /F /Q "%WHEEL_FILE%" 2>nul
    )
)

REM Download to a temporary name so an interrupted request cannot look complete.
if not exist "%WHEEL_FILE%" (
    echo          No local wheel found - downloading from GitHub Releases ^(~228 MB^)...
    if not exist "%~dp0dist-v0.33.0" mkdir "%~dp0dist-v0.33.0"
    del /F /Q "%WHEEL_PART%" 2>nul
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ErrorActionPreference = 'Stop';" ^
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
        "$ProgressPreference = 'SilentlyContinue';" ^
        "Invoke-WebRequest -UseBasicParsing -Uri '%WHEEL_URL%' -OutFile '%WHEEL_PART%'"
    if errorlevel 1 (
        echo          FAILED: Could not download the vLLM wheel.
        echo          URL: %WHEEL_URL%
        echo          Download it manually and place it in: %~dp0dist-v0.33.0\
        del /F /Q "%WHEEL_PART%" 2>nul
        goto :fail
    )

    "%~dp0python\python.exe" "%~dp0verify_artifact.py" "%WHEEL_PART%" "%WHEEL_SHA256%" %WHEEL_SIZE%
    if errorlevel 1 (
        echo          FAILED: Downloaded wheel failed its integrity check.
        del /F /Q "%WHEEL_PART%" 2>nul
        goto :fail
    )

    move /Y "%WHEEL_PART%" "%WHEEL_FILE%" >nul
    if errorlevel 1 (
        echo          FAILED: Could not move the verified wheel into dist-v0.33.0\.
        goto :fail
    )
)

REM Verify again immediately before pip consumes the final file.
"%~dp0python\python.exe" "%~dp0verify_artifact.py" "%WHEEL_FILE%" "%WHEEL_SHA256%" %WHEEL_SIZE%
if errorlevel 1 (
    echo          FAILED: Wheel integrity check failed.
    goto :fail
)
echo          SHA256 verified
echo          Installing corrected vLLM wheel...
"%~dp0python\python.exe" -m pip install "%WHEEL_FILE%" --force-reinstall --no-deps --no-warn-script-location
if errorlevel 1 (
    echo          FAILED: vLLM installation error - check output above
    goto :fail
)
echo          Resolving vLLM dependencies...
"%~dp0python\python.exe" -m pip install "%WHEEL_FILE%" --no-warn-script-location
if errorlevel 1 (
    echo          FAILED: vLLM dependency installation error - check output above
    goto :fail
)

if exist "%MTQ_FILE%" (
    echo          Checking local Multi-TurboQuant wheel...
    "%~dp0python\python.exe" "%~dp0verify_artifact.py" "%MTQ_FILE%" "%MTQ_SHA256%" %MTQ_SIZE%
    if errorlevel 1 (
        echo          Local Multi-TurboQuant wheel is stale - downloading a clean copy.
        del /F /Q "%MTQ_FILE%" 2>nul
    )
)
if not exist "%MTQ_FILE%" (
    echo          Downloading pinned Multi-TurboQuant wheel...
    del /F /Q "%MTQ_PART%" 2>nul
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$ErrorActionPreference = 'Stop';" ^
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
        "$ProgressPreference = 'SilentlyContinue';" ^
        "Invoke-WebRequest -UseBasicParsing -Uri '%MTQ_URL%' -OutFile '%MTQ_PART%'"
    if errorlevel 1 (
        echo          FAILED: Could not download the Multi-TurboQuant wheel.
        del /F /Q "%MTQ_PART%" 2>nul
        goto :fail
    )
    "%~dp0python\python.exe" "%~dp0verify_artifact.py" "%MTQ_PART%" "%MTQ_SHA256%" %MTQ_SIZE%
    if errorlevel 1 (
        echo          FAILED: Multi-TurboQuant wheel failed its integrity check.
        del /F /Q "%MTQ_PART%" 2>nul
        goto :fail
    )
    move /Y "%MTQ_PART%" "%MTQ_FILE%" >nul
    if errorlevel 1 (
        echo          FAILED: Could not move the verified Multi-TurboQuant wheel.
        goto :fail
    )
)
"%~dp0python\python.exe" "%~dp0verify_artifact.py" "%MTQ_FILE%" "%MTQ_SHA256%" %MTQ_SIZE%
if errorlevel 1 goto :fail
echo          Installing pinned Multi-TurboQuant...
"%~dp0python\python.exe" -m pip install "%MTQ_FILE%" --force-reinstall --no-deps --no-warn-script-location
if errorlevel 1 (
    echo          FAILED: Multi-TurboQuant installation error.
    goto :fail
)

REM The Windows patch adds AMD64 to vLLM's dependency markers. Keep an
REM explicit install here as a repair step for environments created by older
REM wheels or pip metadata caches.
echo          Installing structured-output backends (llguidance, xgrammar)...
"%~dp0python\python.exe" -m pip install "llguidance>=1.7.0,<1.8.0" "xgrammar==0.2.8" --no-warn-script-location
if errorlevel 1 (
    echo          FAILED: llguidance/xgrammar installation error.
    goto :fail
)
echo          OK

:stage5
REM ============================================================
REM  STAGE 5: Verify Installation
REM ============================================================
echo [5/%STAGES_TOTAL%] Verification...
if exist "%TRITON_NVIDIA_DIR%\bin\ptxas.exe" (
    set "CUDA_PATH=%TRITON_NVIDIA_DIR%"
    set "CUDA_HOME=%TRITON_NVIDIA_DIR%"
    set "PATH=%TRITON_NVIDIA_DIR%\bin;%PATH%"
    echo          Using bundled Triton CUDA toolkit: %TRITON_NVIDIA_DIR%
)
"%~dp0python\python.exe" "%~dp0verify_install.py" --root "%~dp0." --cuda
if errorlevel 1 (
    echo          FAILED: vLLM runtime verification failed.
    echo          Make sure an NVIDIA driver is installed and rerun install.bat.
    echo          If you see Python.h or python313.lib errors, delete python\ and rerun install.bat.
    goto :fail
)
> "%~dp0python\.vllm-installed" echo WHEEL_SHA256=%WHEEL_SHA256%
>> "%~dp0python\.vllm-installed" echo MTQ_SHA256=%MTQ_SHA256%
echo          OK

echo.
echo  ============================================================
echo                   Installation Complete
echo  ============================================================
echo.
echo  To start vLLM:
echo    launch.bat                                  (interactive model selector)
echo    launch.bat --model path\to\model            (direct launch)
echo    launch.bat --model path\to\model --port 8000
echo.
echo  Or manually:
echo    python\python.exe vllm_launcher.py --model path\to\model
echo.
endlocal
exit /b 0

:fail
endlocal
exit /b 1
