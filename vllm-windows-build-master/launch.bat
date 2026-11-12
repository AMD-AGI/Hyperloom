@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"

echo.
echo  vLLM Windows Server Launcher
echo  =============================
echo.

REM ============================================================
REM  Check Python installation
REM ============================================================
set "NEEDS_INSTALL=0"
set "EXPECTED_WHEEL_SHA256=28C45D24E246D515D1AE87E8CA286CF7D3B1DADBD8AEC4FEB685B01B93627321"
set "EXPECTED_MTQ_SHA256=5B310E05904B588539D9A8E3374DFA6C160F025F9C2099BA5C7877C79B2FA149"
if not exist "%~dp0python\python.exe" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\.torch-installed" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\.vllm-installed" set "NEEDS_INSTALL=1"
if exist "%~dp0python\.vllm-installed" findstr /I /X /C:"WHEEL_SHA256=%EXPECTED_WHEEL_SHA256%" "%~dp0python\.vllm-installed" >nul 2>nul || set "NEEDS_INSTALL=1"
if exist "%~dp0python\.vllm-installed" findstr /I /X /C:"MTQ_SHA256=%EXPECTED_MTQ_SHA256%" "%~dp0python\.vllm-installed" >nul 2>nul || set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Include\Python.h" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\libs\python313.lib" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\python313._pth" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\torch" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\triton" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\vllm" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\llguidance" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\multi_turboquant" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\xgrammar" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\vllm\vllm-rs.exe" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\vllm\model_executor\models\qwen3_5.py" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\vllm\model_executor\models\qwen3_vl.py" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\vllm\vllm_flash_attn\layers\rotary.py" set "NEEDS_INSTALL=1"
if not exist "%~dp0python\Lib\site-packages\vllm\vllm_flash_attn\ops\triton\rotary.py" set "NEEDS_INSTALL=1"
if not exist "%~dp0vllm_launcher.py" set "NEEDS_INSTALL=1"
if not exist "%~dp0engine_dispatcher.py" set "NEEDS_INSTALL=1"
if not exist "%~dp0verify_install.py" set "NEEDS_INSTALL=1"

if "%NEEDS_INSTALL%"=="1" (
    echo  Python install is missing or incomplete. Running installer...
    echo.
    call "%~dp0install.bat"
    if errorlevel 1 (
        echo.
        echo  Installation failed. Please check errors above.
        if not defined VLLM_NO_PAUSE pause
        exit /b 1
    )
    echo.
)

REM ============================================================
REM  Configure environment
REM ============================================================
set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0python\Library\bin;%PATH%"
if not defined CUDA_DEVICE_ORDER set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TRITON_NVIDIA_DIR=%~dp0python\Lib\site-packages\triton\backends\nvidia"
if exist "%TRITON_NVIDIA_DIR%\bin\ptxas.exe" (
    set "CUDA_PATH=%TRITON_NVIDIA_DIR%"
    set "CUDA_HOME=%TRITON_NVIDIA_DIR%"
    set "PATH=%TRITON_NVIDIA_DIR%\bin;%PATH%"
)
set "VLLM_HOST_IP=127.0.0.1"
if not defined PYTHONHASHSEED set "PYTHONHASHSEED=0"

REM Suppress tokenizer parallelism warning
set "TOKENIZERS_PARALLELISM=false"

REM ============================================================
REM  Launch vLLM server
REM ============================================================
REM All arguments are forwarded to vllm_launcher.py.
REM If no --model is passed, the interactive model selector activates.

"%~dp0python\python.exe" "%~dp0vllm_launcher.py" %*

set "SERVER_EXIT=%ERRORLEVEL%"
if not "%SERVER_EXIT%"=="0" (
    echo.
    echo  Server exited with error code %SERVER_EXIT%
    if not defined VLLM_NO_PAUSE pause
    exit /b %SERVER_EXIT%
)

endlocal
exit /b 0
