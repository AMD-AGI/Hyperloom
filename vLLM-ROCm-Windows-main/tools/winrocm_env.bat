@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
rem Shared toolchain setup for the native HIP build scripts: puts MSVC on PATH and resolves the HIP
rem SDK, without hardcoding any machine-specific location. Called by the build_*.bat wrappers as
rem   call "%~dp0..\..\tools\winrocm_env.bat" || exit /b 1
rem Override the auto-detection by setting VCVARS64 and/or HIP_PATH before calling.
rem No setlocal here on purpose: vcvars64 and HIP_PATH must reach the caller.

rem --- MSVC: skip entirely if a compiler is already on PATH (e.g. a Developer Prompt) ---
where cl.exe >nul 2>&1
if %errorlevel%==0 goto :hip

if defined VCVARS64 if exist "%VCVARS64%" call "%VCVARS64%" >nul
where cl.exe >nul 2>&1
if %errorlevel%==0 goto :hip

rem Deliberately no parenthesised blocks below: the vswhere path contains "(x86)", and cmd would let
rem that closing paren terminate an enclosing block early. Plain gotos keep the parser out of it.
set "_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%_VSWHERE%" set "_VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%_VSWHERE%" goto :vsprobe

rem vswhere output goes through a temp file rather than for /f: its own path contains "(x86)", and
rem inside the for's in(...) that paren closes the construct early and cmd then tries to run a
rem mangled command, printing a spurious "vswhere.exe is not recognized".
set "_VSDIR="
set "_VSTMP=%TEMP%\_winrocm_vswhere.txt"
"%_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%_VSTMP%" 2>nul
if exist "%_VSTMP%" set /p _VSDIR=<"%_VSTMP%"
del "%_VSTMP%" 2>nul
set "_VSTMP="
if not defined _VSDIR goto :vsprobe
if exist "%_VSDIR%\VC\Auxiliary\Build\vcvars64.bat" call "%_VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
goto :vsdone

:vsprobe
rem No vswhere (or it found nothing): fall back to the default install locations.
for %%e in (BuildTools Community Professional Enterprise) do if not defined _VSDIR if exist "%ProgramFiles%\Microsoft Visual Studio\2022\%%e\VC\Auxiliary\Build\vcvars64.bat" set "_VSDIR=%ProgramFiles%\Microsoft Visual Studio\2022\%%e"
if defined _VSDIR if exist "%_VSDIR%\VC\Auxiliary\Build\vcvars64.bat" call "%_VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul

:vsdone
set "_VSWHERE="
set "_VSDIR="

where cl.exe >nul 2>&1
if not %errorlevel%==0 (
    echo [winrocm_env] MSVC not found.
    echo               Install Visual Studio 2022 Build Tools with "Desktop development with C++",
    echo               or set VCVARS64 to the full path of your vcvars64.bat.
    exit /b 1
)

:hip
rem --- HIP SDK ---
if not defined HIP_PATH if defined ROCM_PATH set "HIP_PATH=%ROCM_PATH%"
if not defined HIP_PATH if defined ROCM_HOME set "HIP_PATH=%ROCM_HOME%"
if not defined HIP_PATH if exist "C:\HIP-SDK\bin\hipcc.exe" set "HIP_PATH=C:\HIP-SDK"
if not defined HIP_PATH (
    for /f "delims=" %%d in ('dir /b /ad /o-n "%ProgramFiles%\AMD\ROCm" 2^>nul') do (
        if not defined HIP_PATH if exist "%ProgramFiles%\AMD\ROCm\%%d\bin\hipcc.exe" set "HIP_PATH=%ProgramFiles%\AMD\ROCm\%%d"
    )
)
if not defined HIP_PATH (
    echo [winrocm_env] HIP SDK not found.
    echo               Install the AMD HIP SDK, or set HIP_PATH to its install root, e.g.
    echo                   set HIP_PATH=C:\Program Files\AMD\ROCm\7.2
    exit /b 1
)
set "ROCM_PATH=%HIP_PATH%"
set "ROCM_HOME=%HIP_PATH%"

rem --- Scratch build dirs: keep them off the drive root when an ASCII location exists ---
rem C:\vw_cext_build and C:\vw_cext_hip are created here by default (short path: ninja/MSVC
rem hit the Windows 260-char limit on deep trees). VLLM_WIN_BUILD_ROOT moves them elsewhere;
rem default to C:\AI\build (the ASCII-first location this fork documents) when unset.
if not defined VLLM_WIN_BUILD_ROOT if exist "C:\AI\" set "VLLM_WIN_BUILD_ROOT=C:\AI\build"
exit /b 0
